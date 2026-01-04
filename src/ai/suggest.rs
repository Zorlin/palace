use anyhow::Result;
use anthropic_ffi::{models, Client as AnthropicClient};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Lightweight project context (metadata only, not file contents)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectContext {
    /// Project root path
    pub root: PathBuf,
    /// File metadata (path -> size in bytes)
    pub files: HashMap<String, FileInfo>,
    /// Git status output
    pub git_status: Option<String>,
    /// Git branch
    pub git_branch: Option<String>,
    /// Recent history from .palace/history.jsonl
    pub recent_history: Vec<serde_json::Value>,
    /// Project type detection
    pub project_type: ProjectType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub size: u64,
    pub is_dir: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProjectType {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Unknown,
}

/// A suggested next action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    /// Short title for the suggestion
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Category (build, test, fix, refactor, docs, etc.)
    pub category: String,
    /// Priority (1-5, 1 being highest)
    pub priority: u8,
    /// Optional command to execute
    pub command: Option<String>,
}

/// Engine for generating project suggestions
pub struct SuggestionEngine {
    /// Anthropic client via Go FFI
    client: AnthropicClient,
    /// Model to use
    model: String,
}

impl ProjectContext {
    /// Gather context from a project directory
    pub fn gather(project_path: &Path) -> Result<Self> {
        let root = project_path.canonicalize()?;

        // Scan files (top 2 levels, skip hidden and large dirs)
        let mut files = HashMap::new();
        Self::scan_files(&root, &root, &mut files, 0, 2)?;

        // Get git status
        let git_status = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(&root)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string());

        // Get git branch
        let git_branch = Command::new("git")
            .args(["branch", "--show-current"])
            .current_dir(&root)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

        // Load recent history
        let history_path = root.join(".palace/history.jsonl");
        let recent_history = if history_path.exists() {
            std::fs::read_to_string(&history_path)
                .ok()
                .map(|s| {
                    s.lines()
                        .rev()
                        .take(10)
                        .filter_map(|l| serde_json::from_str(l).ok())
                        .collect()
                })
                .unwrap_or_default()
        } else {
            vec![]
        };

        // Detect project type
        let project_type = Self::detect_type(&root, &files);

        Ok(Self {
            root,
            files,
            git_status,
            git_branch,
            recent_history,
            project_type,
        })
    }

    fn scan_files(
        root: &Path,
        dir: &Path,
        files: &mut HashMap<String, FileInfo>,
        depth: usize,
        max_depth: usize,
    ) -> Result<()> {
        if depth > max_depth {
            return Ok(());
        }

        let entries = std::fs::read_dir(dir)?;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            // Skip hidden files and common large directories
            if name.starts_with('.') || name == "node_modules" || name == "target" || name == "__pycache__" || name == "venv" || name == ".venv" {
                continue;
            }

            let metadata = entry.metadata()?;
            let rel_path = path.strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            files.insert(rel_path.clone(), FileInfo {
                size: metadata.len(),
                is_dir: metadata.is_dir(),
            });

            if metadata.is_dir() {
                Self::scan_files(root, &path, files, depth + 1, max_depth)?;
            }
        }

        Ok(())
    }

    fn detect_type(root: &Path, files: &HashMap<String, FileInfo>) -> ProjectType {
        if root.join("Cargo.toml").exists() || files.contains_key("Cargo.toml") {
            ProjectType::Rust
        } else if root.join("pyproject.toml").exists() || root.join("setup.py").exists() || files.contains_key("pyproject.toml") {
            ProjectType::Python
        } else if root.join("package.json").exists() || files.contains_key("package.json") {
            if root.join("tsconfig.json").exists() || files.contains_key("tsconfig.json") {
                ProjectType::TypeScript
            } else {
                ProjectType::JavaScript
            }
        } else if root.join("go.mod").exists() || files.contains_key("go.mod") {
            ProjectType::Go
        } else {
            ProjectType::Unknown
        }
    }

    /// Convert to JSON for API call
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Get a compact summary for the prompt
    pub fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str(&format!("Project: {}\n", self.root.display()));
        s.push_str(&format!("Type: {:?}\n", self.project_type));

        if let Some(ref branch) = self.git_branch {
            s.push_str(&format!("Branch: {}\n", branch));
        }

        if let Some(ref status) = self.git_status {
            if !status.is_empty() {
                s.push_str(&format!("\nGit status:\n{}\n", status));
            } else {
                s.push_str("\nGit: clean\n");
            }
        }

        s.push_str(&format!("\nFiles ({} total):\n", self.files.len()));
        let mut sorted_files: Vec<_> = self.files.iter().collect();
        sorted_files.sort_by_key(|(k, _)| k.as_str());
        for (path, info) in sorted_files.iter().take(30) {
            if info.is_dir {
                s.push_str(&format!("  {}/\n", path));
            } else {
                s.push_str(&format!("  {} ({} bytes)\n", path, info.size));
            }
        }
        if self.files.len() > 30 {
            s.push_str(&format!("  ... and {} more\n", self.files.len() - 30));
        }

        if !self.recent_history.is_empty() {
            s.push_str("\nRecent actions:\n");
            for action in &self.recent_history {
                if let Some(a) = action.get("action") {
                    s.push_str(&format!("  - {}\n", a));
                }
            }
        }

        s
    }
}

impl SuggestionEngine {
    /// Create a new suggestion engine with explicit API key
    pub fn new(api_key: Option<&str>, model: Option<&str>) -> Result<Self> {
        let client = AnthropicClient::new(api_key)?;
        let model = model
            .map(|s| s.to_string())
            .unwrap_or_else(|| models::CLAUDE_SONNET_4_5.to_string());

        Ok(Self { client, model })
    }

    /// Create from environment variables
    pub fn from_env() -> Result<Self> {
        Self::new(None, std::env::var("ANTHROPIC_MODEL").ok().as_deref())
    }

    /// Generate suggestions for a project
    pub fn suggest(&self, context: &ProjectContext) -> Result<Vec<Suggestion>> {
        let prompt = self.build_prompt(context);
        let response = self.call_api(&prompt)?;
        self.parse_suggestions(&response)
    }

    fn build_prompt(&self, context: &ProjectContext) -> String {
        format!(r#"Analyze this project and suggest possible next actions.

{}

Provide as many options as you see fit - there may be many valid paths forward.
Be concrete and actionable.

Respond in JSON array format:
```json
[
  {{"title": "Short label", "description": "What to do", "category": "test", "priority": 1, "command": "cargo test"}}
]
```"#,
            context.summary()
        )
    }

    fn call_api(&self, prompt: &str) -> Result<String> {
        let system = "You are Palace, an AI assistant that analyzes software projects and suggests actionable next steps. Always respond with valid JSON.";

        let response = self.client.message(
            &self.model,
            Some(system),
            prompt,
            4096,
        )?;

        Ok(response.content)
    }

    fn parse_suggestions(&self, response: &str) -> Result<Vec<Suggestion>> {
        // Find JSON array in response (may be wrapped in markdown code block)
        let json_str = if let Some(start) = response.find('[') {
            if let Some(end) = response.rfind(']') {
                &response[start..=end]
            } else {
                response
            }
        } else {
            response
        };

        let suggestions: Vec<Suggestion> = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse suggestions: {}. Response: {}", e, response))?;

        Ok(suggestions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_project_type_detection() {
        let mut files = HashMap::new();
        files.insert("Cargo.toml".to_string(), FileInfo { size: 100, is_dir: false });

        let ctx = ProjectContext {
            root: PathBuf::from("/test"),
            files,
            git_status: None,
            git_branch: None,
            recent_history: vec![],
            project_type: ProjectType::Unknown,
        };

        assert_eq!(ProjectContext::detect_type(&ctx.root, &ctx.files), ProjectType::Rust);
    }

    #[test]
    fn test_context_summary() {
        let mut files = HashMap::new();
        files.insert("src/main.rs".to_string(), FileInfo { size: 1000, is_dir: false });
        files.insert("Cargo.toml".to_string(), FileInfo { size: 200, is_dir: false });

        let ctx = ProjectContext {
            root: PathBuf::from("/test/project"),
            files,
            git_status: Some("M src/main.rs\n".to_string()),
            git_branch: Some("main".to_string()),
            recent_history: vec![],
            project_type: ProjectType::Rust,
        };

        let summary = ctx.summary();
        assert!(summary.contains("Rust"));
        assert!(summary.contains("main"));
        assert!(summary.contains("src/main.rs"));
    }

    #[test]
    fn test_parse_suggestions() {
        let engine = SuggestionEngine::new("http://test", "key", "model");

        let response = r#"Here are my suggestions:
```json
[
  {
    "title": "Run tests",
    "description": "Ensure all tests pass",
    "category": "test",
    "priority": 1,
    "command": "cargo test"
  }
]
```"#;

        let suggestions = engine.parse_suggestions(response).unwrap();
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].title, "Run tests");
        assert_eq!(suggestions[0].command, Some("cargo test".to_string()));
    }
}
