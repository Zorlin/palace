//! Context gathering for Palace CLI
//!
//! Gathers project context to provide Claude with relevant information:
//! - Git status
//! - Key files (SPEC.md, ROADMAP.md, etc.)
//! - Project structure
//! - Recent history

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// Gathered project context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectContext {
    /// Key files and their contents (truncated)
    pub files: HashMap<String, FileInfo>,

    /// Git status output
    pub git_status: Option<String>,

    /// Recent git log
    pub git_log: Option<String>,

    /// Directory structure
    pub structure: Option<String>,

    /// Project type (rust, python, node, etc.)
    pub project_type: Option<String>,

    /// Any errors encountered during gathering
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    /// File exists
    pub exists: bool,
    /// File size in bytes
    pub size: u64,
    /// First N lines of content (for context)
    pub preview: Option<String>,
}

/// Key files to look for in a project
const KEY_FILES: &[&str] = &[
    "SPEC.md",
    "ROADMAP.md",
    "README.md",
    "CLAUDE.md",
    "TODO.md",
    "ARCHITECTURE.md",
    "CHANGELOG.md",
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    ".palace/config.json",
];

/// Gather context about the current project
pub fn gather_context(project_root: &Path) -> Result<ProjectContext> {
    let mut context = ProjectContext {
        files: HashMap::new(),
        git_status: None,
        git_log: None,
        structure: None,
        project_type: None,
        errors: Vec::new(),
    };

    // Detect project type
    if project_root.join("Cargo.toml").exists() {
        context.project_type = Some("rust".to_string());
    } else if project_root.join("package.json").exists() {
        context.project_type = Some("node".to_string());
    } else if project_root.join("pyproject.toml").exists() || project_root.join("setup.py").exists()
    {
        context.project_type = Some("python".to_string());
    }

    // Gather key files
    for filename in KEY_FILES {
        let file_path = project_root.join(filename);
        if file_path.exists() {
            match std::fs::metadata(&file_path) {
                Ok(meta) => {
                    let preview = if meta.len() < 10000 {
                        std::fs::read_to_string(&file_path).ok()
                    } else {
                        // Read first 100 lines for large files
                        std::fs::read_to_string(&file_path)
                            .ok()
                            .map(|content| {
                                content
                                    .lines()
                                    .take(100)
                                    .collect::<Vec<_>>()
                                    .join("\n")
                                    + "\n... (truncated)"
                            })
                    };

                    context.files.insert(
                        filename.to_string(),
                        FileInfo {
                            exists: true,
                            size: meta.len(),
                            preview,
                        },
                    );
                }
                Err(e) => {
                    context.errors.push(format!("Error reading {}: {}", filename, e));
                }
            }
        }
    }

    // Get git status
    if project_root.join(".git").exists() {
        let git_status = Command::new("git")
            .args(["status", "--short"])
            .current_dir(project_root)
            .output();

        if let Ok(output) = git_status {
            if output.status.success() {
                context.git_status = Some(String::from_utf8_lossy(&output.stdout).to_string());
            }
        }

        // Get recent git log
        let git_log = Command::new("git")
            .args(["log", "--oneline", "-10"])
            .current_dir(project_root)
            .output();

        if let Ok(output) = git_log {
            if output.status.success() {
                context.git_log = Some(String::from_utf8_lossy(&output.stdout).to_string());
            }
        }
    }

    // Get directory structure (limited depth)
    let structure = get_directory_structure(project_root, 3);
    context.structure = Some(structure);

    Ok(context)
}

/// Get a tree-like directory structure
fn get_directory_structure(root: &Path, max_depth: usize) -> String {
    let mut lines = Vec::new();
    build_tree(root, "", max_depth, 0, &mut lines);
    lines.join("\n")
}

fn build_tree(
    path: &Path,
    prefix: &str,
    max_depth: usize,
    current_depth: usize,
    lines: &mut Vec<String>,
) {
    if current_depth > max_depth {
        return;
    }

    // Skip hidden directories and common noise
    let skip_dirs = [
        "target",
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".next",
        ".cache",
    ];

    let entries: Vec<_> = match std::fs::read_dir(path) {
        Ok(entries) => entries.filter_map(|e| e.ok()).collect(),
        Err(_) => return,
    };

    let mut sorted_entries: Vec<_> = entries
        .into_iter()
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            !name.starts_with('.') && !skip_dirs.contains(&name.as_str())
        })
        .collect();

    sorted_entries.sort_by_key(|e| {
        let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
        (!is_dir, e.file_name())
    });

    let count = sorted_entries.len();
    for (i, entry) in sorted_entries.into_iter().enumerate() {
        let is_last = i == count - 1;
        let connector = if is_last { "└── " } else { "├── " };
        let name = entry.file_name().to_string_lossy().to_string();

        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let suffix = if is_dir { "/" } else { "" };

        lines.push(format!("{}{}{}{}", prefix, connector, name, suffix));

        if is_dir {
            let new_prefix = if is_last {
                format!("{}    ", prefix)
            } else {
                format!("{}│   ", prefix)
            };
            build_tree(&entry.path(), &new_prefix, max_depth, current_depth + 1, lines);
        }
    }
}

/// Build a full prompt with context
pub fn build_prompt(user_prompt: &str, context: &ProjectContext) -> String {
    let mut sections = Vec::new();

    // Project type
    if let Some(pt) = &context.project_type {
        sections.push(format!("PROJECT TYPE: {}\n", pt));
    }

    // Key files
    let mut file_sections = Vec::new();
    for (name, info) in &context.files {
        if info.exists {
            if let Some(preview) = &info.preview {
                file_sections.push(format!(
                    "=== {} ({} bytes) ===\n{}\n",
                    name, info.size, preview
                ));
            } else {
                file_sections.push(format!("=== {} ({} bytes) ===\n(binary or too large)\n", name, info.size));
            }
        }
    }
    if !file_sections.is_empty() {
        sections.push(format!("KEY FILES:\n\n{}", file_sections.join("\n")));
    }

    // Git status
    if let Some(status) = &context.git_status {
        if !status.trim().is_empty() {
            sections.push(format!("GIT STATUS:\n{}", status));
        }
    }

    // Git log
    if let Some(log) = &context.git_log {
        if !log.trim().is_empty() {
            sections.push(format!("RECENT COMMITS:\n{}", log));
        }
    }

    // Directory structure
    if let Some(structure) = &context.structure {
        if !structure.trim().is_empty() {
            sections.push(format!("DIRECTORY STRUCTURE:\n{}", structure));
        }
    }

    // Combine everything
    format!(
        "{}\n\n---\n\nCONTEXT:\n\n{}\n\n---\n\n{}",
        user_prompt,
        sections.join("\n\n"),
        "Please analyze this project and provide your suggestions."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_gather_context_empty_dir() {
        let dir = tempdir().unwrap();
        let context = gather_context(dir.path()).unwrap();
        assert!(context.files.is_empty());
        assert!(context.project_type.is_none());
    }

    #[test]
    fn test_gather_context_rust_project() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        fs::write(dir.path().join("SPEC.md"), "# Test Spec\n\nThis is a test.").unwrap();

        let context = gather_context(dir.path()).unwrap();
        assert_eq!(context.project_type, Some("rust".to_string()));
        assert!(context.files.contains_key("Cargo.toml"));
        assert!(context.files.contains_key("SPEC.md"));
    }

    #[test]
    fn test_build_prompt() {
        let context = ProjectContext {
            files: HashMap::new(),
            git_status: Some("M src/main.rs".to_string()),
            git_log: None,
            structure: None,
            project_type: Some("rust".to_string()),
            errors: Vec::new(),
        };

        let prompt = build_prompt("What should I do next?", &context);
        assert!(prompt.contains("What should I do next?"));
        assert!(prompt.contains("PROJECT TYPE: rust"));
        assert!(prompt.contains("GIT STATUS:"));
    }
}
