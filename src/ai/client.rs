use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

/// Anthropic API client (via Go SDK)
pub struct AnthropicClient {
    base_url: String,
    model: String,
}

/// Request to Go SDK
#[derive(Debug, Serialize)]
struct SdkRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    work_dir: Option<String>,
    enable_tools: bool,
}

/// Event from Go SDK
#[derive(Debug, Deserialize)]
struct SdkEvent {
    #[serde(rename = "type")]
    event_type: String,
    text: Option<String>,
    error: Option<String>,
    tool: Option<String>,
    detail: Option<String>,
}

/// Parsed task suggestion from Claude
#[derive(Debug, Clone)]
pub struct TaskSuggestion {
    pub label: String,
    pub description: String,
    pub time_estimate: Option<String>,
    pub complexity: Option<String>,
    pub affected_files: Vec<String>,
}

impl AnthropicClient {
    pub fn new(base_url: &str, model: &str) -> Result<Self> {
        // Verify API key exists
        if std::env::var("ZAI_API_KEY").is_err() && std::env::var("ANTHROPIC_API_KEY").is_err() {
            return Err(anyhow!("ZAI_API_KEY or ANTHROPIC_API_KEY environment variable not set"));
        }

        Ok(Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
        })
    }

    /// Ask Claude what tasks to work on (uses SDK with tools for codebase exploration)
    pub async fn generate_tasks(&self, project_path: &Path) -> Result<Vec<TaskSuggestion>> {
        let system = r#"You are analyzing a codebase to suggest actionable tasks.

Use the available tools to explore the codebase:
- file_tree: Get project structure overview
- read_file: Read specific files
- list_files: List files in directories
- grep: Search for patterns

After exploring, respond with a YAML list of suggested actions:

```yaml
actions:
  - label: Short action title (under 60 chars)
    description: Detailed description of what this involves
    time_estimate: "15 min"
    complexity: simple
    affected_files:
      - src/main.rs
```

RULES:
- Explore the codebase FIRST using tools
- label: Required, keep under 60 chars
- description: Required, be detailed
- Focus on gamepad-related features and improvements
- Suggest concrete actions Claude can execute"#;

        let prompt = format!(
            "Analyze this project at {} and suggest what I should work on next. Start by exploring the codebase structure.",
            project_path.display()
        );

        let response = self.call_sdk(&prompt, Some(system), Some(project_path), true).await?;
        self.parse_yaml_tasks(&response)
    }

    /// Execute a task by sending it to Claude
    #[allow(dead_code)]
    pub async fn execute_task(&self, task: &TaskSuggestion, project_path: &Path) -> Result<String> {
        let system = r#"You are a coding assistant executing a task.
Use the available tools to explore and modify the codebase.
Be thorough but concise. Show what you're doing."#;

        let prompt = format!(
            "Task: {}\nDescription: {}\n\nExecute this task.",
            task.label,
            task.description
        );

        self.call_sdk(&prompt, Some(system), Some(project_path), true).await
    }

    /// Raw chat without tools
    pub async fn chat_raw(&self, prompt: &str, system: Option<&str>) -> Result<String> {
        self.call_sdk(prompt, system, None, false).await
    }

    /// Call the Go SDK binary
    async fn call_sdk(
        &self,
        prompt: &str,
        system: Option<&str>,
        work_dir: Option<&Path>,
        enable_tools: bool,
    ) -> Result<String> {
        let sdk_path = self.find_sdk_binary()?;

        let mut cmd = Command::new(&sdk_path);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Pass through API keys
        if let Ok(key) = std::env::var("ZAI_API_KEY") {
            cmd.env("ZAI_API_KEY", key);
        }
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            cmd.env("ANTHROPIC_API_KEY", key);
        }

        let mut child = cmd.spawn()?;

        // Build request
        let request = SdkRequest {
            prompt: prompt.to_string(),
            system: system.map(|s| s.to_string()),
            model: Some(self.model.clone()),
            base_url: Some(self.base_url.clone()),
            work_dir: work_dir.map(|p| p.to_string_lossy().to_string()),
            enable_tools,
        };

        // Send request and close stdin to signal EOF
        {
            let stdin = child.stdin.as_mut().ok_or_else(|| anyhow!("No stdin"))?;
            let mut request_json = serde_json::to_string(&request)?;
            request_json.push('\n');
            stdin.write_all(request_json.as_bytes()).await?;
            stdin.flush().await?;
        }
        // Take ownership and drop to close stdin
        drop(child.stdin.take());

        // Read response
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("No stdout"))?;
        let mut reader = BufReader::new(stdout).lines();

        let mut response_text = String::new();

        while let Some(line) = reader.next_line().await? {
            if let Ok(event) = serde_json::from_str::<SdkEvent>(&line) {
                match event.event_type.as_str() {
                    "text" => {
                        if let Some(text) = event.text {
                            // Print in realtime
                            print!("{}", text);
                            use std::io::Write;
                            std::io::stdout().flush().ok();
                            response_text.push_str(&text);
                        }
                    }
                    "error" => {
                        if let Some(error) = event.error {
                            return Err(anyhow!("SDK error: {}", error));
                        }
                    }
                    "tool_start" => {
                        if let Some(tool) = &event.tool {
                            println!("\n\x1b[33m[{}]\x1b[0m", tool);
                        }
                    }
                    "tool_result" => {
                        if let Some(detail) = &event.detail {
                            println!("\x1b[90m{}\x1b[0m", detail);
                        }
                    }
                    "done" => {
                        println!(); // Final newline
                        break;
                    }
                    _ => {}
                }
            }
        }

        // Wait for process
        let status = child.wait().await?;
        if !status.success() && response_text.is_empty() {
            return Err(anyhow!("SDK exited with code {:?}", status.code()));
        }

        Ok(response_text)
    }

    fn find_sdk_binary(&self) -> Result<PathBuf> {
        let candidates = [
            // In target directory
            PathBuf::from("target/palace-sdk"),
            // In same directory as current exe
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.join("palace-sdk")))
                .unwrap_or_default(),
            // In PATH
            which::which("palace-sdk").unwrap_or_default(),
        ];

        for path in candidates {
            if path.exists() && path.is_file() {
                return Ok(path);
            }
        }

        Err(anyhow!(
            "palace-sdk binary not found. Build with: cd go && go build -o ../target/palace-sdk ."
        ))
    }

    fn parse_yaml_tasks(&self, response: &str) -> Result<Vec<TaskSuggestion>> {
        let mut tasks = Vec::new();

        // Extract YAML block
        let yaml_content = if let Some(start) = response.find("```yaml") {
            let start = start + 7;
            if let Some(end) = response[start..].find("```") {
                &response[start..start + end]
            } else {
                response
            }
        } else if let Some(start) = response.find("```") {
            let start = start + 3;
            if let Some(end) = response[start..].find("```") {
                &response[start..start + end]
            } else {
                response
            }
        } else {
            response
        };

        // Parse YAML
        #[derive(Debug, Clone, Copy, PartialEq)]
        enum ParseMode {
            Normal,
            MultiLineDesc,
            FilesList,
        }

        let mut current_label: Option<String> = None;
        let mut current_desc: Option<String> = None;
        let mut current_time: Option<String> = None;
        let mut current_complexity: Option<String> = None;
        let mut current_files: Vec<String> = Vec::new();
        let mut parse_mode = ParseMode::Normal;
        let mut multiline_indent: usize = 0;

        for line in yaml_content.lines() {
            let trimmed = line.trim();
            let leading_spaces = line.len() - line.trim_start().len();

            if parse_mode == ParseMode::MultiLineDesc {
                if leading_spaces > multiline_indent && !trimmed.is_empty() {
                    if let Some(ref mut desc) = current_desc {
                        if !desc.is_empty() {
                            desc.push(' ');
                        }
                        desc.push_str(trimmed);
                    }
                    continue;
                } else {
                    parse_mode = ParseMode::Normal;
                }
            }

            if trimmed.starts_with("- label:") {
                if let (Some(label), desc) = (current_label.take(), current_desc.take()) {
                    let desc = desc.unwrap_or_default();
                    let desc = if desc == "|" || desc == "-" || desc == ">" {
                        String::new()
                    } else {
                        desc
                    };
                    if !desc.is_empty() {
                        tasks.push(TaskSuggestion {
                            label,
                            description: desc,
                            time_estimate: current_time.take(),
                            complexity: current_complexity.take(),
                            affected_files: std::mem::take(&mut current_files),
                        });
                    }
                }
                parse_mode = ParseMode::Normal;
                current_label = Some(trimmed.trim_start_matches("- label:").trim().trim_matches('"').to_string());
            } else if trimmed.starts_with("description:") {
                let value = trimmed.trim_start_matches("description:").trim();
                if value == "|" || value == ">" || value == "-" || value.is_empty() {
                    parse_mode = ParseMode::MultiLineDesc;
                    multiline_indent = leading_spaces;
                    current_desc = Some(String::new());
                } else {
                    parse_mode = ParseMode::Normal;
                    current_desc = Some(value.trim_matches('"').to_string());
                }
            } else if trimmed.starts_with("time_estimate:") {
                parse_mode = ParseMode::Normal;
                current_time = Some(trimmed.trim_start_matches("time_estimate:").trim().trim_matches('"').to_string());
            } else if trimmed.starts_with("complexity:") {
                parse_mode = ParseMode::Normal;
                current_complexity = Some(trimmed.trim_start_matches("complexity:").trim().to_string());
            } else if trimmed.starts_with("affected_files:") {
                parse_mode = ParseMode::FilesList;
            } else if parse_mode == ParseMode::FilesList && trimmed.starts_with("- ") {
                let file = trimmed.trim_start_matches("- ").trim().to_string();
                if !file.is_empty() {
                    current_files.push(file);
                }
            } else if !trimmed.starts_with("-") && !trimmed.is_empty() && !trimmed.starts_with("#") {
                if parse_mode == ParseMode::FilesList {
                    parse_mode = ParseMode::Normal;
                }
            }
        }

        // Don't forget the last one
        if let (Some(label), desc) = (current_label, current_desc) {
            let desc = desc.unwrap_or_default();
            let desc = if desc == "|" || desc == "-" || desc == ">" {
                String::new()
            } else {
                desc
            };
            if !desc.is_empty() {
                tasks.push(TaskSuggestion {
                    label,
                    description: desc,
                    time_estimate: current_time,
                    complexity: current_complexity,
                    affected_files: current_files,
                });
            }
        }

        Ok(tasks)
    }
}
