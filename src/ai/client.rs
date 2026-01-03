use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Anthropic API client
pub struct AnthropicClient {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    system: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    text: Option<String>,
}

/// Parsed task suggestion from Claude
#[derive(Debug, Clone)]
pub struct TaskSuggestion {
    pub label: String,
    pub description: String,
}

impl AnthropicClient {
    pub fn new(base_url: &str, model: &str) -> Result<Self> {
        // Try ZAI_API_KEY first (for z.ai), fall back to ANTHROPIC_API_KEY
        let api_key = std::env::var("ZAI_API_KEY")
            .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
            .map_err(|_| anyhow!("ZAI_API_KEY or ANTHROPIC_API_KEY environment variable not set"))?;

        Ok(Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model: model.to_string(),
        })
    }

    /// Ask Claude what tasks to work on
    pub async fn generate_tasks(&self, project_path: &Path) -> Result<Vec<TaskSuggestion>> {
        // Gather context
        let context = self.gather_context(project_path).await?;

        let system = r#"You are analyzing a codebase to suggest actionable tasks.
Respond ONLY with a YAML list of suggested actions. No other text.

Format:
```yaml
actions:
  - label: Short action title
    description: What this involves
  - label: Another action
    description: Details about it
```

Be specific and actionable. Suggest as many tasks as make sense - no artificial limits."#;

        let prompt = format!(
            "Here's the current state of the project at {}:\n\n{}\n\nWhat tasks should I work on next?",
            project_path.display(),
            context
        );

        let response = self.chat(&prompt, Some(system)).await?;
        self.parse_yaml_tasks(&response)
    }

    /// Execute a task by sending it to Claude
    pub async fn execute_task(&self, task: &TaskSuggestion, project_path: &Path) -> Result<String> {
        let context = self.gather_context(project_path).await?;

        let system = r#"You are a coding assistant executing a task.
Analyze the codebase and perform the requested task.
Be thorough but concise. Show what you're doing."#;

        let prompt = format!(
            "Project: {}\n\nContext:\n{}\n\nTask: {}\nDescription: {}\n\nExecute this task.",
            project_path.display(),
            context,
            task.label,
            task.description
        );

        self.chat(&prompt, Some(system)).await
    }

    async fn chat(&self, prompt: &str, system: Option<&str>) -> Result<String> {
        let request = ChatRequest {
            model: self.model.clone(),
            max_tokens: 8192,
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            system: system.map(|s| s.to_string()),
        };

        let url = format!("{}/v1/messages", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("API error {}: {}", status, body));
        }

        let chat_response: ChatResponse = response.json().await?;

        chat_response
            .content
            .into_iter()
            .filter_map(|c| c.text)
            .next()
            .ok_or_else(|| anyhow!("No text in response"))
    }

    async fn gather_context(&self, project_path: &Path) -> Result<String> {
        let mut context = String::new();

        // File tree (limited depth)
        context.push_str("## File Structure\n```\n");
        if let Ok(output) = tokio::process::Command::new("find")
            .args([
                project_path.to_str().unwrap_or("."),
                "-type", "f",
                "-not", "-path", "*/.git/*",
                "-not", "-path", "*/node_modules/*",
                "-not", "-path", "*/target/*",
                "-not", "-path", "*/__pycache__/*",
                "-not", "-path", "*/.palace/*",
            ])
            .output()
            .await
        {
            let files = String::from_utf8_lossy(&output.stdout);
            // Limit to first 100 files
            for (i, line) in files.lines().enumerate() {
                if i >= 100 {
                    context.push_str("... (truncated)\n");
                    break;
                }
                context.push_str(line);
                context.push('\n');
            }
        }
        context.push_str("```\n\n");

        // Key files content
        let key_files = ["README.md", "SPEC.md", "Cargo.toml", "package.json", "pyproject.toml"];
        for filename in key_files {
            let file_path = project_path.join(filename);
            if file_path.exists() {
                if let Ok(content) = tokio::fs::read_to_string(&file_path).await {
                    context.push_str(&format!("## {}\n```\n", filename));
                    // Limit file content
                    let truncated: String = content.chars().take(2000).collect();
                    context.push_str(&truncated);
                    if content.len() > 2000 {
                        context.push_str("\n... (truncated)");
                    }
                    context.push_str("\n```\n\n");
                }
            }
        }

        Ok(context)
    }

    fn parse_yaml_tasks(&self, response: &str) -> Result<Vec<TaskSuggestion>> {
        let mut tasks = Vec::new();

        // Extract YAML block if wrapped in ```yaml
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

        // Simple YAML parsing for actions list
        let mut current_label: Option<String> = None;
        let mut current_desc: Option<String> = None;

        for line in yaml_content.lines() {
            let trimmed = line.trim();

            if trimmed.starts_with("- label:") {
                // Save previous task if exists
                if let (Some(label), Some(desc)) = (current_label.take(), current_desc.take()) {
                    tasks.push(TaskSuggestion {
                        label,
                        description: desc,
                    });
                }
                current_label = Some(trimmed.trim_start_matches("- label:").trim().to_string());
            } else if trimmed.starts_with("description:") {
                current_desc = Some(trimmed.trim_start_matches("description:").trim().to_string());
            }
        }

        // Don't forget the last one
        if let (Some(label), Some(desc)) = (current_label, current_desc) {
            tasks.push(TaskSuggestion {
                label,
                description: desc,
            });
        }

        Ok(tasks)
    }
}
