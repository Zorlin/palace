//! Streaming executor that wraps the Go Anthropic SDK
//!
//! This module handles:
//! - Spawning the Go SDK wrapper binary
//! - Streaming output in realtime to the TUI
//! - Processing events from the SDK

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

/// Events from the Claude execution stream
#[derive(Debug, Clone)]
pub enum ExecutionEvent {
    /// Text output from Claude
    Text(String),

    /// Claude is working on something
    Progress { tool: String, detail: String },

    /// Permission request - needs user response
    Permission(PermissionRequest),

    /// Claude is asking a question (via tool use)
    Question {
        id: String,
        question: String,
        options: Vec<DialogueOption>,
        multi_select: bool,
    },

    /// Execution complete
    Done,

    /// Error occurred
    Error(String),
}

/// A permission request from Claude
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionRequest {
    pub id: String,
    pub tool: String,
    pub description: String,
    pub target: Option<String>,
}

/// User response to a permission request
#[derive(Debug, Clone, Serialize)]
pub enum PermissionResponse {
    Allow,
    AlwaysAllow,
    Deny { reason: String },
}

/// A dialogue option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueOption {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    pub hint: Option<String>,
}

/// Request sent to Go SDK wrapper
#[derive(Debug, Serialize)]
struct SdkRequest {
    prompt: String,
    system: Option<String>,
    model: Option<String>,
}

/// Event from Go SDK wrapper
#[derive(Debug, Deserialize)]
struct SdkEvent {
    #[serde(rename = "type")]
    event_type: String,
    text: Option<String>,
    error: Option<String>,
    tool: Option<String>,
    detail: Option<String>,
}

/// User response to prompts
#[derive(Debug, Clone)]
pub enum UserResponse {
    Permission(String, PermissionResponse),
    Dialogue(String, Vec<String>),
    Cancel,
}

/// Streaming executor using Go SDK
pub struct Executor {
    project_path: std::path::PathBuf,
    event_tx: mpsc::Sender<ExecutionEvent>,
    response_rx: mpsc::Receiver<UserResponse>,
}

impl Executor {
    pub fn new(
        project_path: &Path,
        event_tx: mpsc::Sender<ExecutionEvent>,
        response_rx: mpsc::Receiver<UserResponse>,
    ) -> Self {
        Self {
            project_path: project_path.to_path_buf(),
            event_tx,
            response_rx,
        }
    }

    /// Execute tasks using Go SDK wrapper
    pub async fn execute(&mut self, tasks: Vec<super::TaskSuggestion>) -> Result<()> {
        // Build the prompt
        let task_list: Vec<String> = tasks
            .iter()
            .enumerate()
            .map(|(i, t)| format!("{}. **{}**\n   {}", i + 1, t.label, t.description))
            .collect();

        let prompt = format!(
            "Execute these tasks in order:\n\n{}\n\nWork through each task, showing progress.",
            task_list.join("\n\n")
        );

        let system = Some("You are a coding assistant. Execute the requested tasks carefully and show your progress.".to_string());

        // Spawn Go SDK wrapper
        let mut child = self.spawn_sdk(&prompt, system.as_deref()).await?;

        // Process the stream
        self.process_stream(&mut child).await?;

        // Wait for completion
        let status = child.wait().await?;
        if !status.success() {
            self.event_tx
                .send(ExecutionEvent::Error(format!(
                    "SDK exited with code {:?}",
                    status.code()
                )))
                .await
                .ok();
        }

        self.event_tx.send(ExecutionEvent::Done).await.ok();
        Ok(())
    }

    async fn spawn_sdk(&self, prompt: &str, system: Option<&str>) -> Result<Child> {
        // Find the Go SDK binary
        let sdk_path = self.find_sdk_binary()?;

        let mut cmd = Command::new(&sdk_path);
        cmd.current_dir(&self.project_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Pass through API key environment variables if set
        if let Ok(key) = std::env::var("ZAI_API_KEY") {
            cmd.env("ZAI_API_KEY", key);
        }
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            cmd.env("ANTHROPIC_API_KEY", key);
        }

        let mut child = cmd.spawn()?;

        // Send the request
        let request = SdkRequest {
            prompt: prompt.to_string(),
            system: system.map(|s| s.to_string()),
            model: None, // Use default
        };

        let stdin = child.stdin.as_mut().ok_or_else(|| anyhow!("No stdin"))?;
        let mut request_json = serde_json::to_string(&request)?;
        request_json.push('\n');
        stdin.write_all(request_json.as_bytes()).await?;
        stdin.flush().await?;

        Ok(child)
    }

    fn find_sdk_binary(&self) -> Result<std::path::PathBuf> {
        // Check common locations
        let candidates = [
            // In target directory (dev build)
            self.project_path.join("target/palace-sdk"),
            // In same directory as palace binary
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
            "palace-sdk binary not found. Build it with: cd go && go build -o ../target/palace-sdk ."
        ))
    }

    async fn process_stream(&mut self, child: &mut Child) -> Result<()> {
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("No stdout"))?;
        let stderr = child.stderr.take().ok_or_else(|| anyhow!("No stderr"))?;

        let mut reader = BufReader::new(stdout).lines();
        let mut stderr_reader = BufReader::new(stderr).lines();

        // Spawn task to read stderr
        let event_tx = self.event_tx.clone();
        tokio::spawn(async move {
            while let Ok(Some(line)) = stderr_reader.next_line().await {
                if !line.trim().is_empty() {
                    event_tx
                        .send(ExecutionEvent::Error(format!("stderr: {}", line)))
                        .await
                        .ok();
                }
            }
        });

        loop {
            tokio::select! {
                // Read from SDK
                line = reader.next_line() => {
                    match line? {
                        Some(line) => {
                            if let Err(e) = self.handle_line(&line).await {
                                tracing::warn!("Error handling line: {}", e);
                            }
                        }
                        None => break, // EOF
                    }
                }

                // Handle user responses (for future permission handling)
                response = self.response_rx.recv() => {
                    match response {
                        Some(UserResponse::Cancel) => {
                            // Kill the process
                            break;
                        }
                        _ => {
                            // Other responses not implemented for direct SDK mode
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn handle_line(&self, line: &str) -> Result<()> {
        // Parse JSON event from Go SDK
        let event: SdkEvent = serde_json::from_str(line)
            .map_err(|e| anyhow!("Failed to parse SDK event: {} - line: {}", e, line))?;

        match event.event_type.as_str() {
            "text" => {
                if let Some(text) = event.text {
                    self.event_tx.send(ExecutionEvent::Text(text)).await.ok();
                }
            }
            "tool_start" => {
                if let (Some(tool), Some(detail)) = (event.tool, event.detail) {
                    self.event_tx
                        .send(ExecutionEvent::Progress { tool, detail })
                        .await
                        .ok();
                }
            }
            "error" => {
                if let Some(error) = event.error {
                    self.event_tx
                        .send(ExecutionEvent::Error(error))
                        .await
                        .ok();
                }
            }
            "done" => {
                // Done event handled by caller
            }
            _ => {
                tracing::trace!("Unknown event type: {}", event.event_type);
            }
        }

        Ok(())
    }
}
