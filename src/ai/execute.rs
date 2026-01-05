//! Task execution using the same streaming SDK pattern as analysis
//!
//! This runs selected cards through an agentic loop, streaming output
//! back to the UI just like the Palace Loop analysis phase.

use anyhow::Result;
use anthropic_ffi::{models, Client as AnthropicClient, StreamEvent};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use winit::event_loop::EventLoopProxy;

use crate::app::AppEvent;
use crate::state::{ExecuteOption, SuggestionCard};

/// Execute tasks using the streaming agentic loop
pub struct TaskExecutor {
    client: AnthropicClient,
    model: String,
    project_path: std::path::PathBuf,
    executor: ExecuteOption,
}

impl TaskExecutor {
    pub fn new(project_path: std::path::PathBuf, executor: ExecuteOption) -> Result<Self> {
        // Configure based on executor option
        let model = match executor {
            ExecuteOption::Claude => {
                // Use Anthropic API with Claude
                std::env::var("ANTHROPIC_MODEL")
                    .unwrap_or_else(|_| models::CLAUDE_SONNET_4_5.to_string())
            }
            ExecuteOption::ZAi | ExecuteOption::ZAiTurbo => {
                // Use Z.ai API - they support Claude models
                std::env::var("ANTHROPIC_MODEL")
                    .unwrap_or_else(|_| models::CLAUDE_SONNET_4_5.to_string())
            }
        };

        tracing::info!("Creating TaskExecutor with {:?}, model: {}", executor, model);

        let client = AnthropicClient::new(None)?;
        Ok(Self { client, model, project_path, executor })
    }

    /// Execute selected cards, streaming output to the event loop
    pub fn execute(
        &self,
        cards: &[SuggestionCard],
        event_proxy: EventLoopProxy<AppEvent>,
    ) -> Result<()> {
        let total = cards.len();

        for (idx, card) in cards.iter().enumerate() {
            // Progress update
            let _ = event_proxy.send_event(AppEvent::ExecutionProgress {
                current: idx,
                total,
            });

            // Timestamp + task header (goes to thought column)
            let timestamp = chrono_lite_timestamp();
            let _ = event_proxy.send_event(AppEvent::ExecutionThought(format!(
                "[{}] 📋 Task {}/{}: {}",
                timestamp, idx + 1, total, card.title
            )));

            // Execute this card
            if let Err(e) = self.execute_card(card, &event_proxy) {
                let _ = event_proxy.send_event(AppEvent::ExecutionError(format!(
                    "Task '{}' failed: {}", card.title, e
                )));
                return Err(e);
            }

            let _ = event_proxy.send_event(AppEvent::ExecutionThought(format!(
                "[{}] ✅ Completed: {}", chrono_lite_timestamp(), card.title
            )));
        }

        let _ = event_proxy.send_event(AppEvent::ExecutionComplete);
        Ok(())
    }

    /// Execute a single card using the agentic loop
    fn execute_card(
        &self,
        card: &SuggestionCard,
        event_proxy: &EventLoopProxy<AppEvent>,
    ) -> Result<()> {
        tracing::info!("execute_card: {} - {}", card.title, card.description);

        let system = format!(
            r#"You are an AI assistant executing a specific task.

Project: {}
Working directory: {}

CRITICAL RULE: When ANYTHING is unclear, ambiguous, or not specified - USE THE ask_user TOOL.

NEVER GUESS. NEVER ASSUME. The user is RIGHT THERE - ask them!

Keep questions FOCUSED. Don't cram multiple unrelated things into one question.
After each answer, you can ask follow-up questions based on what you learned.
This "20 questions" pattern lets you narrow down progressively.

The ask_user tool shows a survey UI. Use it liberally.
- question: The question to ask
- header: Short label for context
- options: Array of choices, or empty for free-form input
- multi_select: true if user can pick multiple options

Execute the task precisely. Use available tools as needed.
When done, the task is complete - no need to call a "done" tool."#,
            self.project_path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "project".to_string()),
            self.project_path.display()
        );

        // Build the task prompt
        let prompt = if let Some(cmd) = &card.command {
            format!(
                "Execute this task:\n\n## {}\n\n{}\n\nSuggested command: `{}`",
                card.title, card.description, cmd
            )
        } else {
            format!(
                "Execute this task:\n\n## {}\n\n{}",
                card.title, card.description
            )
        };

        // Standard Claude Code tools
        let tools_json = include_str!("tools.json");

        // State for tracking tool calls
        let pending_tools: Arc<Mutex<VecDeque<(String, String)>>> = Arc::new(Mutex::new(VecDeque::new()));
        let current_tool_input = Arc::new(Mutex::new(String::new()));
        let current_tool_name = Arc::new(Mutex::new(String::new()));
        let mut in_tool = false;

        let proxy = event_proxy.clone();
        let pending_clone = pending_tools.clone();
        let input_clone = current_tool_input.clone();
        let name_clone = current_tool_name.clone();

        // Buffer for accumulating text before sending
        let text_buffer = Arc::new(Mutex::new(String::new()));
        let text_buffer_clone = text_buffer.clone();

        // Create tool context with event proxy for ask_user
        let tool_ctx = Arc::new(ToolContext {
            project_path: self.project_path.clone(),
            event_proxy: event_proxy.clone(),
        });
        let tool_ctx_clone = tool_ctx.clone();

        self.client.agentic_loop(
            &self.model,
            Some(&system),
            &prompt,
            8192,
            tools_json,
            move |event| {
                match event {
                    StreamEvent::Text(text) => {
                        // Accumulate text for the right column (Claude's commentary)
                        let mut buffer = text_buffer_clone.lock().unwrap();
                        buffer.push_str(&text);

                        // Flush complete lines to thought log
                        while let Some(newline_pos) = buffer.find('\n') {
                            let line = buffer[..newline_pos].to_string();
                            *buffer = buffer[newline_pos + 1..].to_string();
                            if !line.trim().is_empty() {
                                let timestamp = chrono_lite_timestamp();
                                let _ = proxy.send_event(AppEvent::ExecutionThought(
                                    format!("[{}] {}", timestamp, line.trim())
                                ));
                            }
                        }
                    }
                    StreamEvent::ToolUseStart(tool_info) => {
                        in_tool = true;
                        let name = tool_info.split(':').last().unwrap_or(&tool_info);
                        *name_clone.lock().unwrap() = name.to_string();
                        input_clone.lock().unwrap().clear();
                    }
                    StreamEvent::ToolUseInput(json) => {
                        if in_tool {
                            input_clone.lock().unwrap().push_str(&json);
                        }
                    }
                    StreamEvent::ToolUseEnd => {
                        if in_tool {
                            let name = name_clone.lock().unwrap().clone();
                            let input = input_clone.lock().unwrap().clone();
                            pending_clone.lock().unwrap().push_back((name, input));
                            in_tool = false;
                        }
                    }
                    StreamEvent::ToolResult(result) => {
                        if let Some((name, input)) = pending_clone.lock().unwrap().pop_front() {
                            // Format: [timestamp] icon action  summary [dot]
                            let timestamp = chrono_lite_timestamp();
                            let (maybe_dot, summary) = parse_tool_result(&name, &result);
                            let (icon, action) = format_tool_call(&name, &input);
                            let is_error = maybe_dot.is_some() && summary.contains("error");

                            // Build the formatted line
                            let line = if let Some(dot) = maybe_dot {
                                format!("[{}] {} {}  {} {}", timestamp, icon, action, summary, dot)
                            } else {
                                format!("[{}] {} {}  {}", timestamp, icon, action, summary)
                            };

                            // Include error flag for coloring in renderer
                            let _ = proxy.send_event(AppEvent::ExecutionToolCall(
                                if is_error { format!("ERR:{}", line) } else { line }
                            ));
                        }
                    }
                    StreamEvent::Thinking(thought) => {
                        // Thinking goes to right column - no truncation, UI wraps
                        let timestamp = chrono_lite_timestamp();
                        let _ = proxy.send_event(AppEvent::ExecutionThought(
                            format!("[{}] 💭 {}", timestamp, thought.replace('\n', " "))
                        ));
                    }
                    StreamEvent::Done => {
                        // Flush any remaining text buffer
                        let buffer = text_buffer_clone.lock().unwrap();
                        if !buffer.trim().is_empty() {
                            let timestamp = chrono_lite_timestamp();
                            let _ = proxy.send_event(AppEvent::ExecutionThought(
                                format!("[{}] {}", timestamp, buffer.trim())
                            ));
                        }
                    }
                    _ => {}
                }
            },
            move |tool_name, tool_input| {
                tracing::debug!("Executing tool: {} with input: {}", tool_name, tool_input);
                let result = execute_tool(&tool_ctx_clone, tool_name, tool_input);
                tracing::debug!("Tool result: {}", &result[..result.len().min(200)]);
                result
            },
        )?;

        tracing::info!("execute_card complete: {}", card.title);
        Ok(())
    }
}

/// Tool icons for visual identification
fn tool_icon(name: &str) -> &'static str {
    match name.to_lowercase().as_str() {
        "bash" => "💻",
        "read_file" | "read" => "📖",
        "write_file" | "write" => "✏️",
        "edit_file" | "edit" => "✏️",
        "glob" => "🔍",
        "grep" => "🔍",
        "ask_user" => "❓",
        _ => "🔧",
    }
}

/// Format tool call with emoji icon and clean display
/// Returns (icon, action_text) - no "ToolName()" wrapper, no truncation (UI wraps)
fn format_tool_call(name: &str, input: &str) -> (String, String) {
    let parsed: serde_json::Value = serde_json::from_str(input).unwrap_or_default();
    let icon = tool_icon(name).to_string();

    let action = match name.to_lowercase().as_str() {
        "bash" => {
            parsed["command"].as_str().unwrap_or("...").to_string()
        }
        "read_file" | "read" => {
            parsed["file_path"].as_str()
                .or_else(|| parsed["path"].as_str())
                .unwrap_or("...").to_string()
        }
        "write_file" | "write" => {
            let path = parsed["file_path"].as_str()
                .or_else(|| parsed["path"].as_str())
                .unwrap_or("...");
            format!("write {}", path)
        }
        "edit_file" | "edit" => {
            let path = parsed["file_path"].as_str()
                .or_else(|| parsed["path"].as_str())
                .unwrap_or("...");
            format!("edit {}", path)
        }
        "glob" => {
            parsed["pattern"].as_str().unwrap_or("...").to_string()
        }
        "grep" => {
            parsed["pattern"].as_str().unwrap_or("...").to_string()
        }
        "ask_user" => {
            parsed["question"].as_str().unwrap_or("...").to_string()
        }
        _ => name.to_string(),
    };

    (icon, action)
}

/// Parse tool result and return (status_dot, summary)
/// Status dots: only for bash (always), read (on failure only)
/// Dot goes to the RIGHT of the summary
fn parse_tool_result(name: &str, result: &str) -> (Option<&'static str>, String) {
    const GREEN_DOT: &str = "●";
    const RED_DOT: &str = "●"; // Same char, colored differently in renderer

    // Check for explicit error markers
    let is_error = result.contains("error:") || result.contains("Error:") ||
                   result.contains("FAILED") || result.contains("failed:") ||
                   result.starts_with("Error") || result.starts_with("Failed");

    match name.to_lowercase().as_str() {
        "bash" => {
            // Cargo test parsing
            if result.contains("test result:") {
                if let Some(summary) = parse_cargo_test_result(result) {
                    let dot = if summary.contains("FAILED") { RED_DOT } else { GREEN_DOT };
                    return (Some(dot), summary);
                }
            }
            // Cargo build parsing
            if result.contains("Compiling") || result.contains("Finished") {
                if let Some(summary) = parse_cargo_build_result(result) {
                    let dot = if is_error { RED_DOT } else { GREEN_DOT };
                    return (Some(dot), summary);
                }
            }
            // Generic bash result
            let lines = result.lines().count();
            if is_error {
                (Some(RED_DOT), format!("error, {} lines", lines))
            } else if result.trim().is_empty() {
                (Some(GREEN_DOT), "ok".to_string())
            } else {
                (Some(GREEN_DOT), format!("{} lines", lines))
            }
        }
        "read_file" | "read" => {
            // Only show dot on failure
            if is_error {
                (Some(RED_DOT), "failed to read".to_string())
            } else {
                let lines = result.lines().count();
                (None, format!("{} lines", lines))
            }
        }
        "write_file" | "write" => {
            if is_error {
                (Some(RED_DOT), "write failed".to_string())
            } else {
                (None, "written".to_string())
            }
        }
        "edit_file" | "edit" => {
            if is_error {
                (Some(RED_DOT), "edit failed".to_string())
            } else {
                (None, "edited".to_string())
            }
        }
        "glob" => {
            let files = result.lines().filter(|l| !l.is_empty()).count();
            (None, format!("{} files", files))
        }
        "grep" => {
            let matches = result.lines().filter(|l| !l.is_empty()).count();
            (None, format!("{} matches", matches))
        }
        "ask_user" => {
            (None, "answered".to_string())
        }
        _ => {
            if is_error {
                (Some(RED_DOT), "error".to_string())
            } else {
                (None, "ok".to_string())
            }
        }
    }
}

/// Parse cargo test output for test count summary
fn parse_cargo_test_result(output: &str) -> Option<String> {
    // Look for "test result: ok. X passed; Y failed; Z ignored"
    for line in output.lines() {
        if line.contains("test result:") {
            // Extract numbers
            let passed = extract_number_before(line, "passed").unwrap_or(0);
            let failed = extract_number_before(line, "failed").unwrap_or(0);

            if failed > 0 {
                return Some(format!("{}/{} tests, {} FAILED", passed, passed + failed, failed));
            } else {
                return Some(format!("{}/{} tests passed", passed, passed));
            }
        }
    }
    None
}

/// Parse cargo build output for summary
fn parse_cargo_build_result(output: &str) -> Option<String> {
    // Count errors and warnings
    let errors = output.matches("error[E").count() + output.matches("error:").count();
    let warnings = output.matches("warning:").count();

    if errors > 0 {
        Some(format!("{} errors, {} warnings", errors, warnings))
    } else if output.contains("Finished") {
        if warnings > 0 {
            Some(format!("built, {} warnings", warnings))
        } else {
            Some("built successfully".to_string())
        }
    } else {
        None
    }
}

/// Extract number appearing before a word in a string
fn extract_number_before(s: &str, word: &str) -> Option<usize> {
    if let Some(pos) = s.find(word) {
        let before = &s[..pos];
        // Find last number in the string before the word
        let num_str: String = before.chars().rev()
            .take_while(|c| c.is_ascii_digit() || *c == ' ')
            .collect::<String>()
            .chars().rev().collect();
        num_str.trim().parse().ok()
    } else {
        None
    }
}

/// Context for tool execution (includes event proxy for ask_user)
struct ToolContext {
    project_path: std::path::PathBuf,
    event_proxy: EventLoopProxy<AppEvent>,
}

/// Execute a tool and return result
fn execute_tool(ctx: &ToolContext, tool_name: &str, tool_input: &str) -> String {
    let input: serde_json::Value = match serde_json::from_str(tool_input) {
        Ok(v) => v,
        Err(e) => return format!("Invalid JSON: {}", e),
    };

    match tool_name {
        "bash" | "Bash" => {
            let cmd = input["command"].as_str().unwrap_or("");
            execute_bash(&ctx.project_path, cmd)
        }
        "read_file" | "Read" => {
            let path = input["file_path"].as_str()
                .or_else(|| input["path"].as_str())
                .unwrap_or("");
            read_file(&ctx.project_path, path)
        }
        "write_file" | "Write" => {
            let path = input["file_path"].as_str()
                .or_else(|| input["path"].as_str())
                .unwrap_or("");
            let content = input["content"].as_str().unwrap_or("");
            write_file(&ctx.project_path, path, content)
        }
        "edit_file" | "Edit" => {
            let path = input["file_path"].as_str()
                .or_else(|| input["path"].as_str())
                .unwrap_or("");
            let old = input["old_string"].as_str().unwrap_or("");
            let new = input["new_string"].as_str().unwrap_or("");
            edit_file(&ctx.project_path, path, old, new)
        }
        "glob" | "Glob" => {
            let pattern = input["pattern"].as_str().unwrap_or("*");
            glob_files(&ctx.project_path, pattern)
        }
        "grep" | "Grep" => {
            let pattern = input["pattern"].as_str().unwrap_or("");
            let path = input["path"].as_str();
            grep_files(&ctx.project_path, pattern, path)
        }
        "ask_user" => {
            execute_ask_user(ctx, &input)
        }
        _ => format!("Unknown tool: {}", tool_name),
    }
}

/// Execute ask_user tool - show survey UI and wait for response
fn execute_ask_user(ctx: &ToolContext, input: &serde_json::Value) -> String {
    use crate::state::{SurveyOption, SurveyResponse};
    use std::sync::mpsc;

    let question = input["question"].as_str().unwrap_or("").to_string();
    let header = input["header"].as_str().unwrap_or("Question").to_string();
    let multi_select = input["multi_select"].as_bool().unwrap_or(false);

    // Parse options
    let options: Vec<SurveyOption> = input["options"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|opt| SurveyOption {
                    label: opt["label"].as_str().unwrap_or("").to_string(),
                    description: opt["description"].as_str().unwrap_or("").to_string(),
                })
                .collect()
        })
        .unwrap_or_default();

    // Options can be empty - user can still provide custom input via "Other"

    // Create channel for response
    let (tx, rx) = mpsc::channel();

    // Send survey request to UI
    let _ = ctx.event_proxy.send_event(AppEvent::SurveyRequest {
        question: question.clone(),
        header,
        options: options.clone(),
        multi_select,
        response_tx: tx,
    });

    // Block waiting for user response
    match rx.recv() {
        Ok(SurveyResponse::Selected(indices)) => {
            if indices.is_empty() {
                "User made no selection".to_string()
            } else if indices.len() == 1 {
                let idx = indices[0];
                if idx < options.len() {
                    format!("User selected: {}", options[idx].label)
                } else {
                    "User selected: Other (custom)".to_string()
                }
            } else {
                let labels: Vec<_> = indices.iter()
                    .filter_map(|&i| options.get(i).map(|o| o.label.as_str()))
                    .collect();
                format!("User selected: {}", labels.join(", "))
            }
        }
        Ok(SurveyResponse::Custom(text)) => {
            format!("User provided custom answer: {}", text)
        }
        Ok(SurveyResponse::Cancelled) => {
            "User cancelled the survey".to_string()
        }
        Err(_) => {
            "Error: Survey response channel closed".to_string()
        }
    }
}

fn execute_bash(project_path: &Path, cmd: &str) -> String {
    match std::process::Command::new("bash")
        .arg("-c")
        .arg(cmd)
        .current_dir(project_path)
        .output()
    {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            if output.status.success() {
                format!("{}{}", stdout, stderr)
            } else {
                format!("Exit {}: {}{}",
                    output.status.code().unwrap_or(-1), stdout, stderr)
            }
        }
        Err(e) => format!("Failed: {}", e),
    }
}

fn read_file(project_path: &Path, path: &str) -> String {
    let full = if std::path::Path::new(path).is_absolute() {
        std::path::PathBuf::from(path)
    } else {
        project_path.join(path)
    };
    match std::fs::read_to_string(&full) {
        Ok(content) => content,
        Err(e) => format!("Failed to read {}: {}", path, e),
    }
}

fn write_file(project_path: &Path, path: &str, content: &str) -> String {
    let full = if std::path::Path::new(path).is_absolute() {
        std::path::PathBuf::from(path)
    } else {
        project_path.join(path)
    };
    if let Some(parent) = full.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&full, content) {
        Ok(_) => format!("Wrote {} bytes to {}", content.len(), path),
        Err(e) => format!("Failed to write {}: {}", path, e),
    }
}

fn edit_file(project_path: &Path, path: &str, old: &str, new: &str) -> String {
    let full = if std::path::Path::new(path).is_absolute() {
        std::path::PathBuf::from(path)
    } else {
        project_path.join(path)
    };
    match std::fs::read_to_string(&full) {
        Ok(content) => {
            if !content.contains(old) {
                return format!("old_string not found in {}", path);
            }
            let updated = content.replacen(old, new, 1);
            match std::fs::write(&full, &updated) {
                Ok(_) => format!("Edited {}", path),
                Err(e) => format!("Failed to write {}: {}", path, e),
            }
        }
        Err(e) => format!("Failed to read {}: {}", path, e),
    }
}

fn glob_files(project_path: &Path, pattern: &str) -> String {
    let full_pattern = project_path.join(pattern);
    match glob::glob(full_pattern.to_str().unwrap_or("")) {
        Ok(paths) => {
            let files: Vec<_> = paths
                .filter_map(|p| p.ok())
                .map(|p| p.strip_prefix(project_path)
                    .map(|r| r.display().to_string())
                    .unwrap_or_else(|_| p.display().to_string()))
                .collect();
            if files.is_empty() {
                "No matches".to_string()
            } else {
                files.join("\n")
            }
        }
        Err(e) => format!("Glob error: {}", e),
    }
}

fn grep_files(project_path: &Path, pattern: &str, path: Option<&str>) -> String {
    let search_path = path.map(|p| project_path.join(p))
        .unwrap_or_else(|| project_path.to_path_buf());

    // Use ripgrep if available, otherwise basic grep
    let output = std::process::Command::new("rg")
        .args(["--line-number", "--no-heading", pattern])
        .current_dir(&search_path)
        .output();

    match output {
        Ok(out) if out.status.success() => {
            String::from_utf8_lossy(&out.stdout).to_string()
        }
        _ => {
            // Fallback to grep
            match std::process::Command::new("grep")
                .args(["-rn", pattern, "."])
                .current_dir(&search_path)
                .output()
            {
                Ok(out) => String::from_utf8_lossy(&out.stdout).to_string(),
                Err(e) => format!("Grep failed: {}", e),
            }
        }
    }
}

/// Simple timestamp without chrono dependency
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Just show HH:MM:SS in UTC-ish
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let secs = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, secs)
}

/// Spawn execution in a background thread
pub fn spawn_execution(
    executor: ExecuteOption,
    project_path: std::path::PathBuf,
    cards: Vec<SuggestionCard>,
    event_proxy: EventLoopProxy<AppEvent>,
) {
    tracing::info!(
        "spawn_execution: {} cards with {:?}",
        cards.len(),
        executor
    );

    std::thread::spawn(move || {
        tracing::info!("Execution thread started with {:?}", executor);

        match TaskExecutor::new(project_path, executor) {
            Ok(exec) => {
                tracing::info!("TaskExecutor created successfully");

                if let Err(e) = exec.execute(&cards, event_proxy.clone()) {
                    tracing::error!("Execution failed: {}", e);
                    let _ = event_proxy.send_event(AppEvent::ExecutionError(e.to_string()));
                }
            }
            Err(e) => {
                tracing::error!("Failed to create executor: {}", e);
                let _ = event_proxy.send_event(AppEvent::ExecutionError(format!(
                    "Failed to create executor: {}", e
                )));
            }
        }
    });
}
