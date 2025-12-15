//! Palace Binary - Full RHSI CLI with advanced task selection
//!
//! This binary provides the full `palace` command with:
//! - Interactive TUI task selection with ratatui
//! - Simple text-based selection mode
//! - Generate mode to create tasks.md
//! - Action mode to spawn RHSI loops
//! - Full Mistral and Turbo integration

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tracing::warn;

mod context;
mod menu;
mod tui_menu;
#[allow(dead_code)]
mod session;

/// Palace - RHSI Development Assistant (Full Feature Set)
#[derive(Parser)]
#[command(name = "palace")]
#[command(author = "Riff Labs <wings@riff.cc>")]
#[command(version)]
#[command(about = "Palace - Full RHSI Development Assistant", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Ask Claude what to do next (RHSI core) with advanced task selection
    Next {
        /// User guidance for what to work on
        #[arg(trailing_var_arg = true)]
        guidance: Vec<String>,

        /// Use Claude Opus (highest quality)
        #[arg(long)]
        opus: bool,

        /// Use GLM-4.6 (cost savings)
        #[arg(long)]
        glm: bool,

        /// Use GLM-4.6v (vision model)
        #[arg(long)]
        glmv: bool,

        /// Use Devstral 2 (123B via Mistral API)
        #[arg(long, alias = "mistral")]
        devstral: bool,

        /// Use Devstral Small 2 (24B local)
        #[arg(long, alias = "smol")]
        devstral_small: bool,

        /// Use local Ollama backend (for devstral-small)
        #[arg(long)]
        ollama: bool,

        /// Mixed mode: GLM for prompter, Claude for swarm
        #[arg(long)]
        mixed: bool,

        /// Turbo mode: parallel swarm execution
        #[arg(long)]
        turbo: bool,

        /// Resume a paused session
        #[arg(long)]
        resume: Option<String>,

        /// Select actions (e.g., "1,2,3" or "do 1 but skip tests")
        #[arg(long)]
        select: Option<String>,

        /// Fast mode: skip slow checks
        #[arg(long)]
        fast: bool,

        /// Simple menu: just numbered list (no TUI)
        #[arg(long)]
        simple_menu: bool,

        /// Generate mode: create tasks.md without executing
        #[arg(long)]
        generate: bool,

        /// Action mode: execute specific tasks by number
        #[arg(long)]
        action: bool,

        /// Task numbers to execute (for action mode)
        #[arg(trailing_var_arg = true)]
        tasks: Vec<String>,
    },

    /// Run tests with Claude's help
    Test {
        /// Test pattern to run
        #[arg(trailing_var_arg = true)]
        pattern: Vec<String>,
    },

    /// Create a new project
    New {
        /// Project name
        name: Option<String>,

        /// Project type (rust, python, etc.)
        #[arg(long, short = 't')]
        project_type: Option<String>,
    },

    /// Initialize Palace in current directory
    Init,

    /// List saved sessions
    Sessions,

    /// Manage the translator daemon
    Translator {
        #[command(subcommand)]
        action: TranslatorAction,
    },

    /// Check build status of watched projects
    BuildStatus,

    /// Install Palace skills/masks
    Skill {
        #[command(subcommand)]
        action: SkillAction,
    },
}

#[derive(Subcommand)]
enum TranslatorAction {
    /// Start the translator daemon
    Start,
    /// Stop the translator daemon
    Stop,
    /// Check translator status
    Status,
}

#[derive(Subcommand)]
enum SkillAction {
    /// Install a skill
    Install {
        /// Skill name or --path to folder
        name: Option<String>,

        /// Path to skill folder
        #[arg(long)]
        path: Option<PathBuf>,
    },
    /// List installed skills
    List,
    /// Remove a skill
    Remove {
        /// Skill name
        name: String,
    },
}

/// Configuration for model selection
#[derive(Debug, Clone)]
struct ModelConfig {
    model: String,
    env_override: Option<HashMap<String, String>>,
    display_name: String,
}

impl ModelConfig {
    fn sonnet() -> Self {
        Self {
            model: "claude-sonnet-4-5".to_string(),
            env_override: None,
            display_name: "Claude Sonnet".to_string(),
        }
    }

    fn opus() -> Self {
        Self {
            model: "claude-opus-4-5-20251101".to_string(),
            env_override: None,
            display_name: "Claude Opus".to_string(),
        }
    }

    fn glm() -> Self {
        Self {
            model: "glm-4.6".to_string(),
            env_override: Some(Self::translator_env("https://open.bigmodel.cn/api/paas/v4", "glm-4.6")),
            display_name: "GLM-4.6".to_string(),
        }
    }

    fn glmv() -> Self {
        Self {
            model: "glm-4.6v".to_string(),
            env_override: Some(Self::translator_env("https://open.bigmodel.cn/api/paas/v4", "glm-4.6v")),
            display_name: "GLM-4.6v (vision)".to_string(),
        }
    }

    fn devstral() -> Self {
        Self {
            model: "devstral-2512".to_string(),
            env_override: Some(Self::translator_env("https://api.mistral.ai/v1", "devstral-2512")),
            display_name: "Devstral 2 (123B)".to_string(),
        }
    }

    fn devstral_small_local() -> Self {
        let endpoint = std::env::var("DEVSTRAL_SMALL_ENDPOINT")
            .unwrap_or_else(|_| "http://10.7.1.135:11434/v1".to_string());
        Self {
            model: "devstral".to_string(),
            env_override: Some(Self::translator_env(&endpoint, "devstral")),
            display_name: "Devstral Small 2 (24B local)".to_string(),
        }
    }

    fn translator_env(backend_url: &str, _model: &str) -> HashMap<String, String> {
        let mut env = HashMap::new();
        // Palace-daemon handles the translation - we just need to point to it
        // The translator runs on port 19848 by default
        env.insert("ANTHROPIC_BASE_URL".to_string(), "http://127.0.0.1:19848".to_string());
        // Remove API key - translator handles auth
        env.insert("_PALACE_BACKEND_URL".to_string(), backend_url.to_string());
        env
    }
}

/// RHSI "next" command implementation with advanced features
fn cmd_next(
    guidance: Vec<String>,
    model_config: ModelConfig,
    turbo: bool,
    resume: Option<String>,
    select: Option<String>,
    fast: bool,
    simple_menu: bool,
    generate: bool,
    action: bool,
    tasks: Vec<String>,
) -> Result<()> {
    // Check we're not running as root
    if unsafe { libc::getuid() } == 0 {
        eprintln!("❌ ERROR: Palace cannot run as root");
        eprintln!();
        eprintln!("   Claude CLI blocks permission bypass for root users.");
        eprintln!("   This is a security feature with no override.");
        eprintln!();
        eprintln!("   Run as a normal user instead:");
        eprintln!("     sudo -u wings palace next [options]");
        std::process::exit(1);
    }

    let guidance_text = guidance.join(" ");
    let project_root = std::env::current_dir()?;

    // Generate mode: create tasks.md without executing
    if generate {
        return cmd_next_generate(guidance_text, model_config, project_root);
    }

    // Action mode: execute specific tasks
    if action {
        return cmd_next_action(tasks, model_config, turbo, project_root);
    }

    // Normal interactive mode
    cmd_next_interactive(
        guidance_text,
        model_config,
        turbo,
        resume,
        select,
        fast,
        simple_menu,
        project_root,
    )
}

/// Generate mode: create tasks.md file
fn cmd_next_generate(
    guidance_text: String,
    model_config: ModelConfig,
    project_root: PathBuf,
) -> Result<()> {
    println!("📝 Palace Generate Mode - Creating tasks.md...");

    // Build the prompt
    let initial_prompt = if !guidance_text.is_empty() {
        format!(
            r#"Analyze this project and suggest possible next actions.

USER GUIDANCE: {}

Focus your suggestions on what the user has asked for above.
Check SPEC.md and ROADMAP.md if they exist for context.

Provide as many options as you see fit - there may be many valid paths forward.
Be concrete and actionable. The user will select which action(s) to execute."#,
            guidance_text
        )
    } else {
        r#"Analyze this project and suggest possible next actions.

Consider what exists, what's in progress, and what could come next.
Check SPEC.md and ROADMAP.md if they exist.

Provide as many options as you see fit - there may be many valid paths forward.
Be concrete and actionable. The user will select which action(s) to execute."#
            .to_string()
    };

    // Menu format instructions
    let menu_prompt = r#"IMPORTANT: End your response with suggested actions in a YAML code block.

Format EXACTLY like this:

```yaml
actions:
  - label: Short action label here
    description: Brief description of what this action does
  - label: Another action label
    description: Description for this action
```

RULES:
- Use a ```yaml code block (required for parsing)
- The key must be "actions:" containing a list
- Each action needs "label:" (required) and "description:" (optional but recommended)
- Keep labels concise (under 60 chars)
- Descriptions can be detailed (up to 1000 chars) - they provide important context for execution
- INCLUDE ALL SUGGESTED ACTIONS - do NOT filter or reduce the list
- If you identified 50 tasks, include all 50 in the actions block
- The user will select which ones to execute from the full list"#;

    // Gather context
    let context = context::gather_context(&project_root)?;

    // Build the full prompt with context
    let full_prompt = context::build_prompt(&initial_prompt, &context);

    // Build claude command
    let mut cmd = Command::new("claude");
    cmd.arg("--print")
        .arg("--model")
        .arg(&model_config.model)
        .arg("--append-system-prompt")
        .arg(menu_prompt)
        .arg("--verbose")
        .arg("--input-format")
        .arg("stream-json")
        .arg("--output-format")
        .arg("stream-json")
        .arg("--dangerously-skip-permissions")
        .current_dir(&project_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    // Apply environment overrides
    if let Some(env_override) = &model_config.env_override {
        for (key, value) in env_override {
            if !key.starts_with('_') {
                cmd.env(key, value);
            }
        }
    }

    println!("🏛️  Palace - Invoking Claude to generate tasks...");
    println!("📦 Model: {}", model_config.display_name);
    if !guidance_text.is_empty() {
        println!("📝 Guidance: {}", guidance_text);
    }
    println!();

    // Start the process
    let mut child = cmd.spawn().context("Failed to start claude CLI")?;

    // Send the prompt via stream-json
    {
        let stdin = child.stdin.as_mut().context("Failed to get stdin")?;
        let initial_message = serde_json::json!({
            "type": "user",
            "message": {"role": "user", "content": full_prompt}
        });
        writeln!(stdin, "{}", serde_json::to_string(&initial_message)?)?;
    }

    // Read and process output
    let stdout = child.stdout.take().context("Failed to get stdout")?;
    let actions = process_stream_output(stdout, false)?;

    let exit_status = child.wait()?;

    if !exit_status.success() {
        warn!("Claude exited with code {:?}", exit_status.code());
    }

    // Save actions to tasks.md
    if let Some(actions) = actions {
        if !actions.is_empty() {
            save_tasks_to_file(&actions, &project_root)?;
            println!("✅ Generated tasks saved to .palace/tasks.md");
            println!("   Use: palace next --action --mistral --turbo [task_numbers]");
        }
    }

    Ok(())
}

/// Save tasks to .palace/tasks.md file
fn save_tasks_to_file(actions: &[Action], project_root: &PathBuf) -> Result<()> {
    let palace_dir = project_root.join(".palace");
    std::fs::create_dir_all(&palace_dir)?;

    let tasks_file = palace_dir.join("tasks.md");
    let mut content = String::new();

    content.push_str("# Generated Tasks\n\n");
    content.push_str("These tasks were generated by Palace in response to your guidance.\n\n");

    for (i, action) in actions.iter().enumerate() {
        let num = i + 1;
        content.push_str(&format!("## Task {}: {}\n\n", num, action.label));
        if let Some(desc) = &action.description {
            content.push_str(&format!("{}\n\n", desc));
        }
        content.push_str(&format!("**Command**: `palace next --action --mistral --turbo {}`\n\n", num));
    }

    content.push_str("## Usage\n\n");
    content.push_str("- To execute specific tasks: `palace next --action --mistral --turbo 1 3 5`\n");
    content.push_str("- To execute all tasks: `palace next --action --mistral --turbo $(seq 1 ");
    content.push_str(&actions.len().to_string());
    content.push_str(")`\n");
    content.push_str("- To use interactive selection: `palace next --mistral --turbo`\n");

    std::fs::write(tasks_file, content)?;

    // Add to .gitignore if not already there
    let gitignore_path = project_root.join(".gitignore");
    let mut gitignore_content = if gitignore_path.exists() {
        std::fs::read_to_string(&gitignore_path)?
    } else {
        String::new()
    };

    if !gitignore_content.contains("tasks.md") && !gitignore_content.contains(".palace/") {
        gitignore_content.push_str("\n# Palace generated files\n.palace/tasks.md\n");
        std::fs::write(gitignore_path, gitignore_content)?;
    }

    Ok(())
}

/// Action mode: execute specific tasks by number
fn cmd_next_action(
    task_numbers: Vec<String>,
    model_config: ModelConfig,
    turbo: bool,
    project_root: PathBuf,
) -> Result<()> {
    println!("🚀 Palace Action Mode - Executing tasks...");

    // Read tasks from file
    let palace_dir = project_root.join(".palace");
    let tasks_file = palace_dir.join("tasks.md");

    if !tasks_file.exists() {
        println!("❌ No tasks.md found. Run generate mode first:");
        println!("   palace next --generate --mistral");
        return Ok(());
    }

    let tasks_content = std::fs::read_to_string(&tasks_file)?;
    let actions = parse_tasks_from_file(&tasks_content)?;

    if actions.is_empty() {
        println!("❌ No tasks found in tasks.md");
        return Ok(());
    }

    // Parse task numbers
    let mut selected_indices = Vec::new();
    for task_str in task_numbers {
        if let Ok(num) = task_str.parse::<usize>() {
            if num >= 1 && num <= actions.len() {
                selected_indices.push(num - 1);
            }
        }
    }

    if selected_indices.is_empty() {
        println!("❌ No valid task numbers provided");
        println!("   Available tasks: 1-{}", actions.len());
        return Ok(());
    }

    // Remove duplicates and sort
    selected_indices.sort();
    selected_indices.dedup();

    let selected_actions: Vec<Action> = selected_indices
        .into_iter()
        .map(|i| actions[i].clone())
        .collect();

    println!("📋 Executing {} task(s):", selected_actions.len());
    for action in &selected_actions {
        println!("   • {}", action.label);
    }
    println!();

    // Execute tasks in turbo mode if requested
    if turbo {
        execute_tasks_turbo(selected_actions, model_config, project_root)?;
    } else {
        execute_tasks_sequential(selected_actions, model_config, project_root)?;
    }

    Ok(())
}

/// Parse tasks from tasks.md file
fn parse_tasks_from_file(content: &str) -> Result<Vec<Action>> {
    let mut actions = Vec::new();
    let mut current_label = None;
    let mut current_description = None;

    for line in content.lines() {
        if line.starts_with("## Task ") {
            // Save previous action if exists
            if let Some(label) = current_label.take() {
                actions.push(Action {
                    label,
                    description: current_description.take(),
                });
            }

            // Parse new task
            if let Some(rest) = line.strip_prefix("## Task ") {
                if let Some((num, label)) = rest.split_once(": ") {
                    current_label = Some(label.to_string());
                    current_description = Some(String::new());
                }
            }
        } else if let Some(label) = &current_label {
            // Accumulate description lines (skip command lines)
            if !line.starts_with("**Command**") && !line.is_empty() && !line.starts_with("#") {
                if let Some(desc) = &mut current_description {
                    if !desc.is_empty() {
                        desc.push_str(" ");
                    }
                    desc.push_str(line.trim());
                }
            }
        }
    }

    // Save last action
    if let Some(label) = current_label {
        actions.push(Action {
            label,
            description: current_description,
        });
    }

    Ok(actions)
}

/// Execute tasks in turbo mode (parallel)
fn execute_tasks_turbo(
    actions: Vec<Action>,
    model_config: ModelConfig,
    project_root: PathBuf,
) -> Result<()> {
    println!("🚀 Turbo Mode: Executing tasks in parallel...");

    // For now, we'll implement sequential execution
    // In a real implementation, this would spawn multiple Claude instances
    execute_tasks_sequential(actions, model_config, project_root)
}

/// Execute tasks sequentially
fn execute_tasks_sequential(
    actions: Vec<Action>,
    model_config: ModelConfig,
    project_root: PathBuf,
) -> Result<()> {
    println!("🔄 Sequential Mode: Executing tasks one by one...");

    for (i, action) in actions.iter().enumerate() {
        println!("\n📋 Task {}: {}", i + 1, action.label);

        // Build execution prompt
        let execution_prompt = format!(
            r#"Execute this task:

TASK: {}

CONTEXT: {}

Please complete this task. Be thorough and ensure all requirements are met.
When complete, provide a summary of what was accomplished."#,
            action.label,
            action.description.as_deref().unwrap_or("No additional context")
        );

        // Build claude command
        let mut cmd = Command::new("claude");
        cmd.arg("--print")
            .arg("--model")
            .arg(&model_config.model)
            .arg("--verbose")
            .arg("--input-format")
            .arg("stream-json")
            .arg("--output-format")
            .arg("stream-json")
            .arg("--dangerously-skip-permissions")
            .current_dir(&project_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

        // Apply environment overrides
        if let Some(env_override) = &model_config.env_override {
            for (key, value) in env_override {
                if !key.starts_with('_') {
                    cmd.env(key, value);
                }
            }
        }

        // Start the process
        let mut child = cmd.spawn().context("Failed to start claude CLI")?;

        // Send the prompt via stream-json
        {
            let stdin = child.stdin.as_mut().context("Failed to get stdin")?;
            let initial_message = serde_json::json!({
                "type": "user",
                "message": {"role": "user", "content": execution_prompt}
            });
            writeln!(stdin, "{}", serde_json::to_string(&initial_message)?)?;
        }

        let exit_status = child.wait()?;

        if !exit_status.success() {
            warn!("Task {} failed with code {:?}", i + 1, exit_status.code());
        } else {
            println!("✅ Task {} completed", i + 1);
        }
    }

    println!("\n🎉 All tasks completed!");
    Ok(())
}

/// Interactive mode: normal palace next functionality
fn cmd_next_interactive(
    guidance_text: String,
    model_config: ModelConfig,
    turbo: bool,
    resume: Option<String>,
    select: Option<String>,
    fast: bool,
    simple_menu: bool,
    project_root: PathBuf,
) -> Result<()> {
    // Build the prompt
    let initial_prompt = if !guidance_text.is_empty() {
        format!(
            r#"Analyze this project and suggest possible next actions.

USER GUIDANCE: {}

Focus your suggestions on what the user has asked for above.
Check SPEC.md and ROADMAP.md if they exist for context.

Provide as many options as you see fit - there may be many valid paths forward.
Be concrete and actionable. The user will select which action(s) to execute."#,
            guidance_text
        )
    } else {
        r#"Analyze this project and suggest possible next actions.

Consider what exists, what's in progress, and what could come next.
Check SPEC.md and ROADMAP.md if they exist.

Provide as many options as you see fit - there may be many valid paths forward.
Be concrete and actionable. The user will select which action(s) to execute."#
            .to_string()
    };

    // Menu format instructions
    let menu_prompt = r#"IMPORTANT: End your response with suggested actions in a YAML code block.

Format EXACTLY like this:

```yaml
actions:
  - label: Short action label here
    description: Brief description of what this action does
  - label: Another action label
    description: Description for this action
```

RULES:
- Use a ```yaml code block (required for parsing)
- The key must be "actions:" containing a list
- Each action needs "label:" (required) and "description:" (optional but recommended)
- Keep labels concise (under 60 chars)
- Descriptions can be detailed (up to 1000 chars) - they provide important context for execution
- INCLUDE ALL SUGGESTED ACTIONS - do NOT filter or reduce the list
- If you identified 50 tasks, include all 50 in the actions block
- The user will select which ones to execute from the full list"#;

    // Gather context
    let context = context::gather_context(&project_root)?;

    // Build the full prompt with context
    let full_prompt = context::build_prompt(&initial_prompt, &context);

    // Build claude command
    let mut cmd = Command::new("claude");
    cmd.arg("--print")
        .arg("--model")
        .arg(&model_config.model)
        .arg("--append-system-prompt")
        .arg(menu_prompt)
        .arg("--verbose")
        .arg("--input-format")
        .arg("stream-json")
        .arg("--output-format")
        .arg("stream-json")
        .arg("--dangerously-skip-permissions")
        .current_dir(&project_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    // Apply environment overrides
    if let Some(env_override) = &model_config.env_override {
        for (key, value) in env_override {
            if !key.starts_with('_') {
                cmd.env(key, value);
            }
        }
    }

    println!("🏛️  Palace - Invoking Claude...");
    println!("📦 Model: {}", model_config.display_name);
    if !guidance_text.is_empty() {
        println!("📝 Guidance: {}", guidance_text);
    }
    println!();

    // Start the process
    let mut child = cmd.spawn().context("Failed to start claude CLI")?;

    // Send the prompt via stream-json
    {
        let stdin = child.stdin.as_mut().context("Failed to get stdin")?;
        let initial_message = serde_json::json!({
            "type": "user",
            "message": {"role": "user", "content": full_prompt}
        });
        writeln!(stdin, "{}", serde_json::to_string(&initial_message)?)?;
    }

    // Read and process output
    let stdout = child.stdout.take().context("Failed to get stdout")?;
    let actions = process_stream_output(stdout, simple_menu)?;

    let exit_status = child.wait()?;

    if !exit_status.success() {
        warn!("Claude exited with code {:?}", exit_status.code());
    }

    // Present action menu if we got actions
    if let Some(actions) = actions {
        if !actions.is_empty() {
            if simple_menu {
                // Use simple text menu
                menu::present_action_menu(&actions, true)?;
            } else {
                // Use TUI menu with ratatui
                tui_menu::present_tui_action_menu(&actions)?;
            }
        }
    }

    Ok(())
}

// Re-export Action from menu module
use menu::Action;

/// Stream events from Claude CLI
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    Assistant {
        message: serde_json::Value,
    },
    Result {
        result: serde_json::Value,
        #[serde(default)]
        #[allow(dead_code)]
        subtype: Option<String>,
    },
    #[serde(other)]
    Other,
}

/// Process streaming JSON output from Claude CLI
fn process_stream_output(
    stdout: impl std::io::Read,
    _simple_menu: bool,
) -> Result<Option<Vec<Action>>> {
    let reader = BufReader::new(stdout);
    let mut full_response = String::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        // Try to parse as JSON
        if let Ok(event) = serde_json::from_str::<StreamEvent>(&line) {
            match event {
                StreamEvent::Assistant { message } => {
                    if let Some(content) = message.get("content") {
                        if let Some(text) = content.as_str() {
                            print!("{}", text);
                            std::io::stdout().flush().ok();
                            full_response.push_str(text);
                        } else if let Some(blocks) = content.as_array() {
                            for block in blocks {
                                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                    print!("{}", text);
                                    std::io::stdout().flush().ok();
                                    full_response.push_str(text);
                                }
                            }
                        }
                    }
                }
                StreamEvent::Result { result, .. } => {
                    // Final result - try to extract actions
                    if let Some(content) = result.get("content") {
                        if let Some(text) = content.as_str() {
                            full_response = text.to_string();
                        }
                    }
                }
                _ => {}
            }
        } else {
            // Raw text output
            print!("{}", line);
            std::io::stdout().flush().ok();
            full_response.push_str(&line);
            full_response.push('\n');
        }
    }

    println!(); // Final newline

    // Parse actions from the response
    let actions = parse_actions_from_response(&full_response);

    Ok(actions)
}

/// Parse YAML action blocks from Claude's response
fn parse_actions_from_response(response: &str) -> Option<Vec<Action>> {
    // Look for ```yaml ... ``` blocks
    let yaml_start = response.find("```yaml")?;
    let content_start = yaml_start + 7; // Skip "```yaml"
    let yaml_end = response[content_start..].find("```")?;
    let yaml_content = &response[content_start..content_start + yaml_end].trim();

    // Parse the YAML
    let parsed: serde_yaml::Value = serde_yaml::from_str(yaml_content).ok()?;
    let actions_array = parsed.get("actions")?.as_sequence()?;

    let actions: Vec<Action> = actions_array
        .iter()
        .filter_map(|item| {
            let label = item.get("label")?.as_str()?.to_string();
            let description = item
                .get("description")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string());
            Some(Action { label, description })
        })
        .collect();

    if actions.is_empty() {
        None
    } else {
        Some(actions)
    }
}

// Other command implementations (copied from main.rs)
fn cmd_test(pattern: Vec<String>) -> Result<()> {
    println!("🧪 Palace Test - Running tests...");

    let test_pattern = pattern.join(" ");
    let project_root = std::env::current_dir()?;

    // Detect project type and run appropriate tests
    if project_root.join("Cargo.toml").exists() {
        let mut cmd = Command::new("cargo");
        cmd.arg("test");
        if !test_pattern.is_empty() {
            cmd.arg(&test_pattern);
        }
        cmd.current_dir(&project_root);

        let status = cmd.status()?;
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }
    } else if project_root.join("package.json").exists() {
        let mut cmd = Command::new("npm");
        cmd.arg("test");
        cmd.current_dir(&project_root);

        let status = cmd.status()?;
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }
    } else if project_root.join("pyproject.toml").exists() || project_root.join("setup.py").exists()
    {
        let mut cmd = Command::new("pytest");
        if !test_pattern.is_empty() {
            cmd.arg(&test_pattern);
        }
        cmd.current_dir(&project_root);

        let status = cmd.status()?;
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }
    } else {
        println!("⚠️  No recognized project type found");
        println!("   Supported: Rust (Cargo.toml), Node (package.json), Python (pyproject.toml)");
    }

    Ok(())
}

fn cmd_init() -> Result<()> {
    let project_root = std::env::current_dir()?;
    let palace_dir = project_root.join(".palace");

    if palace_dir.exists() {
        println!("✅ Palace already initialized in this directory");
        return Ok(());
    }

    std::fs::create_dir_all(&palace_dir)?;
    println!("🏛️  Palace initialized in {}", project_root.display());
    println!();
    println!("   Next steps:");
    println!("   1. Run `palace next` to get started");
    println!("   2. Create SPEC.md to guide Claude");
    println!("   3. Create ROADMAP.md for long-term planning");

    Ok(())
}

fn cmd_sessions() -> Result<()> {
    let palace_dir = dirs::home_dir()
        .context("Could not find home directory")?
        .join(".palace")
        .join("sessions");

    if !palace_dir.exists() {
        println!("📋 No sessions found");
        return Ok(());
    }

    println!("📋 Saved Sessions:");
    println!();

    let mut entries: Vec<_> = std::fs::read_dir(&palace_dir)?
        .filter_map(|e| e.ok())
        .collect();

    entries.sort_by_key(|e| std::cmp::Reverse(e.metadata().ok().and_then(|m| m.modified().ok())));

    for entry in entries.iter().take(10) {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.ends_with(".json") {
            println!("  • {}", name_str.trim_end_matches(".json"));
        }
    }

    Ok(())
}

fn cmd_translator(action: TranslatorAction) -> Result<()> {
    match action {
        TranslatorAction::Start => {
            println!("🚀 Starting translator daemon...");
            // The translator is now palace-daemon, managed by systemd
            let status = Command::new("systemctl")
                .args(["--user", "start", "palace-daemon"])
                .status();

            match status {
                Ok(s) if s.success() => println!("✅ Translator started"),
                _ => {
                    println!("⚠️  Could not start via systemd, trying direct...");
                    // TODO: Direct start
                }
            }
        }
        TranslatorAction::Stop => {
            println!("🛑 Stopping translator daemon...");
            let _ = Command::new("systemctl")
                .args(["--user", "stop", "palace-daemon"])
                .status();
            println!("✅ Translator stopped");
        }
        TranslatorAction::Status => {
            // Check if palace-daemon is running
            let output = Command::new("curl")
                .args(["-s", "http://127.0.0.1:19848/health"])
                .output();

            match output {
                Ok(o) if o.status.success() => {
                    println!("✅ Translator is running (port 19848)");
                    if let Ok(body) = String::from_utf8(o.stdout) {
                        println!("   {}", body.trim());
                    }
                }
                _ => {
                    println!("❌ Translator is not running");
                    println!("   Start with: palace translator start");
                }
            }
        }
    }
    Ok(())
}

fn cmd_skill(action: SkillAction) -> Result<()> {
    let skills_dir = dirs::home_dir()
        .context("Could not find home directory")?
        .join(".claude")
        .join("skills");

    match action {
        SkillAction::Install { name, path } => {
            if let Some(path) = path {
                // Install from local path
                let skill_md = path.join("SKILL.md");
                if !skill_md.exists() {
                    anyhow::bail!("No SKILL.md found in {}", path.display());
                }

                let skill_name = path
                    .file_name()
                    .context("Invalid path")?
                    .to_string_lossy()
                    .to_string();

                let dest = skills_dir.join(&skill_name);
                std::fs::create_dir_all(&dest)?;

                // Copy all files
                for entry in std::fs::read_dir(&path)? {
                    let entry = entry?;
                    let dest_path = dest.join(entry.file_name());
                    std::fs::copy(entry.path(), dest_path)?;
                }

                println!("✅ Installed skill: {}", skill_name);
            } else if let Some(name) = name {
                // Install from palace-skills repo
                println!("📦 Installing {} from palace-skills...", name);
                // TODO: Download from repo
                println!("⚠️  Remote installation not yet implemented");
                println!("   Use --path to install from local folder");
            } else {
                println!("Usage: palace skill install <name> or palace skill install --path <folder>");
            }
        }
        SkillAction::List => {
            if !skills_dir.exists() {
                println!("📋 No skills installed");
                return Ok(());
            }

            println!("📋 Installed Skills:");
            println!();

            for entry in std::fs::read_dir(&skills_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    let skill_md = entry.path().join("SKILL.md");
                    if skill_md.exists() {
                        println!("  • {}", name);
                    }
                }
            }
        }
        SkillAction::Remove { name } => {
            let skill_path = skills_dir.join(&name);
            if skill_path.exists() {
                std::fs::remove_dir_all(&skill_path)?;
                println!("✅ Removed skill: {}", name);
            } else {
                println!("⚠️  Skill not found: {}", name);
            }
        }
    }
    Ok(())
}

fn cmd_build_status() -> Result<()> {
    println!("🔍 Checking build status...");

    let project_root = std::env::current_dir()?;

    // Check for Rust project
    if project_root.join("Cargo.toml").exists() {
        println!();
        println!("📦 Rust project detected");

        // Run cargo check
        let output = Command::new("cargo")
            .args(["check", "--message-format=short"])
            .current_dir(&project_root)
            .output()?;

        if output.status.success() {
            println!("   ✅ Build: OK");
        } else {
            println!("   ❌ Build: FAILED");
            let stderr = String::from_utf8_lossy(&output.stderr);
            for line in stderr.lines().take(10) {
                println!("      {}", line);
            }
        }

        // Run cargo test --no-run to check test compilation
        let test_output = Command::new("cargo")
            .args(["test", "--no-run", "--message-format=short"])
            .current_dir(&project_root)
            .output()?;

        if test_output.status.success() {
            println!("   ✅ Tests compile: OK");
        } else {
            println!("   ❌ Tests compile: FAILED");
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Next {
            guidance,
            opus,
            glm,
            glmv,
            devstral,
            devstral_small,
            ollama,
            mixed: _,
            turbo,
            resume,
            select,
            fast,
            simple_menu,
            generate,
            action,
            tasks,
        } => {
            // Determine model configuration
            let model_config = if opus {
                ModelConfig::opus()
            } else if glm {
                ModelConfig::glm()
            } else if glmv {
                ModelConfig::glmv()
            } else if devstral {
                ModelConfig::devstral()
            } else if devstral_small || ollama {
                ModelConfig::devstral_small_local()
            } else {
                ModelConfig::sonnet()
            };

            cmd_next(
                guidance,
                model_config,
                turbo,
                resume,
                select,
                fast,
                simple_menu,
                generate,
                action,
                tasks,
            )
        }
        Commands::Test { pattern } => cmd_test(pattern),
        Commands::New { name, project_type } => {
            println!("🆕 Creating new project...");
            if let Some(name) = name {
                println!("   Name: {}", name);
            }
            if let Some(pt) = project_type {
                println!("   Type: {}", pt);
            }
            println!("⚠️  Project creation not yet implemented");
            Ok(())
        }
        Commands::Init => cmd_init(),
        Commands::Sessions => cmd_sessions(),
        Commands::Translator { action } => cmd_translator(action),
        Commands::BuildStatus => cmd_build_status(),
        Commands::Skill { action } => cmd_skill(action),
    }
}