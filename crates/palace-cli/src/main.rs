//! Palace CLI - RHSI (Really Hard Software Intelligence) Command Line Interface
//!
//! This is the main entry point for Palace, providing commands like:
//! - `pal next` - Ask Claude what to do next (core RHSI loop)
//! - `pal test` - Run tests with Claude's help
//! - `pal new` - Create a new project
//! - `pal init` - Initialize Palace in current directory
//!
//! The CLI communicates with Claude Code CLI, which routes through palace-daemon
//! for model translation and @command handling.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tracing::warn;

mod context;
mod menu;
#[allow(dead_code)]
mod session;

/// Palace - RHSI Development Assistant
#[derive(Parser)]
#[command(name = "pal")]
#[command(author = "Riff Labs <wings@riff.cc>")]
#[command(version)]
#[command(about = "Palace CLI - RHSI Development Assistant", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Ask Claude what to do next (RHSI core)
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
        #[arg(long)]
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

        /// Simple menu: just numbered list
        #[arg(long)]
        simple_menu: bool,
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

/// RHSI "next" command implementation
fn cmd_next(
    guidance: Vec<String>,
    model_config: ModelConfig,
    _turbo: bool,
    _resume: Option<String>,
    _select: Option<String>,
    _fast: bool,
    simple_menu: bool,
) -> Result<()> {
    // Check we're not running as root
    if unsafe { libc::getuid() } == 0 {
        eprintln!("❌ ERROR: Palace cannot run as root");
        eprintln!();
        eprintln!("   Claude CLI blocks permission bypass for root users.");
        eprintln!("   This is a security feature with no override.");
        eprintln!();
        eprintln!("   Run as a normal user instead:");
        eprintln!("     sudo -u wings pal next [options]");
        std::process::exit(1);
    }

    let guidance_text = guidance.join(" ");
    let project_root = std::env::current_dir()?;

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
            menu::present_action_menu(&actions, simple_menu)?;
        }
    }

    Ok(())
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
    println!("   1. Run `pal next` to get started");
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
                    println!("   Start with: pal translator start");
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
                println!("Usage: pal skill install <name> or pal skill install --path <folder>");
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

            cmd_next(guidance, model_config, turbo, resume, select, fast, simple_menu)
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
