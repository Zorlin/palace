mod ai;
mod app;
mod debug;
mod display;
mod persistence;
mod projects;
mod renderer;
mod state;

use anyhow::Result;
use clap::{Parser, Subcommand};
use projects::{detect_launch_context, LaunchContext, ProjectsConfig};
use state::AppState;
use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use winit::event_loop::EventLoop;

#[derive(Parser)]
#[command(name = "palace")]
#[command(about = "GPU-native project launcher")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Restore state from file (used by restart command)
    #[arg(long, value_name = "FILE")]
    restore: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Take screenshot(s) of a running Palace instance
    Screenshot {
        /// Number of screenshots to take
        #[arg(default_value = "1")]
        count: u32,

        /// Delay between screenshots (e.g., "5s", "500ms")
        #[arg(default_value = "0s")]
        delay: String,

        /// Output path (default: /tmp/palace-screenshot-<timestamp>.png)
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Restart a running Palace instance (rebuilds and relaunches)
    Restart,

    /// Manage displays for Palace
    Display {
        #[command(subcommand)]
        action: DisplayAction,
    },

    /// Generate suggestions for a project (headless mode)
    Suggest {
        /// Project path (defaults to current directory)
        #[arg(short, long)]
        project: Option<String>,

        /// Output format: text, json, or render (PNG screenshot)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Output path for rendered screenshot (only with --format=render)
        #[arg(short, long)]
        output: Option<String>,

        /// Disable streaming (wait for full response)
        #[arg(long)]
        no_stream: bool,
    },

    /// Run headless with test rendering
    Headless {
        /// Width of the virtual screen
        #[arg(long, default_value = "1920")]
        width: u32,

        /// Height of the virtual screen
        #[arg(long, default_value = "1080")]
        height: u32,

        /// Output PNG path
        #[arg(short, long)]
        output: Option<String>,

        /// Project path to display
        #[arg(short, long)]
        project: Option<String>,
    },
}

#[derive(Subcommand)]
enum DisplayAction {
    /// List available displays
    List,
    /// Switch to a different display
    Switch {
        /// Display index (from 'palace display list')
        index: usize,
    },
}

fn main() -> Result<()> {
    // Install SIGSEGV handler for segfaults (GPU driver crashes, etc)
    install_signal_handlers();

    // Install custom panic hook for better crash output
    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("\n\x1b[31m╔═══════════════════════════════════════════════════════════════╗\x1b[0m");
        eprintln!("\x1b[31m║                     PALACE CRASHED                             ║\x1b[0m");
        eprintln!("\x1b[31m╚═══════════════════════════════════════════════════════════════╝\x1b[0m\n");

        // Print panic message
        if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            eprintln!("\x1b[1;33mPanic:\x1b[0m {}", s);
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            eprintln!("\x1b[1;33mPanic:\x1b[0m {}", s);
        } else {
            eprintln!("\x1b[1;33mPanic:\x1b[0m Unknown error");
        }

        // Print location
        if let Some(location) = panic_info.location() {
            eprintln!("\x1b[1;36mLocation:\x1b[0m {}:{}:{}", location.file(), location.line(), location.column());
        }

        // Print backtrace
        let backtrace = std::backtrace::Backtrace::capture();
        if backtrace.status() == std::backtrace::BacktraceStatus::Captured {
            eprintln!("\n\x1b[1;35mBacktrace:\x1b[0m\n{}", backtrace);
        } else {
            eprintln!("\n\x1b[1;33mNote:\x1b[0m Set RUST_BACKTRACE=1 for full backtrace");
        }

        eprintln!("\n\x1b[2mPlease report this issue at: https://github.com/yourrepo/palace/issues\x1b[0m");
    }));

    // Load .env file from ~/.config/palace/.env if present (silently)
    if let Some(config_dir) = dirs::config_dir() {
        let env_path = config_dir.join("palace").join(".env");
        if env_path.exists() {
            let _ = dotenvy::from_path(&env_path);
        }
    }

    let cli = Cli::parse();

    // Handle subcommands that don't need the full app
    if let Some(command) = cli.command {
        return match command {
            Commands::Screenshot {
                count,
                delay,
                output,
            } => run_screenshot_command(count, delay, output),
            Commands::Restart => run_restart_command(),
            Commands::Display { action } => run_display_command(action),
            Commands::Suggest { project, format, output, no_stream } => {
                run_suggest_command(project, format, output, !no_stream)
            }
            Commands::Headless { width, height, output, project } => {
                run_headless_command(width, height, output, project)
            }
        };
    }

    // Default: run the GUI application
    run_app(cli.restore)
}

/// Parse duration string like "5s", "500ms", "1m"
fn parse_duration(s: &str) -> Result<Duration> {
    let s = s.trim();
    if s == "0" || s == "0s" || s == "0ms" {
        return Ok(Duration::ZERO);
    }

    if let Some(secs) = s.strip_suffix("s") {
        if let Some(ms) = secs.strip_suffix("m") {
            // milliseconds: "500ms"
            let ms: u64 = ms.parse().map_err(|_| anyhow::anyhow!("Invalid duration: {}", s))?;
            return Ok(Duration::from_millis(ms));
        }
        // seconds: "5s"
        let secs: f64 = secs.parse().map_err(|_| anyhow::anyhow!("Invalid duration: {}", s))?;
        return Ok(Duration::from_secs_f64(secs));
    }

    if let Some(mins) = s.strip_suffix("m") {
        let mins: f64 = mins.parse().map_err(|_| anyhow::anyhow!("Invalid duration: {}", s))?;
        return Ok(Duration::from_secs_f64(mins * 60.0));
    }

    // Try parsing as plain seconds
    if let Ok(secs) = s.parse::<f64>() {
        return Ok(Duration::from_secs_f64(secs));
    }

    anyhow::bail!("Invalid duration format: {}. Use '5s', '500ms', or '1m'", s)
}

/// Run the screenshot command by connecting to the debug socket
fn run_screenshot_command(count: u32, delay: String, output: Option<String>) -> Result<()> {
    const SOCKET_PATH: &str = "/tmp/palace-debug.sock";

    let delay_duration = parse_duration(&delay)?;

    // Connect to the debug socket
    let mut stream = UnixStream::connect(SOCKET_PATH)
        .map_err(|e| anyhow::anyhow!("Failed to connect to Palace debug socket at {}: {}\nIs Palace running?", SOCKET_PATH, e))?;

    stream.set_read_timeout(Some(Duration::from_secs(5)))?;

    let mut reader = BufReader::new(stream.try_clone()?);

    // Read welcome message
    let mut welcome = String::new();
    reader.read_line(&mut welcome)?;
    reader.read_line(&mut welcome)?; // "Type 'help' for commands"

    // Skip the initial prompt
    let mut prompt = [0u8; 2];
    let _ = reader.read_exact(&mut prompt);

    for i in 0..count {
        // Build command
        let cmd = if let Some(ref path) = output {
            if count > 1 {
                // Add index to filename for multiple screenshots
                let path = std::path::Path::new(path);
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("screenshot");
                let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("png");
                let parent = path.parent().unwrap_or(std::path::Path::new("/tmp"));
                format!("screenshot {}/{}-{:04}.{}\n", parent.display(), stem, i + 1, ext)
            } else {
                format!("screenshot {}\n", path)
            }
        } else {
            "screenshot\n".to_string()
        };

        // Send screenshot command
        stream.write_all(cmd.as_bytes())?;
        stream.flush()?;

        // Read response
        let mut response = String::new();
        reader.read_line(&mut response)?;
        print!("{}", response);

        // Skip prompt
        let _ = reader.read_exact(&mut prompt);

        // Delay between screenshots (except after the last one)
        if i < count - 1 && delay_duration > Duration::ZERO {
            std::thread::sleep(delay_duration);
        }
    }

    // Send quit command
    stream.write_all(b"quit\n")?;

    Ok(())
}

const STATE_FILE: &str = "/tmp/palace-restart-state.json";

/// Restart Palace: rebuild first, then transfer state and restart
fn run_restart_command() -> Result<()> {
    use std::process::{Command, Stdio};

    const SOCKET_PATH: &str = "/tmp/palace-debug.sock";

    println!("🔄 Palace restart initiated...");

    // Step 1: Build FIRST (before touching the running instance)
    println!("🔨 Rebuilding Palace...");
    let build_status = Command::new("cargo")
        .args(["build"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()?;

    if !build_status.success() {
        anyhow::bail!("Build failed! Running instance unchanged.");
    }
    println!("✅ Build successful");

    // Step 2: Get state from running instance (if any)
    let mut got_state = false;
    let instance_running = if let Ok(mut stream) = UnixStream::connect(SOCKET_PATH) {
        stream.set_read_timeout(Some(Duration::from_secs(1)))?;
        stream.set_write_timeout(Some(Duration::from_secs(1)))?;

        let mut reader = BufReader::new(stream.try_clone()?);
        let mut line = String::new();

        // Try to read welcome - if this fails, socket is stale
        if reader.read_line(&mut line).is_err() {
            println!("ℹ️  Stale socket detected, no running instance");
            false
        } else {
            println!("📡 Getting state from running instance...");
            // Skip second welcome line and prompt
            let _ = reader.read_line(&mut line);
            let mut prompt = [0u8; 2];
            let _ = reader.read_exact(&mut prompt);

            // Request state
            stream.write_all(b"getstate\n")?;
            stream.flush()?;

            // Read state response
            line.clear();
            if reader.read_line(&mut line).is_ok() && line.starts_with('{') {
                if std::fs::write(STATE_FILE, line.trim()).is_ok() {
                    got_state = true;
                    println!("💾 State saved");
                }
            }

            // Signal kill and get PID
            stream.write_all(b"kill\n")?;
            stream.flush()?;

            // Read PID response and wait for exit
            line.clear();
            if reader.read_line(&mut line).is_ok() {
                if let Some(pid_str) = line.trim().strip_prefix("PID:") {
                    if let Ok(pid) = pid_str.parse::<i32>() {
                        drop(stream);
                        drop(reader);

                        print!("⏳ Waiting for PID {} to exit", pid);
                        std::io::stdout().flush().ok();

                        loop {
                            let result = unsafe { libc::kill(pid, 0) };
                            if result != 0 {
                                println!(" done");
                                break;
                            }
                            std::thread::yield_now();
                        }
                    }
                }
            }
            true
        }
    } else {
        println!("ℹ️  No running instance found");
        false
    };
    let _ = instance_running; // suppress unused warning

    // Step 3: Get the path to the new binary
    let binary_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("debug")
        .join("palace");

    // Step 4: Spawn new instance (detached)
    println!("🚀 Launching new Palace instance...");

    // Pass state file if we got state
    let mut args = vec!["-f", binary_path.to_str().unwrap()];
    let state_arg;
    if got_state {
        state_arg = format!("--restore={}", STATE_FILE);
        args.push(&state_arg);
    }

    let child = Command::new("setsid")
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    println!("✨ New Palace launched (pid: {})", child.id());
    println!("🏛️  Restart complete!");

    Ok(())
}

/// Run the suggest command - analyze project and generate AI suggestions
fn run_suggest_command(project: Option<String>, format: String, output: Option<String>, stream: bool) -> Result<()> {
    use crate::ai::{ProjectContext, SuggestionEngine};
    

    let project_path = project
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    eprintln!("🔍 Analyzing project: {}", project_path.display());

    let context = ProjectContext::gather(&project_path)?;
    let engine = SuggestionEngine::from_env()?;

    if stream {
        eprintln!("📡 Exploring project with Claude...\n");

        // Use agentic exploration with tool calls
        engine.suggest_with_exploration(&context)?;

        eprintln!();
    } else {
        eprintln!("📡 Generating suggestions with Claude...\n");

        let suggestions = engine.suggest(&context)?;

        match format.as_str() {
            "json" => {
                let json = serde_json::to_string_pretty(&suggestions)?;
                if let Some(ref path) = output {
                    std::fs::write(path, &json)?;
                    println!("Written to: {}", path);
                } else {
                    println!("{}", json);
                }
            }
            "render" => {
                println!("Render mode not yet implemented. Use 'text' or 'json'.");
            }
            _ => {
                println!("💡 Suggestions:\n");
                for (i, s) in suggestions.iter().enumerate() {
                    println!("{}. [{}] {}", i + 1, s.category.to_uppercase(), s.title);
                    println!("   Priority: {}/5", s.priority);
                    println!("   {}", s.description);
                    if let Some(ref cmd) = s.command {
                        println!("   Command: {}", cmd);
                    }
                    println!();
                }
            }
        }
    }

    Ok(())
}

/// Run headless mode - render without a window
fn run_headless_command(
    _width: u32,
    _height: u32,
    _output: Option<String>,
    _project: Option<String>,
) -> Result<()> {
    println!("🖥️  Headless mode not yet implemented.");
    println!("   This will render Palace UI to a PNG without needing a display.");
    Ok(())
}

/// Display configuration file path
const DISPLAY_CONFIG_PATH: &str = "/home/wings/.config/palace/display.json";

/// Run the display management command
fn run_display_command(action: DisplayAction) -> Result<()> {
    

    // Try GNOME's Mutter DisplayConfig first (works on GNOME/Wayland)
    let displays = get_gnome_displays().or_else(|_| get_randr_displays())?;

    match action {
        DisplayAction::List => {
            println!("📺 Available displays:\n");

            if displays.is_empty() {
                println!("  No displays found.");
                return Ok(());
            }

            let current_display = load_display_config();

            for (i, (name, desc, res)) in displays.iter().enumerate() {
                let is_selected = current_display.as_ref().map_or(i == 0, |d| d == name);
                let marker = if is_selected { " ← active" } else { "" };

                println!("  {} {}{}", i, name, marker);
                if !desc.is_empty() {
                    println!("      {}", desc);
                }
                println!("      {}", res);
                println!();
            }

            println!("Use 'palace display switch <index>' to switch displays.");
        }
        DisplayAction::Switch { index } => {
            if displays.is_empty() {
                anyhow::bail!("No displays found.");
            }

            if index >= displays.len() {
                anyhow::bail!(
                    "Invalid display index {}. Available: 0-{}",
                    index,
                    displays.len().saturating_sub(1)
                );
            }

            let (name, _, _) = &displays[index];

            // Save to config
            save_display_config(name)?;

            println!("📺 Switched to display: {}", name);
            println!("   Restart Palace for the change to take effect.");
        }
    }

    Ok(())
}

/// Get displays via GNOME's Mutter D-Bus interface
fn get_gnome_displays() -> Result<Vec<(String, String, String)>> {
    use std::process::Command;

    let output = Command::new("gdbus")
        .args([
            "call", "--session",
            "--dest", "org.gnome.Mutter.DisplayConfig",
            "--object-path", "/org/gnome/Mutter/DisplayConfig",
            "--method", "org.gnome.Mutter.DisplayConfig.GetCurrentState",
        ])
        .env("WAYLAND_DISPLAY", std::env::var("WAYLAND_DISPLAY").unwrap_or_else(|_| "wayland-0".to_string()))
        .output()?;

    if !output.status.success() {
        anyhow::bail!("gdbus call failed");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_gnome_displays(&stdout)
}

/// Parse GNOME Mutter DisplayConfig output
fn parse_gnome_displays(output: &str) -> Result<Vec<(String, String, String)>> {
    let mut displays = Vec::new();

    // Look for display entries: (('connector', 'vendor', 'product', 'serial'), ...)
    // Pattern: (('DP-2', 'RTK', 'WCS Display', 'demoset-1'), [modes...], {props})
    let mut remaining = output;

    while let Some(start) = remaining.find("(('") {
        remaining = &remaining[start + 3..];

        // Extract connector name (first field)
        if let Some(end) = remaining.find("',") {
            let connector = remaining[..end].to_string();
            remaining = &remaining[end + 2..];

            // Skip to get vendor/product
            let mut desc = String::new();
            if remaining.starts_with(" '") {
                remaining = &remaining[2..];
                if let Some(end) = remaining.find("', '") {
                    let vendor = &remaining[..end];
                    remaining = &remaining[end + 4..];

                    if let Some(end) = remaining.find("',") {
                        let product = &remaining[..end];
                        desc = format!("{} {}", vendor, product);
                    }
                }
            }

            // Find current resolution in modes list
            let mut resolution = String::new();
            if let Some(modes_start) = remaining.find('[') {
                let modes_section = &remaining[modes_start..];
                // Look for 'is-current': <true>
                if let Some(current_pos) = modes_section.find("'is-current': <true>") {
                    // Go back to find the mode string before this
                    let before_current = &modes_section[..current_pos];
                    // Find the last mode pattern like '1920x1080@60.000'
                    if let Some(mode_start) = before_current.rfind("('") {
                        let mode_str = &before_current[mode_start + 2..];
                        if let Some(mode_end) = mode_str.find("',") {
                            resolution = mode_str[..mode_end].to_string();
                        }
                    }
                }
            }

            if resolution.is_empty() {
                resolution = "unknown".to_string();
            }

            displays.push((connector, desc, resolution));
        }
    }

    if displays.is_empty() {
        anyhow::bail!("No displays found in GNOME output");
    }

    Ok(displays)
}

/// Fallback: get displays via wlr-randr or xrandr
fn get_randr_displays() -> Result<Vec<(String, String, String)>> {
    use std::process::Command;

    let output = if std::env::var("WAYLAND_DISPLAY").is_ok() {
        Command::new("wlr-randr").output()
    } else {
        Command::new("xrandr").arg("--query").output()
    }?;

    if !output.status.success() {
        anyhow::bail!("randr command failed");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut displays = Vec::new();

    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(' ') || line.starts_with('\t') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        let name = parts[0].to_string();
        if name.contains(':') || name.starts_with("--") {
            continue;
        }

        // Find resolution
        let mut resolution = String::new();
        for part in parts.iter().skip(1) {
            if part.contains('x') && part.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                resolution = part.to_string();
                break;
            }
        }

        if resolution.is_empty() {
            resolution = "unknown".to_string();
        }

        displays.push((name, String::new(), resolution));
    }

    if displays.is_empty() {
        anyhow::bail!("No displays found via randr");
    }

    Ok(displays)
}

/// Load display config (returns the display name if set)
fn load_display_config() -> Option<String> {
    let path = std::path::Path::new(DISPLAY_CONFIG_PATH);
    if path.exists() {
        if let Ok(json) = std::fs::read_to_string(path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&json) {
                return config.get("display").and_then(|v| v.as_str()).map(|s| s.to_string());
            }
        }
    }
    None
}

/// Save display config
fn save_display_config(display_name: &str) -> Result<()> {
    let path = std::path::Path::new(DISPLAY_CONFIG_PATH);

    // Create parent directory if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let config = serde_json::json!({
        "display": display_name
    });

    std::fs::write(path, serde_json::to_string_pretty(&config)?)?;
    Ok(())
}

/// Run the main GUI application
/// Serializable state for restart transfer
#[derive(serde::Serialize, serde::Deserialize)]
struct RestoreState {
    /// "chooser", "project", or "settings"
    view: String,
    /// Project path if in project view
    #[serde(default)]
    project_path: Option<String>,
    /// Selected index
    #[serde(default)]
    selected: usize,
    /// Previous state when in settings menu
    #[serde(default)]
    previous: Option<Box<RestoreState>>,
}

/// Convert RestoreState to AppState
fn restore_state_to_app_state(state: RestoreState) -> AppState {
    match state.view.as_str() {
        "project" => {
            if let Some(project_path) = state.project_path {
                AppState::ProjectView {
                    project_path: std::path::PathBuf::from(project_path),
                    selected_action: state.selected,
                }
            } else {
                AppState::ProjectChooser {
                    selected_index: state.selected,
                }
            }
        }
        "settings" => {
            // Restore settings menu with previous state
            let previous = if let Some(prev) = state.previous {
                Box::new(restore_state_to_app_state(*prev))
            } else {
                Box::new(AppState::project_chooser())
            };
            AppState::SettingsMenu {
                selected_item: state.selected,
                previous_state: previous,
            }
        }
        _ => AppState::ProjectChooser {
            selected_index: state.selected,
        },
    }
}

fn run_app(restore_path: Option<String>) -> Result<()> {
    // Set default Wayland display for remote/headless scenarios (GPD Win 4 target)
    if env::var("WAYLAND_DISPLAY").is_err() && env::var("DISPLAY").is_err() {
        env::set_var("WAYLAND_DISPLAY", "wayland-0");
    }

    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "palace=debug,wgpu=warn".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Palace GPU starting...");

    // Create tokio runtime for async tasks (debug server, etc.)
    let runtime = tokio::runtime::Runtime::new()?;
    let _guard = runtime.enter();

    // Create event loop with custom event type for event-driven architecture
    let event_loop = EventLoop::<app::AppEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();

    // Start debug server with event proxy (event-driven, no polling)
    let debug_proxy = proxy.clone();
    let _debug_server = debug::DebugServer::start_with_proxy(debug_proxy);

    // Start gamepad thread with event proxy (event-driven, no polling)
    let gamepad_proxy = proxy.clone();
    std::thread::spawn(move || {
        run_gamepad_thread(gamepad_proxy);
    });

    // Load projects config
    let mut projects_config = ProjectsConfig::load().unwrap_or_default();

    // Try to restore state from file, otherwise detect from launch context
    let initial_state = if let Some(ref path) = restore_path {
        if let Ok(json) = std::fs::read_to_string(path) {
            if let Ok(state) = serde_json::from_str::<RestoreState>(&json) {
                tracing::info!("Restoring state from {}", path);
                // Clean up the state file
                let _ = std::fs::remove_file(path);

                restore_state_to_app_state(state)
            } else {
                tracing::warn!("Failed to parse restore state");
                detect_initial_state(&mut projects_config)
            }
        } else {
            tracing::warn!("Failed to read restore file: {}", path);
            detect_initial_state(&mut projects_config)
        }
    } else {
        detect_initial_state(&mut projects_config)
    };

    // Create and run application
    let app_proxy = proxy.clone();
    let mut app = app::App::new(initial_state, projects_config, app_proxy);

    event_loop.run_app(&mut app)?;

    Ok(())
}

/// Detect initial state from launch directory
fn detect_initial_state(projects_config: &mut ProjectsConfig) -> AppState {
    match detect_launch_context() {
        LaunchContext::Home => {
            tracing::info!("Launched from home - showing project chooser");
            AppState::project_chooser()
        }
        LaunchContext::Project(path) => {
            tracing::info!("Launched from project: {}", path.display());
            // Add to projects list if new
            if projects_config.add_project(path.clone()) {
                if let Err(e) = projects_config.save() {
                    tracing::warn!("Failed to save projects config: {}", e);
                }
            }
            AppState::project_view(path)
        }
    }
}

/// Gamepad event thread - polls gilrs and sends events to main loop
fn run_gamepad_thread(proxy: winit::event_loop::EventLoopProxy<app::AppEvent>) {
    use gilrs::{Event as GilrsEvent, Gilrs};

    let mut gilrs = match Gilrs::new() {
        Ok(g) => {
            // Log connected gamepads
            for (id, gamepad) in g.gamepads() {
                tracing::info!("Gamepad found: {} ({:?})", gamepad.name(), id);
                // Notify main thread
                let _ = proxy.send_event(app::AppEvent::GamepadConnected);
            }
            g
        }
        Err(e) => {
            tracing::warn!("Failed to initialize gamepad support: {}", e);
            return;
        }
    };

    // Event-driven: gilrs provides blocking next_event_blocking()
    loop {
        // Block until gamepad event occurs (true event-driven, zero CPU when idle)
        if let Some(GilrsEvent { event, .. }) = gilrs.next_event_blocking(None) {
            match event {
                gilrs::EventType::ButtonPressed(button, _) => {
                    if proxy.send_event(app::AppEvent::GamepadButton(button)).is_err() {
                        break; // Event loop closed
                    }
                }
                gilrs::EventType::ButtonReleased(button, _) => {
                    if proxy.send_event(app::AppEvent::GamepadButtonReleased(button)).is_err() {
                        break;
                    }
                }
                gilrs::EventType::Connected => {
                    if proxy.send_event(app::AppEvent::GamepadConnected).is_err() {
                        break;
                    }
                }
                gilrs::EventType::Disconnected => {
                    if proxy.send_event(app::AppEvent::GamepadDisconnected).is_err() {
                        break;
                    }
                }
                gilrs::EventType::AxisChanged(axis, value, _) => {
                    // Track left and right stick X and Y, send combined events
                    use gilrs::Axis;
                    static mut LEFT_X: f32 = 0.0;
                    static mut LEFT_Y: f32 = 0.0;
                    static mut RIGHT_X: f32 = 0.0;
                    static mut RIGHT_Y: f32 = 0.0;

                    let (send_left, send_right) = unsafe {
                        match axis {
                            Axis::LeftStickX => {
                                LEFT_X = value;
                                (true, false)
                            }
                            Axis::LeftStickY => {
                                LEFT_Y = value;
                                (true, false)
                            }
                            Axis::RightStickX => {
                                RIGHT_X = value;
                                (false, true)
                            }
                            Axis::RightStickY => {
                                RIGHT_Y = value;
                                (false, true)
                            }
                            _ => (false, false),
                        }
                    };

                    if send_left {
                        let (x, y) = unsafe { (LEFT_X, LEFT_Y) };
                        if proxy.send_event(app::AppEvent::GamepadStick { x, y }).is_err() {
                            break;
                        }
                    }
                    if send_right {
                        let (x, y) = unsafe { (RIGHT_X, RIGHT_Y) };
                        if proxy.send_event(app::AppEvent::GamepadRightStick { x, y }).is_err() {
                            break;
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

/// Install signal handlers for SIGSEGV and other crash signals
fn install_signal_handlers() {
    extern "C" fn crash_handler(sig: libc::c_int) {
        // Use write() directly since we can't use allocation in signal handler
        let msg: &[u8] = match sig {
            libc::SIGSEGV => b"\n\x1b[31m=== PALACE SEGFAULT (SIGSEGV) ===\x1b[0m\n\nMemory access violation - likely GPU driver issue or invalid memory access.\nSet RUST_BACKTRACE=full and run again for more details.\n\n",
            libc::SIGBUS => b"\n\x1b[31m=== PALACE BUS ERROR (SIGBUS) ===\x1b[0m\n\nBus error - memory alignment or I/O issue.\nSet RUST_BACKTRACE=full and run again for more details.\n\n",
            libc::SIGFPE => b"\n\x1b[31m=== PALACE FLOATING POINT EXCEPTION ===\x1b[0m\n\nArithmetic error (division by zero, etc).\n\n",
            libc::SIGILL => b"\n\x1b[31m=== PALACE ILLEGAL INSTRUCTION ===\x1b[0m\n\nIllegal CPU instruction - possibly corrupted code.\n\n",
            libc::SIGABRT => b"\n\x1b[31m=== PALACE ABORTED (SIGABRT) ===\x1b[0m\n\nProcess aborted - assertion failure or abort() called.\n\n",
            _ => b"\n\x1b[31m=== PALACE CRASHED ===\x1b[0m\n\nUnexpected signal received.\n\n",
        };

        // Write directly to stderr (fd 2)
        unsafe {
            libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
        }

        // Re-raise the signal with default handler to get core dump if enabled
        unsafe {
            libc::signal(sig, libc::SIG_DFL);
            libc::raise(sig);
        }
    }

    // Install handlers for crash signals
    unsafe {
        libc::signal(libc::SIGSEGV, crash_handler as libc::sighandler_t);
        libc::signal(libc::SIGBUS, crash_handler as libc::sighandler_t);
        libc::signal(libc::SIGFPE, crash_handler as libc::sighandler_t);
        libc::signal(libc::SIGILL, crash_handler as libc::sighandler_t);
        libc::signal(libc::SIGABRT, crash_handler as libc::sighandler_t);
    }
}
