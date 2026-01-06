use anyhow::Result;
use anthropic_ffi::{models, Client as AnthropicClient, StreamEvent};
use glob::glob;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Interactive menu with arrow key navigation
/// Returns the index of the selected option, or None if cancelled
fn select_menu(prompt: &str, options: &[&str]) -> Option<usize> {
    use std::os::unix::io::AsRawFd;

    let stdin = io::stdin();
    let stdin_fd = stdin.as_raw_fd();

    // Save terminal settings
    let orig_termios = unsafe {
        let mut termios: libc::termios = std::mem::zeroed();
        libc::tcgetattr(stdin_fd, &mut termios);
        termios
    };

    // Set raw mode
    unsafe {
        let mut raw = orig_termios;
        libc::cfmakeraw(&mut raw);
        // Keep some settings for better experience
        raw.c_oflag |= libc::OPOST; // Output processing
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;
        libc::tcsetattr(stdin_fd, libc::TCSANOW, &raw);
    }

    let mut selected: usize = 0;
    let result;

    // Print prompt
    eprint!("\n\x1b[33m⚠ {}\x1b[0m\n", prompt);

    loop {
        // Clear and redraw options
        for (i, opt) in options.iter().enumerate() {
            if i == selected {
                eprintln!("\x1b[36m❯ {}\x1b[0m", opt);
            } else {
                eprintln!("  {}", opt);
            }
        }
        let _ = io::stderr().flush();

        // Read input
        let mut buf = [0u8; 3];
        let n = io::stdin().read(&mut buf).unwrap_or(0);

        if n == 0 {
            result = None;
            break;
        }

        match &buf[..n] {
            // Enter
            [13] | [10] => {
                result = Some(selected);
                break;
            }
            // Escape or Ctrl-C
            [27] | [3] => {
                result = None;
                break;
            }
            // Arrow up
            [27, 91, 65] => {
                if selected > 0 {
                    selected -= 1;
                }
            }
            // Arrow down
            [27, 91, 66] => {
                if selected < options.len() - 1 {
                    selected += 1;
                }
            }
            // Number keys 1-9
            [c] if *c >= b'1' && *c <= b'9' => {
                let idx = (*c - b'1') as usize;
                if idx < options.len() {
                    result = Some(idx);
                    break;
                }
            }
            _ => {}
        }

        // Move cursor up to redraw
        eprint!("\x1b[{}A", options.len());
    }

    // Restore terminal
    unsafe {
        libc::tcsetattr(stdin_fd, libc::TCSANOW, &orig_termios);
    }

    // Clear the menu lines
    for _ in 0..options.len() {
        eprintln!("\x1b[2K"); // Clear line
    }
    eprint!("\x1b[{}A", options.len()); // Move back up

    result
}

use crate::state::PermissionResponse;

/// Permission requester callback type
/// Returns PermissionResponse indicating user's choice
pub type PermissionRequester = Box<dyn Fn(&str) -> PermissionResponse + Send + Sync>;

/// Execute a tool and return its result
fn execute_tool(
    tool_name: &str,
    tool_input: &str,
    project_root: &Path,
    permission_requester: Option<&PermissionRequester>,
) -> String {
    let input: serde_json::Value = serde_json::from_str(tool_input).unwrap_or_default();

    match tool_name {
        "read_file" => {
            let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let full_path = project_root.join(path);
            match std::fs::read_to_string(&full_path) {
                Ok(content) => {
                    // Truncate if too long (use chars to avoid UTF-8 boundary issues)
                    if content.len() > 8000 {
                        let truncated: String = content.chars().take(8000).collect();
                        format!("{}\n... (truncated, {} bytes total)", truncated, content.len())
                    } else {
                        content
                    }
                }
                Err(e) => format!("Error reading file: {}", e),
            }
        }
        "list_directory" => {
            let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            let full_path = project_root.join(path);
            match std::fs::read_dir(&full_path) {
                Ok(entries) => {
                    let mut files: Vec<String> = entries
                        .flatten()
                        .map(|e| {
                            let name = e.file_name().to_string_lossy().to_string();
                            if e.path().is_dir() {
                                format!("{}/", name)
                            } else {
                                name
                            }
                        })
                        .collect();
                    files.sort();
                    files.join("\n")
                }
                Err(e) => format!("Error listing directory: {}", e),
            }
        }
        "search_files" => {
            let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("*");
            let full_pattern = project_root.join(pattern);
            match glob(full_pattern.to_str().unwrap_or("*")) {
                Ok(paths) => {
                    let files: Vec<String> = paths
                        .flatten()
                        .take(50)
                        .filter_map(|p| p.strip_prefix(project_root).ok().map(|p| p.to_string_lossy().to_string()))
                        .collect();
                    if files.is_empty() {
                        "No files found matching pattern".to_string()
                    } else {
                        files.join("\n")
                    }
                }
                Err(e) => format!("Error searching files: {}", e),
            }
        }
        "run_command" => {
            let cmd = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
            // Safe read-only commands that don't need permission
            let safe_commands = [
                "git status", "git log", "git diff", "git show", "git branch",
                "cargo check", "cargo clippy", "cargo test", "cargo build",
                "ls", "cat", "head", "tail", "wc", "grep", "find", "tree",
            ];
            let is_safe = safe_commands.iter().any(|a| cmd.starts_with(a));

            // If not safe, prompt for permission
            if !is_safe {
                // Check if user has approved this command pattern before
                use std::sync::{OnceLock, Mutex as StdMutex};
                static APPROVED: OnceLock<StdMutex<Vec<String>>> = OnceLock::new();
                let approved = APPROVED.get_or_init(|| StdMutex::new(Vec::new()));

                // Check for "always" approval of command prefix
                let cmd_prefix = cmd.split_whitespace().next().unwrap_or(cmd);
                {
                    let list = approved.lock().unwrap();
                    if list.iter().any(|a| cmd.starts_with(a)) {
                        // Already approved
                    } else {
                        drop(list); // Release lock before prompting

                        // Use GUI permission requester if available, otherwise terminal
                        let response = if let Some(requester) = permission_requester {
                            requester(cmd)
                        } else {
                            // Fallback to terminal menu
                            let prompt = format!("Permission required: {}", cmd);
                            let always_opt = format!("Yes (always for '{}')", cmd_prefix);
                            let options = ["Yes (once)", &always_opt, "No"];

                            match select_menu(&prompt, &options) {
                                Some(0) => PermissionResponse::Approved,
                                Some(1) => {
                                    // Always approve this prefix
                                    approved.lock().unwrap().push(cmd_prefix.to_string());
                                    PermissionResponse::ApprovedAlways(cmd_prefix.to_string())
                                }
                                _ => PermissionResponse::Denied,
                            }
                        };

                        match &response {
                            PermissionResponse::Approved => {
                                // Continue with execution
                            }
                            PermissionResponse::ApprovedAlways(prefix) => {
                                // Add to approved list
                                approved.lock().unwrap().push(prefix.clone());
                            }
                            PermissionResponse::Denied => {
                                return format!("Denied: {}", cmd);
                            }
                            PermissionResponse::SuggestElse { original_command } => {
                                // User wants alternatives - return special marker
                                return format!("SUGGEST_ELSE: User rejected '{}' and wants alternative approaches", original_command);
                            }
                        }
                    }
                }
            }

            match Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(project_root)
                .output()
            {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    if output.status.success() {
                        if stdout.len() > 4000 {
                            let truncated: String = stdout.chars().take(4000).collect();
                            format!("{}\n... (truncated)", truncated)
                        } else {
                            stdout.to_string()
                        }
                    } else {
                        format!("Command failed: {}", stderr)
                    }
                }
                Err(e) => format!("Error running command: {}", e),
            }
        }
        _ => format!("Unknown tool: {}", tool_name),
    }
}

/// Get emoji for tool type
fn tool_emoji(tool_name: &str) -> &'static str {
    match tool_name {
        "read_file" => "📖",
        "list_directory" => "📁",
        "search_files" => "🔍",
        "run_command" => "💻",
        _ => "⚙",
    }
}

/// Convert tool call to human-readable description with emoji
fn tool_description(tool_name: &str, input_json: &str) -> String {
    let input: serde_json::Value = serde_json::from_str(input_json).unwrap_or_default();

    match tool_name {
        // Claude Code tools
        "Read" => {
            let path = input
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            format!("📖 Reading {}", path)
        }
        "Bash" => {
            let cmd = input
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            if cmd.len() > 50 {
                format!("💻 {:.50}...", cmd)
            } else {
                format!("💻 {}", cmd)
            }
        }
        "Grep" => {
            let pattern = input
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("*");
            format!("🔍 grep {}", pattern)
        }
        "Glob" => {
            let pattern = input
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("*");
            format!("📁 glob {}", pattern)
        }
        "Edit" => {
            let path = input
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            format!("✏️ Editing {}", path)
        }
        "Write" => {
            let path = input
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            format!("📝 Writing {}", path)
        }
        // Legacy tool names
        "read_file" => {
            let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            format!("📖 Reading {}", path)
        }
        "list_directory" => {
            let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            format!("📁 Listing {}/", path)
        }
        "search_files" => {
            let pattern = input
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("*");
            format!("🔍 Searching {}", pattern)
        }
        "run_command" => {
            let cmd = input
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            if cmd.len() > 50 {
                format!("💻 {:.50}...", cmd)
            } else {
                format!("💻 {}", cmd)
            }
        }
        _ => format!("🔧 {} {:?}", tool_name, input),
    }
}

/// Create a short preview of tool result
fn result_preview(result: &str) -> String {
    // Take first few lines, join with separator
    let lines: Vec<&str> = result.lines().take(2).collect();
    let preview = lines.join(" ");

    // Truncate if too long (use chars to avoid UTF-8 boundary issues)
    if preview.len() > 80 {
        let truncated: String = preview.chars().take(80).collect();
        format!("{}...", truncated)
    } else {
        preview
    }
}

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
    #[allow(dead_code)]
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

    /// Generate suggestions with streaming output in YAML format
    /// Streams plain text directly to stdout as it arrives
    #[allow(dead_code)]
    pub fn suggest_streaming(&self, context: &ProjectContext) -> Result<()> {
        let prompt = self.build_streaming_prompt(context);
        let system = "You are Palace, an AI assistant that analyzes software projects and suggests actionable next steps. Output clean YAML only, no markdown fences.";

        let mut stdout = io::stdout();
        let mut in_tool = false;

        self.client.message_stream(
            &self.model,
            Some(system),
            &prompt,
            4096,
            move |event| {
                match event {
                    StreamEvent::Text(chunk) => {
                        print!("{}", chunk);
                        let _ = stdout.flush();
                    }
                    StreamEvent::ToolUseStart(tool_info) => {
                        in_tool = true;
                        // Parse "id:name" format
                        let name = tool_info.split(':').last().unwrap_or(&tool_info);
                        eprint!("\n\x1b[36m⚙ Tool: {}\x1b[0m ", name);
                        let _ = io::stderr().flush();
                    }
                    StreamEvent::ToolUseInput(json) => {
                        if in_tool {
                            eprint!("{}", json);
                            let _ = io::stderr().flush();
                        }
                    }
                    StreamEvent::ToolUseEnd => {
                        if in_tool {
                            eprintln!();
                            in_tool = false;
                        }
                    }
                    StreamEvent::Thinking(text) => {
                        eprint!("\x1b[90m{}\x1b[0m", text);
                        let _ = io::stderr().flush();
                    }
                    StreamEvent::Done => {
                        println!();
                    }
                    StreamEvent::Error(e) => {
                        eprintln!("\n\x1b[31mStream error: {}\x1b[0m", e);
                    }
                    StreamEvent::ToolResult(result) => {
                        eprintln!("\x1b[32m  → {}\x1b[0m", result);
                    }
                }
            },
        )?;

        Ok(())
    }

    /// Generate suggestions with agentic exploration
    /// Claude will explore the project using tools before suggesting actions
    pub fn suggest_with_exploration(&self, context: &ProjectContext) -> Result<()> {
        let prompt = self.build_exploration_prompt(context);
        let system = r#"You are Palace, an AI that explores software projects to understand them deeply before suggesting actions.

IMPORTANT: First use your tools to explore the project structure and read key files. Then provide your suggestions.

When exploring:
1. Read important config files (Cargo.toml, package.json, etc.)
2. Check git status for recent changes
3. Look at key source files to understand the architecture
4. Check for TODO comments or issues

After exploring, output your suggestions in YAML format (no markdown fences)."#;

        let tools_json = self.get_exploration_tools();
        let project_root = context.root.clone();
        let mut in_tool = false;

        // Track tool calls in order, results come in same order
        use std::sync::{Arc, Mutex};
        use std::collections::VecDeque;
        let pending_tools: Arc<Mutex<VecDeque<(String, String)>>> = Arc::new(Mutex::new(VecDeque::new()));
        let current_tool_input = Arc::new(Mutex::new(String::new()));
        let current_tool_name = Arc::new(Mutex::new(String::new()));
        let had_text = Arc::new(Mutex::new(false));
        let pending_clone = pending_tools.clone();
        let input_clone = current_tool_input.clone();
        let name_clone = current_tool_name.clone();
        let had_text_clone = had_text.clone();
        let had_text_result = had_text.clone();

        self.client.agentic_loop(
            &self.model,
            Some(system),
            &prompt,
            8192,
            &tools_json,
            move |event| {
                match event {
                    StreamEvent::Text(chunk) => {
                        print!("{}", chunk);
                        let _ = io::stdout().flush();
                        *had_text_clone.lock().unwrap() = true;
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
                            // Buffer the tool call - will print when result arrives
                            pending_clone.lock().unwrap().push_back((name, input));
                            in_tool = false;
                        }
                    }
                    StreamEvent::ToolResult(result) => {
                        // Newline after text before first tool output
                        {
                            let mut had = had_text_result.lock().unwrap();
                            if *had {
                                eprintln!();
                                *had = false;
                            }
                        }
                        // Get matching tool call and print together
                        if let Some((name, input)) = pending_clone.lock().unwrap().pop_front() {
                            let emoji = tool_emoji(&name);
                            let desc = tool_description(&name, &input);
                            let preview = result_preview(&result);
                            eprintln!("{} {} {}", emoji, desc, preview);
                        }
                    }
                    StreamEvent::Thinking(text) => {
                        eprint!("\x1b[90m{}\x1b[0m", text);
                        let _ = io::stderr().flush();
                    }
                    StreamEvent::Done => {
                        println!();
                    }
                    StreamEvent::Error(e) => {
                        eprintln!("\n\x1b[31mError: {}\x1b[0m", e);
                    }
                }
            },
            move |tool_name, tool_input| {
                execute_tool(tool_name, tool_input, &project_root, None)
            },
        )?;

        Ok(())
    }

    fn build_exploration_prompt(&self, context: &ProjectContext) -> String {
        format!(r#"Analyze this project by exploring its files.

{}

First, use tools to explore key files and understand the project. Then output your suggestions in YAML:

suggestions:
  - title: Short action title
    category: test|fix|build|refactor|docs
    description: What to do
    command: optional shell command"#,
            context.summary()
        )
    }

    fn get_exploration_tools(&self) -> String {
        r#"[
            {
                "name": "read_file",
                "description": "Read the contents of a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file relative to project root"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "list_directory",
                "description": "List files in a directory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path relative to project root"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "search_files",
                "description": "Search for files matching a pattern",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern like *.rs or src/**/*.ts"}
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "run_command",
                "description": "Run a shell command (read-only commands only)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run (e.g., git status, cargo check)"}
                    },
                    "required": ["command"]
                }
            }
        ]"#.to_string()
    }

    #[allow(dead_code)]
    fn build_streaming_prompt(&self, context: &ProjectContext) -> String {
        format!(r#"Analyze this project and suggest possible next actions.

{}

Output YAML format (no markdown fences, just raw YAML):

suggestions:
  - title: Short action title
    category: test|fix|build|refactor|docs
    description: What to do
    command: optional shell command

Provide as many suggestions as make sense. Be concrete and actionable."#,
            context.summary()
        )
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

    /// Stream suggestions to an event loop proxy for GUI display
    /// Parses YAML as it streams and sends AppEvents
    pub fn stream_to_gui<F>(
        &self,
        context: &ProjectContext,
        send_event: F,
        permission_requester: Option<PermissionRequester>,
    ) -> Result<()>
    where
        F: Fn(SuggestionEvent) + Send + 'static,
    {
        let prompt = self.build_exploration_prompt(context);
        let system = r#"You are Palace, an AI that explores software projects to understand them deeply before suggesting actions.

IMPORTANT: First use your tools to explore the project structure and read key files. Then provide your suggestions.

After exploring, output your suggestions in YAML format (no markdown fences):

suggestions:
  - title: Short action title
    category: test|fix|build|refactor|docs
    description: What to do
    command: optional shell command"#;

        let tools_json = self.get_exploration_tools();
        let project_root = context.root.clone();

        // State for parsing YAML suggestions as they stream
        use std::sync::{Arc, Mutex};
        use std::collections::VecDeque;

        let pending_tools: Arc<Mutex<VecDeque<(String, String)>>> = Arc::new(Mutex::new(VecDeque::new()));
        let current_tool_input = Arc::new(Mutex::new(String::new()));
        let current_tool_name = Arc::new(Mutex::new(String::new()));
        let yaml_buffer = Arc::new(Mutex::new(String::new()));
        let chatter_buffer = Arc::new(Mutex::new(String::new()));
        let parser_state = Arc::new(Mutex::new(YamlParserState::default()));
        let in_suggestions = Arc::new(Mutex::new(false));

        let pending_clone = pending_tools.clone();
        let input_clone = current_tool_input.clone();
        let name_clone = current_tool_name.clone();
        let yaml_clone = yaml_buffer.clone();
        let chatter_clone = chatter_buffer.clone();
        let parser_clone = parser_state.clone();
        let in_sugg_clone = in_suggestions.clone();
        let send_clone = Arc::new(send_event);
        let send_tool = send_clone.clone();
        let send_result = send_clone.clone();

        let mut in_tool = false;

        self.client.agentic_loop(
            &self.model,
            Some(system),
            &prompt,
            8192,
            &tools_json,
            move |event| {
                match event {
                    StreamEvent::Text(chunk) => {
                        // Buffer YAML and parse suggestions incrementally
                        let mut buf = yaml_clone.lock().unwrap();
                        buf.push_str(&chunk);

                        // Check if we've started suggestions section
                        if !*in_sugg_clone.lock().unwrap() {
                            if buf.contains("suggestions:") {
                                *in_sugg_clone.lock().unwrap() = true;
                            } else {
                                // Before suggestions: buffer text, emit on sentence/paragraph end
                                let mut chatter = chatter_clone.lock().unwrap();
                                chatter.push_str(&chunk);

                                // Flush on sentence endings or newlines
                                loop {
                                    // Find first flush point: newline or sentence end
                                    let flush_at = chatter
                                        .find('\n')
                                        .or_else(|| {
                                            // Look for sentence endings followed by space or end
                                            chatter.find(". ").map(|i| i + 1)
                                                .or_else(|| chatter.find("! ").map(|i| i + 1))
                                                .or_else(|| chatter.find("? ").map(|i| i + 1))
                                        });

                                    if let Some(pos) = flush_at {
                                        let line = chatter[..pos].trim().to_string();
                                        *chatter = chatter[pos..].trim_start().to_string();

                                        if !line.is_empty() && !line.starts_with("```") {
                                            send_clone(SuggestionEvent::Chatter(line));
                                        }
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }

                        // Try to parse complete suggestion entries
                        if *in_sugg_clone.lock().unwrap() {
                            parse_streaming_yaml(&buf, &parser_clone, &send_clone);
                        }
                    }
                    StreamEvent::ToolUseStart(tool_info) => {
                        // Flush any pending chatter - tool start means thought ended
                        {
                            let mut chatter = chatter_clone.lock().unwrap();
                            let remaining = chatter.trim().to_string();
                            if !remaining.is_empty() && !remaining.starts_with("```") {
                                send_clone(SuggestionEvent::Chatter(remaining));
                            }
                            chatter.clear();
                        }

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
                    StreamEvent::ToolResult(_result) => {
                        if let Some((name, input)) = pending_clone.lock().unwrap().pop_front() {
                            let desc = tool_description(&name, &input);
                            send_tool(SuggestionEvent::ToolCall(desc));
                        }
                    }
                    StreamEvent::Done => {
                        send_result(SuggestionEvent::Done);
                    }
                    StreamEvent::Error(e) => {
                        send_result(SuggestionEvent::Error(e));
                    }
                    _ => {}
                }
            },
            move |tool_name, tool_input| {
                execute_tool(tool_name, tool_input, &project_root, permission_requester.as_ref())
            },
        )?;

        Ok(())
    }
}

/// Events sent during suggestion streaming
#[derive(Debug, Clone)]
pub enum SuggestionEvent {
    /// Tool being called (human-readable description)
    ToolCall(String),
    /// AI thinking/commentary text
    Chatter(String),
    /// New suggestion card started
    CardStart { id: usize },
    /// Card field updated
    CardUpdate { id: usize, field: String, value: String },
    /// Card is complete
    CardComplete { id: usize },
    /// All done
    Done,
    /// Error occurred
    Error(String),
}

/// State for incremental YAML parsing
#[derive(Debug, Default)]
struct YamlParserState {
    /// Number of complete lines we've already processed
    processed_lines: usize,
    /// Current card being built (if any)
    current_card_id: Option<usize>,
    /// Next card ID to assign
    next_card_id: usize,
    /// Fields we've already emitted for current card
    emitted_fields: std::collections::HashSet<String>,
}

/// Parse YAML buffer incrementally, only processing new complete lines
fn parse_streaming_yaml(
    buffer: &str,
    state: &std::sync::Arc<std::sync::Mutex<YamlParserState>>,
    send: &std::sync::Arc<impl Fn(SuggestionEvent)>,
) {
    let mut state = state.lock().unwrap();

    // Split into lines, keeping track of complete vs incomplete
    let lines: Vec<&str> = buffer.lines().collect();
    let buffer_ends_with_newline = buffer.ends_with('\n');

    // Determine how many complete lines we have
    let complete_lines = if buffer_ends_with_newline {
        lines.len()
    } else if lines.is_empty() {
        0
    } else {
        lines.len() - 1  // Last line might be incomplete
    };

    // Only process new complete lines
    for i in state.processed_lines..complete_lines {
        let line = lines[i];
        let trimmed = line.trim();

        // New card starts with "- title:"
        if trimmed.starts_with("- title:") {
            // Complete previous card if any
            if let Some(id) = state.current_card_id.take() {
                send(SuggestionEvent::CardComplete { id });
            }

            // Start new card
            let id = state.next_card_id;
            state.next_card_id += 1;
            state.current_card_id = Some(id);
            state.emitted_fields.clear();

            send(SuggestionEvent::CardStart { id });

            if let Some(value) = trimmed.strip_prefix("- title:") {
                let value = value.trim();
                if !value.is_empty() {
                    send(SuggestionEvent::CardUpdate {
                        id,
                        field: "title".to_string(),
                        value: value.to_string(),
                    });
                    state.emitted_fields.insert("title".to_string());
                }
            }
        } else if let Some(id) = state.current_card_id {
            // Parse fields for current card
            let field_value = if let Some(v) = trimmed.strip_prefix("category:") {
                Some(("category", v.trim()))
            } else if let Some(v) = trimmed.strip_prefix("description:") {
                Some(("description", v.trim()))
            } else if let Some(v) = trimmed.strip_prefix("command:") {
                Some(("command", v.trim()))
            } else if let Some(v) = trimmed.strip_prefix("title:") {
                // title without "- " prefix (continuation)
                Some(("title", v.trim()))
            } else {
                None
            };

            if let Some((field, value)) = field_value {
                if !value.is_empty() && !state.emitted_fields.contains(field) {
                    send(SuggestionEvent::CardUpdate {
                        id,
                        field: field.to_string(),
                        value: value.to_string(),
                    });
                    state.emitted_fields.insert(field.to_string());
                }
            }

            // Empty line or new list item might signal end of card
            if trimmed.is_empty() || (trimmed.starts_with('-') && !trimmed.starts_with("- title:")) {
                // Mark card complete if we have required fields
                if state.emitted_fields.contains("title") {
                    send(SuggestionEvent::CardComplete { id });
                    state.current_card_id = None;
                    state.emitted_fields.clear();
                }
            }
        }
    }

    state.processed_lines = complete_lines;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ============== ProjectType Detection Tests ==============
    #[test]
    fn test_project_type_detection_rust() {
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

        assert_eq!(
            ProjectContext::detect_type(&ctx.root, &ctx.files),
            ProjectType::Rust
        );
    }

    #[test]
    fn test_project_type_detection_python() {
        let mut files = HashMap::new();
        files.insert("pyproject.toml".to_string(), FileInfo { size: 100, is_dir: false });

        let ctx = ProjectContext {
            root: PathBuf::from("/test"),
            files,
            git_status: None,
            git_branch: None,
            recent_history: vec![],
            project_type: ProjectType::Unknown,
        };

        assert_eq!(
            ProjectContext::detect_type(&ctx.root, &ctx.files),
            ProjectType::Python
        );
    }

    #[test]
    fn test_project_type_detection_typescript() {
        let mut files = HashMap::new();
        files.insert("package.json".to_string(), FileInfo { size: 100, is_dir: false });
        files.insert("tsconfig.json".to_string(), FileInfo { size: 50, is_dir: false });

        let ctx = ProjectContext {
            root: PathBuf::from("/test"),
            files,
            git_status: None,
            git_branch: None,
            recent_history: vec![],
            project_type: ProjectType::Unknown,
        };

        assert_eq!(
            ProjectContext::detect_type(&ctx.root, &ctx.files),
            ProjectType::TypeScript
        );
    }

    #[test]
    fn test_project_type_detection_javascript() {
        let mut files = HashMap::new();
        files.insert("package.json".to_string(), FileInfo { size: 100, is_dir: false });

        let ctx = ProjectContext {
            root: PathBuf::from("/test"),
            files,
            git_status: None,
            git_branch: None,
            recent_history: vec![],
            project_type: ProjectType::Unknown,
        };

        assert_eq!(
            ProjectContext::detect_type(&ctx.root, &ctx.files),
            ProjectType::JavaScript
        );
    }

    #[test]
    fn test_project_type_detection_go() {
        let mut files = HashMap::new();
        files.insert("go.mod".to_string(), FileInfo { size: 100, is_dir: false });

        let ctx = ProjectContext {
            root: PathBuf::from("/test"),
            files,
            git_status: None,
            git_branch: None,
            recent_history: vec![],
            project_type: ProjectType::Unknown,
        };

        assert_eq!(
            ProjectContext::detect_type(&ctx.root, &ctx.files),
            ProjectType::Go
        );
    }

    #[test]
    fn test_project_type_detection_unknown() {
        let files = HashMap::new();

        let ctx = ProjectContext {
            root: PathBuf::from("/test"),
            files,
            git_status: None,
            git_branch: None,
            recent_history: vec![],
            project_type: ProjectType::Unknown,
        };

        assert_eq!(
            ProjectContext::detect_type(&ctx.root, &ctx.files),
            ProjectType::Unknown
        );
    }

    // ============== ProjectContext Summary Tests ==============
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
    fn test_context_summary_with_clean_git() {
        let mut files = HashMap::new();
        files.insert("src/main.rs".to_string(), FileInfo { size: 1000, is_dir: false });

        let ctx = ProjectContext {
            root: PathBuf::from("/test/project"),
            files,
            git_status: Some("".to_string()), // Clean status
            git_branch: Some("main".to_string()),
            recent_history: vec![],
            project_type: ProjectType::Rust,
        };

        let summary = ctx.summary();
        assert!(summary.contains("clean"));
    }

    #[test]
    fn test_context_summary_with_history() {
        let mut files = HashMap::new();
        files.insert("src/main.rs".to_string(), FileInfo { size: 1000, is_dir: false });

        let mut history = Vec::new();
        history.push(serde_json::json!({"action": "ran tests"}));
        history.push(serde_json::json!({"action": "built project"}));

        let ctx = ProjectContext {
            root: PathBuf::from("/test/project"),
            files,
            git_status: None,
            git_branch: None,
            recent_history: history,
            project_type: ProjectType::Rust,
        };

        let summary = ctx.summary();
        assert!(summary.contains("Recent actions"));
    }

    // ============== Suggestion Parsing Tests ==============
    #[test]
    fn test_parse_suggestions() {
        let engine = SuggestionEngine::new(Some("test-key"), Some("test-model")).unwrap();

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

    #[test]
    fn test_parse_multiple_suggestions() {
        let engine = SuggestionEngine::new(Some("test-key"), Some("test-model")).unwrap();

        let response = r#"[
  {
    "title": "Run tests",
    "description": "Ensure all tests pass",
    "category": "test",
    "priority": 1,
    "command": "cargo test"
  },
  {
    "title": "Fix linting",
    "description": "Fix clippy warnings",
    "category": "fix",
    "priority": 2,
    "command": "cargo clippy --fix"
  },
  {
    "title": "Update docs",
    "description": "Update README with new features",
    "category": "docs",
    "priority": 3
  }
]"#;

        let suggestions = engine.parse_suggestions(response).unwrap();
        assert_eq!(suggestions.len(), 3);
        assert_eq!(suggestions[0].title, "Run tests");
        assert_eq!(suggestions[1].category, "fix");
        assert_eq!(suggestions[2].category, "docs");
        assert!(suggestions[2].command.is_none());
    }

    #[test]
    fn test_parse_suggestions_with_markdown() {
        let engine = SuggestionEngine::new(Some("test-key"), Some("test-model")).unwrap();

        // Response with markdown code blocks
        let response = r#"Here are my suggestions for your project:

```json
[
  {
    "title": "Add tests",
    "description": "Write unit tests",
    "category": "test",
    "priority": 1
  }
]
```

Let me know if you need more!"#;

        let suggestions = engine.parse_suggestions(response).unwrap();
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].title, "Add tests");
    }

    #[test]
    fn test_parse_invalid_json() {
        let engine = SuggestionEngine::new(Some("test-key"), Some("test-model")).unwrap();

        let response = "This is not valid JSON {][}";

        let result = engine.parse_suggestions(response);
        assert!(result.is_err());
    }

    // ============== Tool Description Tests ==============
    #[test]
    fn test_tool_description_read() {
        let desc = tool_description("Read", r#"{"file_path": "/test.rs"}"#);
        assert!(desc.starts_with("📖"), "Read tool should start with 📖, got: {}", desc);
        assert!(desc.contains("/test.rs"));
    }

    #[test]
    fn test_tool_description_bash() {
        let desc = tool_description("Bash", r#"{"command": "cargo build"}"#);
        assert!(desc.starts_with("💻"), "Bash tool should start with 💻, got: {}", desc);

        // Test truncation of long commands
        let long_cmd = "cargo build --release --features all --target x86_64-unknown-linux-gnu && echo done";
        let desc_long = tool_description("Bash", &serde_json::json!({"command": long_cmd}).to_string());
        // Total: emoji (4 bytes) + space (1) + 50 chars + "..." (3) = 58 bytes
        assert!(desc_long.len() <= 58, "Expected ≤58 bytes, got {}", desc_long.len());
        assert!(desc_long.ends_with("..."));
    }

    #[test]
    fn test_tool_description_grep() {
        let desc = tool_description("Grep", r#"{"pattern": "TODO"}"#);
        assert!(desc.starts_with("🔍"), "Grep tool should start with 🔍, got: {}", desc);
    }

    #[test]
    fn test_tool_description_glob() {
        let desc = tool_description("Glob", r#"{"pattern": "*.rs"}"#);
        assert!(desc.starts_with("📁"), "Glob tool should start with 📁, got: {}", desc);
    }

    #[test]
    fn test_tool_description_edit() {
        let desc = tool_description("Edit", r#"{"file_path": "/test.rs"}"#);
        assert!(desc.starts_with("✏️"), "Edit tool should start with ✏️, got: {}", desc);
    }

    #[test]
    fn test_tool_description_write() {
        let desc = tool_description("Write", r#"{"file_path": "/new.rs"}"#);
        assert!(desc.starts_with("📝"), "Write tool should start with 📝, got: {}", desc);
    }

    #[test]
    fn test_tool_description_unknown() {
        let desc = tool_description("SomeRandomTool", r#"{}"#);
        assert!(desc.starts_with("🔧"), "Unknown tool should start with 🔧, got: {}", desc);
    }

    // ============== Result Preview Tests ==============
    #[test]
    fn test_result_preview_single_line() {
        let result = "Error: something went wrong";
        let preview = result_preview(result);
        // Should preserve the full line since it's not a tool_name: format
        assert_eq!(preview, "Error: something went wrong");
    }

    #[test]
    fn test_result_preview_two_lines() {
        let result = "line one\nline two";
        let preview = result_preview(result);
        assert_eq!(preview, "line one line two");
    }

    #[test]
    fn test_result_preview_many_lines() {
        let result = "line one\nline two\nline three\nline four";
        let preview = result_preview(result);
        // Should only take first 2 lines
        assert_eq!(preview, "line one line two");
    }

    #[test]
    fn test_result_preview_with_prefix() {
        let result = "read_file: This is the content";
        let preview = result_preview(result);
        // result_preview doesn't strip tool_name: prefix anymore
        assert_eq!(preview, "read_file: This is the content");
    }

    #[test]
    fn test_result_preview_truncation() {
        let long_content = "a".repeat(100);
        let result = format!("read_file: {}", long_content);
        let preview = result_preview(&result);
        assert!(preview.len() <= 83); // 80 chars + "..."
        assert!(preview.ends_with("..."));
    }

    // ============== Tool Emoji Tests ==============
    #[test]
    fn test_tool_emoji_read_file() {
        assert_eq!(tool_emoji("read_file"), "📖");
    }

    #[test]
    fn test_tool_emoji_list_directory() {
        assert_eq!(tool_emoji("list_directory"), "📁");
    }

    #[test]
    fn test_tool_emoji_search_files() {
        assert_eq!(tool_emoji("search_files"), "🔍");
    }

    #[test]
    fn test_tool_emoji_run_command() {
        assert_eq!(tool_emoji("run_command"), "💻");
    }

    #[test]
    fn test_tool_emoji_unknown() {
        assert_eq!(tool_emoji("unknown_tool"), "⚙");
    }

    // ============== SuggestionEngine Creation Tests ==============
    #[test]
    fn test_suggestion_engine_new_with_key() {
        let engine = SuggestionEngine::new(Some("test-api-key"), Some("claude-3-5-sonnet-4"));
        assert!(engine.is_ok());
    }

    #[test]
    fn test_suggestion_engine_new_default_model() {
        let engine = SuggestionEngine::new(Some("test-api-key"), None);
        assert!(engine.is_ok());
    }

    // ============== Chatter Event Tests ==============
    #[test]
    fn test_chatter_event_generated() {
        use std::sync::{Arc, Mutex};

        let events: Arc<Mutex<Vec<SuggestionEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();

        // Simulate the text processing logic from stream_to_gui
        let in_suggestions = false;
        let text_before_suggestions = "Let me analyze your project.\nLooking at the codebase structure.\n";

        // This simulates what happens in StreamEvent::Text handler
        if !in_suggestions {
            for line in text_before_suggestions.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() && !trimmed.starts_with("```") {
                    events_clone
                        .lock()
                        .unwrap()
                        .push(SuggestionEvent::Chatter(trimmed.to_string()));
                }
            }
        }

        let collected = events.lock().unwrap();
        assert_eq!(collected.len(), 2, "Should have 2 chatter events");
        assert!(matches!(
            &collected[0],
            SuggestionEvent::Chatter(s) if s == "Let me analyze your project."
        ));
        assert!(matches!(
            &collected[1],
            SuggestionEvent::Chatter(s) if s == "Looking at the codebase structure."
        ));
    }

    #[test]
    fn test_chatter_not_generated_after_suggestions_start() {
        use std::sync::{Arc, Mutex};

        let events: Arc<Mutex<Vec<SuggestionEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();

        // Once in_suggestions is true, no chatter should be generated
        let in_suggestions = true;
        let text_after_suggestions = "suggestions:\n- title: Fix bug\n";

        if !in_suggestions {
            for line in text_after_suggestions.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() && !trimmed.starts_with("```") {
                    events_clone
                        .lock()
                        .unwrap()
                        .push(SuggestionEvent::Chatter(trimmed.to_string()));
                }
            }
        }

        let collected = events.lock().unwrap();
        assert_eq!(collected.len(), 0, "No chatter events after suggestions start");
    }

    // ============== FileInfo Tests ==============
    #[test]
    fn test_file_info_directory() {
        let info = FileInfo {
            size: 4096,
            is_dir: true,
        };
        assert!(info.is_dir);
        assert_eq!(info.size, 4096);
    }

    #[test]
    fn test_file_info_file() {
        let info = FileInfo {
            size: 1024,
            is_dir: false,
        };
        assert!(!info.is_dir);
        assert_eq!(info.size, 1024);
    }

    // ============== ProjectContext JSON Tests ==============
    #[test]
    fn test_project_context_to_json() {
        let mut files = HashMap::new();
        files.insert(
            "Cargo.toml".to_string(),
            FileInfo { size: 100, is_dir: false },
        );

        let ctx = ProjectContext {
            root: PathBuf::from("/test"),
            files,
            git_status: None,
            git_branch: None,
            recent_history: vec![],
            project_type: ProjectType::Rust,
        };

        let json = ctx.to_json();
        assert!(json.contains("\"root\""));
        assert!(json.contains("\"files\""));
        assert!(json.contains("\"Rust\""));
    }

    // ============== YAML Parsing State Tests ==============
    #[test]
    fn test_yaml_parser_state_default() {
        let state = YamlParserState::default();
        assert_eq!(state.processed_lines, 0);
        assert!(state.current_card_id.is_none());
        assert_eq!(state.next_card_id, 0);
        assert!(state.emitted_fields.is_empty());
    }

    // ============== SuggestionEvent Variant Tests ==============
    #[test]
    fn test_suggestion_event_variants() {
        // Test ToolCall
        let event = SuggestionEvent::ToolCall("Reading file".to_string());
        match event {
            SuggestionEvent::ToolCall(desc) => {
                assert_eq!(desc, "Reading file");
            }
            _ => panic!("Expected ToolCall variant"),
        }

        // Test Chatter
        let event = SuggestionEvent::Chatter("Thinking...".to_string());
        match event {
            SuggestionEvent::Chatter(text) => {
                assert_eq!(text, "Thinking...");
            }
            _ => panic!("Expected Chatter variant"),
        }

        // Test CardStart
        let event = SuggestionEvent::CardStart { id: 5 };
        match event {
            SuggestionEvent::CardStart { id } => {
                assert_eq!(id, 5);
            }
            _ => panic!("Expected CardStart variant"),
        }

        // Test CardUpdate
        let event = SuggestionEvent::CardUpdate {
            id: 5,
            field: "title".to_string(),
            value: "Fix bug".to_string(),
        };
        match event {
            SuggestionEvent::CardUpdate { id, field, value } => {
                assert_eq!(id, 5);
                assert_eq!(field, "title");
                assert_eq!(value, "Fix bug");
            }
            _ => panic!("Expected CardUpdate variant"),
        }

        // Test CardComplete
        let event = SuggestionEvent::CardComplete { id: 5 };
        match event {
            SuggestionEvent::CardComplete { id } => {
                assert_eq!(id, 5);
            }
            _ => panic!("Expected CardComplete variant"),
        }

        // Test Done
        let event = SuggestionEvent::Done;
        match event {
            SuggestionEvent::Done => {}
            _ => panic!("Expected Done variant"),
        }

        // Test Error
        let event = SuggestionEvent::Error("Test error".to_string());
        match event {
            SuggestionEvent::Error(msg) => {
                assert_eq!(msg, "Test error");
            }
            _ => panic!("Expected Error variant"),
        }
    }
}
