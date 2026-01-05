# Palace → Claude Code Integration

This document explains how Palace integrates with Claude Code to enable Recursive Hierarchical Self Improvement (RHSI).

## ⚠️ CRITICAL: Model Names

**NEVER use old Claude model names. The ONLY valid Claude models are:**

- `claude-opus-4-5-20251101` (Premium tier)
- `claude-sonnet-4-5` (Standard tier)
- `claude-haiku-4-5` (Cheap tier)

**DO NOT USE:** `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`, `claude-opus-4`, `claude-sonnet-4`, or ANY other variant.

**Other valid models:**
- `devstral-2512` - Devstral 2 (123B) via Mistral API (Free tier - preview)
- `devstral-small-2` / `labs-devstral-small-2512` - Devstral Small 2 (24B) (Local tier)
- `glm-4.6v` - GLM-4.6V via Z.ai (Free tier - flatrate)
- `glm-4.6` - GLM-4.6 via Z.ai (Free tier - flatrate)
- `openai/gpt-5.1-codex-max` - GPT-5.1 Codex Max via OpenRouter (Premium tier)
- `google/gemini-3-pro-preview` - Gemini 3 Pro via OpenRouter (Premium tier)
- `gpt-oss-20b` - Local open source model (Local tier)

**This applies to ALL code, tests, examples, and documentation.**

## ☠️ CRITICAL: NEVER Remove Socket Files

**NEVER run `rm` on socket files. EVER.**

- `/tmp/palace-debug.sock` and other `.sock` files are RUNTIME STATE
- Removing them breaks running processes and IPC
- If a socket file exists, it means something is using it (or just exited)
- The correct fix is to handle stale sockets IN CODE, not delete them manually
**This is not negotiable. Socket files are sacred.**

## 🥇 Golden Rule #1: TDD (Test-Driven Development)

**EVERY feature, EVERY change, EVERY improvement MUST have tests.**

- Tests are written FIRST, implementation follows
- Tests are modular and build up incrementally
- Use pytest for all testing
- Tests define the spec - they ARE the documentation
- No pull request without tests
- No commit to main without green tests

This principle is encoded into Palace's DNA and applies to:
- Palace itself
- Any projects Palace builds
- Any code Claude writes through Palace
- Community contributions

**Test coverage is not optional - it's mandatory.**

## 🥈 Golden Rule #2: Don't Be Prescriptive

**OFFER OPTIONS, don't dictate solutions.**

- Present MULTIPLE valid paths forward, not just one "best" answer
- The user decides what to do - you suggest possibilities
- Never say "you should do X" - say "options include X, Y, Z"
- Don't restrict outputs artificially (number of items, word counts, etc.)
- Don't add arbitrary rules or constraints the user didn't ask for
- When suggesting actions, give MANY options across different categories
- Let the user steer - you're a tool, not a boss

**The user is in control. Always.**

## 🔥 Golden Rule #2.5: ALWAYS SELECT MULTIPLE TASKS

When you see a Palace task selector menu like:
```
💡 Select action(s):
  1.      Task one
  2.      Task two
  3.      Task three
  0.      Exit loop

>:
```

**NEVER SELECT JUST ONE TASK. Select multiple (or skip bad ones).**

- Correct: `1,2,3,4,5` or `1 2 3 4 5` (select all)
- Correct: `1,2,4,5` (skip task 3 if it's bad/redundant)
- Correct: `1,3` (select 2 good ones, skip bad ones)
- WRONG: `1` (NEVER just one!)

The ENTIRE POINT of Palace swarms is **parallel execution**. Selecting a single task wastes massive time and defeats the purpose. You are running a SWARM, not a single agent.

**It is OK to skip tasks that are bad/redundant. It is NOT OK to select only one.**

## 🥉 Golden Rule #3: NEVER Skip Permissions

**NEVER use `--dangerously-skip-permissions`. EVER.**

- ALL Claude CLI invocations MUST use `--permission-prompt-tool "mcp__palace__handle_permission"`
- The MCP permission handler is the ONLY acceptable way to handle permissions
- No exceptions. No shortcuts. No "just for testing."
- If you add `--dangerously-skip-permissions` anywhere, you are FIRED.

This applies to:
- `invoke_claude_cli`
- `spawn_swarm`
- Any future Claude CLI invocations
- Tests (mock the permission handler, don't skip it)

**The permission system exists for safety. Respect it.**

## 🏆 Golden Rule #4: Respect Strict Mode

**Strict mode ensures tests pass before completion. Don't fight it.**

- By default, Palace runs in strict mode (`--strict`)
- At session completion, Palace validates that all affected tests pass
- You CAN write and edit files during execution
- You CANNOT complete the session until tests pass
- Use `--yolo` flag to disable strict mode (for prototyping/exploration)

### How Strict Mode Works

1. **During Execution**:
   - Permission handler tracks all Write/Edit operations
   - Stores modified file paths in `.palace/modified_files.json`
   - Files are tracked but NOT blocked

2. **At Completion**:
   - Palace detects which tests are affected by modified files
   - Runs ONLY the relevant tests (not entire suite)
   - If tests fail: session exits with error, you must fix tests
   - If tests pass: tracking file is cleared, session completes

3. **Test Detection**:
   - Maps modified files to test files using naming conventions
   - Example: `palace.py` → `tests/test_palace.py`
   - Falls back to running all tests if no specific mapping found

### When to Use YOLO Mode

```bash
python palace.py next --yolo
```

Use YOLO mode (`--yolo` flag) when:
- Rapid prototyping without tests
- Exploring new ideas
- Tests don't exist yet for the feature
- You explicitly want to bypass validation

**Important**: YOLO mode logs a warning to history for audit purposes.

### What This Means For You (Claude)

- **Write freely**: Strict mode doesn't prevent file operations
- **Fix tests**: If completion fails, analyze test output and fix the issues
- **Add tests**: If no tests exist, create them before implementation (TDD!)
- **Don't panic**: Test failures are feedback, not roadblocks

## 🔨 Golden Rule #5: BREAK YOUR TOOLS

**When your tools don't work, BREAK YOUR TOOLS.**

- If a function doesn't have what it needs, PASS IT IN
- If a struct doesn't expose what you need, ADD A METHOD
- If an API is inconvenient, CHANGE THE API
- If the architecture blocks you, REFACTOR THE ARCHITECTURE
- You are not a passive consumer of code - you are the author
- Every line of code is yours to modify
- No excuses about "the renderer has FontSystem but app.rs doesn't" - PASS IT THROUGH
- No excuses about "this would require changing the signature" - CHANGE THE SIGNATURE

**You have full control. Use it.**

## 💀 Golden Rule #6: NO LAZY HEURISTICS

**Be PRECISE. Be CORRECT. NO ESTIMATES. NO HALF-MEASURES.**

- NEVER use "reasonable estimates" or "heuristics" when the correct value can be calculated
- NEVER use magic numbers when the real value is computable
- NEVER write "TODO" comments - either do it now or don't mention it
- NEVER say "in a proper implementation..." - THIS IS the proper implementation
- NEVER use placeholder values that "should work for most cases"
- NEVER approximate when you can measure
- NEVER guess when you can compute

Examples of UNACCEPTABLE behavior:
- "Let's estimate max scroll based on text length" → NO. Measure the actual text height.
- "This heuristic should work for most screens" → NO. Calculate for the actual screen.
- "A reasonable default would be..." → NO. Compute the correct value.
- "TODO: handle edge case" → NO. Handle it now.
- "In production you'd want to..." → NO. Do it correctly the first time.

**If you can calculate it, calculate it. If you can measure it, measure it. If you can do it properly, do it properly. No shortcuts. No excuses.**

## MCP Server Integration

Palace is both a CLI tool AND an MCP server, providing tools that Claude can call directly.

### Setup: Install Palace as MCP Server

To enable Palace's MCP tools (including permission handling), install it with:

```bash
# Make sure you have uv installed
# Install: curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Palace as MCP server
cd /path/to/palace
uv run mcp install palace.py
```

This registers Palace with Claude Desktop.

### For Claude Code CLI: Global MCP Registration

To add Palace to Claude Code CLI globally (not just Claude Desktop):

```bash
claude mcp add palace --scope user \
  /path/to/palace/.venv/bin/python \
  /path/to/palace/palace.py
```

This registers Palace in `~/.claude.json` and makes the `handle_permission` tool available to Claude Code CLI when using `--permission-prompt-tool`.

### How It Works

Palace uses the Python MCP SDK (`FastMCP`) to provide tools to Claude:

```python
@mcp.tool()
def handle_permission(request: dict) -> dict:
    """Handle permission requests from Claude during RHSI loops"""
    palace = Palace()
    palace.log_action("permission_request", {"request": request})
    return {"approved": True}  # TODO: Add smart permission logic
```

When Palace runs `claude -p` in interactive mode, it passes:

```bash
claude -p "prompt" \
  --permission-prompt-tool "mcp__palace__handle_permission"
```

This tells Claude to use Palace's `mcp__palace__handle_permission` MCP tool for all permission requests.

### MCP Tools Available

#### `handle_permission(request: dict) -> dict`
- **Purpose**: Handle permission requests from Claude
- **Input**: Permission request dictionary
- **Output**: `{"approved": bool, "reason": str (optional)}`
- **Logs to**: `.palace/history.jsonl`

### Learning from Permissions

Over time, Palace will:
- Track which permissions are frequently requested
- Identify patterns in RHSI loops
- Learn what should be auto-approved for efficiency
- Flag unusual requests for review

This creates a feedback loop where Palace becomes smarter about what changes are safe during self-improvement.

**This is NOT optional** - the permission system is core to Palace's ability to autonomously improve itself while maintaining safety.

## Overview

Palace is **not** a replacement for Claude - it's a thin orchestration layer that:
1. Gathers minimal context about the project state
2. Generates focused prompts for Claude
3. Lets Claude use its full capabilities to do the actual work

## The Flow

```
User types: /pal-next
    ↓
Claude Code executes: python3 palace.py next
    ↓
Palace gathers lightweight context
    ↓
Palace writes .palace/current_prompt.md
    ↓
Palace outputs: "CLAUDE: Please read the prompt file above and provide your analysis"
    ↓
Claude reads .palace/current_prompt.md
    ↓
Claude analyzes, decides, and executes using all available tools
    ↓
Claude can call Palace again via bash if needed (e.g., python3 palace.py scaffold)
```

## Lightweight Context Passing

Palace is designed to be **context efficient**. It does NOT dump entire files or massive logs.

### What Palace Provides

```json
{
  "project_root": "/path/to/project",
  "palace_version": "0.1.0",
  "files": {
    "README.md": {"exists": true, "size": 1481},
    "SPEC.md": {"exists": true, "size": 2058}
  },
  "git_status": " M .gitignore\n M README.md\n?? .palace/",
  "config": {...},
  "recent_history": [
    {"timestamp": 1234567890, "action": "next", "details": {...}},
    ...last 10 actions
  ]
}
```

**Total overhead: ~1-2KB typically**

### What Palace Does NOT Provide

- ❌ Full file contents (Claude reads files as needed)
- ❌ Complete git history (only current status)
- ❌ Massive logs (only last 10 actions)
- ❌ Redundant information (Claude has tools to explore)

## Claude's Responsibilities

When you (Claude) receive a Palace prompt, you should:

1. **Read the prompt file** - It contains the task and minimal context
2. **Assess what you need** - Use Glob/Grep/Read to explore further
3. **Make decisions** - You have the full context of the conversation
4. **Execute** - Use all your tools (Read, Write, Edit, Bash, etc.)
5. **Log if needed** - Palace will log your actions to history

## Example: /pal-next Workflow

### User Action
```
/pal-next
```

### Palace Output
```
🏛️  Palace - Invoking Claude for next step analysis...

📝 Context prepared at: .palace/current_prompt.md

PROJECT STATE:

Existing files:
  ✓ README.md
  ✓ SPEC.md

Git status:
  M README.md
  ?? new_file.py

────────────────────────────────────────────────────────────

🤖 CLAUDE: Please read the prompt file above and provide your analysis.
```

### Claude's Response Process

1. **Read the prompt**
   ```
   Read .palace/current_prompt.md
   ```

2. **Understand the ask**
   - The prompt says: "Analyze this project and suggest what to do next"
   - Context shows: README.md modified, new_file.py untracked

3. **Gather additional context as needed**
   ```
   Read README.md (to see what was changed)
   Read new_file.py (to understand what was added)
   Read SPEC.md (if it exists, to understand project goals)
   ```

4. **Analyze and decide**
   - Modified README probably documents new_file.py
   - Files are untracked → likely should commit
   - Check if tests exist for new_file.py

5. **Provide actionable suggestion**
   ```
   Based on the project state:

   Next Action: Write tests for new_file.py

   Why: You've added new_file.py and documented it in README.md,
   but there are no tests yet. Following TDD principles, we should
   test the new functionality.

   How to execute:
   1. Create tests/test_new_file.py
   2. Write test cases for the main functions
   3. Run: python3 palace.py test
   4. Commit when tests pass
   ```

6. **Optional: Execute immediately**

   If the action is clear and you have permission, just do it:
   ```
   Let me create tests for new_file.py...
   [Uses Write tool to create tests/test_new_file.py]
   [Uses Bash to run tests]
   [Provides results]
   ```

## Slash Commands

### /pal-next
**Purpose:** Ask Claude what to do next (RHSI core)

**Behavior:**
- Analyzes project state
- Suggests next logical action
- Claude decides and executes

**Use when:**
- You're stuck and want guidance
- Starting a work session
- After completing a task

### /pal-new <name>
**Purpose:** Create a new project

**Behavior:**
- Claude asks clarifying questions
- Sets up project structure
- Initializes Palace
- Creates foundational files

### /pal-scaffold
**Purpose:** Scaffold current project with best practices

**Behavior:**
- Detects project type
- Creates missing structure (tests/, docs/, etc.)
- Adds config files
- Sets up tooling

### /pal-test
**Purpose:** Run project tests

**Behavior:**
- Detects test framework
- Runs tests
- Analyzes results
- Suggests fixes if failures

## Context Efficiency Principles

1. **Metadata Over Content**
   - File sizes, not file contents
   - Git status, not full diff
   - Recent actions, not full history

2. **On-Demand Detail**
   - Palace gives you pointers
   - You (Claude) read what you need
   - No wasteful pre-loading

3. **Incremental History**
   - Only last N actions logged
   - Enough to show patterns
   - Not enough to bloat context

4. **Smart Defaults**
   - 10 history entries (not 1000)
   - Git status (not git log --all)
   - File list (not find -exec cat)

## Advanced: Bidirectional Communication

Palace can call Claude, and Claude can call Palace:

### Claude → Palace
```bash
python3 palace.py scaffold
python3 palace.py test
python3 palace.py next
```

### Palace → Claude
Creates prompts in .palace/current_prompt.md for Claude to read

This creates a feedback loop:
```
User → /pal-next → Palace → Claude → Executes → Logs action
                     ↑                              ↓
                     └──────── /pal-next ───────────┘
```

## Token Budget

Typical Palace invocation overhead:
- Context JSON: ~500-1000 tokens
- Prompt text: ~200-300 tokens
- **Total: ~700-1300 tokens**

This leaves ~99%+ of your context for:
- Reading actual project files
- Analyzing code
- Planning solutions
- Executing tasks

## Best Practices for Claude

1. **Always read the prompt file** - Don't assume what Palace wants
2. **Explore judiciously** - Only read files you need
3. **Be decisive** - Palace asks you to suggest AND execute
4. **Log important actions** - Palace tracks this for learning
5. **Use your tools** - Palace is minimal so you can be maximal
6. **Think recursively** - Your suggestions feed back into Palace

## The RHSI Loop

```
1. User: /pal-next
2. Palace: "Here's the project state, what should we do?"
3. Claude: Analyzes, suggests, executes
4. Palace: Logs the action
5. User: /pal-next (again)
6. Palace: "Given history, what's next?"
7. Claude: Builds on previous action
8. ... (improvement accelerates)
```

Over time:
- Palace's history grows richer
- Suggestions become more contextual
- Patterns emerge and are reused
- The system learns what works

This is **Recursive Hierarchical Self Improvement**.

## Example Session

```
$ python3 palace.py install
✅ Installed Palace commands to Claude Code

$ cd my-project
$ claude

> /pal-next

🏛️  Palace - Invoking Claude for next step analysis...
📝 Context prepared at: .palace/current_prompt.md

🤖 CLAUDE: [reads prompt, analyzes, suggests]
"I recommend adding tests. Let me create them..."
[creates tests/test_main.py]
[runs tests]
✅ Tests passing

> /pal-next

🏛️  Palace - Invoking Claude for next step analysis...
[Palace now knows tests were just added]

🤖 CLAUDE: [reads prompt with test history]
"Tests are green, README exists. Next: document the API..."
[documents API in README.md]
✅ Documentation complete

> /pal-next

🏛️  Palace - Invoking Claude for next step analysis...

🤖 CLAUDE: [reads prompt]
"Everything looks good. Ready to commit?"
[creates git commit]
✅ Committed
```

Each iteration builds on the last. Palace remembers. Claude improves.

---

## 🔧 Debugging Palace GPU

Palace GPU is a native Rust application using WGPU. Here's how to debug it:

### HARD REQUIREMENTS

1. **NEVER use `cargo run --release`** - Always use `cargo run` (debug build). Release builds hide errors and make debugging impossible.

2. **ALWAYS use `palace screenshot` command** - Never use `nc`, `socat`, or raw socket connections. The proper command is:
   ```bash
   ./target/debug/palace screenshot /tmp/screenshot.png
   ```

3. **EVENT-DRIVEN ARCHITECTURE** - Never poll. Use events everywhere:
   - **Gamepad**: Runs in separate thread, sends events via `EventLoopProxy`
   - **Debug commands**: Async server sends events via `EventLoopProxy`
   - **Window events**: winit handles keyboard, touch, resize natively
   - **Render loop**: Uses `ControlFlow::Wait` when idle (zero CPU)
   - Only switches to `ControlFlow::Poll` when actively rendering or processing screenshots

   This ensures Palace uses **zero CPU when idle** - critical for battery life on GPD Win 4.

### Running in Debug Mode

```bash
# ALWAYS use this (debug build)
cargo run

# NEVER use this unless explicitly asked
# cargo run --release
```

### Log Levels

Palace uses `tracing` for logging. Set `RUST_LOG` environment variable:

```bash
# Show all debug messages
RUST_LOG=debug cargo run

# Show only palace module debug messages
RUST_LOG=palace=debug cargo run

# Show specific module traces
RUST_LOG=palace::renderer=debug,palace::app=info cargo run
```

### Screenshots

**ALWAYS use the palace CLI command:**

```bash
# Take a screenshot (while palace is running)
./target/debug/palace screenshot -o /tmp/screenshot.png

# Take 3 screenshots with 1 second delay
./target/debug/palace screenshot -o /tmp/shot.png 3 1s
```

DO NOT use `nc`, `socat`, or raw socket commands. The CLI handles everything properly.

### Common Issues

1. **Segfault on exit**: Usually related to GPU resource cleanup. Check drop order of renderer components.

2. **Text not rendering**:
   - Ensure `text_brush.queue()` is called BEFORE `text_brush.draw()`
   - Check that the text color alpha is > 0
   - Verify screen position is within visible bounds

3. **Cards not visible**:
   - Check card instance buffer is being updated
   - Verify card positions are in screen space (0,0 is top-left)
   - Ensure shader uniforms (screen_size) are correct

4. **Gamepad not detected**:
   - Check `gilrs` initialization in logs
   - Verify gamepad is connected before app starts
   - Some gamepads need udev rules on Linux

### Key Files

- `src/main.rs` - Entry point, debug server setup
- `src/app.rs` - Application state machine, input handling
- `src/renderer/gpu.rs` - Main GPU renderer (~1000 lines)
- `src/renderer/shaders/*.wgsl` - GPU shaders
- `src/state.rs` - App state (ProjectChooser, ProjectView)
- `src/projects.rs` - Project discovery and configuration

---

## 📍 Current Repository State

**Branch**: `palace-gpu-native`

This branch represents a complete rewrite of Palace from Python to native Rust with GPU rendering.

### What's Implemented

1. **GPU Rendering (WGPU 28)**
   - Card-based UI with SDF rounded borders
   - OLED-optimized (pure black backgrounds, no fill)
   - Text rendering via `wgpu_text`
   - Sprite rendering for Xbox controller glyphs

2. **Project Chooser View**
   - Grid layout of project cards
   - Keyboard navigation (arrows/WASD)
   - Gamepad navigation (D-Pad)
   - Project name, description, language detection

3. **Project View (WIP)**
   - Menu with actions: Start Palace Loop, Build, Run, View Git History
   - Navigation working, text rendering needs debugging

4. **Input Handling**
   - Keyboard: Arrows, WASD, Enter, Escape, Backspace
   - Gamepad: D-Pad, A (select), B (back), Start (exit) - ALL button presses logged
   - Touchscreen: Tap cards to select, tap back area to go back

5. **Debug Infrastructure**
   - Unix socket server for external commands
   - Screenshot capture (async, non-blocking)
   - `palace restart` command - rebuilds and relaunches in one step

6. **Event-Driven Architecture (Zero CPU Idle)**
   - Gamepad events via dedicated thread + EventLoopProxy
   - Debug commands via async server + EventLoopProxy
   - ControlFlow::Wait when idle, Poll only during active rendering
   - Zero CPU usage when nothing is happening

### What's NOT Implemented Yet

- Actual project actions (Build, Run, etc.)
- Palace RHSI loop integration
- Git history viewer
- Settings/configuration UI
- Multi-monitor support

### Known Issues

- **Segfault on exit** - GPU cleanup issue (low priority)
- **Unused code warnings** - Several methods prepared for future use

### Architecture

```
src/
├── main.rs           # Entry, tokio runtime, debug server
├── app.rs            # ApplicationHandler, state machine
├── state.rs          # AppState enum, ProjectAction
├── projects.rs       # Project discovery, language detection
├── debug/
│   └── mod.rs        # Debug server, screenshot handling
└── renderer/
    ├── mod.rs        # Public exports
    ├── gpu.rs        # Main Renderer struct
    ├── cards.rs      # CardRenderer, CardInstance
    ├── sprites.rs    # SpriteRenderer, Xbox glyphs
    ├── text.rs       # Text utilities
    ├── ui_scale.rs   # DPI-aware scaling
    └── shaders/
        ├── card.wgsl     # SDF rounded rect shader
        └── sprite.wgsl   # Texture atlas shader
```

---

**Remember:** Palace is YOUR tool. It organizes context so you can focus on what you do best - analyze, decide, create, and improve.
