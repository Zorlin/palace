# Palace

Gamepad-native AI coding assistant. You select tasks, Claude executes them.

## Priority: GAMEPAD

This is a **gamepad-first** application for handheld gaming PCs (GPD Win 4, Steam Deck).

All task suggestions should prioritize gamepad UX:
- Controller navigation
- Button mappings (A/B/X/Y)
- D-pad controls
- Trigger combos (RT+LT for intervention)
- Haptic feedback
- BG3/Skyrim-style dialogue selection

Keyboard is fallback, not primary.

## How It Works

1. Palace analyzes your codebase
2. Claude suggests actionable tasks (things IT can do)
3. You navigate with gamepad/keyboard, select tasks
4. Press Execute - Claude does the work while you watch

**You are the director, not the typist.** Pick what matters, let AI handle implementation.

## Controls

| Action | Keyboard | Gamepad |
|--------|----------|---------|
| Navigate | Arrow keys / j,k | D-pad |
| Toggle select | Space | A |
| Select all | A | Y |
| Execute selected | Enter | Start |
| Expand details | E | - |
| History view | H | - |
| Quit | q / Ctrl+C | B (hold) |

## Task Types

Good tasks (Claude executes these):
- "Add error handling to payment module"
- "Write tests for auth flow"
- "Refactor database queries for performance"
- "Fix the race condition in cache invalidation"

Bad tasks (require human keyboard input):
- "Consider using a different database"
- "Review the architecture decisions"
- "Think about error handling strategy"

## Architecture

```
Palace (Rust TUI)
    ↓ spawns
palace-sdk (Go binary)
    ↓ calls
Anthropic API (streaming)
    ↓ returns
Streamed text output
```

## Building

```bash
# Build Go SDK wrapper
cd go && go build -o ../target/palace-sdk .

# Build Palace
cargo build --release
```

## Configuration

`~/.config/palace/config.toml`:

```toml
[api]
base_url = "https://api.anthropic.com"
model = "claude-sonnet-4-5"

[gamepad]
enabled = true
deadzone = 0.15
```

Environment: `ANTHROPIC_API_KEY`
