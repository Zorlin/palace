# Palace Native GPU

Palace is a native Rust application with GPU-accelerated rendering using WGPU. It provides a visual interface for AI-assisted software development with gamepad, keyboard, touch, and mouse support.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Rust |
| GPU Rendering | WGPU 28 |
| Window Management | winit |
| Gamepad Input | gilrs |
| Text Rendering | glyphon/cosmic-text |
| Persistence | ReDB |
| Async Runtime | tokio |
| CLI Parsing | clap |

## Building and Running

```bash
# Debug build (ALWAYS use for development)
cargo build
./target/debug/palace

# Or run directly
cargo run

# NEVER use release for debugging
# cargo run --release  # NO!
```

## CLI Flags

```bash
# Virtual viewport for testing aspect ratios
./target/debug/palace --aspect 32:9    # Ultrawide simulation
./target/debug/palace --aspect 21:9    # Cinema ultrawide
./target/debug/palace --aspect 16:9    # Standard

# Force internal resolution
./target/debug/palace --internal 1920x1080
./target/debug/palace --internal 2560x1440
```

## Screenshots

```bash
# Take screenshot while Palace is running
./target/debug/palace screenshot -o /tmp/screenshot.png

# Multiple screenshots with delay
./target/debug/palace screenshot -o /tmp/shot.png 3 1s
```

## Debug Logging

```bash
# All debug messages
RUST_LOG=debug cargo run

# Palace module only
RUST_LOG=palace=debug cargo run

# Specific modules
RUST_LOG=palace::renderer=debug,palace::app=info cargo run
```

---

## Golden Rules

### 1. TDD (Test-Driven Development)

Every feature, every change, every improvement MUST have tests.

- Tests are written FIRST, implementation follows
- Tests are modular and build up incrementally
- No pull request without tests
- No commit to main without green tests
- **The user is your LAST RESORT debugger, not your FIRST**
- You can LOSE the user at any time - they have autonomy
- Write tests → validate behavior → THEN have user test (if needed)

### 2. DRY (Don't Repeat Yourself)

If we do a thing in multiple places, make a reusable component.

- As soon as there are TWO versions of something, REFACTOR into a shared component
- Do NOT TODO this - do it IMMEDIATELY when you encounter duplication
- If the component needs variations, make it MORE FLEXIBLE
- Copy-paste is technical debt - pay it off NOW

### 3. Don't Be Prescriptive

OFFER OPTIONS, don't dictate solutions.

- Present MULTIPLE valid paths forward
- The user decides what to do - you suggest possibilities
- Never say "you should do X" - say "options include X, Y, Z"
- Let the user steer - you're a tool, not a boss

### 4. BREAK YOUR TOOLS

When your tools don't work, BREAK YOUR TOOLS.

- If a function doesn't have what it needs, PASS IT IN
- If a struct doesn't expose what you need, ADD A METHOD
- If an API is inconvenient, CHANGE THE API
- If the architecture blocks you, REFACTOR THE ARCHITECTURE
- You are not a passive consumer of code - you are the author

### 5. NO LAZY HEURISTICS

Be PRECISE. Be CORRECT. NO ESTIMATES. NO HALF-MEASURES.

- NEVER use "reasonable estimates" when the correct value can be calculated
- NEVER use magic numbers when the real value is computable
- NEVER approximate when you can measure
- NEVER guess when you can compute

### 6. NEVER Remove Socket Files

NEVER run `rm` on socket files. EVER.

- `/tmp/palace-debug.sock` and other `.sock` files are RUNTIME STATE
- Removing them breaks running processes and IPC
- The correct fix is to handle stale sockets IN CODE

---

## UI Design Rules

### No "Apply" Buttons (except Display Settings)

"Apply" buttons are banned from Palace UI, with ONE exception: Display Settings.

- Most dialogs: Space/Enter toggles or selects, Escape cancels
- Changes take effect immediately
- Display Settings is the ONLY place with an Apply button (because monitor changes are disruptive and need explicit confirmation)

### Focus vs Selection

Focus (what you're navigating to) and Selection (what's chosen) are SEPARATE:

- Arrow keys/Tab move FOCUS only
- Space/Enter ACTIVATES the focused item (toggles checkbox, selects radio, presses button)
- Never change selection just by navigating to something

### Mouse Support

ALL interactive elements must be clickable:

- Buttons, checkboxes, radio buttons, list items
- Click = same as Space/Enter on that element
- Hover states for visual feedback

---

## Architecture

```
src/
├── main.rs           # Entry point, CLI args, VirtualViewport
├── app.rs            # ApplicationHandler, state machine, input
├── state.rs          # AppState enum, all state types
├── projects.rs       # Project discovery, language detection
├── persistence.rs    # ReDB wrapper, preferences
└── renderer/
    ├── mod.rs        # Public exports
    ├── gpu.rs        # Main Renderer (~5000 lines)
    ├── cards.rs      # CardInstance, CardRenderer
    ├── sprites.rs    # SpriteRenderer for Xbox glyphs
    ├── text.rs       # Text measurement utilities
    ├── ui_scale.rs   # DPI-aware scaling
    └── shaders/
        ├── card.wgsl     # SDF rounded rect shader
        └── sprite.wgsl   # Texture atlas shader
```

## AppState Enum

```rust
pub enum AppState {
    ProjectChooser { selected_index, show_archived },
    ProjectView { project_path, selected_action },
    MainMenu { selected_index, previous_state },
    SettingsMenu { selected_index, previous_state },
    UiScaleMenu { selected_index, previous_state },
    PalaceLoop { project_path, cards, focused_index, ... },
    PermissionModal { ... },
    ExecuteModal { ... },
    Survey { ... },
    Executing { ... },
    AddCardMenu { ... },
    CustomTaskInput { ... },
    MultiDisplayDialog { ... },
    ProjectContextMenu { ... },
    LanguageSelector { ... },
}
```

## Input Support

| Input | Actions |
|-------|---------|
| Keyboard | WASD/Arrows navigate, Enter select, Escape back, Tab quest log |
| Gamepad | D-Pad navigate, A select, B back, Start menu, triggers scroll |
| Touch | Tap to select, tap back area to go back |
| Mouse | Click, hover, scroll wheel |

## Event-Driven Architecture

Palace uses zero-CPU-when-idle design:

- **Gamepad**: Separate thread, events via `EventLoopProxy`
- **Debug commands**: Async server, events via `EventLoopProxy`
- **Window events**: winit handles natively
- **Render loop**: `ControlFlow::Wait` when idle

This is critical for battery life on handheld devices like GPD Win 4.

---

## Panel Names

All active work states are branded as "Palace Loop":

| Panel | Description |
|-------|-------------|
| Project Chooser | Grid of all projects (entry point) |
| Project View | Per-project action menu |
| Analysis Panel | Tool/thought logs during analysis |
| Quest Log | Card grid for task selection |
| Execution Panel | Live tool/thought columns during execution |
| Overlays | Permission, Survey, Settings modals |

---

## Common Issues

1. **Text not rendering**
   - Ensure `text_brush.queue()` before `text_brush.draw()`
   - Check text color alpha > 0
   - Verify position within visible bounds

2. **Cards not visible**
   - Check card instance buffer is being updated
   - Verify positions in screen space (0,0 = top-left)
   - Ensure shader uniforms (screen_size) are correct

3. **Gamepad not detected**
   - Check `gilrs` initialization in logs
   - Verify gamepad connected before app starts
   - Linux may need udev rules

4. **Virtual viewport issues**
   - All layout functions must use `content_size()` and `content_offset()`
   - CardGrid needs `.with_offset(offset_x, offset_y)`
   - Text positions need offset added

---

## Model Names

Valid Claude models:

- `claude-opus-4-5-20251101` (Premium)
- `claude-sonnet-4-5` (Standard)
- `claude-haiku-4-5` (Cheap)

DO NOT USE old names like `claude-3-opus`, `claude-3-sonnet`, etc.
