# Palace

**A GPU-accelerated visual interface for AI-assisted software development.**

Palace is a native Rust application that provides a card-based UI for managing projects and orchestrating AI coding tasks. Designed for gamepad-first interaction with full keyboard, mouse, and touch support.

## Features

- **GPU Rendering** - WGPU-based rendering with OLED-optimized black backgrounds
- **Universal Input** - Gamepad (Xbox), keyboard, mouse, and touchscreen
- **Project Discovery** - Automatic detection of projects with language identification
- **Virtual Viewport** - Test any aspect ratio (32:9 ultrawide, 21:9, etc.)
- **Zero CPU Idle** - Event-driven architecture for battery efficiency
- **Card-Based UI** - SDF rounded borders, smooth animations

## Installation

### Requirements

- Rust 1.75+
- GPU with Vulkan, Metal, or DX12 support

### Build

```bash
git clone https://github.com/Zorlin/palace.git
cd palace
cargo build
```

### Run

```bash
# Standard launch
./target/debug/palace

# Test ultrawide aspect ratio
./target/debug/palace --aspect 32:9

# Force specific internal resolution
./target/debug/palace --internal 1920x1080
```

## Usage

### Navigation

| Input | Action |
|-------|--------|
| WASD / Arrows / D-Pad | Navigate |
| Enter / A Button | Select |
| Escape / B Button | Back |
| Tab | Toggle Quest Log |
| Alt+F | Toggle Fullscreen |
| Start | Open Menu |

### Screens

1. **Project Chooser** - Grid of discovered projects
2. **Project View** - Actions for selected project (Start Loop, Build, Run, etc.)
3. **Palace Loop** - AI task cards, approval workflow, execution monitoring
4. **Settings** - UI scale, display preferences

### Screenshots

Take screenshots while Palace is running:

```bash
./target/debug/palace screenshot -o /tmp/screenshot.png
```

## Architecture

```
src/
├── main.rs           # Entry, CLI args, VirtualViewport
├── app.rs            # ApplicationHandler, state machine
├── state.rs          # AppState enum
├── projects.rs       # Project discovery
├── persistence.rs    # ReDB database
└── renderer/
    ├── gpu.rs        # Main renderer
    ├── cards.rs      # Card rendering
    ├── sprites.rs    # Xbox button glyphs
    └── shaders/      # WGSL shaders
```

## Technology

| Component | Library |
|-----------|---------|
| GPU | wgpu 28 |
| Windowing | winit |
| Gamepad | gilrs |
| Text | glyphon/cosmic-text |
| Database | redb |
| Async | tokio |

## Credits

**Xbox Controller Button Icons** by [Arks](https://arks.itch.io/xbox-buttons) - Licensed under CC-BY.

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
