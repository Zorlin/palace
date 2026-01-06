# Palace

**A GPU-accelerated visual interface for AI-assisted software development.**

Palace is a native Rust application providing a card-based UI for managing projects and orchestrating AI coding tasks. Designed for gamepad-first interaction with full keyboard, mouse, and touch support.

## Features

- **GPU Rendering** - WGPU-based rendering with OLED-optimized black backgrounds
- **Multi-Monitor** - Simultaneous display on multiple monitors, each with independent windows
- **Universal Input** - Gamepad (Xbox), keyboard, mouse, and touchscreen
- **Project Discovery** - Automatic detection of projects with language identification
- **Virtual Viewport** - Test any aspect ratio (32:9 ultrawide, 21:9, etc.)
- **Zero CPU Idle** - Event-driven architecture for battery efficiency on handhelds
- **Card-Based UI** - SDF rounded borders, smooth animations
- **Composable Workspaces** - Panel-based layout system (Android widget-style)

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
| F2 | Edit Layout (panel mode) |
| L1/R1 | Switch between panels |

### Screens

1. **Project Chooser** - Grid of discovered projects
2. **Project View** - Actions for selected project (Start Loop, Build, Run, etc.)
3. **Palace Loop** - AI task cards, approval workflow, execution monitoring
4. **Settings** - UI scale, display preferences

### Multi-Monitor

Palace supports multiple monitors simultaneously:

1. Open Settings (Escape → Settings)
2. Go to Display Settings
3. Enable monitors you want to use
4. Apply changes

Each monitor runs an independent window. Settings persist across restarts.

### Screenshots

Take screenshots while Palace is running:

```bash
./target/debug/palace screenshot -o /tmp/screenshot.png
```

## Architecture

```
src/
├── main.rs              # Entry, CLI args, VirtualViewport
├── app/                 # Application logic
│   ├── mod.rs           # App struct, initialization
│   ├── events.rs        # winit ApplicationHandler
│   ├── input.rs         # Keyboard input
│   ├── gamepad.rs       # Gamepad thread
│   └── menus.rs         # Menu navigation
├── palace_window.rs     # Per-monitor window
├── state.rs             # AppState enum
├── projects.rs          # Project discovery
├── persistence.rs       # ReDB database
├── ui/                  # Reusable UI components
│   └── menu.rs          # Menu component
├── panels/              # Panel system
│   ├── trait.rs         # Panel trait
│   ├── registry.rs      # Panel management
│   └── layout.rs        # Grid layout
└── renderer/
    ├── gpu.rs           # Main renderer
    ├── shared_gpu.rs    # Shared GPU resources
    ├── cards.rs         # Card rendering
    ├── sprites.rs       # Xbox button glyphs
    └── shaders/         # WGSL shaders
```

### Panel System

Palace uses a composable panel architecture:

| Panel | Description |
|-------|-------------|
| Project Chooser | Grid of all projects |
| Project View | Per-project action menu |
| Quest Log | Card grid for task selection |
| Execution Panel | Live tool/thought display |
| Analysis Panel | Tool/thought logs |

Panels can be rearranged by entering edit mode (F2) and dragging to grid positions.

### Multi-Window Architecture

- **SharedGpuResources** - Device/queue/adapter shared across windows
- **PalaceWindow** - Per-monitor window with own renderer and surface
- **Exclusive Fullscreen** - Each window uses OS-configured resolution
- Display settings persisted via ReDB

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
