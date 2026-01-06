//! Panel system for composable workspaces
//!
//! Panels are UI primitives that can be arranged in a grid layout.
//! Screens are saved arrangements of panels.

mod bounds;
mod edit_mode;
mod layout;
mod registry;

pub use bounds::{GridCell, GridPosition, PanelBounds};
pub use edit_mode::{available_panel_types, EditModeState, PanelTypeInfo, ResizeHandle, UIPanel, UIPanelBounds, UIPanelDragState, UIPanelResizeState, LONG_PRESS_THRESHOLD};
pub use layout::{LayoutPreset, PanelLayout, PushDirection, ReorderSolution};
pub use registry::{PanelId, PanelRegistry};

use crate::renderer::Renderer;
use winit::keyboard::KeyCode;

/// Input events routed to panels
#[derive(Debug, Clone)]
pub enum PanelInput {
    /// Keyboard key press
    Key(KeyCode),
    /// Gamepad button (using gilrs button index)
    GamepadButton(u32),
    /// Touch/click at normalized position (0.0-1.0 within panel bounds)
    Touch { x: f32, y: f32 },
    /// Scroll delta
    Scroll { delta: f32 },
}

/// Navigation targets for panel actions
#[derive(Debug, Clone)]
pub enum NavigationTarget {
    /// Go to project chooser
    ProjectChooser,
    /// Go to project view for specific project
    ProjectView { project_path: std::path::PathBuf },
    /// Start Palace Loop for project
    PalaceLoop { project_path: std::path::PathBuf },
}

/// Actions panels can request
#[derive(Debug, Clone)]
pub enum PanelAction {
    /// Nothing happened
    None,
    /// Request this panel to be focused
    RequestFocus,
    /// Close this panel
    Close,
    /// Spawn a new panel of given type
    SpawnPanel {
        panel_type: String,
        bounds: Option<GridPosition>,
    },
    /// Navigate to a different screen/state
    Navigate { to: NavigationTarget },
    /// Show an overlay (modal)
    ShowOverlay { overlay_type: String },
}

/// Shared state that panels can read
///
/// Uses Arc for efficient sharing across panels.
/// State is updated by the App and read by panels during rendering.
pub struct SharedPanelState {
    /// All discovered projects
    pub projects: std::sync::Arc<crate::projects::ProjectsConfig>,
    /// Currently focused project (if any)
    pub focused_project: Option<std::path::PathBuf>,
}

impl Default for SharedPanelState {
    fn default() -> Self {
        Self {
            projects: std::sync::Arc::new(crate::projects::ProjectsConfig::default()),
            focused_project: None,
        }
    }
}

/// Panel trait - each panel type implements this
///
/// Panels are the fundamental building blocks of the Palace UI.
/// They can be arranged in a grid layout and customized by the user.
pub trait Panel: Send + Sync {
    /// Unique type identifier (e.g., "quest_log", "execution")
    fn panel_type(&self) -> &'static str;

    /// Human-readable name for panel chooser
    fn display_name(&self) -> &'static str;

    /// Minimum grid cells this panel needs (width, height)
    fn min_size(&self) -> (u32, u32);

    /// Render this panel within bounds
    fn render(
        &self,
        renderer: &mut Renderer,
        bounds: &PanelBounds,
        state: &SharedPanelState,
    );

    /// Handle input when this panel is focused
    fn handle_input(&mut self, input: PanelInput) -> PanelAction;

    /// Can this panel be closed? (Some panels like main view cannot)
    fn closeable(&self) -> bool {
        true
    }

    /// Can this panel be resized smaller than min_size?
    fn resizable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test panel for unit tests
    struct TestPanel {
        panel_type: &'static str,
        min_size: (u32, u32),
    }

    impl Panel for TestPanel {
        fn panel_type(&self) -> &'static str {
            self.panel_type
        }

        fn display_name(&self) -> &'static str {
            "Test Panel"
        }

        fn min_size(&self) -> (u32, u32) {
            self.min_size
        }

        fn render(&self, _renderer: &mut Renderer, _bounds: &PanelBounds, _state: &SharedPanelState) {
            // No-op for testing
        }

        fn handle_input(&mut self, _input: PanelInput) -> PanelAction {
            PanelAction::None
        }
    }

    #[test]
    fn test_panel_trait_basics() {
        let panel = TestPanel {
            panel_type: "test",
            min_size: (2, 2),
        };

        assert_eq!(panel.panel_type(), "test");
        assert_eq!(panel.min_size(), (2, 2));
        assert!(panel.closeable());
        assert!(panel.resizable());
    }

    #[test]
    fn test_grid_cell() {
        let cell = GridCell { col: 3, row: 5 };
        assert_eq!(cell.col, 3);
        assert_eq!(cell.row, 5);
    }

    #[test]
    fn test_grid_position() {
        let pos = GridPosition {
            start: GridCell { col: 0, row: 0 },
            width: 4,
            height: 3,
        };

        assert_eq!(pos.width, 4);
        assert_eq!(pos.height, 3);
        assert_eq!(pos.cells_covered(), 12);
    }

    #[test]
    fn test_panel_bounds_calculation() {
        // 8x6 grid on 1920x1080 screen
        let bounds = PanelBounds::from_grid(
            &GridPosition {
                start: GridCell { col: 0, row: 0 },
                width: 4,
                height: 3,
            },
            8,
            6,
            1920.0,
            1080.0,
            10.0, // gap
            20.0, // margin
        );

        // Panel should be in top-left quadrant
        assert_eq!(bounds.x, 20.0); // margin
        assert_eq!(bounds.y, 20.0); // margin

        // Width should be roughly half screen minus margins and gaps
        assert!(bounds.width > 800.0);
        assert!(bounds.width < 1000.0);
    }
}
