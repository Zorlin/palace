//! Palace application core
//!
//! This module contains the main App struct and event handling for Palace.
//! Functionality is split across submodules for maintainability:
//!
//! - `gamepad`: Gamepad input and focus management
//! - `touch`: Touch, mouse, and cursor input
//! - `menus`: Menu state navigation
//! - `debug`: Debug commands and state serialization
//! - `ai`: AI suggestion execution
//! - `input`: Keyboard input handling
//! - `events`: winit ApplicationHandler implementation

mod ai;
mod debug;
mod events;
mod gamepad;
mod input;
mod menus;
mod touch;

// Re-export traits so they're available when using App
pub use gamepad::GamepadFocusTracker;

use crate::debug::{DebugCommand, DebugResponse, ScreenshotCapture};
use crate::palace_window::PalaceWindow;
use crate::persistence::PalaceDB;
use crate::projects::ProjectsConfig;
use crate::renderer::{Renderer, SharedGpuResources};
use crate::state::{AppState, TaskStatus};

use gilrs::Button;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use winit::event::Modifiers;
use winit::event_loop::{ActiveEventLoop, EventLoopProxy};
use winit::window::WindowId;

/// User preference keys
const PREF_UI_SCALE: &str = "ui_scale";
const PREF_MULTI_DISPLAY: &str = "multi_display";
const PREF_DISPLAY_SETTINGS: &str = "display_settings";

/// Custom events sent to the main event loop from background threads
#[derive(Debug)]
#[allow(dead_code)]
pub enum AppEvent {
    /// Gamepad button pressed
    GamepadButton(Button),
    /// Gamepad button released
    GamepadButtonReleased(Button),
    /// Gamepad left stick moved (axis, value -1.0 to 1.0)
    GamepadStick { x: f32, y: f32 },
    /// Gamepad right stick moved (for scrolling)
    #[allow(dead_code)]
    GamepadRightStick { x: f32, y: f32 },
    /// Gamepad connected
    GamepadConnected,
    /// Gamepad disconnected
    GamepadDisconnected,
    /// Debug command received
    DebugCommand(DebugCommand, mpsc::Sender<DebugResponse>),
    /// Get current state (for restart)
    GetState(tokio::sync::oneshot::Sender<String>),
    /// Shutdown the application
    Shutdown,
    /// AI is exploring with a tool
    AiToolCall(String),
    /// AI thinking/commentary (shows in waterfall)
    AiChatter(String),
    /// New suggestion card started
    SuggestionStart { id: usize },
    /// Suggestion field updated (title, category, description, command)
    SuggestionUpdate { id: usize, field: String, value: String },
    /// Suggestion card complete
    SuggestionComplete { id: usize },
    /// All suggestions done
    SuggestionsDone,
    /// AI error
    AiError(String),
    /// Execution tool call (left column) - format: "[HH:MM:SS] icon action"
    ExecutionToolCall(String),
    /// Execution thought/commentary (right column)
    ExecutionThought(String),
    /// Execution progress update
    ExecutionProgress { current: usize, total: usize },
    /// Execution completed successfully
    ExecutionComplete,
    /// Execution failed
    ExecutionError(String),
    /// Execution token usage update (cumulative)
    ExecutionTokens(u64),
    /// Execution request started (set request_active = true)
    ExecutionRequestStart,
    /// Execution request ended (set request_active = false)
    ExecutionRequestEnd,
    /// Survey request from AI (ask_user tool) - show survey UI and return response
    SurveyRequest {
        question: String,
        header: String,
        options: Vec<crate::state::SurveyOption>,
        multi_select: bool,
        response_tx: std::sync::mpsc::Sender<crate::state::SurveyResponse>,
    },
    /// Permission request from AI - requires user approval
    PermissionRequest {
        /// Unique ID for this request
        #[allow(dead_code)]
        id: usize,
        /// The command or action requesting permission
        command: String,
        /// Channel to send response back
        response_tx: tokio::sync::oneshot::Sender<PermissionResponse>,
    },
    /// Task status update from AI (task_update tool)
    TaskStatusUpdate {
        task_index: usize,
        status: TaskStatus,
        message: Option<String>,
    },
    /// Switch to a different display/monitor
    SwitchDisplay {
        /// Monitor name to switch to
        monitor_name: String,
    },
    /// Create a new window on a specific monitor
    CreateWindow {
        /// Monitor name to create window on
        monitor_name: String,
    },
    /// Extend to all available monitors (create windows on each)
    ExtendToAllMonitors,
}

// Re-export PermissionResponse from state for convenience
pub use crate::state::PermissionResponse;

/// The main Palace application
pub struct App {
    /// Shared GPU resources (device, queue) - created once, shared by all windows
    pub(crate) shared_gpu: Option<SharedGpuResources>,
    /// All Palace windows, keyed by WindowId
    pub(crate) windows: HashMap<WindowId, PalaceWindow>,
    /// Currently focused window (receives keyboard/gamepad input)
    pub(crate) focused_window: Option<WindowId>,
    /// Current application state
    pub(crate) state: AppState,
    /// Project configuration
    pub(crate) projects: ProjectsConfig,
    /// Screenshot capture handler
    pub(crate) screenshot_capture: ScreenshotCapture,
    /// Pending screenshot path
    pub(crate) pending_screenshot: Option<PathBuf>,
    /// Flag to track if we need to redraw (idle optimization)
    pub(crate) needs_redraw: bool,
    /// Whether a gamepad is connected (updated via events)
    pub(crate) gamepad_connected: bool,
    /// Exit requested via menu
    pub(crate) exit_requested: bool,
    /// Dark mode enabled (true = dark, false = light)
    pub(crate) dark_mode: bool,
    /// Gamepad passthrough mode (true = Palace ignores gamepad, other apps can use it)
    pub(crate) gamepad_passthrough: bool,
    /// Track held buttons for combos (L3, R3)
    pub(crate) l3_held: bool,
    pub(crate) r3_held: bool,
    /// Right stick Y position for continuous scrolling
    pub(crate) right_stick_y: f32,
    /// Event loop proxy for sending events from background threads
    pub(crate) event_proxy: Arc<EventLoopProxy<AppEvent>>,
    /// Pending permission response sender (set when permission modal is shown)
    pub(crate) permission_response_tx: Option<tokio::sync::oneshot::Sender<PermissionResponse>>,
    /// System-detected UI scale (from display settings)
    pub(crate) detected_scale: f32,
    /// User's scale override (None = use auto-detected)
    pub(crate) user_scale_override: Option<f32>,
    /// Flag to track if hover state changed (to minimize redraws)
    pub(crate) hover_changed: bool,
    /// Touch hold tracking: (start_time, start_x, start_y, card_index)
    pub(crate) touch_hold: Option<(std::time::Instant, f32, f32, Option<usize>)>,
    /// Whether touch hold has triggered (to prevent tap on release)
    pub(crate) touch_hold_triggered: bool,
    /// Current keyboard modifiers (for Alt+F, Ctrl+, etc.)
    pub(crate) modifiers: Modifiers,
    /// Whether window is currently fullscreen
    pub(crate) is_fullscreen: bool,
    /// Whether window is currently focused (for focus-gated inputs)
    pub(crate) window_focused: bool,
    /// Database for persistent storage (preferences, tasks, etc.)
    pub(crate) db: Option<PalaceDB>,
    /// Known monitor IDs (to detect new monitors)
    pub(crate) known_monitors: Vec<String>,
    /// Virtual viewport settings for testing different aspect ratios
    pub(crate) virtual_viewport: crate::VirtualViewport,
    /// Pending primary monitor from saved settings (set in restore_display_settings)
    /// When CreateWindow creates a window on this monitor, it becomes the focused window
    pub(crate) pending_primary_monitor: Option<String>,
}

impl App {
    pub fn new(
        initial_state: AppState,
        projects: ProjectsConfig,
        event_proxy: EventLoopProxy<AppEvent>,
        virtual_viewport: crate::VirtualViewport,
    ) -> Self {
        Self {
            permission_response_tx: None,
            shared_gpu: None,
            windows: HashMap::new(),
            focused_window: None,
            state: initial_state,
            projects,
            screenshot_capture: ScreenshotCapture::new(),
            pending_screenshot: None,
            needs_redraw: true,
            gamepad_connected: false,
            exit_requested: false,
            dark_mode: true,
            gamepad_passthrough: false,
            l3_held: false,
            r3_held: false,
            right_stick_y: 0.0,
            event_proxy: Arc::new(event_proxy),
            detected_scale: 1.0,
            user_scale_override: None,
            hover_changed: false,
            touch_hold: None,
            touch_hold_triggered: false,
            modifiers: Modifiers::default(),
            is_fullscreen: true,
            window_focused: true,
            db: PalaceDB::open().ok(),
            known_monitors: Vec::new(),
            virtual_viewport,
            pending_primary_monitor: None,
        }
    }

    /// Load saved preferences from database
    pub(crate) fn load_preferences(&mut self) {
        if let Some(ref db) = self.db {
            // Load UI scale preference
            if let Ok(Some(scale)) = db.get_pref::<Option<f32>>(PREF_UI_SCALE) {
                self.user_scale_override = scale;
                tracing::info!("Loaded UI scale preference: {:?}", scale);
            }
        }
    }

    /// Save UI scale preference to database
    pub(crate) fn save_scale_preference(&self) {
        if let Some(ref db) = self.db {
            if let Err(e) = db.set_pref(PREF_UI_SCALE, &self.user_scale_override) {
                tracing::warn!("Failed to save UI scale preference: {}", e);
            }
        }
    }

    /// Request redraw on all windows
    pub(crate) fn request_redraw(&mut self) {
        self.needs_redraw = true;
        for palace_window in self.windows.values_mut() {
            palace_window.request_redraw();
        }
    }

    /// Request redraw on a specific window
    #[allow(dead_code)]
    pub(crate) fn request_redraw_window(&mut self, window_id: WindowId) {
        self.needs_redraw = true;
        if let Some(palace_window) = self.windows.get_mut(&window_id) {
            palace_window.request_redraw();
        }
    }

    /// Get renderer from focused window (immutable)
    pub(crate) fn focused_renderer(&self) -> Option<&Renderer> {
        self.focused_window
            .and_then(|id| self.windows.get(&id))
            .map(|w| &w.renderer)
    }

    /// Get renderer from focused window (mutable)
    pub(crate) fn focused_renderer_mut(&mut self) -> Option<&mut Renderer> {
        self.focused_window
            .and_then(|id| self.windows.get_mut(&id))
            .map(|w| &mut w.renderer)
    }

    /// Calculate number of columns for PalaceLoop card grid
    pub(crate) fn palace_loop_columns(&self) -> usize {
        if let Some(renderer) = self.focused_renderer() {
            let ui_scale = renderer.ui_scale();
            let scale = |v: f32| v * ui_scale;

            let base_card_width = 237.0;
            let base_gap = 16.0;
            let base_margin = 40.0;
            let screen_width = renderer.size().width as f32;

            let card_width = scale(base_card_width);
            let gap = scale(base_gap);
            let margin_x = scale(base_margin);

            let available_width = screen_width - margin_x * 2.0;
            let columns = ((available_width + gap) / (card_width + gap)).floor() as usize;
            columns.max(1)
        } else {
            5
        }
    }

    /// Get card height for PalaceLoop
    pub(crate) fn palace_loop_card_height(&self) -> f32 {
        let ui_scale = self.focused_renderer().map(|r| r.ui_scale()).unwrap_or(1.5);
        let base_card_width = 237.0;
        let card_width = base_card_width * ui_scale;
        card_width * 0.6 // Same aspect ratio as CardGrid
    }

    /// Get row height for PalaceLoop (card + gap)
    pub(crate) fn palace_loop_row_height(&self) -> f32 {
        let ui_scale = self.focused_renderer().map(|r| r.ui_scale()).unwrap_or(1.5);
        let base_gap = 16.0;
        self.palace_loop_card_height() + base_gap * ui_scale
    }

    /// Restore saved display settings - creates windows on all saved enabled monitors
    pub(crate) fn restore_display_settings(&mut self) {
        use crate::state::DisplaySettings;

        let saved: DisplaySettings = self.db
            .as_ref()
            .and_then(|db| db.get_pref::<DisplaySettings>(PREF_DISPLAY_SETTINGS).ok())
            .flatten()
            .unwrap_or_default();

        if saved.enabled.is_empty() {
            tracing::info!("No saved display settings to restore");
            return;
        }

        tracing::info!("Restoring display settings: enabled={:?}, primary={:?}",
            saved.enabled, saved.primary);

        // Store the saved primary so CreateWindow can set focus correctly
        self.pending_primary_monitor = saved.primary.clone();

        // Create windows on all saved enabled monitors (except ones we already have)
        for monitor_name in &saved.enabled {
            let already_have_window = self.windows.values()
                .any(|w| &w.monitor_name == monitor_name);

            if !already_have_window && self.known_monitors.contains(monitor_name) {
                tracing::info!("Restoring window on saved monitor: {}", monitor_name);
                let _ = self.event_proxy.send_event(AppEvent::CreateWindow {
                    monitor_name: monitor_name.clone(),
                });
            }
        }

        // If the saved primary already has a window, switch to it now
        if let Some(ref primary_name) = saved.primary {
            let primary_window_id = self.windows.iter()
                .find(|(_, w)| &w.monitor_name == primary_name)
                .map(|(id, _)| *id);

            if let Some(window_id) = primary_window_id {
                tracing::info!("Focusing existing window on saved primary: {}", primary_name);
                self.focused_window = Some(window_id);
                self.pending_primary_monitor = None;
            }
        }
    }

    /// Check for new monitors and show dialog if needed
    pub(crate) fn check_for_new_monitors(&mut self, event_loop: &ActiveEventLoop) {
        use crate::state::DisplayOption;

        // Get current monitors
        let current_monitors: Vec<_> = event_loop.available_monitors().collect();
        let current_names: Vec<String> = current_monitors
            .iter()
            .filter_map(|m| m.name())
            .collect();

        // Check if there are any new monitors
        let new_monitors: Vec<_> = current_names
            .iter()
            .filter(|name| !self.known_monitors.contains(name))
            .collect();

        if new_monitors.is_empty() {
            return;
        }

        tracing::info!("New monitor(s) detected: {:?}", new_monitors);

        // Load saved display settings
        use crate::state::DisplaySettings;
        let saved: DisplaySettings = self.db
            .as_ref()
            .and_then(|db| db.get_pref::<DisplaySettings>(PREF_DISPLAY_SETTINGS).ok())
            .flatten()
            .unwrap_or_default();

        // Check if we should auto-apply saved preferences
        let has_saved_prefs = !saved.enabled.is_empty();
        if has_saved_prefs {
            // First, switch to saved primary if different from current
            if let Some(ref primary_name) = saved.primary {
                let current_primary = self.focused_window
                    .and_then(|id| self.windows.get(&id))
                    .map(|w| &w.monitor_name);

                if current_primary != Some(primary_name) && current_names.contains(primary_name) {
                    tracing::info!("Switching to saved primary monitor: {}", primary_name);
                    let _ = self.event_proxy.send_event(AppEvent::SwitchDisplay {
                        monitor_name: primary_name.clone(),
                    });
                }
            }

            // Then, auto-create windows on saved enabled monitors that we don't have yet
            for monitor_name in &saved.enabled {
                // Skip if we already have a window on this monitor
                let already_have_window = self.windows.values()
                    .any(|w| &w.monitor_name == monitor_name);

                if !already_have_window && current_names.contains(monitor_name) {
                    tracing::info!("Auto-creating window on saved monitor: {}", monitor_name);
                    let _ = self.event_proxy.send_event(AppEvent::CreateWindow {
                        monitor_name: monitor_name.clone(),
                    });
                }
            }

            // Update known monitors and return - don't show dialog
            self.known_monitors = current_names;
            return;
        }

        // Only show dialog if we have more than one monitor
        if current_monitors.len() <= 1 {
            self.known_monitors = current_names;
            return;
        }

        // Don't show dialog if already showing one
        if matches!(self.state, AppState::MultiDisplayDialog { .. }) {
            return;
        }

        // Fallback primary to focused window if not saved
        let current_monitor_name: Option<String> = saved.primary.or_else(|| {
            self.focused_window
                .and_then(|id| self.windows.get(&id))
                .and_then(|w| w.window.current_monitor())
                .and_then(|m| m.name())
        });

        // Fallback enabled to current windows if not saved
        let enabled_monitors: Vec<String> = if saved.enabled.is_empty() {
            self.windows.values()
                .filter_map(|w| Some(w.monitor_name.clone()))
                .collect()
        } else {
            saved.enabled
        };

        let options: Vec<DisplayOption> = current_monitors
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let raw_name = m.name().unwrap_or_else(|| format!("Monitor {}", i + 1));
                let size = m.size();
                let pos = m.position();
                // Make name more identifiable
                let name = if pos.x == 0 && pos.y == 0 {
                    format!("{} (Main)", raw_name)
                } else if pos.x < 0 {
                    format!("{} (Left)", raw_name)
                } else if pos.x > 0 {
                    format!("{} (Right)", raw_name)
                } else {
                    format!("{} @ ({},{})", raw_name, pos.x, pos.y)
                };
                let mut opt = DisplayOption::new(name, raw_name.clone(), size.width, size.height);
                opt.is_primary = current_monitor_name.as_ref() == Some(&raw_name);
                opt.enabled = enabled_monitors.contains(&raw_name) || opt.is_primary;
                opt
            })
            .collect();

        // Show dialog - each monitor has enable toggle, Primary pill is draggable
        let previous = std::mem::replace(&mut self.state, AppState::project_chooser());
        self.state = AppState::MultiDisplayDialog {
            focus_index: 0,
            options,
            remember_choice: false,
            focused_row: 0,
            primary_pill_drag: None,
            previous_state: Box::new(previous),
        };

        self.known_monitors = current_names;
        self.request_redraw();
    }

    /// Find the target monitor from config or return None for default
    pub(crate) fn find_target_monitor(
        &self,
        event_loop: &ActiveEventLoop,
    ) -> Option<winit::monitor::MonitorHandle> {
        // Try to load display config
        let config_path = dirs::config_dir()
            .map(|p| p.join("palace/display.json"))
            .unwrap_or_else(|| PathBuf::from("/tmp/palace/display.json"));

        if config_path.exists() {
            if let Ok(json) = std::fs::read_to_string(&config_path) {
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&json) {
                    if let Some(display_name) = config.get("display").and_then(|v| v.as_str()) {
                        // Search for matching monitor by name
                        for monitor in event_loop.available_monitors() {
                            if let Some(name) = monitor.name() {
                                if name == display_name {
                                    return Some(monitor);
                                }
                            }
                        }
                        tracing::warn!(
                            "Configured display '{}' not found, using default",
                            display_name
                        );
                    }
                }
            }
        }
        None // Use default (primary) monitor
    }

    /// Add a command prefix to the approved list
    pub(crate) fn approve_command_prefix(prefix: &str) {
        use std::sync::{Mutex as StdMutex, OnceLock};
        static APPROVED: OnceLock<StdMutex<Vec<String>>> = OnceLock::new();
        let approved = APPROVED.get_or_init(|| StdMutex::new(Vec::new()));
        let mut list = approved.lock().unwrap();
        if !list.contains(&prefix.to_string()) {
            list.push(prefix.to_string());
            tracing::info!("Approved command prefix: {}", prefix);
        }
    }

    /// Check if a command is approved (prefix match)
    #[allow(dead_code)]
    pub fn is_command_approved(cmd: &str) -> bool {
        use std::sync::{Mutex as StdMutex, OnceLock};
        static APPROVED: OnceLock<StdMutex<Vec<String>>> = OnceLock::new();
        let approved = APPROVED.get_or_init(|| StdMutex::new(Vec::new()));
        let list = approved.lock().unwrap();
        list.iter().any(|a| cmd.starts_with(a))
    }
}
