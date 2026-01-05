use crate::debug::{DebugCommand, DebugResponse, ScreenshotCapture};
use crate::display::DisplayScaling;
use crate::projects::ProjectsConfig;
use crate::renderer::Renderer;
use crate::state::{AppState, ExecuteOption, ExecutionStatus, MainMenuItem, SettingsItem, SuggestionCard, UiScaleOption};
use gilrs::Button;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, Touch, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

/// Custom events sent to the main event loop from background threads
#[derive(Debug)]
pub enum AppEvent {
    /// Gamepad button pressed
    GamepadButton(Button),
    /// Gamepad button released
    GamepadButtonReleased(Button),
    /// Gamepad left stick moved (axis, value -1.0 to 1.0)
    GamepadStick { x: f32, y: f32 },
    /// Gamepad right stick moved (for scrolling)
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
        id: usize,
        /// The command or action requesting permission
        command: String,
        /// Channel to send response back
        response_tx: tokio::sync::oneshot::Sender<PermissionResponse>,
    },
}

// Re-export PermissionResponse from state for convenience
pub use crate::state::PermissionResponse;

pub struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    state: AppState,
    projects: ProjectsConfig,
    screenshot_capture: ScreenshotCapture,
    pending_screenshot: Option<PathBuf>,
    /// Flag to track if we need to redraw (idle optimization)
    needs_redraw: bool,
    /// Whether a gamepad is connected (updated via events)
    gamepad_connected: bool,
    /// Exit requested via menu
    exit_requested: bool,
    /// Dark mode enabled (true = dark, false = light)
    dark_mode: bool,
    /// Gamepad passthrough mode (true = Palace ignores gamepad, other apps can use it)
    gamepad_passthrough: bool,
    /// Track held buttons for combos (L3, R3)
    l3_held: bool,
    r3_held: bool,
    /// Right stick Y position for continuous scrolling
    right_stick_y: f32,
    /// Event loop proxy for sending events from background threads
    event_proxy: Arc<EventLoopProxy<AppEvent>>,
    /// Pending permission response sender (set when permission modal is shown)
    permission_response_tx: Option<tokio::sync::oneshot::Sender<PermissionResponse>>,
    /// System-detected UI scale (from display settings)
    detected_scale: f32,
    /// User's scale override (None = use auto-detected)
    user_scale_override: Option<f32>,
    /// Flag to track if hover state changed (to minimize redraws)
    hover_changed: bool,
    /// Touch hold tracking: (start_time, start_x, start_y, card_index)
    touch_hold: Option<(std::time::Instant, f32, f32, Option<usize>)>,
    /// Whether touch hold has triggered (to prevent tap on release)
    touch_hold_triggered: bool,
}

impl App {
    pub fn new(initial_state: AppState, projects: ProjectsConfig, event_proxy: EventLoopProxy<AppEvent>) -> Self {
        Self {
            permission_response_tx: None,
            window: None,
            renderer: None,
            state: initial_state,
            projects,
            screenshot_capture: ScreenshotCapture::new(),
            pending_screenshot: None,
            needs_redraw: true, // Initial render needed
            gamepad_connected: false,
            exit_requested: false,
            dark_mode: true, // Default to dark mode (OLED-friendly)
            gamepad_passthrough: false,
            l3_held: false,
            r3_held: false,
            right_stick_y: 0.0,
            event_proxy: Arc::new(event_proxy),
            detected_scale: 1.0, // Will be set when display is detected
            user_scale_override: None, // Auto-detect by default
            hover_changed: false,
            touch_hold: None,
            touch_hold_triggered: false,
        }
    }

    /// Request a redraw on the next frame
    fn request_redraw(&mut self) {
        self.needs_redraw = true;
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    /// Open the main menu (Start menu)
    fn open_main_menu(&mut self) {
        // Don't open menu if we're already in one
        match &self.state {
            AppState::MainMenu { .. } | AppState::SettingsMenu { .. } | AppState::UiScaleMenu { .. } => {
                // Already in a menu, go back instead
                self.go_back_from_menu();
            }
            _ => {
                tracing::info!("Opening main menu");
                self.state = AppState::MainMenu {
                    selected_item: 0,
                    previous_state: Box::new(self.state.clone()),
                };
            }
        }
    }

    /// Go back from current menu state
    fn go_back_from_menu(&mut self) {
        match &self.state {
            AppState::MainMenu { previous_state, .. } => {
                self.state = *previous_state.clone();
            }
            AppState::SettingsMenu { previous_state, .. } => {
                self.state = *previous_state.clone();
            }
            AppState::UiScaleMenu { previous_state, .. } => {
                self.state = *previous_state.clone();
            }
            AppState::ExecuteModal { previous_state, .. } => {
                self.state = *previous_state.clone();
            }
            _ => {}
        }
    }

    /// Get the index of the current UI scale in UiScaleOption::all()
    fn get_current_scale_index(&self) -> usize {
        let current_scale = self.renderer.as_ref()
            .map(|r| r.ui_scale())
            .unwrap_or(1.0);
        let current_option = UiScaleOption::from_setting(self.user_scale_override);
        UiScaleOption::all().iter()
            .position(|o| *o == current_option)
            .unwrap_or(1) // Default to 100% (index 1)
    }

    fn handle_gamepad_button_pressed(&mut self, button: Button) {
        // Track L3/R3 for focus switching combo
        match button {
            Button::LeftThumb => {
                self.l3_held = true;
                // Check for L3+R3 combo to recapture
                if self.r3_held && self.gamepad_passthrough {
                    self.recapture_gamepad();
                    return;
                }
            }
            Button::RightThumb => {
                self.r3_held = true;
                // Check for L3+R3 combo to recapture
                if self.l3_held && self.gamepad_passthrough {
                    self.recapture_gamepad();
                    return;
                }
            }
            _ => {}
        }

        // If in passthrough mode, only respond to L3/R3 combo (handled above)
        if self.gamepad_passthrough {
            tracing::debug!("Gamepad passthrough - ignoring {:?}", button);
            return;
        }

        // Log button presses when not in passthrough
        tracing::info!("Gamepad button pressed: {:?}", button);

        // Map gamepad buttons to keyboard actions
        match button {
            Button::DPadUp => self.handle_input(KeyCode::ArrowUp),
            Button::DPadDown => self.handle_input(KeyCode::ArrowDown),
            Button::DPadLeft => self.handle_input(KeyCode::ArrowLeft),
            Button::DPadRight => self.handle_input(KeyCode::ArrowRight),
            Button::South => self.handle_input(KeyCode::Enter), // A button
            Button::East => self.handle_input(KeyCode::Backspace), // B button
            Button::West => self.handle_input(KeyCode::KeyX), // X button
            Button::North => self.handle_input(KeyCode::KeyY), // Y button
            Button::Start => {
                // Open settings menu (or close if already in settings)
                self.toggle_main_menu();
            }
            Button::LeftThumb => {
                // L3 alone = release gamepad focus
                if !self.r3_held {
                    self.release_gamepad_focus();
                }
            }
            _ => {
                tracing::debug!("Unmapped button: {:?}", button);
            }
        }
    }

    fn handle_gamepad_button_released(&mut self, button: Button) {
        match button {
            Button::LeftThumb => self.l3_held = false,
            Button::RightThumb => self.r3_held = false,
            _ => {}
        }
    }

    /// Handle left thumbstick movement for navigation
    /// Uses deadzone + threshold to convert analog to digital nav
    fn handle_gamepad_stick(&mut self, x: f32, y: f32) {
        const DEADZONE: f32 = 0.3;
        const THRESHOLD: f32 = 0.7;

        // If in passthrough mode, ignore stick input
        if self.gamepad_passthrough {
            return;
        }

        // Track last stick direction to avoid repeated inputs
        // Use a simple threshold-crossing approach
        static mut LAST_X: i8 = 0;
        static mut LAST_Y: i8 = 0;

        let new_x = if x > THRESHOLD { 1 } else if x < -THRESHOLD { -1 } else if x.abs() < DEADZONE { 0 } else { unsafe { LAST_X } };
        let new_y = if y > THRESHOLD { 1 } else if y < -THRESHOLD { -1 } else if y.abs() < DEADZONE { 0 } else { unsafe { LAST_Y } };

        unsafe {
            // Only trigger when crossing threshold, not when held
            if new_x != LAST_X {
                if new_x > 0 {
                    self.handle_input(KeyCode::ArrowRight);
                    self.request_redraw();
                } else if new_x < 0 {
                    self.handle_input(KeyCode::ArrowLeft);
                    self.request_redraw();
                }
                LAST_X = new_x;
            }

            // Y axis: negative = down (stick pushed away from you = up)
            if new_y != LAST_Y {
                if new_y > 0 {
                    self.handle_input(KeyCode::ArrowUp);
                    self.request_redraw();
                } else if new_y < 0 {
                    self.handle_input(KeyCode::ArrowDown);
                    self.request_redraw();
                }
                LAST_Y = new_y;
            }
        }
    }

    /// Handle right thumbstick for scrolling (logs when no cards, description when cards exist)
    /// Analog: scroll speed proportional to stick deflection, continuous while held
    fn handle_gamepad_right_stick(&mut self, y: f32) {
        const DEADZONE: f32 = 0.15;
        const MAX_SCROLL_SPEED: f32 = 12.0; // Max pixels per frame at full deflection

        if self.gamepad_passthrough {
            return;
        }

        if y.abs() < DEADZONE {
            return;
        }

        let scroll_amount = -y * MAX_SCROLL_SPEED; // Invert: push up = positive scroll

        // Phase 1: Extract info we need for calculation
        enum ScrollAction {
            LogScroll,
            CardScroll { card: crate::state::SuggestionCard, current: f32 },
            None,
        }

        let action = if let AppState::PalaceLoop { cards, focused_index, tool_log, thought_log, log_scroll_offset, detail_scroll_offset, .. } = &mut self.state {
            if cards.is_empty() {
                let max_scroll = tool_log.len().max(thought_log.len()).saturating_sub(1);
                if scroll_amount > 2.0 {
                    *log_scroll_offset = (*log_scroll_offset + 1).min(max_scroll);
                    ScrollAction::LogScroll
                } else if scroll_amount < -2.0 {
                    *log_scroll_offset = log_scroll_offset.saturating_sub(1);
                    ScrollAction::LogScroll
                } else {
                    ScrollAction::None
                }
            } else if let Some(card) = cards.get(*focused_index) {
                ScrollAction::CardScroll { card: card.clone(), current: *detail_scroll_offset }
            } else {
                ScrollAction::None
            }
        } else {
            ScrollAction::None
        };

        // Phase 2: Calculate max scroll (needs renderer)
        let new_offset = match &action {
            ScrollAction::CardScroll { card, current } => {
                if let Some(renderer) = self.renderer.as_mut() {
                    let max_scroll = renderer.calculate_card_max_scroll(card);
                    Some((*current + scroll_amount).clamp(0.0, max_scroll))
                } else {
                    None
                }
            }
            _ => None,
        };

        // Phase 3: Write back to state
        match action {
            ScrollAction::LogScroll => self.request_redraw(),
            ScrollAction::CardScroll { .. } => {
                if let Some(offset) = new_offset {
                    if let AppState::PalaceLoop { detail_scroll_offset, .. } = &mut self.state {
                        *detail_scroll_offset = offset;
                        self.request_redraw();
                    }
                }
            }
            ScrollAction::None => {}
        }
    }

    /// Release gamepad focus - minimize Palace and let other apps use the gamepad
    fn release_gamepad_focus(&mut self) {
        if self.gamepad_passthrough {
            return; // Already in passthrough
        }

        tracing::info!("🎮 Releasing gamepad focus (L3+R3 to recapture)");
        self.gamepad_passthrough = true;

        // Minimize the window
        if let Some(window) = &self.window {
            window.set_minimized(true);
        }
    }

    /// Recapture gamepad focus - restore Palace and take back gamepad control
    fn recapture_gamepad(&mut self) {
        if !self.gamepad_passthrough {
            return; // Already captured
        }

        tracing::info!("🎮 Recaptured gamepad focus");
        self.gamepad_passthrough = false;

        // Restore the window to fullscreen
        if let Some(window) = &self.window {
            // First ensure it's visible
            window.set_visible(true);
            // Unminimize
            window.set_minimized(false);
            // Restore fullscreen
            window.set_fullscreen(Some(Fullscreen::Borderless(None)));
            // Try to focus
            window.focus_window();
        }

        // On GNOME/Wayland, show overview so user can tap Palace to focus
        self.show_gnome_overview();

        self.request_redraw();
    }

    /// Add a command prefix to the approved list
    fn approve_command_prefix(prefix: &str) {
        use std::sync::{OnceLock, Mutex as StdMutex};
        static APPROVED: OnceLock<StdMutex<Vec<String>>> = OnceLock::new();
        let approved = APPROVED.get_or_init(|| StdMutex::new(Vec::new()));
        let mut list = approved.lock().unwrap();
        if !list.contains(&prefix.to_string()) {
            list.push(prefix.to_string());
            tracing::info!("Approved command prefix: {}", prefix);
        }
    }

    /// Check if a command is approved (prefix match)
    pub fn is_command_approved(cmd: &str) -> bool {
        use std::sync::{OnceLock, Mutex as StdMutex};
        static APPROVED: OnceLock<StdMutex<Vec<String>>> = OnceLock::new();
        let approved = APPROVED.get_or_init(|| StdMutex::new(Vec::new()));
        let list = approved.lock().unwrap();
        list.iter().any(|a| cmd.starts_with(a))
    }

    /// Show GNOME Overview so user can see and select Palace window
    /// (Wayland prevents apps from stealing focus, so we show overview instead)
    fn show_gnome_overview(&self) {
        use std::process::Command;

        let wayland_display = std::env::var("WAYLAND_DISPLAY").unwrap_or_else(|_| "wayland-0".to_string());

        // Set OverviewActive = true to show the GNOME overview
        match Command::new("gdbus")
            .args([
                "call", "--session",
                "--dest", "org.gnome.Shell",
                "--object-path", "/org/gnome/Shell",
                "--method", "org.freedesktop.DBus.Properties.Set",
                "org.gnome.Shell", "OverviewActive", "<true>",
            ])
            .env("WAYLAND_DISPLAY", &wayland_display)
            .output()
        {
            Ok(output) => {
                if output.status.success() {
                    tracing::info!("🎮 Opened GNOME Overview - tap Palace to focus");
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    tracing::warn!("Failed to open overview: {}", stderr);
                }
            }
            Err(e) => {
                tracing::debug!("Failed to run gdbus: {}", e);
            }
        }
    }

    /// Toggle the main menu (Resume, Settings, Exit)
    fn toggle_main_menu(&mut self) {
        match &self.state {
            AppState::MainMenu { previous_state, .. } => {
                // Close menu, return to previous state (Resume)
                tracing::info!("Closing main menu");
                self.state = *previous_state.clone();
            }
            AppState::SettingsMenu { previous_state, .. } => {
                // Go back to main menu from settings
                tracing::info!("Going back to main menu from settings");
                self.state = *previous_state.clone();
            }
            _ => {
                // Open main menu
                tracing::info!("Opening main menu");
                self.state = AppState::MainMenu {
                    selected_item: 0,
                    previous_state: Box::new(self.state.clone()),
                };
            }
        }
    }

    fn handle_input(&mut self, key: KeyCode) {
        // Calculate columns first to avoid borrow issues
        let columns = self.grid_columns();

        match &mut self.state {
            AppState::ProjectChooser { selected_index } => {
                let project_count = self.projects.projects.len();
                if project_count == 0 {
                    return;
                }

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        // Move up one row
                        if *selected_index >= columns {
                            *selected_index -= columns;
                        } else {
                            // Wrap to last row
                            let last_row_start = (project_count / columns) * columns;
                            let target = last_row_start + (*selected_index % columns);
                            *selected_index = target.min(project_count - 1);
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        // Move down one row
                        let next = *selected_index + columns;
                        if next < project_count {
                            *selected_index = next;
                        } else {
                            // Wrap to first row
                            *selected_index = *selected_index % columns;
                        }
                    }
                    KeyCode::ArrowLeft | KeyCode::KeyA => {
                        if *selected_index > 0 {
                            *selected_index -= 1;
                        } else {
                            *selected_index = project_count - 1;
                        }
                    }
                    KeyCode::ArrowRight | KeyCode::KeyD => {
                        if *selected_index < project_count - 1 {
                            *selected_index += 1;
                        } else {
                            *selected_index = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        if let Some(project) = self.projects.projects.get(*selected_index) {
                            tracing::info!("Selected project: {}", project.name);
                            self.state = AppState::project_view(project.path.clone());
                        }
                    }
                    _ => {}
                }
            }
            AppState::ProjectView { project_path, selected_action } => {
                let action_count = crate::state::ProjectAction::all().len();
                let project_path = project_path.clone();

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_action > 0 {
                            *selected_action -= 1;
                        } else {
                            *selected_action = action_count - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_action < action_count - 1 {
                            *selected_action += 1;
                        } else {
                            *selected_action = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        use crate::state::ProjectAction;
                        let action = ProjectAction::all()[*selected_action];
                        tracing::info!("Selected action: {:?}", action);

                        match action {
                            ProjectAction::StartPalaceLoop => {
                                // Transition to Palace Loop state
                                self.state = AppState::PalaceLoop {
                                    project_path: project_path.clone(),
                                    cards: Vec::new(),
                                    focused_index: 0,
                                    hovered_index: None,
                                    generating: true,
                                    current_tool: None,
                                    tool_log: Vec::new(),
                                    thought_log: Vec::new(),
                                    log_scroll_offset: 0,
                                    detail_scroll_offset: 0.0,
                                    detail_max_scroll: 0.0,
                                };

                                // Spawn AI suggestion thread
                                let proxy = self.event_proxy.clone();
                                let path = project_path.clone();
                                std::thread::spawn(move || {
                                    Self::run_ai_suggestions(path, proxy);
                                });
                            }
                            ProjectAction::Build => {
                                tracing::info!("Build action not yet implemented");
                            }
                            ProjectAction::Run => {
                                tracing::info!("Run action not yet implemented");
                            }
                            ProjectAction::ViewGitHistory => {
                                tracing::info!("Git history action not yet implemented");
                            }
                        }
                    }
                    KeyCode::Backspace => {
                        // Go back to project chooser
                        self.state = AppState::project_chooser();
                    }
                    _ => {}
                }
            }
            AppState::MainMenu {
                selected_item,
                previous_state,
            } => {
                use crate::state::MainMenuItem;
                let item_count = MainMenuItem::all().len();

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_item > 0 {
                            *selected_item -= 1;
                        } else {
                            *selected_item = item_count - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_item < item_count - 1 {
                            *selected_item += 1;
                        } else {
                            *selected_item = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        let item = MainMenuItem::all()[*selected_item];
                        match item {
                            MainMenuItem::Resume => {
                                tracing::info!("Resume - closing menu");
                                self.state = *previous_state.clone();
                            }
                            MainMenuItem::Settings => {
                                tracing::info!("Opening settings submenu");
                                self.state = AppState::SettingsMenu {
                                    selected_item: 0,
                                    previous_state: Box::new(self.state.clone()),
                                };
                            }
                            MainMenuItem::Exit => {
                                tracing::info!("Exit requested");
                                self.exit_requested = true;
                            }
                        }
                    }
                    KeyCode::Backspace | KeyCode::Escape => {
                        // Go back to previous state (Resume)
                        self.state = *previous_state.clone();
                    }
                    _ => {}
                }
            }
            AppState::SettingsMenu {
                selected_item,
                previous_state,
            } => {
                use crate::state::SettingsItem;
                let item_count = SettingsItem::all().len();

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_item > 0 {
                            *selected_item -= 1;
                        } else {
                            *selected_item = item_count - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_item < item_count - 1 {
                            *selected_item += 1;
                        } else {
                            *selected_item = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        // Toggle or adjust the selected setting
                        let item = SettingsItem::all()[*selected_item];
                        match item {
                            SettingsItem::DarkMode => {
                                self.dark_mode = !self.dark_mode;
                                tracing::info!("Dark mode: {}", if self.dark_mode { "ON" } else { "OFF" });
                                if let Some(renderer) = &mut self.renderer {
                                    renderer.set_dark_mode(self.dark_mode);
                                }
                            }
                            SettingsItem::UiScale => {
                                // Open UI Scale submenu
                                self.state = AppState::UiScaleMenu {
                                    selected_item: self.get_current_scale_index(),
                                    previous_state: Box::new(self.state.clone()),
                                    user_scale_override: self.user_scale_override,
                                };
                            }
                        }
                    }
                    KeyCode::Backspace | KeyCode::Escape => {
                        // Go back to main menu
                        self.state = *previous_state.clone();
                    }
                    _ => {}
                }
            }
            AppState::UiScaleMenu {
                selected_item,
                previous_state,
                ..
            } => {
                let scale_count = UiScaleOption::all().len();

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_item > 0 {
                            *selected_item -= 1;
                        } else {
                            *selected_item = scale_count - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_item < scale_count - 1 {
                            *selected_item += 1;
                        } else {
                            *selected_item = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        // Apply the selected scale
                        let scale_option = UiScaleOption::all()[*selected_item];
                        self.user_scale_override = scale_option.value();
                        // Use override if set, otherwise use detected
                        let actual_scale = self.user_scale_override.unwrap_or(self.detected_scale);
                        let is_auto = self.user_scale_override.is_none();
                        tracing::info!("Setting UI scale to {} ({})", scale_option.label(), actual_scale);
                        if let Some(renderer) = &mut self.renderer {
                            renderer.set_ui_scale(actual_scale, is_auto);
                        }
                        // Go back to settings menu
                        self.state = *previous_state.clone();
                    }
                    KeyCode::Backspace | KeyCode::Escape => {
                        // Go back to settings menu
                        self.state = *previous_state.clone();
                    }
                    _ => {}
                }
            }
            AppState::PermissionModal {
                selected_choice,
                previous_state,
                command_prefix,
                command,
            } => {
                use crate::state::PermissionChoice;
                let choice_count = PermissionChoice::all().len();
                let command_prefix = command_prefix.clone();
                let command = command.clone();
                let previous_state = previous_state.clone();

                // Permission modal uses direct button mappings (QTE style):
                // A = Yes once, X = Yes always, B = No, Y = Suggest else
                match key {
                    // Navigation still works for mouse/touch users
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_choice > 0 {
                            *selected_choice -= 1;
                        } else {
                            *selected_choice = choice_count - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_choice < choice_count - 1 {
                            *selected_choice += 1;
                        } else {
                            *selected_choice = 0;
                        }
                    }
                    // A button / Enter / Space → Yes once
                    KeyCode::Enter | KeyCode::Space => {
                        if let Some(tx) = self.permission_response_tx.take() {
                            let _ = tx.send(PermissionResponse::Approved);
                        }
                        self.state = *previous_state;
                    }
                    // X button → Yes always
                    KeyCode::KeyX => {
                        Self::approve_command_prefix(&command_prefix);
                        if let Some(tx) = self.permission_response_tx.take() {
                            let _ = tx.send(PermissionResponse::ApprovedAlways(command_prefix));
                        }
                        self.state = *previous_state;
                    }
                    // B button / Backspace / Escape → No
                    KeyCode::Backspace | KeyCode::Escape => {
                        if let Some(tx) = self.permission_response_tx.take() {
                            let _ = tx.send(PermissionResponse::Denied);
                        }
                        self.state = *previous_state;
                    }
                    // Y button → Suggest something else (ask Z.ai for alternatives)
                    KeyCode::KeyY => {
                        tracing::info!("Suggest else requested for: {}", command);
                        if let Some(tx) = self.permission_response_tx.take() {
                            let _ = tx.send(PermissionResponse::SuggestElse {
                                original_command: command,
                            });
                        }
                        self.state = *previous_state;
                    }
                    _ => {}
                }
            }
            AppState::PalaceLoop {
                cards,
                focused_index,
                detail_scroll_offset,
                generating,
                ..
            } => {
                let card_count = cards.len();
                let palace_columns = 5; // PalaceLoop uses 5-column grid

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        // Move up one row
                        if *focused_index >= palace_columns {
                            *focused_index -= palace_columns;
                            *detail_scroll_offset = 0.0; // Reset scroll on focus change
                        } else if card_count > 0 {
                            // Wrap to last row (same column or nearest)
                            let last_row_start = (card_count.saturating_sub(1) / palace_columns) * palace_columns;
                            let target = last_row_start + (*focused_index % palace_columns);
                            *focused_index = target.min(card_count - 1);
                            *detail_scroll_offset = 0.0;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        // Move down one row
                        let next = *focused_index + palace_columns;
                        if next < card_count {
                            *focused_index = next;
                            *detail_scroll_offset = 0.0; // Reset scroll on focus change
                        } else if card_count > 0 {
                            // Wrap to first row (same column)
                            *focused_index = *focused_index % palace_columns;
                            if *focused_index >= card_count {
                                *focused_index = 0;
                            }
                            *detail_scroll_offset = 0.0;
                        }
                    }
                    KeyCode::ArrowLeft | KeyCode::KeyA => {
                        if *focused_index > 0 {
                            *focused_index -= 1;
                            *detail_scroll_offset = 0.0; // Reset scroll on focus change
                        } else if card_count > 0 {
                            *focused_index = card_count - 1;
                            *detail_scroll_offset = 0.0;
                        }
                    }
                    KeyCode::ArrowRight | KeyCode::KeyD => {
                        if card_count > 0 && *focused_index < card_count - 1 {
                            *focused_index += 1;
                            *detail_scroll_offset = 0.0; // Reset scroll on focus change
                        } else {
                            *focused_index = 0;
                            *detail_scroll_offset = 0.0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        // Toggle selection on focused card
                        if let Some(card) = cards.get_mut(*focused_index) {
                            card.selected = !card.selected;
                            tracing::info!("Card {} selected: {}", card.id, card.selected);
                        }
                    }
                    KeyCode::Backspace | KeyCode::Escape => {
                        // Go back to project view
                        if let AppState::PalaceLoop { project_path, .. } = &self.state {
                            self.state = AppState::project_view(project_path.clone());
                        }
                    }
                    KeyCode::KeyX => {
                        // Show execute options modal (only when done generating and has selections)
                        if !*generating {
                            let has_selected = cards.iter().any(|c| c.selected);
                            if has_selected {
                                self.state = AppState::ExecuteModal {
                                    selected_option: 0,
                                    previous_state: Box::new(self.state.clone()),
                                };
                            }
                        }
                    }
                    _ => {}
                }
            }
            AppState::ExecuteModal {
                selected_option,
                previous_state,
            } => {
                use crate::state::ExecuteOption;
                let option_count = ExecuteOption::all().len();
                let previous_state = previous_state.clone();

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_option > 0 {
                            *selected_option -= 1;
                        } else {
                            *selected_option = option_count - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_option < option_count - 1 {
                            *selected_option += 1;
                        } else {
                            *selected_option = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        let option = ExecuteOption::all()[*selected_option];
                        tracing::info!("Execute option selected: {:?}", option);

                        // Extract project path and selected cards from PalaceLoop
                        if let AppState::PalaceLoop { project_path, cards, .. } = &*previous_state {
                            let selected_cards: Vec<SuggestionCard> = cards
                                .iter()
                                .filter(|c| c.selected)
                                .cloned()
                                .collect();

                            if selected_cards.is_empty() {
                                tracing::warn!("No cards selected for execution");
                                self.state = *previous_state;
                                return;
                            }

                            let project_path = project_path.clone();
                            let card_count = selected_cards.len();
                            tracing::info!(
                                "Starting execution of {} cards with {:?}",
                                card_count,
                                option
                            );

                            // Transition to Executing state
                            self.state = AppState::Executing {
                                project_path: project_path.clone(),
                                executing_cards: selected_cards.clone(),
                                status: ExecutionStatus::Running {
                                    current_card: 0,
                                    total_cards: card_count,
                                },
                                tool_log: Vec::new(),
                                thought_log: Vec::new(),
                                log_scroll_offset: 0,
                                executor: option,
                                previous_state: previous_state.clone(),
                            };

                            // Spawn the executor in a background thread
                            let proxy = (*self.event_proxy).clone();
                            crate::ai::spawn_execution(
                                option,
                                project_path,
                                selected_cards,
                                proxy,
                            );
                        } else {
                            // Shouldn't happen, but handle gracefully
                            tracing::error!("ExecuteModal previous_state is not PalaceLoop");
                            self.state = *previous_state;
                        }
                    }
                    KeyCode::Backspace | KeyCode::Escape => {
                        // Go back to PalaceLoop
                        self.state = *previous_state;
                    }
                    _ => {}
                }
            }
            AppState::Survey {
                options,
                focused_index,
                custom_input,
                custom_active,
                multi_select,
                selected_indices,
                use_quick_select,
                scroll_offset,
                previous_state,
                response_tx,
                ..
            } => {
                use crate::state::SurveyResponse;
                let option_count = options.len();
                let previous_state = previous_state.clone();
                let has_custom = true; // Always have "Other" option
                let total = option_count + if has_custom { 1 } else { 0 };

                // Calculate visible_count based on actual screen size
                let visible_count = if let Some(ref renderer) = self.renderer {
                    let size = renderer.size();
                    let ui_scale = renderer.ui_scale();
                    let scale = |v: f32| v * ui_scale;
                    let margin = scale(48.0);
                    let title_height = scale(80.0);
                    let card_height = scale(60.0);
                    let card_gap = scale(10.0);
                    let available_height = size.height as f32 - margin * 2.0 - title_height - scale(60.0);
                    (available_height / (card_height + card_gap)).floor() as usize
                } else {
                    8 // Fallback
                };

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        // Navigation disables quick-select
                        *use_quick_select = false;
                        if !*custom_active {
                            if *focused_index > 0 {
                                *focused_index -= 1;
                            } else {
                                *focused_index = total - 1;
                                // Wrap to bottom - scroll to show last item
                                *scroll_offset = total.saturating_sub(visible_count);
                            }
                            // Update scroll to keep focused item visible
                            if *focused_index < *scroll_offset {
                                *scroll_offset = *focused_index;
                            }
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        // Navigation disables quick-select
                        *use_quick_select = false;
                        if !*custom_active {
                            if *focused_index < total - 1 {
                                *focused_index += 1;
                            } else {
                                *focused_index = 0;
                                *scroll_offset = 0; // Wrap to top
                            }
                            // Update scroll to keep focused item visible
                            if *focused_index >= *scroll_offset + visible_count {
                                *scroll_offset = focused_index.saturating_sub(visible_count - 1);
                            }
                            if *focused_index < *scroll_offset {
                                *scroll_offset = *focused_index;
                            }
                        }
                    }
                    // A button / Enter / Space → Confirm focused or quick-select option 0
                    KeyCode::Enter | KeyCode::Space => {
                        if *custom_active {
                            // In custom mode, submit if there's input
                            if !custom_input.is_empty() {
                                if let Some(tx) = response_tx.take() {
                                    let _ = tx.send(SurveyResponse::Custom(custom_input.clone()));
                                }
                                self.state = *previous_state;
                            }
                        } else if *use_quick_select && option_count > 0 {
                            // Quick-select option 0
                            if let Some(tx) = response_tx.take() {
                                let _ = tx.send(SurveyResponse::Selected(vec![0]));
                            }
                            self.state = *previous_state;
                        } else if *focused_index >= option_count {
                            // Focused on "Other" option - activate custom input
                            *custom_active = true;
                        } else if *multi_select {
                            // Toggle selection in multi-select mode
                            if selected_indices.contains(focused_index) {
                                selected_indices.retain(|&i| i != *focused_index);
                            } else {
                                selected_indices.push(*focused_index);
                            }
                        } else {
                            // Single select - submit focused
                            if let Some(tx) = response_tx.take() {
                                let _ = tx.send(SurveyResponse::Selected(vec![*focused_index]));
                            }
                            self.state = *previous_state;
                        }
                    }
                    // X button → Quick-select option 1, or confirm multi-select
                    KeyCode::KeyX => {
                        if *custom_active {
                            // Ignore in custom mode
                        } else if *multi_select {
                            // Confirm multi-selection
                            if !selected_indices.is_empty() {
                                if let Some(tx) = response_tx.take() {
                                    let _ = tx.send(SurveyResponse::Selected(selected_indices.clone()));
                                }
                                self.state = *previous_state;
                            }
                        } else if *use_quick_select && option_count > 1 {
                            // Quick-select option 1
                            if let Some(tx) = response_tx.take() {
                                let _ = tx.send(SurveyResponse::Selected(vec![1]));
                            }
                            self.state = *previous_state;
                        }
                    }
                    // B button / Backspace → Quick-select option 2, or backspace
                    KeyCode::Backspace => {
                        if *custom_active {
                            custom_input.pop();
                        } else if *use_quick_select && option_count > 2 {
                            // Quick-select option 2
                            if let Some(tx) = response_tx.take() {
                                let _ = tx.send(SurveyResponse::Selected(vec![2]));
                            }
                            self.state = *previous_state;
                        }
                    }
                    // Y button → Quick-select option 3
                    KeyCode::KeyY => {
                        if !*custom_active && *use_quick_select && option_count > 3 {
                            // Quick-select option 3
                            if let Some(tx) = response_tx.take() {
                                let _ = tx.send(SurveyResponse::Selected(vec![3]));
                            }
                            self.state = *previous_state;
                        }
                    }
                    // Escape → Cancel survey or exit custom mode
                    KeyCode::Escape => {
                        if *custom_active {
                            *custom_active = false;
                        } else {
                            // Cancel the survey
                            if let Some(tx) = response_tx.take() {
                                let _ = tx.send(SurveyResponse::Cancelled);
                            }
                            self.state = *previous_state;
                        }
                    }
                    _ => {}
                }
            }
            AppState::Executing { status, previous_state, log_scroll_offset, .. } => {
                match key {
                    KeyCode::Escape | KeyCode::Backspace => {
                        // Cancel execution if running, or go back if done
                        if status.is_done() {
                            self.state = *previous_state.clone();
                        } else {
                            // Mark as cancelled
                            *status = crate::state::ExecutionStatus::Cancelled;
                        }
                    }
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        *log_scroll_offset = log_scroll_offset.saturating_sub(1);
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        *log_scroll_offset += 1;
                    }
                    _ => {}
                }
            }
        }
    }

    fn grid_columns(&self) -> usize {
        // Calculate columns based on window size if available
        if let Some(window) = &self.window {
            let width = window.inner_size().width as f32;
            // Same calculation as CardGrid: (available_width + gap) / (card_width + gap)
            let margin = 72.0; // scaled margin
            let card_width = 384.0; // scaled card width
            let gap = 28.8; // scaled gap
            let available = width - margin * 2.0;
            let cols = ((available + gap) / (card_width + gap)).floor() as usize;
            cols.max(1)
        } else {
            4 // Default for 1920px width
        }
    }

    fn handle_touch(&mut self, touch: Touch) {
        let x = touch.location.x as f32;
        let y = touch.location.y as f32;

        match touch.phase {
            TouchPhase::Started => {
                tracing::info!("Touch started at ({:.0}, {:.0}), id: {:?}", x, y, touch.id);
                // Record touch start for hold detection
                let card_under_touch = self.card_at_position(x, y);
                self.touch_hold = Some((std::time::Instant::now(), x, y, card_under_touch));
                self.touch_hold_triggered = false;
            }
            TouchPhase::Ended => {
                tracing::info!("Touch ended at ({:.0}, {:.0}), id: {:?}", x, y, touch.id);

                // Check if this was a tap (short hold, minimal movement)
                let should_tap = if let Some((start_time, start_x, start_y, _)) = self.touch_hold {
                    let held_ms = start_time.elapsed().as_millis();
                    let dx = (x - start_x).abs();
                    let dy = (y - start_y).abs();
                    let moved = dx > 20.0 || dy > 20.0;

                    // Tap = short hold AND didn't move much AND hold didn't already trigger
                    held_ms < 300 && !moved && !self.touch_hold_triggered
                } else {
                    false
                };

                if should_tap {
                    self.handle_tap(x, y);
                }

                // Clear touch state and hover
                self.touch_hold = None;
                self.touch_hold_triggered = false;

                // Clear hover when touch ends (in PalaceLoop)
                if let AppState::PalaceLoop { hovered_index, .. } = &mut self.state {
                    if hovered_index.is_some() {
                        *hovered_index = None;
                        self.hover_changed = true;
                    }
                }
            }
            TouchPhase::Moved => {
                tracing::debug!("Touch moved to ({:.0}, {:.0})", x, y);

                // Check for hold-to-reveal (300ms threshold for initial trigger)
                if let Some((start_time, start_x, start_y, _)) = self.touch_hold {
                    // If already triggered, immediately update hovered card (no delay)
                    if self.touch_hold_triggered {
                        let card_under = self.card_at_position(x, y);
                        if let AppState::PalaceLoop { hovered_index, .. } = &mut self.state {
                            if *hovered_index != card_under {
                                *hovered_index = card_under;
                                self.hover_changed = true;
                            }
                        }
                    } else {
                        // Not yet triggered - check if we should trigger
                        let held_ms = start_time.elapsed().as_millis();
                        let dx = (x - start_x).abs();
                        let dy = (y - start_y).abs();
                        let moved_enough = dx > 20.0 || dy > 20.0;

                        // After 300ms hold, or after moving 20px: trigger reveal mode
                        if held_ms >= 300 || moved_enough {
                            self.touch_hold_triggered = true;

                            // Immediately show description of card under finger
                            let card_under = self.card_at_position(x, y);
                            if let AppState::PalaceLoop { hovered_index, .. } = &mut self.state {
                                if *hovered_index != card_under {
                                    *hovered_index = card_under;
                                    self.hover_changed = true;
                                }
                            }
                        }
                    }
                }
            }
            TouchPhase::Cancelled => {
                tracing::debug!("Touch cancelled");
                self.touch_hold = None;
                self.touch_hold_triggered = false;

                // Clear hover on cancel
                if let AppState::PalaceLoop { hovered_index, .. } = &mut self.state {
                    if hovered_index.is_some() {
                        *hovered_index = None;
                        self.hover_changed = true;
                    }
                }
            }
        }
    }

    /// Find which card (if any) is at the given screen position
    fn card_at_position(&self, x: f32, y: f32) -> Option<usize> {
        if let AppState::PalaceLoop { cards, .. } = &self.state {
            let window_size = self.window.as_ref().map(|w| w.inner_size());
            let Some(size) = window_size else { return None };

            // Must match CardGrid::for_palace_loop() in gpu.rs
            let ui_scale = self.renderer.as_ref().map(|r| r.ui_scale()).unwrap_or(1.0);
            let scale = |v: f32| v * ui_scale;

            let target_columns = 5usize;
            let base_gap = 16.0;
            let base_margin = 40.0;
            let screen_width = size.width as f32;

            let gap = scale(base_gap);
            let margin_x = scale(base_margin);
            let margin_y = scale(base_margin + 80.0); // Extra space for title + subtitle

            // Calculate card width to fit exactly 5 columns
            let available_width = screen_width - margin_x * 2.0;
            let card_width = (available_width - gap * (target_columns - 1) as f32) / target_columns as f32;
            let card_height = card_width * 0.6; // Same aspect ratio as renderer

            for (i, _card) in cards.iter().enumerate() {
                let col = i % target_columns;
                let row = i / target_columns;
                let card_x = margin_x + col as f32 * (card_width + gap);
                let card_y = margin_y + row as f32 * (card_height + gap);

                if x >= card_x && x <= card_x + card_width
                   && y >= card_y && y <= card_y + card_height {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Handle cursor movement for hover detection
    fn handle_cursor_moved(&mut self, x: f32, y: f32) {
        // Only handle hover in PalaceLoop state
        if let AppState::PalaceLoop { cards, hovered_index, .. } = &mut self.state {
            let window_size = self.window.as_ref().map(|w| w.inner_size());
            let Some(size) = window_size else { return };

            // Must match CardGrid::for_palace_loop() in gpu.rs
            let ui_scale = self.renderer.as_ref().map(|r| r.ui_scale()).unwrap_or(1.0);
            let scale = |v: f32| v * ui_scale;

            let target_columns = 5usize;
            let base_gap = 16.0;
            let base_margin = 40.0;
            let screen_width = size.width as f32;

            let gap = scale(base_gap);
            let margin_x = scale(base_margin);
            let margin_y = scale(base_margin + 80.0); // Extra space for title + subtitle

            // Calculate card width to fit exactly 5 columns
            let available_width = screen_width - margin_x * 2.0;
            let card_width = (available_width - gap * (target_columns - 1) as f32) / target_columns as f32;
            let card_height = card_width * 0.6; // Same aspect ratio as renderer

            // Find which card (if any) the cursor is over
            let mut new_hovered = None;
            for (i, _card) in cards.iter().enumerate() {
                let col = i % target_columns;
                let row = i / target_columns;
                let card_x = margin_x + col as f32 * (card_width + gap);
                let card_y = margin_y + row as f32 * (card_height + gap);

                if x >= card_x && x <= card_x + card_width
                   && y >= card_y && y <= card_y + card_height {
                    new_hovered = Some(i);
                    break;
                }
            }

            // Only set hover_changed if the hovered card actually changed
            if *hovered_index != new_hovered {
                *hovered_index = new_hovered;
                self.hover_changed = true;
            }
        }
    }

    fn handle_tap(&mut self, x: f32, y: f32) {
        let window_size = self.window.as_ref().map(|w| w.inner_size());
        let Some(size) = window_size else { return };
        let columns = self.grid_columns();

        match &mut self.state {
            AppState::ProjectChooser { selected_index } => {
                // Check if tap is on a project card
                let margin = 72.0;
                let card_width = 384.0;
                let card_height = 216.0;
                let gap = 28.8;
                let top_offset = 120.0; // Space for title

                for (i, _project) in self.projects.projects.iter().enumerate() {
                    let col = i % columns;
                    let row = i / columns;
                    let card_x = margin + col as f32 * (card_width + gap);
                    let card_y = top_offset + row as f32 * (card_height + gap);

                    if x >= card_x && x <= card_x + card_width
                       && y >= card_y && y <= card_y + card_height {
                        tracing::info!("Tapped project card {}", i);
                        *selected_index = i;
                        // Double-tap or long-press could select
                        self.handle_input(KeyCode::Enter);
                        return;
                    }
                }
            }
            AppState::ProjectView { selected_action, .. } => {
                // Check if tap is on an action card
                let margin = 72.0;
                let card_height = 84.0; // ui_scale.px(70.0) ≈ 84
                let card_gap = 19.2;
                let menu_start_y = 180.0; // Approx: top_margin + title_scale + 60

                let actions = crate::state::ProjectAction::all();
                for (i, _action) in actions.iter().enumerate() {
                    let card_y = menu_start_y + i as f32 * (card_height + card_gap);
                    let card_width = size.width as f32 - margin * 2.0;

                    if x >= margin && x <= margin + card_width
                       && y >= card_y && y <= card_y + card_height {
                        tracing::info!("Tapped action card {}", i);
                        *selected_action = i;
                        self.handle_input(KeyCode::Enter);
                        return;
                    }
                }

                // Check if tap is in back area (bottom left)
                if y > size.height as f32 - 100.0 && x < 200.0 {
                    tracing::info!("Tapped back area");
                    self.handle_input(KeyCode::Backspace);
                }
            }
            AppState::MainMenu { selected_item, .. } => {
                // Tap on menu items
                let modal_width = 400.0f32.min(size.width as f32 - 40.0);
                let modal_height = 220.0;
                let modal_x = (size.width as f32 - modal_width) / 2.0;
                let modal_y = (size.height as f32 - modal_height) / 2.0;
                let card_height = 50.0;
                let card_gap = 10.0;
                let inner_padding = 20.0;
                let title_height = 10.0;

                let items = crate::state::MainMenuItem::all();
                for (i, _item) in items.iter().enumerate() {
                    let card_y = modal_y + title_height + i as f32 * (card_height + card_gap);
                    let card_width = modal_width - inner_padding * 2.0;
                    let card_x = modal_x + inner_padding;

                    if x >= card_x && x <= card_x + card_width
                       && y >= card_y && y <= card_y + card_height {
                        tracing::info!("Tapped menu item {}", i);
                        *selected_item = i;
                        self.handle_input(KeyCode::Enter);
                        return;
                    }
                }
            }
            AppState::SettingsMenu { selected_item, .. } => {
                // Tap on settings items
                let modal_width = 500.0f32.min(size.width as f32 - 40.0);
                let modal_height = 250.0;
                let modal_x = (size.width as f32 - modal_width) / 2.0;
                let modal_y = (size.height as f32 - modal_height) / 2.0;
                let card_height = 60.0;
                let card_gap = 12.0;
                let inner_padding = 20.0;
                let title_height = 50.0;

                let items = crate::state::SettingsItem::all();
                for (i, _item) in items.iter().enumerate() {
                    let card_y = modal_y + title_height + i as f32 * (card_height + card_gap);
                    let card_width = modal_width - inner_padding * 2.0;
                    let card_x = modal_x + inner_padding;

                    if x >= card_x && x <= card_x + card_width
                       && y >= card_y && y <= card_y + card_height {
                        tracing::info!("Tapped settings item {}", i);
                        *selected_item = i;
                        self.handle_input(KeyCode::Enter);
                        return;
                    }
                }
            }
            AppState::PalaceLoop {
                cards,
                focused_index,
                ..
            } => {
                // Tap on suggestion cards to toggle selection
                // Must match CardGrid::for_palace_loop() in gpu.rs
                let ui_scale = self.renderer.as_ref().map(|r| r.ui_scale()).unwrap_or(1.0);
                let scale = |v: f32| v * ui_scale;

                let target_columns = 5usize;
                let base_gap = 16.0;
                let base_margin = 40.0;
                let screen_width = size.width as f32;

                let gap = scale(base_gap);
                let margin_x = scale(base_margin);
                let margin_y = scale(base_margin + 80.0); // Extra space for title + subtitle

                // Calculate card width to fit exactly 5 columns
                let available_width = screen_width - margin_x * 2.0;
                let card_width = (available_width - gap * (target_columns - 1) as f32) / target_columns as f32;
                let card_height = card_width * 0.6; // Same aspect ratio as renderer

                for (i, _card) in cards.iter().enumerate() {
                    let col = i % target_columns;
                    let row = i / target_columns;
                    let card_x = margin_x + col as f32 * (card_width + gap);
                    let card_y = margin_y + row as f32 * (card_height + gap);

                    if x >= card_x && x <= card_x + card_width
                       && y >= card_y && y <= card_y + card_height {
                        tracing::info!("Tapped suggestion card {} at ({}, {})", i, col, row);
                        *focused_index = i;
                        // Toggle selection
                        if let Some(card) = cards.get_mut(i) {
                            card.selected = !card.selected;
                        }
                        self.request_redraw();
                        return;
                    }
                }
            }
            AppState::UiScaleMenu { selected_item, .. } => {
                // Tap on scale option items
                let modal_width = 400.0f32.min(size.width as f32 - 40.0);
                let item_count = UiScaleOption::all().len() as f32;
                let card_height = 50.0;
                let card_gap = 8.0;
                let inner_padding = 20.0;
                let title_height = 50.0;
                let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
                let modal_x = (size.width as f32 - modal_width) / 2.0;
                let modal_y = (size.height as f32 - modal_height) / 2.0;

                let items = UiScaleOption::all();
                for (i, _item) in items.iter().enumerate() {
                    let card_y = modal_y + title_height + i as f32 * (card_height + card_gap);
                    let card_width = modal_width - inner_padding * 2.0;
                    let card_x = modal_x + inner_padding;

                    if x >= card_x && x <= card_x + card_width
                       && y >= card_y && y <= card_y + card_height {
                        tracing::info!("Tapped scale option {}", i);
                        *selected_item = i;
                        self.handle_input(KeyCode::Enter);
                        return;
                    }
                }
            }
            AppState::PermissionModal { selected_choice, command, .. } => {
                // Tap on permission choice options - dynamic sizing based on command
                use crate::state::PermissionChoice;
                let ui_scale = self.renderer.as_ref().map(|r| r.ui_scale()).unwrap_or(1.0);
                let scale = |v: f32| v * ui_scale;

                let max_width = scale(800.0).min(size.width as f32 * 0.9);
                let min_width = scale(400.0);
                let inner_padding = scale(20.0);
                let command_scale = scale(16.0);

                // Estimate chars per line
                let char_width = command_scale * 0.55;
                let usable_width = max_width - inner_padding * 2.0;
                let chars_per_line = (usable_width / char_width).floor() as usize;
                let command_lines = ((command.len() as f32) / chars_per_line as f32).ceil() as usize;
                let command_lines = command_lines.max(1).min(10);

                let command_text_width = (command.len().min(chars_per_line) as f32 * char_width) + inner_padding * 2.0;
                let modal_width = command_text_width.max(min_width).min(max_width);

                let item_count = PermissionChoice::all().len() as f32;
                let card_height = scale(50.0);
                let card_gap = scale(8.0);
                let title_height = scale(50.0);
                let line_height = command_scale * 1.4;
                let command_height = line_height * command_lines as f32 + scale(20.0);
                let modal_height = title_height + command_height + inner_padding + item_count * (card_height + card_gap);
                let modal_x = (size.width as f32 - modal_width) / 2.0;
                let modal_y = (size.height as f32 - modal_height) / 2.0;

                let items = PermissionChoice::all();
                let card_start_y = modal_y + title_height + command_height;
                for (i, _item) in items.iter().enumerate() {
                    let card_y = card_start_y + i as f32 * (card_height + card_gap);
                    let card_width = modal_width - inner_padding * 2.0;
                    let card_x = modal_x + inner_padding;

                    if x >= card_x && x <= card_x + card_width
                       && y >= card_y && y <= card_y + card_height {
                        tracing::info!("Tapped permission choice {}", i);
                        *selected_choice = i;
                        self.handle_input(KeyCode::Enter);
                        return;
                    }
                }
            }
            AppState::ExecuteModal { selected_option, .. } => {
                // Tap on execute option items
                use crate::state::ExecuteOption;
                let ui_scale = self.renderer.as_ref().map(|r| r.ui_scale()).unwrap_or(1.0);
                let scale = |v: f32| v * ui_scale;

                let modal_width = scale(400.0).min(size.width as f32 - 40.0);
                let item_count = ExecuteOption::all().len() as f32;
                let card_height = scale(50.0);
                let card_gap = scale(8.0);
                let inner_padding = scale(20.0);
                let title_height = scale(50.0);
                let modal_height = title_height + inner_padding + item_count * (card_height + card_gap);
                let modal_x = (size.width as f32 - modal_width) / 2.0;
                let modal_y = (size.height as f32 - modal_height) / 2.0;

                let items = ExecuteOption::all();
                let card_start_y = modal_y + title_height;
                for (i, _item) in items.iter().enumerate() {
                    let card_y = card_start_y + i as f32 * (card_height + card_gap);
                    let card_width = modal_width - inner_padding * 2.0;
                    let card_x = modal_x + inner_padding;

                    if x >= card_x && x <= card_x + card_width
                       && y >= card_y && y <= card_y + card_height {
                        tracing::info!("Tapped execute option {}", i);
                        *selected_option = i;
                        self.handle_input(KeyCode::Enter);
                        return;
                    }
                }
            }
            AppState::Survey { options, focused_index, scroll_offset, .. } => {
                // Tap on survey option items - full-width layout with scroll
                let ui_scale = self.renderer.as_ref().map(|r| r.ui_scale()).unwrap_or(1.0);
                let scale = |v: f32| v * ui_scale;

                let margin = scale(48.0);
                let modal_width = size.width as f32 - margin * 2.0;
                let total_options = options.len() + 1; // +1 for "Other"
                let card_height = scale(60.0);
                let card_gap = scale(10.0);
                let inner_padding = scale(20.0);
                let title_height = scale(80.0);

                // Calculate visible options
                let available_height = size.height as f32 - margin * 2.0 - title_height - scale(60.0);
                let max_visible = (available_height / (card_height + card_gap)).floor() as usize;
                let visible_count = max_visible.min(total_options);

                let modal_height = title_height + inner_padding + visible_count as f32 * (card_height + card_gap);
                let modal_x = margin;
                let modal_y = (size.height as f32 - modal_height) / 2.0;

                let card_start_y = modal_y + title_height;
                let card_width = modal_width - inner_padding * 2.0;
                let card_x = modal_x + inner_padding;

                // Check visible cards (accounting for scroll)
                for visible_idx in 0..visible_count {
                    let actual_idx = *scroll_offset + visible_idx;
                    if actual_idx >= total_options { break; }

                    let card_y = card_start_y + visible_idx as f32 * (card_height + card_gap);

                    if x >= card_x && x <= card_x + card_width
                       && y >= card_y && y <= card_y + card_height {
                        tracing::info!("Tapped survey option {}", actual_idx);
                        *focused_index = actual_idx;
                        self.handle_input(KeyCode::Enter);
                        return;
                    }
                }
            }
            AppState::Executing { .. } => {
                // Tap anywhere to go back when done, or show cancel confirmation
                self.handle_input(KeyCode::Escape);
            }
        }
    }

    fn handle_debug_command(&mut self, command: DebugCommand) -> DebugResponse {
        match command {
            DebugCommand::Screenshot { path } => {
                if self.renderer.is_none() {
                    return DebugResponse::Error {
                        message: "Renderer not initialized".to_string(),
                    };
                }

                // Generate default path if not specified
                let path = path.unwrap_or_else(|| {
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    PathBuf::from(format!("/tmp/palace-screenshot-{}.png", timestamp))
                });

                // Store the request - will be captured after next render
                self.pending_screenshot = Some(path.clone());
                DebugResponse::ScreenshotPending { path }
            }
        }
    }

    /// Find the target monitor from config or return None for default
    fn find_target_monitor(&self, event_loop: &ActiveEventLoop) -> Option<winit::monitor::MonitorHandle> {
        // Try to load display config
        let config_path = std::path::Path::new("/home/wings/.config/palace/display.json");
        if config_path.exists() {
            if let Ok(json) = std::fs::read_to_string(config_path) {
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
                        tracing::warn!("Configured display '{}' not found, using default", display_name);
                    }
                }
            }
        }
        None // Use default (primary) monitor
    }
}

impl ApplicationHandler<AppEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        tracing::info!("Creating fullscreen window...");

        // Find the target monitor (from config or default to primary)
        let target_monitor = self.find_target_monitor(event_loop);
        if let Some(ref mon) = target_monitor {
            tracing::info!("Using display: {:?}", mon.name());
        }

        // Create fullscreen window
        let window_attrs = Window::default_attributes()
            .with_title("Palace")
            .with_fullscreen(Some(Fullscreen::Borderless(target_monitor)));

        match event_loop.create_window(window_attrs) {
            Ok(window) => {
                tracing::info!("Window created: {:?}", window.inner_size());

                // Initialize renderer
                match pollster::block_on(Renderer::new(&window)) {
                    Ok(mut renderer) => {
                        tracing::info!("Renderer initialized");

                        // Detect and apply display scaling
                        let size = window.inner_size();
                        let display_name = window.current_monitor()
                            .and_then(|m| m.name())
                            .unwrap_or_else(|| "unknown".to_string());

                        let scaling_config = DisplayScaling::load();
                        let detected = scaling_config.get_scale_for_display(
                            &display_name,
                            size.width,
                            size.height,
                        );
                        self.detected_scale = detected;
                        // Use user override if set, otherwise auto-detected
                        let scale = self.user_scale_override.unwrap_or(detected);
                        let is_auto = self.user_scale_override.is_none();
                        renderer.set_ui_scale(scale, is_auto);

                        // Set initial gamepad connection state
                        renderer.set_gamepad_connected(self.gamepad_connected);
                        self.renderer = Some(renderer);
                    }
                    Err(e) => {
                        tracing::error!("Failed to create renderer: {}", e);
                        event_loop.exit();
                        return;
                    }
                }

                self.window = Some(window);
            }
            Err(e) => {
                tracing::error!("Failed to create window: {}", e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                tracing::info!("Close requested");
                event_loop.exit();
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                // Escape opens the main menu (instead of exiting)
                if key == KeyCode::Escape {
                    self.open_main_menu();
                    self.request_redraw();
                } else {
                    self.handle_input(key);
                    self.request_redraw();
                }
            }

            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(new_size);
                }
                self.request_redraw();
            }

            WindowEvent::Touch(touch) => {
                self.handle_touch(touch);
                self.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.handle_cursor_moved(position.x as f32, position.y as f32);
                // Only redraw if hover state changed
                if self.hover_changed {
                    self.hover_changed = false;
                    self.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                // Clear redraw flag - we're rendering now
                self.needs_redraw = false;

                // Poll pending screenshot captures
                if let Some(renderer) = &self.renderer {
                    renderer.poll_screenshots(&mut self.screenshot_capture);
                }

                if let Some(renderer) = &mut self.renderer {
                    // Check if we need to capture a screenshot
                    let screenshot_path = self.pending_screenshot.take();

                    match renderer.render_with_screenshot(
                        &self.state,
                        &self.projects,
                        screenshot_path.as_ref(),
                        &mut self.screenshot_capture,
                    ) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            if let Some(window) = &self.window {
                                renderer.resize(window.inner_size());
                            }
                            // Need to redraw after resize
                            self.needs_redraw = true;
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            tracing::error!("Out of GPU memory");
                            event_loop.exit();
                        }
                        Err(e) => {
                            tracing::warn!("Render error: {:?}", e);
                        }
                    }
                }

                // Only request another frame if needed (idle optimization)
                // Note: gamepad/debug polling happens in about_to_wait

                // Check if exit was requested via menu
                if self.exit_requested {
                    event_loop.exit();
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        use winit::event_loop::ControlFlow;

        // Check if we have pending screenshot captures that need GPU polling
        let has_pending_captures = self.screenshot_capture.has_pending();

        // Process continuous right stick scrolling
        let stick_active = self.right_stick_y.abs() > 0.15;
        if stick_active && !self.gamepad_passthrough {
            self.handle_gamepad_right_stick(self.right_stick_y);
        }

        // Request redraw if needed
        if self.needs_redraw || self.pending_screenshot.is_some() || has_pending_captures || stick_active {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            // Use Poll to ensure redraw is processed immediately
            event_loop.set_control_flow(ControlFlow::Poll);
        } else {
            // True event-driven, zero CPU when idle
            // Gamepad events come via user_event from the gamepad thread
            // Debug commands come via user_event from the debug server
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::GamepadButton(button) => {
                self.handle_gamepad_button_pressed(button);
                self.request_redraw();
            }
            AppEvent::GamepadButtonReleased(button) => {
                self.handle_gamepad_button_released(button);
            }
            AppEvent::GamepadStick { x, y } => {
                self.handle_gamepad_stick(x, y);
            }
            AppEvent::GamepadRightStick { x: _, y } => {
                // Store stick position for continuous scrolling in about_to_wait
                self.right_stick_y = y;
            }
            AppEvent::GamepadConnected => {
                tracing::info!("Gamepad connected");
                self.gamepad_connected = true;
                if let Some(renderer) = &mut self.renderer {
                    renderer.set_gamepad_connected(true);
                }
                self.request_redraw();
            }
            AppEvent::GamepadDisconnected => {
                tracing::info!("Gamepad disconnected");
                self.gamepad_connected = false;
                if let Some(renderer) = &mut self.renderer {
                    renderer.set_gamepad_connected(false);
                }
                self.request_redraw();
            }
            AppEvent::DebugCommand(command, response_tx) => {
                let response = self.handle_debug_command(command);
                let _ = response_tx.blocking_send(response);
                self.request_redraw();
            }
            AppEvent::GetState(response_tx) => {
                let state_json = self.get_state_json();
                let _ = response_tx.send(state_json);
            }
            AppEvent::Shutdown => {
                tracing::info!("Shutdown requested via debug command");
                event_loop.exit();
            }
            AppEvent::AiToolCall(description) => {
                // Update current tool display and add to tool log
                if let AppState::PalaceLoop { current_tool, tool_log, .. } = &mut self.state {
                    *current_tool = Some(description.clone());
                    // Add to front of log (most recent first) - description already has emoji
                    tool_log.insert(0, description);
                    // Keep log reasonably sized
                    if tool_log.len() > 500 {
                        tool_log.truncate(500);
                    }
                    self.request_redraw();
                }
            }
            AppEvent::AiChatter(text) => {
                // AI thinking/commentary - add to thought log (right side)
                if let AppState::PalaceLoop { thought_log, .. } = &mut self.state {
                    thought_log.insert(0, text);
                    if thought_log.len() > 500 {
                        thought_log.truncate(500);
                    }
                    self.request_redraw();
                }
            }
            AppEvent::SuggestionStart { id } => {
                // Add new card, clear logs when first suggestion arrives
                if let AppState::PalaceLoop { cards, tool_log, thought_log, .. } = &mut self.state {
                    if cards.is_empty() {
                        // First suggestion - clear the context-gathering waterfalls
                        tool_log.clear();
                        thought_log.clear();
                    }
                    cards.push(SuggestionCard::new(id));
                    self.request_redraw();
                }
            }
            AppEvent::SuggestionUpdate { id, field, value } => {
                // Update card field
                if let AppState::PalaceLoop { cards, .. } = &mut self.state {
                    if let Some(card) = cards.iter_mut().find(|c| c.id == id) {
                        match field.as_str() {
                            "title" => card.title = value,
                            "category" => card.category = value,
                            "description" => card.description = value,
                            "command" => card.command = Some(value),
                            _ => {}
                        }
                        self.request_redraw();
                    }
                }
            }
            AppEvent::SuggestionComplete { id } => {
                // Mark card as complete (not streaming)
                if let AppState::PalaceLoop { cards, .. } = &mut self.state {
                    if let Some(card) = cards.iter_mut().find(|c| c.id == id) {
                        card.streaming = false;
                        self.request_redraw();
                    }
                }
            }
            AppEvent::SuggestionsDone => {
                // Mark generation as complete
                if let AppState::PalaceLoop { generating, current_tool, .. } = &mut self.state {
                    *generating = false;
                    *current_tool = None;
                    self.request_redraw();
                }
            }
            AppEvent::AiError(error) => {
                tracing::error!("AI Error: {}", error);
                if let AppState::PalaceLoop { generating, current_tool, .. } = &mut self.state {
                    *generating = false;
                    *current_tool = Some(format!("Error: {}", error));
                    self.request_redraw();
                }
            }
            AppEvent::PermissionRequest { id: _, command, response_tx } => {
                // Show permission modal
                tracing::info!("Permission requested: {}", command);
                let cmd_prefix = command.split_whitespace().next().unwrap_or(&command).to_string();
                // Store the sender in the app (not the state, since state derives Clone)
                self.permission_response_tx = Some(response_tx);
                self.state = AppState::PermissionModal {
                    command: command.clone(),
                    command_prefix: cmd_prefix,
                    selected_choice: 0,
                    previous_state: Box::new(self.state.clone()),
                };
                self.request_redraw();
            }
            AppEvent::ExecutionToolCall(line) => {
                // Add tool call to left column
                if let AppState::Executing { tool_log, .. } = &mut self.state {
                    tool_log.insert(0, line); // Newest at top
                    if tool_log.len() > 500 { tool_log.truncate(500); }
                    self.request_redraw();
                }
            }
            AppEvent::ExecutionThought(line) => {
                // Add thought/commentary to right column
                if let AppState::Executing { thought_log, .. } = &mut self.state {
                    thought_log.insert(0, line); // Newest at top
                    if thought_log.len() > 500 { thought_log.truncate(500); }
                    self.request_redraw();
                }
            }
            AppEvent::ExecutionProgress { current, total } => {
                // Update execution progress
                if let AppState::Executing { status, .. } = &mut self.state {
                    *status = ExecutionStatus::Running {
                        current_card: current,
                        total_cards: total,
                    };
                    self.request_redraw();
                }
            }
            AppEvent::ExecutionComplete => {
                // Mark execution as complete
                if let AppState::Executing { status, tool_log, .. } = &mut self.state {
                    *status = ExecutionStatus::Completed;
                    tool_log.insert(0, "✅ All tasks completed".to_string());
                    self.request_redraw();
                }
            }
            AppEvent::ExecutionError(error) => {
                // Mark execution as failed
                if let AppState::Executing { status, tool_log, .. } = &mut self.state {
                    *status = ExecutionStatus::Failed(error.clone());
                    tool_log.insert(0, format!("❌ Failed: {}", error));
                    self.request_redraw();
                }
            }
            AppEvent::SurveyRequest { question, header, options, multi_select, response_tx } => {
                // Show survey UI - store response channel and transition to Survey state
                tracing::info!("Survey request: {}", question);
                self.state = AppState::Survey {
                    question,
                    header,
                    options,
                    focused_index: 0,
                    custom_input: String::new(),
                    custom_active: false,
                    multi_select,
                    selected_indices: Vec::new(),
                    // Quick-select only for single-select mode (disabled when user navigates)
                    use_quick_select: !multi_select,
                    scroll_offset: 0,
                    previous_state: Box::new(self.state.clone()),
                    response_tx: Some(response_tx),
                };
                self.request_redraw();
            }
        }
    }
}

impl App {
    /// Get current state as JSON for restart transfer
    fn get_state_json(&self) -> String {
        use crate::state::AppState;

        match &self.state {
            AppState::ProjectChooser { selected_index } => {
                serde_json::json!({
                    "view": "chooser",
                    "selected": selected_index
                })
            }
            AppState::ProjectView {
                project_path,
                selected_action,
            } => {
                serde_json::json!({
                    "view": "project",
                    "project_path": project_path.to_string_lossy(),
                    "selected": selected_action
                })
            }
            AppState::MainMenu {
                selected_item,
                previous_state,
            } => {
                // Serialize the main menu AND the previous state
                let prev = match previous_state.as_ref() {
                    AppState::ProjectChooser { selected_index } => {
                        serde_json::json!({
                            "view": "chooser",
                            "selected": selected_index
                        })
                    }
                    AppState::ProjectView {
                        project_path,
                        selected_action,
                    } => {
                        serde_json::json!({
                            "view": "project",
                            "project_path": project_path.to_string_lossy(),
                            "selected": selected_action
                        })
                    }
                    _ => serde_json::json!({"view": "chooser", "selected": 0}),
                };
                serde_json::json!({
                    "view": "main_menu",
                    "selected": selected_item,
                    "previous": prev
                })
            }
            AppState::SettingsMenu {
                selected_item,
                previous_state,
            } => {
                // Serialize the settings menu AND the previous state
                let prev = match previous_state.as_ref() {
                    AppState::ProjectChooser { selected_index } => {
                        serde_json::json!({
                            "view": "chooser",
                            "selected": selected_index
                        })
                    }
                    AppState::ProjectView {
                        project_path,
                        selected_action,
                    } => {
                        serde_json::json!({
                            "view": "project",
                            "project_path": project_path.to_string_lossy(),
                            "selected": selected_action
                        })
                    }
                    _ => serde_json::json!({"view": "chooser", "selected": 0}),
                };
                serde_json::json!({
                    "view": "settings",
                    "selected": selected_item,
                    "previous": prev
                })
            }
            AppState::PalaceLoop {
                project_path,
                focused_index,
                ..
            } => {
                serde_json::json!({
                    "view": "palace_loop",
                    "project_path": project_path.to_string_lossy(),
                    "focused": focused_index
                })
            }
            AppState::UiScaleMenu {
                selected_item,
                ..
            } => {
                // For UI scale menu, just return a simple state - we'll go back to settings on restart
                serde_json::json!({
                    "view": "ui_scale",
                    "selected": selected_item
                })
            }
            AppState::PermissionModal { .. } => {
                // Permission modal is transient - just return to chooser on restart
                serde_json::json!({
                    "view": "chooser",
                    "selected": 0
                })
            }
            AppState::ExecuteModal { .. } => {
                // Execute modal is transient - just return to chooser on restart
                serde_json::json!({
                    "view": "chooser",
                    "selected": 0
                })
            }
            AppState::Survey { .. } => {
                // Survey is transient - just return to chooser on restart
                serde_json::json!({
                    "view": "chooser",
                    "selected": 0
                })
            }
            AppState::Executing { project_path, .. } => {
                // Executing is transient - return to project view on restart
                serde_json::json!({
                    "view": "project",
                    "project_path": project_path.to_string_lossy(),
                    "selected": 0
                })
            }
        }
        .to_string()
    }

    /// Run AI suggestion generation in a background thread
    fn run_ai_suggestions(project_path: PathBuf, proxy: Arc<EventLoopProxy<AppEvent>>) {
        use crate::ai::{ProjectContext, SuggestionEngine, SuggestionEvent};

        tracing::info!("Starting AI suggestions for {:?}", project_path);

        // Gather project context
        let context = match ProjectContext::gather(&project_path) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("Failed to gather project context: {}", e);
                let _ = proxy.send_event(AppEvent::AiError(format!("Failed to gather context: {}", e)));
                return;
            }
        };

        // Create suggestion engine
        let engine = match SuggestionEngine::new(None, None) {
            Ok(e) => e,
            Err(e) => {
                tracing::error!("Failed to create suggestion engine: {}", e);
                let _ = proxy.send_event(AppEvent::AiError(format!("Failed to create engine: {}", e)));
                return;
            }
        };

        // Define callback that sends events to the main thread
        let callback = {
            let proxy = proxy.clone();
            move |event: SuggestionEvent| {
                let app_event = match event {
                    SuggestionEvent::ToolCall(desc) => AppEvent::AiToolCall(desc),
                    SuggestionEvent::Chatter(text) => AppEvent::AiChatter(text),
                    SuggestionEvent::CardStart { id } => AppEvent::SuggestionStart { id },
                    SuggestionEvent::CardUpdate { id, field, value } => {
                        AppEvent::SuggestionUpdate { id, field, value }
                    }
                    SuggestionEvent::CardComplete { id } => AppEvent::SuggestionComplete { id },
                    SuggestionEvent::Done => AppEvent::SuggestionsDone,
                    SuggestionEvent::Error(e) => AppEvent::AiError(e),
                };
                let _ = proxy.send_event(app_event);
            }
        };

        // Create permission requester that sends events to the GUI
        let permission_proxy = proxy.clone();
        let permission_requester: crate::ai::PermissionRequester = Box::new(move |command: &str| {
            // Create oneshot channel for response
            let (tx, rx) = tokio::sync::oneshot::channel::<PermissionResponse>();

            // Send permission request to GUI
            static PERMISSION_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let id = PERMISSION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            if permission_proxy.send_event(AppEvent::PermissionRequest {
                id,
                command: command.to_string(),
                response_tx: tx,
            }).is_err() {
                return PermissionResponse::Denied; // Event loop closed
            }

            // Block waiting for user response
            match rx.blocking_recv() {
                Ok(response) => response,
                Err(_) => PermissionResponse::Denied, // Channel closed = deny
            }
        });

        // Run streaming suggestions
        if let Err(e) = engine.stream_to_gui(&context, callback, Some(permission_requester)) {
            tracing::error!("AI suggestion error: {}", e);
            let _ = proxy.send_event(AppEvent::AiError(e.to_string()));
        }

        // Signal completion
        let _ = proxy.send_event(AppEvent::SuggestionsDone);
    }
}

/// Testable gamepad focus state machine (no window dependencies)
#[derive(Debug, Default)]
pub struct GamepadFocusTracker {
    /// Whether we're in passthrough mode (Palace ignores gamepad)
    pub passthrough: bool,
    /// L3 (LeftThumb) is currently held
    pub l3_held: bool,
    /// R3 (RightThumb) is currently held
    pub r3_held: bool,
}

/// Result of processing a gamepad button
#[derive(Debug, PartialEq)]
pub enum FocusAction {
    /// No focus change needed
    None,
    /// Release focus (minimize window, enter passthrough)
    Release,
    /// Recapture focus (restore window, exit passthrough)
    Recapture,
    /// Forward to normal input handling
    Forward(Button),
}

impl GamepadFocusTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a button press, returns action to take
    pub fn button_pressed(&mut self, button: Button) -> FocusAction {
        match button {
            Button::LeftThumb => {
                self.l3_held = true;
                // L3+R3 combo while in passthrough = recapture
                if self.r3_held && self.passthrough {
                    self.passthrough = false;
                    return FocusAction::Recapture;
                }
            }
            Button::RightThumb => {
                self.r3_held = true;
                // L3+R3 combo while in passthrough = recapture
                if self.l3_held && self.passthrough {
                    self.passthrough = false;
                    return FocusAction::Recapture;
                }
            }
            _ => {}
        }

        // In passthrough mode, ignore all buttons except the L3+R3 combo (handled above)
        if self.passthrough {
            return FocusAction::None;
        }

        // L3 alone (without R3) = release focus
        if button == Button::LeftThumb && !self.r3_held {
            self.passthrough = true;
            return FocusAction::Release;
        }

        // Forward other buttons to normal handling
        FocusAction::Forward(button)
    }

    /// Process a button release
    pub fn button_released(&mut self, button: Button) {
        match button {
            Button::LeftThumb => self.l3_held = false,
            Button::RightThumb => self.r3_held = false,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gilrs::Button;

    #[test]
    fn test_initial_state() {
        let tracker = GamepadFocusTracker::new();
        assert!(!tracker.passthrough);
        assert!(!tracker.l3_held);
        assert!(!tracker.r3_held);
    }

    #[test]
    fn test_l3_releases_focus() {
        let mut tracker = GamepadFocusTracker::new();

        // Press L3 alone
        let action = tracker.button_pressed(Button::LeftThumb);

        assert_eq!(action, FocusAction::Release);
        assert!(tracker.passthrough);
        assert!(tracker.l3_held);
    }

    #[test]
    fn test_passthrough_ignores_buttons() {
        let mut tracker = GamepadFocusTracker::new();
        tracker.passthrough = true;

        // Regular buttons should be ignored
        assert_eq!(tracker.button_pressed(Button::South), FocusAction::None);
        assert_eq!(tracker.button_pressed(Button::DPadUp), FocusAction::None);
        assert_eq!(tracker.button_pressed(Button::Start), FocusAction::None);
    }

    #[test]
    fn test_l3_r3_combo_recaptures() {
        let mut tracker = GamepadFocusTracker::new();
        tracker.passthrough = true;

        // Press L3 first
        let action = tracker.button_pressed(Button::LeftThumb);
        assert_eq!(action, FocusAction::None); // Just L3, not combo yet

        // Press R3 while L3 held
        let action = tracker.button_pressed(Button::RightThumb);
        assert_eq!(action, FocusAction::Recapture);
        assert!(!tracker.passthrough);
    }

    #[test]
    fn test_r3_l3_combo_recaptures() {
        let mut tracker = GamepadFocusTracker::new();
        tracker.passthrough = true;

        // Press R3 first
        let action = tracker.button_pressed(Button::RightThumb);
        assert_eq!(action, FocusAction::None);

        // Press L3 while R3 held
        let action = tracker.button_pressed(Button::LeftThumb);
        assert_eq!(action, FocusAction::Recapture);
        assert!(!tracker.passthrough);
    }

    #[test]
    fn test_button_release_tracking() {
        let mut tracker = GamepadFocusTracker::new();

        tracker.button_pressed(Button::LeftThumb);
        assert!(tracker.l3_held);

        tracker.button_released(Button::LeftThumb);
        assert!(!tracker.l3_held);

        tracker.button_pressed(Button::RightThumb);
        assert!(tracker.r3_held);

        tracker.button_released(Button::RightThumb);
        assert!(!tracker.r3_held);
    }

    #[test]
    fn test_normal_buttons_forward() {
        let mut tracker = GamepadFocusTracker::new();

        assert_eq!(
            tracker.button_pressed(Button::South),
            FocusAction::Forward(Button::South)
        );
        assert_eq!(
            tracker.button_pressed(Button::DPadUp),
            FocusAction::Forward(Button::DPadUp)
        );
        assert_eq!(
            tracker.button_pressed(Button::Start),
            FocusAction::Forward(Button::Start)
        );
    }

    #[test]
    fn test_full_focus_cycle() {
        let mut tracker = GamepadFocusTracker::new();

        // Start: not in passthrough
        assert!(!tracker.passthrough);

        // L3 to release
        assert_eq!(
            tracker.button_pressed(Button::LeftThumb),
            FocusAction::Release
        );
        assert!(tracker.passthrough);

        // Release L3
        tracker.button_released(Button::LeftThumb);

        // Buttons ignored in passthrough
        assert_eq!(tracker.button_pressed(Button::South), FocusAction::None);

        // L3+R3 to recapture
        tracker.button_pressed(Button::LeftThumb);
        assert_eq!(
            tracker.button_pressed(Button::RightThumb),
            FocusAction::Recapture
        );
        assert!(!tracker.passthrough);

        // Normal button handling restored
        tracker.button_released(Button::LeftThumb);
        tracker.button_released(Button::RightThumb);
        assert_eq!(
            tracker.button_pressed(Button::South),
            FocusAction::Forward(Button::South)
        );
    }

    #[test]
    fn test_l3_while_not_passthrough_and_r3_held() {
        let mut tracker = GamepadFocusTracker::new();

        // Hold R3 first (not in passthrough)
        tracker.button_pressed(Button::RightThumb);
        assert!(tracker.r3_held);

        // L3 while R3 held but NOT in passthrough - should release (L3 alone behavior)
        // Actually, since R3 is held, it's not "L3 alone", so it should forward
        let action = tracker.button_pressed(Button::LeftThumb);
        // When not in passthrough and L3+R3, no special action
        assert_eq!(action, FocusAction::Forward(Button::LeftThumb));
    }
}
