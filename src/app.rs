use crate::debug::{DebugCommand, DebugResponse, ScreenshotCapture};
use crate::projects::ProjectsConfig;
use crate::renderer::Renderer;
use crate::state::AppState;
use gilrs::Button;
use std::path::PathBuf;
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
}

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
}

impl App {
    pub fn new(initial_state: AppState, projects: ProjectsConfig) -> Self {
        Self {
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
        }
    }

    /// Request a redraw on the next frame
    fn request_redraw(&mut self) {
        self.needs_redraw = true;
        if let Some(window) = &self.window {
            window.request_redraw();
        }
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
            AppState::ProjectView { selected_action, .. } => {
                let action_count = crate::state::ProjectAction::all().len();

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
                        let action = crate::state::ProjectAction::all()[*selected_action];
                        tracing::info!("Selected action: {:?}", action);
                        // TODO: Execute action
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
                                // TODO: Cycle through scale options
                                tracing::info!("UI Scale toggle (not implemented)");
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
            }
            TouchPhase::Ended => {
                tracing::info!("Touch ended at ({:.0}, {:.0}), id: {:?}", x, y, touch.id);
                // Treat touch end as a tap - find what was touched
                self.handle_tap(x, y);
            }
            TouchPhase::Moved => {
                // Could implement swipe gestures here
                tracing::debug!("Touch moved to ({:.0}, {:.0})", x, y);
            }
            TouchPhase::Cancelled => {
                tracing::debug!("Touch cancelled");
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
                if key == KeyCode::Escape {
                    tracing::info!("Escape pressed, exiting...");
                    event_loop.exit();
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

        // Request redraw if needed
        if self.needs_redraw || self.pending_screenshot.is_some() || has_pending_captures {
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
        }
        .to_string()
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
