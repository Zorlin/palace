//! winit ApplicationHandler implementation
//!
//! Handles:
//! - Window lifecycle (resumed, close)
//! - Window events (resize, redraw, input)
//! - User events from background threads

use crate::display::DisplayScaling;
use crate::palace_window::PalaceWindow;
use crate::renderer::SharedGpuResources;
use crate::state::{AppState, ExecutionStatus, SuggestionCard, TaskStatus};

use super::debug::DebugCommands;
use super::gamepad::GamepadInput;
use super::input::KeyboardInput;
use super::menus::MenuNavigation;
use super::touch::TouchInput;
use super::{App, AppEvent};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowId;

impl ApplicationHandler<AppEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        use crate::state::DisplaySettings;
        use super::PREF_DISPLAY_SETTINGS;

        // Skip if already initialized
        if !self.windows.is_empty() {
            return;
        }

        // Load saved preferences (including UI scale)
        self.load_preferences();

        tracing::info!("Initializing Palace multi-window system...");

        // Create shared GPU resources (device, queue) - done once
        if self.shared_gpu.is_none() {
            match pollster::block_on(SharedGpuResources::new()) {
                Ok(shared) => {
                    tracing::info!("Shared GPU resources initialized");
                    self.shared_gpu = Some(shared);
                }
                Err(e) => {
                    tracing::error!("Failed to create shared GPU resources: {}", e);
                    event_loop.exit();
                    return;
                }
            }
        }

        let shared_gpu = self.shared_gpu.as_ref().unwrap();

        // Initialize known monitors FIRST
        self.known_monitors = event_loop
            .available_monitors()
            .filter_map(|m| m.name())
            .collect();
        tracing::info!(
            "Detected {} monitors: {:?}",
            self.known_monitors.len(),
            self.known_monitors
        );

        // Load saved display settings to know which monitors to create windows on
        let saved: DisplaySettings = self.db
            .as_ref()
            .and_then(|db| db.get_pref::<DisplaySettings>(PREF_DISPLAY_SETTINGS).ok())
            .flatten()
            .unwrap_or_default();

        tracing::info!("Saved display settings: enabled={:?}, primary={:?}",
            saved.enabled, saved.primary);

        // Determine which monitors to create windows on
        let monitors_to_use: Vec<String> = if saved.enabled.is_empty() {
            // No saved settings - just use first available monitor
            self.known_monitors.first().cloned().into_iter().collect()
        } else {
            // Use saved enabled monitors that still exist
            saved.enabled.iter()
                .filter(|name| self.known_monitors.contains(name))
                .cloned()
                .collect()
        };

        // Determine which is primary
        let primary_monitor = saved.primary
            .filter(|p| monitors_to_use.contains(p))
            .or_else(|| monitors_to_use.first().cloned());

        tracing::info!("Will create windows on: {:?}, primary: {:?}",
            monitors_to_use, primary_monitor);

        // Create windows on all enabled monitors - PRIMARY FIRST
        let scaling_config = DisplayScaling::load();

        // Sort so primary is first
        let mut sorted_monitors = monitors_to_use.clone();
        if let Some(ref primary) = primary_monitor {
            sorted_monitors.sort_by(|a, b| {
                if a == primary { std::cmp::Ordering::Less }
                else if b == primary { std::cmp::Ordering::Greater }
                else { std::cmp::Ordering::Equal }
            });
        }

        for monitor_name in sorted_monitors {
            let target_monitor = event_loop
                .available_monitors()
                .find(|m| m.name().as_deref() == Some(&monitor_name));

            match PalaceWindow::new(event_loop, target_monitor, shared_gpu, &self.virtual_viewport) {
                Ok(mut palace_window) => {
                    let window_id = palace_window.id();
                    let actual_name = palace_window.monitor_name.clone();
                    tracing::info!("Palace window created on {}", actual_name);

                    // Detect and apply display scaling
                    let size = palace_window.window.inner_size();
                    let detected = scaling_config.get_scale_for_display(&actual_name, size.width, size.height);
                    if self.detected_scale == 0.0 {
                        self.detected_scale = detected;
                    }

                    // Use user override if set, otherwise auto-detected
                    let scale = self.user_scale_override.unwrap_or(detected);
                    let is_auto = self.user_scale_override.is_none();
                    palace_window.renderer.set_ui_scale(scale, is_auto);

                    // Set initial gamepad connection state
                    palace_window.renderer.set_gamepad_connected(self.gamepad_connected);

                    // Store the window
                    self.windows.insert(window_id, palace_window);

                    // Set as focused if this is the primary
                    if primary_monitor.as_ref() == Some(&actual_name) {
                        tracing::info!("Setting {} as primary (focused)", actual_name);
                        self.focused_window = Some(window_id);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to create Palace window on {}: {}", monitor_name, e);
                }
            }
        }

        // Ensure we have at least one focused window
        if self.focused_window.is_none() {
            self.focused_window = self.windows.keys().next().copied();
        }

        if self.windows.is_empty() {
            tracing::error!("Failed to create any windows");
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                // Remove the window
                self.windows.remove(&window_id);

                // If this was the focused window, clear focus
                if self.focused_window == Some(window_id) {
                    self.focused_window = self.windows.keys().next().copied();
                }

                // Exit if no windows left
                if self.windows.is_empty() {
                    tracing::info!("Last window closed, exiting");
                    event_loop.exit();
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ElementState::Pressed,
                        text,
                        ..
                    },
                ..
            } => {
                // Alt+F toggles fullscreen (works in any state)
                if key == KeyCode::KeyF && self.modifiers.state().alt_key() {
                    self.toggle_fullscreen();
                    return;
                }

                // Alt+E extends to all monitors
                if key == KeyCode::KeyE && self.modifiers.state().alt_key() {
                    let _ = self.event_proxy.send_event(AppEvent::ExtendToAllMonitors);
                    return;
                }

                // Escape opens the main menu (instead of exiting)
                if key == KeyCode::Escape {
                    self.open_main_menu();
                    self.request_redraw();
                } else {
                    // Handle text input for custom survey input
                    if let AppState::Survey { custom_active: true, custom_input, .. } = &mut self.state {
                        if let Some(ref txt) = text {
                            if !txt.is_empty() && !txt.chars().next().map(|c| c.is_control()).unwrap_or(true) {
                                custom_input.push_str(txt);
                                self.request_redraw();
                                return;
                            }
                        }
                    }
                    // Handle text input for custom task input
                    if let AppState::CustomTaskInput { name, description, active_field, cursor, .. } = &mut self.state {
                        if let Some(ref txt) = text {
                            if !txt.is_empty() && !txt.chars().next().map(|c| c.is_control()).unwrap_or(true) {
                                let active_text = if *active_field == 0 { name } else { description };
                                active_text.insert_str(*cursor, txt.as_str());
                                *cursor += txt.len();
                                self.request_redraw();
                                return;
                            }
                        }
                    }
                    // Handle text input for diff viewer custom feedback
                    if let AppState::ScenarioDiffViewer { feedback_mode: true, custom_feedback: Some(ref mut feedback_text), custom_cursor, .. } = &mut self.state {
                        if let Some(ref txt) = text {
                            if !txt.is_empty() && !txt.chars().next().map(|c| c.is_control()).unwrap_or(true) {
                                feedback_text.insert_str(*custom_cursor, txt.as_str());
                                *custom_cursor += txt.len();
                                self.request_redraw();
                                return;
                            }
                        }
                    }
                    // Handle text input for generator description input
                    if let AppState::ScenarioGenerator { generator, .. } = &mut self.state {
                        use crate::scenario::GeneratorState;
                        if let GeneratorState::DescriptionInput { text: input_text, cursor, .. } = generator.current_state_mut() {
                            if let Some(ref txt) = text {
                                if !txt.is_empty() && !txt.chars().next().map(|c| c.is_control()).unwrap_or(true) {
                                    input_text.insert_str(*cursor, txt.as_str());
                                    *cursor += txt.len();
                                    self.request_redraw();
                                    return;
                                }
                            }
                        }
                    }
                    self.handle_input(key);
                    self.request_redraw();
                }
            }

            WindowEvent::Resized(new_size) => {
                if let Some(palace_window) = self.windows.get_mut(&window_id) {
                    palace_window.resize(new_size);
                }
                self.request_redraw();
            }

            WindowEvent::Touch(touch) => {
                self.handle_touch(touch);
                self.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.handle_cursor_moved(position.x as f32, position.y as f32);
                if self.hover_changed {
                    self.hover_changed = false;
                    self.request_redraw();
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                self.handle_mouse_wheel(delta);
                self.request_redraw();
            }

            WindowEvent::MouseInput { state, button, .. } => {
                use winit::event::MouseButton;
                use crate::panels::{UIPanel, UIPanelBounds};

                if button == MouseButton::Left {
                    match state {
                        ElementState::Pressed => {
                            // Handle left click in edit mode - use last known cursor position
                            if self.edit_mode.active {
                                if let Some(pos) = self.last_cursor_position {
                                    // Refresh ui_panels from renderer before handling tap
                                    if let Some(renderer) = self.focused_renderer() {
                                        let panels = renderer.compute_ui_panels(&self.state);
                                        self.edit_mode.ui_panels = panels
                                            .into_iter()
                                            .map(|(id, x, y, w, h)| UIPanel {
                                                id,
                                                bounds: UIPanelBounds { x, y, width: w, height: h },
                                            })
                                            .collect();
                                        // Apply any saved overrides to the panel positions
                                        self.edit_mode.apply_overrides_to_ui_panels();
                                    }
                                    self.handle_edit_mode_tap(pos.0, pos.1);
                                    self.request_redraw();
                                }
                            }
                        }
                        ElementState::Released => {
                            // Handle mouse button release for edit mode resize or drag
                            if self.edit_mode.active {
                                if let Some(ref resize) = self.edit_mode.ui_resize {
                                    let (x, y) = resize.current_pos;
                                    self.handle_edit_mode_release(x, y);
                                    self.request_redraw();
                                } else if let Some(ref drag) = self.edit_mode.ui_drag {
                                    let (x, y) = drag.current_pos;
                                    self.handle_edit_mode_release(x, y);
                                    self.request_redraw();
                                }
                            }
                        }
                    }
                }
            }

            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers;
            }

            WindowEvent::Focused(focused) => {
                self.window_focused = focused;
                if focused {
                    self.focused_window = Some(window_id);
                }
                tracing::debug!("Window {:?} focus changed: {}", window_id, focused);
            }

            WindowEvent::RedrawRequested => {
                // Clear redraw flag
                self.needs_redraw = false;

                tracing::debug!("RedrawRequested for window {:?}", window_id);

                // Get the window for this redraw
                if let Some(palace_window) = self.windows.get_mut(&window_id) {
                    tracing::debug!("  Rendering on monitor: {}", palace_window.monitor_name);
                    palace_window.needs_redraw = false;

                    // Poll pending screenshot captures
                    palace_window.renderer.poll_screenshots(&mut self.screenshot_capture);

                    // Check if we need to capture a screenshot
                    let screenshot_path = self.pending_screenshot.take();

                    // Update snap animations (applies animated positions to panel_overrides)
                    self.edit_mode.update_snap_animations();

                    // Merge panel_overrides (persisted) with reflow_previews (active drag)
                    // This ensures panels stay at their saved positions after resize completes
                    let mut panel_positions = self.edit_mode.panel_overrides.clone();
                    // Active reflow previews override saved positions during drag
                    for (&id, &pos) in &self.edit_mode.reflow_previews {
                        panel_positions.insert(id, pos);
                    }

                    // Compute resize preview bounds for the actively resizing panel
                    // This handles all handle types correctly (including Top/Left which move x/y)
                    let resize_preview = if let Some(ref resize) = self.edit_mode.ui_resize {
                        let delta_x = resize.current_pos.0 - resize.start_pos.0;
                        let delta_y = resize.current_pos.1 - resize.start_pos.1;
                        let (new_x, new_y, new_w, new_h) = self.edit_mode.apply_resize_delta_pub(
                            resize.original_bounds,
                            resize.handle,
                            delta_x,
                            delta_y,
                        );
                        // Add resize preview to panel_positions so renderer draws it correctly
                        panel_positions.insert(resize.panel_id, (new_x, new_y, new_w, new_h));
                        None // No separate resize_preview needed - it's in panel_positions
                    } else {
                        None
                    };

                    // Compute drag preview bounds for the actively dragging panel
                    if let Some(ref drag) = self.edit_mode.ui_drag {
                        let (new_x, new_y) = drag.current_pos;
                        let orig = drag.original_bounds;
                        // Add drag preview to panel_positions so renderer draws it at new position
                        panel_positions.insert(drag.panel_id, (new_x, new_y, orig.width, orig.height));
                    }

                    // Build panel chooser state if open
                    let panel_chooser = if self.edit_mode.panel_chooser_open {
                        self.edit_mode.spawn_target.map(|pos| (pos, self.edit_mode.chooser_selection))
                    } else {
                        None
                    };

                    match palace_window.renderer.render_with_screenshot(
                        &self.state,
                        &self.projects,
                        screenshot_path.as_ref(),
                        &mut self.screenshot_capture,
                        self.edit_mode.active,
                        self.edit_mode.selected_panel,
                        resize_preview,
                        &panel_positions,
                        panel_chooser,
                    ) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            let size = palace_window.window.inner_size();
                            palace_window.renderer.resize(size);
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

                // Check if exit was requested via menu
                if self.exit_requested {
                    event_loop.exit();
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Check for new monitors (lightweight - just compares string lists)
        self.check_for_new_monitors(event_loop);

        // Check if we have pending screenshot captures that need GPU polling
        let has_pending_captures = self.screenshot_capture.has_pending();

        // Process continuous right stick scrolling
        let stick_active = self.right_stick_y.abs() > 0.15;
        if stick_active && !self.gamepad_passthrough {
            self.handle_gamepad_right_stick(self.right_stick_y);
        }

        // Check if KITT scanner animation should be running
        let animation_active = matches!(
            &self.state,
            AppState::Executing { request_active: true, .. }
        );

        // Edit mode has pulsing animations and snap animations
        let edit_mode_active = self.edit_mode.active || self.edit_mode.has_active_animations();

        // Recording indicator has pulsing animation (only when not paused)
        let recording_active = matches!(
            &self.state,
            AppState::Recording { paused: false, .. }
        );

        // Request redraw if needed
        if self.needs_redraw || self.pending_screenshot.is_some() || has_pending_captures || stick_active || animation_active || edit_mode_active || recording_active {
            for palace_window in self.windows.values() {
                palace_window.window.request_redraw();
            }
            event_loop.set_control_flow(ControlFlow::Poll);
        } else {
            // True event-driven, zero CPU when idle
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
                self.right_stick_y = y;
            }
            AppEvent::GamepadConnected => {
                tracing::info!("Gamepad connected");
                self.gamepad_connected = true;
                if let Some(renderer) = self.focused_renderer_mut() {
                    renderer.set_gamepad_connected(true);
                }
                self.request_redraw();
            }
            AppEvent::GamepadDisconnected => {
                tracing::info!("Gamepad disconnected");
                self.gamepad_connected = false;
                if let Some(renderer) = self.focused_renderer_mut() {
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
                if let AppState::PalaceLoop { current_tool, tool_log, .. } = &mut self.state {
                    *current_tool = Some(description.clone());
                    tool_log.insert(0, description);
                    if tool_log.len() > 500 {
                        tool_log.truncate(500);
                    }
                    self.request_redraw();
                }
            }
            AppEvent::AiChatter(text) => {
                if let AppState::PalaceLoop { thought_log, .. } = &mut self.state {
                    thought_log.insert(0, text);
                    if thought_log.len() > 500 {
                        thought_log.truncate(500);
                    }
                    self.request_redraw();
                }
            }
            AppEvent::SuggestionStart { id } => {
                if let AppState::PalaceLoop { cards, tool_log, thought_log, .. } = &mut self.state {
                    if cards.is_empty() {
                        tool_log.clear();
                        thought_log.clear();
                    }
                    cards.push(SuggestionCard::new(id));
                    self.request_redraw();
                }
            }
            AppEvent::SuggestionUpdate { id, field, value } => {
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
                if let AppState::PalaceLoop { cards, .. } = &mut self.state {
                    if let Some(card) = cards.iter_mut().find(|c| c.id == id) {
                        card.streaming = false;
                        self.request_redraw();
                    }
                }
            }
            AppEvent::SuggestionsDone => {
                // Get card count for recording before mutating state
                let card_count = match &self.state {
                    AppState::PalaceLoop { cards, .. } => cards.len(),
                    AppState::Recording { inner_state, .. } => {
                        if let AppState::PalaceLoop { cards, .. } = inner_state.as_ref() {
                            cards.len()
                        } else {
                            0
                        }
                    }
                    _ => 0,
                };

                if let AppState::PalaceLoop { generating, current_tool, .. } = &mut self.state {
                    *generating = false;
                    *current_tool = None;
                    self.request_redraw();
                }

                // Record cards received if in recording mode
                if card_count > 0 {
                    self.record_cards_received(card_count);
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
                tracing::info!("Permission requested: {}", command);
                let cmd_prefix = command.split_whitespace().next().unwrap_or(&command).to_string();
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
                if let AppState::Executing { tool_log, .. } = &mut self.state {
                    tool_log.insert(0, line);
                    if tool_log.len() > 500 { tool_log.truncate(500); }
                    self.request_redraw();
                }
            }
            AppEvent::ExecutionThought(line) => {
                if let AppState::Executing { thought_log, .. } = &mut self.state {
                    thought_log.insert(0, line);
                    if thought_log.len() > 500 { thought_log.truncate(500); }
                    self.request_redraw();
                }
            }
            AppEvent::ExecutionProgress { current, total } => {
                if let AppState::Executing { status, task_statuses, .. } = &mut self.state {
                    *status = ExecutionStatus::Running {
                        current_card: current,
                        total_cards: total,
                    };
                    if current > 0 && current - 1 < task_statuses.len() {
                        if task_statuses[current - 1] == TaskStatus::InProgress {
                            task_statuses[current - 1] = TaskStatus::Completed;
                        }
                    }
                    if current < task_statuses.len() && task_statuses[current] == TaskStatus::Pending {
                        task_statuses[current] = TaskStatus::InProgress;
                    }
                    self.request_redraw();
                }
            }
            AppEvent::ExecutionComplete => {
                if let AppState::Executing { status, task_statuses, tool_log, .. } = &mut self.state {
                    *status = ExecutionStatus::Completed;
                    for ts in task_statuses.iter_mut() {
                        if *ts == TaskStatus::InProgress {
                            *ts = TaskStatus::Completed;
                        }
                    }
                    tool_log.insert(0, "✅ All tasks completed".to_string());
                    self.request_redraw();
                }
                // Record completion if in recording mode
                self.record_execution_completed(true);
            }
            AppEvent::ExecutionError(error) => {
                if let AppState::Executing { status, task_statuses, tool_log, .. } = &mut self.state {
                    *status = ExecutionStatus::Failed(error.clone());
                    for ts in task_statuses.iter_mut() {
                        if *ts == TaskStatus::InProgress {
                            *ts = TaskStatus::Blocked;
                            break;
                        }
                    }
                    tool_log.insert(0, format!("❌ Failed: {}", error));
                    self.request_redraw();
                }
                // Record failure if in recording mode
                self.record_execution_completed(false);
            }
            AppEvent::ExecutionTokens(tokens) => {
                if let AppState::Executing { tokens_used, .. } = &mut self.state {
                    *tokens_used = tokens;
                    self.request_redraw();
                }
            }
            AppEvent::ExecutionRequestStart => {
                if let AppState::Executing { request_active, .. } = &mut self.state {
                    *request_active = true;
                    self.request_redraw();
                }
            }
            AppEvent::ExecutionRequestEnd => {
                if let AppState::Executing { request_active, .. } = &mut self.state {
                    *request_active = false;
                    self.request_redraw();
                }
            }
            AppEvent::TaskStatusUpdate { task_index, status, message } => {
                if let AppState::Executing { task_statuses, tool_log, .. } = &mut self.state {
                    if task_index < task_statuses.len() {
                        task_statuses[task_index] = status;
                        if let Some(msg) = message {
                            tool_log.insert(0, format!("📋 Task {}: {} - {}", task_index + 1, status.badge_text(), msg));
                        }
                        self.request_redraw();
                    }
                }
            }
            AppEvent::SurveyRequest { question, header, options, multi_select, response_tx } => {
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
                    use_quick_select: !multi_select,
                    scroll_offset: 0,
                    previous_state: Box::new(self.state.clone()),
                    response_tx: Some(response_tx),
                };
                self.request_redraw();
            }
            AppEvent::SwitchDisplay { monitor_name } => {
                // SwitchDisplay ONLY changes which window is primary (focused)
                // It NEVER moves windows between monitors
                tracing::info!("Setting primary display to: {}", monitor_name);

                // Find window on this monitor
                let window_on_monitor = self.windows.iter()
                    .find(|(_, w)| w.monitor_name == monitor_name)
                    .map(|(id, _)| *id);

                if let Some(window_id) = window_on_monitor {
                    tracing::info!("Focusing window on {}", monitor_name);
                    self.focused_window = Some(window_id);
                    self.request_redraw();
                } else {
                    tracing::warn!("No window on monitor '{}' to focus", monitor_name);
                }
            }
            AppEvent::CreateWindow { monitor_name } => {
                tracing::info!("Creating window on display: {}", monitor_name);

                // Check if we already have a window on this monitor
                let already_have_window = self.windows.values()
                    .any(|w| w.monitor_name == monitor_name);

                if already_have_window {
                    tracing::info!("Already have window on {}, skipping", monitor_name);
                    return;
                }

                // Find the target monitor
                let target_monitor = event_loop
                    .available_monitors()
                    .find(|m| m.name().as_deref() == Some(&monitor_name));

                if target_monitor.is_none() {
                    tracing::warn!("Monitor '{}' not found in available monitors:", monitor_name);
                    for m in event_loop.available_monitors() {
                        tracing::warn!("  Available: {:?}", m.name());
                    }
                    return; // Don't create window if we can't find the monitor
                }

                if let Some(shared_gpu) = &self.shared_gpu {
                    // Get the primary window to use as parent (for Dock/taskbar grouping)
                    let parent_window = self.focused_window
                        .and_then(|id| self.windows.get(&id))
                        .map(|pw| &pw.window);

                    let result = PalaceWindow::new_with_parent(
                        event_loop,
                        target_monitor,
                        shared_gpu,
                        &self.virtual_viewport,
                        parent_window,
                    );

                    match result {
                        Ok(mut palace_window) => {
                            let window_id = palace_window.id();
                            let this_monitor = palace_window.monitor_name.clone();

                            // Apply current UI scale
                            let scale = self.user_scale_override.unwrap_or(self.detected_scale);
                            let is_auto = self.user_scale_override.is_none();
                            palace_window.renderer.set_ui_scale(scale, is_auto);
                            palace_window.renderer.set_gamepad_connected(self.gamepad_connected);

                            // Force a resize to ensure surface is properly configured
                            let size = palace_window.window.inner_size();
                            palace_window.renderer.resize(size);

                            tracing::info!("New window created on {}", this_monitor);
                            self.windows.insert(window_id, palace_window);

                            // Check if this is the pending primary from saved settings
                            if self.pending_primary_monitor.as_ref() == Some(&this_monitor) {
                                tracing::info!("This window is the saved primary, setting focus");
                                self.focused_window = Some(window_id);
                                self.pending_primary_monitor = None;
                            }

                            // Log all windows
                            tracing::info!("Total windows after create: {}, focused: {:?}",
                                self.windows.len(), self.focused_window);
                            for (id, w) in &self.windows {
                                let is_primary = self.focused_window == Some(*id);
                                tracing::info!("  Window {:?} on {}{}", id, w.monitor_name,
                                    if is_primary { " (PRIMARY)" } else { "" });
                            }

                            self.request_redraw();
                        }
                        Err(e) => {
                            tracing::error!("Failed to create window on {}: {}", monitor_name, e);
                        }
                    }
                }
            }
            AppEvent::ExtendToAllMonitors => {
                tracing::info!("Extending to all monitors");

                // Collect monitors we don't have windows on yet
                let existing_monitors: Vec<String> = self.windows.values()
                    .map(|w| w.monitor_name.clone())
                    .collect();

                let monitors_to_add: Vec<_> = event_loop
                    .available_monitors()
                    .filter(|m| {
                        let name = m.name().unwrap_or_default();
                        !existing_monitors.contains(&name)
                    })
                    .collect();

                if monitors_to_add.is_empty() {
                    tracing::info!("Already have windows on all monitors");
                    return;
                }

                if let Some(shared_gpu) = &self.shared_gpu {
                    // Get the primary window to use as parent
                    let primary_window_id = self.focused_window
                        .or_else(|| self.windows.keys().next().copied());

                    for monitor in monitors_to_add {
                        let monitor_name = monitor.name().unwrap_or_else(|| "Unknown".to_string());

                        // Get parent reference (must re-borrow each iteration)
                        let parent_window = primary_window_id
                            .and_then(|id| self.windows.get(&id))
                            .map(|pw| &pw.window);

                        let result = PalaceWindow::new_with_parent(
                            event_loop,
                            Some(monitor),
                            shared_gpu,
                            &self.virtual_viewport,
                            parent_window,
                        );

                        match result {
                            Ok(mut palace_window) => {
                                let window_id = palace_window.id();

                                // Apply current UI scale
                                let scale = self.user_scale_override.unwrap_or(self.detected_scale);
                                let is_auto = self.user_scale_override.is_none();
                                palace_window.renderer.set_ui_scale(scale, is_auto);
                                palace_window.renderer.set_gamepad_connected(self.gamepad_connected);

                                tracing::info!("Extended to {}", palace_window.monitor_name);
                                self.windows.insert(window_id, palace_window);
                            }
                            Err(e) => {
                                tracing::error!("Failed to extend to {}: {}", monitor_name, e);
                            }
                        }
                    }

                    self.request_redraw();
                }
            }
        }
    }
}
