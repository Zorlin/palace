//! Touch, mouse, and cursor input handling
//!
//! Handles:
//! - Touch gestures (tap, hold-to-reveal)
//! - Mouse cursor hover detection
//! - Mouse wheel scrolling
//! - Hit testing for card positions
//! - Edit mode panel selection and resize

use crate::panels::{ResizeHandle, UIPanel, UIPanelBounds, UIPanelResizeState};
use crate::state::{AppState, ProjectAction, UiScaleOption};
use winit::event::{MouseScrollDelta, Touch, TouchPhase};
use winit::keyboard::KeyCode;

use super::input::KeyboardInput;

/// Edge detection threshold for ICS-style resize (pixels)
const EDGE_THRESHOLD: f32 = 20.0;

/// Touch and mouse input handling methods for App
pub trait TouchInput {
    fn handle_touch(&mut self, touch: Touch);
    fn card_at_position(&self, x: f32, y: f32) -> Option<usize>;
    fn handle_cursor_moved(&mut self, x: f32, y: f32);
    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta);
    fn handle_tap(&mut self, x: f32, y: f32);
    fn grid_columns(&self) -> usize;
}

impl TouchInput for super::App {
    fn grid_columns(&self) -> usize {
        // Calculate columns based on focused window size if available
        if let Some(window_id) = self.focused_window {
            if let Some(palace_window) = self.windows.get(&window_id) {
                let width = palace_window.window.inner_size().width as f32;
                // Same calculation as CardGrid: (available_width + gap) / (card_width + gap)
                let margin = 72.0; // scaled margin
                let card_width = 384.0; // scaled card width
                let gap = 28.8; // scaled gap
                let available = width - margin * 2.0;
                let cols = ((available + gap) / (card_width + gap)).floor() as usize;
                return cols.max(1);
            }
        }
        4 // Default for 1920px width
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

                // Handle edit mode resize release
                if self.edit_mode.active && self.edit_mode.ui_resize.is_some() {
                    self.handle_edit_mode_release(x, y);
                    self.touch_hold = None;
                    self.touch_hold_triggered = false;
                    return;
                }

                // Handle Primary pill drag drop (state is inside MultiDisplayDialog)
                // Get values before mutable borrow of self.state
                let ui_scale = self.focused_renderer().map(|r| r.ui_scale()).unwrap_or(1.5);
                let size = self.focused_window
                    .and_then(|id| self.windows.get(&id))
                    .map(|w| w.window.inner_size())
                    .unwrap_or_else(|| winit::dpi::PhysicalSize::new(1920, 1080));

                if let AppState::MultiDisplayDialog { primary_pill_drag, options, .. } = &mut self.state {
                    if let Some((source_idx, _, _)) = primary_pill_drag.take() {
                        // Compute drop target (same layout as handle_tap)
                        let scale = |v: f32| v * ui_scale;
                        let inner_padding = scale(24.0);
                        let card_height = scale(60.0);
                        let card_gap = scale(12.0);
                        let title_height = scale(60.0);
                        let toggle_height = scale(40.0);
                        let toggle_gap = scale(8.0);
                        let button_height = scale(44.0);

                        let modal_width = scale(500.0).min(size.width as f32 * 0.8);
                        let item_count = options.len() as f32;
                        let modal_height = title_height + inner_padding + item_count * (card_height + card_gap) + toggle_height + toggle_gap * 2.0 + button_height;
                        let modal_x = (size.width as f32 - modal_width) / 2.0;
                        let modal_y = (size.height as f32 - modal_height) / 2.0;
                        let card_start_y = modal_y + title_height;
                        let card_width = modal_width - inner_padding * 2.0;

                        // Find which card the drop landed on
                        let mut target_idx = None;
                        for i in 0..options.len() {
                            let card_y = card_start_y + i as f32 * (card_height + card_gap);
                            let card_x = modal_x + inner_padding;

                            if x >= card_x && x <= card_x + card_width
                               && y >= card_y && y <= card_y + card_height {
                                target_idx = Some(i);
                                break;
                            }
                        }

                        // If dropped on a different monitor, move primary
                        if let Some(target) = target_idx {
                            if target != source_idx {
                                tracing::info!("Dropped Primary pill on monitor {}", target);
                                for (i, opt) in options.iter_mut().enumerate() {
                                    if i == target {
                                        opt.is_primary = true;
                                        opt.enabled = true;
                                    } else {
                                        opt.is_primary = false;
                                    }
                                }
                            }
                        }
                        self.request_redraw();
                        return;
                    }
                }

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

                // Handle edit mode resize dragging
                if self.edit_mode.active && self.edit_mode.ui_resize.is_some() {
                    self.handle_edit_mode_move(x, y);
                    return;
                }

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
            // Get window size from focused window
            let window_size = self.focused_window
                .and_then(|id| self.windows.get(&id))
                .map(|w| w.window.inner_size());
            let Some(size) = window_size else { return None };

            // Must match CardGrid::for_palace_loop() in gpu.rs
            let ui_scale = self.focused_renderer().map(|r| r.ui_scale()).unwrap_or(1.0);
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
        // Store cursor position for click handling
        self.last_cursor_position = Some((x, y));

        // Handle edit mode resize dragging
        if self.edit_mode.active && self.edit_mode.ui_resize.is_some() {
            self.handle_edit_mode_move(x, y);
            return;
        }

        // Handle Primary pill drag in MultiDisplayDialog
        if let AppState::MultiDisplayDialog { primary_pill_drag, .. } = &mut self.state {
            if let Some((source_idx, _, _)) = *primary_pill_drag {
                // Update drag position
                *primary_pill_drag = Some((source_idx, x, y));
                self.hover_changed = true;
                return;
            }
        }

        // Get window info before mutable state borrow
        let window_info = self.focused_window
            .and_then(|id| self.windows.get(&id))
            .map(|w| (w.window.inner_size(), w.renderer.ui_scale()));
        let Some((size, ui_scale)) = window_info else { return };

        // Only handle hover in PalaceLoop state
        if let AppState::PalaceLoop { cards, hovered_index, card_scroll_offset, .. } = &mut self.state {
            let scale = |v: f32| v * ui_scale;

            // Dynamic column calculation
            let base_card_width = 237.0;
            let base_gap = 16.0;
            let base_margin = 40.0;
            let screen_width = size.width as f32;

            let card_width = scale(base_card_width);
            let gap = scale(base_gap);
            let margin_x = scale(base_margin);
            let margin_y = scale(base_margin + 80.0); // Extra space for title + subtitle

            // Calculate columns dynamically
            let available_width = screen_width - margin_x * 2.0;
            let columns = ((available_width + gap) / (card_width + gap)).floor() as usize;
            let columns = columns.max(1);
            let card_height = card_width * 0.6; // Same aspect ratio as renderer

            // Find which card (if any) the cursor is over
            let mut new_hovered = None;
            for (i, _card) in cards.iter().enumerate() {
                let col = i % columns;
                let row = i / columns;
                let card_x = margin_x + col as f32 * (card_width + gap);
                let card_y = margin_y + row as f32 * (card_height + gap) - *card_scroll_offset;

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

    /// Handle mouse wheel scrolling
    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        // Extract scroll amount (positive = scroll up, negative = scroll down)
        let scroll_amount = match delta {
            MouseScrollDelta::LineDelta(_, y) => y * 50.0, // Lines to pixels
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
        };

        // Pre-compute layout values to avoid borrow issues
        let row_height = self.palace_loop_row_height();
        let columns = self.palace_loop_columns();
        let ui_scale = self.focused_renderer().map(|r| r.ui_scale()).unwrap_or(1.5);
        let screen_height = self.focused_window
            .and_then(|id| self.windows.get(&id))
            .map(|w| w.window.inner_size().height as f32)
            .unwrap_or(1080.0);
        let margin_y = (40.0 + 80.0) * ui_scale;
        let visible_height = screen_height - margin_y;

        // Handle scroll in PalaceLoop state
        if let AppState::PalaceLoop { cards, card_scroll_offset, .. } = &mut self.state {
            let total_rows = (cards.len() + columns - 1) / columns;
            let content_height = total_rows as f32 * row_height;
            let max_scroll = (content_height - visible_height).max(0.0);

            // Apply scroll (inverted: wheel up = scroll up = decrease offset)
            *card_scroll_offset = (*card_scroll_offset - scroll_amount).clamp(0.0, max_scroll);
        }
    }

    fn handle_tap(&mut self, x: f32, y: f32) {
        // Handle edit mode taps first (panel selection / handle clicks)
        if self.edit_mode.active {
            // Refresh ui_panels from renderer before handling tap
            if let Some(renderer) = self.focused_renderer() {
                let panels = renderer.compute_ui_panels(&self.state);
                self.edit_mode.ui_panels = panels
                    .into_iter()
                    .map(|(id, px, py, w, h)| UIPanel {
                        id,
                        bounds: UIPanelBounds { x: px, y: py, width: w, height: h },
                    })
                    .collect();
            }
            self.handle_edit_mode_tap(x, y);
            return;
        }

        // Extract window info before mutable state borrow
        let window_info = self.focused_window
            .and_then(|id| self.windows.get(&id))
            .map(|w| (w.window.inner_size(), w.renderer.ui_scale()));
        let Some((size, ui_scale)) = window_info else { return };
        let columns = self.grid_columns();

        match &mut self.state {
            AppState::ProjectChooser { selected_index, .. } => {
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
                card_scroll_offset,
                ..
            } => {
                // Tap on suggestion cards to toggle selection
                // Must match CardGrid::for_palace_loop() in gpu.rs
                let scale = |v: f32| v * ui_scale;

                // Dynamic column calculation (same as palace_loop_columns)
                let base_card_width = 237.0;
                let base_gap = 16.0;
                let base_margin = 40.0;
                let screen_width = size.width as f32;

                let card_width = scale(base_card_width);
                let gap = scale(base_gap);
                let margin_x = scale(base_margin);
                let margin_y = scale(base_margin + 80.0); // Extra space for title + subtitle

                // Calculate columns dynamically
                let available_width = screen_width - margin_x * 2.0;
                let columns = ((available_width + gap) / (card_width + gap)).floor() as usize;
                let columns = columns.max(1);
                let card_height = card_width * 0.6; // Same aspect ratio as renderer

                for (i, _card) in cards.iter().enumerate() {
                    let col = i % columns;
                    let row = i / columns;
                    let card_x = margin_x + col as f32 * (card_width + gap);
                    let card_y = margin_y + row as f32 * (card_height + gap) - *card_scroll_offset;

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
                // No touchscreen actions during execution
            }
            AppState::AddCardMenu { .. } => {
                // TODO: Handle menu touches
            }
            AppState::CustomTaskInput { .. } => {
                // TODO: Handle text input touches
            }
            AppState::MultiDisplayDialog {
                focus_index,
                options,
                remember_choice,
                focused_row,
                primary_pill_drag,
                ..
            } => {
                // Handle clicks on display dialog elements
                // Per-monitor: click toggle area to enable/disable, click Primary pill to drag
                let scale = |v: f32| v * ui_scale;

                let inner_padding = scale(24.0);
                let card_height = scale(60.0);
                let card_gap = scale(12.0);
                let title_height = scale(60.0);
                let toggle_height = scale(40.0);
                let toggle_gap = scale(8.0);
                let button_height = scale(44.0);

                let modal_width = scale(500.0).min(size.width as f32 * 0.8);
                let item_count = options.len() as f32;
                // Adjusted: no extend toggle, just remember + apply
                let modal_height = title_height + inner_padding + item_count * (card_height + card_gap) + toggle_height + toggle_gap * 2.0 + button_height;
                let modal_x = (size.width as f32 - modal_width) / 2.0;
                let modal_y = (size.height as f32 - modal_height) / 2.0;
                let card_start_y = modal_y + title_height;
                let card_width = modal_width - inner_padding * 2.0;

                // Define hit areas within each card
                let toggle_box_width = scale(48.0);
                let pill_width = scale(80.0);

                // Check monitor cards
                for i in 0..options.len() {
                    let card_y = card_start_y + i as f32 * (card_height + card_gap);
                    let card_x = modal_x + inner_padding;

                    if y >= card_y && y <= card_y + card_height {
                        // Check toggle box (left side) - toggles enabled
                        if x >= card_x && x <= card_x + toggle_box_width {
                            tracing::info!("Clicked toggle for monitor {}", i);
                            if let Some(opt) = options.get_mut(i) {
                                if opt.is_primary {
                                    tracing::info!("Cannot disable primary - set another as primary first");
                                } else {
                                    opt.enabled = !opt.enabled;
                                }
                            }
                            *focus_index = i;
                            *focused_row = 0;
                            self.request_redraw();
                            return;
                        }

                        // Check Primary pill (right side) - start drag or set primary
                        let pill_x = card_x + card_width - pill_width - scale(8.0);
                        if x >= pill_x && x <= pill_x + pill_width {
                            if options.get(i).map(|o| o.is_primary).unwrap_or(false) {
                                // Start dragging the Primary pill
                                tracing::info!("Started dragging Primary pill from monitor {}", i);
                                *primary_pill_drag = Some((i, x, y));
                            } else {
                                // Click on non-primary = set as primary
                                tracing::info!("Set monitor {} as primary", i);
                                for (j, opt) in options.iter_mut().enumerate() {
                                    if j == i {
                                        opt.is_primary = true;
                                        opt.enabled = true;
                                    } else {
                                        opt.is_primary = false;
                                    }
                                }
                            }
                            *focus_index = i;
                            *focused_row = 0;
                            self.request_redraw();
                            return;
                        }

                        // Click elsewhere on card = just focus
                        *focus_index = i;
                        *focused_row = 0;
                        self.request_redraw();
                        return;
                    }
                }

                // Check remember toggle
                let remember_y = card_start_y + item_count * (card_height + card_gap) + toggle_gap;
                if x >= modal_x + inner_padding && x <= modal_x + inner_padding + card_width
                   && y >= remember_y - scale(4.0) && y <= remember_y + scale(36.0) {
                    tracing::info!("Clicked remember toggle");
                    *remember_choice = !*remember_choice;
                    *focused_row = 1;
                    self.request_redraw();
                    return;
                }

                // Check apply button
                let apply_y = remember_y + toggle_height + toggle_gap;
                let button_width = scale(120.0);
                let button_x = modal_x + (modal_width - button_width) / 2.0;
                if x >= button_x && x <= button_x + button_width
                   && y >= apply_y - scale(4.0) && y <= apply_y + scale(36.0) {
                    tracing::info!("Clicked apply button");
                    *focused_row = 2;
                    // Trigger Enter key to apply
                    self.handle_input(KeyCode::Enter);
                    return;
                }
            }
            AppState::ProjectContextMenu { .. } => {
                // TODO: Handle context menu touches
            }
            AppState::LanguageSelector { .. } => {
                // TODO: Handle language selector touches
            }
            AppState::NewMonitorDialog { focus_index, .. } => {
                // Simple dialog with 3 options: Yes, No, Never
                let scale = |v: f32| v * ui_scale;

                let modal_width = scale(400.0).min(size.width as f32 * 0.8);
                let modal_height = scale(200.0);
                let modal_x = (size.width as f32 - modal_width) / 2.0;
                let modal_y = (size.height as f32 - modal_height) / 2.0;

                let button_height = scale(40.0);
                let button_gap = scale(12.0);
                let button_start_y = modal_y + scale(80.0);

                // Check which button was tapped
                for i in 0..3 {
                    let button_y = button_start_y + i as f32 * (button_height + button_gap);
                    if y >= button_y && y <= button_y + button_height
                       && x >= modal_x + scale(24.0) && x <= modal_x + modal_width - scale(24.0) {
                        *focus_index = i;
                        self.handle_input(KeyCode::Enter);
                        return;
                    }
                }
            }
        }
    }
}

/// ICS-style edge detection: detects which edge/handle the point is on
/// Returns Some(handle) if on an edge, None if inside or outside the panel
fn detect_edge_handle(x: f32, y: f32, bounds: &UIPanelBounds) -> Option<ResizeHandle> {
    let left = bounds.x;
    let right = bounds.right();
    let top = bounds.y;
    let bottom = bounds.bottom();

    // Check if within extended bounds (panel bounds + threshold)
    let in_x_range = x >= left - EDGE_THRESHOLD && x <= right + EDGE_THRESHOLD;
    let in_y_range = y >= top - EDGE_THRESHOLD && y <= bottom + EDGE_THRESHOLD;
    if !in_x_range || !in_y_range {
        return None;
    }

    // Detect edges (ICS style - entire edge is grabbable)
    let on_left = x >= left - EDGE_THRESHOLD && x <= left + EDGE_THRESHOLD;
    let on_right = x >= right - EDGE_THRESHOLD && x <= right + EDGE_THRESHOLD;
    let on_top = y >= top - EDGE_THRESHOLD && y <= top + EDGE_THRESHOLD;
    let on_bottom = y >= bottom - EDGE_THRESHOLD && y <= bottom + EDGE_THRESHOLD;

    // Must be within the panel's extent (plus threshold) to count as on that edge
    let within_horizontal = x >= left - EDGE_THRESHOLD && x <= right + EDGE_THRESHOLD;
    let within_vertical = y >= top - EDGE_THRESHOLD && y <= bottom + EDGE_THRESHOLD;

    // Corners first (where edges overlap)
    if on_top && on_left && within_horizontal && within_vertical {
        return Some(ResizeHandle::TopLeft);
    }
    if on_top && on_right && within_horizontal && within_vertical {
        return Some(ResizeHandle::TopRight);
    }
    if on_bottom && on_left && within_horizontal && within_vertical {
        return Some(ResizeHandle::BottomLeft);
    }
    if on_bottom && on_right && within_horizontal && within_vertical {
        return Some(ResizeHandle::BottomRight);
    }

    // Edges (only if within the panel's extent on the perpendicular axis)
    if on_left && within_vertical {
        return Some(ResizeHandle::Left);
    }
    if on_right && within_vertical {
        return Some(ResizeHandle::Right);
    }
    if on_top && within_horizontal {
        return Some(ResizeHandle::Top);
    }
    if on_bottom && within_horizontal {
        return Some(ResizeHandle::Bottom);
    }

    None
}

/// Edit mode touch handling methods
impl super::App {
    /// Handle taps in edit mode (panel selection and resize handle detection)
    pub(crate) fn handle_edit_mode_tap(&mut self, x: f32, y: f32) {
        // Clone ui_panels to avoid borrow issues
        let panels: Vec<UIPanel> = self.edit_mode.ui_panels.clone();

        tracing::debug!(
            "Edit mode tap at ({:.0}, {:.0}), {} panels available",
            x, y, panels.len()
        );

        // First, check if tap is on any panel's edge (for resize)
        for panel in &panels {
            // Debug: show panel bounds
            let b = &panel.bounds;
            tracing::debug!(
                "  Checking '{}': x={:.0}-{:.0}, y={:.0}-{:.0}",
                panel.id, b.x, b.x + b.width, b.y, b.y + b.height
            );

            if let Some(handle) = detect_edge_handle(x, y, &panel.bounds) {
                tracing::info!(
                    "Edit mode: Starting resize of '{}' from {:?} handle at ({:.0}, {:.0})",
                    panel.id, handle, x, y
                );

                // Start resize operation
                self.edit_mode.ui_resize = Some(UIPanelResizeState::new(
                    panel.id,
                    handle,
                    panel.bounds,
                    x,
                    y,
                ));
                self.edit_mode.selected_panel = Some(panel.id);
                self.request_redraw();
                return;
            }
        }

        // Not on an edge - check if tap is inside any panel (for selection)
        for panel in &panels {
            if panel.bounds.contains(x, y) {
                tracing::info!(
                    "Edit mode: Selected panel '{}' at ({:.0}, {:.0})",
                    panel.id, x, y
                );
                self.edit_mode.selected_panel = Some(panel.id);
                self.request_redraw();
                return;
            }
        }

        // Tap outside all panels - deselect
        if self.edit_mode.selected_panel.is_some() {
            tracing::info!("Edit mode: Deselected panel (tap outside)");
            self.edit_mode.selected_panel = None;
            self.request_redraw();
        }
    }

    /// Handle touch/mouse move during edit mode (for resize dragging)
    pub(crate) fn handle_edit_mode_move(&mut self, x: f32, y: f32) {
        if let Some(ref mut resize) = self.edit_mode.ui_resize {
            let delta_x = x - resize.start_pos.0;
            let delta_y = y - resize.start_pos.1;
            tracing::debug!(
                "Edit mode move: panel='{}' delta=({:.0}, {:.0})",
                resize.panel_id, delta_x, delta_y
            );
            resize.current_pos = (x, y);

            // Compute reflow for displaced panels
            if let Some(window_id) = self.focused_window {
                if let Some(palace_window) = self.windows.get(&window_id) {
                    let size = palace_window.window.inner_size();
                    let screen_width = size.width as f32;
                    let screen_height = size.height as f32;
                    self.edit_mode.compute_reflow(
                        &mut self.panel_layout,
                        screen_width,
                        screen_height,
                    );
                }
            }

            self.request_redraw();
        }
    }

    /// Handle touch/mouse release during edit mode (finish resize)
    pub(crate) fn handle_edit_mode_release(&mut self, _x: f32, _y: f32) {
        // Check if reflow solution was valid (must check before taking ui_resize)
        let reflow_valid = self.edit_mode.reflow_solution
            .as_ref()
            .map(|s| s.valid)
            .unwrap_or(true);

        if reflow_valid {
            // Apply resize to panel_overrides BEFORE taking ui_resize
            // This persists the new position for the resized panel and any displaced panels
            self.edit_mode.apply_resize_to_overrides();
        }

        if let Some(resize) = self.edit_mode.ui_resize.take() {
            let delta_x = resize.current_pos.0 - resize.start_pos.0;
            let delta_y = resize.current_pos.1 - resize.start_pos.1;

            if reflow_valid {
                tracing::info!(
                    "Edit mode: Finished resize of '{}' - delta: ({:.0}, {:.0}), applied to overrides",
                    resize.panel_id, delta_x, delta_y
                );

                // Log override count
                tracing::info!(
                    "  Panel overrides now contains {} panels",
                    self.edit_mode.panel_overrides.len()
                );
            } else {
                tracing::warn!(
                    "Edit mode: Resize of '{}' blocked - no valid reflow solution",
                    resize.panel_id
                );
            }

            // Clear reflow state (previews already applied to overrides)
            self.edit_mode.clear_reflow();
            self.request_redraw();
        }
    }
}
