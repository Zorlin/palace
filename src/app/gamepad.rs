//! Gamepad input handling and focus management
//!
//! Handles:
//! - Gamepad button press/release mapping to keyboard actions
//! - Left stick analog-to-digital navigation
//! - Right stick continuous scrolling
//! - L3/R3 focus release/recapture (passthrough mode)

use crate::state::{AppState, SuggestionCard};
use gilrs::Button;
use winit::keyboard::KeyCode;
use winit::window::Fullscreen;

use super::input::KeyboardInput;
use super::menus::MenuNavigation;

/// Gamepad focus state tracker
///
/// Manages the passthrough mode where Palace releases gamepad control
/// to other applications (e.g., for gaming), and recaptures via L3+R3 combo.
#[derive(Default)]
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
#[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a button press, returns action to take
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn button_released(&mut self, button: Button) {
        match button {
            Button::LeftThumb => self.l3_held = false,
            Button::RightThumb => self.r3_held = false,
            _ => {}
        }
    }
}

/// Gamepad input handling methods for App
pub trait GamepadInput {
    fn handle_gamepad_button_pressed(&mut self, button: Button);
    fn handle_gamepad_button_released(&mut self, button: Button);
    fn handle_gamepad_stick(&mut self, x: f32, y: f32);
    fn handle_gamepad_right_stick(&mut self, y: f32);
    fn release_gamepad_focus(&mut self);
    fn recapture_gamepad(&mut self);
    fn show_gnome_overview(&self);
}

impl GamepadInput for super::App {
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
            Button::Select => {
                // Select button = toggle quest log in Executing state
                self.handle_input(KeyCode::Tab);
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
            CardScroll { card: SuggestionCard, current: f32 },
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

        // Phase 2: Calculate max scroll (needs renderer from focused window)
        let new_offset = match &action {
            ScrollAction::CardScroll { card, current } => {
                if let Some(renderer) = self.focused_renderer_mut() {
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

        // Minimize the focused window
        if let Some(window_id) = self.focused_window {
            if let Some(palace_window) = self.windows.get(&window_id) {
                palace_window.window.set_minimized(true);
            }
        }
    }

    /// Recapture gamepad focus - restore Palace and take back gamepad control
    fn recapture_gamepad(&mut self) {
        if !self.gamepad_passthrough {
            return; // Already captured
        }

        tracing::info!("🎮 Recaptured gamepad focus");
        self.gamepad_passthrough = false;

        // Restore the focused window to fullscreen
        if let Some(window_id) = self.focused_window {
            if let Some(palace_window) = self.windows.get(&window_id) {
                let window = &palace_window.window;
                // First ensure it's visible
                window.set_visible(true);
                // Unminimize
                window.set_minimized(false);
                // Restore fullscreen
                window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                // Try to focus
                window.focus_window();
            }
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

        // L3 while R3 held but NOT in passthrough - should forward (not "L3 alone")
        let action = tracker.button_pressed(Button::LeftThumb);
        assert_eq!(action, FocusAction::Forward(Button::LeftThumb));
    }
}
