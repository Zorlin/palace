//! Menu state management and navigation
//!
//! Handles:
//! - Main menu (Resume, Settings, Exit)
//! - Settings menu
//! - UI scale menu
//! - Fullscreen toggling
//! - Menu navigation (back, toggle)

use crate::state::{AppState, UiScaleOption};
use winit::window::Fullscreen;

/// Menu navigation methods for App
pub trait MenuNavigation {
    fn toggle_fullscreen(&mut self);
    fn open_main_menu(&mut self);
    fn go_back_from_menu(&mut self);
    fn get_current_scale_index(&self) -> usize;
    fn toggle_main_menu(&mut self);
}

impl MenuNavigation for super::App {
    fn toggle_fullscreen(&mut self) {
        if let Some(window_id) = self.focused_window {
            if let Some(palace_window) = self.windows.get(&window_id) {
                if self.is_fullscreen {
                    palace_window.window.set_fullscreen(None);
                    tracing::info!("Switched to windowed mode");
                } else {
                    palace_window.window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                    tracing::info!("Switched to fullscreen mode");
                }
                self.is_fullscreen = !self.is_fullscreen;
                self.request_redraw();
            }
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
        let _current_scale = self.focused_renderer()
            .map(|r| r.ui_scale())
            .unwrap_or(1.0);
        let current_option = UiScaleOption::from_setting(self.user_scale_override);
        UiScaleOption::all().iter()
            .position(|o| *o == current_option)
            .unwrap_or(1) // Default to 100% (index 1)
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
}
