mod keyboard;
mod gamepad;

use crate::config::Config;
use anyhow::Result;
use crossterm::event::{self, Event as CrosstermEvent, KeyCode, KeyModifiers, MouseEventKind};
use std::time::Duration;

/// Unified input events from keyboard, mouse, and gamepad
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    Quit,
    NavigateUp,
    NavigateDown,
    ToggleSelect,
    SelectAll,
    DeselectAll,
    Execute,
    Delete,
    MouseClick { row: u16, col: u16 },
}

/// Handles input from all sources: keyboard, mouse, gamepad
pub struct InputHandler {
    #[allow(dead_code)]
    gamepad: Option<gamepad::GamepadHandler>,
    mouse_enabled: bool,
}

impl InputHandler {
    pub fn new(config: &Config) -> Result<Self> {
        let gamepad = if config.gamepad.enabled {
            match gamepad::GamepadHandler::new(config.gamepad.deadzone) {
                Ok(g) => Some(g),
                Err(e) => {
                    tracing::warn!("Gamepad init failed: {e}, continuing without gamepad");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            gamepad,
            mouse_enabled: config.ui.mouse_enabled,
        })
    }

    /// Get the next input event (async)
    pub async fn next_event(&mut self) -> Result<Option<Event>> {
        // Poll for crossterm events (keyboard/mouse)
        if event::poll(Duration::from_millis(16))? {
            match event::read()? {
                CrosstermEvent::Key(key) => {
                    return Ok(self.handle_key(key.code, key.modifiers));
                }
                CrosstermEvent::Mouse(mouse) if self.mouse_enabled => {
                    if let MouseEventKind::Down(_) = mouse.kind {
                        return Ok(Some(Event::MouseClick {
                            row: mouse.row,
                            col: mouse.column,
                        }));
                    }
                }
                _ => {}
            }
        }

        // Poll gamepad
        if let Some(ref mut gamepad) = self.gamepad {
            if let Some(event) = gamepad.poll()? {
                return Ok(Some(event));
            }
        }

        Ok(None)
    }

    fn handle_key(&self, code: KeyCode, modifiers: KeyModifiers) -> Option<Event> {
        match (code, modifiers) {
            // Quit
            (KeyCode::Char('q'), KeyModifiers::NONE) => Some(Event::Quit),
            (KeyCode::Esc, _) => Some(Event::Quit),

            // Navigation
            (KeyCode::Up, _) | (KeyCode::Char('k'), KeyModifiers::NONE) => Some(Event::NavigateUp),
            (KeyCode::Down, _) | (KeyCode::Char('j'), KeyModifiers::NONE) => {
                Some(Event::NavigateDown)
            }

            // Selection
            (KeyCode::Char(' '), _) => Some(Event::ToggleSelect),
            (KeyCode::Char('a'), KeyModifiers::CONTROL) => Some(Event::SelectAll),
            (KeyCode::Char('d'), KeyModifiers::CONTROL) => Some(Event::DeselectAll),

            // Actions
            (KeyCode::Enter, _) => Some(Event::Execute),
            (KeyCode::Delete, _) | (KeyCode::Backspace, _) => Some(Event::Delete),

            _ => None,
        }
    }
}
