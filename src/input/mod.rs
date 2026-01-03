mod keyboard;
mod gamepad;

use crate::config::Config;
use anyhow::Result;
use crossterm::event::{self, Event as CrosstermEvent, KeyCode, KeyModifiers, MouseButton, MouseEventKind};
use std::time::Duration;

/// Unified input events from keyboard, mouse, and gamepad
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    Quit,
    NavigateUp,
    NavigateDown,
    PageUp,
    PageDown,
    JumpUp,      // Ctrl+Up - jump 5 items
    JumpDown,    // Ctrl+Down - jump 5 items
    SelectUp,    // Shift+Up - extend selection upward
    SelectDown,  // Shift+Down - extend selection downward
    Home,
    End,
    ToggleSelect,
    ToggleSelectAll,  // A - toggle all selected/deselected
    Execute,
    Delete,
    ToggleExpand,  // E - toggle expanded view for focused task
    MouseClick { row: u16, col: u16 },
    MouseShiftClick { row: u16, col: u16 }, // Shift+click for range select
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
                    if let MouseEventKind::Down(MouseButton::Left) = mouse.kind {
                        if mouse.modifiers.contains(KeyModifiers::SHIFT) {
                            return Ok(Some(Event::MouseShiftClick {
                                row: mouse.row,
                                col: mouse.column,
                            }));
                        } else {
                            return Ok(Some(Event::MouseClick {
                                row: mouse.row,
                                col: mouse.column,
                            }));
                        }
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

            // Navigation - single step
            (KeyCode::Up, KeyModifiers::NONE) | (KeyCode::Char('k'), KeyModifiers::NONE) => {
                Some(Event::NavigateUp)
            }
            (KeyCode::Down, KeyModifiers::NONE) | (KeyCode::Char('j'), KeyModifiers::NONE) => {
                Some(Event::NavigateDown)
            }

            // Navigation - jump (Ctrl+Up/Down)
            (KeyCode::Up, KeyModifiers::CONTROL) => Some(Event::JumpUp),
            (KeyCode::Down, KeyModifiers::CONTROL) => Some(Event::JumpDown),

            // Selection extension (Shift+Up/Down)
            (KeyCode::Up, KeyModifiers::SHIFT) => Some(Event::SelectUp),
            (KeyCode::Down, KeyModifiers::SHIFT) => Some(Event::SelectDown),

            // Navigation - page
            (KeyCode::PageUp, _) => Some(Event::PageUp),
            (KeyCode::PageDown, _) => Some(Event::PageDown),

            // Navigation - home/end
            (KeyCode::Home, _) | (KeyCode::Char('g'), KeyModifiers::NONE) => Some(Event::Home),
            (KeyCode::End, _) | (KeyCode::Char('G'), KeyModifiers::SHIFT) => Some(Event::End),

            // Selection
            (KeyCode::Char(' '), _) => Some(Event::ToggleSelect),
            (KeyCode::Char('a'), KeyModifiers::NONE) | (KeyCode::Char('A'), _) => Some(Event::ToggleSelectAll),

            // Actions
            (KeyCode::Enter, _) => Some(Event::Execute),
            (KeyCode::Delete, _) | (KeyCode::Backspace, _) => Some(Event::Delete),
            (KeyCode::Char('e'), KeyModifiers::NONE) => Some(Event::ToggleExpand),

            _ => None,
        }
    }
}
