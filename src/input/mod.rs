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
    ToggleExpand,   // E - toggle expanded view for focused task
    ToggleHistory,  // H - toggle between fresh tasks and history
    MouseClick { row: u16, col: u16 },
    MouseShiftClick { row: u16, col: u16 }, // Shift+click for range select

    // Dialogue events (BG3/Skyrim style)
    DialogueUp,          // Navigate dialogue options
    DialogueDown,
    DialogueAccept,      // A button - accept/allow
    DialogueAlwaysAllow, // X button - always allow this type
    DialogueDeny,        // B button - deny/cancel
    DialogueToggle,      // Space - toggle multi-select option
    DialogueConfirm,     // Enter - confirm selection

    // Special combo: RT+LT intervention (interrupt and suggest alternatives)
    Intervention,
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
            // Quit - Ctrl+C, q, or Esc
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => Some(Event::Quit),
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
            (KeyCode::Char('h'), KeyModifiers::NONE) | (KeyCode::Char('H'), _) => Some(Event::ToggleHistory),

            // Intervention: Ctrl+I or Ctrl+Shift+I
            (KeyCode::Char('i'), KeyModifiers::CONTROL) => Some(Event::Intervention),

            _ => None,
        }
    }

    /// Handle keys when in dialogue mode (permission prompts, surveys)
    fn handle_dialogue_key(&self, code: KeyCode, modifiers: KeyModifiers) -> Option<Event> {
        match (code, modifiers) {
            // Quit - Ctrl+C always works
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => Some(Event::Quit),

            // Navigation
            (KeyCode::Up, KeyModifiers::NONE) | (KeyCode::Char('k'), KeyModifiers::NONE) => {
                Some(Event::DialogueUp)
            }
            (KeyCode::Down, KeyModifiers::NONE) | (KeyCode::Char('j'), KeyModifiers::NONE) => {
                Some(Event::DialogueDown)
            }

            // Accept (A button)
            (KeyCode::Char('a'), KeyModifiers::NONE) | (KeyCode::Char('A'), _) => {
                Some(Event::DialogueAccept)
            }

            // Always allow (X button)
            (KeyCode::Char('x'), KeyModifiers::NONE) | (KeyCode::Char('X'), _) => {
                Some(Event::DialogueAlwaysAllow)
            }

            // Deny/cancel (B button)
            (KeyCode::Char('b'), KeyModifiers::NONE) | (KeyCode::Char('B'), _) => {
                Some(Event::DialogueDeny)
            }
            (KeyCode::Esc, _) => Some(Event::DialogueDeny),

            // Toggle option in multi-select
            (KeyCode::Char(' '), _) => Some(Event::DialogueToggle),

            // Confirm selection
            (KeyCode::Enter, _) => Some(Event::DialogueConfirm),

            // Intervention also works in dialogue mode
            (KeyCode::Char('i'), KeyModifiers::CONTROL) => Some(Event::Intervention),

            _ => None,
        }
    }

    /// Get next event in dialogue mode
    pub async fn next_dialogue_event(&mut self) -> Result<Option<Event>> {
        if event::poll(Duration::from_millis(16))? {
            if let CrosstermEvent::Key(key) = event::read()? {
                return Ok(self.handle_dialogue_key(key.code, key.modifiers));
            }
        }

        // Poll gamepad in dialogue mode
        if let Some(ref mut gamepad) = self.gamepad {
            if let Some(event) = gamepad.poll_dialogue()? {
                return Ok(Some(event));
            }
        }

        Ok(None)
    }
}
