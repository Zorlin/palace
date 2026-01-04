use super::Event;
use anyhow::Result;
use gilrs::{Button, EventType, Gilrs};

pub struct GamepadHandler {
    gilrs: Gilrs,
    #[allow(dead_code)]
    deadzone: f32,
    // Track trigger states for RT+LT intervention combo
    left_trigger_held: bool,
    right_trigger_held: bool,
}

impl GamepadHandler {
    pub fn new(deadzone: f32) -> Result<Self> {
        let gilrs = Gilrs::new().map_err(|e| anyhow::anyhow!("Failed to init gilrs: {e}"))?;

        // Log connected gamepads
        for (id, gamepad) in gilrs.gamepads() {
            tracing::info!(
                "Gamepad connected: {} (id: {:?})",
                gamepad.name(),
                id
            );
        }

        Ok(Self {
            gilrs,
            deadzone,
            left_trigger_held: false,
            right_trigger_held: false,
        })
    }

    pub fn poll(&mut self) -> Result<Option<Event>> {
        while let Some(gilrs::Event { event, .. }) = self.gilrs.next_event() {
            if let Some(e) = self.handle_event(event) {
                return Ok(Some(e));
            }
        }
        Ok(None)
    }

    fn handle_event(&mut self, event: EventType) -> Option<Event> {
        match event {
            EventType::ButtonPressed(button, _) => {
                // Track trigger presses
                match button {
                    Button::LeftTrigger | Button::LeftTrigger2 => {
                        self.left_trigger_held = true;
                        // Check for intervention combo
                        if self.right_trigger_held {
                            return Some(Event::Intervention);
                        }
                    }
                    Button::RightTrigger | Button::RightTrigger2 => {
                        self.right_trigger_held = true;
                        // Check for intervention combo
                        if self.left_trigger_held {
                            return Some(Event::Intervention);
                        }
                    }
                    _ => {}
                }
                self.handle_button(button)
            }
            EventType::ButtonReleased(button, _) => {
                // Track trigger releases
                match button {
                    Button::LeftTrigger | Button::LeftTrigger2 => {
                        self.left_trigger_held = false;
                    }
                    Button::RightTrigger | Button::RightTrigger2 => {
                        self.right_trigger_held = false;
                    }
                    _ => {}
                }
                None
            }
            // D-pad via axis
            EventType::AxisChanged(axis, value, _) => self.handle_axis(axis, value),
            _ => None,
        }
    }

    fn handle_button(&self, button: Button) -> Option<Event> {
        match button {
            // A button - select/deselect
            Button::South => Some(Event::ToggleSelect),

            // Y button - select all toggle
            Button::North => Some(Event::ToggleSelectAll),

            // X button - execute
            Button::West => Some(Event::Execute),

            // Start - also execute
            Button::Start => Some(Event::Execute),

            // B button (tap) - quit/back
            Button::East => Some(Event::Quit),

            // D-pad
            Button::DPadUp => Some(Event::NavigateUp),
            Button::DPadDown => Some(Event::NavigateDown),

            _ => None,
        }
    }

    fn handle_axis(&self, axis: gilrs::Axis, value: f32) -> Option<Event> {
        // Handle left stick for navigation
        match axis {
            gilrs::Axis::LeftStickY => {
                if value > 0.5 {
                    Some(Event::NavigateUp)
                } else if value < -0.5 {
                    Some(Event::NavigateDown)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Poll gamepad in dialogue mode
    pub fn poll_dialogue(&mut self) -> Result<Option<Event>> {
        while let Some(gilrs::Event { event, .. }) = self.gilrs.next_event() {
            if let Some(e) = self.handle_dialogue_event(event) {
                return Ok(Some(e));
            }
        }
        Ok(None)
    }

    fn handle_dialogue_event(&self, event: EventType) -> Option<Event> {
        match event {
            EventType::ButtonPressed(button, _) => self.handle_dialogue_button(button),
            EventType::AxisChanged(axis, value, _) => self.handle_dialogue_axis(axis, value),
            _ => None,
        }
    }

    fn handle_dialogue_button(&self, button: Button) -> Option<Event> {
        match button {
            // A button - accept/allow (in dialogue mode)
            Button::South => Some(Event::DialogueAccept),

            // X button - always allow
            Button::West => Some(Event::DialogueAlwaysAllow),

            // B button - deny/cancel
            Button::East => Some(Event::DialogueDeny),

            // Y button - toggle selection in multi-select
            Button::North => Some(Event::DialogueToggle),

            // Start - confirm selection
            Button::Start => Some(Event::DialogueConfirm),

            // D-pad navigation
            Button::DPadUp => Some(Event::DialogueUp),
            Button::DPadDown => Some(Event::DialogueDown),

            _ => None,
        }
    }

    fn handle_dialogue_axis(&self, axis: gilrs::Axis, value: f32) -> Option<Event> {
        match axis {
            gilrs::Axis::LeftStickY => {
                if value > 0.5 {
                    Some(Event::DialogueUp)
                } else if value < -0.5 {
                    Some(Event::DialogueDown)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
