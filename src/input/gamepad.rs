use super::Event;
use anyhow::Result;
use gilrs::{Button, EventType, Gilrs};

pub struct GamepadHandler {
    gilrs: Gilrs,
    #[allow(dead_code)]
    deadzone: f32,
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

        Ok(Self { gilrs, deadzone })
    }

    pub fn poll(&mut self) -> Result<Option<Event>> {
        while let Some(gilrs::Event { event, .. }) = self.gilrs.next_event() {
            if let Some(e) = self.handle_event(event) {
                return Ok(Some(e));
            }
        }
        Ok(None)
    }

    fn handle_event(&self, event: EventType) -> Option<Event> {
        match event {
            EventType::ButtonPressed(button, _) => self.handle_button(button),
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
}
