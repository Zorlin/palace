//! Integration tests for gamepad input handling
//!
//! Tests the gamepad focus state machine and button mapping logic.

use palace::app::{FocusAction, GamepadFocusTracker};
use gilrs::Button;

// ============== GamepadFocusTracker Unit Tests ==============

#[test]
fn test_gamepad_focus_tracker_initial_state() {
    let tracker = GamepadFocusTracker::new();
    assert!(!tracker.passthrough, "Should not start in passthrough mode");
    assert!(!tracker.l3_held, "L3 should not be held initially");
    assert!(!tracker.r3_held, "R3 should not be held initially");
}

#[test]
fn test_gamepad_normal_button_forwarding() {
    let mut tracker = GamepadFocusTracker::new();

    // Most buttons should be forwarded when not in passthrough
    assert_eq!(
        tracker.button_pressed(Button::South),
        FocusAction::Forward(Button::South)
    );
    assert_eq!(
        tracker.button_pressed(Button::East),
        FocusAction::Forward(Button::East)
    );
    assert_eq!(
        tracker.button_pressed(Button::North),
        FocusAction::Forward(Button::North)
    );
    assert_eq!(
        tracker.button_pressed(Button::West),
        FocusAction::Forward(Button::West)
    );
}

#[test]
fn test_gamepad_dpad_navigation() {
    let mut tracker = GamepadFocusTracker::new();

    // D-pad buttons should forward
    assert_eq!(
        tracker.button_pressed(Button::DPadUp),
        FocusAction::Forward(Button::DPadUp)
    );
    assert_eq!(
        tracker.button_pressed(Button::DPadDown),
        FocusAction::Forward(Button::DPadDown)
    );
    assert_eq!(
        tracker.button_pressed(Button::DPadLeft),
        FocusAction::Forward(Button::DPadLeft)
    );
    assert_eq!(
        tracker.button_pressed(Button::DPadRight),
        FocusAction::Forward(Button::DPadRight)
    );
}

#[test]
fn test_gamepad_face_button_labels() {
    // Verify the button mapping matches gamepad conventions:
    // South = A (confirm)
    // East = B (cancel/back)
    // West = X
    // North = Y

    let mut tracker = GamepadFocusTracker::new();

    let south_action = tracker.button_pressed(Button::South);
    assert!(matches!(south_action, FocusAction::Forward(Button::South)));

    let east_action = tracker.button_pressed(Button::East);
    assert!(matches!(east_action, FocusAction::Forward(Button::East)));

    let west_action = tracker.button_pressed(Button::West);
    assert!(matches!(west_action, FocusAction::Forward(Button::West)));

    let north_action = tracker.button_pressed(Button::North);
    assert!(matches!(north_action, FocusAction::Forward(Button::North)));
}

#[test]
fn test_gamepad_start_button() {
    let mut tracker = GamepadFocusTracker::new();

    // Start button should forward (opens menu)
    assert_eq!(
        tracker.button_pressed(Button::Start),
        FocusAction::Forward(Button::Start)
    );
}

#[test]
fn test_gamepad_select_button() {
    let mut tracker = GamepadFocusTracker::new();

    // Select button should forward (toggles quest log)
    assert_eq!(
        tracker.button_pressed(Button::Select),
        FocusAction::Forward(Button::Select)
    );
}

// ============== Focus Release Tests ==============

#[test]
fn test_gamepad_l3_releases_focus() {
    let mut tracker = GamepadFocusTracker::new();

    // Press L3 alone (without R3)
    let action = tracker.button_pressed(Button::LeftThumb);

    assert_eq!(action, FocusAction::Release, "L3 alone should release focus");
    assert!(tracker.passthrough, "Should enter passthrough mode");
    assert!(tracker.l3_held, "L3 should be marked as held");
}

#[test]
fn test_gamepad_passthrough_ignores_buttons() {
    let mut tracker = GamepadFocusTracker::new();
    tracker.passthrough = true;

    // In passthrough mode, all buttons except L3+R3 combo should be ignored
    assert_eq!(
        tracker.button_pressed(Button::South),
        FocusAction::None,
        "South button should be ignored in passthrough"
    );
    assert_eq!(
        tracker.button_pressed(Button::East),
        FocusAction::None,
        "East button should be ignored in passthrough"
    );
    assert_eq!(
        tracker.button_pressed(Button::DPadUp),
        FocusAction::None,
        "DPad should be ignored in passthrough"
    );
    assert_eq!(
        tracker.button_pressed(Button::Start),
        FocusAction::None,
        "Start should be ignored in passthrough"
    );
}

// ============== Focus Recapture Tests ==============

#[test]
fn test_gamepad_l3_r3_combo_recaptures() {
    let mut tracker = GamepadFocusTracker::new();
    tracker.passthrough = true;

    // Press L3 first
    let action1 = tracker.button_pressed(Button::LeftThumb);
    assert_eq!(
        action1, FocusAction::None,
        "Just L3 in passthrough should do nothing"
    );
    assert!(tracker.l3_held);

    // Press R3 while L3 is held
    let action2 = tracker.button_pressed(Button::RightThumb);
    assert_eq!(
        action2, FocusAction::Recapture,
        "L3+R3 combo should recapture focus"
    );
    assert!(!tracker.passthrough, "Should exit passthrough mode");
    assert!(tracker.r3_held);
}

#[test]
fn test_gamepad_r3_l3_combo_recaptures() {
    let mut tracker = GamepadFocusTracker::new();
    tracker.passthrough = true;

    // Press R3 first
    let action1 = tracker.button_pressed(Button::RightThumb);
    assert_eq!(action1, FocusAction::None);
    assert!(tracker.r3_held);

    // Press L3 while R3 is held
    let action2 = tracker.button_pressed(Button::LeftThumb);
    assert_eq!(
        action2, FocusAction::Recapture,
        "R3+L3 combo should recapture focus"
    );
    assert!(!tracker.passthrough);
    assert!(tracker.l3_held);
}

#[test]
fn test_gamepad_l3_r3_outside_passthrough() {
    let mut tracker = GamepadFocusTracker::new();
    // NOT in passthrough mode

    // Press L3
    tracker.button_pressed(Button::LeftThumb);
    assert!(tracker.l3_held);

    // Press R3 while L3 held (but NOT in passthrough)
    let action = tracker.button_pressed(Button::RightThumb);
    // Should NOT recapture (only recaptures when in passthrough)
    assert_eq!(
        action, FocusAction::Forward(Button::RightThumb),
        "L3+R3 outside passthrough should forward R3"
    );
}

// ============== Button Release Tracking ==============

#[test]
fn test_gamepad_button_release_tracking() {
    let mut tracker = GamepadFocusTracker::new();

    // Press and release L3
    tracker.button_pressed(Button::LeftThumb);
    assert!(tracker.l3_held);

    tracker.button_released(Button::LeftThumb);
    assert!(!tracker.l3_held, "L3 should be released");

    // Press and release R3
    tracker.button_pressed(Button::RightThumb);
    assert!(tracker.r3_held);

    tracker.button_released(Button::RightThumb);
    assert!(!tracker.r3_held, "R3 should be released");
}

#[test]
fn test_gamepad_release_non_stick_buttons() {
    let mut tracker = GamepadFocusTracker::new();

    // Releasing non-stick buttons should not affect stick state
    tracker.button_released(Button::South);
    tracker.button_released(Button::East);
    tracker.button_released(Button::DPadUp);

    assert!(!tracker.l3_held);
    assert!(!tracker.r3_held);
}

// ============== Full Focus Cycle Tests ==============

#[test]
fn test_gamepad_full_focus_cycle() {
    let mut tracker = GamepadFocusTracker::new();

    // Start: not in passthrough
    assert!(!tracker.passthrough);

    // 1. Normal operation - buttons forward
    assert!(matches!(
        tracker.button_pressed(Button::South),
        FocusAction::Forward(Button::South)
    ));

    // 2. Release focus with L3
    assert_eq!(
        tracker.button_pressed(Button::LeftThumb),
        FocusAction::Release
    );
    assert!(tracker.passthrough);

    // 3. Release L3
    tracker.button_released(Button::LeftThumb);
    assert!(!tracker.l3_held);

    // 4. Buttons ignored in passthrough
    assert_eq!(
        tracker.button_pressed(Button::South),
        FocusAction::None
    );

    // 5. Recapture with L3+R3 combo
    tracker.button_pressed(Button::LeftThumb);
    assert_eq!(
        tracker.button_pressed(Button::RightThumb),
        FocusAction::Recapture
    );
    assert!(!tracker.passthrough);

    // 6. Release both sticks
    tracker.button_released(Button::LeftThumb);
    tracker.button_released(Button::RightThumb);
    assert!(!tracker.l3_held);
    assert!(!tracker.r3_held);

    // 7. Normal button handling restored
    assert!(matches!(
        tracker.button_pressed(Button::South),
        FocusAction::Forward(Button::South)
    ));
}

#[test]
fn test_gamepad_multiple_release_cycles() {
    let mut tracker = GamepadFocusTracker::new();

    // First cycle
    tracker.button_pressed(Button::LeftThumb); // Release
    assert!(tracker.passthrough);

    tracker.button_pressed(Button::LeftThumb); // L3 held
    tracker.button_pressed(Button::RightThumb); // Recapture
    assert!(!tracker.passthrough);

    tracker.button_released(Button::LeftThumb);
    tracker.button_released(Button::RightThumb);

    // Second cycle
    tracker.button_pressed(Button::LeftThumb); // Release again
    assert!(tracker.passthrough);

    tracker.button_pressed(Button::RightThumb); // R3 held
    tracker.button_pressed(Button::LeftThumb); // Recapture
    assert!(!tracker.passthrough);
}

#[test]
fn test_gamepad_partial_button_combinations() {
    let mut tracker = GamepadFocusTracker::new();

    // Test that holding one thumbstick doesn't interfere with normal operation
    tracker.button_pressed(Button::LeftThumb);
    assert!(tracker.l3_held);
    assert!(!tracker.r3_held);

    // Other buttons should still work normally
    assert!(matches!(
        tracker.button_pressed(Button::South),
        FocusAction::Forward(Button::South)
    ));

    // Release L3
    tracker.button_released(Button::LeftThumb);

    // Now hold R3
    tracker.button_pressed(Button::RightThumb);
    assert!(tracker.r3_held);
    assert!(!tracker.l3_held);

    // Other buttons should still work
    assert!(matches!(
        tracker.button_pressed(Button::East),
        FocusAction::Forward(Button::East)
    ));

    // Release R3
    tracker.button_released(Button::RightThumb);
}

#[test]
fn test_gamepad_focus_state_independence() {
    let mut tracker1 = GamepadFocusTracker::new();
    let mut tracker2 = GamepadFocusTracker::new();

    // Each tracker should maintain independent state
    tracker1.button_pressed(Button::LeftThumb);
    assert!(tracker1.passthrough);
    assert!(!tracker2.passthrough);

    tracker2.button_pressed(Button::LeftThumb);
    assert!(tracker1.passthrough);
    assert!(tracker2.passthrough);

    // Recapture in tracker1 shouldn't affect tracker2
    tracker1.button_pressed(Button::RightThumb);
    assert!(!tracker1.passthrough);
    assert!(tracker2.passthrough);
}

// ============== Edge Case Tests ==============

#[test]
fn test_gamepad_rapid_button_presses() {
    let mut tracker = GamepadFocusTracker::new();

    // Rapid button presses should all be handled
    for _ in 0..10 {
        tracker.button_pressed(Button::South);
        tracker.button_released(Button::South);
    }

    assert!(!tracker.passthrough);
    assert!(!tracker.l3_held);
    assert!(!tracker.r3_held);
}

#[test]
fn test_gamepad_stuck_button_recovery() {
    let mut tracker = GamepadFocusTracker::new();

    // Simulate stuck L3 (pressed but not released)
    tracker.button_pressed(Button::LeftThumb);
    assert!(tracker.l3_held);

    // Even with stuck L3, R3 should still work
    tracker.button_pressed(Button::RightThumb);
    assert!(tracker.r3_held);

    // Manually release to simulate hardware fix
    tracker.button_released(Button::LeftThumb);
    tracker.button_released(Button::RightThumb);

    // State should be clean
    assert!(!tracker.l3_held);
    assert!(!tracker.r3_held);
}

#[test]
fn test_gamepad_concurrent_button_presses() {
    let mut tracker = GamepadFocusTracker::new();

    // Press multiple face buttons
    let south = tracker.button_pressed(Button::South);
    let east = tracker.button_pressed(Button::East);
    let north = tracker.button_pressed(Button::North);

    // All should forward independently
    assert!(matches!(south, FocusAction::Forward(Button::South)));
    assert!(matches!(east, FocusAction::Forward(Button::East)));
    assert!(matches!(north, FocusAction::Forward(Button::North)));
}

#[test]
fn test_gamepad_passthrough_with_combo_detection() {
    let mut tracker = GamepadFocusTracker::new();
    tracker.passthrough = true;

    // In passthrough, single L3 or R3 should do nothing
    let action1 = tracker.button_pressed(Button::LeftThumb);
    assert_eq!(action1, FocusAction::None);

    tracker.button_released(Button::LeftThumb);

    let action2 = tracker.button_pressed(Button::RightThumb);
    assert_eq!(action2, FocusAction::None);

    // Only the combo should trigger recapture
    tracker.button_pressed(Button::LeftThumb);
    let combo_action = tracker.button_pressed(Button::RightThumb);
    assert_eq!(combo_action, FocusAction::Recapture);
}
