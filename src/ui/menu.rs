//! Reusable menu component with vertical navigation

use winit::keyboard::KeyCode;

/// A menu item that can be focused and activated
#[derive(Debug, Clone)]
pub enum MenuItem {
    /// Simple button - activates on Enter
    Button { label: String, id: String },
    /// Toggle/checkbox - toggles on Enter
    Toggle { label: String, id: String, enabled: bool },
    /// Label only - not selectable, skipped during navigation
    Label { text: String },
    /// Separator - visual divider, not selectable
    Separator,
}

impl MenuItem {
    pub fn button(label: impl Into<String>, id: impl Into<String>) -> Self {
        MenuItem::Button { label: label.into(), id: id.into() }
    }

    pub fn toggle(label: impl Into<String>, id: impl Into<String>, enabled: bool) -> Self {
        MenuItem::Toggle { label: label.into(), id: id.into(), enabled }
    }

    pub fn text_label(text: impl Into<String>) -> Self {
        MenuItem::Label { text: text.into() }
    }

    pub fn separator() -> Self {
        MenuItem::Separator
    }

    /// Can this item be focused/selected?
    pub fn is_selectable(&self) -> bool {
        matches!(self, MenuItem::Button { .. } | MenuItem::Toggle { .. })
    }

    /// Get the ID if this item has one
    pub fn id(&self) -> Option<&str> {
        match self {
            MenuItem::Button { id, .. } | MenuItem::Toggle { id, .. } => Some(id),
            _ => None,
        }
    }

    /// Get the label/text
    pub fn label(&self) -> &str {
        match self {
            MenuItem::Button { label, .. } => label,
            MenuItem::Toggle { label, .. } => label,
            MenuItem::Label { text } => text,
            MenuItem::Separator => "",
        }
    }
}

/// Action returned from menu input handling
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MenuAction {
    /// Nothing happened
    None,
    /// Focus moved
    FocusChanged,
    /// Item was activated (Enter pressed on button)
    Activated(String),
    /// Toggle was toggled (returns new state)
    Toggled(String, bool),
    /// Menu was closed (Escape/Back)
    Closed,
    /// Custom key pressed on item (key, item_id)
    CustomKey(KeyCode, String),
}

/// A vertical menu with keyboard navigation
#[derive(Debug, Clone)]
pub struct Menu {
    items: Vec<MenuItem>,
    focus_index: usize,
}

impl Menu {
    pub fn new(items: Vec<MenuItem>) -> Self {
        let mut menu = Self { items, focus_index: 0 };
        // Ensure focus starts on a selectable item
        // Only advance if current item is not selectable
        if !menu.items.is_empty() && !menu.items[0].is_selectable() {
            menu.focus_next_selectable(true);
        }
        menu
    }

    pub fn items(&self) -> &[MenuItem] {
        &self.items
    }

    pub fn items_mut(&mut self) -> &mut [MenuItem] {
        &mut self.items
    }

    pub fn focus_index(&self) -> usize {
        self.focus_index
    }

    pub fn focused_item(&self) -> Option<&MenuItem> {
        self.items.get(self.focus_index)
    }

    pub fn focused_id(&self) -> Option<&str> {
        self.focused_item().and_then(|item| item.id())
    }

    /// Set toggle state by ID
    pub fn set_toggle(&mut self, id: &str, enabled: bool) {
        for item in &mut self.items {
            if let MenuItem::Toggle { id: item_id, enabled: item_enabled, .. } = item {
                if item_id == id {
                    *item_enabled = enabled;
                    break;
                }
            }
        }
    }

    /// Get toggle state by ID
    pub fn get_toggle(&self, id: &str) -> Option<bool> {
        for item in &self.items {
            if let MenuItem::Toggle { id: item_id, enabled, .. } = item {
                if item_id == id {
                    return Some(*enabled);
                }
            }
        }
        None
    }

    /// Move focus to next selectable item
    fn focus_next_selectable(&mut self, forward: bool) {
        if self.items.is_empty() {
            return;
        }

        let len = self.items.len();
        let mut attempts = 0;

        loop {
            if forward {
                self.focus_index = (self.focus_index + 1) % len;
            } else {
                self.focus_index = (self.focus_index + len - 1) % len;
            }

            if self.items[self.focus_index].is_selectable() {
                break;
            }

            attempts += 1;
            if attempts >= len {
                // No selectable items, just stop
                break;
            }
        }
    }

    /// Handle keyboard input, returns action
    pub fn handle_key(&mut self, key: KeyCode) -> MenuAction {
        match key {
            KeyCode::ArrowUp | KeyCode::KeyW => {
                self.focus_next_selectable(false);
                MenuAction::FocusChanged
            }
            KeyCode::ArrowDown | KeyCode::KeyS => {
                self.focus_next_selectable(true);
                MenuAction::FocusChanged
            }
            KeyCode::Enter | KeyCode::Space => {
                if let Some(item) = self.items.get_mut(self.focus_index) {
                    match item {
                        MenuItem::Button { id, .. } => {
                            MenuAction::Activated(id.clone())
                        }
                        MenuItem::Toggle { id, enabled, .. } => {
                            *enabled = !*enabled;
                            MenuAction::Toggled(id.clone(), *enabled)
                        }
                        _ => MenuAction::None,
                    }
                } else {
                    MenuAction::None
                }
            }
            KeyCode::Escape | KeyCode::Backspace => {
                MenuAction::Closed
            }
            _ => MenuAction::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_menu_navigation_skips_non_selectable() {
        let mut menu = Menu::new(vec![
            MenuItem::button("Button 1", "btn1"),
            MenuItem::text_label("Just a label"),
            MenuItem::separator(),
            MenuItem::button("Button 2", "btn2"),
            MenuItem::toggle("Toggle", "toggle", false),
        ]);

        // Should start on first selectable (btn1)
        assert_eq!(menu.focused_id(), Some("btn1"));

        // Down should skip label and separator, land on btn2
        menu.handle_key(KeyCode::ArrowDown);
        assert_eq!(menu.focused_id(), Some("btn2"));

        // Down again to toggle
        menu.handle_key(KeyCode::ArrowDown);
        assert_eq!(menu.focused_id(), Some("toggle"));

        // Down wraps to btn1
        menu.handle_key(KeyCode::ArrowDown);
        assert_eq!(menu.focused_id(), Some("btn1"));

        // Up wraps to toggle
        menu.handle_key(KeyCode::ArrowUp);
        assert_eq!(menu.focused_id(), Some("toggle"));
    }

    #[test]
    fn test_menu_toggle() {
        let mut menu = Menu::new(vec![
            MenuItem::toggle("Dark Mode", "dark", false),
            MenuItem::button("Apply", "apply"),
        ]);

        assert_eq!(menu.get_toggle("dark"), Some(false));

        // Toggle it
        let action = menu.handle_key(KeyCode::Enter);
        assert_eq!(action, MenuAction::Toggled("dark".to_string(), true));
        assert_eq!(menu.get_toggle("dark"), Some(true));

        // Toggle again
        let action = menu.handle_key(KeyCode::Enter);
        assert_eq!(action, MenuAction::Toggled("dark".to_string(), false));
        assert_eq!(menu.get_toggle("dark"), Some(false));
    }

    #[test]
    fn test_menu_button_activation() {
        let mut menu = Menu::new(vec![
            MenuItem::button("Cancel", "cancel"),
            MenuItem::button("Apply", "apply"),
        ]);

        menu.handle_key(KeyCode::ArrowDown);
        assert_eq!(menu.focused_id(), Some("apply"));

        let action = menu.handle_key(KeyCode::Enter);
        assert_eq!(action, MenuAction::Activated("apply".to_string()));
    }
}
