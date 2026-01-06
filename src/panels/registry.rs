//! Panel registry for managing active panel instances

use super::{GridPosition, Panel, PanelBounds};
use std::collections::HashMap;

/// Unique identifier for panel instances
pub type PanelId = u64;

/// Manages active panel instances and their layout
pub struct PanelRegistry {
    /// Active panel instances
    panels: HashMap<PanelId, Box<dyn Panel>>,
    /// Panel grid positions
    positions: HashMap<PanelId, GridPosition>,
    /// Next panel ID to assign
    next_id: PanelId,
    /// Currently focused panel
    focused: Option<PanelId>,
    /// Grid dimensions
    grid_columns: u32,
    grid_rows: u32,
    /// Layout configuration
    gap: f32,
    margin: f32,
}

impl Default for PanelRegistry {
    fn default() -> Self {
        Self::new(8, 6) // Default 8x6 grid for 16:9
    }
}

impl PanelRegistry {
    /// Create a new panel registry with specified grid dimensions
    pub fn new(columns: u32, rows: u32) -> Self {
        Self {
            panels: HashMap::new(),
            positions: HashMap::new(),
            next_id: 1,
            focused: None,
            grid_columns: columns,
            grid_rows: rows,
            gap: 10.0,
            margin: 20.0,
        }
    }

    /// Create registry with grid dimensions based on aspect ratio
    pub fn for_aspect_ratio(aspect: f32) -> Self {
        let (columns, rows) = match aspect {
            a if a > 2.5 => (12, 6), // 32:9 super ultrawide
            a if a > 2.0 => (10, 6), // 21:9 ultrawide
            a if a < 1.4 => (6, 6),  // 4:3 or portrait
            _ => (8, 6),             // 16:9 standard
        };
        Self::new(columns, rows)
    }

    /// Set gap between cells
    pub fn set_gap(&mut self, gap: f32) {
        self.gap = gap;
    }

    /// Set margin around grid
    pub fn set_margin(&mut self, margin: f32) {
        self.margin = margin;
    }

    /// Get grid dimensions
    pub fn grid_size(&self) -> (u32, u32) {
        (self.grid_columns, self.grid_rows)
    }

    /// Add a panel at a specific grid position
    ///
    /// Returns the assigned panel ID, or None if position invalid/occupied
    pub fn add(&mut self, panel: Box<dyn Panel>, position: GridPosition) -> Option<PanelId> {
        // Validate position fits in grid
        if !position.fits_in_grid(self.grid_columns, self.grid_rows) {
            return None;
        }

        // Check for overlaps with existing panels
        for (_, existing_pos) in &self.positions {
            if position.overlaps(existing_pos) {
                return None;
            }
        }

        // Check minimum size requirements
        let (min_w, min_h) = panel.min_size();
        if position.width < min_w || position.height < min_h {
            return None;
        }

        let id = self.next_id;
        self.next_id += 1;

        self.panels.insert(id, panel);
        self.positions.insert(id, position);

        // Focus first panel added
        if self.focused.is_none() {
            self.focused = Some(id);
        }

        Some(id)
    }

    /// Remove a panel by ID
    ///
    /// Returns true if panel was removed, false if not found or not closeable
    pub fn remove(&mut self, id: PanelId) -> bool {
        if let Some(panel) = self.panels.get(&id) {
            if !panel.closeable() {
                return false;
            }
        } else {
            return false;
        }

        self.panels.remove(&id);
        self.positions.remove(&id);

        // Clear focus if removed panel was focused
        if self.focused == Some(id) {
            self.focused = self.panels.keys().next().copied();
        }

        true
    }

    /// Move a panel to a new position
    ///
    /// Returns true if move was successful
    pub fn move_panel(&mut self, id: PanelId, new_position: GridPosition) -> bool {
        // Validate position fits in grid
        if !new_position.fits_in_grid(self.grid_columns, self.grid_rows) {
            return false;
        }

        // Check minimum size requirements
        if let Some(panel) = self.panels.get(&id) {
            let (min_w, min_h) = panel.min_size();
            if new_position.width < min_w || new_position.height < min_h {
                return false;
            }
        } else {
            return false;
        }

        // Check for overlaps (excluding self)
        for (&other_id, existing_pos) in &self.positions {
            if other_id != id && new_position.overlaps(existing_pos) {
                return false;
            }
        }

        self.positions.insert(id, new_position);
        true
    }

    /// Resize a panel
    ///
    /// Returns true if resize was successful
    pub fn resize_panel(&mut self, id: PanelId, new_width: u32, new_height: u32) -> bool {
        if let Some(pos) = self.positions.get(&id).cloned() {
            let new_pos = GridPosition {
                start: pos.start,
                width: new_width,
                height: new_height,
            };
            self.move_panel(id, new_pos)
        } else {
            false
        }
    }

    /// Set focused panel
    pub fn set_focus(&mut self, id: PanelId) {
        if self.panels.contains_key(&id) {
            self.focused = Some(id);
        }
    }

    /// Get focused panel ID
    pub fn focused(&self) -> Option<PanelId> {
        self.focused
    }

    /// Focus next panel (for L1/R1 switching)
    pub fn focus_next(&mut self) {
        let ids: Vec<_> = self.panels.keys().copied().collect();
        if ids.is_empty() {
            return;
        }

        let current_idx = self
            .focused
            .and_then(|f| ids.iter().position(|&id| id == f))
            .unwrap_or(0);

        let next_idx = (current_idx + 1) % ids.len();
        self.focused = Some(ids[next_idx]);
    }

    /// Focus previous panel
    pub fn focus_prev(&mut self) {
        let ids: Vec<_> = self.panels.keys().copied().collect();
        if ids.is_empty() {
            return;
        }

        let current_idx = self
            .focused
            .and_then(|f| ids.iter().position(|&id| id == f))
            .unwrap_or(0);

        let prev_idx = if current_idx == 0 {
            ids.len() - 1
        } else {
            current_idx - 1
        };
        self.focused = Some(ids[prev_idx]);
    }

    /// Get a panel by ID
    pub fn get(&self, id: PanelId) -> Option<&dyn Panel> {
        self.panels.get(&id).map(|p| p.as_ref())
    }

    /// Get a mutable panel by ID
    pub fn get_mut(&mut self, id: PanelId) -> Option<&mut Box<dyn Panel>> {
        self.panels.get_mut(&id)
    }

    /// Get panel position by ID
    pub fn get_position(&self, id: PanelId) -> Option<&GridPosition> {
        self.positions.get(&id)
    }

    /// Calculate pixel bounds for a panel
    pub fn get_bounds(&self, id: PanelId, screen_width: f32, screen_height: f32) -> Option<PanelBounds> {
        self.positions.get(&id).map(|pos| {
            PanelBounds::from_grid(
                pos,
                self.grid_columns,
                self.grid_rows,
                screen_width,
                screen_height,
                self.gap,
                self.margin,
            )
        })
    }

    /// Iterate over all panels with their IDs
    pub fn iter(&self) -> impl Iterator<Item = (PanelId, &dyn Panel)> {
        self.panels.iter().map(|(&id, panel)| (id, panel.as_ref()))
    }

    /// Number of active panels
    pub fn len(&self) -> usize {
        self.panels.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.panels.is_empty()
    }

    /// Find empty space for a panel of given minimum size
    ///
    /// Returns the first available position that fits, or None if no space
    pub fn find_empty_space(&self, min_width: u32, min_height: u32) -> Option<GridPosition> {
        // Simple greedy algorithm: scan grid left-to-right, top-to-bottom
        for row in 0..self.grid_rows {
            for col in 0..self.grid_columns {
                let pos = GridPosition::new(col, row, min_width, min_height);

                // Check if position fits in grid
                if !pos.fits_in_grid(self.grid_columns, self.grid_rows) {
                    continue;
                }

                // Check for overlaps
                let overlaps = self.positions.values().any(|p| pos.overlaps(p));
                if !overlaps {
                    return Some(pos);
                }
            }
        }
        None
    }

    /// Find panel at screen coordinates
    pub fn panel_at(&self, x: f32, y: f32, screen_width: f32, screen_height: f32) -> Option<PanelId> {
        for &id in self.panels.keys() {
            if let Some(bounds) = self.get_bounds(id, screen_width, screen_height) {
                if bounds.contains(x, y) {
                    return Some(id);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panels::{PanelAction, PanelInput, SharedPanelState};
    use crate::renderer::Renderer;

    // Minimal test panel
    struct TestPanel {
        name: &'static str,
        min_size: (u32, u32),
        closeable: bool,
    }

    impl TestPanel {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                min_size: (1, 1),
                closeable: true,
            }
        }

        fn with_min_size(mut self, w: u32, h: u32) -> Self {
            self.min_size = (w, h);
            self
        }

        fn not_closeable(mut self) -> Self {
            self.closeable = false;
            self
        }
    }

    impl super::super::Panel for TestPanel {
        fn panel_type(&self) -> &'static str {
            self.name
        }

        fn display_name(&self) -> &'static str {
            self.name
        }

        fn min_size(&self) -> (u32, u32) {
            self.min_size
        }

        fn render(&self, _: &mut Renderer, _: &PanelBounds, _: &SharedPanelState) {}

        fn handle_input(&mut self, _: PanelInput) -> PanelAction {
            PanelAction::None
        }

        fn closeable(&self) -> bool {
            self.closeable
        }
    }

    #[test]
    fn test_registry_add_and_remove() {
        let mut registry = PanelRegistry::new(8, 6);

        let id = registry
            .add(
                Box::new(TestPanel::new("test")),
                GridPosition::new(0, 0, 2, 2),
            )
            .unwrap();

        assert_eq!(registry.len(), 1);
        assert!(registry.get(id).is_some());

        assert!(registry.remove(id));
        assert_eq!(registry.len(), 0);
        assert!(registry.get(id).is_none());
    }

    #[test]
    fn test_registry_overlap_prevention() {
        let mut registry = PanelRegistry::new(8, 6);

        // Add first panel
        let _id1 = registry
            .add(
                Box::new(TestPanel::new("panel1")),
                GridPosition::new(0, 0, 3, 3),
            )
            .unwrap();

        // Try to add overlapping panel - should fail
        let result = registry.add(
            Box::new(TestPanel::new("panel2")),
            GridPosition::new(2, 2, 2, 2),
        );
        assert!(result.is_none());

        // Add non-overlapping panel - should succeed
        let id2 = registry
            .add(
                Box::new(TestPanel::new("panel3")),
                GridPosition::new(4, 0, 2, 2),
            )
            .unwrap();
        assert!(registry.get(id2).is_some());
    }

    #[test]
    fn test_registry_bounds_check() {
        let mut registry = PanelRegistry::new(8, 6);

        // Panel extending beyond grid - should fail
        let result = registry.add(
            Box::new(TestPanel::new("too_wide")),
            GridPosition::new(6, 0, 4, 2), // extends to column 10
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_registry_min_size_enforcement() {
        let mut registry = PanelRegistry::new(8, 6);

        // Panel smaller than minimum - should fail
        let result = registry.add(
            Box::new(TestPanel::new("big").with_min_size(3, 3)),
            GridPosition::new(0, 0, 2, 2), // too small
        );
        assert!(result.is_none());

        // Panel meeting minimum - should succeed
        let id = registry
            .add(
                Box::new(TestPanel::new("big").with_min_size(3, 3)),
                GridPosition::new(0, 0, 3, 3),
            )
            .unwrap();
        assert!(registry.get(id).is_some());
    }

    #[test]
    fn test_registry_focus_cycling() {
        let mut registry = PanelRegistry::new(8, 6);

        let id1 = registry
            .add(
                Box::new(TestPanel::new("a")),
                GridPosition::new(0, 0, 2, 2),
            )
            .unwrap();
        let id2 = registry
            .add(
                Box::new(TestPanel::new("b")),
                GridPosition::new(3, 0, 2, 2),
            )
            .unwrap();

        // First panel gets focus by default
        assert_eq!(registry.focused(), Some(id1));

        // Cycle forward
        registry.focus_next();
        assert_eq!(registry.focused(), Some(id2));

        // Cycle forward wraps
        registry.focus_next();
        assert_eq!(registry.focused(), Some(id1));

        // Cycle backward
        registry.focus_prev();
        assert_eq!(registry.focused(), Some(id2));
    }

    #[test]
    fn test_registry_cannot_remove_uncloseable() {
        let mut registry = PanelRegistry::new(8, 6);

        let id = registry
            .add(
                Box::new(TestPanel::new("permanent").not_closeable()),
                GridPosition::new(0, 0, 2, 2),
            )
            .unwrap();

        assert!(!registry.remove(id));
        assert!(registry.get(id).is_some());
    }

    #[test]
    fn test_registry_find_empty_space() {
        let mut registry = PanelRegistry::new(8, 6);

        // Add panel in top-left
        registry
            .add(
                Box::new(TestPanel::new("a")),
                GridPosition::new(0, 0, 4, 3),
            )
            .unwrap();

        // Should find space to the right
        let space = registry.find_empty_space(2, 2).unwrap();
        assert!(space.start.col >= 4 || space.start.row >= 3);
    }

    #[test]
    fn test_registry_aspect_ratio_grid_sizes() {
        // 16:9 standard
        let r1 = PanelRegistry::for_aspect_ratio(16.0 / 9.0);
        assert_eq!(r1.grid_size(), (8, 6));

        // 21:9 ultrawide
        let r2 = PanelRegistry::for_aspect_ratio(21.0 / 9.0);
        assert_eq!(r2.grid_size(), (10, 6));

        // 32:9 super ultrawide
        let r3 = PanelRegistry::for_aspect_ratio(32.0 / 9.0);
        assert_eq!(r3.grid_size(), (12, 6));
    }
}
