//! Grid layout system for panel arrangement
//!
//! The layout system manages the grid configuration and panel positions
//! independently from panel instances. This allows for:
//! - Ghost previews during drag operations
//! - Layout persistence without panel state
//! - Layout presets and switching

use super::{GridCell, GridPosition, PanelBounds, PanelId};
use std::collections::HashMap;

/// Direction hint for the reflow algorithm
/// Tells panels which way to "push" when displaced
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushDirection {
    /// Push panels to the right
    Right,
    /// Push panels to the left
    Left,
    /// Push panels down
    Down,
    /// Push panels up
    Up,
    /// Push panels outward from center (default for drag)
    Outward,
}

/// Solution for reordering displaced panels
#[derive(Debug, Clone)]
pub struct ReorderSolution {
    /// Map of panel_id -> (original_position, new_position)
    pub moves: HashMap<PanelId, (GridPosition, GridPosition)>,
    /// Whether the solution is valid (all panels fit)
    pub valid: bool,
}

impl ReorderSolution {
    pub fn new() -> Self {
        Self {
            moves: HashMap::new(),
            valid: true,
        }
    }

    /// Check if any panels need to move
    pub fn has_moves(&self) -> bool {
        !self.moves.is_empty()
    }

    /// Get the new position for a panel (if it needs to move)
    pub fn get_move(&self, id: PanelId) -> Option<&GridPosition> {
        self.moves.get(&id).map(|(_, new)| new)
    }
}

impl Default for ReorderSolution {
    fn default() -> Self {
        Self::new()
    }
}

impl PushDirection {
    /// Determine push direction from resize handle
    pub fn from_resize_handle(handle: &super::ResizeHandle) -> Self {
        use super::ResizeHandle;
        match handle {
            ResizeHandle::Right | ResizeHandle::TopRight | ResizeHandle::BottomRight => {
                PushDirection::Right
            }
            ResizeHandle::Left | ResizeHandle::TopLeft | ResizeHandle::BottomLeft => {
                PushDirection::Left
            }
            ResizeHandle::Bottom => PushDirection::Down,
            ResizeHandle::Top => PushDirection::Up,
        }
    }
}

/// Layout configuration for a display
#[derive(Debug, Clone)]
pub struct PanelLayout {
    /// Grid columns
    pub columns: u32,
    /// Grid rows
    pub rows: u32,
    /// Panel positions by ID
    positions: HashMap<PanelId, GridPosition>,
    /// Gap between cells (pixels)
    pub gap: f32,
    /// Margin around grid (pixels)
    pub margin: f32,
}

/// Target cell size in pixels for fine-grained collision detection
const TARGET_CELL_SIZE: f32 = 40.0;

impl Default for PanelLayout {
    fn default() -> Self {
        // Default to a reasonable fine grid - will be recalculated for actual screen
        Self::new(48, 27) // ~40px cells on 1920x1080
    }
}

impl PanelLayout {
    /// Create a new layout with specified grid dimensions
    pub fn new(columns: u32, rows: u32) -> Self {
        Self {
            columns,
            rows,
            positions: HashMap::new(),
            gap: 0.0,   // No gap - fine grid for pixel-accurate collision
            margin: 0.0, // No margin - use full screen
        }
    }

    /// Create layout sized for a specific screen resolution
    /// Uses ~40px cells for fine-grained collision detection
    pub fn for_screen_size(width: f32, height: f32) -> Self {
        let columns = (width / TARGET_CELL_SIZE).ceil() as u32;
        let rows = (height / TARGET_CELL_SIZE).ceil() as u32;
        Self::new(columns, rows)
    }

    /// Create layout with grid dimensions based on aspect ratio (legacy)
    pub fn for_aspect_ratio(aspect: f32) -> Self {
        let (columns, rows) = Self::grid_for_aspect(aspect);
        Self::new(columns, rows)
    }

    /// Get recommended grid dimensions for an aspect ratio (legacy - coarse grid)
    pub fn grid_for_aspect(aspect: f32) -> (u32, u32) {
        match aspect {
            a if a > 2.5 => (12, 6), // 32:9 super ultrawide
            a if a > 2.0 => (10, 6), // 21:9 ultrawide
            a if a < 1.4 => (6, 6),  // 4:3 or portrait
            _ => (8, 6),             // 16:9 standard
        }
    }

    /// Set gap between cells
    pub fn with_gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }

    /// Set margin around grid
    pub fn with_margin(mut self, margin: f32) -> Self {
        self.margin = margin;
        self
    }

    /// Get all panel positions
    pub fn positions(&self) -> &HashMap<PanelId, GridPosition> {
        &self.positions
    }

    /// Set a panel's position
    ///
    /// Returns false if position is invalid (out of bounds or overlaps)
    pub fn set_position(&mut self, id: PanelId, position: GridPosition) -> bool {
        // Check bounds
        if !position.fits_in_grid(self.columns, self.rows) {
            return false;
        }

        // Check overlaps (excluding self)
        for (&other_id, other_pos) in &self.positions {
            if other_id != id && position.overlaps(other_pos) {
                return false;
            }
        }

        self.positions.insert(id, position);
        true
    }

    /// Force set a panel's position without overlap validation
    ///
    /// Used when building collision state from existing (potentially overlapping) panels.
    /// This allows the reflow algorithm to detect and resolve overlaps.
    pub fn force_set_position(&mut self, id: PanelId, position: GridPosition) {
        self.positions.insert(id, position);
    }

    /// Remove a panel from the layout
    pub fn remove(&mut self, id: PanelId) {
        self.positions.remove(&id);
    }

    /// Clear all panel positions
    pub fn clear(&mut self) {
        self.positions.clear();
    }

    /// Get a panel's position
    pub fn get_position(&self, id: PanelId) -> Option<&GridPosition> {
        self.positions.get(&id)
    }

    /// Calculate pixel bounds for a grid position
    pub fn bounds_for(&self, pos: &GridPosition, screen_width: f32, screen_height: f32) -> PanelBounds {
        PanelBounds::from_grid(
            pos,
            self.columns,
            self.rows,
            screen_width,
            screen_height,
            self.gap,
            self.margin,
        )
    }

    /// Calculate pixel bounds for a panel by ID
    pub fn panel_bounds(&self, id: PanelId, screen_width: f32, screen_height: f32) -> Option<PanelBounds> {
        self.positions
            .get(&id)
            .map(|pos| self.bounds_for(pos, screen_width, screen_height))
    }

    /// Find empty space for a panel of given minimum size
    ///
    /// Returns the first available position that fits, or None if no space
    pub fn find_empty_space(&self, min_width: u32, min_height: u32) -> Option<GridPosition> {
        // Scan grid left-to-right, top-to-bottom
        for row in 0..self.rows {
            for col in 0..self.columns {
                let pos = GridPosition::new(col, row, min_width, min_height);

                // Check if position fits in grid
                if !pos.fits_in_grid(self.columns, self.rows) {
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
        for (&id, pos) in &self.positions {
            let bounds = self.bounds_for(pos, screen_width, screen_height);
            if bounds.contains(x, y) {
                return Some(id);
            }
        }
        None
    }

    /// Find which grid cell a screen coordinate falls into
    pub fn cell_at(&self, x: f32, y: f32, screen_width: f32, screen_height: f32) -> Option<GridCell> {
        // Check if within grid area
        if x < self.margin || y < self.margin {
            return None;
        }

        let available_width = screen_width - self.margin * 2.0;
        let available_height = screen_height - self.margin * 2.0;

        let total_gap_width = self.gap * (self.columns - 1) as f32;
        let total_gap_height = self.gap * (self.rows - 1) as f32;

        let cell_width = (available_width - total_gap_width) / self.columns as f32;
        let cell_height = (available_height - total_gap_height) / self.rows as f32;

        // Adjust position relative to margin
        let rel_x = x - self.margin;
        let rel_y = y - self.margin;

        // Calculate cell (accounting for gaps)
        let cell_with_gap_w = cell_width + self.gap;
        let cell_with_gap_h = cell_height + self.gap;

        let col = (rel_x / cell_with_gap_w).floor() as u32;
        let row = (rel_y / cell_with_gap_h).floor() as u32;

        // Bounds check
        if col >= self.columns || row >= self.rows {
            return None;
        }

        // Check if we're in the gap between cells
        let cell_x = col as f32 * cell_with_gap_w;
        let cell_y = row as f32 * cell_with_gap_h;

        if rel_x > cell_x + cell_width || rel_y > cell_y + cell_height {
            // In a gap
            return None;
        }

        Some(GridCell::new(col, row))
    }

    /// Convert pixel bounds to approximate grid position
    ///
    /// Used to convert static UI panel bounds to grid positions for reflow calculation.
    /// Returns the grid position that best approximates the pixel bounds.
    pub fn grid_position_from_pixels(
        &self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        screen_width: f32,
        screen_height: f32,
    ) -> GridPosition {
        // Available space after margins
        let available_width = screen_width - self.margin * 2.0;
        let available_height = screen_height - self.margin * 2.0;

        let total_gap_width = self.gap * (self.columns - 1) as f32;
        let total_gap_height = self.gap * (self.rows - 1) as f32;

        let cell_width = (available_width - total_gap_width) / self.columns as f32;
        let cell_height = (available_height - total_gap_height) / self.rows as f32;

        let cell_with_gap_w = cell_width + self.gap;
        let cell_with_gap_h = cell_height + self.gap;

        // Convert position to grid coordinates
        let rel_x = (x - self.margin).max(0.0);
        let rel_y = (y - self.margin).max(0.0);

        let start_col = (rel_x / cell_with_gap_w).floor() as u32;
        let start_row = (rel_y / cell_with_gap_h).floor() as u32;

        // Calculate how many cells the width/height spans
        let grid_width = ((width + self.gap) / cell_with_gap_w).ceil().max(1.0) as u32;
        let grid_height = ((height + self.gap) / cell_with_gap_h).ceil().max(1.0) as u32;

        // Clamp to grid bounds
        let start_col = start_col.min(self.columns.saturating_sub(1));
        let start_row = start_row.min(self.rows.saturating_sub(1));
        let grid_width = grid_width.min(self.columns - start_col);
        let grid_height = grid_height.min(self.rows - start_row);

        GridPosition::new(start_col, start_row, grid_width, grid_height)
    }

    /// Check if a position is valid (fits in grid, no overlaps with existing positions except `exclude_id`)
    pub fn is_valid_position(&self, pos: &GridPosition, exclude_id: Option<PanelId>) -> bool {
        if !pos.fits_in_grid(self.columns, self.rows) {
            return false;
        }

        for (&other_id, other_pos) in &self.positions {
            if Some(other_id) == exclude_id {
                continue;
            }
            if pos.overlaps(other_pos) {
                return false;
            }
        }

        true
    }

    /// Get all grid cells that are occupied
    pub fn occupied_cells(&self) -> Vec<GridCell> {
        let mut cells = Vec::new();
        for pos in self.positions.values() {
            cells.extend(pos.cells());
        }
        cells
    }

    /// Get all grid cells that are empty
    pub fn empty_cells(&self) -> Vec<GridCell> {
        let occupied = self.occupied_cells();
        let mut empty = Vec::new();

        for row in 0..self.rows {
            for col in 0..self.columns {
                let cell = GridCell::new(col, row);
                if !occupied.contains(&cell) {
                    empty.push(cell);
                }
            }
        }
        empty
    }

    /// Number of panels in the layout
    pub fn panel_count(&self) -> usize {
        self.positions.len()
    }

    /// Total cells in the grid
    pub fn total_cells(&self) -> u32 {
        self.columns * self.rows
    }

    /// Cells currently in use
    pub fn used_cells(&self) -> u32 {
        self.positions.values().map(|p| p.cells_covered()).sum()
    }

    /// Fraction of grid in use (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        self.used_cells() as f32 / self.total_cells() as f32
    }

    /// Find a reorder solution for panel resize/move
    ///
    /// ICS-style algorithm: computes where ALL affected panels need to move
    /// when one panel claims new territory.
    ///
    /// # Arguments
    /// * `resizing_id` - The panel being resized/moved
    /// * `new_position` - The desired new position
    /// * `direction` - Hint for which way to push displaced panels
    ///
    /// # Returns
    /// A `ReorderSolution` with new positions for all affected panels
    pub fn find_reorder_solution(
        &self,
        resizing_id: PanelId,
        new_position: GridPosition,
        direction: PushDirection,
    ) -> ReorderSolution {
        let mut solution = ReorderSolution::new();

        tracing::trace!(
            "find_reorder_solution: resizing_id={}, new_pos=({},{}){}x{}, positions_count={}",
            resizing_id, new_position.start.col, new_position.start.row,
            new_position.width, new_position.height,
            self.positions.len()
        );

        // Check if new position fits in grid
        if !new_position.fits_in_grid(self.columns, self.rows) {
            tracing::trace!("  new position doesn't fit in {}x{} grid", self.columns, self.rows);
            solution.valid = false;
            return solution;
        }

        // Find all panels that would overlap with the new position
        let mut displaced: Vec<(PanelId, GridPosition)> = Vec::new();
        for (&id, pos) in &self.positions {
            tracing::trace!(
                "  checking panel {}: ({},{}){}x{} - skip={}",
                id, pos.start.col, pos.start.row, pos.width, pos.height,
                id == resizing_id
            );
            if id == resizing_id {
                continue;
            }
            let overlaps = new_position.overlaps(pos);
            tracing::trace!("    overlaps={}", overlaps);
            if overlaps {
                displaced.push((id, pos.clone()));
            }
        }

        // If nothing displaced, we're done
        if displaced.is_empty() {
            tracing::trace!("  no displaced panels");
            return solution;
        }

        // Build working state: all current positions with the resizing panel at new position
        let mut working_positions: HashMap<PanelId, GridPosition> = self.positions.clone();
        working_positions.insert(resizing_id, new_position.clone());

        // Try to find positions for all displaced panels
        for (displaced_id, original_pos) in displaced {
            // Remove displaced panel from working state temporarily
            working_positions.remove(&displaced_id);

            // Find nearest valid position
            if let Some(new_pos) = self.find_nearest_valid_position(
                &original_pos,
                displaced_id,
                &working_positions,
                direction,
            ) {
                solution.moves.insert(displaced_id, (original_pos, new_pos.clone()));
                working_positions.insert(displaced_id, new_pos);
            } else {
                // No valid position found - solution is invalid
                solution.valid = false;
                return solution;
            }
        }

        solution
    }

    /// Find the nearest valid position for a panel, given working positions
    ///
    /// Searches outward from the original position, biased by direction hint
    fn find_nearest_valid_position(
        &self,
        original: &GridPosition,
        panel_id: PanelId,
        working_positions: &HashMap<PanelId, GridPosition>,
        direction: PushDirection,
    ) -> Option<GridPosition> {
        // Generate candidate positions sorted by distance, biased by direction
        let candidates = self.generate_push_candidates(original, direction);

        for candidate in candidates {
            // Check if candidate fits in grid
            if !candidate.fits_in_grid(self.columns, self.rows) {
                continue;
            }

            // Check if candidate overlaps any other panel in working state
            let overlaps = working_positions.iter().any(|(&id, pos)| {
                id != panel_id && candidate.overlaps(pos)
            });

            if !overlaps {
                return Some(candidate);
            }
        }

        None
    }

    /// Generate candidate positions for pushing a panel, ordered by preference
    fn generate_push_candidates(
        &self,
        original: &GridPosition,
        direction: PushDirection,
    ) -> Vec<GridPosition> {
        let mut candidates = Vec::new();
        let max_distance = (self.columns.max(self.rows) as i32) + 1;

        // Generate positions at increasing distances
        for distance in 1..=max_distance {
            // Get offsets for this distance, ordered by direction preference
            let offsets = self.offsets_for_distance(distance, direction);

            for (dx, dy) in offsets {
                let new_col = original.start.col as i32 + dx;
                let new_row = original.start.row as i32 + dy;

                if new_col >= 0 && new_row >= 0 {
                    candidates.push(GridPosition::new(
                        new_col as u32,
                        new_row as u32,
                        original.width,
                        original.height,
                    ));
                }
            }
        }

        candidates
    }

    /// Get offsets at a given manhattan distance, ordered by direction preference
    fn offsets_for_distance(&self, distance: i32, direction: PushDirection) -> Vec<(i32, i32)> {
        let mut offsets = Vec::new();

        // Generate all offsets at this manhattan distance
        for d in 0..=distance {
            let other = distance - d;

            // All combinations that sum to distance
            if d == 0 {
                offsets.push((0, other));
                offsets.push((0, -other));
            } else if other == 0 {
                offsets.push((d, 0));
                offsets.push((-d, 0));
            } else {
                offsets.push((d, other));
                offsets.push((d, -other));
                offsets.push((-d, other));
                offsets.push((-d, -other));
            }
        }

        // Remove duplicates (0,0) case
        offsets.retain(|&(dx, dy)| dx != 0 || dy != 0);
        offsets.sort();
        offsets.dedup();

        // Reorder based on direction preference
        match direction {
            PushDirection::Right => {
                offsets.sort_by_key(|&(dx, dy)| (-dx, dy.abs())); // Prefer positive dx
            }
            PushDirection::Left => {
                offsets.sort_by_key(|&(dx, dy)| (dx, dy.abs())); // Prefer negative dx
            }
            PushDirection::Down => {
                offsets.sort_by_key(|&(dx, dy)| (-dy, dx.abs())); // Prefer positive dy
            }
            PushDirection::Up => {
                offsets.sort_by_key(|&(dx, dy)| (dy, dx.abs())); // Prefer negative dy
            }
            PushDirection::Outward => {
                // Default: no preference, just manhattan distance order
            }
        }

        offsets
    }
}

/// Layout preset definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutPreset {
    /// Full screen single panel
    Single,
    /// Two columns: left | right
    TwoColumn,
    /// Three columns: left | center | right (ultrawide)
    ThreeColumn,
    /// Project browser + active project
    BrowserAndProject,
}

impl LayoutPreset {
    /// Get recommended preset for aspect ratio
    pub fn for_aspect_ratio(aspect: f32) -> Self {
        match aspect {
            a if a > 2.5 => LayoutPreset::ThreeColumn,  // 32:9
            a if a > 2.0 => LayoutPreset::TwoColumn,    // 21:9
            _ => LayoutPreset::Single,                   // 16:9 and below
        }
    }

    /// Create a layout with this preset applied
    pub fn create_layout(&self, aspect: f32) -> PanelLayout {
        let mut layout = PanelLayout::for_aspect_ratio(aspect);

        // Note: Actual panel positions would be set when panels are added
        // The preset just configures the grid, actual panel placement
        // depends on what panels exist

        layout
    }

    /// Get default panel positions for this preset
    ///
    /// Returns (panel_type, grid_position) pairs for typical usage
    pub fn default_positions(&self) -> Vec<(&'static str, GridPosition)> {
        match self {
            LayoutPreset::Single => {
                // Full screen for main content
                vec![("quest_log", GridPosition::new(0, 0, 8, 6))]
            }
            LayoutPreset::TwoColumn => {
                // Left: Quest Log, Right: Execution
                vec![
                    ("quest_log", GridPosition::new(0, 0, 4, 6)),
                    ("execution", GridPosition::new(4, 0, 4, 6)),
                ]
            }
            LayoutPreset::ThreeColumn => {
                // Left: Analysis, Center: Quest Log, Right: Execution
                vec![
                    ("analysis", GridPosition::new(0, 0, 4, 6)),
                    ("quest_log", GridPosition::new(4, 0, 4, 6)),
                    ("execution", GridPosition::new(8, 0, 4, 6)),
                ]
            }
            LayoutPreset::BrowserAndProject => {
                // Left: Project browser, Right: Active project content
                vec![
                    ("project_chooser", GridPosition::new(0, 0, 3, 6)),
                    ("quest_log", GridPosition::new(3, 0, 5, 6)),
                ]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_creation() {
        let layout = PanelLayout::new(8, 6);
        assert_eq!(layout.columns, 8);
        assert_eq!(layout.rows, 6);
        assert_eq!(layout.panel_count(), 0);
    }

    #[test]
    fn test_layout_aspect_ratios() {
        // 16:9 standard
        assert_eq!(PanelLayout::grid_for_aspect(16.0 / 9.0), (8, 6));

        // 21:9 ultrawide
        assert_eq!(PanelLayout::grid_for_aspect(21.0 / 9.0), (10, 6));

        // 32:9 super ultrawide
        assert_eq!(PanelLayout::grid_for_aspect(32.0 / 9.0), (12, 6));
    }

    #[test]
    fn test_layout_set_position() {
        let mut layout = PanelLayout::new(8, 6);

        // Valid position
        assert!(layout.set_position(1, GridPosition::new(0, 0, 4, 3)));
        assert_eq!(layout.panel_count(), 1);

        // Overlapping position should fail
        assert!(!layout.set_position(2, GridPosition::new(2, 2, 3, 3)));
        assert_eq!(layout.panel_count(), 1);

        // Non-overlapping position should succeed
        assert!(layout.set_position(2, GridPosition::new(4, 0, 4, 3)));
        assert_eq!(layout.panel_count(), 2);

        // Out of bounds should fail
        assert!(!layout.set_position(3, GridPosition::new(6, 4, 4, 4)));
    }

    #[test]
    fn test_layout_find_empty_space() {
        let mut layout = PanelLayout::new(8, 6);

        // Empty layout should find space at origin
        let space = layout.find_empty_space(2, 2).unwrap();
        assert_eq!(space.start.col, 0);
        assert_eq!(space.start.row, 0);

        // Add panel, should find space elsewhere
        layout.set_position(1, GridPosition::new(0, 0, 4, 6));
        let space = layout.find_empty_space(2, 2).unwrap();
        assert!(space.start.col >= 4);
    }

    #[test]
    fn test_layout_cell_at() {
        let layout = PanelLayout::new(8, 6)
            .with_gap(10.0)
            .with_margin(20.0);

        // Test at origin cell (after margin)
        let cell = layout.cell_at(25.0, 25.0, 1920.0, 1080.0);
        assert_eq!(cell, Some(GridCell::new(0, 0)));

        // Test in margin (should be None)
        let cell = layout.cell_at(5.0, 5.0, 1920.0, 1080.0);
        assert_eq!(cell, None);
    }

    #[test]
    fn test_layout_utilization() {
        let mut layout = PanelLayout::new(8, 6);

        assert_eq!(layout.utilization(), 0.0);

        // Add panel covering half the grid
        layout.set_position(1, GridPosition::new(0, 0, 4, 6));
        assert_eq!(layout.utilization(), 0.5);

        // Add another panel
        layout.set_position(2, GridPosition::new(4, 0, 4, 6));
        assert_eq!(layout.utilization(), 1.0);
    }

    #[test]
    fn test_preset_positions() {
        let positions = LayoutPreset::TwoColumn.default_positions();
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].0, "quest_log");
        assert_eq!(positions[1].0, "execution");
    }

    #[test]
    fn test_preset_for_aspect() {
        assert_eq!(LayoutPreset::for_aspect_ratio(16.0 / 9.0), LayoutPreset::Single);
        assert_eq!(LayoutPreset::for_aspect_ratio(21.0 / 9.0), LayoutPreset::TwoColumn);
        assert_eq!(LayoutPreset::for_aspect_ratio(32.0 / 9.0), LayoutPreset::ThreeColumn);
    }

    #[test]
    fn test_layout_is_valid_position() {
        let mut layout = PanelLayout::new(8, 6);
        layout.set_position(1, GridPosition::new(0, 0, 4, 3));

        // New position overlapping should be invalid
        assert!(!layout.is_valid_position(&GridPosition::new(2, 2, 3, 3), None));

        // Same position for existing panel (with exclude) should be valid
        assert!(layout.is_valid_position(&GridPosition::new(0, 0, 4, 3), Some(1)));

        // Out of bounds should be invalid
        assert!(!layout.is_valid_position(&GridPosition::new(6, 4, 4, 4), None));
    }

    // === ICS-style Reflow Tests ===

    #[test]
    fn test_reflow_no_displacement_needed() {
        let mut layout = PanelLayout::new(8, 6);
        // Panel 1: left half, Panel 2: right half
        layout.set_position(1, GridPosition::new(0, 0, 4, 6));
        layout.set_position(2, GridPosition::new(4, 0, 4, 6));

        // Resize panel 1 - doesn't overlap panel 2
        let solution = layout.find_reorder_solution(
            1,
            GridPosition::new(0, 0, 3, 6), // Shrink, no overlap
            PushDirection::Right,
        );

        assert!(solution.valid);
        assert!(!solution.has_moves()); // No panels displaced
    }

    #[test]
    fn test_reflow_simple_push_right() {
        let mut layout = PanelLayout::new(8, 6);
        // Two 2x2 panels side by side
        layout.set_position(1, GridPosition::new(0, 0, 2, 2));
        layout.set_position(2, GridPosition::new(2, 0, 2, 2));

        // Resize panel 1 to 3 wide - overlaps panel 2
        let solution = layout.find_reorder_solution(
            1,
            GridPosition::new(0, 0, 3, 2),
            PushDirection::Right,
        );

        assert!(solution.valid);
        assert!(solution.has_moves());

        // Panel 2 should be pushed right
        let new_pos = solution.get_move(2).unwrap();
        assert!(new_pos.start.col >= 3, "Panel 2 should be pushed right");
    }

    #[test]
    fn test_reflow_simple_push_down() {
        let mut layout = PanelLayout::new(8, 6);
        // Two 2x2 panels stacked
        layout.set_position(1, GridPosition::new(0, 0, 2, 2));
        layout.set_position(2, GridPosition::new(0, 2, 2, 2));

        // Resize panel 1 to 3 tall - overlaps panel 2
        let solution = layout.find_reorder_solution(
            1,
            GridPosition::new(0, 0, 2, 3),
            PushDirection::Down,
        );

        assert!(solution.valid);
        assert!(solution.has_moves());

        // Panel 2 should be pushed down
        let new_pos = solution.get_move(2).unwrap();
        assert!(new_pos.start.row >= 3, "Panel 2 should be pushed down");
    }

    #[test]
    fn test_reflow_no_room_invalid() {
        let mut layout = PanelLayout::new(4, 4);
        // Fill most of grid
        layout.set_position(1, GridPosition::new(0, 0, 2, 4)); // Left half
        layout.set_position(2, GridPosition::new(2, 0, 2, 2)); // Top right
        layout.set_position(3, GridPosition::new(2, 2, 2, 2)); // Bottom right

        // Try to make panel 1 span entire width - no room for others
        let solution = layout.find_reorder_solution(
            1,
            GridPosition::new(0, 0, 4, 4),
            PushDirection::Right,
        );

        assert!(!solution.valid);
    }

    #[test]
    fn test_reflow_out_of_bounds_invalid() {
        let mut layout = PanelLayout::new(8, 6);
        layout.set_position(1, GridPosition::new(0, 0, 4, 3));

        // Try to resize beyond grid bounds
        let solution = layout.find_reorder_solution(
            1,
            GridPosition::new(0, 0, 10, 3), // Too wide
            PushDirection::Right,
        );

        assert!(!solution.valid);
    }

    #[test]
    fn test_push_direction_affects_solution() {
        let mut layout = PanelLayout::new(8, 6);
        // Panel in center
        layout.set_position(1, GridPosition::new(0, 2, 4, 2));
        layout.set_position(2, GridPosition::new(4, 2, 2, 2)); // To the right

        // Resize panel 1 to overlap panel 2
        let solution_right = layout.find_reorder_solution(
            1,
            GridPosition::new(0, 2, 6, 2),
            PushDirection::Right,
        );

        let solution_down = layout.find_reorder_solution(
            1,
            GridPosition::new(0, 2, 6, 2),
            PushDirection::Down,
        );

        assert!(solution_right.valid);
        assert!(solution_down.valid);

        // Both should find a place for panel 2, but biased differently
        let pos_right = solution_right.get_move(2).unwrap();
        let pos_down = solution_down.get_move(2).unwrap();

        // With PushDirection::Right, panel should prefer moving right
        // With PushDirection::Down, panel should prefer moving down
        // The exact positions depend on what's available
        assert!(pos_right.start.col >= 6 || pos_down.start.row >= 4,
            "At least one direction should successfully push");
    }

    #[test]
    fn test_reorder_solution_api() {
        let mut solution = ReorderSolution::new();
        assert!(solution.valid);
        assert!(!solution.has_moves());

        solution.moves.insert(1, (
            GridPosition::new(0, 0, 2, 2),
            GridPosition::new(2, 0, 2, 2),
        ));

        assert!(solution.has_moves());
        assert!(solution.get_move(1).is_some());
        assert!(solution.get_move(999).is_none());
    }
}
