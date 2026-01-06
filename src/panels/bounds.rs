//! Grid layout primitives for panel positioning

/// A single cell in the grid
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridCell {
    pub col: u32,
    pub row: u32,
}

impl GridCell {
    pub fn new(col: u32, row: u32) -> Self {
        Self { col, row }
    }
}

/// A panel's position in the grid (can span multiple cells)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridPosition {
    /// Top-left cell of this panel
    pub start: GridCell,
    /// How many cells wide
    pub width: u32,
    /// How many cells tall
    pub height: u32,
}

impl GridPosition {
    pub fn new(col: u32, row: u32, width: u32, height: u32) -> Self {
        Self {
            start: GridCell::new(col, row),
            width,
            height,
        }
    }

    /// Total cells covered by this position
    pub fn cells_covered(&self) -> u32 {
        self.width * self.height
    }

    /// End column (exclusive)
    pub fn end_col(&self) -> u32 {
        self.start.col + self.width
    }

    /// End row (exclusive)
    pub fn end_row(&self) -> u32 {
        self.start.row + self.height
    }

    /// Check if this position overlaps with another
    pub fn overlaps(&self, other: &GridPosition) -> bool {
        // No overlap if one is completely to the left, right, above, or below
        let no_overlap = self.end_col() <= other.start.col
            || other.end_col() <= self.start.col
            || self.end_row() <= other.start.row
            || other.end_row() <= self.start.row;

        !no_overlap
    }

    /// Check if this position fits within grid bounds
    pub fn fits_in_grid(&self, columns: u32, rows: u32) -> bool {
        self.end_col() <= columns && self.end_row() <= rows
    }

    /// Iterate over all cells covered by this position
    pub fn cells(&self) -> impl Iterator<Item = GridCell> {
        let start_col = self.start.col;
        let start_row = self.start.row;
        let width = self.width;
        let height = self.height;

        (0..height).flat_map(move |dy| {
            (0..width).map(move |dx| GridCell::new(start_col + dx, start_row + dy))
        })
    }
}

/// Pixel bounds for rendering a panel
#[derive(Clone, Debug)]
pub struct PanelBounds {
    /// Left edge in pixels
    pub x: f32,
    /// Top edge in pixels
    pub y: f32,
    /// Width in pixels
    pub width: f32,
    /// Height in pixels
    pub height: f32,
}

impl PanelBounds {
    /// Calculate pixel bounds from grid position
    ///
    /// # Arguments
    /// * `pos` - Grid position of the panel
    /// * `grid_cols` - Total columns in grid
    /// * `grid_rows` - Total rows in grid
    /// * `screen_width` - Screen width in pixels
    /// * `screen_height` - Screen height in pixels
    /// * `gap` - Gap between cells in pixels
    /// * `margin` - Margin around grid in pixels
    pub fn from_grid(
        pos: &GridPosition,
        grid_cols: u32,
        grid_rows: u32,
        screen_width: f32,
        screen_height: f32,
        gap: f32,
        margin: f32,
    ) -> Self {
        // Available space after margins
        let available_width = screen_width - margin * 2.0;
        let available_height = screen_height - margin * 2.0;

        // Cell size (accounting for gaps between cells)
        let total_gap_width = gap * (grid_cols - 1) as f32;
        let total_gap_height = gap * (grid_rows - 1) as f32;

        let cell_width = (available_width - total_gap_width) / grid_cols as f32;
        let cell_height = (available_height - total_gap_height) / grid_rows as f32;

        // Position calculation
        let x = margin + pos.start.col as f32 * (cell_width + gap);
        let y = margin + pos.start.row as f32 * (cell_height + gap);

        // Size calculation (spanning multiple cells includes internal gaps)
        let width = pos.width as f32 * cell_width + (pos.width - 1) as f32 * gap;
        let height = pos.height as f32 * cell_height + (pos.height - 1) as f32 * gap;

        Self { x, y, width, height }
    }

    /// Right edge in pixels
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    /// Bottom edge in pixels
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    /// Center X coordinate
    pub fn center_x(&self) -> f32 {
        self.x + self.width / 2.0
    }

    /// Center Y coordinate
    pub fn center_y(&self) -> f32 {
        self.y + self.height / 2.0
    }

    /// Check if a point is inside these bounds
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.right() && py >= self.y && py < self.bottom()
    }

    /// Shrink bounds by padding on all sides
    pub fn inset(&self, padding: f32) -> Self {
        Self {
            x: self.x + padding,
            y: self.y + padding,
            width: (self.width - padding * 2.0).max(0.0),
            height: (self.height - padding * 2.0).max(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_position_overlap_detection() {
        let pos1 = GridPosition::new(0, 0, 2, 2);
        let pos2 = GridPosition::new(1, 1, 2, 2); // overlaps
        let pos3 = GridPosition::new(2, 0, 2, 2); // adjacent, no overlap
        let pos4 = GridPosition::new(0, 2, 2, 2); // below, no overlap

        assert!(pos1.overlaps(&pos2));
        assert!(pos2.overlaps(&pos1));
        assert!(!pos1.overlaps(&pos3));
        assert!(!pos1.overlaps(&pos4));
    }

    #[test]
    fn test_grid_position_fits_in_grid() {
        let pos = GridPosition::new(6, 4, 2, 2);

        // 8x6 grid - should fit
        assert!(pos.fits_in_grid(8, 6));

        // Too narrow
        assert!(!pos.fits_in_grid(7, 6));

        // Too short
        assert!(!pos.fits_in_grid(8, 5));
    }

    #[test]
    fn test_grid_position_cells() {
        let pos = GridPosition::new(1, 2, 2, 2);
        let cells: Vec<_> = pos.cells().collect();

        assert_eq!(cells.len(), 4);
        assert!(cells.contains(&GridCell::new(1, 2)));
        assert!(cells.contains(&GridCell::new(2, 2)));
        assert!(cells.contains(&GridCell::new(1, 3)));
        assert!(cells.contains(&GridCell::new(2, 3)));
    }

    #[test]
    fn test_panel_bounds_contains() {
        let bounds = PanelBounds {
            x: 100.0,
            y: 100.0,
            width: 200.0,
            height: 150.0,
        };

        assert!(bounds.contains(150.0, 150.0)); // center
        assert!(bounds.contains(100.0, 100.0)); // top-left corner
        assert!(!bounds.contains(300.0, 250.0)); // just outside
        assert!(!bounds.contains(50.0, 150.0)); // left of bounds
    }

    #[test]
    fn test_panel_bounds_inset() {
        let bounds = PanelBounds {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
        };

        let inset = bounds.inset(10.0);
        assert_eq!(inset.x, 10.0);
        assert_eq!(inset.y, 10.0);
        assert_eq!(inset.width, 80.0);
        assert_eq!(inset.height, 80.0);
    }

    #[test]
    fn test_panel_bounds_from_grid_full_screen() {
        // Single panel spanning entire 8x6 grid on 1920x1080
        let pos = GridPosition::new(0, 0, 8, 6);
        let bounds = PanelBounds::from_grid(&pos, 8, 6, 1920.0, 1080.0, 0.0, 0.0);

        // Should fill entire screen with no gap/margin
        assert_eq!(bounds.x, 0.0);
        assert_eq!(bounds.y, 0.0);
        assert_eq!(bounds.width, 1920.0);
        assert_eq!(bounds.height, 1080.0);
    }

    #[test]
    fn test_panel_bounds_from_grid_with_margin() {
        let pos = GridPosition::new(0, 0, 8, 6);
        let bounds = PanelBounds::from_grid(&pos, 8, 6, 1920.0, 1080.0, 0.0, 20.0);

        // Should have 20px margin on all sides
        assert_eq!(bounds.x, 20.0);
        assert_eq!(bounds.y, 20.0);
        assert_eq!(bounds.width, 1880.0); // 1920 - 40
        assert_eq!(bounds.height, 1040.0); // 1080 - 40
    }
}
