//! Edit mode for panel layout customization
//!
//! Edit mode allows users to:
//! - Add new panels by tapping empty cells
//! - Drag panels to new positions
//! - Resize panels using handles
//! - Remove panels
//!
//! Entry points:
//! - F2 key
//! - Main Menu → "Edit Layout"
//! - Long-press on empty area (500ms)

use super::{GridCell, GridPosition, PanelId, PanelLayout, PushDirection, ReorderSolution};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Threshold for long-press to enter edit mode
pub const LONG_PRESS_THRESHOLD: Duration = Duration::from_millis(500);

/// Resize handle positions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeHandle {
    TopLeft,
    Top,
    TopRight,
    Left,
    Right,
    BottomLeft,
    Bottom,
    BottomRight,
}

impl ResizeHandle {
    /// All resize handles
    pub fn all() -> &'static [ResizeHandle] {
        &[
            ResizeHandle::TopLeft,
            ResizeHandle::Top,
            ResizeHandle::TopRight,
            ResizeHandle::Left,
            ResizeHandle::Right,
            ResizeHandle::BottomLeft,
            ResizeHandle::Bottom,
            ResizeHandle::BottomRight,
        ]
    }

    /// Is this a corner handle?
    pub fn is_corner(&self) -> bool {
        matches!(
            self,
            ResizeHandle::TopLeft
                | ResizeHandle::TopRight
                | ResizeHandle::BottomLeft
                | ResizeHandle::BottomRight
        )
    }
}

/// Pixel bounds for a UI panel (computed at render time)
#[derive(Debug, Clone, Copy)]
pub struct UIPanelBounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl UIPanelBounds {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }
}

/// A UI panel that can be selected/resized in edit mode
#[derive(Debug, Clone)]
pub struct UIPanel {
    /// Panel identifier
    pub id: &'static str,
    /// Pixel bounds
    pub bounds: UIPanelBounds,
}

/// State for edit mode
#[derive(Debug, Clone)]
pub struct EditModeState {
    /// Currently active (edit mode is on)
    pub active: bool,

    /// Panel being dragged (if any)
    pub dragging: Option<DragState>,

    /// Panel being resized (if any)
    pub resizing: Option<ResizeState>,

    /// UI panel being resized (for static UI panels)
    pub ui_resize: Option<UIPanelResizeState>,

    /// UI panel being dragged (for static UI panels)
    pub ui_drag: Option<UIPanelDragState>,

    /// Long press tracking for edit mode entry
    pub long_press: Option<LongPressState>,

    /// Panel chooser dialog open
    pub panel_chooser_open: bool,

    /// Target cell for new panel (where user tapped)
    pub spawn_target: Option<GridCell>,

    /// Selected panel in chooser
    pub chooser_selection: usize,

    /// Ghost preview position (during drag/resize)
    pub ghost_position: Option<GridPosition>,

    /// Hovered panel (for showing handles)
    pub hovered_panel: Option<PanelId>,

    /// Currently selected panel in edit mode (by id)
    pub selected_panel: Option<&'static str>,

    /// UI panels visible on current screen (populated by renderer)
    pub ui_panels: Vec<UIPanel>,

    /// Preview positions for displaced panels (from reflow algorithm)
    /// Maps panel string ID to preview pixel bounds (x, y, w, h)
    pub reflow_previews: HashMap<&'static str, (f32, f32, f32, f32)>,

    /// Current reflow solution (if valid)
    pub reflow_solution: Option<ReorderSolution>,

    /// Override positions for panels (persists during edit mode)
    /// When set, these positions are used instead of the computed defaults
    pub panel_overrides: HashMap<&'static str, (f32, f32, f32, f32)>,
}

/// State for panel drag operation
#[derive(Debug, Clone)]
pub struct DragState {
    /// Panel being dragged
    pub panel_id: PanelId,
    /// Original position before drag
    pub original_position: GridPosition,
    /// Screen position where drag started
    pub start_screen_pos: (f32, f32),
    /// Current screen position
    pub current_screen_pos: (f32, f32),
}

/// State for panel resize operation
#[derive(Debug, Clone)]
pub struct ResizeState {
    /// Panel being resized
    pub panel_id: PanelId,
    /// Which handle is being dragged
    pub handle: ResizeHandle,
    /// Original position before resize
    pub original_position: GridPosition,
    /// Screen position where resize started
    pub start_screen_pos: (f32, f32),
}

/// State for tracking long press
#[derive(Debug, Clone)]
pub struct LongPressState {
    /// Screen position of press
    pub position: (f32, f32),
    /// When press started
    pub started: Instant,
    /// Whether we've moved too much (cancels long press)
    pub cancelled: bool,
}

/// State for UI panel resize operation (for static UI panels, not registry panels)
#[derive(Debug, Clone)]
pub struct UIPanelResizeState {
    /// Panel being resized (id string)
    pub panel_id: &'static str,
    /// Which handle is being dragged
    pub handle: ResizeHandle,
    /// Original pixel bounds before resize
    pub original_bounds: UIPanelBounds,
    /// Screen position where resize started
    pub start_pos: (f32, f32),
    /// Current screen position
    pub current_pos: (f32, f32),
}

/// State for UI panel drag operation (moving entire panel)
#[derive(Debug, Clone)]
pub struct UIPanelDragState {
    /// Panel being dragged (id string)
    pub panel_id: &'static str,
    /// Original pixel bounds before drag
    pub original_bounds: UIPanelBounds,
    /// Offset from panel corner to mouse position at drag start
    pub grab_offset: (f32, f32),
    /// Current panel position (top-left corner)
    pub current_pos: (f32, f32),
}

impl UIPanelResizeState {
    pub fn new(panel_id: &'static str, handle: ResizeHandle, bounds: UIPanelBounds, x: f32, y: f32) -> Self {
        Self {
            panel_id,
            handle,
            original_bounds: bounds,
            start_pos: (x, y),
            current_pos: (x, y),
        }
    }
}

impl Default for EditModeState {
    fn default() -> Self {
        Self::new()
    }
}

impl EditModeState {
    pub fn new() -> Self {
        Self {
            active: false,
            dragging: None,
            resizing: None,
            ui_resize: None,
            ui_drag: None,
            long_press: None,
            panel_chooser_open: false,
            spawn_target: None,
            chooser_selection: 0,
            ghost_position: None,
            hovered_panel: None,
            selected_panel: None,
            ui_panels: Vec::new(),
            reflow_previews: HashMap::new(),
            reflow_solution: None,
            panel_overrides: HashMap::new(),
        }
    }

    /// Enter edit mode
    pub fn enter(&mut self) {
        self.active = true;
        self.dragging = None;
        self.resizing = None;
        self.ui_resize = None;
        self.ui_drag = None;
        self.panel_chooser_open = false;
        self.spawn_target = None;
        self.ghost_position = None;
        self.reflow_previews.clear();
        self.reflow_solution = None;
    }

    /// Exit edit mode
    pub fn exit(&mut self) {
        self.active = false;
        self.dragging = None;
        self.resizing = None;
        self.ui_resize = None;
        self.ui_drag = None;
        self.panel_chooser_open = false;
        self.spawn_target = None;
        self.ghost_position = None;
        self.hovered_panel = None;
        self.selected_panel = None;
        self.ui_panels.clear();
        self.reflow_previews.clear();
        self.reflow_solution = None;
        // Clear overrides when exiting - positions reset to defaults
        // TODO: In the future, persist to ReDB for layout persistence
        self.panel_overrides.clear();
    }

    /// Toggle edit mode
    pub fn toggle(&mut self) {
        if self.active {
            self.exit();
        } else {
            self.enter();
        }
    }

    /// Start tracking a potential long press
    pub fn start_long_press(&mut self, x: f32, y: f32) {
        self.long_press = Some(LongPressState {
            position: (x, y),
            started: Instant::now(),
            cancelled: false,
        });
    }

    /// Update long press tracking on move
    /// Returns true if movement cancelled the long press
    pub fn update_long_press_position(&mut self, x: f32, y: f32) -> bool {
        const MAX_MOVEMENT: f32 = 20.0; // pixels

        if let Some(ref mut state) = self.long_press {
            let dx = x - state.position.0;
            let dy = y - state.position.1;
            let distance = (dx * dx + dy * dy).sqrt();

            if distance > MAX_MOVEMENT {
                state.cancelled = true;
                return true;
            }
        }
        false
    }

    /// Check if long press threshold reached
    /// Returns true if we should enter edit mode
    pub fn check_long_press(&mut self) -> bool {
        if let Some(ref state) = self.long_press {
            if !state.cancelled && state.started.elapsed() >= LONG_PRESS_THRESHOLD {
                self.long_press = None;
                return true;
            }
        }
        false
    }

    /// Cancel long press tracking
    pub fn cancel_long_press(&mut self) {
        self.long_press = None;
    }

    /// Start dragging a panel
    pub fn start_drag(&mut self, panel_id: PanelId, position: GridPosition, screen_x: f32, screen_y: f32) {
        self.dragging = Some(DragState {
            panel_id,
            original_position: position.clone(),
            start_screen_pos: (screen_x, screen_y),
            current_screen_pos: (screen_x, screen_y),
        });
        self.ghost_position = Some(position);
    }

    /// Update drag position
    pub fn update_drag(&mut self, screen_x: f32, screen_y: f32, layout: &PanelLayout, screen_width: f32, screen_height: f32) {
        if let Some(ref mut drag) = self.dragging {
            drag.current_screen_pos = (screen_x, screen_y);

            // Calculate new grid position based on cursor
            if let Some(cell) = layout.cell_at(screen_x, screen_y, screen_width, screen_height) {
                let new_pos = GridPosition::new(
                    cell.col,
                    cell.row,
                    drag.original_position.width,
                    drag.original_position.height,
                );

                // Only update ghost if position is valid
                if layout.is_valid_position(&new_pos, Some(drag.panel_id)) {
                    self.ghost_position = Some(new_pos);
                }
            }
        }
    }

    /// Finish drag operation
    /// Returns the new position if drag completed, None if cancelled
    pub fn finish_drag(&mut self) -> Option<(PanelId, GridPosition)> {
        if let Some(drag) = self.dragging.take() {
            if let Some(new_pos) = self.ghost_position.take() {
                // Only return new position if it changed
                if new_pos != drag.original_position {
                    return Some((drag.panel_id, new_pos));
                }
            }
        }
        self.ghost_position = None;
        None
    }

    /// Cancel drag operation
    pub fn cancel_drag(&mut self) {
        self.dragging = None;
        self.ghost_position = None;
    }

    /// Start resizing a panel
    pub fn start_resize(&mut self, panel_id: PanelId, handle: ResizeHandle, position: GridPosition, screen_x: f32, screen_y: f32) {
        self.resizing = Some(ResizeState {
            panel_id,
            handle,
            original_position: position.clone(),
            start_screen_pos: (screen_x, screen_y),
        });
        self.ghost_position = Some(position);
    }

    /// Update resize based on cursor movement
    pub fn update_resize(&mut self, screen_x: f32, screen_y: f32, layout: &PanelLayout, screen_width: f32, screen_height: f32, min_size: (u32, u32)) {
        if let Some(ref resize) = self.resizing {
            // Get target cell
            if let Some(target_cell) = layout.cell_at(screen_x, screen_y, screen_width, screen_height) {
                let orig = &resize.original_position;
                let (min_w, min_h) = min_size;

                // Calculate new position based on which handle is being dragged
                let new_pos = match resize.handle {
                    ResizeHandle::BottomRight => {
                        let new_width = (target_cell.col + 1).saturating_sub(orig.start.col).max(min_w);
                        let new_height = (target_cell.row + 1).saturating_sub(orig.start.row).max(min_h);
                        GridPosition::new(orig.start.col, orig.start.row, new_width, new_height)
                    }
                    ResizeHandle::Right => {
                        let new_width = (target_cell.col + 1).saturating_sub(orig.start.col).max(min_w);
                        GridPosition::new(orig.start.col, orig.start.row, new_width, orig.height)
                    }
                    ResizeHandle::Bottom => {
                        let new_height = (target_cell.row + 1).saturating_sub(orig.start.row).max(min_h);
                        GridPosition::new(orig.start.col, orig.start.row, orig.width, new_height)
                    }
                    ResizeHandle::TopLeft => {
                        let new_col = target_cell.col.min(orig.end_col().saturating_sub(min_w));
                        let new_row = target_cell.row.min(orig.end_row().saturating_sub(min_h));
                        let new_width = orig.end_col().saturating_sub(new_col).max(min_w);
                        let new_height = orig.end_row().saturating_sub(new_row).max(min_h);
                        GridPosition::new(new_col, new_row, new_width, new_height)
                    }
                    ResizeHandle::Top => {
                        let new_row = target_cell.row.min(orig.end_row().saturating_sub(min_h));
                        let new_height = orig.end_row().saturating_sub(new_row).max(min_h);
                        GridPosition::new(orig.start.col, new_row, orig.width, new_height)
                    }
                    ResizeHandle::Left => {
                        let new_col = target_cell.col.min(orig.end_col().saturating_sub(min_w));
                        let new_width = orig.end_col().saturating_sub(new_col).max(min_w);
                        GridPosition::new(new_col, orig.start.row, new_width, orig.height)
                    }
                    ResizeHandle::TopRight => {
                        let new_row = target_cell.row.min(orig.end_row().saturating_sub(min_h));
                        let new_width = (target_cell.col + 1).saturating_sub(orig.start.col).max(min_w);
                        let new_height = orig.end_row().saturating_sub(new_row).max(min_h);
                        GridPosition::new(orig.start.col, new_row, new_width, new_height)
                    }
                    ResizeHandle::BottomLeft => {
                        let new_col = target_cell.col.min(orig.end_col().saturating_sub(min_w));
                        let new_width = orig.end_col().saturating_sub(new_col).max(min_w);
                        let new_height = (target_cell.row + 1).saturating_sub(orig.start.row).max(min_h);
                        GridPosition::new(new_col, orig.start.row, new_width, new_height)
                    }
                };

                // Only update ghost if position is valid
                if layout.is_valid_position(&new_pos, Some(resize.panel_id)) {
                    self.ghost_position = Some(new_pos);
                }
            }
        }
    }

    /// Finish resize operation
    pub fn finish_resize(&mut self) -> Option<(PanelId, GridPosition)> {
        if let Some(resize) = self.resizing.take() {
            if let Some(new_pos) = self.ghost_position.take() {
                if new_pos != resize.original_position {
                    return Some((resize.panel_id, new_pos));
                }
            }
        }
        self.ghost_position = None;
        None
    }

    /// Cancel resize operation
    pub fn cancel_resize(&mut self) {
        self.resizing = None;
        self.ghost_position = None;
    }

    /// Open panel chooser at a specific cell
    pub fn open_panel_chooser(&mut self, target: GridCell) {
        self.panel_chooser_open = true;
        self.spawn_target = Some(target);
        self.chooser_selection = 0;
    }

    /// Close panel chooser
    pub fn close_panel_chooser(&mut self) {
        self.panel_chooser_open = false;
        self.spawn_target = None;
    }

    /// Move chooser selection
    pub fn move_chooser_selection(&mut self, delta: i32, max_items: usize) {
        if max_items == 0 {
            return;
        }
        let new_index = self.chooser_selection as i32 + delta;
        self.chooser_selection = new_index.rem_euclid(max_items as i32) as usize;
    }

    /// Is any operation in progress?
    pub fn is_busy(&self) -> bool {
        self.dragging.is_some() || self.resizing.is_some() || self.ui_resize.is_some() || self.panel_chooser_open
    }

    /// Compute reflow during UI panel resize
    ///
    /// This converts static UI panels to grid positions, runs the reflow algorithm,
    /// and stores preview positions for displaced panels.
    ///
    /// # Arguments
    /// * `layout` - The panel layout to use for grid calculations
    /// * `screen_width` - Screen width in pixels
    /// * `screen_height` - Screen height in pixels
    pub fn compute_reflow(
        &mut self,
        layout: &mut PanelLayout,
        screen_width: f32,
        screen_height: f32,
    ) {
        // Clear previous previews
        self.reflow_previews.clear();
        self.reflow_solution = None;

        // Get current resize state
        let resize = match &self.ui_resize {
            Some(r) => r,
            None => return,
        };

        // Map string IDs to numeric IDs for the layout system
        let panel_ids: HashMap<&str, PanelId> = self.ui_panels
            .iter()
            .enumerate()
            .map(|(i, p)| (p.id, i as PanelId + 1))
            .collect();

        let reverse_ids: HashMap<PanelId, &'static str> = panel_ids
            .iter()
            .map(|(&name, &id)| (id, name))
            .collect();

        // Get the ID of the panel being resized
        let resizing_id = match panel_ids.get(resize.panel_id) {
            Some(&id) => id,
            None => return,
        };

        // Reconfigure layout for current screen size (~40px cells)
        *layout = PanelLayout::for_screen_size(screen_width, screen_height);

        // Populate layout with current UI panel positions
        for panel in &self.ui_panels {
            if let Some(&id) = panel_ids.get(panel.id) {
                let grid_pos = layout.grid_position_from_pixels(
                    panel.bounds.x,
                    panel.bounds.y,
                    panel.bounds.width,
                    panel.bounds.height,
                    screen_width,
                    screen_height,
                );
                tracing::trace!(
                    "Panel '{}' pixel bounds ({:.0},{:.0},{:.0},{:.0}) -> grid ({},{}) {}x{}",
                    panel.id, panel.bounds.x, panel.bounds.y, panel.bounds.width, panel.bounds.height,
                    grid_pos.start.col, grid_pos.start.row, grid_pos.width, grid_pos.height
                );
                // Force insert - we're building collision state, not checking validity
                layout.force_set_position(id, grid_pos);
            }
        }

        // Calculate the new position for the resizing panel
        let delta_x = resize.current_pos.0 - resize.start_pos.0;
        let delta_y = resize.current_pos.1 - resize.start_pos.1;

        // Apply resize delta based on handle
        let (new_x, new_y, new_w, new_h) = self.apply_resize_delta_pub(
            resize.original_bounds,
            resize.handle,
            delta_x,
            delta_y,
        );

        // Convert new pixel bounds to grid position
        let new_grid_pos = layout.grid_position_from_pixels(
            new_x, new_y, new_w, new_h,
            screen_width, screen_height,
        );

        tracing::trace!(
            "Resizing '{}' to pixel ({:.0},{:.0},{:.0},{:.0}) -> grid ({},{}) {}x{}",
            resize.panel_id, new_x, new_y, new_w, new_h,
            new_grid_pos.start.col, new_grid_pos.start.row, new_grid_pos.width, new_grid_pos.height
        );

        // Determine push direction from handle
        let direction = PushDirection::from_resize_handle(&resize.handle);

        // Run the reflow algorithm
        let solution = layout.find_reorder_solution(resizing_id, new_grid_pos.clone(), direction);

        tracing::trace!(
            "Reflow solution: valid={}, moves={}",
            solution.valid, solution.moves.len()
        );

        if solution.valid && solution.has_moves() {
            // Convert grid moves back to pixel previews
            for (&id, (orig_pos, new_pos)) in &solution.moves {
                if let Some(&name) = reverse_ids.get(&id) {
                    let bounds = layout.bounds_for(new_pos, screen_width, screen_height);
                    self.reflow_previews.insert(name, (bounds.x, bounds.y, bounds.width, bounds.height));
                    tracing::debug!(
                        "Reflow: '{}' displaced from ({},{}) to ({},{}) -> pixel ({:.0},{:.0})",
                        name, orig_pos.start.col, orig_pos.start.row,
                        new_pos.start.col, new_pos.start.row,
                        bounds.x, bounds.y
                    );
                }
            }
        }

        self.reflow_solution = Some(solution);
    }

    /// Apply resize delta to bounds based on handle
    ///
    /// Public version for use by events.rs to compute resize preview.
    pub fn apply_resize_delta_pub(
        &self,
        orig: UIPanelBounds,
        handle: ResizeHandle,
        delta_x: f32,
        delta_y: f32,
    ) -> (f32, f32, f32, f32) {
        let min_size = 50.0;

        match handle {
            ResizeHandle::Right => {
                (orig.x, orig.y, (orig.width + delta_x).max(min_size), orig.height)
            }
            ResizeHandle::Bottom => {
                (orig.x, orig.y, orig.width, (orig.height + delta_y).max(min_size))
            }
            ResizeHandle::BottomRight => {
                (orig.x, orig.y, (orig.width + delta_x).max(min_size), (orig.height + delta_y).max(min_size))
            }
            ResizeHandle::Left => {
                let new_w = (orig.width - delta_x).max(min_size);
                let new_x = orig.x + orig.width - new_w;
                (new_x, orig.y, new_w, orig.height)
            }
            ResizeHandle::Top => {
                let new_h = (orig.height - delta_y).max(min_size);
                let new_y = orig.y + orig.height - new_h;
                (orig.x, new_y, orig.width, new_h)
            }
            ResizeHandle::TopLeft => {
                let new_w = (orig.width - delta_x).max(min_size);
                let new_h = (orig.height - delta_y).max(min_size);
                let new_x = orig.x + orig.width - new_w;
                let new_y = orig.y + orig.height - new_h;
                (new_x, new_y, new_w, new_h)
            }
            ResizeHandle::TopRight => {
                let new_h = (orig.height - delta_y).max(min_size);
                let new_y = orig.y + orig.height - new_h;
                (orig.x, new_y, (orig.width + delta_x).max(min_size), new_h)
            }
            ResizeHandle::BottomLeft => {
                let new_w = (orig.width - delta_x).max(min_size);
                let new_x = orig.x + orig.width - new_w;
                (new_x, orig.y, new_w, (orig.height + delta_y).max(min_size))
            }
        }
    }

    /// Clear reflow state
    pub fn clear_reflow(&mut self) {
        self.reflow_previews.clear();
        self.reflow_solution = None;
    }

    /// Compute reflow during UI panel drag (ICS-style collision detection)
    ///
    /// Similar to compute_reflow but for drag operations instead of resize.
    pub fn compute_drag_reflow(
        &mut self,
        layout: &mut PanelLayout,
        screen_width: f32,
        screen_height: f32,
    ) {
        // Clear previous previews
        self.reflow_previews.clear();
        self.reflow_solution = None;

        // Get current drag state
        let drag = match &self.ui_drag {
            Some(d) => d,
            None => return,
        };

        // Map string IDs to numeric IDs for the layout system
        let panel_ids: HashMap<&str, PanelId> = self.ui_panels
            .iter()
            .enumerate()
            .map(|(i, p)| (p.id, i as PanelId + 1))
            .collect();

        let reverse_ids: HashMap<PanelId, &'static str> = panel_ids
            .iter()
            .map(|(&name, &id)| (id, name))
            .collect();

        // Get the ID of the panel being dragged
        let dragging_id = match panel_ids.get(drag.panel_id) {
            Some(&id) => id,
            None => return,
        };

        // Reconfigure layout for current screen size (~40px cells)
        *layout = PanelLayout::for_screen_size(screen_width, screen_height);

        // Populate layout with current UI panel positions
        for panel in &self.ui_panels {
            if let Some(&id) = panel_ids.get(panel.id) {
                let grid_pos = layout.grid_position_from_pixels(
                    panel.bounds.x,
                    panel.bounds.y,
                    panel.bounds.width,
                    panel.bounds.height,
                    screen_width,
                    screen_height,
                );
                layout.force_set_position(id, grid_pos);
            }
        }

        // Calculate the new position for the dragging panel
        let (new_x, new_y) = drag.current_pos;
        let orig = drag.original_bounds;

        // Convert new pixel bounds to grid position (same size, new location)
        let new_grid_pos = layout.grid_position_from_pixels(
            new_x, new_y, orig.width, orig.height,
            screen_width, screen_height,
        );

        tracing::trace!(
            "Dragging '{}' to pixel ({:.0},{:.0}) -> grid ({},{}) {}x{}",
            drag.panel_id, new_x, new_y,
            new_grid_pos.start.col, new_grid_pos.start.row, new_grid_pos.width, new_grid_pos.height
        );

        // Use Outward direction for drag (push panels away from drag point)
        let direction = PushDirection::Outward;

        // Run the reflow algorithm
        let solution = layout.find_reorder_solution(dragging_id, new_grid_pos.clone(), direction);

        tracing::trace!(
            "Drag reflow solution: valid={}, moves={}",
            solution.valid, solution.moves.len()
        );

        if solution.valid && solution.has_moves() {
            // Convert grid moves back to pixel previews
            for (&id, (orig_pos, new_pos)) in &solution.moves {
                if let Some(&name) = reverse_ids.get(&id) {
                    let bounds = layout.bounds_for(new_pos, screen_width, screen_height);
                    self.reflow_previews.insert(name, (bounds.x, bounds.y, bounds.width, bounds.height));
                    tracing::debug!(
                        "Drag reflow: '{}' displaced from ({},{}) to ({},{}) -> pixel ({:.0},{:.0})",
                        name, orig_pos.start.col, orig_pos.start.row,
                        new_pos.start.col, new_pos.start.row,
                        bounds.x, bounds.y
                    );
                }
            }
        }

        self.reflow_solution = Some(solution);
    }

    /// Apply the current resize operation to panel_overrides
    ///
    /// Call this on release to persist the new panel position.
    /// Also applies any reflow moves from displaced panels.
    pub fn apply_resize_to_overrides(&mut self) {
        // Get resize state (must still be present)
        if let Some(ref resize) = self.ui_resize {
            let delta_x = resize.current_pos.0 - resize.start_pos.0;
            let delta_y = resize.current_pos.1 - resize.start_pos.1;

            // Calculate new bounds for resized panel
            let (new_x, new_y, new_w, new_h) = self.apply_resize_delta_pub(
                resize.original_bounds,
                resize.handle,
                delta_x,
                delta_y,
            );

            // Store override for the resized panel
            self.panel_overrides.insert(resize.panel_id, (new_x, new_y, new_w, new_h));
            tracing::debug!(
                "Applied resize override for '{}': ({:.0}, {:.0}, {:.0}, {:.0})",
                resize.panel_id, new_x, new_y, new_w, new_h
            );
        }

        // Apply reflow positions for displaced panels
        for (&panel_id, &(x, y, w, h)) in &self.reflow_previews {
            self.panel_overrides.insert(panel_id, (x, y, w, h));
            tracing::debug!(
                "Applied reflow override for '{}': ({:.0}, {:.0}, {:.0}, {:.0})",
                panel_id, x, y, w, h
            );
        }
    }

    /// Apply the current drag operation to panel_overrides
    ///
    /// Call this on release to persist the new panel position.
    /// Also applies any reflow moves from displaced panels.
    pub fn apply_drag_to_overrides(&mut self) {
        // Get drag state (must still be present)
        if let Some(ref drag) = self.ui_drag {
            let (new_x, new_y) = drag.current_pos;
            let orig = drag.original_bounds;

            // Store override for the dragged panel (same size, new position)
            self.panel_overrides.insert(drag.panel_id, (new_x, new_y, orig.width, orig.height));
            tracing::debug!(
                "Applied drag override for '{}': ({:.0}, {:.0}, {:.0}, {:.0})",
                drag.panel_id, new_x, new_y, orig.width, orig.height
            );
        }

        // Apply reflow positions for displaced panels
        for (&panel_id, &(x, y, w, h)) in &self.reflow_previews {
            self.panel_overrides.insert(panel_id, (x, y, w, h));
            tracing::debug!(
                "Applied drag reflow override for '{}': ({:.0}, {:.0}, {:.0}, {:.0})",
                panel_id, x, y, w, h
            );
        }
    }

    /// Get panel bounds, checking overrides first then falling back to computed
    ///
    /// Returns (x, y, width, height) for the panel, or None if not found.
    pub fn get_panel_bounds(&self, panel_id: &str) -> Option<(f32, f32, f32, f32)> {
        // Check overrides first
        if let Some(&bounds) = self.panel_overrides.get(panel_id) {
            return Some(bounds);
        }

        // Fall back to ui_panels
        for panel in &self.ui_panels {
            if panel.id == panel_id {
                return Some((
                    panel.bounds.x,
                    panel.bounds.y,
                    panel.bounds.width,
                    panel.bounds.height,
                ));
            }
        }

        None
    }

    /// Update ui_panels list, applying any overrides
    ///
    /// Call this after compute_ui_panels() to apply saved overrides.
    pub fn apply_overrides_to_ui_panels(&mut self) {
        for panel in &mut self.ui_panels {
            if let Some(&(x, y, w, h)) = self.panel_overrides.get(panel.id) {
                panel.bounds = UIPanelBounds::new(x, y, w, h);
            }
        }
    }
}

/// Available panel types for the chooser
#[derive(Debug, Clone)]
pub struct PanelTypeInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub min_size: (u32, u32),
}

/// Get all available panel types for the chooser
pub fn available_panel_types() -> Vec<PanelTypeInfo> {
    vec![
        PanelTypeInfo {
            id: "quest_log",
            name: "Quest Log",
            description: "Card grid for task selection",
            min_size: (3, 3),
        },
        PanelTypeInfo {
            id: "execution",
            name: "Execution",
            description: "Live tool/thought display",
            min_size: (3, 4),
        },
        PanelTypeInfo {
            id: "analysis",
            name: "Analysis",
            description: "Tool/thought logs",
            min_size: (2, 3),
        },
        PanelTypeInfo {
            id: "project_chooser",
            name: "Project Chooser",
            description: "Grid of all projects",
            min_size: (3, 3),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_mode_toggle() {
        let mut state = EditModeState::new();
        assert!(!state.active);

        state.toggle();
        assert!(state.active);

        state.toggle();
        assert!(!state.active);
    }

    #[test]
    fn test_edit_mode_enter_exit() {
        let mut state = EditModeState::new();

        state.enter();
        assert!(state.active);

        // Start some operations
        state.panel_chooser_open = true;
        state.ghost_position = Some(GridPosition::new(0, 0, 2, 2));

        state.exit();
        assert!(!state.active);
        assert!(!state.panel_chooser_open);
        assert!(state.ghost_position.is_none());
    }

    #[test]
    fn test_long_press_threshold() {
        let mut state = EditModeState::new();

        state.start_long_press(100.0, 100.0);
        assert!(state.long_press.is_some());

        // Immediately checking should return false
        assert!(!state.check_long_press());

        // Movement should be tracked
        assert!(!state.update_long_press_position(105.0, 105.0)); // Small movement OK

        // Large movement cancels
        state.start_long_press(100.0, 100.0);
        assert!(state.update_long_press_position(200.0, 200.0)); // Cancelled
    }

    #[test]
    fn test_drag_state() {
        let mut state = EditModeState::new();
        state.enter();

        let pos = GridPosition::new(0, 0, 2, 2);
        state.start_drag(1, pos.clone(), 100.0, 100.0);

        assert!(state.dragging.is_some());
        assert!(state.ghost_position.is_some());

        state.cancel_drag();
        assert!(state.dragging.is_none());
        assert!(state.ghost_position.is_none());
    }

    #[test]
    fn test_panel_chooser() {
        let mut state = EditModeState::new();
        state.enter();

        state.open_panel_chooser(GridCell::new(2, 3));
        assert!(state.panel_chooser_open);
        assert_eq!(state.spawn_target, Some(GridCell::new(2, 3)));
        assert_eq!(state.chooser_selection, 0);

        // Test selection navigation
        state.move_chooser_selection(1, 4);
        assert_eq!(state.chooser_selection, 1);

        state.move_chooser_selection(-1, 4);
        assert_eq!(state.chooser_selection, 0);

        // Wrap around
        state.move_chooser_selection(-1, 4);
        assert_eq!(state.chooser_selection, 3);

        state.close_panel_chooser();
        assert!(!state.panel_chooser_open);
        assert!(state.spawn_target.is_none());
    }

    #[test]
    fn test_is_busy() {
        let mut state = EditModeState::new();
        state.enter();

        assert!(!state.is_busy());

        state.start_drag(1, GridPosition::new(0, 0, 2, 2), 0.0, 0.0);
        assert!(state.is_busy());

        state.cancel_drag();
        assert!(!state.is_busy());

        state.open_panel_chooser(GridCell::new(0, 0));
        assert!(state.is_busy());
    }

    #[test]
    fn test_available_panel_types() {
        let types = available_panel_types();
        assert!(!types.is_empty());

        // Should have quest_log
        assert!(types.iter().any(|t| t.id == "quest_log"));
    }

    #[test]
    fn test_resize_handles() {
        assert!(ResizeHandle::TopLeft.is_corner());
        assert!(ResizeHandle::BottomRight.is_corner());
        assert!(!ResizeHandle::Top.is_corner());
        assert!(!ResizeHandle::Left.is_corner());

        assert_eq!(ResizeHandle::all().len(), 8);
    }
}
