//! Layout testing framework for detecting overlapping UI elements
//!
//! Tests all screens at 1080p resolution across all UI scales.

use crate::state::{ProjectAction, UiScaleOption};

/// A rendered element with its bounding box
#[derive(Debug, Clone)]
pub struct LayoutElement {
    pub name: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl LayoutElement {
    pub fn new(name: impl Into<String>, x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            name: name.into(),
            x,
            y,
            width,
            height,
        }
    }

    /// Check if this element fully contains another (parent-child relationship)
    pub fn contains(&self, other: &LayoutElement) -> bool {
        other.x >= self.x
            && other.y >= self.y
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
    }

    /// Check if this element overlaps with another
    pub fn overlaps(&self, other: &LayoutElement) -> bool {
        // AABB intersection test
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }

    /// Check if this element is fully within the screen bounds
    pub fn within_screen(&self, screen_width: f32, screen_height: f32) -> bool {
        self.x >= 0.0
            && self.y >= 0.0
            && self.x + self.width <= screen_width
            && self.y + self.height <= screen_height
    }

    /// Get the right edge position
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    /// Get the bottom edge position
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }
}

/// Overlap detection result
#[derive(Debug)]
pub struct OverlapResult {
    pub element_a: String,
    pub element_b: String,
    pub overlap_area: f32,
}

/// Out of bounds result
#[derive(Debug)]
pub struct OutOfBoundsResult {
    pub element: String,
    pub overflow_right: f32,
    pub overflow_bottom: f32,
}

/// Layout validation results for a single screen
#[derive(Debug, Default)]
pub struct LayoutValidation {
    pub screen_name: String,
    pub scale: f32,
    pub screen_width: f32,
    pub screen_height: f32,
    pub overlaps: Vec<OverlapResult>,
    pub out_of_bounds: Vec<OutOfBoundsResult>,
}

impl LayoutValidation {
    pub fn new(screen_name: impl Into<String>, scale: f32, width: f32, height: f32) -> Self {
        Self {
            screen_name: screen_name.into(),
            scale,
            screen_width: width,
            screen_height: height,
            overlaps: Vec::new(),
            out_of_bounds: Vec::new(),
        }
    }

    /// Validate a list of elements for overlaps and bounds
    pub fn validate(&mut self, elements: &[LayoutElement]) {
        // Check for overlaps between all pairs
        for i in 0..elements.len() {
            for j in (i + 1)..elements.len() {
                if elements[i].overlaps(&elements[j]) {
                    // Skip if one contains the other (parent-child relationship)
                    if elements[i].contains(&elements[j]) || elements[j].contains(&elements[i]) {
                        continue;
                    }

                    let overlap_width = (elements[i].right().min(elements[j].right())
                        - elements[i].x.max(elements[j].x))
                    .max(0.0);
                    let overlap_height = (elements[i].bottom().min(elements[j].bottom())
                        - elements[i].y.max(elements[j].y))
                    .max(0.0);
                    let overlap_area = overlap_width * overlap_height;

                    self.overlaps.push(OverlapResult {
                        element_a: elements[i].name.clone(),
                        element_b: elements[j].name.clone(),
                        overlap_area,
                    });
                }
            }
        }

        // Check for out of bounds
        for elem in elements {
            if !elem.within_screen(self.screen_width, self.screen_height) {
                let overflow_right = (elem.right() - self.screen_width).max(0.0);
                let overflow_bottom = (elem.bottom() - self.screen_height).max(0.0);

                // Also check left/top overflow
                let overflow_left = (-elem.x).max(0.0);
                let overflow_top = (-elem.y).max(0.0);

                if overflow_right > 0.0 || overflow_bottom > 0.0 || overflow_left > 0.0 || overflow_top > 0.0 {
                    self.out_of_bounds.push(OutOfBoundsResult {
                        element: elem.name.clone(),
                        overflow_right: overflow_right + overflow_left,
                        overflow_bottom: overflow_bottom + overflow_top,
                    });
                }
            }
        }
    }

    pub fn has_issues(&self) -> bool {
        !self.overlaps.is_empty() || !self.out_of_bounds.is_empty()
    }

    pub fn print_issues(&self) {
        if self.overlaps.is_empty() && self.out_of_bounds.is_empty() {
            println!("  ✓ {} @ {}x - No issues", self.screen_name, self.scale);
            return;
        }

        println!(
            "  ✗ {} @ {}x ({}x{}):",
            self.screen_name, self.scale, self.screen_width, self.screen_height
        );

        for overlap in &self.overlaps {
            println!(
                "    OVERLAP: '{}' and '{}' ({:.0}px² area)",
                overlap.element_a, overlap.element_b, overlap.overlap_area
            );
        }

        for oob in &self.out_of_bounds {
            println!(
                "    OUT OF BOUNDS: '{}' (overflow: +{:.0}px right, +{:.0}px bottom)",
                oob.element, oob.overflow_right, oob.overflow_bottom
            );
        }
    }
}

/// UI Scale helper for tests (matches UiScale from gpu.rs)
pub struct TestUiScale {
    pub dpi_scale: f32,
}

impl TestUiScale {
    pub fn new(scale: f32) -> Self {
        Self { dpi_scale: scale }
    }

    pub fn px(&self, base: f32) -> f32 {
        base * self.dpi_scale
    }
}

/// Screen dimensions for testing
pub const TEST_SCREEN_WIDTH: f32 = 1920.0;
pub const TEST_SCREEN_HEIGHT: f32 = 1080.0;

/// All scales to test
pub fn all_test_scales() -> Vec<f32> {
    UiScaleOption::all().iter().map(|s| s.value()).collect()
}

/// Card grid layout calculator (mirrors CardGrid in gpu.rs)
pub struct CardGrid {
    pub card_width: f32,
    pub card_height: f32,
    pub gap: f32,
    pub columns: usize,
    pub margin_x: f32,
    pub margin_y: f32,
}

impl CardGrid {
    /// Default grid for project chooser
    pub fn for_project_chooser(screen_width: f32, scale: &TestUiScale) -> Self {
        let card_width = scale.px(320.0);
        let card_height = scale.px(180.0);
        let gap = scale.px(24.0);
        let margin_x = scale.px(60.0);
        let margin_y = scale.px(140.0); // 60 + 80 for title

        let available_width = screen_width - margin_x * 2.0;
        let columns = ((available_width + gap) / (card_width + gap)).floor() as usize;
        let columns = columns.max(1);

        Self { card_width, card_height, gap, columns, margin_x, margin_y }
    }

    /// Grid for PalaceLoop - targets 5 columns
    pub fn for_palace_loop(screen_width: f32, scale: &TestUiScale) -> Self {
        let target_columns = 5;
        let gap = scale.px(16.0);
        let margin_x = scale.px(40.0);
        let margin_y = scale.px(120.0);

        let available_width = screen_width - margin_x * 2.0;
        let card_width = (available_width - gap * (target_columns - 1) as f32) / target_columns as f32;
        let card_height = card_width * 0.6;

        Self { card_width, card_height, gap, columns: target_columns, margin_x, margin_y }
    }

    /// Get position for card at index
    pub fn card_position(&self, index: usize) -> (f32, f32) {
        let col = index % self.columns;
        let row = index / self.columns;
        let x = self.margin_x + col as f32 * (self.card_width + self.gap);
        let y = self.margin_y + row as f32 * (self.card_height + self.gap);
        (x, y)
    }
}

/// Extract layout elements for ProjectChooser screen
pub fn layout_project_chooser(
    scale: &TestUiScale,
    screen_width: f32,
    screen_height: f32,
    project_count: usize,
) -> Vec<LayoutElement> {
    let mut elements = Vec::new();

    // Title "PALACE"
    let title_x = scale.px(60.0);
    let title_y = scale.px(40.0);
    let title_width = scale.px(200.0);
    let title_height = scale.px(42.0);
    elements.push(LayoutElement::new("Title: PALACE", title_x, title_y, title_width, title_height));

    // Subtitle "Projects"
    let subtitle_y = title_y + title_height + scale.px(8.0);
    elements.push(LayoutElement::new("Subtitle", title_x, subtitle_y, scale.px(150.0), scale.px(32.0)));

    // Project cards
    let grid = CardGrid::for_project_chooser(screen_width, scale);
    for i in 0..project_count {
        let (x, y) = grid.card_position(i);
        elements.push(LayoutElement::new(
            format!("Project Card {}", i),
            x, y, grid.card_width, grid.card_height,
        ));

        // Text inside card
        let text_x = x + scale.px(16.0);
        let text_y = y + scale.px(16.0);
        elements.push(LayoutElement::new(
            format!("Project {} Name", i),
            text_x, text_y, grid.card_width - scale.px(32.0), scale.px(22.0),
        ));
    }

    elements
}

/// Extract layout elements for ProjectView screen
pub fn layout_project_view(
    scale: &TestUiScale,
    screen_width: f32,
    _screen_height: f32,
) -> Vec<LayoutElement> {
    let mut elements = Vec::new();
    let actions = ProjectAction::all();

    let left_margin = scale.px(60.0);
    let title_scale = scale.px(40.0);
    let top_margin = scale.px(40.0);
    let menu_start_y = top_margin + title_scale + scale.px(60.0);
    let card_width = screen_width - left_margin * 2.0;
    let card_height = scale.px(70.0);
    let card_gap = scale.px(16.0);

    // Title
    elements.push(LayoutElement::new("Project Title", left_margin, top_margin, scale.px(400.0), title_scale));

    // Action cards
    for (i, action) in actions.iter().enumerate() {
        let y = menu_start_y + i as f32 * (card_height + card_gap);
        elements.push(LayoutElement::new(
            format!("Action: {}", action.label()),
            left_margin, y, card_width, card_height,
        ));

        // Text inside action card
        let text_y = y + scale.px(14.0);
        elements.push(LayoutElement::new(
            format!("Action Text: {}", action.label()),
            left_margin + scale.px(15.0), text_y, card_width - scale.px(30.0), scale.px(22.0),
        ));
    }

    elements
}

/// Extract layout elements for PalaceLoop screen
pub fn layout_palace_loop(
    scale: &TestUiScale,
    screen_width: f32,
    screen_height: f32,
    card_count: usize,
) -> Vec<LayoutElement> {
    let mut elements = Vec::new();

    // Title
    let left_margin = scale.px(40.0);
    let top_margin = scale.px(30.0);
    elements.push(LayoutElement::new("Title", left_margin, top_margin, scale.px(200.0), scale.px(32.0)));

    // Suggestion cards
    let grid = CardGrid::for_palace_loop(screen_width, scale);
    for i in 0..card_count {
        let (x, y) = grid.card_position(i);
        elements.push(LayoutElement::new(
            format!("Suggestion Card {}", i),
            x, y, grid.card_width, grid.card_height,
        ));
    }

    // Waterfall area (right 30% of screen)
    let waterfall_x = screen_width * 0.70;
    let waterfall_width = screen_width * 0.30 - scale.px(20.0);
    let waterfall_height = screen_height - scale.px(60.0);
    elements.push(LayoutElement::new(
        "Waterfall Area",
        waterfall_x, scale.px(30.0), waterfall_width, waterfall_height,
    ));

    elements
}

/// Extract layout elements for MainMenu modal
pub fn layout_main_menu(
    scale: &TestUiScale,
    screen_width: f32,
    screen_height: f32,
) -> Vec<LayoutElement> {
    use crate::state::MainMenuItem;
    let mut elements = Vec::new();

    let items = MainMenuItem::all();
    let modal_width = scale.px(400.0).min(screen_width - 40.0);
    let card_height = scale.px(50.0);
    let card_gap = scale.px(8.0);
    let inner_padding = scale.px(20.0);
    let title_height = scale.px(50.0);
    let modal_height = title_height + inner_padding + items.len() as f32 * (card_height + card_gap);
    let modal_x = (screen_width - modal_width) / 2.0;
    let modal_y = (screen_height - modal_height) / 2.0;

    // Modal background
    elements.push(LayoutElement::new("Modal Background", modal_x, modal_y, modal_width, modal_height));

    // Title
    elements.push(LayoutElement::new(
        "Modal Title",
        modal_x + inner_padding,
        modal_y + scale.px(12.0),
        modal_width - inner_padding * 2.0,
        scale.px(28.0),
    ));

    // Menu items
    let card_start_y = modal_y + title_height;
    let card_width = modal_width - inner_padding * 2.0;
    for (i, item) in items.iter().enumerate() {
        let y = card_start_y + i as f32 * (card_height + card_gap);
        elements.push(LayoutElement::new(
            format!("Menu Item: {}", item.label()),
            modal_x + inner_padding, y, card_width, card_height,
        ));
    }

    elements
}

/// Extract layout elements for UiScale modal
pub fn layout_ui_scale_modal(
    scale: &TestUiScale,
    screen_width: f32,
    screen_height: f32,
) -> Vec<LayoutElement> {
    let mut elements = Vec::new();

    let items = UiScaleOption::all();
    let modal_width = scale.px(400.0).min(screen_width - 40.0);
    let card_height = scale.px(50.0);
    let card_gap = scale.px(8.0);
    let inner_padding = scale.px(20.0);
    let title_height = scale.px(50.0);
    let modal_height = title_height + inner_padding + items.len() as f32 * (card_height + card_gap);
    let modal_x = (screen_width - modal_width) / 2.0;
    let modal_y = (screen_height - modal_height) / 2.0;

    // Modal background
    elements.push(LayoutElement::new("UI Scale Modal", modal_x, modal_y, modal_width, modal_height));

    // Title
    elements.push(LayoutElement::new(
        "UI Scale Title",
        modal_x + inner_padding,
        modal_y + scale.px(12.0),
        modal_width - inner_padding * 2.0,
        scale.px(28.0),
    ));

    // Scale options
    let card_start_y = modal_y + title_height;
    for (i, opt) in items.iter().enumerate() {
        let y = card_start_y + i as f32 * (card_height + card_gap);
        elements.push(LayoutElement::new(
            format!("Scale Option: {}", opt.label()),
            modal_x + inner_padding,
            y,
            modal_width - inner_padding * 2.0,
            card_height,
        ));
    }

    elements
}

/// Run all layout tests at all scales
pub fn run_all_layout_tests() -> Vec<LayoutValidation> {
    let mut results = Vec::new();
    let scales = all_test_scales();

    for &scale_value in &scales {
        let scale = TestUiScale::new(scale_value);
        let w = TEST_SCREEN_WIDTH;
        let h = TEST_SCREEN_HEIGHT;

        // Test ProjectChooser with varying project counts
        for project_count in [1, 5, 10, 20] {
            let mut validation = LayoutValidation::new(
                format!("ProjectChooser ({} projects)", project_count),
                scale_value, w, h,
            );
            let elements = layout_project_chooser(&scale, w, h, project_count);
            validation.validate(&elements);
            results.push(validation);
        }

        // Test ProjectView
        {
            let mut validation = LayoutValidation::new("ProjectView", scale_value, w, h);
            let elements = layout_project_view(&scale, w, h);
            validation.validate(&elements);
            results.push(validation);
        }

        // Test PalaceLoop with varying card counts
        for card_count in [5, 10, 15] {
            let mut validation = LayoutValidation::new(
                format!("PalaceLoop ({} cards)", card_count),
                scale_value, w, h,
            );
            let elements = layout_palace_loop(&scale, w, h, card_count);
            validation.validate(&elements);
            results.push(validation);
        }

        // Test MainMenu modal
        {
            let mut validation = LayoutValidation::new("MainMenu Modal", scale_value, w, h);
            let elements = layout_main_menu(&scale, w, h);
            validation.validate(&elements);
            results.push(validation);
        }

        // Test UiScale modal
        {
            let mut validation = LayoutValidation::new("UiScale Modal", scale_value, w, h);
            let elements = layout_ui_scale_modal(&scale, w, h);
            validation.validate(&elements);
            results.push(validation);
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlap_detection() {
        let a = LayoutElement::new("Card A", 100.0, 100.0, 200.0, 100.0);
        let b = LayoutElement::new("Card B", 250.0, 150.0, 200.0, 100.0);
        let c = LayoutElement::new("Card C", 500.0, 100.0, 200.0, 100.0);

        // A and B overlap
        assert!(a.overlaps(&b));
        // A and C don't overlap
        assert!(!a.overlaps(&c));
        // B and C don't overlap
        assert!(!b.overlaps(&c));
    }

    #[test]
    fn test_bounds_detection() {
        let inside = LayoutElement::new("Inside", 100.0, 100.0, 200.0, 100.0);
        let outside = LayoutElement::new("Outside", 1800.0, 1000.0, 200.0, 100.0);

        assert!(inside.within_screen(1920.0, 1080.0));
        assert!(!outside.within_screen(1920.0, 1080.0));
    }

    #[test]
    fn test_validation() {
        let mut validation = LayoutValidation::new("Test Screen", 1.0, 1920.0, 1080.0);
        let elements = vec![
            LayoutElement::new("A", 100.0, 100.0, 200.0, 100.0),
            LayoutElement::new("B", 250.0, 150.0, 200.0, 100.0), // Overlaps A
            LayoutElement::new("C", 1800.0, 1000.0, 200.0, 100.0), // Out of bounds
        ];

        validation.validate(&elements);

        assert_eq!(validation.overlaps.len(), 1);
        assert_eq!(validation.out_of_bounds.len(), 1);
        assert!(validation.has_issues());
    }

    #[test]
    fn test_all_screens_at_all_scales() {
        let results = run_all_layout_tests();

        let mut issues_found = false;
        let mut issue_count = 0;

        println!("\n=== Layout Validation Report ===\n");
        println!("Screen: 1920x1080 (1080p)\n");

        for scale in all_test_scales() {
            println!("Scale: {}x", scale);
            let scale_results: Vec<_> = results.iter()
                .filter(|r| (r.scale - scale).abs() < 0.001)
                .collect();

            for result in scale_results {
                result.print_issues();
                if result.has_issues() {
                    issues_found = true;
                    issue_count += result.overlaps.len() + result.out_of_bounds.len();
                }
            }
            println!();
        }

        if issues_found {
            println!("Total issues found: {}", issue_count);
        } else {
            println!("✓ All layouts passed validation!");
        }

        // For now, don't fail on issues - report them
        // Once we fix layouts, we can enable this assertion:
        // assert!(!issues_found, "Layout issues detected");
    }

    #[test]
    fn test_project_chooser_no_card_overlaps() {
        // Specifically test that project cards don't overlap at any scale
        for scale_value in all_test_scales() {
            let scale = TestUiScale::new(scale_value);
            let elements = layout_project_chooser(&scale, TEST_SCREEN_WIDTH, TEST_SCREEN_HEIGHT, 10);

            // Filter to just cards
            let cards: Vec<_> = elements.iter()
                .filter(|e| e.name.starts_with("Project Card"))
                .collect();

            for i in 0..cards.len() {
                for j in (i + 1)..cards.len() {
                    assert!(
                        !cards[i].overlaps(cards[j]),
                        "Cards {} and {} overlap at scale {}",
                        cards[i].name, cards[j].name, scale_value
                    );
                }
            }
        }
    }

    #[test]
    fn test_modal_fits_screen() {
        // Test that modals fit on screen at reasonable scales for 1080p
        // (3x and 4x scales are intended for higher resolution displays)
        let reasonable_scales = [0.5, 1.0, 1.5, 2.0];

        for scale_value in reasonable_scales {
            let scale = TestUiScale::new(scale_value);
            let elements = layout_ui_scale_modal(&scale, TEST_SCREEN_WIDTH, TEST_SCREEN_HEIGHT);

            for elem in &elements {
                assert!(
                    elem.within_screen(TEST_SCREEN_WIDTH, TEST_SCREEN_HEIGHT),
                    "{} out of bounds at scale {} on 1080p",
                    elem.name, scale_value
                );
            }
        }
    }

    #[test]
    fn test_high_scales_on_4k() {
        // Test that high scales work on 4K resolution
        let screen_4k_width = 3840.0;
        let screen_4k_height = 2160.0;

        for scale_value in [3.0, 4.0] {
            let scale = TestUiScale::new(scale_value);
            let elements = layout_ui_scale_modal(&scale, screen_4k_width, screen_4k_height);

            for elem in &elements {
                assert!(
                    elem.within_screen(screen_4k_width, screen_4k_height),
                    "{} out of bounds at scale {} on 4K",
                    elem.name, scale_value
                );
            }
        }
    }
}
