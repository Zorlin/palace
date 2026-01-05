//! Text rendering abstraction using glyphon (cosmic-text based)
//!
//! Provides a simple API for rendering text with color emoji support.

use glyphon::{
    Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport, Wrap,
    cosmic_text::Align,
};

/// A text rendering request - lightweight data before buffer creation
#[derive(Clone)]
pub struct TextRequest {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub color: [f32; 4],
    pub bounds_width: Option<f32>,
    pub bounds_height: Option<f32>,
}

impl TextRequest {
    pub fn new(text: impl Into<String>, x: f32, y: f32, scale: f32, color: [f32; 4]) -> Self {
        Self {
            text: text.into(),
            x,
            y,
            scale,
            color,
            bounds_width: None,
            bounds_height: None,
        }
    }

    pub fn with_bounds(mut self, width: f32, height: f32) -> Self {
        self.bounds_width = Some(width);
        self.bounds_height = Some(height);
        self
    }
}

/// Collects text rendering requests for a frame
#[derive(Default)]
pub struct TextQueue {
    requests: Vec<TextRequest>,
}

impl TextQueue {
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
        }
    }

    pub fn add(&mut self, request: TextRequest) {
        self.requests.push(request);
    }

    pub fn push(&mut self, text: impl Into<String>, x: f32, y: f32, scale: f32, color: [f32; 4]) {
        self.requests.push(TextRequest::new(text, x, y, scale, color));
    }

    pub fn push_bounded(
        &mut self,
        text: impl Into<String>,
        x: f32,
        y: f32,
        scale: f32,
        color: [f32; 4],
        bounds_width: f32,
        bounds_height: f32,
    ) {
        self.requests.push(
            TextRequest::new(text, x, y, scale, color).with_bounds(bounds_width, bounds_height),
        );
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn clear(&mut self) {
        self.requests.clear();
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Consume the queue and return the requests
    pub fn take(&mut self) -> Vec<TextRequest> {
        std::mem::take(&mut self.requests)
    }

    /// Get requests as slice
    pub fn requests(&self) -> &[TextRequest] {
        &self.requests
    }
}

/// Prepared text buffers ready for rendering
/// Must be kept alive until after render() is called
pub struct PreparedText {
    buffers: Vec<Buffer>,
    requests: Vec<TextRequest>,
}

impl PreparedText {
    /// Create buffers for all text requests
    pub fn prepare(
        requests: Vec<TextRequest>,
        font_system: &mut FontSystem,
        screen_width: f32,
    ) -> Self {
        let mut buffers = Vec::with_capacity(requests.len());

        for req in &requests {
            let metrics = Metrics::new(req.scale, req.scale * 1.2);
            let mut buffer = Buffer::new(font_system, metrics);

            let bounds_width = req.bounds_width.unwrap_or(screen_width - req.x);
            buffer.set_size(font_system, Some(bounds_width), None);
            let attrs = Attrs::new().family(Family::Monospace);
            buffer.set_text(
                font_system,
                &req.text,
                &attrs,
                Shaping::Advanced,
                Some(Align::Left),
            );
            buffer.set_wrap(font_system, Wrap::Word);
            buffer.shape_until_scroll(font_system, false);

            buffers.push(buffer);
        }

        Self { buffers, requests }
    }

    /// Build text areas referencing the buffers
    /// The returned TextAreas borrow from self, so self must outlive them
    pub fn text_areas(&self, screen_width: f32, screen_height: f32) -> Vec<TextArea<'_>> {
        self.requests
            .iter()
            .zip(self.buffers.iter())
            .map(|(req, buffer)| {
                let color = Color::rgba(
                    (req.color[0] * 255.0) as u8,
                    (req.color[1] * 255.0) as u8,
                    (req.color[2] * 255.0) as u8,
                    (req.color[3] * 255.0) as u8,
                );

                let bounds_width = req.bounds_width.unwrap_or(screen_width - req.x);
                let bounds_height = req.bounds_height.unwrap_or(screen_height - req.y);

                TextArea {
                    buffer,
                    left: req.x,
                    top: req.y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: req.x as i32,
                        top: req.y as i32,
                        right: (req.x + bounds_width) as i32,
                        bottom: (req.y + bounds_height) as i32,
                    },
                    default_color: color,
                    custom_glyphs: &[],
                }
            })
            .collect()
    }

    /// Prepare the text renderer with all text areas
    pub fn upload(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        font_system: &mut FontSystem,
        swash_cache: &mut SwashCache,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_renderer: &mut TextRenderer,
        screen_width: f32,
        screen_height: f32,
    ) -> Result<(), glyphon::PrepareError> {
        let text_areas = self.text_areas(screen_width, screen_height);
        text_renderer.prepare(
            device,
            queue,
            font_system,
            atlas,
            viewport,
            text_areas,
            swash_cache,
        )
    }
}

/// Measure text dimensions for a given width constraint
/// Returns (width, height, line_count)
pub fn measure_text(
    font_system: &mut FontSystem,
    text: &str,
    font_size: f32,
    max_width: f32,
) -> (f32, f32, usize) {
    let line_height = font_size * 1.2;
    let metrics = Metrics::new(font_size, line_height);
    let mut buffer = Buffer::new(font_system, metrics);

    buffer.set_size(font_system, Some(max_width), None);
    let attrs = Attrs::new().family(Family::Monospace);
    buffer.set_text(font_system, text, &attrs, Shaping::Advanced, Some(Align::Left));
    buffer.set_wrap(font_system, Wrap::Word);
    buffer.shape_until_scroll(font_system, false);

    // Count visual lines from layout (not logical lines from buffer.lines)
    let mut visual_line_count = 0usize;
    let mut max_line_width: f32 = 0.0;
    for line in buffer.lines.iter() {
        if let Some(layout) = line.layout_opt() {
            for layout_line in layout.iter() {
                visual_line_count += 1;
                let line_width = layout_line.w;
                if line_width > max_line_width {
                    max_line_width = line_width;
                }
            }
        }
    }

    // Ensure at least 1 line for non-empty text
    let visual_line_count = visual_line_count.max(if text.is_empty() { 0 } else { 1 });
    let height = visual_line_count as f32 * line_height;
    (max_line_width, height, visual_line_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_wrapping() {
        let mut font_system = FontSystem::new();

        // Short text - should be 1 line
        let short = "Hello world";
        let (_, _, lines) = measure_text(&mut font_system, short, 16.0, 500.0);
        assert_eq!(lines, 1, "Short text should be 1 line");

        // Long text with narrow width - should wrap
        let long = "I'll explore the project structure and key files to understand this Rust project and provide suggestions.";
        let (_, _, lines) = measure_text(&mut font_system, long, 16.0, 200.0);
        assert!(lines > 1, "Long text at 200px width should wrap, got {} lines", lines);

        // Same long text with wide width - should be fewer lines
        let (_, _, lines_wide) = measure_text(&mut font_system, long, 16.0, 800.0);
        assert!(lines_wide < lines, "Wide width should have fewer lines than narrow");

        // Very narrow - should wrap a lot
        let (_, _, lines_narrow) = measure_text(&mut font_system, long, 16.0, 100.0);
        assert!(lines_narrow > lines, "Very narrow should wrap more, got {} vs {}", lines_narrow, lines);
    }

    #[test]
    fn test_wrap_height_calculation() {
        let mut font_system = FontSystem::new();
        let font_size = 16.0;
        let line_height = font_size * 1.2;

        let text = "This is a test string that should definitely wrap when given a narrow width constraint.";
        let (_, height, lines) = measure_text(&mut font_system, text, font_size, 150.0);

        let expected_height = lines as f32 * line_height;
        assert!((height - expected_height).abs() < 0.01, "Height {} should match {} lines * {}", height, lines, line_height);
    }
}
