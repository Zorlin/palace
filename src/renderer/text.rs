//! Text rendering abstraction using glyphon (cosmic-text based)
//!
//! Provides a simple API for rendering text with color emoji support.
//! Includes basic markdown rendering support (bold, italic, code, headings, bullets).

use glyphon::{
    Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport, Wrap, Weight, Style,
    cosmic_text::Align,
};
use pulldown_cmark::{Event, Parser, Tag, TagEnd};

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
    pub scroll_offset: f32,
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
            scroll_offset: 0.0,
        }
    }

    pub fn with_bounds(mut self, width: f32, height: f32) -> Self {
        self.bounds_width = Some(width);
        self.bounds_height = Some(height);
        self
    }

    pub fn with_scroll(mut self, scroll: f32) -> Self {
        self.scroll_offset = scroll;
        self
    }
}

/// Collects text rendering requests for a frame
#[derive(Default)]
pub struct TextQueue {
    requests: Vec<TextRequest>,
    markdown_requests: Vec<MarkdownRequest>,
}

impl TextQueue {
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
            markdown_requests: Vec::new(),
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

    pub fn push_bounded_scroll(
        &mut self,
        text: impl Into<String>,
        x: f32,
        y: f32,
        scale: f32,
        color: [f32; 4],
        bounds_width: f32,
        bounds_height: f32,
        scroll_offset: f32,
    ) {
        self.requests.push(
            TextRequest::new(text, x, y, scale, color)
                .with_bounds(bounds_width, bounds_height)
                .with_scroll(scroll_offset),
        );
    }

    /// Push markdown text with bounds (no scroll)
    pub fn push_markdown_bounded(
        &mut self,
        markdown: &str,
        x: f32,
        y: f32,
        scale: f32,
        color: [f32; 4],
        bounds_width: f32,
        bounds_height: f32,
    ) {
        self.markdown_requests.push(
            MarkdownRequest::new(markdown, x, y, scale, color)
                .with_bounds(bounds_width, bounds_height),
        );
    }

    /// Push markdown text with bounds and scroll
    pub fn push_markdown_scroll(
        &mut self,
        markdown: &str,
        x: f32,
        y: f32,
        scale: f32,
        color: [f32; 4],
        bounds_width: f32,
        bounds_height: f32,
        scroll_offset: f32,
    ) {
        self.markdown_requests.push(
            MarkdownRequest::new(markdown, x, y, scale, color)
                .with_bounds(bounds_width, bounds_height)
                .with_scroll(scroll_offset),
        );
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty() && self.markdown_requests.is_empty()
    }

    pub fn clear(&mut self) {
        self.requests.clear();
        self.markdown_requests.clear();
    }

    pub fn len(&self) -> usize {
        self.requests.len() + self.markdown_requests.len()
    }

    /// Consume the queue and return plain text requests
    pub fn take(&mut self) -> Vec<TextRequest> {
        std::mem::take(&mut self.requests)
    }

    /// Consume the queue and return markdown requests
    pub fn take_markdown(&mut self) -> Vec<MarkdownRequest> {
        std::mem::take(&mut self.markdown_requests)
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
    markdown_buffers: Vec<Buffer>,
    markdown_requests: Vec<MarkdownRequest>,
}

impl PreparedText {
    /// Create buffers for all text requests (plain and markdown)
    pub fn prepare(
        requests: Vec<TextRequest>,
        markdown_requests: Vec<MarkdownRequest>,
        font_system: &mut FontSystem,
        screen_width: f32,
    ) -> Self {
        // Prepare plain text buffers
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

        // Prepare markdown buffers with rich text
        let mut markdown_buffers = Vec::with_capacity(markdown_requests.len());
        for req in &markdown_requests {
            let metrics = Metrics::new(req.base_scale, req.base_scale * 1.2);
            let mut buffer = Buffer::new(font_system, metrics);

            let bounds_width = req.bounds_width.unwrap_or(screen_width - req.x);
            buffer.set_size(font_system, Some(bounds_width), None);

            // Build rich text from parsed markdown spans
            let rich_spans: Vec<(&str, Attrs)> = req
                .spans
                .iter()
                .map(|span| (span.text.as_str(), span.attrs()))
                .collect();

            let default_attrs = Attrs::new().family(Family::Monospace);
            buffer.set_rich_text(
                font_system,
                rich_spans.into_iter(),
                &default_attrs,
                Shaping::Advanced,
                Some(Align::Left),
            );
            buffer.set_wrap(font_system, Wrap::Word);
            buffer.shape_until_scroll(font_system, false);

            markdown_buffers.push(buffer);
        }

        Self { buffers, requests, markdown_buffers, markdown_requests }
    }

    /// Build text areas referencing the buffers
    /// The returned TextAreas borrow from self, so self must outlive them
    pub fn text_areas(&self, screen_width: f32, screen_height: f32) -> Vec<TextArea<'_>> {
        // Plain text areas
        let plain_areas = self.requests
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
                let scrolled_top = req.y - req.scroll_offset;

                TextArea {
                    buffer,
                    left: req.x,
                    top: scrolled_top,
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
            });

        // Markdown text areas
        let markdown_areas = self.markdown_requests
            .iter()
            .zip(self.markdown_buffers.iter())
            .map(|(req, buffer)| {
                let color = Color::rgba(
                    (req.color[0] * 255.0) as u8,
                    (req.color[1] * 255.0) as u8,
                    (req.color[2] * 255.0) as u8,
                    (req.color[3] * 255.0) as u8,
                );

                let bounds_width = req.bounds_width.unwrap_or(screen_width - req.x);
                let bounds_height = req.bounds_height.unwrap_or(screen_height - req.y);
                let scrolled_top = req.y - req.scroll_offset;

                TextArea {
                    buffer,
                    left: req.x,
                    top: scrolled_top,
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
            });

        plain_areas.chain(markdown_areas).collect()
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

/// A styled text span from markdown parsing
#[derive(Clone, Debug)]
pub struct StyledSpan {
    pub text: String,
    pub bold: bool,
    pub italic: bool,
    pub code: bool,
    pub heading_level: u8, // 0 = normal, 1-3 = heading
    pub list_depth: u8,    // 0 = none, 1+ = bullet depth
}

impl StyledSpan {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            bold: false,
            italic: false,
            code: false,
            heading_level: 0,
            list_depth: 0,
        }
    }

    /// Build glyphon Attrs for this span
    pub fn attrs(&self) -> Attrs<'static> {
        let mut attrs = Attrs::new();

        // Bold
        if self.bold || self.heading_level > 0 {
            attrs = attrs.weight(Weight::BOLD);
        }

        // Italic
        if self.italic {
            attrs = attrs.style(Style::Italic);
        }

        // Code uses monospace (already default, but explicit)
        if self.code {
            attrs = attrs.family(Family::Monospace);
        } else {
            attrs = attrs.family(Family::Monospace); // Keep monospace for all
        }

        attrs
    }

    /// Get font size multiplier for headings
    pub fn size_multiplier(&self) -> f32 {
        match self.heading_level {
            1 => 1.5,
            2 => 1.3,
            3 => 1.15,
            _ => 1.0,
        }
    }
}

/// Parse markdown into styled spans
/// Supports: **bold**, *italic*, `code`, # headings, - bullets
pub fn parse_markdown(input: &str) -> Vec<StyledSpan> {
    let mut spans = Vec::new();
    let parser = Parser::new(input);

    let mut bold = false;
    let mut italic = false;
    let mut code = false;
    let mut heading_level = 0u8;
    let mut list_depth = 0u8;
    let mut pending_newline = false;

    for event in parser {
        match event {
            Event::Start(Tag::Strong) => bold = true,
            Event::End(TagEnd::Strong) => bold = false,

            Event::Start(Tag::Emphasis) => italic = true,
            Event::End(TagEnd::Emphasis) => italic = false,

            Event::Start(Tag::CodeBlock(_)) | Event::Code(_) => {
                if let Event::Code(text) = event {
                    // Inline code
                    let mut span = StyledSpan::new(text.to_string());
                    span.code = true;
                    span.bold = bold;
                    span.italic = italic;
                    spans.push(span);
                    continue;
                }
                code = true;
            }
            Event::End(TagEnd::CodeBlock) => {
                code = false;
                pending_newline = true;
            }

            Event::Start(Tag::Heading { level, .. }) => {
                heading_level = level as u8;
                if pending_newline {
                    spans.push(StyledSpan::new("\n"));
                    pending_newline = false;
                }
            }
            Event::End(TagEnd::Heading(_)) => {
                heading_level = 0;
                pending_newline = true;
            }

            Event::Start(Tag::List(_)) => {
                list_depth += 1;
            }
            Event::End(TagEnd::List(_)) => {
                list_depth = list_depth.saturating_sub(1);
                if list_depth == 0 {
                    pending_newline = true;
                }
            }

            Event::Start(Tag::Item) => {
                if pending_newline && !spans.is_empty() {
                    spans.push(StyledSpan::new("\n"));
                    pending_newline = false;
                }
                // Add bullet prefix
                let indent = "  ".repeat(list_depth.saturating_sub(1) as usize);
                let mut span = StyledSpan::new(format!("{}• ", indent));
                span.list_depth = list_depth;
                spans.push(span);
            }
            Event::End(TagEnd::Item) => {
                pending_newline = true;
            }

            Event::Start(Tag::Paragraph) => {
                if pending_newline && !spans.is_empty() {
                    spans.push(StyledSpan::new("\n\n"));
                    pending_newline = false;
                }
            }
            Event::End(TagEnd::Paragraph) => {
                pending_newline = true;
            }

            Event::Text(text) => {
                if pending_newline && !spans.is_empty() && list_depth == 0 {
                    spans.push(StyledSpan::new("\n"));
                    pending_newline = false;
                }

                let mut span = StyledSpan::new(text.to_string());
                span.bold = bold;
                span.italic = italic;
                span.code = code;
                span.heading_level = heading_level;
                span.list_depth = list_depth;
                spans.push(span);
            }

            Event::SoftBreak => {
                spans.push(StyledSpan::new(" "));
            }

            Event::HardBreak => {
                spans.push(StyledSpan::new("\n"));
            }

            _ => {}
        }
    }

    // Merge adjacent spans with same style for efficiency
    merge_spans(spans)
}

/// Merge adjacent spans with identical styling
fn merge_spans(spans: Vec<StyledSpan>) -> Vec<StyledSpan> {
    let mut merged: Vec<StyledSpan> = Vec::new();

    for span in spans {
        if let Some(last) = merged.last_mut() {
            if last.bold == span.bold
                && last.italic == span.italic
                && last.code == span.code
                && last.heading_level == span.heading_level
                && last.list_depth == span.list_depth
            {
                last.text.push_str(&span.text);
                continue;
            }
        }
        merged.push(span);
    }

    merged
}

/// Convert styled spans back to plain text (for layout/measurement)
pub fn spans_to_plain_text(spans: &[StyledSpan]) -> String {
    spans.iter().map(|s| s.text.as_str()).collect()
}

/// A markdown text rendering request
#[derive(Clone)]
pub struct MarkdownRequest {
    pub spans: Vec<StyledSpan>,
    pub x: f32,
    pub y: f32,
    pub base_scale: f32,
    pub color: [f32; 4],
    pub bounds_width: Option<f32>,
    pub bounds_height: Option<f32>,
    pub scroll_offset: f32,
}

impl MarkdownRequest {
    pub fn new(markdown: &str, x: f32, y: f32, base_scale: f32, color: [f32; 4]) -> Self {
        Self {
            spans: parse_markdown(markdown),
            x,
            y,
            base_scale,
            color,
            bounds_width: None,
            bounds_height: None,
            scroll_offset: 0.0,
        }
    }

    pub fn with_bounds(mut self, width: f32, height: f32) -> Self {
        self.bounds_width = Some(width);
        self.bounds_height = Some(height);
        self
    }

    pub fn with_scroll(mut self, scroll: f32) -> Self {
        self.scroll_offset = scroll;
        self
    }

    /// Convert to plain text for compatibility
    pub fn as_plain_text(&self) -> String {
        spans_to_plain_text(&self.spans)
    }
}

/// Prepared markdown buffer ready for rendering
pub struct PreparedMarkdown {
    buffer: Buffer,
    request: MarkdownRequest,
}

impl PreparedMarkdown {
    /// Create buffer for markdown request with rich text
    pub fn prepare(
        request: MarkdownRequest,
        font_system: &mut FontSystem,
        screen_width: f32,
    ) -> Self {
        let metrics = Metrics::new(request.base_scale, request.base_scale * 1.2);
        let mut buffer = Buffer::new(font_system, metrics);

        let bounds_width = request.bounds_width.unwrap_or(screen_width - request.x);
        buffer.set_size(font_system, Some(bounds_width), None);

        // Build rich text from spans
        let rich_spans: Vec<(&str, Attrs)> = request
            .spans
            .iter()
            .map(|span| (span.text.as_str(), span.attrs()))
            .collect();

        let default_attrs = Attrs::new().family(Family::Monospace);
        buffer.set_rich_text(
            font_system,
            rich_spans.into_iter(),
            &default_attrs,
            Shaping::Advanced,
            Some(Align::Left),
        );
        buffer.set_wrap(font_system, Wrap::Word);
        buffer.shape_until_scroll(font_system, false);

        Self { buffer, request }
    }

    /// Build text area for rendering
    pub fn text_area(&self, screen_width: f32, screen_height: f32) -> TextArea<'_> {
        let req = &self.request;
        let color = Color::rgba(
            (req.color[0] * 255.0) as u8,
            (req.color[1] * 255.0) as u8,
            (req.color[2] * 255.0) as u8,
            (req.color[3] * 255.0) as u8,
        );

        let bounds_width = req.bounds_width.unwrap_or(screen_width - req.x);
        let bounds_height = req.bounds_height.unwrap_or(screen_height - req.y);
        let scrolled_top = req.y - req.scroll_offset;

        TextArea {
            buffer: &self.buffer,
            left: req.x,
            top: scrolled_top,
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
    }
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
