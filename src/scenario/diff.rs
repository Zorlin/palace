//! Diff computation and visualization for scenario correction
//!
//! Provides intelligent synchronized scrolling with cursor-centric reflow,
//! collapsed markers for added/removed blocks, and chunk-based navigation.

use std::collections::HashSet;

/// A chunk of diff representing a contiguous section of changes
#[derive(Debug, Clone)]
pub struct DiffChunk {
    /// Index of this chunk in the diff
    pub index: usize,
    /// What kind of change this represents
    pub kind: DiffKind,
    /// Lines from the original file
    pub original_lines: Vec<String>,
    /// Lines from the corrected file
    pub corrected_lines: Vec<String>,
    /// Starting line number in original (1-indexed for display)
    pub line_start_original: usize,
    /// Starting line number in corrected (1-indexed for display)
    pub line_start_corrected: usize,
    /// Explanation for this specific change (from LLM)
    pub explanation: Option<String>,
}

impl DiffChunk {
    /// Get the height of this chunk in lines
    pub fn height_lines(&self) -> usize {
        self.original_lines.len().max(self.corrected_lines.len())
    }

    /// Check if this chunk has content on the original side
    pub fn has_original(&self) -> bool {
        !self.original_lines.is_empty()
    }

    /// Check if this chunk has content on the corrected side
    pub fn has_corrected(&self) -> bool {
        !self.corrected_lines.is_empty()
    }
}

/// Type of diff chunk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffKind {
    /// Unchanged context lines
    Context,
    /// Lines added in corrected (not in original)
    Added,
    /// Lines removed from original (not in corrected)
    Removed,
    /// Lines modified (different content on each side)
    Modified,
}

impl DiffKind {
    /// Get the background color for this diff kind (RGBA)
    pub fn background_color(&self) -> [f32; 4] {
        match self {
            DiffKind::Context => [0.0, 0.0, 0.0, 0.0],      // Transparent
            DiffKind::Added => [0.1, 0.4, 0.1, 0.3],        // Green
            DiffKind::Removed => [0.4, 0.1, 0.1, 0.3],      // Red
            DiffKind::Modified => [0.4, 0.4, 0.1, 0.3],     // Yellow
        }
    }

    /// Get the marker bar color for collapsed sections
    pub fn marker_color(&self) -> [f32; 4] {
        match self {
            DiffKind::Context => [0.3, 0.3, 0.3, 0.5],
            DiffKind::Added => [0.1, 0.6, 0.1, 0.8],
            DiffKind::Removed => [0.6, 0.1, 0.1, 0.8],
            DiffKind::Modified => [0.6, 0.6, 0.1, 0.8],
        }
    }
}

/// Diff viewer state with cursor-centric reflow
#[derive(Debug, Clone)]
pub struct DiffViewer {
    /// Original content
    pub original: String,
    /// Corrected content
    pub corrected: String,
    /// Overall explanation from LLM
    pub explanation: String,
    /// Computed diff chunks
    pub chunks: Vec<DiffChunk>,
    /// Which chunk has focus (cursor)
    pub cursor_chunk: usize,
    /// Single scroll offset for synchronized scrolling
    pub scroll_offset: f32,
    /// Which chunks are collapsed
    pub collapsed: HashSet<usize>,
    /// Viewport height (set during render)
    pub viewport_height: f32,
    /// Line height in pixels
    pub line_height: f32,
    /// Whether awaiting user feedback
    pub awaiting_feedback: bool,
    /// Current iteration count
    pub iteration: usize,
}

impl DiffViewer {
    /// Create a new diff viewer from original and corrected content
    pub fn new(original: String, corrected: String, explanation: String) -> Self {
        let chunks = compute_diff(&original, &corrected);

        Self {
            original,
            corrected,
            explanation,
            chunks,
            cursor_chunk: 0,
            scroll_offset: 0.0,
            collapsed: HashSet::new(),
            viewport_height: 600.0,
            line_height: 20.0,
            awaiting_feedback: false,
            iteration: 1,
        }
    }

    /// Update with new corrected content (for iterative refinement)
    pub fn update_correction(&mut self, corrected: String, explanation: String) {
        self.corrected = corrected.clone();
        self.explanation = explanation;
        self.chunks = compute_diff(&self.original, &corrected);
        self.cursor_chunk = 0;
        self.scroll_offset = 0.0;
        self.collapsed.clear();
        self.iteration += 1;
    }

    /// Get total content height in pixels
    pub fn total_height(&self) -> f32 {
        self.chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                if self.collapsed.contains(&i) {
                    self.collapsed_height()
                } else {
                    chunk.height_lines() as f32 * self.line_height
                }
            })
            .sum()
    }

    /// Height of a collapsed marker bar
    pub fn collapsed_height(&self) -> f32 {
        4.0
    }

    /// Maximum scroll offset
    pub fn max_scroll(&self) -> f32 {
        (self.total_height() - self.viewport_height).max(0.0)
    }

    /// Scroll by a delta amount
    pub fn scroll_by(&mut self, delta: f32) {
        self.scroll_offset = (self.scroll_offset + delta)
            .max(0.0)
            .min(self.max_scroll());
    }

    /// Navigate to next/previous chunk
    pub fn navigate_chunk(&mut self, direction: i32) {
        let new_idx = (self.cursor_chunk as i32 + direction)
            .max(0)
            .min(self.chunks.len().saturating_sub(1) as i32) as usize;

        if new_idx != self.cursor_chunk {
            self.cursor_chunk = new_idx;
            self.scroll_to_center_chunk(new_idx);
        }
    }

    /// Navigate to next change (skip context)
    pub fn navigate_next_change(&mut self) {
        for i in (self.cursor_chunk + 1)..self.chunks.len() {
            if self.chunks[i].kind != DiffKind::Context {
                self.cursor_chunk = i;
                self.scroll_to_center_chunk(i);
                return;
            }
        }
    }

    /// Navigate to previous change (skip context)
    pub fn navigate_prev_change(&mut self) {
        for i in (0..self.cursor_chunk).rev() {
            if self.chunks[i].kind != DiffKind::Context {
                self.cursor_chunk = i;
                self.scroll_to_center_chunk(i);
                return;
            }
        }
    }

    /// Toggle collapse on current chunk
    pub fn toggle_collapse(&mut self) {
        if self.collapsed.contains(&self.cursor_chunk) {
            self.collapsed.remove(&self.cursor_chunk);
        } else {
            self.collapsed.insert(self.cursor_chunk);
        }
    }

    /// Collapse all context chunks
    pub fn collapse_all_context(&mut self) {
        for (i, chunk) in self.chunks.iter().enumerate() {
            if chunk.kind == DiffKind::Context {
                self.collapsed.insert(i);
            }
        }
    }

    /// Expand all chunks
    pub fn expand_all(&mut self) {
        self.collapsed.clear();
    }

    /// Get Y position of a chunk (accounting for collapsed state)
    pub fn chunk_y(&self, chunk_idx: usize) -> f32 {
        let mut y = 0.0;
        for (i, chunk) in self.chunks.iter().enumerate() {
            if i == chunk_idx {
                return y;
            }
            if self.collapsed.contains(&i) {
                y += self.collapsed_height();
            } else {
                y += chunk.height_lines() as f32 * self.line_height;
            }
        }
        y
    }

    /// Scroll to center a specific chunk in the viewport
    fn scroll_to_center_chunk(&mut self, chunk_idx: usize) {
        let chunk_y = self.chunk_y(chunk_idx);
        let chunk_height = if self.collapsed.contains(&chunk_idx) {
            self.collapsed_height()
        } else {
            self.chunks[chunk_idx].height_lines() as f32 * self.line_height
        };

        // Target: center the chunk vertically
        let visible_center = self.viewport_height / 2.0;
        let target_scroll = chunk_y + chunk_height / 2.0 - visible_center;

        self.scroll_offset = target_scroll.max(0.0).min(self.max_scroll());
    }

    /// Get the current chunk (at cursor)
    pub fn current_chunk(&self) -> Option<&DiffChunk> {
        self.chunks.get(self.cursor_chunk)
    }

    /// Count of changes (non-context chunks)
    pub fn change_count(&self) -> usize {
        self.chunks
            .iter()
            .filter(|c| c.kind != DiffKind::Context)
            .count()
    }

    /// Current change index (1-indexed, for display)
    pub fn current_change_index(&self) -> Option<usize> {
        let mut count = 0;
        for (i, chunk) in self.chunks.iter().enumerate() {
            if chunk.kind != DiffKind::Context {
                count += 1;
                if i == self.cursor_chunk {
                    return Some(count);
                }
            }
        }
        None
    }
}

/// Compute diff chunks from original and corrected content
pub fn compute_diff(original: &str, corrected: &str) -> Vec<DiffChunk> {
    let _original_lines: Vec<&str> = original.lines().collect();
    let _corrected_lines: Vec<&str> = corrected.lines().collect();

    // Use patience diff algorithm for better results
    let diff = similar::TextDiff::from_lines(original, corrected);

    let mut chunks = Vec::new();
    let mut current_chunk: Option<DiffChunk> = None;
    let mut chunk_index = 0;
    let mut orig_line = 1;
    let mut corr_line = 1;

    for change in diff.iter_all_changes() {
        let kind = match change.tag() {
            similar::ChangeTag::Equal => DiffKind::Context,
            similar::ChangeTag::Insert => DiffKind::Added,
            similar::ChangeTag::Delete => DiffKind::Removed,
        };

        let line = change.value().trim_end_matches('\n').to_string();

        // Check if we need to start a new chunk
        let should_start_new = match &current_chunk {
            None => true,
            Some(chunk) => {
                // Start new chunk if kind changes (except Equal->Equal)
                chunk.kind != kind
            }
        };

        if should_start_new {
            // Push previous chunk if exists
            if let Some(chunk) = current_chunk.take() {
                chunks.push(chunk);
                chunk_index += 1;
            }

            // Start new chunk
            current_chunk = Some(DiffChunk {
                index: chunk_index,
                kind,
                original_lines: Vec::new(),
                corrected_lines: Vec::new(),
                line_start_original: orig_line,
                line_start_corrected: corr_line,
                explanation: None,
            });
        }

        // Add line to current chunk
        if let Some(ref mut chunk) = current_chunk {
            match kind {
                DiffKind::Context => {
                    chunk.original_lines.push(line.clone());
                    chunk.corrected_lines.push(line);
                    orig_line += 1;
                    corr_line += 1;
                }
                DiffKind::Added => {
                    chunk.corrected_lines.push(line);
                    corr_line += 1;
                }
                DiffKind::Removed => {
                    chunk.original_lines.push(line);
                    orig_line += 1;
                }
                DiffKind::Modified => {
                    // Modified chunks are created by merging adjacent Add/Remove
                    unreachable!("Modified chunks are created in post-processing");
                }
            }
        }
    }

    // Push final chunk
    if let Some(chunk) = current_chunk {
        chunks.push(chunk);
    }

    // Post-process: merge adjacent Removed+Added into Modified
    merge_adjacent_changes(&mut chunks);

    // Ensure indices are correct
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.index = i;
    }

    chunks
}

/// Merge adjacent Removed+Added chunks into Modified chunks
fn merge_adjacent_changes(chunks: &mut Vec<DiffChunk>) {
    let mut i = 0;
    while i + 1 < chunks.len() {
        let should_merge = matches!(
            (&chunks[i].kind, &chunks[i + 1].kind),
            (DiffKind::Removed, DiffKind::Added)
        );

        if should_merge {
            // Merge into a Modified chunk
            let next = chunks.remove(i + 1);
            chunks[i].kind = DiffKind::Modified;
            chunks[i].corrected_lines = next.corrected_lines;
        } else {
            i += 1;
        }
    }
}

/// Add chunk-level explanations from LLM response
pub fn add_explanations(chunks: &mut [DiffChunk], explanations: &[(usize, String)]) {
    for (chunk_idx, explanation) in explanations {
        if let Some(chunk) = chunks.get_mut(*chunk_idx) {
            chunk.explanation = Some(explanation.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_diff_simple() {
        let original = "line1\nline2\nline3";
        let corrected = "line1\nmodified\nline3";

        let chunks = compute_diff(original, corrected);

        // Should have: context, modified, context
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_compute_diff_addition() {
        // Use trailing newlines to avoid edge case where last line differs due to newline
        let original = "line1\nline2\n";
        let corrected = "line1\nline2\nline3\n";

        let chunks = compute_diff(original, corrected);

        // Should have added chunk
        assert!(chunks.iter().any(|c| c.kind == DiffKind::Added));
    }

    #[test]
    fn test_compute_diff_removal() {
        let original = "line1\nline2\nline3";
        let corrected = "line1\nline3";

        let chunks = compute_diff(original, corrected);

        // Should have removed chunk
        assert!(chunks.iter().any(|c| c.kind == DiffKind::Removed));
    }

    #[test]
    fn test_diff_viewer_navigation() {
        let viewer = DiffViewer::new(
            "a\nb\nc".to_string(),
            "a\nB\nc".to_string(),
            "Changed b to B".to_string(),
        );

        assert_eq!(viewer.cursor_chunk, 0);
        assert!(viewer.chunks.len() > 0);
    }

    #[test]
    fn test_diff_viewer_collapse() {
        let mut viewer = DiffViewer::new(
            "context\nchanged\ncontext".to_string(),
            "context\nCHANGED\ncontext".to_string(),
            "Test".to_string(),
        );

        assert!(!viewer.collapsed.contains(&0));
        viewer.toggle_collapse();
        assert!(viewer.collapsed.contains(&0));
        viewer.toggle_collapse();
        assert!(!viewer.collapsed.contains(&0));
    }

    #[test]
    fn test_merge_adjacent_changes() {
        let original = "old line";
        let corrected = "new line";

        let chunks = compute_diff(original, corrected);

        // Should be a single Modified chunk, not separate Removed+Added
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, DiffKind::Modified);
    }
}
