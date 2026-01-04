//! Dialogue widget - BG3/Skyrim style dialogue interface
//!
//! Displays:
//! - Permission requests (A=Allow, X=Always, B=Deny)
//! - Multi-select surveys
//! - Streaming output with dialogue overlay

use crate::ai::{DialogueOption, PermissionRequest};
use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Clear, Paragraph, Widget, Wrap},
};

/// A dialogue/survey to present to the user
#[derive(Debug, Clone)]
pub enum Dialogue {
    /// Permission request from Claude
    Permission(PermissionRequest),

    /// Multi-select survey
    Survey {
        id: String,
        question: String,
        options: Vec<DialogueOption>,
        multi_select: bool,
    },
}

/// State for the dialogue widget
pub struct DialogueState {
    /// The current dialogue
    pub dialogue: Option<Dialogue>,
    /// Currently focused option index
    pub cursor: usize,
    /// Selected options (for multi-select)
    pub selected: Vec<usize>,
    /// Streaming output buffer
    pub output_lines: Vec<OutputLine>,
    /// Progress indicator
    pub progress: Option<String>,
}

/// A line of streaming output
#[derive(Debug, Clone)]
pub struct OutputLine {
    pub text: String,
    pub style: OutputStyle,
}

#[derive(Debug, Clone, Copy)]
pub enum OutputStyle {
    Normal,
    Progress,
    Error,
    Success,
}

impl Default for DialogueState {
    fn default() -> Self {
        Self {
            dialogue: None,
            cursor: 0,
            selected: Vec::new(),
            output_lines: Vec::new(),
            progress: None,
        }
    }
}

impl DialogueState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_permission(&mut self, perm: PermissionRequest) {
        self.dialogue = Some(Dialogue::Permission(perm));
        self.cursor = 0;
        self.selected.clear();
    }

    pub fn set_survey(
        &mut self,
        id: String,
        question: String,
        options: Vec<DialogueOption>,
        multi_select: bool,
    ) {
        self.dialogue = Some(Dialogue::Survey {
            id,
            question,
            options,
            multi_select,
        });
        self.cursor = 0;
        self.selected.clear();
    }

    pub fn clear_dialogue(&mut self) {
        self.dialogue = None;
        self.cursor = 0;
        self.selected.clear();
    }

    pub fn add_output(&mut self, text: String, style: OutputStyle) {
        self.output_lines.push(OutputLine { text, style });
        // Keep last 100 lines
        if self.output_lines.len() > 100 {
            self.output_lines.remove(0);
        }
    }

    pub fn move_up(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    pub fn move_down(&mut self) {
        let max = match &self.dialogue {
            Some(Dialogue::Permission(_)) => 2, // Allow, Always, Deny
            Some(Dialogue::Survey { options, .. }) => options.len().saturating_sub(1),
            None => 0,
        };
        if self.cursor < max {
            self.cursor += 1;
        }
    }

    pub fn toggle_selection(&mut self) {
        if let Some(Dialogue::Survey { multi_select: true, .. }) = &self.dialogue {
            if self.selected.contains(&self.cursor) {
                self.selected.retain(|&x| x != self.cursor);
            } else {
                self.selected.push(self.cursor);
            }
        }
    }

    /// Get the selected option IDs for surveys
    pub fn get_selected_ids(&self) -> Vec<String> {
        if let Some(Dialogue::Survey { options, multi_select, .. }) = &self.dialogue {
            if *multi_select {
                self.selected
                    .iter()
                    .filter_map(|&idx| options.get(idx))
                    .map(|o| o.id.clone())
                    .collect()
            } else {
                options
                    .get(self.cursor)
                    .map(|o| vec![o.id.clone()])
                    .unwrap_or_default()
            }
        } else {
            Vec::new()
        }
    }
}

/// Widget that renders the dialogue overlay
pub struct DialogueWidget<'a> {
    state: &'a DialogueState,
}

impl<'a> DialogueWidget<'a> {
    pub fn new(state: &'a DialogueState) -> Self {
        Self { state }
    }
}

impl Widget for DialogueWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Split into output area (top) and dialogue area (bottom)
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),     // Output
                Constraint::Length(12), // Dialogue box
            ])
            .split(area);

        // Render output
        self.render_output(chunks[0], buf);

        // Render dialogue if present
        if self.state.dialogue.is_some() {
            self.render_dialogue(chunks[1], buf);
        } else if let Some(ref progress) = self.state.progress {
            self.render_progress(chunks[1], buf, progress);
        }
    }
}

impl DialogueWidget<'_> {
    fn render_output(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Claude Output ")
            .style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(area);
        block.render(area, buf);

        // Render output lines (scrolled to bottom)
        let visible_lines = inner.height as usize;
        let start = self.state.output_lines.len().saturating_sub(visible_lines);

        for (i, line) in self.state.output_lines.iter().skip(start).enumerate() {
            if i >= visible_lines {
                break;
            }

            let style = match line.style {
                OutputStyle::Normal => Style::default().fg(Color::White),
                OutputStyle::Progress => Style::default().fg(Color::Yellow),
                OutputStyle::Error => Style::default().fg(Color::Red),
                OutputStyle::Success => Style::default().fg(Color::Green),
            };

            // Truncate long lines
            let text = if line.text.len() > inner.width as usize {
                format!("{}...", &line.text[..inner.width as usize - 3])
            } else {
                line.text.clone()
            };

            buf.set_string(inner.x, inner.y + i as u16, &text, style);
        }
    }

    fn render_dialogue(&self, area: Rect, buf: &mut Buffer) {
        // Clear area for dialogue box
        Clear.render(area, buf);

        match &self.state.dialogue {
            Some(Dialogue::Permission(perm)) => self.render_permission(area, buf, perm),
            Some(Dialogue::Survey {
                question,
                options,
                multi_select,
                ..
            }) => self.render_survey(area, buf, question, options, *multi_select),
            None => {}
        }
    }

    fn render_permission(&self, area: Rect, buf: &mut Buffer, perm: &PermissionRequest) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Permission Required ")
            .title_alignment(Alignment::Center)
            .style(Style::default().fg(Color::Yellow));

        let inner = block.inner(area);
        block.render(area, buf);

        // Permission description
        let desc = format!(
            "{}: {}{}",
            perm.tool,
            perm.description,
            perm.target
                .as_ref()
                .map(|t| format!("\n→ {}", t))
                .unwrap_or_default()
        );

        let desc_para = Paragraph::new(desc)
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: true });

        let desc_area = Rect::new(inner.x + 1, inner.y, inner.width - 2, 3);
        desc_para.render(desc_area, buf);

        // Options
        let options = [
            ("A", "Allow", "Permit this action"),
            ("X", "Always Allow", "Auto-approve this tool"),
            ("B", "Deny", "Cancel and provide feedback"),
        ];

        for (i, (key, label, hint)) in options.iter().enumerate() {
            let y = inner.y + 4 + i as u16;
            if y >= inner.y + inner.height {
                break;
            }

            let is_focused = i == self.state.cursor;

            let key_style = Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD);

            let label_style = if is_focused {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };

            let hint_style = Style::default().fg(Color::DarkGray);

            let cursor = if is_focused { "▶ " } else { "  " };
            buf.set_string(inner.x + 1, y, cursor, label_style);
            buf.set_string(inner.x + 3, y, &format!("[{}]", key), key_style);
            buf.set_string(inner.x + 7, y, *label, label_style);
            buf.set_string(inner.x + 22, y, *hint, hint_style);
        }
    }

    fn render_survey(
        &self,
        area: Rect,
        buf: &mut Buffer,
        question: &str,
        options: &[DialogueOption],
        multi_select: bool,
    ) {
        let title = if multi_select {
            " Select Options (Space to toggle, Enter to confirm) "
        } else {
            " Select Option "
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .title_alignment(Alignment::Center)
            .style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        block.render(area, buf);

        // Question
        let question_para = Paragraph::new(question)
            .style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD))
            .wrap(Wrap { trim: true });

        let question_area = Rect::new(inner.x + 1, inner.y, inner.width - 2, 2);
        question_para.render(question_area, buf);

        // Options
        for (i, opt) in options.iter().enumerate() {
            let y = inner.y + 3 + i as u16;
            if y >= inner.y + inner.height - 1 {
                break;
            }

            let is_focused = i == self.state.cursor;
            let is_selected = self.state.selected.contains(&i);

            let checkbox = if multi_select {
                if is_selected {
                    "[x] "
                } else {
                    "[ ] "
                }
            } else {
                ""
            };

            let cursor = if is_focused { "▶ " } else { "  " };

            let label_style = if is_focused {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else if is_selected {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::White)
            };

            let line = format!("{}{}{}", cursor, checkbox, opt.label);
            buf.set_string(inner.x + 1, y, &line, label_style);

            // Description on same line if space permits
            if let Some(ref desc) = opt.description {
                let desc_x = inner.x + 1 + line.len() as u16 + 2;
                if desc_x < inner.x + inner.width - 3 {
                    let max_desc = (inner.width as usize).saturating_sub(line.len() + 4);
                    let desc_text = if desc.len() > max_desc {
                        format!("{}...", &desc[..max_desc.saturating_sub(3)])
                    } else {
                        desc.clone()
                    };
                    buf.set_string(desc_x, y, &desc_text, Style::default().fg(Color::DarkGray));
                }
            }
        }
    }

    fn render_progress(&self, area: Rect, buf: &mut Buffer, progress: &str) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Working... ")
            .style(Style::default().fg(Color::Yellow));

        let inner = block.inner(area);
        block.render(area, buf);

        let para = Paragraph::new(progress)
            .style(Style::default().fg(Color::Yellow))
            .alignment(Alignment::Center);

        para.render(inner, buf);
    }
}
