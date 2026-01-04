use crate::db::Database;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::Widget,
};

/// Wrap text to fit within max_width, breaking on word boundaries
fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        if current_line.is_empty() {
            if word.len() > max_width {
                // Word too long, force break
                let mut remaining = word;
                while remaining.len() > max_width {
                    lines.push(remaining[..max_width].to_string());
                    remaining = &remaining[max_width..];
                }
                current_line = remaining.to_string();
            } else {
                current_line = word.to_string();
            }
        } else if current_line.len() + 1 + word.len() <= max_width {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            lines.push(current_line);
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

pub struct TaskListWidget<'a> {
    db: &'a Database,
    expanded: bool,
}

impl<'a> TaskListWidget<'a> {
    pub fn new(db: &'a Database, expanded: bool) -> Self {
        Self { db, expanded }
    }
}

impl Widget for TaskListWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let tasks = self.db.tasks();

        if tasks.is_empty() {
            let msg = if self.db.showing_history() {
                "No historical tasks. Press [H] to view fresh tasks."
            } else if self.db.history_count() > 0 {
                "No fresh tasks. Press [H] to view history."
            } else {
                "No tasks. Waiting for Claude to suggest tasks..."
            };
            let x = area.x + 2;
            let y = area.y + area.height / 2;
            buf.set_string(x, y, msg, Style::default().fg(Color::DarkGray));
            return;
        }

        let cursor = self.db.cursor();

        // Estimate lines per task: label + desc, focused task gets +1 for details
        let visible_lines = area.height as usize;

        // Calculate scroll offset - we need to account for the focused task having extra lines
        // Count lines needed up to and including cursor
        let lines_before_cursor: usize = (0..cursor).map(|_| 2).sum();
        let lines_for_cursor = 3; // label + desc + details for focused
        let total_lines_to_cursor = lines_before_cursor + lines_for_cursor;

        let scroll_offset = if total_lines_to_cursor > visible_lines {
            // Find which task to start from
            let mut lines = 0;
            let mut start = 0;
            for i in 0..=cursor {
                let task_lines = if i == cursor { 3 } else { 2 };
                if lines + task_lines > total_lines_to_cursor - visible_lines {
                    start = i;
                    break;
                }
                lines += task_lines;
            }
            start
        } else {
            0
        };

        let mut y = area.y;

        // Render visible tasks
        for (i, task) in tasks.iter().enumerate().skip(scroll_offset) {
            if y >= area.y + area.height {
                break;
            }

            let is_focused = i == cursor;

            // Line 1: Checkbox + Label
            let checkbox = if task.selected { "[x]" } else { "[ ]" };

            let label_style = if is_focused {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else if task.selected {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::White)
            };

            // Add time estimate badge if available
            let time_badge = task
                .time_estimate
                .as_ref()
                .map(|t| format!(" [{}]", t))
                .unwrap_or_default();

            let line = format!(" {} {}{}", checkbox, task.label, time_badge);
            let line = if line.len() > area.width as usize {
                format!("{}...", &line[..area.width as usize - 3])
            } else {
                line
            };

            buf.set_string(area.x, y, &line, label_style);
            y += 1;

            // Line 2+: Description (dimmed, indented)
            // When expanded and focused, show full wrapped description
            if y < area.y + area.height {
                let desc_style = if is_focused {
                    Style::default().fg(Color::Gray)
                } else {
                    Style::default().fg(Color::DarkGray)
                };

                let indent = "       ";
                let max_width = (area.width as usize).saturating_sub(indent.len() + 1);

                if is_focused && self.expanded && !task.description.is_empty() {
                    // Wrap text for expanded view
                    let lines = wrap_text(&task.description, max_width);
                    for line in lines {
                        if y >= area.y + area.height {
                            break;
                        }
                        let desc_line = format!("{}{}", indent, line);
                        buf.set_string(area.x, y, &desc_line, desc_style);
                        y += 1;
                    }
                } else {
                    // Single line truncated
                    let desc = if task.description.len() > max_width {
                        format!("{}...", &task.description[..(max_width.saturating_sub(3)).min(task.description.len())])
                    } else {
                        task.description.clone()
                    };
                    let desc_line = format!("{}{}", indent, desc);
                    buf.set_string(area.x, y, &desc_line, desc_style);
                    y += 1;
                }
            }

            // Line 3: Details (only for focused task) - inline summary
            if is_focused && y < area.y + area.height {
                let mut details = Vec::new();

                if let Some(ref complexity) = task.complexity {
                    let dots = match complexity.as_str() {
                        "trivial" => "●",
                        "simple" => "●●",
                        "moderate" => "●●●",
                        "complex" => "●●●●",
                        _ => "?",
                    };
                    details.push(format!("{} {}", dots, complexity));
                }

                if !task.affected_files.is_empty() {
                    let files_str = if task.affected_files.len() <= 3 {
                        task.affected_files.join(", ")
                    } else {
                        format!("{}, ... +{} more",
                            task.affected_files[..2].join(", "),
                            task.affected_files.len() - 2)
                    };
                    details.push(format!("📁 {}", files_str));
                }

                if !details.is_empty() {
                    let detail_line = format!("       {}", details.join("  │  "));
                    buf.set_string(area.x, y, &detail_line, Style::default().fg(Color::Cyan));
                    y += 1;
                }
            }
        }

        // Show scroll indicator if there are more tasks
        if scroll_offset > 0 {
            buf.set_string(
                area.x + area.width - 5,
                area.y,
                "↑more",
                Style::default().fg(Color::DarkGray),
            );
        }

        // Check if there are more tasks below
        let tasks_after_visible: usize = tasks.iter().enumerate().skip(scroll_offset).count();
        let mut lines_used = 0;
        for (i, _) in tasks.iter().enumerate().skip(scroll_offset) {
            lines_used += if i == cursor { 3 } else { 2 };
            if lines_used > visible_lines {
                break;
            }
        }
        if lines_used > visible_lines || scroll_offset + tasks_after_visible < tasks.len() {
            buf.set_string(
                area.x + area.width - 5,
                area.y + area.height - 1,
                "↓more",
                Style::default().fg(Color::DarkGray),
            );
        }
    }
}
