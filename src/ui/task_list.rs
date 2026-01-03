use crate::db::Database;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::Widget,
};

pub struct TaskListWidget<'a> {
    db: &'a Database,
}

impl<'a> TaskListWidget<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }
}

impl Widget for TaskListWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let tasks = self.db.tasks();

        if tasks.is_empty() {
            let msg = "No tasks. Waiting for Claude to suggest tasks...";
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

            // Line 2: Description (dimmed, indented)
            if y < area.y + area.height {
                let desc_style = if is_focused {
                    Style::default().fg(Color::Gray)
                } else {
                    Style::default().fg(Color::DarkGray)
                };

                let desc = if task.description.len() > (area.width as usize - 8) {
                    format!("{}...", &task.description[..(area.width as usize - 11).min(task.description.len())])
                } else {
                    task.description.clone()
                };

                let desc_line = format!("       {}", desc);
                buf.set_string(area.x, y, &desc_line, desc_style);
                y += 1;
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
