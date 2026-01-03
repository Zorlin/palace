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

        // Each task takes 2 lines: label + description
        let mut y = area.y;

        for (i, task) in tasks.iter().enumerate() {
            if y >= area.y + area.height - 1 {
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
        }
    }
}

/// Widget showing details for the focused task
pub struct TaskDetailWidget<'a> {
    db: &'a Database,
}

impl<'a> TaskDetailWidget<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }
}

impl Widget for TaskDetailWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let tasks = self.db.tasks();
        let cursor = self.db.cursor();

        let Some(task) = tasks.get(cursor) else {
            return;
        };

        let mut y = area.y;

        // Time estimate
        if let Some(ref time) = task.time_estimate {
            let line = format!(" ⏱  {}", time);
            buf.set_string(area.x, y, &line, Style::default().fg(Color::Cyan));
            y += 1;
        }

        // Complexity
        if let Some(ref complexity) = task.complexity {
            let (icon, color) = match complexity.as_str() {
                "trivial" => ("●", Color::Green),
                "simple" => ("●●", Color::Green),
                "moderate" => ("●●●", Color::Yellow),
                "complex" => ("●●●●", Color::Red),
                _ => ("?", Color::Gray),
            };
            let line = format!(" {}  {}", icon, complexity);
            buf.set_string(area.x, y, &line, Style::default().fg(color));
            y += 1;
        }

        // Affected files
        if !task.affected_files.is_empty() {
            buf.set_string(area.x, y, " 📁 Files:", Style::default().fg(Color::Magenta));
            y += 1;

            for file in &task.affected_files {
                if y >= area.y + area.height {
                    break;
                }
                let line = format!("    {}", file);
                buf.set_string(area.x, y, &line, Style::default().fg(Color::DarkGray));
                y += 1;
            }
        }
    }
}
