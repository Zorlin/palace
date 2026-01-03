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
            let msg = "No tasks. Run 'palace analyze' to generate tasks.";
            let x = area.x + 2;
            let y = area.y + area.height / 2;
            buf.set_string(x, y, msg, Style::default().fg(Color::DarkGray));
            return;
        }

        let cursor = self.db.cursor();

        for (i, task) in tasks.iter().enumerate() {
            if i as u16 >= area.height {
                break;
            }

            let y = area.y + i as u16;

            // Selection indicator
            let checkbox = if task.selected { "[x]" } else { "[ ]" };

            // Style based on cursor and selection
            let style = if i == cursor {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else if task.selected {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::White)
            };

            // Render line
            let line = format!(" {} {}", checkbox, task.label);
            let line = if line.len() > area.width as usize {
                format!("{}...", &line[..area.width as usize - 3])
            } else {
                line
            };

            buf.set_string(area.x, y, &line, style);
        }
    }
}
