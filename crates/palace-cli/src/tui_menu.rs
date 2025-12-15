//! TUI Action Menu with ratatui
//!
//! Provides an interactive terminal UI for selecting tasks with:
//! - Arrow key navigation
//! - Space to select/deselect
//! - A to select all
//! - C to add custom tasks
//! - Enter to confirm selection

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    prelude::*,
    widgets::{block::*, *},
};
use std::io::{stdout, Stdout};

use super::menu::Action;

/// Present TUI action menu
pub fn present_tui_action_menu(actions: &[Action]) -> Result<Option<Vec<Action>>> {
    if actions.is_empty() {
        println!("\n📋 No actions suggested.");
        return Ok(None);
    }

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = AppState::new(actions);

    // Run main loop
    let _result = run_app(&mut terminal, &mut app);

    // Cleanup terminal
    disable_raw_mode()?;
    terminal.show_cursor()?;
    terminal.backend_mut().flush()?;

    // Return selected actions
    if app.confirmed && !app.selected_indices.is_empty() {
        let selected_actions: Vec<Action> = app.selected_indices
            .iter()
            .map(|&i| actions[i].clone())
            .collect();
        Ok(Some(selected_actions))
    } else {
        Ok(None)
    }
}

/// App state for TUI
struct AppState {
    actions: Vec<Action>,
    selected_indices: Vec<usize>,
    cursor: usize,
    confirmed: bool,
    custom_tasks: Vec<String>,
    custom_input: String,
    custom_input_active: bool,
}

impl AppState {
    fn new(actions: &[Action]) -> Self {
        Self {
            actions: actions.to_vec(),
            selected_indices: Vec::new(),
            cursor: 0,
            confirmed: false,
            custom_tasks: Vec::new(),
            custom_input: String::new(),
            custom_input_active: false,
        }
    }

    fn toggle_selection(&mut self) {
        let index = self.cursor;
        if self.selected_indices.contains(&index) {
            self.selected_indices.retain(|&i| i != index);
        } else {
            self.selected_indices.push(index);
        }
    }

    fn select_all(&mut self) {
        self.selected_indices = (0..self.actions.len()).collect();
    }

    fn move_cursor_up(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    fn move_cursor_down(&mut self) {
        if self.cursor < self.actions.len() - 1 {
            self.cursor += 1;
        }
    }

    fn add_custom_task(&mut self) {
        if !self.custom_input.is_empty() {
            self.custom_tasks.push(self.custom_input.clone());
            self.custom_input.clear();
            self.custom_input_active = false;
        }
    }

    fn handle_custom_input(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char(c) => {
                self.custom_input.push(c);
            }
            KeyCode::Backspace => {
                self.custom_input.pop();
            }
            KeyCode::Enter => {
                self.add_custom_task();
            }
            KeyCode::Esc => {
                self.custom_input.clear();
                self.custom_input_active = false;
            }
            _ => {}
        }
    }
}

/// Run the TUI application
fn run_app(terminal: &mut Terminal<impl Backend>, app: &mut AppState) -> Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        if event::poll(std::time::Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Up => {
                            if app.custom_input_active {
                                // Custom input active - don't move cursor
                            } else {
                                app.move_cursor_up();
                            }
                        }
                        KeyCode::Down => {
                            if app.custom_input_active {
                                // Custom input active - don't move cursor
                            } else {
                                app.move_cursor_down();
                            }
                        }
                        KeyCode::Char(' ') => {
                            if !app.custom_input_active {
                                app.toggle_selection();
                            }
                        }
                        KeyCode::Char('a') | KeyCode::Char('A') => {
                            if !app.custom_input_active {
                                app.select_all();
                            }
                        }
                        KeyCode::Char('c') | KeyCode::Char('C') => {
                            if !app.custom_input_active {
                                app.custom_input_active = true;
                            }
                        }
                        KeyCode::Enter => {
                            if app.custom_input_active {
                                app.add_custom_task();
                            } else {
                                app.confirmed = true;
                                return Ok(());
                            }
                        }
                        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => {
                            app.confirmed = false;
                            return Ok(());
                        }
                        _ => {
                            if app.custom_input_active {
                                app.handle_custom_input(key.code);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Draw the UI
fn ui(f: &mut Frame, app: &AppState) {
    let rects = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Actions list
            Constraint::Length(5), // Footer/Help
        ])
        .split(f.size());

    // Header
    let header_block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan));
    let header = Paragraph::new("📋 SUGGESTED ACTIONS - Use ↑↓ arrows, Space to select, A for all, C for custom, Enter to confirm")
        .block(header_block)
        .alignment(Alignment::Center);
    f.render_widget(header, rects[0]);

    // Actions list
    let items: Vec<ListItem> = app.actions.iter().enumerate().map(|(i, action)| {
        let selected = app.selected_indices.contains(&i);
        let cursor = i == app.cursor;

        let mut line = format!("  {}. {}", i + 1, action.label);
        if let Some(desc) = &action.description {
            line.push_str(&format!(" - {}", desc));
        }

        let symbol = if selected { "✓ " } else { "  " };
        let style = if cursor {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else if selected {
            Style::default().fg(Color::Green)
        } else {
            Style::default()
        };

        ListItem::new(format!("{}{}", symbol, line)).style(style)
    }).collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL))
        .highlight_style(Style::default().add_modifier(Modifier::BOLD));
    f.render_widget(list, rects[1]);

    // Custom tasks section (if any)
    if !app.custom_tasks.is_empty() {
        let custom_rect = Rect {
            x: rects[1].x,
            y: rects[1].y + rects[1].height + 1,
            width: rects[1].width,
            height: 5,
        };

        let custom_block = Block::default()
            .borders(Borders::ALL)
            .title("Custom Tasks")
            .style(Style::default().fg(Color::Magenta));

        let custom_items: Vec<ListItem> = app.custom_tasks.iter().enumerate().map(|(i, task)| {
            ListItem::new(format!("  {}. {}", i + 1, task))
        }).collect();

        let custom_list = List::new(custom_items)
            .block(custom_block);
        f.render_widget(custom_list, custom_rect);
    }

    // Custom input (if active)
    if app.custom_input_active {
        let input_rect = Rect {
            x: rects[1].x,
            y: rects[1].y + rects[1].height + 1,
            width: rects[1].width,
            height: 3,
        };

        let input_block = Block::default()
            .borders(Borders::ALL)
            .title("Add Custom Task (Enter to add, Esc to cancel)")
            .style(Style::default().fg(Color::Yellow));

        let input_paragraph = Paragraph::new(format!("> {}", app.custom_input))
            .block(input_block);
        f.render_widget(input_paragraph, input_rect);
    }

    // Footer/Help
    let help_text = vec![
        Line::from("↑↓: Navigate  Space: Select  A: Select All  C: Add Custom  Enter: Confirm  Q/Esc: Quit"),
        Line::from(format!("Selected: {}  Total: {}", app.selected_indices.len(), app.actions.len())),
    ];

    let help_block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Gray));

    let help_paragraph = Paragraph::new(help_text)
        .block(help_block)
        .alignment(Alignment::Center);
    f.render_widget(help_paragraph, rects[2]);
}