mod task_list;

use crate::app::App;
use crate::config::Config;
use anyhow::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};
use std::io::{self, Stdout};

pub use task_list::TaskListWidget;

pub struct Ui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    mouse_enabled: bool,
}

impl Ui {
    pub fn new(config: &Config) -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();

        if config.ui.mouse_enabled {
            execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        } else {
            execute!(stdout, EnterAlternateScreen)?;
        }

        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        Ok(Self {
            terminal,
            mouse_enabled: config.ui.mouse_enabled,
        })
    }

    pub fn draw(&mut self, app: &App) -> Result<()> {
        self.terminal.draw(|frame| {
            render(frame, app);
        })?;
        Ok(())
    }

    pub fn restore(&mut self) -> Result<()> {
        disable_raw_mode()?;
        if self.mouse_enabled {
            execute!(
                self.terminal.backend_mut(),
                LeaveAlternateScreen,
                DisableMouseCapture
            )?;
        } else {
            execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
        }
        self.terminal.show_cursor()?;
        Ok(())
    }
}

fn render(frame: &mut Frame, app: &App) {
    use crate::app::AppState;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Task list
            Constraint::Length(3), // Footer/controls
        ])
        .split(frame.area());

    // Header - show state
    let state_indicator = match app.state {
        AppState::Generating => " [Asking Claude...]",
        AppState::Executing => " [Claude working...]",
        _ => "",
    };

    let header = Paragraph::new(format!(
        " Palace │ {} │ {} tasks{}",
        app.project_path.display(),
        app.db.task_count(),
        state_indicator
    ))
    .style(Style::default().fg(Color::Cyan))
    .block(Block::default().borders(Borders::BOTTOM));
    frame.render_widget(header, chunks[0]);

    // Main content area
    match app.state {
        AppState::Generating => {
            let msg = Paragraph::new(" Asking Claude what to work on...")
                .style(Style::default().fg(Color::Yellow));
            frame.render_widget(msg, chunks[1]);
        }
        AppState::Executing => {
            let msg = Paragraph::new(" Claude is working on selected tasks...")
                .style(Style::default().fg(Color::Green));
            frame.render_widget(msg, chunks[1]);
        }
        _ => {
            let task_widget = TaskListWidget::new(&app.db);
            frame.render_widget(task_widget, chunks[1]);
        }
    }

    // Footer - show error if any, otherwise controls
    let footer_text = if let Some(ref err) = app.error_message {
        format!(" ⚠ {}", err)
    } else {
        " [Space] Select  [Enter] Execute  [Ctrl+A] All  [Del] Delete  [q] Quit".to_string()
    };

    let footer_style = if app.error_message.is_some() {
        Style::default().fg(Color::Red)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let footer = Paragraph::new(footer_text)
        .style(footer_style)
        .block(Block::default().borders(Borders::TOP));
    frame.render_widget(footer, chunks[2]);
}
