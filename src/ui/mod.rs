mod dialogue;
mod task_list;

use crate::app::App;
pub use dialogue::{Dialogue, DialogueState, DialogueWidget, OutputLine, OutputStyle};
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
    pub fn new(_config: &Config) -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();

        // Never capture mouse - user needs to select/copy text from terminal
        execute!(stdout, EnterAlternateScreen)?;

        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        Ok(Self {
            terminal,
            mouse_enabled: false,
        })
    }

    pub fn draw(&mut self, app: &mut App) -> Result<()> {
        self.terminal.draw(|frame| {
            render(frame, app);
        })?;
        Ok(())
    }

    /// Enter execution mode: leave TUI, show simple CLI output
    pub fn enter_execution_mode(&mut self) -> Result<()> {
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

        // Print header for execution mode
        println!("\n\x1b[36m━━━ Palace: Claude Execution Mode ━━━\x1b[0m\n");
        Ok(())
    }

    /// Leave execution mode: return to TUI
    pub fn leave_execution_mode(&mut self) -> Result<()> {
        enable_raw_mode()?;
        if self.mouse_enabled {
            execute!(
                self.terminal.backend_mut(),
                EnterAlternateScreen,
                EnableMouseCapture
            )?;
        } else {
            execute!(self.terminal.backend_mut(), EnterAlternateScreen)?;
        }
        self.terminal.hide_cursor()?;
        self.terminal.clear()?;
        Ok(())
    }

    /// Print a status line during execution (simple CLI output)
    pub fn print_status(&self, msg: &str) {
        println!("{}", msg);
    }

    /// Print a progress indicator
    pub fn print_progress(&self, tool: &str, detail: &str) {
        println!("\x1b[33m[{}]\x1b[0m {}", tool, detail);
    }

    /// Print an error
    pub fn print_error(&self, msg: &str) {
        eprintln!("\x1b[31mError:\x1b[0m {}", msg);
    }

    /// Print success
    pub fn print_success(&self, msg: &str) {
        println!("\x1b[32m✓\x1b[0m {}", msg);
    }

    /// Print permission prompt (returns to TUI mode for this)
    pub fn print_permission_prompt(&self, tool: &str, description: &str, target: Option<&str>) {
        println!("\n\x1b[33m━━━ Permission Required ━━━\x1b[0m");
        println!("\x1b[1m{}\x1b[0m: {}", tool, description);
        if let Some(t) = target {
            println!("  → {}", t);
        }
        println!("\n  \x1b[36m[A]\x1b[0m Allow   \x1b[36m[X]\x1b[0m Always   \x1b[36m[B]\x1b[0m Deny\n");
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

fn render(frame: &mut Frame, app: &mut App) {
    use crate::app::AppState;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(10),    // Task list (details shown inline)
            Constraint::Length(3),  // Footer/controls
        ])
        .split(frame.area());

    // Calculate visible tasks based on actual task list area
    // Most tasks are 2 lines, focused task is 3 lines
    app.visible_tasks = (chunks[1].height as usize) / 2;

    // Header - show state
    let state_indicator = match app.state {
        AppState::Generating => " [Asking Claude...]",
        AppState::Executing => " [Claude working...]",
        _ => "",
    };

    let selected_count = app.db.tasks().iter().filter(|t| t.selected).count();
    let selected_info = if selected_count > 0 {
        format!(" │ {} selected", selected_count)
    } else {
        String::new()
    };

    // Position indicator
    let position_info = if app.db.task_count() > 0 {
        format!(" │ {}/{}", app.db.cursor() + 1, app.db.task_count())
    } else {
        String::new()
    };

    // History mode indicator
    let history_indicator = if app.db.showing_history() {
        let count = app.db.history_count();
        format!(" │ HISTORY ({})", count)
    } else {
        let fresh = app.db.fresh_count();
        let hist = app.db.history_count();
        if hist > 0 {
            format!(" │ Fresh {} (+{} hist)", fresh, hist)
        } else {
            String::new()
        }
    };

    let header = Paragraph::new(format!(
        " Palace │ {}{}{}{}{}",
        app.project_path.display(),
        position_info,
        selected_info,
        history_indicator,
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
            // Show dialogue widget with streaming output
            let dialogue_widget = DialogueWidget::new(&app.dialogue_state);
            frame.render_widget(dialogue_widget, chunks[1]);
        }
        _ => {
            let task_widget = TaskListWidget::new(&app.db, app.expanded);
            frame.render_widget(task_widget, chunks[1]);
        }
    }

    // Footer - show error if any, otherwise context-appropriate controls
    let footer_text = if let Some(ref err) = app.error_message {
        format!(" ⚠ {}", err)
    } else if app.state == AppState::Executing {
        if app.dialogue_state.dialogue.is_some() {
            " [A] Accept  [X] Always  [B] Deny/Cancel  [↑↓] Navigate  [Space] Toggle  [Enter] Confirm".to_string()
        } else {
            " [B] Cancel  │  Claude is working...".to_string()
        }
    } else if app.db.showing_history() {
        " [H] Fresh Tasks  [Space] Toggle  [Enter] Execute  [Del] Delete  [q] Quit".to_string()
    } else {
        " [Space] Toggle  [Shift+↑↓] Range  [A] All  [E] Expand  [H] History  [Enter] Execute  [q] Quit".to_string()
    };

    let footer_style = if app.error_message.is_some() {
        Style::default().fg(Color::Red)
    } else if app.state == AppState::Executing {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let footer = Paragraph::new(footer_text)
        .style(footer_style)
        .block(Block::default().borders(Borders::TOP));
    frame.render_widget(footer, chunks[2]);
}
