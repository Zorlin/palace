use crate::ai::AnthropicClient;
use crate::config::{Backend, Config};
use crate::db::Database;
use crate::input::{Event, InputHandler};
use crate::ui::Ui;
use anyhow::Result;
use std::path::PathBuf;

/// Application state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    /// Claude is generating task suggestions
    Generating,
    /// Browsing task list
    TaskList,
    /// Claude is executing selected tasks
    Executing,
    /// Viewing history
    History,
    /// Switching projects
    ProjectSelect,
}

/// Main application
pub struct App {
    pub config: Config,
    pub backend: Backend,
    pub project_path: PathBuf,
    pub state: AppState,
    pub db: Database,
    pub should_quit: bool,
    pub ai_client: Option<AnthropicClient>,
    pub error_message: Option<String>,
    /// Visible height of task list (for page navigation)
    pub visible_tasks: usize,
    /// Whether to show expanded description for focused task
    pub expanded: bool,
}

impl App {
    pub fn new(config: Config, backend: Backend, project_path: PathBuf) -> Result<Self> {
        let project_path = project_path.canonicalize().unwrap_or(project_path);
        let db = Database::open(&project_path)?;

        // Initialize AI client
        let ai_client = match backend {
            Backend::Anthropic => {
                match AnthropicClient::new(&config.api.base_url, &config.api.model) {
                    Ok(client) => Some(client),
                    Err(e) => {
                        tracing::warn!("Failed to init AI client: {e}");
                        None
                    }
                }
            }
            Backend::OpenCode => {
                // TODO: OpenCode client
                None
            }
        };

        Ok(Self {
            config,
            backend,
            project_path,
            state: AppState::TaskList,
            db,
            should_quit: false,
            ai_client,
            error_message: None,
            visible_tasks: 10, // Default, updated by UI
            expanded: false,
        })
    }

    /// Run the TUI application
    pub async fn run(&mut self) -> Result<()> {
        // Initialize terminal
        let mut ui = Ui::new(&self.config)?;

        // Initialize input handler (keyboard, mouse, gamepad)
        let mut input = InputHandler::new(&self.config)?;

        // Ask Claude for task suggestions on startup
        self.state = AppState::Generating;
        ui.draw(self)?;
        self.generate_tasks().await?;
        self.state = AppState::TaskList;

        // Main event loop
        while !self.should_quit {
            // Render UI
            ui.draw(self)?;

            // Handle input
            if let Some(event) = input.next_event().await? {
                self.handle_event(event).await?;
            }
        }

        // Cleanup
        ui.restore()?;

        Ok(())
    }

    /// Handle input events
    async fn handle_event(&mut self, event: Event) -> Result<()> {
        match event {
            Event::Quit => {
                self.should_quit = true;
            }
            Event::NavigateUp => {
                self.db.select_previous();
            }
            Event::NavigateDown => {
                self.db.select_next();
            }
            Event::JumpUp => {
                self.db.jump_up(5);
            }
            Event::JumpDown => {
                self.db.jump_down(5);
            }
            Event::PageUp => {
                self.db.page_up(self.visible_tasks);
            }
            Event::PageDown => {
                self.db.page_down(self.visible_tasks);
            }
            Event::Home => {
                self.db.jump_to_start();
            }
            Event::End => {
                self.db.jump_to_end();
            }
            Event::SelectUp => {
                self.db.select_and_move_up();
            }
            Event::SelectDown => {
                self.db.select_and_move_down();
            }
            Event::ToggleSelect => {
                self.db.toggle_current_selection();
            }
            Event::ToggleSelectAll => {
                self.db.toggle_select_all();
            }
            Event::Execute => {
                if self.db.has_selection() {
                    self.state = AppState::Executing;
                    // TODO: Send selected tasks to Claude for execution
                    self.execute_tasks().await?;
                    // After execution, ask Claude for new suggestions
                    self.state = AppState::Generating;
                    self.generate_tasks().await?;
                    self.state = AppState::TaskList;
                }
            }
            Event::Delete => {
                self.db.delete_selected()?;
            }
            Event::ToggleExpand => {
                self.expanded = !self.expanded;
            }
            Event::MouseClick { row, .. } => {
                // Click on a task row to focus it (offset for header)
                if row >= 3 {
                    let task_idx = (row - 3) as usize / 2; // 2 lines per task
                    self.db.select_index(task_idx);
                }
            }
            Event::MouseShiftClick { row, .. } => {
                // Shift+click for range selection
                if row >= 3 {
                    let task_idx = (row - 3) as usize / 2;
                    self.db.select_range_to(task_idx);
                }
            }
        }
        Ok(())
    }

    /// Ask Claude/OpenCode to generate task suggestions
    pub async fn generate_tasks(&mut self) -> Result<()> {
        tracing::info!(project = ?self.project_path, backend = ?self.backend, "Generating tasks");

        let Some(ref client) = self.ai_client else {
            self.error_message = Some("No AI client configured. Set ANTHROPIC_API_KEY.".to_string());
            return Ok(());
        };

        match client.generate_tasks(&self.project_path).await {
            Ok(suggestions) => {
                self.error_message = None;
                let mut added = 0;
                for suggestion in suggestions {
                    // Skip duplicates
                    if !self.db.has_task_with_label(&suggestion.label) {
                        self.db.add_task(
                            suggestion.label,
                            suggestion.description,
                            suggestion.time_estimate,
                            suggestion.complexity,
                            suggestion.affected_files,
                        )?;
                        added += 1;
                    }
                }
                tracing::info!(added, "Added new task suggestions");
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to generate tasks: {e}"));
                tracing::error!(?e, "Failed to generate tasks");
            }
        }

        Ok(())
    }

    /// Send selected tasks to Claude for execution
    pub async fn execute_tasks(&mut self) -> Result<()> {
        let selected: Vec<_> = self.db.selected_tasks().iter().map(|t| {
            crate::ai::TaskSuggestion {
                label: t.label.clone(),
                description: t.description.clone(),
                time_estimate: t.time_estimate.clone(),
                complexity: t.complexity.clone(),
                affected_files: t.affected_files.clone(),
            }
        }).collect();

        tracing::info!(count = selected.len(), "Executing tasks");

        let Some(ref client) = self.ai_client else {
            self.error_message = Some("No AI client configured.".to_string());
            return Ok(());
        };

        // Execute each task
        for task in &selected {
            tracing::info!(label = %task.label, "Executing task");
            match client.execute_task(task, &self.project_path).await {
                Ok(output) => {
                    tracing::info!(label = %task.label, "Task completed");
                    // TODO: Display output in UI, mark as completed
                    tracing::debug!(output = %output, "Task output");
                }
                Err(e) => {
                    self.error_message = Some(format!("Task failed: {e}"));
                    tracing::error!(?e, label = %task.label, "Task failed");
                }
            }
        }

        // Clear selection after execution
        self.db.deselect_all();

        Ok(())
    }

    /// Switch to a different project
    pub fn switch_project(&mut self, path: PathBuf) -> Result<()> {
        let path = path.canonicalize().unwrap_or(path);
        tracing::info!(?path, "Switching project");
        self.project_path = path.clone();
        self.db = Database::open(&path)?;
        Ok(())
    }
}
