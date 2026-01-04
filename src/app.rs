use crate::ai::{AnthropicClient, ExecutionEvent, Executor, PermissionResponse, UserResponse};
use crate::config::{Backend, Config};
use crate::db::Database;
use crate::input::{Event, InputHandler};
use crate::ui::{Dialogue, DialogueState, OutputStyle, Ui};
use anyhow::Result;
use std::path::PathBuf;
use tokio::sync::mpsc;

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
    /// Dialogue state for permission prompts and surveys
    pub dialogue_state: DialogueState,
    /// Channel to receive execution events from Executor
    event_rx: Option<mpsc::Receiver<ExecutionEvent>>,
    /// Channel to send user responses to Executor
    response_tx: Option<mpsc::Sender<UserResponse>>,
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
            dialogue_state: DialogueState::new(),
            event_rx: None,
            response_tx: None,
        })
    }

    /// Run the TUI application
    pub async fn run(&mut self) -> Result<()> {
        // Initialize terminal
        let mut ui = Ui::new(&self.config)?;

        // Initialize input handler (keyboard, mouse, gamepad)
        let mut input = InputHandler::new(&self.config)?;

        // Track if we're in execution CLI mode (vs TUI mode)
        let mut in_execution_cli = false;
        // Track if executor has been spawned (so we only spawn once per execution)
        let mut executor_spawned = false;

        // Ask Claude for task suggestions on startup
        self.state = AppState::Generating;
        ui.draw(self)?;
        self.generate_tasks().await?;
        self.state = AppState::TaskList;

        // Main event loop
        while !self.should_quit {
            match self.state {
                AppState::Executing => {
                    // Enter CLI mode if not already
                    if !in_execution_cli {
                        ui.enter_execution_mode()?;
                        in_execution_cli = true;
                    }

                    // Spawn executor after entering CLI mode (so output doesn't mix with TUI)
                    if !executor_spawned {
                        self.execute_tasks().await?;
                        executor_spawned = true;
                    }

                    // Poll execution events and simple CLI input
                    tokio::select! {
                        // Check for execution events
                        event = async {
                            if let Some(ref mut rx) = self.event_rx {
                                rx.recv().await
                            } else {
                                None
                            }
                        } => {
                            if let Some(exec_event) = event {
                                self.handle_execution_event_cli(&mut ui, exec_event).await?;
                            }
                        }
                        // Check for user input (simple CLI mode)
                        input_event = input.next_dialogue_event() => {
                            if let Some(event) = input_event? {
                                self.handle_dialogue_event(event).await?;
                            }
                        }
                    }

                    // If we've left Executing state, return to TUI
                    if self.state != AppState::Executing && in_execution_cli {
                        ui.leave_execution_mode()?;
                        in_execution_cli = false;
                        executor_spawned = false; // Reset for next execution
                    }
                }
                AppState::Generating => {
                    // Enter CLI mode so we can see streaming output
                    if !in_execution_cli {
                        ui.enter_execution_mode()?;
                        in_execution_cli = true;
                    }

                    // Generate tasks with streaming output visible
                    self.generate_tasks().await?;
                    self.state = AppState::TaskList;

                    // Return to TUI mode
                    ui.leave_execution_mode()?;
                    in_execution_cli = false;
                    executor_spawned = false;
                }
                _ => {
                    // Make sure we're in TUI mode
                    if in_execution_cli {
                        ui.leave_execution_mode()?;
                        in_execution_cli = false;
                        executor_spawned = false; // Reset for next execution
                    }

                    // Render UI
                    ui.draw(self)?;

                    // Normal input handling
                    if let Some(event) = input.next_event().await? {
                        self.handle_event(event).await?;
                    }
                }
            }
        }

        // Cleanup - make sure we restore properly
        if in_execution_cli {
            // Already in normal terminal mode, just need to clean up
        } else {
            ui.restore()?;
        }

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
                    // Just set state - the main loop will spawn the executor
                    // after entering CLI mode (to avoid output mixing with TUI)
                    self.state = AppState::Executing;
                    // Don't call execute_tasks here - defer to main loop
                }
            }
            Event::Delete => {
                self.db.delete_selected()?;
            }
            Event::ToggleExpand => {
                self.expanded = !self.expanded;
            }
            Event::ToggleHistory => {
                self.db.toggle_history();
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
            // Dialogue events are handled in handle_dialogue_event
            Event::DialogueUp
            | Event::DialogueDown
            | Event::DialogueAccept
            | Event::DialogueAlwaysAllow
            | Event::DialogueDeny
            | Event::DialogueToggle
            | Event::DialogueConfirm => {
                // These are only handled when in Executing state
            }
            Event::Intervention => {
                // RT+LT combo or Ctrl+I: interrupt and suggest alternatives
                self.trigger_intervention().await?;
            }
        }
        Ok(())
    }

    /// Handle execution events from the streaming executor (TUI mode - kept for compatibility)
    #[allow(dead_code)]
    async fn handle_execution_event(&mut self, event: ExecutionEvent) -> Result<()> {
        match event {
            ExecutionEvent::Text(text) => {
                self.dialogue_state.add_output(text, OutputStyle::Normal);
            }
            ExecutionEvent::Progress { tool, detail } => {
                let msg = format!("[{}] {}", tool, detail);
                self.dialogue_state.add_output(msg, OutputStyle::Progress);
            }
            ExecutionEvent::Permission(perm) => {
                self.dialogue_state.set_permission(perm);
            }
            ExecutionEvent::Question {
                id,
                question,
                options,
                multi_select,
            } => {
                self.dialogue_state.set_survey(id, question, options, multi_select);
            }
            ExecutionEvent::Done => {
                self.dialogue_state.add_output("Execution complete".to_string(), OutputStyle::Success);
                self.dialogue_state.clear_dialogue();
                // Clear selection and go back to task list
                self.db.deselect_all();
                self.state = AppState::Generating;
            }
            ExecutionEvent::Error(err) => {
                self.dialogue_state.add_output(format!("Error: {}", err), OutputStyle::Error);
                self.error_message = Some(err);
            }
        }
        Ok(())
    }

    /// Handle execution events in CLI mode (simple terminal output)
    async fn handle_execution_event_cli(&mut self, ui: &mut crate::ui::Ui, event: ExecutionEvent) -> Result<()> {
        match event {
            ExecutionEvent::Text(text) => {
                ui.print_status(&text);
            }
            ExecutionEvent::Progress { tool, detail } => {
                ui.print_progress(&tool, &detail);
            }
            ExecutionEvent::Permission(ref perm) => {
                ui.print_permission_prompt(
                    &perm.tool,
                    &perm.description,
                    perm.target.as_deref(),
                );
                // Store for handling
                self.dialogue_state.set_permission(perm.clone());
            }
            ExecutionEvent::Question {
                id,
                question,
                options,
                multi_select,
            } => {
                // For complex questions, we might want to switch back to TUI
                // For now, print them CLI-style
                println!("\n\x1b[36m━━━ Question ━━━\x1b[0m");
                println!("{}", question);
                for (i, opt) in options.iter().enumerate() {
                    let selected = if multi_select { "[ ]" } else { "" };
                    println!("  {} {}: {}", selected, i + 1, opt.label);
                    if let Some(ref desc) = opt.description {
                        println!("       {}", desc);
                    }
                }
                println!();
                self.dialogue_state.set_survey(id, question, options, multi_select);
            }
            ExecutionEvent::Done => {
                ui.print_success("Execution complete!");
                println!();
                self.dialogue_state.clear_dialogue();
                // Clear selection and go back to task list
                self.db.deselect_all();
                self.state = AppState::Generating;
            }
            ExecutionEvent::Error(err) => {
                ui.print_error(&err);
                self.error_message = Some(err);
            }
        }
        Ok(())
    }

    /// Handle dialogue input events (permission prompts, surveys)
    async fn handle_dialogue_event(&mut self, event: Event) -> Result<()> {
        match event {
            Event::Quit => {
                // Ctrl+C - always quit immediately
                if let Some(ref tx) = self.response_tx {
                    tx.send(UserResponse::Cancel).await.ok();
                }
                self.should_quit = true;
            }
            Event::DialogueUp => {
                self.dialogue_state.move_up();
            }
            Event::DialogueDown => {
                self.dialogue_state.move_down();
            }
            Event::DialogueToggle => {
                self.dialogue_state.toggle_selection();
            }
            Event::DialogueAccept => {
                // A button - Accept/Allow
                if let Some(Dialogue::Permission(ref perm)) = self.dialogue_state.dialogue {
                    let id = perm.id.clone();
                    if let Some(ref tx) = self.response_tx {
                        tx.send(UserResponse::Permission(id, PermissionResponse::Allow))
                            .await
                            .ok();
                    }
                    self.dialogue_state.clear_dialogue();
                } else if self.dialogue_state.dialogue.is_some() {
                    // For surveys, Accept = confirm current selection
                    self.confirm_dialogue_selection().await?;
                }
            }
            Event::DialogueAlwaysAllow => {
                // X button - Always Allow
                if let Some(Dialogue::Permission(ref perm)) = self.dialogue_state.dialogue {
                    let id = perm.id.clone();
                    if let Some(ref tx) = self.response_tx {
                        tx.send(UserResponse::Permission(id, PermissionResponse::AlwaysAllow))
                            .await
                            .ok();
                    }
                    self.dialogue_state.clear_dialogue();
                }
            }
            Event::DialogueDeny => {
                // B button - Deny/Cancel
                if let Some(Dialogue::Permission(ref perm)) = self.dialogue_state.dialogue {
                    let id = perm.id.clone();
                    if let Some(ref tx) = self.response_tx {
                        tx.send(UserResponse::Permission(
                            id,
                            PermissionResponse::Deny {
                                reason: "User denied".to_string(),
                            },
                        ))
                        .await
                        .ok();
                    }
                    self.dialogue_state.clear_dialogue();
                } else if let Some(ref tx) = self.response_tx {
                    // Cancel entire execution
                    tx.send(UserResponse::Cancel).await.ok();
                    self.dialogue_state.clear_dialogue();
                    self.state = AppState::TaskList;
                }
            }
            Event::DialogueConfirm => {
                // Enter - Confirm selection
                self.confirm_dialogue_selection().await?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Confirm dialogue selection (for surveys)
    async fn confirm_dialogue_selection(&mut self) -> Result<()> {
        if let Some(Dialogue::Survey { ref id, .. }) = self.dialogue_state.dialogue {
            let id = id.clone();
            let selected = self.dialogue_state.get_selected_ids();
            if let Some(ref tx) = self.response_tx {
                tx.send(UserResponse::Dialogue(id, selected)).await.ok();
            }
            self.dialogue_state.clear_dialogue();
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

    /// Send selected tasks to Claude Code for execution
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

        if selected.is_empty() {
            return Ok(());
        }

        tracing::info!(count = selected.len(), "Starting streaming execution");

        // Create channels for communication with executor
        let (event_tx, event_rx) = mpsc::channel::<ExecutionEvent>(100);
        let (response_tx, response_rx) = mpsc::channel::<UserResponse>(10);

        // Store channels in app
        self.event_rx = Some(event_rx);
        self.response_tx = Some(response_tx);

        // Clear dialogue state
        self.dialogue_state = DialogueState::new();
        self.dialogue_state.progress = Some("Starting Claude Code...".to_string());

        // Spawn the executor in a background task
        let project_path = self.project_path.clone();
        let mut executor = Executor::new(&project_path, event_tx, response_rx);

        tokio::spawn(async move {
            if let Err(e) = executor.execute(selected).await {
                tracing::error!(?e, "Executor failed");
            }
        });

        // State is already set to Executing by handle_event
        // The main loop will now poll event_rx for updates
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

    /// Trigger intervention: interrupt and suggest objections/alternatives
    pub async fn trigger_intervention(&mut self) -> Result<()> {
        tracing::info!("Intervention triggered!");

        // Cancel current execution if running
        if self.state == AppState::Executing {
            if let Some(ref tx) = self.response_tx {
                tx.send(UserResponse::Cancel).await.ok();
            }
        }

        // Get context: selected tasks or current focus
        let context = if self.db.has_selection() {
            let selected: Vec<_> = self.db.selected_tasks().iter()
                .map(|t| format!("- {}: {}", t.label, t.description))
                .collect();
            format!("Currently selected tasks:\n{}", selected.join("\n"))
        } else if let Some(task) = self.db.tasks().get(self.db.cursor()) {
            format!("Current task: {} - {}", task.label, task.description)
        } else {
            "No tasks selected or focused".to_string()
        };

        // Ask Claude for objections/alternatives
        let Some(ref client) = self.ai_client else {
            self.error_message = Some("No AI client configured".to_string());
            return Ok(());
        };

        self.dialogue_state.progress = Some("Generating alternatives...".to_string());

        let system = r#"You are an assistant helping to evaluate a task.
The user has triggered an intervention, meaning they want to reconsider what they're doing.
Provide objections, concerns, or alternative approaches.

Respond ONLY with YAML:
```yaml
objections:
  - label: Short objection title
    description: Why this might be a concern
alternatives:
  - label: Alternative approach title
    description: What this alternative involves
```"#;

        let prompt = format!(
            "The user is reconsidering their current work:\n\n{}\n\nProvide objections and alternatives.",
            context
        );

        match client.chat_raw(&prompt, Some(system)).await {
            Ok(response) => {
                // Parse and present as survey
                let options = self.parse_intervention_response(&response);
                if !options.is_empty() {
                    self.dialogue_state.set_survey(
                        "intervention".to_string(),
                        "Intervention: Consider these alternatives".to_string(),
                        options,
                        true, // multi-select
                    );
                } else {
                    self.error_message = Some("No alternatives generated".to_string());
                }
            }
            Err(e) => {
                self.error_message = Some(format!("Intervention failed: {e}"));
            }
        }

        self.dialogue_state.progress = None;
        self.state = AppState::Executing; // Show dialogue
        Ok(())
    }

    /// Parse intervention response into dialogue options
    fn parse_intervention_response(&self, response: &str) -> Vec<crate::ai::DialogueOption> {
        let mut options = Vec::new();

        // Simple YAML parsing for objections and alternatives
        let mut in_objections = false;
        let mut current_label: Option<String> = None;
        let mut current_desc: Option<String> = None;

        for line in response.lines() {
            let trimmed = line.trim();

            if trimmed.starts_with("objections:") {
                in_objections = true;
            } else if trimmed.starts_with("alternatives:") {
                in_objections = false;
            } else if trimmed.starts_with("- label:") {
                // Save previous item
                if let (Some(label), desc) = (current_label.take(), current_desc.take()) {
                    let prefix = if in_objections { "⚠️ " } else { "💡 " };
                    options.push(crate::ai::DialogueOption {
                        id: format!("opt_{}", options.len()),
                        label: format!("{}{}", prefix, label),
                        description: desc,
                        hint: None,
                    });
                }
                current_label = Some(trimmed.trim_start_matches("- label:").trim().trim_matches('"').to_string());
            } else if trimmed.starts_with("description:") {
                current_desc = Some(trimmed.trim_start_matches("description:").trim().trim_matches('"').to_string());
            }
        }

        // Don't forget the last one
        if let (Some(label), desc) = (current_label, current_desc) {
            let prefix = if in_objections { "⚠️ " } else { "💡 " };
            options.push(crate::ai::DialogueOption {
                id: format!("opt_{}", options.len()),
                label: format!("{}{}", prefix, label),
                description: desc,
                hint: None,
            });
        }

        // Add a "Continue anyway" option
        options.push(crate::ai::DialogueOption {
            id: "continue".to_string(),
            label: "✓ Continue with current task".to_string(),
            description: Some("Dismiss this intervention and continue".to_string()),
            hint: Some("B".to_string()),
        });

        options
    }
}
