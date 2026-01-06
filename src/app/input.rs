//! Keyboard input handling
//!
//! Contains the massive handle_input state machine that processes
//! keyboard input based on current AppState.

use crate::state::{
    AppState, ExecutionStatus, PermissionResponse, SuggestionCard, TaskStatus, UiScaleOption,
};
use winit::keyboard::KeyCode;

/// Keyboard input handling trait for App
pub trait KeyboardInput {
    fn handle_input(&mut self, key: KeyCode);
}

impl KeyboardInput for super::App {
    fn handle_input(&mut self, key: KeyCode) {
        tracing::debug!("handle_input received key: {:?}", key);

        // F2 toggles edit mode globally (works from any state)
        if key == KeyCode::F2 {
            self.edit_mode.toggle();
            tracing::info!("Edit mode: {}", if self.edit_mode.active { "ON" } else { "OFF" });
            self.request_redraw();
            return;
        }

        // Escape exits edit mode if active
        if key == KeyCode::Escape && self.edit_mode.active {
            self.edit_mode.exit();
            tracing::info!("Edit mode: OFF");
            self.request_redraw();
            return;
        }

        // Calculate layout values first to avoid borrow issues
        let columns = self.grid_columns();
        let palace_columns = self.palace_loop_columns();
        let row_height = self.palace_loop_row_height();
        let ui_scale = self.focused_renderer().map(|r| r.ui_scale()).unwrap_or(1.5);
        let screen_height = self
            .focused_window
            .and_then(|id| self.windows.get(&id))
            .map(|w| w.window.inner_size().height as f32)
            .unwrap_or(1080.0);
        let margin_y = (40.0 + 80.0) * ui_scale;
        let visible_height = screen_height - margin_y;

        match &mut self.state {
            AppState::ProjectChooser {
                selected_index,
                show_archived,
            } => {
                // Filter projects based on show_archived toggle
                let visible_projects: Vec<usize> = self
                    .projects
                    .projects
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| *show_archived || !p.archived)
                    .map(|(i, _)| i)
                    .collect();
                let project_count = visible_projects.len();
                if project_count == 0 {
                    return;
                }

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_index >= columns {
                            *selected_index -= columns;
                        } else {
                            let last_row_start = (project_count / columns) * columns;
                            let target = last_row_start + (*selected_index % columns);
                            *selected_index = target.min(project_count - 1);
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        let next = *selected_index + columns;
                        if next < project_count {
                            *selected_index = next;
                        } else {
                            *selected_index = *selected_index % columns;
                        }
                    }
                    KeyCode::ArrowLeft | KeyCode::KeyA => {
                        if *selected_index > 0 {
                            *selected_index -= 1;
                        } else {
                            *selected_index = project_count - 1;
                        }
                    }
                    KeyCode::ArrowRight | KeyCode::KeyD => {
                        if *selected_index < project_count - 1 {
                            *selected_index += 1;
                        } else {
                            *selected_index = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        if let Some(&actual_index) = visible_projects.get(*selected_index) {
                            if let Some(project) = self.projects.projects.get(actual_index) {
                                tracing::info!("Selected project: {}", project.name);
                                self.state = AppState::project_view(project.path.clone());
                            }
                        }
                    }
                    KeyCode::KeyX => {
                        if let Some(&actual_index) = visible_projects.get(*selected_index) {
                            let current_state =
                                std::mem::replace(&mut self.state, AppState::project_chooser());
                            self.state = AppState::ProjectContextMenu {
                                project_index: actual_index,
                                selected_option: 0,
                                previous_state: Box::new(current_state),
                            };
                        }
                    }
                    KeyCode::Tab => {
                        *show_archived = !*show_archived;
                        *selected_index = 0;
                    }
                    _ => {}
                }
            }
            AppState::ProjectView {
                project_path,
                selected_action,
            } => {
                let action_count = crate::state::ProjectAction::all().len();
                let project_path = project_path.clone();

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_action > 0 {
                            *selected_action -= 1;
                        } else {
                            *selected_action = action_count - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_action < action_count - 1 {
                            *selected_action += 1;
                        } else {
                            *selected_action = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        use crate::state::ProjectAction;
                        let action = ProjectAction::all()[*selected_action];
                        tracing::info!("Selected action: {:?}", action);

                        match action {
                            ProjectAction::StartPalaceLoop => {
                                self.state = AppState::PalaceLoop {
                                    project_path: project_path.clone(),
                                    cards: Vec::new(),
                                    focused_index: 0,
                                    hovered_index: None,
                                    generating: true,
                                    current_tool: None,
                                    tool_log: Vec::new(),
                                    thought_log: Vec::new(),
                                    log_scroll_offset: 0,
                                    detail_scroll_offset: 0.0,
                                    detail_max_scroll: 0.0,
                                    card_scroll_offset: 0.0,
                                };

                                let proxy = self.event_proxy.clone();
                                let path = project_path.clone();
                                std::thread::spawn(move || {
                                    super::ai::run_ai_suggestions(path, proxy);
                                });
                            }
                            ProjectAction::Build => {
                                tracing::info!("Build action not yet implemented");
                            }
                            ProjectAction::Run => {
                                tracing::info!("Run action not yet implemented");
                            }
                            ProjectAction::ViewGitHistory => {
                                tracing::info!("Git history action not yet implemented");
                            }
                        }
                    }
                    KeyCode::Backspace => {
                        self.state = AppState::project_chooser();
                    }
                    _ => {}
                }
            }
            AppState::MainMenu {
                selected_item,
                previous_state,
            } => {
                use crate::state::MainMenuItem;
                let item_count = MainMenuItem::all().len();

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_item > 0 {
                            *selected_item -= 1;
                        } else {
                            *selected_item = item_count - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_item < item_count - 1 {
                            *selected_item += 1;
                        } else {
                            *selected_item = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        let item = MainMenuItem::all()[*selected_item];
                        match item {
                            MainMenuItem::Resume => {
                                tracing::info!("Resume - closing menu");
                                self.state = *previous_state.clone();
                            }
                            MainMenuItem::Settings => {
                                tracing::info!("Opening settings submenu");
                                self.state = AppState::SettingsMenu {
                                    selected_item: 0,
                                    previous_state: Box::new(self.state.clone()),
                                };
                            }
                            MainMenuItem::Exit => {
                                tracing::info!("Exit requested");
                                self.exit_requested = true;
                            }
                        }
                    }
                    KeyCode::Backspace | KeyCode::Escape => {
                        self.state = *previous_state.clone();
                    }
                    _ => {}
                }
            }
            AppState::SettingsMenu {
                selected_item,
                previous_state,
            } => {
                use crate::state::SettingsItem;
                let item_count = SettingsItem::all().len();

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_item > 0 {
                            *selected_item -= 1;
                        } else {
                            *selected_item = item_count - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_item < item_count - 1 {
                            *selected_item += 1;
                        } else {
                            *selected_item = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        let item = SettingsItem::all()[*selected_item];
                        match item {
                            SettingsItem::Display => {
                                // Build display options from known monitors
                                use crate::state::{DisplayOption, DisplaySettings};

                                // Load saved display settings
                                let saved: DisplaySettings = self.db
                                    .as_ref()
                                    .and_then(|db| db.get_pref::<DisplaySettings>(super::PREF_DISPLAY_SETTINGS).ok())
                                    .flatten()
                                    .unwrap_or_default();

                                // Fallback primary to focused window if not saved
                                let primary_monitor: Option<String> = saved.primary.or_else(|| {
                                    self.focused_window
                                        .and_then(|id| self.windows.get(&id))
                                        .map(|w| w.monitor_name.clone())
                                });

                                // Fallback enabled to current windows if not saved
                                let enabled_monitors: Vec<String> = if saved.enabled.is_empty() {
                                    self.windows.values()
                                        .filter_map(|w| Some(w.monitor_name.clone()))
                                        .collect()
                                } else {
                                    saved.enabled
                                };

                                let options: Vec<DisplayOption> = self.known_monitors
                                    .iter()
                                    .map(|raw_name| {
                                        let mut opt = DisplayOption::new(raw_name.clone(), raw_name.clone(), 0, 0);
                                        opt.is_primary = primary_monitor.as_ref() == Some(raw_name);
                                        opt.enabled = enabled_monitors.contains(raw_name) || opt.is_primary;
                                        opt
                                    })
                                    .collect();

                                // Show display dialog
                                self.state = AppState::MultiDisplayDialog {
                                    focus_index: 0,
                                    options,
                                    remember_choice: false,
                                    focused_row: 0,
                                    primary_pill_drag: None,
                                    previous_state: Box::new(self.state.clone()),
                                };
                            }
                            SettingsItem::DarkMode => {
                                self.dark_mode = !self.dark_mode;
                                let dark_mode = self.dark_mode;
                                tracing::info!(
                                    "Dark mode: {}",
                                    if dark_mode { "ON" } else { "OFF" }
                                );
                                if let Some(renderer) = self.focused_renderer_mut() {
                                    renderer.set_dark_mode(dark_mode);
                                }
                            }
                            SettingsItem::UiScale => {
                                use super::menus::MenuNavigation;
                                self.state = AppState::UiScaleMenu {
                                    selected_item: self.get_current_scale_index(),
                                    previous_state: Box::new(self.state.clone()),
                                    user_scale_override: self.user_scale_override,
                                };
                            }
                        }
                    }
                    KeyCode::Backspace | KeyCode::Escape => {
                        self.state = *previous_state.clone();
                    }
                    _ => {}
                }
            }
            AppState::UiScaleMenu {
                selected_item,
                previous_state,
                ..
            } => {
                let scale_count = UiScaleOption::all().len();

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_item > 0 {
                            *selected_item -= 1;
                        } else {
                            *selected_item = scale_count - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_item < scale_count - 1 {
                            *selected_item += 1;
                        } else {
                            *selected_item = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        let scale_option = UiScaleOption::all()[*selected_item];
                        let new_state = *previous_state.clone();
                        self.user_scale_override = scale_option.value();
                        let actual_scale = self.user_scale_override.unwrap_or(self.detected_scale);
                        let is_auto = self.user_scale_override.is_none();
                        tracing::info!(
                            "Setting UI scale to {} ({})",
                            scale_option.label(),
                            actual_scale
                        );
                        if let Some(renderer) = self.focused_renderer_mut() {
                            renderer.set_ui_scale(actual_scale, is_auto);
                        }
                        self.state = new_state;
                        self.save_scale_preference();
                    }
                    KeyCode::Backspace | KeyCode::Escape => {
                        self.state = *previous_state.clone();
                    }
                    _ => {}
                }
            }
            AppState::PermissionModal {
                selected_choice,
                previous_state,
                command_prefix,
                command,
            } => {
                use crate::state::PermissionChoice;
                let choice_count = PermissionChoice::all().len();
                let command_prefix = command_prefix.clone();
                let command = command.clone();
                let previous_state = previous_state.clone();

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_choice > 0 {
                            *selected_choice -= 1;
                        } else {
                            *selected_choice = choice_count - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_choice < choice_count - 1 {
                            *selected_choice += 1;
                        } else {
                            *selected_choice = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        if let Some(tx) = self.permission_response_tx.take() {
                            let _ = tx.send(PermissionResponse::Approved);
                        }
                        self.state = *previous_state;
                    }
                    KeyCode::KeyX => {
                        super::App::approve_command_prefix(&command_prefix);
                        if let Some(tx) = self.permission_response_tx.take() {
                            let _ = tx.send(PermissionResponse::ApprovedAlways(command_prefix));
                        }
                        self.state = *previous_state;
                    }
                    KeyCode::Backspace | KeyCode::Escape => {
                        if let Some(tx) = self.permission_response_tx.take() {
                            let _ = tx.send(PermissionResponse::Denied);
                        }
                        self.state = *previous_state;
                    }
                    KeyCode::KeyY => {
                        tracing::info!("Suggest else requested for: {}", command);
                        if let Some(tx) = self.permission_response_tx.take() {
                            let _ = tx.send(PermissionResponse::SuggestElse {
                                original_command: command,
                            });
                        }
                        self.state = *previous_state;
                    }
                    _ => {}
                }
            }
            AppState::PalaceLoop {
                cards,
                focused_index,
                detail_scroll_offset,
                card_scroll_offset,
                generating,
                ..
            } => {
                let card_count = cards.len();
                let total_items = card_count + 1;

                let scroll_into_view = |focused: usize, scroll: &mut f32| {
                    let focused_row = focused / palace_columns;
                    let card_top = focused_row as f32 * row_height;
                    let card_bottom = card_top + row_height;

                    if card_top < *scroll {
                        *scroll = card_top;
                    }
                    if card_bottom > *scroll + visible_height {
                        *scroll = card_bottom - visible_height;
                    }
                    *scroll = scroll.max(0.0);
                };

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *focused_index >= palace_columns {
                            *focused_index -= palace_columns;
                            *detail_scroll_offset = 0.0;
                        } else if total_items > 0 {
                            let last_row_start =
                                (total_items.saturating_sub(1) / palace_columns) * palace_columns;
                            let target = last_row_start + (*focused_index % palace_columns);
                            *focused_index = target.min(total_items - 1);
                            *detail_scroll_offset = 0.0;
                        }
                        scroll_into_view(*focused_index, card_scroll_offset);
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        let next = *focused_index + palace_columns;
                        if next < total_items {
                            *focused_index = next;
                            *detail_scroll_offset = 0.0;
                        } else if total_items > 0 {
                            *focused_index = *focused_index % palace_columns;
                            if *focused_index >= total_items {
                                *focused_index = 0;
                            }
                            *detail_scroll_offset = 0.0;
                        }
                        scroll_into_view(*focused_index, card_scroll_offset);
                    }
                    KeyCode::ArrowLeft | KeyCode::KeyA => {
                        if *focused_index > 0 {
                            *focused_index -= 1;
                            *detail_scroll_offset = 0.0;
                        } else if total_items > 0 {
                            *focused_index = total_items - 1;
                            *detail_scroll_offset = 0.0;
                        }
                        scroll_into_view(*focused_index, card_scroll_offset);
                    }
                    KeyCode::ArrowRight | KeyCode::KeyD => {
                        if total_items > 0 && *focused_index < total_items - 1 {
                            *focused_index += 1;
                            *detail_scroll_offset = 0.0;
                        } else {
                            *focused_index = 0;
                            *detail_scroll_offset = 0.0;
                        }
                        scroll_into_view(*focused_index, card_scroll_offset);
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        if *focused_index == card_count {
                            self.state = AppState::AddCardMenu {
                                selected_option: 0,
                                previous_state: Box::new(self.state.clone()),
                            };
                        } else if let Some(card) = cards.get_mut(*focused_index) {
                            card.selected = !card.selected;
                            tracing::info!("Card {} selected: {}", card.id, card.selected);
                        }
                    }
                    KeyCode::Backspace | KeyCode::Escape => {
                        if let AppState::PalaceLoop { project_path, .. } = &self.state {
                            self.state = AppState::project_view(project_path.clone());
                        }
                    }
                    KeyCode::KeyZ => {
                        if let Some(card) = cards.get_mut(0) {
                            card.selected = !card.selected;
                            tracing::debug!("Quick-select Z: card 0 = {}", card.selected);
                        }
                    }
                    KeyCode::KeyX => {
                        if *generating {
                            if let Some(card) = cards.get_mut(1) {
                                card.selected = !card.selected;
                                tracing::debug!("Quick-select X: card 1 = {}", card.selected);
                            }
                        } else {
                            let has_selected = cards.iter().any(|c| c.selected);
                            if has_selected {
                                self.state = AppState::ExecuteModal {
                                    selected_option: 0,
                                    previous_state: Box::new(self.state.clone()),
                                };
                            }
                        }
                    }
                    KeyCode::KeyC => {
                        if let Some(card) = cards.get_mut(2) {
                            card.selected = !card.selected;
                            tracing::debug!("Quick-select C: card 2 = {}", card.selected);
                        }
                    }
                    KeyCode::KeyV => {
                        if let Some(card) = cards.get_mut(3) {
                            card.selected = !card.selected;
                            tracing::debug!("Quick-select V: card 3 = {}", card.selected);
                        }
                    }
                    _ => {}
                }
            }
            AppState::ExecuteModal {
                selected_option,
                previous_state,
            } => {
                use crate::state::ExecuteOption;
                let option_count = ExecuteOption::all().len();
                let previous_state = previous_state.clone();

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_option > 0 {
                            *selected_option -= 1;
                        } else {
                            *selected_option = option_count - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_option < option_count - 1 {
                            *selected_option += 1;
                        } else {
                            *selected_option = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        let option = ExecuteOption::all()[*selected_option];
                        tracing::info!("Execute option selected: {:?}", option);

                        if let AppState::PalaceLoop {
                            project_path,
                            cards,
                            ..
                        } = &*previous_state
                        {
                            let selected_cards: Vec<SuggestionCard> =
                                cards.iter().filter(|c| c.selected).cloned().collect();

                            if selected_cards.is_empty() {
                                tracing::warn!("No cards selected for execution");
                                self.state = *previous_state;
                                return;
                            }

                            let project_path = project_path.clone();
                            let card_count = selected_cards.len();
                            tracing::info!(
                                "Starting execution of {} cards with {:?}",
                                card_count,
                                option
                            );

                            let mut task_statuses = vec![TaskStatus::Pending; card_count];
                            if !task_statuses.is_empty() {
                                task_statuses[0] = TaskStatus::InProgress;
                            }

                            self.state = AppState::Executing {
                                project_path: project_path.clone(),
                                executing_cards: selected_cards.clone(),
                                all_cards: cards.clone(),
                                status: ExecutionStatus::Running {
                                    current_card: 0,
                                    total_cards: card_count,
                                },
                                task_statuses,
                                tool_log: Vec::new(),
                                thought_log: Vec::new(),
                                log_scroll_offset: 0.0,
                                executor: option,
                                previous_state: previous_state.clone(),
                                quest_log_visible: false,
                                quest_log_focus: 0,
                                tokens_used: 0,
                                request_active: true,
                            };

                            let proxy = (*self.event_proxy).clone();
                            crate::ai::spawn_execution(option, project_path, selected_cards, proxy);
                        } else {
                            tracing::error!("ExecuteModal previous_state is not PalaceLoop");
                            self.state = *previous_state;
                        }
                    }
                    KeyCode::Backspace | KeyCode::Escape => {
                        self.state = *previous_state;
                    }
                    _ => {}
                }
            }
            AppState::Survey {
                options,
                focused_index,
                custom_input,
                custom_active,
                multi_select,
                selected_indices,
                use_quick_select,
                scroll_offset,
                previous_state,
                response_tx,
                ..
            } => {
                use crate::state::SurveyResponse;
                let option_count = options.len();
                let previous_state = previous_state.clone();
                let has_custom = true;
                let total = option_count + if has_custom { 1 } else { 0 };

                // Calculate visible_count from renderer values captured in local vars
                let (screen_height, ui_scale) = self.focused_window
                    .and_then(|id| self.windows.get(&id))
                    .map(|w| (w.renderer.size().height as f32, w.renderer.ui_scale()))
                    .unwrap_or((600.0, 1.0));
                let scale = |v: f32| v * ui_scale;
                let margin = scale(48.0);
                let title_height = scale(80.0);
                let card_height = scale(60.0);
                let card_gap = scale(10.0);
                let available_height = screen_height - margin * 2.0 - title_height - scale(60.0);
                let visible_count = (available_height / (card_height + card_gap)).floor() as usize;

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        *use_quick_select = false;
                        if !*custom_active {
                            if *focused_index > 0 {
                                *focused_index -= 1;
                            } else {
                                *focused_index = total - 1;
                                *scroll_offset = total.saturating_sub(visible_count);
                            }
                            if *focused_index < *scroll_offset {
                                *scroll_offset = *focused_index;
                            }
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        *use_quick_select = false;
                        if !*custom_active {
                            if *focused_index < total - 1 {
                                *focused_index += 1;
                            } else {
                                *focused_index = 0;
                                *scroll_offset = 0;
                            }
                            if *focused_index >= *scroll_offset + visible_count {
                                *scroll_offset = focused_index.saturating_sub(visible_count - 1);
                            }
                            if *focused_index < *scroll_offset {
                                *scroll_offset = *focused_index;
                            }
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        if *custom_active {
                            if !custom_input.is_empty() {
                                if let Some(tx) = response_tx.take() {
                                    let _ = tx.send(SurveyResponse::Custom(custom_input.clone()));
                                }
                                self.state = *previous_state;
                            }
                        } else if *use_quick_select && option_count > 0 {
                            if let Some(tx) = response_tx.take() {
                                let _ = tx.send(SurveyResponse::Selected(vec![0]));
                            }
                            self.state = *previous_state;
                        } else if *focused_index >= option_count {
                            *custom_active = true;
                        } else if *multi_select {
                            if selected_indices.contains(focused_index) {
                                selected_indices.retain(|&i| i != *focused_index);
                            } else {
                                selected_indices.push(*focused_index);
                            }
                        } else {
                            if let Some(tx) = response_tx.take() {
                                let _ = tx.send(SurveyResponse::Selected(vec![*focused_index]));
                            }
                            self.state = *previous_state;
                        }
                    }
                    KeyCode::KeyX => {
                        if !*custom_active {
                            if *multi_select {
                                if !selected_indices.is_empty() {
                                    if let Some(tx) = response_tx.take() {
                                        let _ = tx.send(SurveyResponse::Selected(
                                            selected_indices.clone(),
                                        ));
                                    }
                                    self.state = *previous_state;
                                }
                            } else if *use_quick_select && option_count > 1 {
                                if let Some(tx) = response_tx.take() {
                                    let _ = tx.send(SurveyResponse::Selected(vec![1]));
                                }
                                self.state = *previous_state;
                            }
                        }
                    }
                    KeyCode::Backspace => {
                        if *custom_active {
                            custom_input.pop();
                        } else if *use_quick_select && option_count > 2 {
                            if let Some(tx) = response_tx.take() {
                                let _ = tx.send(SurveyResponse::Selected(vec![2]));
                            }
                            self.state = *previous_state;
                        }
                    }
                    KeyCode::KeyY => {
                        if !*custom_active && *use_quick_select && option_count > 3 {
                            if let Some(tx) = response_tx.take() {
                                let _ = tx.send(SurveyResponse::Selected(vec![3]));
                            }
                            self.state = *previous_state;
                        }
                    }
                    KeyCode::Escape => {
                        if *custom_active {
                            *custom_active = false;
                        } else {
                            if let Some(tx) = response_tx.take() {
                                let _ = tx.send(SurveyResponse::Cancelled);
                            }
                            self.state = *previous_state;
                        }
                    }
                    _ => {}
                }
            }
            AppState::Executing {
                status,
                previous_state,
                log_scroll_offset,
                quest_log_visible,
                quest_log_focus,
                all_cards,
                ..
            } => {
                let card_count = all_cards.len();
                let columns = 5;

                match key {
                    KeyCode::Enter | KeyCode::Space => {
                        if status.is_done() && !*quest_log_visible {
                            self.state = *previous_state.clone();
                        }
                    }
                    KeyCode::Escape | KeyCode::Backspace => {
                        if *quest_log_visible {
                            *quest_log_visible = false;
                        } else if status.is_done() {
                            self.state = *previous_state.clone();
                        } else {
                            *status = ExecutionStatus::Cancelled;
                        }
                    }
                    KeyCode::Tab => {
                        if self.window_focused {
                            *quest_log_visible = !*quest_log_visible;
                        }
                    }
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *quest_log_visible {
                            if *quest_log_focus >= columns {
                                *quest_log_focus -= columns;
                            } else if card_count > 0 {
                                let last_row_start =
                                    (card_count.saturating_sub(1) / columns) * columns;
                                let target = last_row_start + (*quest_log_focus % columns);
                                *quest_log_focus = target.min(card_count - 1);
                            }
                        } else {
                            *log_scroll_offset = (*log_scroll_offset - 40.0).max(0.0);
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *quest_log_visible {
                            let next = *quest_log_focus + columns;
                            if next < card_count {
                                *quest_log_focus = next;
                            } else if card_count > 0 {
                                *quest_log_focus = *quest_log_focus % columns;
                                if *quest_log_focus >= card_count {
                                    *quest_log_focus = 0;
                                }
                            }
                        } else {
                            *log_scroll_offset += 40.0;
                        }
                    }
                    KeyCode::ArrowLeft | KeyCode::KeyA => {
                        if *quest_log_visible {
                            if *quest_log_focus > 0 {
                                *quest_log_focus -= 1;
                            } else if card_count > 0 {
                                *quest_log_focus = card_count - 1;
                            }
                        }
                    }
                    KeyCode::ArrowRight | KeyCode::KeyD => {
                        if *quest_log_visible {
                            if card_count > 0 && *quest_log_focus < card_count - 1 {
                                *quest_log_focus += 1;
                            } else {
                                *quest_log_focus = 0;
                            }
                        }
                    }
                    _ => {}
                }
            }
            AppState::AddCardMenu {
                selected_option,
                previous_state,
            } => {
                use crate::state::AddCardOption;
                let option_count = AddCardOption::all().len();
                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_option > 0 {
                            *selected_option -= 1;
                        } else {
                            *selected_option = option_count.saturating_sub(1);
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_option < option_count - 1 {
                            *selected_option += 1;
                        } else {
                            *selected_option = 0;
                        }
                    }
                    KeyCode::Enter => {
                        let option = AddCardOption::all()[*selected_option];
                        match option {
                            AddCardOption::GenerateMore => {
                                let prev = std::mem::replace(
                                    previous_state,
                                    Box::new(AppState::project_chooser()),
                                );
                                self.state = *prev;
                                tracing::info!("Generate more suggestions requested");
                            }
                            AddCardOption::InterviewMe => {
                                let prev = std::mem::replace(
                                    previous_state,
                                    Box::new(AppState::project_chooser()),
                                );
                                self.state = *prev;
                                tracing::info!("Interview me requested");
                            }
                            AddCardOption::CustomTask => {
                                let prev = std::mem::replace(
                                    previous_state,
                                    Box::new(AppState::project_chooser()),
                                );
                                self.state = AppState::CustomTaskInput {
                                    name: String::new(),
                                    description: String::new(),
                                    active_field: 0,
                                    cursor: 0,
                                    previous_state: prev,
                                };
                            }
                            AddCardOption::Cancel => {
                                let prev = std::mem::replace(
                                    previous_state,
                                    Box::new(AppState::project_chooser()),
                                );
                                self.state = *prev;
                            }
                        }
                    }
                    KeyCode::Escape | KeyCode::Backspace => {
                        let prev = std::mem::replace(
                            previous_state,
                            Box::new(AppState::project_chooser()),
                        );
                        self.state = *prev;
                    }
                    _ => {}
                }
            }
            AppState::CustomTaskInput {
                name,
                description,
                active_field,
                cursor,
                previous_state,
            } => {
                match key {
                    KeyCode::Enter => {
                        if !name.trim().is_empty() || !description.trim().is_empty() {
                            let title = if name.trim().is_empty() {
                                description.chars().take(50).collect::<String>()
                            } else {
                                name.clone()
                            };

                            let desc = if description.trim().is_empty() {
                                String::new()
                            } else {
                                description.clone()
                            };

                            let new_card = SuggestionCard {
                                id: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| d.as_millis() as usize)
                                    .unwrap_or(0),
                                title,
                                category: "custom".to_string(),
                                description: desc,
                                command: None,
                                selected: false,
                                streaming: false,
                            };

                            let prev = std::mem::replace(
                                previous_state,
                                Box::new(AppState::project_chooser()),
                            );
                            match *prev {
                                AppState::PalaceLoop {
                                    project_path,
                                    mut cards,
                                    focused_index,
                                    hovered_index,
                                    generating,
                                    current_tool,
                                    tool_log,
                                    thought_log,
                                    log_scroll_offset,
                                    detail_scroll_offset,
                                    detail_max_scroll,
                                    card_scroll_offset,
                                } => {
                                    cards.push(new_card);
                                    self.state = AppState::PalaceLoop {
                                        project_path,
                                        cards,
                                        focused_index,
                                        hovered_index,
                                        generating,
                                        current_tool,
                                        tool_log,
                                        thought_log,
                                        log_scroll_offset,
                                        detail_scroll_offset,
                                        detail_max_scroll,
                                        card_scroll_offset,
                                    };
                                }
                                other => {
                                    self.state = other;
                                }
                            }
                        } else {
                            let prev = std::mem::replace(
                                previous_state,
                                Box::new(AppState::project_chooser()),
                            );
                            self.state = *prev;
                        }
                    }
                    KeyCode::Tab | KeyCode::ArrowDown => {
                        *active_field = (*active_field + 1) % 2;
                        *cursor = if *active_field == 0 {
                            name.len()
                        } else {
                            description.len()
                        };
                    }
                    KeyCode::ArrowUp => {
                        *active_field = if *active_field == 0 { 1 } else { 0 };
                        *cursor = if *active_field == 0 {
                            name.len()
                        } else {
                            description.len()
                        };
                    }
                    KeyCode::Escape => {
                        let prev = std::mem::replace(
                            previous_state,
                            Box::new(AppState::project_chooser()),
                        );
                        self.state = *prev;
                    }
                    KeyCode::Backspace => {
                        let active_text = if *active_field == 0 {
                            name
                        } else {
                            description
                        };
                        if *cursor > 0 {
                            active_text.remove(*cursor - 1);
                            *cursor -= 1;
                        }
                    }
                    KeyCode::ArrowLeft => {
                        if *cursor > 0 {
                            *cursor -= 1;
                        }
                    }
                    KeyCode::ArrowRight => {
                        let active_len = if *active_field == 0 {
                            name.len()
                        } else {
                            description.len()
                        };
                        if *cursor < active_len {
                            *cursor += 1;
                        }
                    }
                    _ => {}
                }
            }
            AppState::ProjectContextMenu {
                project_index,
                selected_option,
                previous_state,
            } => {
                use crate::state::ProjectContextOption;
                let option_count = ProjectContextOption::all().len();
                let project_index = *project_index;

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_option > 0 {
                            *selected_option -= 1;
                        } else {
                            *selected_option = option_count.saturating_sub(1);
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_option < option_count - 1 {
                            *selected_option += 1;
                        } else {
                            *selected_option = 0;
                        }
                    }
                    KeyCode::Escape | KeyCode::Backspace => {
                        let prev = std::mem::replace(
                            previous_state,
                            Box::new(AppState::project_chooser()),
                        );
                        self.state = *prev;
                    }
                    KeyCode::Enter => {
                        let option = ProjectContextOption::all()[*selected_option];
                        match option {
                            ProjectContextOption::Remove => {
                                self.projects.projects.remove(project_index);
                                let _ = self.projects.save();
                                self.state = AppState::project_chooser();
                                tracing::info!("Removed project at index {}", project_index);
                            }
                            ProjectContextOption::ChangeLanguage => {
                                let languages = vec![
                                    "Rust".to_string(),
                                    "TypeScript".to_string(),
                                    "Python".to_string(),
                                    "Go".to_string(),
                                    "Vue".to_string(),
                                    "React".to_string(),
                                    "Ruby".to_string(),
                                    "Java".to_string(),
                                    "C#".to_string(),
                                    "Other".to_string(),
                                ];
                                let prev = std::mem::replace(
                                    previous_state,
                                    Box::new(AppState::project_chooser()),
                                );
                                self.state = AppState::LanguageSelector {
                                    project_index,
                                    selected_index: 0,
                                    languages,
                                    previous_state: prev,
                                };
                            }
                            ProjectContextOption::ToggleArchive => {
                                let (name, archived) =
                                    if let Some(project) = self.projects.projects.get_mut(project_index)
                                    {
                                        project.archived = !project.archived;
                                        (project.name.clone(), project.archived)
                                    } else {
                                        (String::new(), false)
                                    };
                                let _ = self.projects.save();
                                tracing::info!(
                                    "Toggled archive for project: {} -> {}",
                                    name,
                                    archived
                                );
                                let prev = std::mem::replace(
                                    previous_state,
                                    Box::new(AppState::project_chooser()),
                                );
                                self.state = *prev;
                            }
                            ProjectContextOption::Cancel => {
                                let prev = std::mem::replace(
                                    previous_state,
                                    Box::new(AppState::project_chooser()),
                                );
                                self.state = *prev;
                            }
                        }
                    }
                    _ => {}
                }
            }
            AppState::LanguageSelector {
                project_index,
                selected_index,
                languages,
                previous_state,
            } => {
                let lang_count = languages.len();
                let project_index = *project_index;

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *selected_index > 0 {
                            *selected_index -= 1;
                        } else {
                            *selected_index = lang_count.saturating_sub(1);
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *selected_index < lang_count - 1 {
                            *selected_index += 1;
                        } else {
                            *selected_index = 0;
                        }
                    }
                    KeyCode::Escape | KeyCode::Backspace => {
                        let prev = std::mem::replace(
                            previous_state,
                            Box::new(AppState::project_chooser()),
                        );
                        self.state = *prev;
                    }
                    KeyCode::Enter => {
                        let lang_name = languages.get(*selected_index).cloned();
                        if let Some(lang) = lang_name {
                            let project_name =
                                if let Some(project) = self.projects.projects.get_mut(project_index)
                                {
                                    project.languages = vec![lang.clone()];
                                    project.name.clone()
                                } else {
                                    String::new()
                                };
                            let _ = self.projects.save();
                            tracing::info!(
                                "Changed language for project {} to {}",
                                project_name,
                                lang
                            );
                        }
                        self.state = AppState::project_chooser();
                    }
                    _ => {}
                }
            }
            AppState::MultiDisplayDialog {
                focus_index,
                options,
                remember_choice,
                focused_row,
                previous_state,
                ..
            } => {
                // Simple vertical list: each monitor, then remember, then apply
                // focused_row: 0..n = monitors, n = remember, n+1 = apply
                // focus_index tracks which monitor when in monitor rows
                let option_count = options.len();
                let total_items = option_count + 2; // monitors + remember + apply

                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *focused_row > 0 {
                            *focused_row -= 1;
                            // Update focus_index when entering monitor section
                            if *focused_row < option_count {
                                *focus_index = *focused_row;
                            }
                        } else {
                            // Wrap to bottom
                            *focused_row = total_items - 1;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *focused_row < total_items - 1 {
                            *focused_row += 1;
                            // Update focus_index when entering monitor section
                            if *focused_row < option_count {
                                *focus_index = *focused_row;
                            }
                        } else {
                            // Wrap to top
                            *focused_row = 0;
                            *focus_index = 0;
                        }
                    }
                    // Tab still works for quick section jumping
                    KeyCode::Tab => {
                        if *focused_row < option_count {
                            *focused_row = option_count; // Jump to remember
                        } else if *focused_row == option_count {
                            *focused_row = option_count + 1; // Jump to apply
                        } else {
                            *focused_row = 0; // Jump back to first monitor
                            *focus_index = 0;
                        }
                    }
                    KeyCode::Space | KeyCode::Enter => {
                        // Activate the focused item based on row
                        if *focused_row < option_count {
                            // Monitor row - toggle enabled
                            if let Some(opt) = options.get_mut(*focused_row) {
                                if opt.is_primary {
                                    // Can't disable primary
                                } else {
                                    opt.enabled = !opt.enabled;
                                }
                            }
                        } else if *focused_row == option_count {
                            // Remember checkbox
                            *remember_choice = !*remember_choice;
                        } else if *focused_row == option_count + 1 {
                            // Apply button
                            Self::apply_display_settings(
                                options,
                                *remember_choice,
                                &self.db,
                                &self.event_proxy,
                            );
                            let prev = std::mem::replace(
                                previous_state,
                                Box::new(AppState::project_chooser()),
                            );
                            self.state = *prev;
                        }
                    }
                    // P key = set as primary (when on a monitor row)
                    KeyCode::KeyP => {
                        if *focused_row < option_count {
                            for (i, opt) in options.iter_mut().enumerate() {
                                if i == *focused_row {
                                    opt.is_primary = true;
                                    opt.enabled = true;
                                } else {
                                    opt.is_primary = false;
                                }
                            }
                        }
                    }
                    KeyCode::Escape | KeyCode::Backspace => {
                        let prev = std::mem::replace(
                            previous_state,
                            Box::new(AppState::project_chooser()),
                        );
                        self.state = *prev;
                    }
                    _ => {}
                }
            }
            AppState::NewMonitorDialog {
                focus_index,
                monitor_name,
                previous_state,
                ..
            } => {
                // Simple dialog: Yes (0), No (1), Never (2)
                match key {
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        if *focus_index > 0 {
                            *focus_index -= 1;
                        } else {
                            *focus_index = 2;
                        }
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        if *focus_index < 2 {
                            *focus_index += 1;
                        } else {
                            *focus_index = 0;
                        }
                    }
                    KeyCode::Enter | KeyCode::Space => {
                        let selected = *focus_index;
                        let name = monitor_name.clone();

                        match selected {
                            0 => {
                                // Yes - extend to this monitor
                                let _ = self.event_proxy.send_event(super::AppEvent::CreateWindow {
                                    monitor_name: name,
                                });
                            }
                            1 => {
                                // No - don't extend (do nothing)
                            }
                            2 => {
                                // Never - save preference to not ask again
                                if let Some(ref db) = self.db {
                                    let _ = db.set_pref(super::PREF_MULTI_DISPLAY, &"never_ask");
                                }
                            }
                            _ => {}
                        }
                        // Return to previous state
                        let prev = std::mem::replace(
                            previous_state,
                            Box::new(AppState::project_chooser()),
                        );
                        self.state = *prev;
                    }
                    KeyCode::Escape | KeyCode::Backspace => {
                        let prev = std::mem::replace(
                            previous_state,
                            Box::new(AppState::project_chooser()),
                        );
                        self.state = *prev;
                    }
                    _ => {}
                }
            }
        }
    }
}

impl super::App {
    /// Apply display settings and send events
    fn apply_display_settings(
        options: &[crate::state::DisplayOption],
        remember: bool,
        db: &Option<crate::persistence::PalaceDB>,
        event_proxy: &std::sync::Arc<winit::event_loop::EventLoopProxy<super::AppEvent>>,
    ) {
        // Find enabled monitors
        let enabled: Vec<_> = options.iter().filter(|o| o.enabled).collect();
        let primary = options.iter().find(|o| o.is_primary);

        // ALWAYS save display settings as a proper struct
        if let Some(ref db) = db {
            let settings = crate::state::DisplaySettings {
                primary: primary.map(|p| p.id.clone()),
                enabled: enabled.iter().map(|o| o.id.clone()).collect(),
            };
            let _ = db.set_pref(super::PREF_DISPLAY_SETTINGS, &settings);
        }

        // Save multi-display preference if requested (for "never ask" behavior)
        if remember {
            if let Some(ref db) = db {
                let _ = db.set_pref(super::PREF_MULTI_DISPLAY, &"enabled");
            }
        }

        tracing::info!(
            "Applying display settings: primary={:?}, enabled={}",
            primary.map(|p| &p.name),
            enabled.len()
        );

        // First, set the primary display
        if let Some(p) = primary {
            let _ = event_proxy.send_event(super::AppEvent::SwitchDisplay {
                monitor_name: p.id.clone(), // Use raw monitor name, not display name with position suffix
            });
        }

        // Then create windows on other enabled monitors
        for opt in enabled.iter() {
            if !opt.is_primary {
                let _ = event_proxy.send_event(super::AppEvent::CreateWindow {
                    monitor_name: opt.id.clone(), // Use raw monitor name, not display name with position suffix
                });
            }
        }

        // TODO: Close windows on disabled monitors
    }

    /// Calculate columns for project chooser grid
    pub(crate) fn grid_columns(&self) -> usize {
        if let Some(window_id) = self.focused_window {
            if let Some(palace_window) = self.windows.get(&window_id) {
                let width = palace_window.window.inner_size().width as f32;
                let margin = 72.0;
                let card_width = 384.0;
                let gap = 28.8;
                let available = width - margin * 2.0;
                let cols = ((available + gap) / (card_width + gap)).floor() as usize;
                return cols.max(1);
            }
        }
        4
    }
}
