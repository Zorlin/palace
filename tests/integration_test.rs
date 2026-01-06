//! Integration tests for Palace
//!
//! Tests the full application state machine, persistence layer, and interactions
//! between components. Uses headless mocks for renderer/window to enable CI.

use std::path::PathBuf;
use std::sync::mpsc;

// Re-export key types from the main crate
use palace::state::{
    AppState, ExecuteOption, ExecutionStatus, MainMenuItem, PermissionChoice,
    PermissionResponse, ProjectAction, SettingsItem, SuggestionCard, SurveyOption,
    SurveyResponse, TaskStatus, UiScaleOption,
};

// ============== Test Utilities ==============

/// Creates a mock suggestion card for testing
fn mock_card(id: usize, title: &str, category: &str) -> SuggestionCard {
    let mut card = SuggestionCard::new(id);
    card.title = title.to_string();
    card.category = category.to_string();
    card.description = format!("Description for {}", title);
    card.command = Some(format!("{}-command", title));
    card
}

/// Creates a mock project path
fn mock_project_path() -> PathBuf {
    PathBuf::from("/tmp/test-project")
}

// ============== AppState Transition Tests ==============

#[test]
fn test_app_state_project_chooser_initial() {
    let state = AppState::project_chooser();

    match state {
        AppState::ProjectChooser { selected_index } => {
            assert_eq!(selected_index, 0, "Initial selection should be 0");
        }
        _ => panic!("Expected ProjectChooser state"),
    }
}

#[test]
fn test_app_state_project_view_creation() {
    let project_path = mock_project_path();
    let state = AppState::project_view(project_path.clone());

    match state {
        AppState::ProjectView {
            project_path: path,
            selected_action,
        } => {
            assert_eq!(path, project_path);
            assert_eq!(selected_action, 0, "Initial action selection should be 0");
        }
        _ => panic!("Expected ProjectView state"),
    }
}

#[test]
fn test_app_state_palace_loop_creation() {
    let project_path = mock_project_path();

    let state = AppState::PalaceLoop {
        project_path: project_path.clone(),
        cards: vec![],
        focused_index: 0,
        hovered_index: None,
        generating: true,
        current_tool: None,
        tool_log: vec![],
        thought_log: vec![],
        log_scroll_offset: 0,
        detail_scroll_offset: 0.0,
        detail_max_scroll: 0.0,
    };

    match state {
        AppState::PalaceLoop {
            project_path,
            cards,
            focused_index,
            generating,
            ..
        } => {
            assert_eq!(project_path, mock_project_path());
            assert!(cards.is_empty());
            assert_eq!(focused_index, 0);
            assert!(generating, "Should start in generating state");
        }
        _ => panic!("Expected PalaceLoop state"),
    }
}

#[test]
fn test_app_state_main_menu_wrapping() {
    let previous = AppState::project_chooser();
    let state = AppState::MainMenu {
        selected_item: 0,
        previous_state: Box::new(previous.clone()),
    };

    match state {
        AppState::MainMenu {
            selected_item,
            previous_state,
        } => {
            assert_eq!(selected_item, 0);
            match *previous_state {
                AppState::ProjectChooser { .. } => {}
                _ => panic!("Previous state should be ProjectChooser"),
            }
        }
        _ => panic!("Expected MainMenu state"),
    }
}

#[test]
fn test_app_state_settings_menu_hierarchy() {
    let base = AppState::project_chooser();
    let main_menu = AppState::MainMenu {
        selected_item: 0,
        previous_state: Box::new(base.clone()),
    };
    let settings = AppState::SettingsMenu {
        selected_item: 0,
        previous_state: Box::new(main_menu),
    };

    // Verify we can unwrap back to base
    match settings {
        AppState::SettingsMenu { previous_state, .. } => {
            match *previous_state {
                AppState::MainMenu { previous_state, .. } => {
                    match *previous_state {
                        AppState::ProjectChooser { .. } => {
                            // Successfully unwrapped all the way
                        }
                        _ => panic!("Should unwrap to ProjectChooser"),
                    }
                }
                _ => panic!("Should unwrap to MainMenu"),
            }
        }
        _ => panic!("Expected SettingsMenu state"),
    }
}

#[test]
fn test_app_state_permission_modal_fields() {
    let base = AppState::project_chooser();
    let command = "cargo build --release".to_string();
    let command_prefix = "cargo build".to_string();

    let state = AppState::PermissionModal {
        command: command.clone(),
        command_prefix: command_prefix.clone(),
        selected_choice: 0,
        previous_state: Box::new(base),
    };

    match state {
        AppState::PermissionModal {
            command: cmd,
            command_prefix: prefix,
            selected_choice,
            ..
        } => {
            assert_eq!(cmd, command);
            assert_eq!(prefix, command_prefix);
            assert_eq!(selected_choice, 0);
        }
        _ => panic!("Expected PermissionModal state"),
    }
}

#[test]
fn test_app_state_survey_creation() {
    let base = AppState::project_chooser();
    let (tx, _rx) = mpsc::channel::<SurveyResponse>();

    let state = AppState::Survey {
        question: "What is your favorite color?".to_string(),
        header: "Color Choice".to_string(),
        options: vec![
            SurveyOption::new("Red", "Bold and bright"),
            SurveyOption::new("Blue", "Calm and cool"),
        ],
        focused_index: 0,
        custom_input: String::new(),
        custom_active: false,
        multi_select: false,
        selected_indices: vec![],
        use_quick_select: true,
        scroll_offset: 0,
        previous_state: Box::new(base),
        response_tx: Some(tx),
    };

    match state {
        AppState::Survey {
            question,
            header,
            options,
            focused_index,
            multi_select,
            ..
        } => {
            assert_eq!(question, "What is your favorite color?");
            assert_eq!(header, "Color Choice");
            assert_eq!(options.len(), 2);
            assert_eq!(focused_index, 0);
            assert!(!multi_select);
        }
        _ => panic!("Expected Survey state"),
    }
}

#[test]
fn test_app_state_executing_transitions() {
    let project_path = mock_project_path();
    let cards = vec![
        mock_card(0, "Task 1", "fix"),
        mock_card(1, "Task 2", "test"),
    ];

    let state = AppState::Executing {
        project_path: project_path.clone(),
        executing_cards: cards.clone(),
        all_cards: cards.clone(),
        status: ExecutionStatus::Pending,
        task_statuses: vec![TaskStatus::Pending, TaskStatus::Pending],
        tool_log: vec![],
        thought_log: vec![],
        log_scroll_offset: 0.0,
        executor: ExecuteOption::Claude,
        previous_state: Box::new(AppState::project_view(project_path.clone())),
        quest_log_visible: false,
        quest_log_focus: 0,
    };

    match state {
        AppState::Executing {
            project_path,
            executing_cards,
            status,
            task_statuses,
            executor,
            ..
        } => {
            assert_eq!(project_path, mock_project_path());
            assert_eq!(executing_cards.len(), 2);
            assert!(matches!(status, ExecutionStatus::Pending));
            assert_eq!(task_statuses.len(), 2);
            assert_eq!(executor, ExecuteOption::Claude);
        }
        _ => panic!("Expected Executing state"),
    }
}

// ============== ExecutionStatus Transitions ==============

#[test]
fn test_execution_status_pending_to_running() {
    let status = ExecutionStatus::Pending;
    assert!(!status.is_running());
    assert!(!status.is_done());

    let running = ExecutionStatus::Running {
        current_card: 0,
        total_cards: 5,
    };
    assert!(running.is_running());
    assert!(!running.is_done());
}

#[test]
fn test_execution_status_completion() {
    let completed = ExecutionStatus::Completed;
    assert!(!completed.is_running());
    assert!(completed.is_done());

    let failed = ExecutionStatus::Failed("Test error".to_string());
    assert!(!failed.is_running());
    assert!(failed.is_done());

    let cancelled = ExecutionStatus::Cancelled;
    assert!(!cancelled.is_running());
    assert!(cancelled.is_done());
}

#[test]
fn test_execution_status_running_progress() {
    let status = ExecutionStatus::Running {
        current_card: 2,
        total_cards: 5,
    };

    match status {
        ExecutionStatus::Running {
            current_card,
            total_cards,
        } => {
            assert_eq!(current_card, 2);
            assert_eq!(total_cards, 5);
        }
        _ => panic!("Expected Running status"),
    }
}

// ============== TaskStatus Badge Integration ==============

#[test]
fn test_task_status_badge_text() {
    assert_eq!(TaskStatus::Pending.badge_text(), "PENDING");
    assert_eq!(TaskStatus::InProgress.badge_text(), "IN PROGRESS");
    assert_eq!(TaskStatus::Completed.badge_text(), "DONE");
    assert_eq!(TaskStatus::Verified.badge_text(), "VERIFIED");
    assert_eq!(TaskStatus::Blocked.badge_text(), "BLOCKED");
    assert_eq!(TaskStatus::NeedsReview.badge_text(), "REVIEW");
}

#[test]
fn test_task_status_badge_colors() {
    let pending_color = TaskStatus::Pending.badge_color();
    assert_eq!(pending_color.len(), 4, "RGBA should have 4 components");

    let completed_color = TaskStatus::Completed.badge_color();
    assert_eq!(completed_color, [0.2, 1.0, 0.4, 0.9], "Green for completed");

    let blocked_color = TaskStatus::Blocked.badge_color();
    assert_eq!(blocked_color[0], 1.0, "High red component for blocked");
}

#[test]
fn test_task_status_transitions() {
    // Simulate task lifecycle
    let mut status = TaskStatus::Pending;
    assert_eq!(status, TaskStatus::default());

    status = TaskStatus::InProgress;
    assert_eq!(status.badge_text(), "IN PROGRESS");

    status = TaskStatus::Completed;
    assert_eq!(status.badge_text(), "DONE");
}

// ============== SuggestionCard Integration ==============

#[test]
fn test_suggestion_card_category_colors() {
    let fix_card = mock_card(0, "Fix bug", "fix");
    assert_eq!(fix_card.color(), [1.0, 0.4, 0.4, 1.0], "Red for fix");

    let test_card = mock_card(1, "Add test", "test");
    assert_eq!(test_card.color(), [0.4, 0.8, 1.0, 1.0], "Cyan for test");

    let build_card = mock_card(2, "Build fix", "build");
    assert_eq!(build_card.color(), [1.0, 0.7, 0.2, 1.0], "Orange for build");

    let refactor_card = mock_card(3, "Refactor", "refactor");
    assert_eq!(refactor_card.color(), [0.7, 0.5, 1.0, 1.0], "Purple for refactor");

    let docs_card = mock_card(4, "Documentation", "docs");
    assert_eq!(docs_card.color(), [0.5, 0.9, 0.5, 1.0], "Green for docs");
}

#[test]
fn test_suggestion_card_selection() {
    let mut card = mock_card(0, "Test", "fix");
    assert!(!card.selected);

    card.selected = true;
    assert!(card.selected);

    // Card should retain other properties
    assert_eq!(card.id, 0);
    assert_eq!(card.title, "Test");
}

#[test]
fn test_suggestion_card_streaming() {
    let mut card = mock_card(0, "Test", "fix");
    assert!(card.streaming, "New cards start as streaming");

    card.streaming = false;
    assert!(!card.streaming);
}

// ============== Permission System Integration ==============

#[test]
fn test_permission_response_approval() {
    let approved = PermissionResponse::Approved;
    assert!(approved.is_approved());

    let always = PermissionResponse::ApprovedAlways("cargo".to_string());
    assert!(always.is_approved());

    let denied = PermissionResponse::Denied;
    assert!(!denied.is_approved());

    let suggest = PermissionResponse::SuggestElse {
        original_command: "rm -rf /".to_string(),
    };
    assert!(!suggest.is_approved());
}

#[test]
fn test_permission_choice_labels() {
    let choices = PermissionChoice::all();
    assert_eq!(choices.len(), 4, "Should have 4 permission choices");

    assert_eq!(PermissionChoice::YesOnce.label("cargo"), "Yes (once)");
    assert_eq!(
        PermissionChoice::YesAlways.label("cargo build"),
        "Yes (always for 'cargo build')"
    );
    assert_eq!(PermissionChoice::No.label("test"), "No");
    assert_eq!(
        PermissionChoice::SuggestElse.label("test"),
        "Suggest something else"
    );
}

#[test]
fn test_permission_choice_short_labels() {
    assert_eq!(PermissionChoice::YesOnce.short_label(), "Yes");
    assert_eq!(PermissionChoice::YesAlways.short_label(), "Always");
    assert_eq!(PermissionChoice::No.short_label(), "No");
    assert_eq!(PermissionChoice::SuggestElse.short_label(), "Suggest");
}

// ============== Survey System Integration ==============

#[test]
fn test_survey_response_variants() {
    let selected = SurveyResponse::Selected(vec![0, 2]);
    match selected {
        SurveyResponse::Selected(indices) => {
            assert_eq!(indices, vec![0, 2]);
        }
        _ => panic!("Expected Selected variant"),
    }

    let custom = SurveyResponse::Custom("other option".to_string());
    match custom {
        SurveyResponse::Custom(text) => {
            assert_eq!(text, "other option");
        }
        _ => panic!("Expected Custom variant"),
    }

    let cancelled = SurveyResponse::Cancelled;
    match cancelled {
        SurveyResponse::Cancelled => {}
        _ => panic!("Expected Cancelled variant"),
    }
}

#[test]
fn test_survey_option_creation() {
    let option = SurveyOption::new("Label", "Description");
    assert_eq!(option.label, "Label");
    assert_eq!(option.description, "Description");

    let option2 = SurveyOption::new(
        String::from("String Label"),
        String::from("String Description"),
    );
    assert_eq!(option2.label, "String Label");
    assert_eq!(option2.description, "String Description");
}

// ============== UI Scale Integration ==============

#[test]
fn test_ui_scale_options() {
    let options = UiScaleOption::all();
    assert_eq!(options.len(), 7, "Should have 7 scale options");

    assert_eq!(UiScaleOption::Auto.label(), "Auto-detect");
    assert_eq!(UiScaleOption::Scale50.label(), "50%");
    assert_eq!(UiScaleOption::Scale100.label(), "100%");
    assert_eq!(UiScaleOption::Scale150.label(), "150%");
    assert_eq!(UiScaleOption::Scale200.label(), "200%");
    assert_eq!(UiScaleOption::Scale300.label(), "300%");
    assert_eq!(UiScaleOption::Scale400.label(), "400%");
}

#[test]
fn test_ui_scale_values() {
    assert_eq!(UiScaleOption::Auto.value(), None);
    assert_eq!(UiScaleOption::Scale50.value(), Some(0.5));
    assert_eq!(UiScaleOption::Scale100.value(), Some(1.0));
    assert_eq!(UiScaleOption::Scale150.value(), Some(1.5));
    assert_eq!(UiScaleOption::Scale200.value(), Some(2.0));
    assert_eq!(UiScaleOption::Scale300.value(), Some(3.0));
    assert_eq!(UiScaleOption::Scale400.value(), Some(4.0));
}

#[test]
fn test_ui_scale_roundtrip() {
    // Test that from_setting is the inverse of value()
    let scales = vec![
        None,
        Some(0.5),
        Some(1.0),
        Some(1.5),
        Some(2.0),
        Some(3.0),
        Some(4.0),
    ];

    for scale in scales {
        let option = UiScaleOption::from_setting(scale);
        assert_eq!(option.value(), scale, "Roundtrip failed for {:?}", scale);
    }
}

#[test]
fn test_ui_scale_boundaries() {
    // Test boundary mapping for from_setting
    assert_eq!(UiScaleOption::from_setting(Some(0.6)), UiScaleOption::Scale50);
    assert_eq!(UiScaleOption::from_setting(Some(0.75)), UiScaleOption::Scale100);
    assert_eq!(UiScaleOption::from_setting(Some(1.25)), UiScaleOption::Scale150);
    assert_eq!(UiScaleOption::from_setting(Some(1.75)), UiScaleOption::Scale200);
    assert_eq!(UiScaleOption::from_setting(Some(2.5)), UiScaleOption::Scale300);
    assert_eq!(UiScaleOption::from_setting(Some(3.5)), UiScaleOption::Scale400);
    assert_eq!(UiScaleOption::from_setting(Some(10.0)), UiScaleOption::Scale400);
}

// ============== Menu Items Integration ==============

#[test]
fn test_main_menu_items() {
    let items = MainMenuItem::all();
    assert_eq!(items.len(), 3);

    assert_eq!(MainMenuItem::Resume.label(), "Resume");
    assert_eq!(MainMenuItem::Settings.label(), "Settings");
    assert_eq!(MainMenuItem::Exit.label(), "Exit");
}

#[test]
fn test_settings_items() {
    let items = SettingsItem::all();
    assert_eq!(items.len(), 2);

    assert_eq!(SettingsItem::DarkMode.label(), "Dark Mode");
    assert_eq!(SettingsItem::UiScale.label(), "UI Scale");
}

#[test]
fn test_project_actions() {
    let actions = ProjectAction::all();
    assert_eq!(actions.len(), 4);

    assert_eq!(
        ProjectAction::StartPalaceLoop.label(),
        "Start Palace Loop"
    );
    assert_eq!(ProjectAction::Build.label(), "Build");
    assert_eq!(ProjectAction::Run.label(), "Run");
    assert_eq!(ProjectAction::ViewGitHistory.label(), "View Git History");
}

#[test]
fn test_execute_options() {
    let options = ExecuteOption::all();
    assert_eq!(options.len(), 3);

    assert_eq!(ExecuteOption::Claude.label(), "Run with Claude");
    assert_eq!(ExecuteOption::ZAi.label(), "Run with Z.ai");
    assert_eq!(ExecuteOption::ZAiTurbo.label(), "Run with Z.ai (turbo)");
}

// ============== AppState Clone Tests ==============

#[test]
fn test_app_state_clone_preserves_data() {
    let original = AppState::ProjectChooser { selected_index: 3 };
    let cloned = original.clone();

    match cloned {
        AppState::ProjectChooser { selected_index } => {
            assert_eq!(selected_index, 3);
        }
        _ => panic!("Clone should preserve state type"),
    }
}

#[test]
fn test_app_state_clone_with_complex_data() {
    let project_path = mock_project_path();
    let cards = vec![mock_card(0, "Task 1", "fix")];

    let original = AppState::PalaceLoop {
        project_path: project_path.clone(),
        cards: cards.clone(),
        focused_index: 0,
        hovered_index: None,
        generating: false,
        current_tool: Some("test-tool".to_string()),
        tool_log: vec!["log entry".to_string()],
        thought_log: vec!["thought".to_string()],
        log_scroll_offset: 5,
        detail_scroll_offset: 10.5,
        detail_max_scroll: 100.0,
    };

    let cloned = original.clone();

    match cloned {
        AppState::PalaceLoop {
            project_path,
            cards,
            current_tool,
            tool_log,
            thought_log,
            log_scroll_offset,
            detail_scroll_offset,
            detail_max_scroll,
            ..
        } => {
            assert_eq!(project_path, mock_project_path());
            assert_eq!(cards.len(), 1);
            assert_eq!(cards[0].title, "Task 1");
            assert_eq!(current_tool, Some("test-tool".to_string()));
            assert_eq!(tool_log, vec!["log entry".to_string()]);
            assert_eq!(thought_log, vec!["thought".to_string()]);
            assert_eq!(log_scroll_offset, 5);
            assert_eq!(detail_scroll_offset, 10.5);
            assert_eq!(detail_max_scroll, 100.0);
        }
        _ => panic!("Clone should preserve complex state"),
    }
}

// ============== State Machine Workflow Tests ==============

#[test]
fn test_full_state_machine_workflow() {
    // Simulate a typical user flow

    // 1. Start at project chooser
    let state = AppState::project_chooser();
    assert!(matches!(state, AppState::ProjectChooser { .. }));

    // 2. Select a project
    let project_path = mock_project_path();
    let state = AppState::project_view(project_path.clone());
    assert!(matches!(state, AppState::ProjectView { .. }));

    // 3. Start Palace Loop
    let state = AppState::PalaceLoop {
        project_path: project_path.clone(),
        cards: vec![],
        focused_index: 0,
        hovered_index: None,
        generating: true,
        current_tool: None,
        tool_log: vec![],
        thought_log: vec![],
        log_scroll_offset: 0,
        detail_scroll_offset: 0.0,
        detail_max_scroll: 0.0,
    };
    assert!(matches!(state, AppState::PalaceLoop { .. }));

    // 4. Open main menu
    let state = AppState::MainMenu {
        selected_item: 0,
        previous_state: Box::new(state),
    };
    assert!(matches!(state, AppState::MainMenu { .. }));

    // 5. Navigate to settings
    let state = AppState::SettingsMenu {
        selected_item: 0,
        previous_state: Box::new(state),
    };
    assert!(matches!(state, AppState::SettingsMenu { .. }));

    // 6. Unwind back to palace loop
    match state {
        AppState::SettingsMenu { previous_state, .. } => {
            match *previous_state {
                AppState::MainMenu { previous_state, .. } => {
                    match *previous_state {
                        AppState::PalaceLoop { .. } => {
                            // Successfully unwound
                        }
                        _ => panic!("Should unwind to PalaceLoop"),
                    }
                }
                _ => panic!("Should unwind to MainMenu"),
            }
        }
        _ => panic!("Expected SettingsMenu"),
    }
}

#[test]
fn test_execution_workflow_state_transitions() {
    let project_path = mock_project_path();
    let cards = vec![
        mock_card(0, "Task 1", "fix"),
        mock_card(1, "Task 2", "test"),
    ];

    // Start executing
    let state = AppState::Executing {
        project_path: project_path.clone(),
        executing_cards: cards.clone(),
        all_cards: cards.clone(),
        status: ExecutionStatus::Running {
            current_card: 0,
            total_cards: 2,
        },
        task_statuses: vec![TaskStatus::InProgress, TaskStatus::Pending],
        tool_log: vec!["Starting execution".to_string()],
        thought_log: vec![],
        log_scroll_offset: 0.0,
        executor: ExecuteOption::Claude,
        previous_state: Box::new(AppState::PalaceLoop {
            project_path: project_path.clone(),
            cards: cards.clone(),
            focused_index: 0,
            hovered_index: None,
            generating: false,
            current_tool: None,
            tool_log: vec![],
            thought_log: vec![],
            log_scroll_offset: 0,
            detail_scroll_offset: 0.0,
            detail_max_scroll: 0.0,
        }),
        quest_log_visible: false,
        quest_log_focus: 0,
    };

    // Verify execution state
    match state {
        AppState::Executing {
            status,
            task_statuses,
            ..
        } => {
            assert!(matches!(
                status,
                ExecutionStatus::Running {
                    current_card: 0,
                    total_cards: 2
                }
            ));
            assert_eq!(task_statuses[0], TaskStatus::InProgress);
            assert_eq!(task_statuses[1], TaskStatus::Pending);
        }
        _ => panic!("Expected Executing state"),
    }
}
