//! Debug commands and state serialization
//!
//! Handles:
//! - Debug command processing (screenshots, etc.)
//! - State serialization for hot reload / restart transfer

use crate::debug::{DebugCommand, DebugResponse};
use crate::state::AppState;
use std::path::PathBuf;

/// Debug command handling methods for App
pub trait DebugCommands {
    fn handle_debug_command(&mut self, command: DebugCommand) -> DebugResponse;
    fn get_state_json(&self) -> String;
}

impl DebugCommands for super::App {
    fn handle_debug_command(&mut self, command: DebugCommand) -> DebugResponse {
        match command {
            DebugCommand::Screenshot { path } => {
                if self.focused_renderer().is_none() {
                    return DebugResponse::Error {
                        message: "Renderer not initialized".to_string(),
                    };
                }

                // Generate default path if not specified
                let path = path.unwrap_or_else(|| {
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    PathBuf::from(format!("/tmp/palace-screenshot-{}.png", timestamp))
                });

                // Store the request - will be captured after next render
                self.pending_screenshot = Some(path.clone());
                DebugResponse::ScreenshotPending { path }
            }
        }
    }

    /// Get current state as JSON for restart transfer
    fn get_state_json(&self) -> String {
        match &self.state {
            AppState::ProjectChooser { selected_index, .. } => {
                serde_json::json!({
                    "view": "chooser",
                    "selected": selected_index
                })
            }
            AppState::ProjectView {
                project_path,
                selected_action,
            } => {
                serde_json::json!({
                    "view": "project",
                    "project_path": project_path.to_string_lossy(),
                    "selected": selected_action
                })
            }
            AppState::MainMenu {
                selected_item,
                previous_state,
            } => {
                // Serialize the main menu AND the previous state
                let prev = match previous_state.as_ref() {
                    AppState::ProjectChooser { selected_index, .. } => {
                        serde_json::json!({
                            "view": "chooser",
                            "selected": selected_index
                        })
                    }
                    AppState::ProjectView {
                        project_path,
                        selected_action,
                    } => {
                        serde_json::json!({
                            "view": "project",
                            "project_path": project_path.to_string_lossy(),
                            "selected": selected_action
                        })
                    }
                    _ => serde_json::json!({"view": "chooser", "selected": 0}),
                };
                serde_json::json!({
                    "view": "main_menu",
                    "selected": selected_item,
                    "previous": prev
                })
            }
            AppState::SettingsMenu {
                selected_item,
                previous_state,
            } => {
                // Serialize the settings menu AND the previous state
                let prev = match previous_state.as_ref() {
                    AppState::ProjectChooser { selected_index, .. } => {
                        serde_json::json!({
                            "view": "chooser",
                            "selected": selected_index
                        })
                    }
                    AppState::ProjectView {
                        project_path,
                        selected_action,
                    } => {
                        serde_json::json!({
                            "view": "project",
                            "project_path": project_path.to_string_lossy(),
                            "selected": selected_action
                        })
                    }
                    _ => serde_json::json!({"view": "chooser", "selected": 0}),
                };
                serde_json::json!({
                    "view": "settings",
                    "selected": selected_item,
                    "previous": prev
                })
            }
            AppState::PalaceLoop {
                project_path,
                focused_index,
                ..
            } => {
                serde_json::json!({
                    "view": "palace_loop",
                    "project_path": project_path.to_string_lossy(),
                    "focused": focused_index
                })
            }
            AppState::UiScaleMenu {
                selected_item,
                ..
            } => {
                // For UI scale menu, just return a simple state - we'll go back to settings on restart
                serde_json::json!({
                    "view": "ui_scale",
                    "selected": selected_item
                })
            }
            AppState::PermissionModal { .. } => {
                // Permission modal is transient - just return to chooser on restart
                serde_json::json!({
                    "view": "chooser",
                    "selected": 0
                })
            }
            AppState::ExecuteModal { .. } => {
                // Execute modal is transient - just return to chooser on restart
                serde_json::json!({
                    "view": "chooser",
                    "selected": 0
                })
            }
            AppState::Survey { .. } => {
                // Survey is transient - just return to chooser on restart
                serde_json::json!({
                    "view": "chooser",
                    "selected": 0
                })
            }
            AppState::Executing { project_path, .. } => {
                // Executing is transient - return to project view on restart
                serde_json::json!({
                    "view": "project",
                    "project_path": project_path.to_string_lossy(),
                    "selected": 0
                })
            }
            AppState::AddCardMenu { previous_state, .. } | AppState::CustomTaskInput { previous_state, .. } => {
                // Menus are transient - return to previous state on restart
                match previous_state.as_ref() {
                    AppState::PalaceLoop { project_path, .. } => {
                        serde_json::json!({
                            "view": "project",
                            "project_path": project_path.to_string_lossy(),
                            "selected": 0
                        })
                    }
                    _ => serde_json::json!({"view": "chooser", "selected": 0}),
                }
            }
            AppState::MultiDisplayDialog { previous_state, .. } => {
                // Dialog is transient - return to previous state on restart
                match previous_state.as_ref() {
                    AppState::ProjectChooser { selected_index, .. } => {
                        serde_json::json!({
                            "view": "chooser",
                            "selected": selected_index
                        })
                    }
                    AppState::ProjectView { project_path, selected_action } => {
                        serde_json::json!({
                            "view": "project",
                            "project_path": project_path.to_string_lossy(),
                            "selected": selected_action
                        })
                    }
                    _ => serde_json::json!({"view": "chooser", "selected": 0}),
                }
            }
            AppState::ProjectContextMenu { .. } | AppState::LanguageSelector { .. } => {
                // Menus are transient - return to project chooser on restart
                serde_json::json!({"view": "chooser", "selected": 0})
            }
            AppState::NewMonitorDialog { .. } => {
                // New monitor dialog is transient - return to project chooser on restart
                serde_json::json!({"view": "chooser", "selected": 0})
            }
        }
        .to_string()
    }
}
