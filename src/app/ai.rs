//! AI suggestion generation and execution
//!
//! Handles running AI suggestions in background threads and communicating
//! results back to the main event loop.

use crate::state::PermissionResponse;
use super::AppEvent;
use std::path::PathBuf;
use std::sync::Arc;
use winit::event_loop::EventLoopProxy;

/// Run AI suggestion generation in a background thread
pub fn run_ai_suggestions(project_path: PathBuf, proxy: Arc<EventLoopProxy<AppEvent>>) {
    use crate::ai::{ProjectContext, SuggestionEngine, SuggestionEvent};

    tracing::info!("Starting AI suggestions for {:?}", project_path);

    // Gather project context
    let context = match ProjectContext::gather(&project_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to gather project context: {}", e);
            let _ = proxy.send_event(AppEvent::AiError(format!("Failed to gather context: {}", e)));
            return;
        }
    };

    // Create suggestion engine
    let engine = match SuggestionEngine::new(None, None) {
        Ok(e) => e,
        Err(e) => {
            tracing::error!("Failed to create suggestion engine: {}", e);
            let _ = proxy.send_event(AppEvent::AiError(format!("Failed to create engine: {}", e)));
            return;
        }
    };

    // Define callback that sends events to the main thread
    let callback = {
        let proxy = proxy.clone();
        move |event: SuggestionEvent| {
            let app_event = match event {
                SuggestionEvent::ToolCall(desc) => AppEvent::AiToolCall(desc),
                SuggestionEvent::Chatter(text) => AppEvent::AiChatter(text),
                SuggestionEvent::CardStart { id } => AppEvent::SuggestionStart { id },
                SuggestionEvent::CardUpdate { id, field, value } => {
                    AppEvent::SuggestionUpdate { id, field, value }
                }
                SuggestionEvent::CardComplete { id } => AppEvent::SuggestionComplete { id },
                SuggestionEvent::Done => AppEvent::SuggestionsDone,
                SuggestionEvent::Error(e) => AppEvent::AiError(e),
            };
            let _ = proxy.send_event(app_event);
        }
    };

    // Create permission requester that sends events to the GUI
    let permission_proxy = proxy.clone();
    let permission_requester: crate::ai::PermissionRequester = Box::new(move |command: &str| {
        // Skip empty commands - auto-approve (nothing to show user)
        if command.trim().is_empty() {
            tracing::debug!("Auto-approving empty permission request");
            return PermissionResponse::Approved;
        }

        // Create oneshot channel for response
        let (tx, rx) = tokio::sync::oneshot::channel::<PermissionResponse>();

        // Send permission request to GUI
        static PERMISSION_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = PERMISSION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if permission_proxy.send_event(AppEvent::PermissionRequest {
            id,
            command: command.to_string(),
            response_tx: tx,
        }).is_err() {
            return PermissionResponse::Denied; // Event loop closed
        }

        // Block waiting for user response
        match rx.blocking_recv() {
            Ok(response) => response,
            Err(_) => PermissionResponse::Denied, // Channel closed = deny
        }
    });

    // Run streaming suggestions
    if let Err(e) = engine.stream_to_gui(&context, callback, Some(permission_requester)) {
        tracing::error!("AI suggestion error: {}", e);
        let _ = proxy.send_event(AppEvent::AiError(e.to_string()));
    }

    // Signal completion
    let _ = proxy.send_event(AppEvent::SuggestionsDone);
}
