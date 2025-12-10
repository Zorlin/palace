//! Palace Daemon
//!
//! High-performance daemon combining:
//! - API Translator (Anthropic → OpenAI format)
//! - Context Cache (swarm consciousness)
//! - Agent Coordination (spawn/announce)
//!
//! HTTP API on port 19848:
//! - POST /v1/messages - Anthropic-compatible messages API (translated to backend)
//! - GET /health - Health check
//! - POST /context/delta - Apply context delta
//! - GET /context/active - Get active context
//!
//! Unix socket /tmp/palace-context-cache.sock for low-latency context ops

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use palace_daemon::translator::{handle_messages, TranslatorState};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::{fmt, EnvFilter};

const DEFAULT_PORT: u16 = 19848;
const DEFAULT_BACKEND_URL: &str = "https://api.mistral.ai/v1";
const DEFAULT_BACKEND_MODEL: &str = "devstral-2512";

#[derive(Clone)]
struct AppState {
    translator: Arc<TranslatorState>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with env filter
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();

    info!("🚀 Palace Daemon starting...");

    // Load config from environment
    let _ = dotenvy::from_filename(
        dirs::home_dir()
            .unwrap_or_default()
            .join(".palace/credentials.env")
    );

    let port: u16 = std::env::var("PALACE_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let backend_url = std::env::var("PALACE_BACKEND_URL")
        .unwrap_or_else(|_| DEFAULT_BACKEND_URL.to_string());

    let backend_api_key = std::env::var("MISTRAL_API_KEY")
        .unwrap_or_default();

    let backend_model = std::env::var("PALACE_BACKEND_MODEL")
        .unwrap_or_else(|_| DEFAULT_BACKEND_MODEL.to_string());

    info!("Backend: {} (model: {})", backend_url, backend_model);

    // Create translator state
    let translator = Arc::new(TranslatorState::new(
        backend_url,
        backend_api_key,
        backend_model,
    ));

    let state = AppState { translator };

    // Build router
    let app = Router::new()
        // Anthropic-compatible messages endpoint
        .route("/v1/messages", post(handle_messages_route))
        // Health check
        .route("/health", get(health_check))
        // Context endpoints (future)
        .route("/context/delta", post(context_delta))
        .route("/context/active", get(context_active))
        // CORS for browser access
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .with_state(state);

    // Bind and serve
    let addr = format!("127.0.0.1:{}", port);
    let listener = TcpListener::bind(&addr).await?;
    info!("🔄 Listening on http://{}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn handle_messages_route(
    State(state): State<AppState>,
    Json(req): Json<Value>,
) -> axum::response::Response {
    // Parse request
    let req: palace_daemon::translator::AnthropicRequest = match serde_json::from_value(req) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": e.to_string()}
                })),
            )
                .into_response();
        }
    };

    handle_messages(State(state.translator), Json(req)).await
}

async fn health_check() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "service": "palace-daemon",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

async fn context_delta(Json(payload): Json<Value>) -> Json<Value> {
    // TODO: Wire to context cache
    let delta = payload["delta"].as_str().unwrap_or("");
    info!("Context delta: {}", delta);
    Json(json!({"status": "ok", "applied": delta}))
}

async fn context_active() -> Json<Value> {
    // TODO: Wire to context cache
    Json(json!({"active_blocks": [], "context": ""}))
}
