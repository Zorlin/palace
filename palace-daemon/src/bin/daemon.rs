//! Palace Daemon
//!
//! High-performance daemon combining:
//! - API Translator (Anthropic → OpenAI format)
//! - Context Cache (swarm consciousness)
//! - Agent Coordination (spawn/announce)
//!
//! ## Model Routing
//!
//! By default, ALL models map to devstral-2512 (free):
//! - opus, sonnet, haiku → devstral-2512
//! - devstral, mistral, codestral → devstral-2512
//!
//! Run with `--real` flag to use actual Claude API for opus/sonnet/haiku:
//! ```bash
//! palace-daemon --real  # Now opus/sonnet/haiku cost real money
//! ```
//!
//! This lets you test swarm architectures for FREE, then flip to real Claude when needed.
//!
//! ## HTTP API (default port 19848)
//!
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
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::{fmt, EnvFilter};

const DEFAULT_PORT: u16 = 19848;
const DEFAULT_BACKEND_URL: &str = "https://api.mistral.ai/v1";
const DEFAULT_BACKEND_MODEL: &str = "devstral-2512";

/// Get runtime directory for PID/config files (same as Python's palace.py)
fn get_runtime_dir() -> PathBuf {
    std::env::var("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join(".palace/run")
        })
        .join("palace")
}

/// Write PID and config files so `pal` can detect us
fn write_runtime_files(port: u16, backend_url: &str, backend_model: &str) -> std::io::Result<()> {
    let runtime_dir = get_runtime_dir();
    std::fs::create_dir_all(&runtime_dir)?;

    // Write PID file
    let pid_file = runtime_dir.join("translator.pid");
    std::fs::write(&pid_file, std::process::id().to_string())?;

    // Write config file (matches Python's format)
    let config_file = runtime_dir.join("translator.json");
    let config = json!({
        "port": port,
        "backend_url": backend_url,
        "backend_model": backend_model,
        "started_at": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    });
    std::fs::write(&config_file, serde_json::to_string(&config)?)?;

    info!("📝 Runtime files: {}", runtime_dir.display());
    Ok(())
}

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
