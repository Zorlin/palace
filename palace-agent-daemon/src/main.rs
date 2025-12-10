//! Palace Agent Daemon
//!
//! Shared rate limit daemon and agent registry for Palace swarms.
//! ALL Palace instances share this daemon for:
//! - Rate limit awareness across providers
//! - Agent registry (DAG of active agents)
//! - Real-time visibility via WebSocket
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                PALACE AGENT DAEMON                        │
//! │  ┌────────────────────────────────────────────────────┐  │
//! │  │  Rate Limiter (shared across all Palace instances) │  │
//! │  │  - Token buckets per provider/model                │  │
//! │  │  - Adaptive backoff with proportional response     │  │
//! │  └────────────────────────────────────────────────────┘  │
//! │  ┌────────────────────────────────────────────────────┐  │
//! │  │  Agent Registry                                     │  │
//! │  │  - DAG of all active agents                        │  │
//! │  │  - Parent/child relationships                       │  │
//! │  │  - Model assignments and spawn decisions           │  │
//! │  └────────────────────────────────────────────────────┘  │
//! │  ┌────────────────────────────────────────────────────┐  │
//! │  │  WebSocket API for WebUI                           │  │
//! │  │  - Real-time agent tree updates                    │  │
//! │  │  - Stream agent outputs                            │  │
//! │  │  - Announcements broadcast                         │  │
//! │  └────────────────────────────────────────────────────┘  │
//! └──────────────────────────────────────────────────────────┘
//! ```

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::broadcast;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

// ============================================================================
// Data Structures
// ============================================================================

/// Unique agent ID
pub type AgentId = String;

/// An agent in the swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: AgentId,
    pub model: String,
    pub parent: Option<AgentId>,
    pub children: Vec<AgentId>,
    pub task: String,
    pub status: AgentStatus,
    pub created_at: u64,
    pub last_activity: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Starting,
    Running,
    WaitingForChildren,
    Completed,
    Failed,
}

/// An announcement propagating through the swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Announcement {
    pub id: String,
    pub source: AgentId,
    pub direction: AnnounceDirection,
    pub hops_remaining: i32,
    pub emoji: Option<String>,
    pub message: String,
    pub channel: Option<String>,
    pub path: Vec<AgentId>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AnnounceDirection {
    Up,
    Down,
    Global,
    Both,
}

/// Rate limit state for a provider
#[derive(Debug)]
pub struct RateLimitState {
    pub provider: String,
    pub tokens_per_minute: u32,
    pub tokens_used: u32,
    pub last_reset: Instant,
    pub backoff_until: Option<Instant>,
    pub consecutive_429s: u32,
}

impl RateLimitState {
    fn new(provider: &str, tokens_per_minute: u32) -> Self {
        Self {
            provider: provider.to_string(),
            tokens_per_minute,
            tokens_used: 0,
            last_reset: Instant::now(),
            backoff_until: None,
            consecutive_429s: 0,
        }
    }

    fn can_proceed(&mut self, tokens_needed: u32) -> bool {
        // Check if we're in backoff
        if let Some(until) = self.backoff_until {
            if Instant::now() < until {
                return false;
            }
            self.backoff_until = None;
        }

        // Reset if minute has passed
        if self.last_reset.elapsed() >= Duration::from_secs(60) {
            self.tokens_used = 0;
            self.last_reset = Instant::now();
        }

        // Check capacity
        self.tokens_used + tokens_needed <= self.tokens_per_minute
    }

    fn consume(&mut self, tokens: u32) {
        self.tokens_used += tokens;
        self.consecutive_429s = 0; // Reset on success
    }

    fn handle_429(&mut self) {
        self.consecutive_429s += 1;

        // Proportional backoff - not exponential panic
        // First 429: wait 5 seconds
        // Second: wait 10 seconds
        // Third+: wait 15 seconds max
        let wait_secs = match self.consecutive_429s {
            1 => 5,
            2 => 10,
            _ => 15,
        };

        self.backoff_until = Some(Instant::now() + Duration::from_secs(wait_secs));

        // Reduce tokens_per_minute slightly (proportional, not panic)
        // Drop by 5% per 429, minimum 50% of original
        self.tokens_per_minute = (self.tokens_per_minute * 95 / 100).max(self.tokens_per_minute / 2);
    }
}

// ============================================================================
// Application State
// ============================================================================

pub struct AppState {
    /// All agents by ID
    agents: DashMap<AgentId, Agent>,

    /// Root agents (no parent)
    roots: RwLock<Vec<AgentId>>,

    /// Rate limits per provider
    rate_limits: DashMap<String, RwLock<RateLimitState>>,

    /// Broadcast channel for real-time updates
    updates_tx: broadcast::Sender<SwarmUpdate>,

    /// Announcement history (last 100)
    announcements: RwLock<Vec<Announcement>>,
}

impl AppState {
    fn new() -> Self {
        let (updates_tx, _) = broadcast::channel(1024);

        // Initialize default rate limits
        let rate_limits = DashMap::new();
        rate_limits.insert(
            "anthropic".to_string(),
            RwLock::new(RateLimitState::new("anthropic", 100000)),
        );
        rate_limits.insert(
            "mistral".to_string(),
            RwLock::new(RateLimitState::new("mistral", 500000)),
        );
        rate_limits.insert(
            "openai".to_string(),
            RwLock::new(RateLimitState::new("openai", 200000)),
        );

        Self {
            agents: DashMap::new(),
            roots: RwLock::new(Vec::new()),
            rate_limits,
            updates_tx,
            announcements: RwLock::new(Vec::new()),
        }
    }
}

/// Real-time updates sent via WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SwarmUpdate {
    AgentSpawned { agent: Agent },
    AgentStatusChanged { agent_id: AgentId, status: AgentStatus },
    AgentCompleted { agent_id: AgentId },
    Announcement { announcement: Announcement },
    RateLimitWarning { provider: String, usage_percent: u32 },
    FullState { agents: Vec<Agent>, announcements: Vec<Announcement> },
}

// ============================================================================
// API Handlers
// ============================================================================

/// Register a new agent
#[derive(Debug, Deserialize)]
struct SpawnRequest {
    model: String,
    task: String,
    parent: Option<AgentId>,
}

async fn spawn_agent(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SpawnRequest>,
) -> Json<Agent> {
    let id = Uuid::new_v4().to_string();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let agent = Agent {
        id: id.clone(),
        model: req.model,
        parent: req.parent.clone(),
        children: vec![],
        task: req.task,
        status: AgentStatus::Starting,
        created_at: now,
        last_activity: now,
    };

    // Update parent's children list
    if let Some(parent_id) = &req.parent {
        if let Some(mut parent) = state.agents.get_mut(parent_id) {
            parent.children.push(id.clone());
        }
    } else {
        // Root agent
        state.roots.write().push(id.clone());
    }

    state.agents.insert(id.clone(), agent.clone());

    // Broadcast update
    let _ = state.updates_tx.send(SwarmUpdate::AgentSpawned { agent: agent.clone() });

    Json(agent)
}

/// Update agent status
#[derive(Debug, Deserialize)]
struct StatusUpdate {
    agent_id: AgentId,
    status: AgentStatus,
}

async fn update_status(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StatusUpdate>,
) -> impl IntoResponse {
    if let Some(mut agent) = state.agents.get_mut(&req.agent_id) {
        agent.status = req.status;
        agent.last_activity = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let _ = state.updates_tx.send(SwarmUpdate::AgentStatusChanged {
            agent_id: req.agent_id,
            status: req.status,
        });
    }
    Json(serde_json::json!({"ok": true}))
}

/// Check rate limit before API call
#[derive(Debug, Deserialize)]
struct RateLimitCheck {
    provider: String,
    tokens_needed: u32,
}

#[derive(Debug, Serialize)]
struct RateLimitResponse {
    allowed: bool,
    wait_seconds: Option<u64>,
    usage_percent: u32,
}

async fn check_rate_limit(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RateLimitCheck>,
) -> Json<RateLimitResponse> {
    let entry = state.rate_limits.entry(req.provider.clone()).or_insert_with(|| {
        RwLock::new(RateLimitState::new(&req.provider, 100000))
    });

    let mut limit = entry.write();
    let allowed = limit.can_proceed(req.tokens_needed);

    let wait_seconds = if !allowed {
        limit.backoff_until.map(|until| {
            until.duration_since(Instant::now()).as_secs()
        })
    } else {
        None
    };

    let usage_percent = (limit.tokens_used * 100 / limit.tokens_per_minute).min(100);

    // Warn at 80%
    if usage_percent >= 80 {
        let _ = state.updates_tx.send(SwarmUpdate::RateLimitWarning {
            provider: req.provider,
            usage_percent,
        });
    }

    Json(RateLimitResponse {
        allowed,
        wait_seconds,
        usage_percent,
    })
}

/// Report rate limit consumption or 429
#[derive(Debug, Deserialize)]
struct RateLimitReport {
    provider: String,
    tokens_used: Option<u32>,
    got_429: bool,
}

async fn report_rate_limit(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RateLimitReport>,
) -> impl IntoResponse {
    if let Some(entry) = state.rate_limits.get(&req.provider) {
        let mut limit = entry.write();
        if req.got_429 {
            limit.handle_429();
            tracing::warn!("429 from {}, backoff applied", req.provider);
        } else if let Some(tokens) = req.tokens_used {
            limit.consume(tokens);
        }
    }
    Json(serde_json::json!({"ok": true}))
}

/// Broadcast an announcement
async fn broadcast_announcement(
    State(state): State<Arc<AppState>>,
    Json(mut announcement): Json<Announcement>,
) -> impl IntoResponse {
    announcement.id = Uuid::new_v4().to_string();
    announcement.timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Store in history
    {
        let mut history = state.announcements.write();
        history.push(announcement.clone());
        if history.len() > 100 {
            history.remove(0);
        }
    }

    // Broadcast to all WebSocket clients
    let _ = state.updates_tx.send(SwarmUpdate::Announcement {
        announcement: announcement.clone(),
    });

    Json(announcement)
}

/// Get full swarm state
async fn get_state(State(state): State<Arc<AppState>>) -> Json<SwarmUpdate> {
    let agents: Vec<Agent> = state.agents.iter().map(|r| r.value().clone()).collect();
    let announcements = state.announcements.read().clone();

    Json(SwarmUpdate::FullState {
        agents,
        announcements,
    })
}

/// WebSocket handler for real-time updates
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(mut socket: WebSocket, state: Arc<AppState>) {
    // Send initial state
    let agents: Vec<Agent> = state.agents.iter().map(|r| r.value().clone()).collect();
    let announcements = state.announcements.read().clone();
    let initial = SwarmUpdate::FullState { agents, announcements };

    if let Ok(json) = serde_json::to_string(&initial) {
        let _ = socket.send(Message::Text(json)).await;
    }

    // Subscribe to updates
    let mut rx = state.updates_tx.subscribe();

    loop {
        tokio::select! {
            Ok(update) = rx.recv() => {
                if let Ok(json) = serde_json::to_string(&update) {
                    if socket.send(Message::Text(json)).await.is_err() {
                        break;
                    }
                }
            }
            Some(msg) = socket.recv() => {
                match msg {
                    Ok(Message::Close(_)) => break,
                    Err(_) => break,
                    _ => {}
                }
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state = Arc::new(AppState::new());

    let app = Router::new()
        .route("/api/spawn", post(spawn_agent))
        .route("/api/status", post(update_status))
        .route("/api/rate-limit/check", post(check_rate_limit))
        .route("/api/rate-limit/report", post(report_rate_limit))
        .route("/api/announce", post(broadcast_announcement))
        .route("/api/state", get(get_state))
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = "127.0.0.1:19850";
    tracing::info!("Palace Agent Daemon running on http://{}", addr);
    tracing::info!("WebSocket available at ws://{}/ws", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_state() {
        let mut state = RateLimitState::new("test", 1000);

        // Should allow within limit
        assert!(state.can_proceed(500));
        state.consume(500);

        // Should allow more
        assert!(state.can_proceed(400));
        state.consume(400);

        // Should deny exceeding limit
        assert!(!state.can_proceed(200));
    }

    #[test]
    fn test_proportional_backoff() {
        let mut state = RateLimitState::new("test", 1000);

        // First 429: 5% reduction
        state.handle_429();
        assert_eq!(state.tokens_per_minute, 950);
        assert_eq!(state.consecutive_429s, 1);

        // Second 429: another 5%
        state.handle_429();
        assert_eq!(state.tokens_per_minute, 902); // 950 * 0.95
        assert_eq!(state.consecutive_429s, 2);
    }
}
