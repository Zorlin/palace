//! Palace Translator Daemon
//!
//! HTTP server that proxies Anthropic API requests to OpenAI-compatible backends.
//! Features:
//! - Rate limiting with adaptive backoff
//! - Request buffering: absorbs 429s and queues requests instead of failing
//! - Silent retry: never returns 429 to clients, just delays responses
//! - System reminder injection for delayed responses
//! - Model packs: @switch command to swap between GLM/Mistral/Anthropic backends
//! - Session-scoped model switching: each session tracks its own active pack

use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::StreamExt;
use palace_core::{anthropic, openai, Provider, RateLimiter, RateLimitResult};
use palace_translator::convert;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, env, path::PathBuf, sync::Arc, time::{Duration, Instant}};
use tokio::sync::{RwLock, Semaphore};
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};

/// A model pack maps Claude tiers (opus/sonnet/haiku) to actual backend models
#[derive(Debug, Clone)]
struct ModelPack {
    name: &'static str,
    description: &'static str,
    opus: &'static str,
    sonnet: &'static str,
    haiku: &'static str,
}

impl ModelPack {
    /// Get all available model packs
    fn all_packs() -> HashMap<&'static str, ModelPack> {
        let mut packs = HashMap::new();

        packs.insert("glm", ModelPack {
            name: "glm",
            description: "GLM models via z.ai (best for coding)",
            opus: "glm-4.6v",
            sonnet: "glm-4.6",
            haiku: "glm-4.5-air",
        });

        packs.insert("mistral", ModelPack {
            name: "mistral",
            description: "Mistral Devstral 2 models (fast coding)",
            opus: "devstral-2512",
            sonnet: "devstral-2512",
            haiku: "devstral-mini-2",
        });

        // Aliases
        packs.insert("devstral", packs["mistral"].clone());
        packs.insert("liefstral", packs["mistral"].clone());

        packs.insert("anthropic", ModelPack {
            name: "anthropic",
            description: "Real Claude models (passthrough to Anthropic)",
            opus: "claude-opus-4-5-20251101",
            sonnet: "claude-sonnet-4-5",
            haiku: "claude-haiku-4-5",
        });

        // Alias
        packs.insert("claude", packs["anthropic"].clone());

        packs
    }

    /// Map a requested model to the pack's equivalent
    /// IMPORTANT: Specific models (devstral, glm, etc.) pass through directly - NO FALLBACK TO EXPENSIVE MODELS
    fn map_model<'a>(&self, requested: &'a str) -> &'a str
    where
        'static: 'a,
    {
        // First: check if it's a specific model that should pass through directly
        // This prevents accidental spending when free/cheap models are explicitly requested
        let lower = requested.to_lowercase();
        if lower.starts_with("devstral") || lower.starts_with("mistral") {
            return requested; // Pass through to Mistral API
        }
        if lower.starts_with("glm") {
            return requested; // Pass through to GLM API
        }
        if lower.starts_with("qwen") {
            return requested; // Pass through to Qwen API
        }
        if lower.contains("gpt-oss") || lower.contains("local") {
            return requested; // Pass through to local
        }

        // Second: check if it's a tier request (from Claude Code defaults)
        if requested.contains("opus") {
            self.opus
        } else if requested.contains("sonnet") {
            self.sonnet
        } else if requested.contains("haiku") {
            self.haiku
        } else {
            // Unknown - default to sonnet tier (but only for tier-like requests)
            self.sonnet
        }
    }
}

/// Maximum concurrent requests per provider when rate limited
const MAX_QUEUED_PER_PROVIDER: usize = 100;

/// Maximum retry attempts for 429s before giving up
const MAX_RETRY_ATTEMPTS: u32 = 10;

/// Maximum output tokens allowed (prevents runaway generation)
/// Can be overridden with PALACE_MAX_OUTPUT_TOKENS env var
/// Default matches Claude Code's CLAUDE_CODE_MAX_OUTPUT_TOKENS default
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 32000;

/// Project build and test status
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectStatus {
    /// Whether the last build passed
    build_passing: bool,
    /// Build error messages (empty if passing)
    build_errors: Vec<String>,
    /// Whether the last test run passed
    test_passing: bool,
    /// Test failure messages (empty if passing)
    test_failures: Vec<String>,
    /// Last time build status was checked
    last_build_check: Option<std::time::SystemTime>,
    /// Last time test status was checked
    last_test_check: Option<std::time::SystemTime>,
}

impl Default for ProjectStatus {
    fn default() -> Self {
        Self {
            build_passing: false,
            build_errors: vec!["Build status not yet checked".to_string()],
            test_passing: false,
            test_failures: vec!["Test status not yet checked".to_string()],
            last_build_check: None,
            last_test_check: None,
        }
    }
}

/// Strict mode - when on, models cannot report success until ALL tests pass
#[derive(Debug, Clone, PartialEq)]
enum StrictMode {
    /// Strict mode off - models can complete without passing tests
    Off,
    /// Strict mode on - all builds AND tests must pass before completion
    On,
}

impl Default for StrictMode {
    fn default() -> Self {
        StrictMode::Off
    }
}

/// Swarm mode - enables parallel worker spawning in hypermiler mode
#[derive(Debug, Clone, PartialEq)]
enum SwarmMode {
    /// Swarm mode off - normal single-model operation
    Off,
    /// Swarm mode on - spawn parallel Claude Code workers (default cap of 20)
    On,
    /// Swarm mode with safety cap - spawn up to N parallel workers
    TaskLimit(u32),
    /// Unlimited mode - no cap, planner decides everything
    Unlimited,
}

impl Default for SwarmMode {
    fn default() -> Self {
        SwarmMode::Off
    }
}

/// Configuration for swarm orchestration
#[derive(Debug, Clone)]
struct SwarmConfig {
    /// Which model generates task list (None = use hypermiler orchestrator)
    planner_model: Option<String>,
    /// Swarm mode state
    swarm_mode: SwarmMode,
    /// Safety cap on parallel workers (default 20, but planner decides actual count)
    max_workers: u32,
    /// Provider for workers (anthropic, z.ai, local)
    worker_provider: String,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            planner_model: None,
            swarm_mode: SwarmMode::Off,
            max_workers: 20,  // Safety cap - planner model decides actual task count
            worker_provider: "anthropic".to_string(),
        }
    }
}

/// Per-conversation state - isolated settings for each Claude Code session
#[derive(Debug, Clone)]
struct ConversationState {
    /// Conversation ID (from header or generated)
    id: String,
    /// Currently active model pack for this conversation
    active_pack: ModelPack,
    /// Continuous mode state
    continuous_mode: ContinuousMode,
    /// When continuous mode started (for time limit tracking)
    continuous_start: Option<Instant>,
    /// Cost tracking for this conversation
    cost_tracker: CostTracker,
    /// Current execution strategy
    strategy: Strategy,
    /// Strict mode - blocks completion until all tests pass
    strict_mode: StrictMode,
    /// Number of turns remaining for multi-model reminder
    reminder_turns_remaining: u32,
    /// Last activity time (for cleanup of stale conversations)
    last_activity: Instant,
    /// Project path for this conversation (from X-Project-Path header)
    project_path: Option<String>,
    /// Last verification result
    last_verification: Option<VerificationStatus>,
    /// Swarm configuration for parallel worker spawning
    swarm_config: SwarmConfig,
    /// User to run workers as (from X-User header)
    run_as_user: Option<String>,
    /// Pending actions from the planner awaiting @action selection
    pending_actions: Vec<PendingAction>,
    /// Tool ID mapper for consistent round-trip between Claude and backend
    tool_id_mapper: convert::ToolIdMapper,
}

/// Verification status from running build/tests
#[derive(Debug, Clone)]
struct VerificationStatus {
    build_passing: bool,
    test_passing: bool,
    build_errors: Vec<String>,
    test_failures: Vec<String>,
    verified_at: Instant,
}

/// A pending action from the planner awaiting user selection
#[derive(Debug, Clone)]
struct PendingAction {
    num: usize,
    label: String,
    description: String,
}

impl ConversationState {
    fn new(id: String, default_pack: ModelPack) -> Self {
        Self {
            id,
            active_pack: default_pack,
            continuous_mode: ContinuousMode::Off,
            continuous_start: None,
            cost_tracker: CostTracker::default(),
            strategy: Strategy::Simple,
            strict_mode: StrictMode::Off,
            reminder_turns_remaining: 0,
            last_activity: Instant::now(),
            project_path: None,
            last_verification: None,
            swarm_config: SwarmConfig::default(),
            run_as_user: None,
            pending_actions: Vec::new(),
            tool_id_mapper: convert::ToolIdMapper::new(),
        }
    }

    /// Touch the conversation to update last activity time
    fn touch(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Check if conversation is stale (no activity for given duration)
    fn is_stale(&self, max_idle: std::time::Duration) -> bool {
        self.last_activity.elapsed() > max_idle
    }
}

/// Duration after which conversation state is cleaned up
const CONVERSATION_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3600); // 1 hour

/// Model registry mapping model names to backend configurations
#[derive(Debug, Clone)]
struct ModelConfig {
    backend_url: String,
    api_key_env: String,
    model_id: String,
    /// If true, backend speaks Anthropic format - skip translation, just proxy with rate limiting
    passthrough: bool,
}

/// Per-provider queue state
struct ProviderQueue {
    /// Semaphore to limit concurrent requests when rate limited
    semaphore: Semaphore,
    /// Current queue depth
    queued: std::sync::atomic::AtomicUsize,
    /// Whether we're currently in backoff mode
    in_backoff: std::sync::atomic::AtomicBool,
    /// Time when backoff started
    backoff_until: RwLock<Option<Instant>>,
}

impl ProviderQueue {
    fn new() -> Self {
        Self {
            semaphore: Semaphore::new(MAX_QUEUED_PER_PROVIDER),
            queued: std::sync::atomic::AtomicUsize::new(0),
            in_backoff: std::sync::atomic::AtomicBool::new(false),
            backoff_until: RwLock::new(None),
        }
    }

    fn queue_depth(&self) -> usize {
        self.queued.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn is_in_backoff(&self) -> bool {
        self.in_backoff.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Path to persist active pack across restarts
const STATE_FILE: &str = "/var/lib/palace-translator/active_pack";

/// Load persisted pack from state file
fn load_persisted_pack(path: &PathBuf, packs: &HashMap<&'static str, ModelPack>) -> Option<ModelPack> {
    let pack_name = std::fs::read_to_string(path).ok()?;
    let pack_name = pack_name.trim();
    let pack = packs.get(pack_name)?;
    info!("Loaded persisted pack: {}", pack_name);
    Some(pack.clone())
}

/// Continuous mode verification tier
#[derive(Debug, Clone, Copy, PartialEq)]
enum VerificationTier {
    /// High confidence (95+): use 3b model for quick check
    Quick,
    /// Medium confidence (70-94): use 8b model for thorough check
    Medium,
    /// Low confidence or disaster (<70): use 24b model for deep analysis
    Deep,
}

/// Continuous mode configuration
#[derive(Debug, Clone)]
struct ContinuousConfig {
    /// Ollama endpoint URL
    ollama_url: String,
    /// Model for quick verification (3b tier)
    quick_model: String,
    /// Model for medium verification (8b tier)
    medium_model: String,
    /// Model for deep verification (24b tier)
    deep_model: String,
}

impl Default for ContinuousConfig {
    fn default() -> Self {
        Self {
            ollama_url: env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://10.7.1.135:11434".to_string()),
            quick_model: env::var("OLLAMA_QUICK_MODEL")
                .unwrap_or_else(|_| "ministral-3:3b".to_string()),
            medium_model: env::var("OLLAMA_MEDIUM_MODEL")
                .unwrap_or_else(|_| "ministral-3:8b".to_string()),
            deep_model: env::var("OLLAMA_DEEP_MODEL")
                .unwrap_or_else(|_| "hf.co/bartowski/mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF:latest".to_string()),
        }
    }
}

/// Model pricing per 1M tokens (input, output) in USD
#[derive(Debug, Clone)]
struct ModelPricing {
    input_per_million: f64,
    output_per_million: f64,
}

impl ModelPricing {
    fn calculate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * self.input_per_million;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.output_per_million;
        input_cost + output_cost
    }

    fn get_pricing(model_id: &str) -> Self {
        let model_lower = model_id.to_lowercase();

        // Claude models (actual Anthropic pricing as of Dec 2025)
        if model_lower.contains("opus") {
            // Claude Opus 4.5: $5/Mtok in, $25/Mtok out
            ModelPricing { input_per_million: 5.0, output_per_million: 25.0 }
        } else if model_lower.contains("sonnet") {
            // Claude Sonnet 4.5: $3/Mtok in, $15/Mtok out (≤200K context)
            ModelPricing { input_per_million: 3.0, output_per_million: 15.0 }
        } else if model_lower.contains("haiku") {
            // Claude Haiku 4.5: $1/Mtok in, $5/Mtok out
            ModelPricing { input_per_million: 1.0, output_per_million: 5.0 }
        }
        // OpenRouter premium models
        else if model_lower.contains("gpt-5") || model_lower.contains("codex-max") {
            // OpenAI GPT-5.1 Codex Max: $1.25/Mtok in, $10/Mtok out
            ModelPricing { input_per_million: 1.25, output_per_million: 10.0 }
        } else if model_lower.contains("gemini-3") || model_lower.contains("gemini-pro") {
            // Google Gemini 3 Pro: $2/Mtok in, $12/Mtok out
            ModelPricing { input_per_million: 2.0, output_per_million: 12.0 }
        }
        // Free/flatrate models
        else if model_lower.contains("devstral") || model_lower.contains("mistral") {
            // Devstral 2: Currently free during preview
            ModelPricing { input_per_million: 0.0, output_per_million: 0.0 }
        } else if model_lower.contains("glm") || model_lower.contains("zhipu") {
            // GLM models via Z.ai: flatrate subscription
            ModelPricing { input_per_million: 0.0, output_per_million: 0.0 }
        }
        // Local models are always free
        else {
            ModelPricing { input_per_million: 0.0, output_per_million: 0.0 }
        }
    }
}

/// Model cost tier for strategy-based selection
#[derive(Debug, Clone, PartialEq)]
enum ModelTier {
    /// Premium models: Opus ($25/Mtok out), GPT-5.1 Codex Max ($10), Gemini 3 Pro ($12)
    Premium,
    /// Standard models: Sonnet ($15/Mtok out)
    Standard,
    /// Cheap models: Haiku ($5/Mtok out)
    Cheap,
    /// Free/flatrate models: GLM (Z.ai flatrate), Devstral 2 (free preview)
    Free,
    /// Local models: Devstral Small 2 (24B), gpt-oss-20b, Ollama models
    Local,
}

impl ModelTier {
    /// Classify a model by its pricing tier
    fn from_model_name(model: &str) -> Self {
        let model_lower = model.to_lowercase();

        // Local models (Ollama, llama.cpp, local endpoints)
        // Check this FIRST because some local models have names that match other patterns
        if model_lower.contains("qwen")
            || model_lower.contains("llama")
            || model_lower.contains("codellama")
            || model_lower.contains("deepseek")
            || model_lower.contains("starcoder")
            || model_lower.contains("gpt-oss")
            || model_lower.starts_with("local:")
            // Ollama models use format "model:tag" (e.g., "devstral:latest")
            || (model_lower.contains(":") && !model_lower.contains("claude") && !model_lower.contains("gpt-5"))
            // Devstral Small 2 specifically is for local use
            || model_lower.contains("devstral-small")
            || model_lower.contains("labs-devstral-small") {
            return ModelTier::Local;
        }

        // Premium tier (>$10/Mtok output)
        if model_lower.contains("opus")
            // OpenRouter premium models
            || model_lower.contains("gpt-5") || model_lower.contains("codex-max")
            || model_lower.contains("gemini-3") || model_lower.contains("gemini-pro") {
            return ModelTier::Premium;
        }

        // Free/flatrate models (GLM via Z.ai, Devstral 2 free preview)
        if model_lower.contains("glm")
            || model_lower.contains("zhipu")
            // Devstral 2 (123B) is free during preview period - NOT devstral-small
            || (model_lower.contains("devstral") && !model_lower.contains("small")) {
            return ModelTier::Free;
        }

        // Cheap models (<$5/Mtok output)
        if model_lower.contains("haiku")
            || model_lower.contains("flash") {
            return ModelTier::Cheap;
        }

        // Standard tier (Sonnet, and unknown models default here)
        ModelTier::Standard
    }

    /// Check if model is at or below the given tier
    fn is_at_or_below(&self, max_tier: &ModelTier) -> bool {
        let self_rank = self.rank();
        let max_rank = max_tier.rank();
        self_rank <= max_rank
    }

    /// Rank for comparison (lower = cheaper)
    fn rank(&self) -> u8 {
        match self {
            ModelTier::Local => 0,
            ModelTier::Free => 1,
            ModelTier::Cheap => 2,
            ModelTier::Standard => 3,
            ModelTier::Premium => 4,
        }
    }

    /// Get cost per million output tokens (approximate)
    fn approx_output_cost_per_mtok(&self) -> f64 {
        match self {
            ModelTier::Local => 0.0,      // Local inference is free
            ModelTier::Free => 0.0,       // GLM (flatrate), Devstral 2 (free preview)
            ModelTier::Cheap => 5.0,      // Haiku ($5/Mtok out)
            ModelTier::Standard => 15.0,  // Sonnet ($15/Mtok out)
            ModelTier::Premium => 25.0,   // Opus ($25/Mtok out) - representative
        }
    }
}

/// Cost tracking state
#[derive(Debug, Clone)]
struct CostTracker {
    /// Total session cost in USD
    session_cost: f64,
    /// Last request cost in USD
    request_cost: f64,
    /// Whether cost tracking is enabled
    enabled: bool,
}

impl Default for CostTracker {
    fn default() -> Self {
        Self {
            session_cost: 0.0,
            request_cost: 0.0,
            enabled: true, // Default to enabled (@costs=on)
        }
    }
}

impl CostTracker {
    fn update(&mut self, model_id: &str, input_tokens: u32, output_tokens: u32) {
        if !self.enabled {
            return;
        }

        let pricing = ModelPricing::get_pricing(model_id);
        self.request_cost = pricing.calculate_cost(input_tokens, output_tokens);
        self.session_cost += self.request_cost;
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Continuous mode state
#[derive(Debug, Clone)]
enum ContinuousMode {
    /// Continuous mode disabled
    Off,
    /// Unlimited continuous mode (no time limit)
    Unlimited,
    /// Time-limited continuous mode (duration limit)
    TimeLimited(Duration),
}

/// Smart orchestrator type
#[derive(Debug, Clone, PartialEq)]
enum SmartOrchestrator {
    ClaudeCode,
    Codex,
}

/// Worker tier for smart mode execution
#[derive(Debug, Clone, PartialEq)]
enum WorkerTier {
    Premium,
    Standard,
    Cheap,
    Free,
    Local,
}

/// Smart mode configuration
#[derive(Debug, Clone, PartialEq)]
struct SmartConfig {
    /// Orchestrator to use (Claude Code or Codex)
    orchestrator: Option<SmartOrchestrator>,
    /// Worker tier for execution
    worker_tier: WorkerTier,
    /// Estimated rate limit per window
    rate_limit: u32,
    /// Rate limit window in minutes
    rate_window_minutes: u32,
    /// Threshold percentage to trigger warnings
    rate_threshold_percent: u8,
}

impl Default for SmartConfig {
    fn default() -> Self {
        Self {
            orchestrator: None, // Auto-detect on first use
            worker_tier: WorkerTier::Standard,
            rate_limit: 100, // Conservative estimate for Claude Code
            rate_window_minutes: 60,
            rate_threshold_percent: 80,
        }
    }
}

/// Structured plan from smart orchestration
#[derive(Debug, Clone)]
struct SmartOrchestrationPlan {
    /// Which orchestrator generated this plan
    orchestrator: SmartOrchestrator,
    /// Raw plan text from orchestrator
    raw_plan: String,
    /// Worker tier for execution
    worker_tier: WorkerTier,
}

/// Configuration for hypermiler strategy
#[derive(Debug, Clone, PartialEq)]
struct HypermilerConfig {
    /// Which tier of workers to delegate to
    worker_tier: WorkerTier,
    /// Explicit orchestrator model (None = auto-select best local)
    orchestrator_model: Option<String>,
}

impl Default for HypermilerConfig {
    fn default() -> Self {
        Self {
            worker_tier: WorkerTier::Premium,
            orchestrator_model: None,  // Auto-select
        }
    }
}

/// Rate limit tracking for smart mode
#[derive(Debug, Clone)]
struct RateLimitTracker {
    /// Requests made in current window
    requests_this_window: u32,
    /// Estimated requests per window
    estimated_limit: u32,
    /// Window start time
    window_start: std::time::Instant,
    /// Window duration
    window_duration: std::time::Duration,
}

impl RateLimitTracker {
    fn new(estimated_limit: u32, window_minutes: u32) -> Self {
        Self {
            requests_this_window: 0,
            estimated_limit,
            window_start: std::time::Instant::now(),
            window_duration: std::time::Duration::from_secs(window_minutes as u64 * 60),
        }
    }

    fn record_request(&mut self) {
        // Reset window if expired
        if self.window_start.elapsed() > self.window_duration {
            self.window_start = std::time::Instant::now();
            self.requests_this_window = 0;
        }
        self.requests_this_window += 1;
    }

    /// Get current usage as percentage of estimated limit
    fn usage_percent(&self) -> u8 {
        if self.estimated_limit == 0 {
            return 0;
        }
        ((self.requests_this_window as f32 / self.estimated_limit as f32) * 100.0) as u8
    }

    /// Check if we're at or above the threshold
    fn at_threshold(&self, threshold_percent: u8) -> bool {
        self.usage_percent() >= threshold_percent
    }
}

/// Error type for orchestration
#[derive(Debug)]
enum OrchestrationError {
    ProcessSpawn(std::io::Error),
    ProcessWait(std::io::Error),
    OutputParse(String),
    Timeout,
    RateLimited,
}

impl std::fmt::Display for OrchestrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrchestrationError::ProcessSpawn(e) => write!(f, "Failed to spawn process: {}", e),
            OrchestrationError::ProcessWait(e) => write!(f, "Failed to wait for process: {}", e),
            OrchestrationError::OutputParse(e) => write!(f, "Failed to parse output: {}", e),
            OrchestrationError::Timeout => write!(f, "Orchestration timed out"),
            OrchestrationError::RateLimited => write!(f, "Rate limited"),
        }
    }
}

/// Available local models (detected from Ollama)
#[derive(Debug, Clone, Default)]
struct LocalModels {
    /// Models available via Ollama
    available: Vec<String>,
    /// Preferred model for orchestration (user configurable)
    preferred_orchestrator: Option<String>,
    /// Preferred model for workers (user configurable)
    preferred_worker: Option<String>,
}

/// Detect available local models from Ollama
async fn detect_local_models(ollama_url: &str) -> LocalModels {
    let client = reqwest::Client::new();
    let url = format!("{}/api/tags", ollama_url.trim_end_matches('/'));

    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(models) = json["models"].as_array() {
                    let available: Vec<String> = models
                        .iter()
                        .filter_map(|m| m["name"].as_str().map(String::from))
                        .collect();

                    info!("LOCAL: Detected {} models from Ollama: {:?}", available.len(), available);

                    // Pick defaults based on what's available
                    let preferred_orchestrator = available.iter()
                        .find(|m| m.contains("devstral-small") || m.contains("devstral:"))
                        .or_else(|| available.iter().find(|m| m.contains("qwen")))
                        .cloned();

                    let preferred_worker = available.iter()
                        .find(|m| m.contains("devstral-small") || m.contains("devstral:"))
                        .or_else(|| available.iter().find(|m| m.contains("gpt-oss")))
                        .or_else(|| available.first())
                        .cloned();

                    return LocalModels {
                        available,
                        preferred_orchestrator,
                        preferred_worker,
                    };
                }
            }
        }
        Ok(resp) => {
            warn!("LOCAL: Ollama returned status {}", resp.status());
        }
        Err(e) => {
            warn!("LOCAL: Failed to connect to Ollama at {}: {}", ollama_url, e);
        }
    }

    LocalModels::default()
}

/// Select best available local model for the given role
fn select_local_model(local_models: &LocalModels, role: &str) -> Option<String> {
    match role {
        "orchestrator" => local_models.preferred_orchestrator.clone()
            .or_else(|| local_models.available.first().cloned()),
        "worker" => local_models.preferred_worker.clone()
            .or_else(|| local_models.available.first().cloned()),
        _ => local_models.available.first().cloned(),
    }
}

/// Execution strategy for continuous mode
#[derive(Debug, Clone, PartialEq)]
enum Strategy {
    /// Run until credit exhaustion, verify locally, stop
    Simple,
    /// Smart orchestration with premium models (Claude/Codex CLI)
    Smart(SmartConfig),
    /// Use premium models without cost optimization
    Premium,
    /// Use models under $4/Mtok, prefer flatrate
    Cheap,
    /// Only free/flatrate models
    Free,
    /// Local models only (with hardware detection)
    Local,
    /// Alias for Local - offline mode
    Airplane,
    /// Local orchestrator, configurable workers
    Hypermiler(HypermilerConfig),
    /// Use best models until credit exhaustion
    Burn,
}

impl Default for Strategy {
    fn default() -> Self {
        Strategy::Simple
    }
}

impl Strategy {
    /// Get the maximum allowed model tier for this strategy
    fn max_tier(&self) -> Option<ModelTier> {
        match self {
            Strategy::Simple => None, // No tier restriction
            Strategy::Smart(_) => None, // Orchestrator decides
            Strategy::Premium => Some(ModelTier::Premium),
            Strategy::Cheap => Some(ModelTier::Cheap),
            Strategy::Free => Some(ModelTier::Free),
            Strategy::Local | Strategy::Airplane => Some(ModelTier::Local),
            Strategy::Hypermiler(_) => None, // Workers can be any tier
            Strategy::Burn => Some(ModelTier::Premium), // Use best available
        }
    }

    /// Check if a model is allowed under this strategy
    fn allows_model(&self, model_name: &str) -> bool {
        let tier = ModelTier::from_model_name(model_name);
        match self.max_tier() {
            None => true, // No restriction
            Some(max) => tier.is_at_or_below(&max),
        }
    }

    /// Get a suggested fallback model when requested model is not allowed
    /// Each tier can use models at or below its tier:
    /// - Premium: Opus, GPT-5.1 Codex Max (but also Sonnet, Haiku, etc if needed)
    /// - Standard: Sonnet, Haiku (not Opus/premium)
    /// - Cheap: Haiku, GLM, Devstral 2 (anything at or below Cheap)
    /// - Free: GLM, Devstral 2
    /// - Local: devstral-small-2, gpt-oss-20b
    fn suggest_fallback(&self, _requested: &str) -> Option<&'static str> {
        match self {
            Strategy::Premium => Some("claude-opus-4-5-20251101"),
            Strategy::Cheap => Some("glm-4.6v"),                   // GLM is free/flatrate, prefer vision
            Strategy::Free => Some("glm-4.6v"),                    // GLM vision for free tier
            Strategy::Local | Strategy::Airplane => Some("devstral-small-2"),
            _ => None, // No fallback needed
        }
    }
}

/// Shared application state
#[derive(Clone)]
struct AppState {
    models: Arc<HashMap<String, ModelConfig>>,
    http_client: reqwest::Client,
    /// Rate limiter shared across all requests
    rate_limiter: Arc<RateLimiter>,
    /// Per-provider queues for buffering
    provider_queues: Arc<HashMap<Provider, Arc<ProviderQueue>>>,
    /// Maximum output tokens (caps runaway generation)
    max_output_tokens: u32,
    /// All available packs
    packs: Arc<HashMap<&'static str, ModelPack>>,
    /// Default pack for new conversations
    default_pack: ModelPack,
    /// Path to state file for persistence
    state_file: PathBuf,
    /// Configuration for continuous mode verification models
    continuous_config: Arc<ContinuousConfig>,
    /// Build and test status (global - per project, not per conversation)
    project_status: Arc<RwLock<ProjectStatus>>,
    /// Per-conversation state (keyed by X-Conversation-Id header)
    conversations: Arc<RwLock<HashMap<String, ConversationState>>>,
}

impl AppState {
    fn new() -> Self {
        let mut models = HashMap::new();

        // === Mistral/Devstral models (OpenAI-compatible, needs translation) ===
        let mistral_url = env::var("DEVSTRAL_API_URL")
            .unwrap_or_else(|_| "https://api.mistral.ai/v1/chat/completions".to_string());

        // Full fat Devstral (default for @@mistral/@@devstral)
        models.insert("devstral".to_string(), ModelConfig {
            backend_url: mistral_url.clone(),
            api_key_env: "MISTRAL_API_KEY".to_string(),
            model_id: "devstral-2512".to_string(),
            passthrough: false,
        });
        models.insert("devstral-2512".to_string(), ModelConfig {
            backend_url: mistral_url.clone(),
            api_key_env: "MISTRAL_API_KEY".to_string(),
            model_id: "devstral-2512".to_string(),
            passthrough: false,
        });
        // Small Devstral (call with @@devstral-small-2)
        models.insert("devstral-small-2".to_string(), ModelConfig {
            backend_url: mistral_url.clone(),
            api_key_env: "MISTRAL_API_KEY".to_string(),
            model_id: "labs-devstral-small-2512".to_string(),
            passthrough: false,
        });
        models.insert("devstral-mini-2".to_string(), ModelConfig {
            backend_url: mistral_url.clone(),
            api_key_env: "MISTRAL_API_KEY".to_string(),
            model_id: "labs-devstral-small-2512".to_string(),
            passthrough: false,
        });

        // === GLM models via z.ai (Anthropic-compatible, passthrough) ===
        let zai_url = "https://api.z.ai/api/anthropic/v1/messages".to_string();

        models.insert("glm-4.6".to_string(), ModelConfig {
            backend_url: zai_url.clone(),
            api_key_env: "ZAI_API_KEY".to_string(),
            model_id: "glm-4.6".to_string(),
            passthrough: true,
        });
        models.insert("glm-4.6v".to_string(), ModelConfig {
            backend_url: zai_url.clone(),
            api_key_env: "ZAI_API_KEY".to_string(),
            model_id: "glm-4.6v".to_string(),
            passthrough: true,
        });
        models.insert("glm-4.5-air".to_string(), ModelConfig {
            backend_url: zai_url.clone(),
            api_key_env: "ZAI_API_KEY".to_string(),
            model_id: "glm-4.5-air".to_string(),
            passthrough: true,
        });

        // === Real Claude models (Anthropic API, passthrough) ===
        let anthropic_url = "https://api.anthropic.com/v1/messages".to_string();

        models.insert("claude-opus-4-5-20251101".to_string(), ModelConfig {
            backend_url: anthropic_url.clone(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            model_id: "claude-opus-4-5-20251101".to_string(),
            passthrough: true,
        });
        models.insert("claude-sonnet-4-5".to_string(), ModelConfig {
            backend_url: anthropic_url.clone(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            model_id: "claude-sonnet-4-5".to_string(),
            passthrough: true,
        });
        models.insert("claude-haiku-4-5".to_string(), ModelConfig {
            backend_url: anthropic_url.clone(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            model_id: "claude-haiku-4-5".to_string(),
            passthrough: true,
        });

        // === Qwen models (OpenAI-compatible, needs translation) ===
        models.insert("qwen".to_string(), ModelConfig {
            backend_url: env::var("QWEN_API_URL")
                .unwrap_or_else(|_| "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions".to_string()),
            api_key_env: "DASHSCOPE_API_KEY".to_string(),
            model_id: "qwen-plus".to_string(),
            passthrough: false,
        });

        // === OpenRouter models (OpenAI-compatible API) ===
        // GPT-5.1 Codex Max: $1.25/Mtok in, $10/Mtok out, 400K context
        models.insert("openai/gpt-5.1-codex-max".to_string(), ModelConfig {
            backend_url: "https://openrouter.ai/api/v1/chat/completions".to_string(),
            api_key_env: "OPENROUTER_API_KEY".to_string(),
            model_id: "openai/gpt-5.1-codex-max".to_string(),
            passthrough: false, // OpenRouter uses OpenAI format
        });
        models.insert("gpt-5.1-codex-max".to_string(), ModelConfig {
            backend_url: "https://openrouter.ai/api/v1/chat/completions".to_string(),
            api_key_env: "OPENROUTER_API_KEY".to_string(),
            model_id: "openai/gpt-5.1-codex-max".to_string(),
            passthrough: false,
        });

        // Gemini 3 Pro: $2/Mtok in, $12/Mtok out, 1M context
        models.insert("google/gemini-3-pro-preview".to_string(), ModelConfig {
            backend_url: "https://openrouter.ai/api/v1/chat/completions".to_string(),
            api_key_env: "OPENROUTER_API_KEY".to_string(),
            model_id: "google/gemini-3-pro-preview".to_string(),
            passthrough: false,
        });
        models.insert("gemini-3-pro".to_string(), ModelConfig {
            backend_url: "https://openrouter.ai/api/v1/chat/completions".to_string(),
            api_key_env: "OPENROUTER_API_KEY".to_string(),
            model_id: "google/gemini-3-pro-preview".to_string(),
            passthrough: false,
        });

        // Initialize provider queues
        let mut provider_queues = HashMap::new();
        for provider in [Provider::Mistral, Provider::Anthropic, Provider::OpenAI, Provider::GLM, Provider::Qwen, Provider::OpenRouter, Provider::Local] {
            provider_queues.insert(provider, Arc::new(ProviderQueue::new()));
        }

        // Parse max output tokens from env, default to 32k
        let max_output_tokens = env::var("PALACE_MAX_OUTPUT_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);

        // Initialize model packs
        let packs = ModelPack::all_packs();

        // State file path (can be overridden via env)
        let state_file = PathBuf::from(
            env::var("PALACE_STATE_FILE").unwrap_or_else(|_| STATE_FILE.to_string())
        );

        // Try to load persisted pack, fall back to env default, then glm
        let default_pack = load_persisted_pack(&state_file, &packs)
            .or_else(|| {
                let name = env::var("PALACE_DEFAULT_PACK").ok()?;
                packs.get(name.as_str()).cloned()
            })
            .unwrap_or_else(|| packs["glm"].clone());

        info!("Active model pack: {} ({}: opus={}, sonnet={}, haiku={})",
            default_pack.name, default_pack.description,
            default_pack.opus, default_pack.sonnet, default_pack.haiku);

        // Continuous mode config
        let continuous_config = ContinuousConfig::default();
        info!("Continuous mode config: ollama={}, quick={}, medium={}, deep={}",
            continuous_config.ollama_url, continuous_config.quick_model,
            continuous_config.medium_model, continuous_config.deep_model);

        Self {
            models: Arc::new(models),
            http_client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300)) // 5 min timeout for long completions
                .build()
                .unwrap(),
            rate_limiter: Arc::new(RateLimiter::new()),
            provider_queues: Arc::new(provider_queues),
            max_output_tokens,
            packs: Arc::new(packs),
            default_pack,
            state_file,
            continuous_config: Arc::new(continuous_config),
            project_status: Arc::new(RwLock::new(ProjectStatus::default())),
            conversations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or create conversation state for a given conversation ID
    async fn get_conversation(&self, conversation_id: &str) -> ConversationState {
        let mut conversations = self.conversations.write().await;

        // Clean up stale conversations while we have the lock
        conversations.retain(|_id, state| !state.is_stale(CONVERSATION_IDLE_TIMEOUT));

        // Get or create conversation state
        if let Some(state) = conversations.get_mut(conversation_id) {
            state.touch();
            state.clone()
        } else {
            let state = ConversationState::new(
                conversation_id.to_string(),
                self.default_pack.clone(),
            );
            conversations.insert(conversation_id.to_string(), state.clone());
            info!("Created new conversation state: {} (total: {})", conversation_id, conversations.len());
            state
        }
    }

    /// Update conversation state
    async fn update_conversation(&self, state: ConversationState) {
        let mut conversations = self.conversations.write().await;
        conversations.insert(state.id.clone(), state);
    }

    /// Save active pack to state file for persistence across restarts
    async fn persist_pack(&self, pack_name: &str) {
        if let Some(parent) = self.state_file.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                warn!("Failed to create state directory: {}", e);
                return;
            }
        }
        if let Err(e) = tokio::fs::write(&self.state_file, pack_name).await {
            warn!("Failed to persist pack selection: {}", e);
        } else {
            info!("Persisted pack selection: {}", pack_name);
        }
    }

    fn get_model_config(&self, model_name: &str) -> Option<&ModelConfig> {
        self.models.get(model_name)
    }

    fn get_queue(&self, provider: Provider) -> Arc<ProviderQueue> {
        self.provider_queues.get(&provider).cloned().unwrap_or_else(|| Arc::new(ProviderQueue::new()))
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("palace_daemon=info".parse().unwrap()),
        )
        .init();

    let state = AppState::new();

    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .route("/health", get(health_check))
        .route("/stats", get(rate_limit_stats))
        .route("/switch", get(get_switch).post(set_switch))
        .route("/status", get(get_status_handler))
        .route("/status/build", get(get_build_status_handler))
        .route("/status/tests", get(get_test_status_handler))
        .route("/status/update", post(update_status_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let port = env::var("PORT").unwrap_or_else(|_| "19848".to_string());
    let addr = format!("127.0.0.1:{}", port);

    info!("Palace Translator Daemon v6.0.0 (model packs) starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_check() -> &'static str {
    "OK"
}

/// Rate limit stats endpoint - now includes queue info
async fn rate_limit_stats(State(state): State<AppState>) -> Json<serde_json::Value> {
    let providers = [
        ("mistral", Provider::Mistral),
        ("anthropic", Provider::Anthropic),
        ("openai", Provider::OpenAI),
        ("glm", Provider::GLM),
        ("qwen", Provider::Qwen),
        ("openrouter", Provider::OpenRouter),
        ("local", Provider::Local),
    ];

    let stats: serde_json::Map<String, serde_json::Value> = providers
        .iter()
        .map(|(name, provider)| {
            let s = state.rate_limiter.stats(*provider);
            let queue = state.get_queue(*provider);
            (
                name.to_string(),
                serde_json::json!({
                    "requests_available": s.requests_available,
                    "tokens_available": s.tokens_available,
                    "backoff_multiplier": s.backoff_multiplier,
                    "preemptive_limiting": provider.preemptive_limiting(),
                    "queue_depth": queue.queue_depth(),
                    "in_backoff": queue.is_in_backoff(),
                }),
            )
        })
        .collect();

    Json(serde_json::Value::Object(stats))
}

/// GET /switch - show default pack and available options
async fn get_switch(State(state): State<AppState>) -> Json<serde_json::Value> {
    let packs: Vec<serde_json::Value> = state.packs.iter()
        .filter(|(k, v)| *k == &v.name) // Only show canonical names, not aliases
        .map(|(_, pack)| serde_json::json!({
            "name": pack.name,
            "description": pack.description,
            "opus": pack.opus,
            "sonnet": pack.sonnet,
            "haiku": pack.haiku,
            "active": pack.name == state.default_pack.name,
        }))
        .collect();

    Json(serde_json::json!({
        "default": {
            "name": state.default_pack.name,
            "description": state.default_pack.description,
            "opus": state.default_pack.opus,
            "sonnet": state.default_pack.sonnet,
            "haiku": state.default_pack.haiku,
        },
        "available": packs,
        "note": "This shows the default pack for new conversations. Each conversation can use @switch=pack to change their own pack.",
        "usage": "Use @switch=glm in message to change pack for a conversation"
    }))
}

/// POST /switch - change default pack for new conversations
async fn set_switch(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, AppError> {
    let pack_name = body.get("pack")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::Upstream("Missing 'pack' field".to_string()))?;

    let new_pack = state.packs.get(pack_name)
        .ok_or_else(|| AppError::UnknownModel(format!("Unknown pack: {}", pack_name)))?;

    info!("Setting default model pack to: {} (opus={}, sonnet={}, haiku={})",
        new_pack.name, new_pack.opus, new_pack.sonnet, new_pack.haiku);

    // Persist for restart - this becomes the default for new conversations
    state.persist_pack(new_pack.name).await;

    Ok(Json(serde_json::json!({
        "status": "default_updated",
        "pack": {
            "name": new_pack.name,
            "description": new_pack.description,
            "opus": new_pack.opus,
            "sonnet": new_pack.sonnet,
            "haiku": new_pack.haiku,
        },
        "note": "This sets the default pack for new conversations. Existing conversations are not affected. Use @switch=pack in messages to change a specific conversation's pack."
    })))
}

/// Extract @switch command from the LAST user message only
fn extract_switch_command(messages: &[anthropic::Message]) -> Option<String> {
    // Only check the LAST user message - not the whole conversation history
    let last_user_msg = messages.iter()
        .rev()
        .find(|m| m.role == anthropic::Role::User)?;

    let text = match &last_user_msg.content {
        anthropic::Content::Text(t) => t.clone(),
        anthropic::Content::Blocks(blocks) => {
            blocks.iter()
                .filter_map(|b| match b {
                    anthropic::ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        }
        anthropic::Content::Unknown(v) => {
            // Try to extract text from unknown content (e.g., GLM's webReader output)
            v.to_string()
        }
    };

    // Look for @switch or @switch=pack
    if text.contains("@switch") {
        if let Some(eq_pos) = text.find("@switch=") {
            let after = &text[eq_pos + 8..];
            let pack_name: String = after.chars()
                .take_while(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .collect();
            if !pack_name.is_empty() {
                return Some(pack_name);
            }
        }
        // Just @switch with no value - return empty to signal "list packs"
        return Some(String::new());
    }
    None
}

/// Strip thinking blocks that lack valid signatures (cross-model compatibility)
/// When switching from GLM to Claude, GLM's thinking blocks don't have signatures
/// but Claude requires them. We strip these to prevent 400 errors.
fn strip_invalid_thinking_blocks(messages: &mut Vec<anthropic::Message>) {
    let mut stripped_count = 0;
    for (msg_idx, msg) in messages.iter_mut().enumerate() {
        if let anthropic::Content::Blocks(blocks) = &mut msg.content {
            let before = blocks.len();
            for (blk_idx, block) in blocks.iter().enumerate() {
                match block {
                    anthropic::ContentBlock::Thinking { signature, .. } => {
                        info!("Found thinking block at msg[{}].content[{}], signature len={}",
                              msg_idx, blk_idx, signature.len());
                    }
                    _ => {}
                }
            }
            blocks.retain(|block| {
                match block {
                    anthropic::ContentBlock::Thinking { signature, .. } => {
                        // Keep only if signature is non-empty
                        let keep = !signature.is_empty();
                        if !keep {
                            info!("Stripping thinking block with empty signature");
                        }
                        keep
                    }
                    _ => true, // Keep all other block types
                }
            });
            stripped_count += before - blocks.len();
        }
    }
    if stripped_count > 0 {
        info!("Stripped {} thinking blocks with invalid signatures", stripped_count);
    }
}

/// Convert ALL tool_use and tool_result blocks to text summaries.
/// This is needed when handing off between different model backends (e.g., Claude -> Mistral)
/// because the receiving model didn't make those tool calls and its backend will reject them.
fn convert_tool_blocks_to_text(messages: &mut Vec<anthropic::Message>) {
    let mut converted_count = 0;

    for message in messages.iter_mut() {
        if let anthropic::Content::Blocks(blocks) = &mut message.content {
            let mut new_blocks = Vec::new();

            for block in blocks.drain(..) {
                match block {
                    anthropic::ContentBlock::ToolUse { name, input, .. } => {
                        // Convert tool_use to a text summary
                        let input_str = serde_json::to_string_pretty(&input)
                            .unwrap_or_else(|_| input.to_string());
                        let summary = format!("[Tool call: {} with input: {}]", name, input_str);
                        new_blocks.push(anthropic::ContentBlock::Text { text: summary });
                        converted_count += 1;
                    }
                    anthropic::ContentBlock::ToolResult { tool_use_id, content, is_error } => {
                        // Convert tool_result to a text summary
                        let result_text = match content {
                            anthropic::ToolResultContent::Text(t) => t,
                            anthropic::ToolResultContent::Blocks(bs) => {
                                bs.into_iter().filter_map(|b| match b {
                                    anthropic::ToolResultBlock::Text { text } => Some(text),
                                    _ => None,
                                }).collect::<Vec<_>>().join("\n")
                            }
                        };
                        let error_prefix = if is_error.unwrap_or(false) { "ERROR: " } else { "" };
                        let summary = format!("[Tool result for {}: {}{}]", tool_use_id, error_prefix,
                            if result_text.len() > 500 {
                                format!("{}...(truncated)", &result_text[..500])
                            } else {
                                result_text
                            });
                        new_blocks.push(anthropic::ContentBlock::Text { text: summary });
                        converted_count += 1;
                    }
                    other => new_blocks.push(other),
                }
            }

            *blocks = new_blocks;
        }
    }

    if converted_count > 0 {
        info!("Converted {} tool blocks to text summaries for cross-model handoff", converted_count);
    }
}

/// Sanitize orphaned tool_result blocks that don't have matching tool_use in preceding message.
/// This fixes errors like "unexpected tool_use_id found in tool_result blocks: Rv01om234"
fn sanitize_orphaned_tool_results(messages: &mut Vec<anthropic::Message>) {
    // Build a set of valid tool_use IDs from the message immediately before each user message
    // tool_result blocks in a user message must reference tool_use blocks from the immediately
    // preceding assistant message

    let mut i = 0;
    while i < messages.len() {
        if messages[i].role == anthropic::Role::User {
            // Get tool_use IDs from the immediately preceding assistant message
            let valid_tool_ids: std::collections::HashSet<String> = if i > 0 {
                if let anthropic::Content::Blocks(blocks) = &messages[i - 1].content {
                    blocks.iter().filter_map(|b| {
                        if let anthropic::ContentBlock::ToolUse { id, .. } = b {
                            Some(id.clone())
                        } else {
                            None
                        }
                    }).collect()
                } else {
                    std::collections::HashSet::new()
                }
            } else {
                std::collections::HashSet::new()
            };

            // Filter out tool_result blocks that reference non-existent tool_use IDs
            if let anthropic::Content::Blocks(blocks) = &mut messages[i].content {
                let before = blocks.len();
                blocks.retain(|block| {
                    if let anthropic::ContentBlock::ToolResult { tool_use_id, .. } = block {
                        let keep = valid_tool_ids.contains(tool_use_id);
                        if !keep {
                            info!("Stripping orphaned tool_result with id {} (no matching tool_use in preceding message)", tool_use_id);
                        }
                        keep
                    } else {
                        true
                    }
                });
                if blocks.len() < before {
                    info!("Removed {} orphaned tool_result blocks from message {}", before - blocks.len(), i);
                }
            }
        }
        i += 1;
    }
}

/// Sanitize unknown content formats (e.g., GLM's webReader output) by converting to text.
/// This prevents 422 errors when forwarding to Anthropic or other backends that don't
/// understand provider-specific content formats.
fn sanitize_unknown_content(messages: &mut Vec<anthropic::Message>) {
    let mut converted_count = 0;

    for message in messages.iter_mut() {
        if let anthropic::Content::Unknown(v) = &message.content {
            // Convert unknown content to a text representation
            let text_content = if let Some(s) = v.as_str() {
                s.to_string()
            } else {
                // For complex unknown content, serialize it as a readable format
                format!("[Provider-specific content: {}]",
                    serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string()))
            };
            message.content = anthropic::Content::Text(text_content);
            converted_count += 1;
        }
    }

    if converted_count > 0 {
        info!("Sanitized {} messages with unknown content format", converted_count);
    }
}

/// Handle incoming Anthropic-format messages with buffering
async fn handle_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut request): Json<anthropic::MessagesRequest>,
) -> Result<Response, AppError> {
    let request_start = Instant::now();

    // Extract conversation ID from headers (fall back to "default" for backwards compat)
    let conversation_id = headers
        .get("x-conversation-id")
        .or_else(|| headers.get("X-Conversation-Id"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "default".to_string());

    // Extract project path from headers (for running tests in the right directory)
    let project_path = headers
        .get("x-project-path")
        .or_else(|| headers.get("X-Project-Path"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Extract user to run as (for spawning workers with correct permissions)
    let run_as_user = headers
        .get("x-user")
        .or_else(|| headers.get("X-User"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Get or create per-conversation state
    let mut conv = state.get_conversation(&conversation_id).await;

    // Update project path - prefer header, fall back to system prompt <env> block
    let effective_project_path = project_path.or_else(|| {
        // Try to extract from system prompt's <env> block
        // Format: <env>\nWorking directory: /path/to/project\n...\n</env>
        request.system.as_ref().and_then(|s| {
            let text = s.to_string();
            extract_working_directory(&text)
        })
    });

    if let Some(ref path) = effective_project_path {
        if conv.project_path.as_ref() != Some(path) {
            info!("Updated project path for conversation {}: {}", conversation_id, path);
            conv.project_path = Some(path.clone());
        }
    }

    // Update run_as_user - prefer header, fall back to project path ownership
    info!("run_as_user extraction: header={:?}, conv.project_path={:?}", run_as_user, conv.project_path);
    let effective_run_as_user = run_as_user.or_else(|| {
        conv.project_path.as_ref().and_then(|p| get_user_from_path_owner(p))
    });
    info!("run_as_user result: {:?}", effective_run_as_user);

    if let Some(ref user) = effective_run_as_user {
        if conv.run_as_user.as_ref() != Some(user) {
            info!("Updated run_as_user for conversation {}: {}", conversation_id, user);
            conv.run_as_user = Some(user.clone());
        }
    }

    // Track if we switched packs (to inject reminder)
    let mut did_switch = false;

    // Check for @continuous toggle in user messages
    let last_user_message: Option<String> = request.messages.iter()
        .filter(|m| matches!(m.role, anthropic::Role::User))
        .last()
        .map(|m| match &m.content {
            anthropic::Content::Text(s) => s.clone(),
            anthropic::Content::Blocks(blocks) => {
                blocks.iter()
                    .filter_map(|b| match b {
                        anthropic::ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            anthropic::Content::Unknown(v) => v.to_string(),
        });

    if let Some(ref msg) = last_user_message {
        // Debug: log message content for toggle detection
        if msg.contains("@continuous") || msg.contains("@costs") || msg.contains("@strategy") || msg.contains("@hypermiler") {
            info!("Toggle detection: message contains toggle keyword, len={}", msg.len());
        }

        if let Some(new_mode) = detect_continuous_toggle(msg) {
            // Check if we're transitioning to an "on" state
            let was_off = matches!(conv.continuous_mode, ContinuousMode::Off);

            conv.continuous_mode = new_mode.clone();

            // Log mode changes and update start time
            match &new_mode {
                ContinuousMode::Off => {
                    info!("[{}] @continuous: Disabled - normal single-turn mode", conversation_id);
                    conv.continuous_start = None;
                }
                ContinuousMode::Unlimited => {
                    if was_off {
                        info!("[{}] @continuous: Enabled (unlimited) - model will continue until success criteria met", conversation_id);
                        conv.continuous_start = Some(Instant::now());
                    }
                }
                ContinuousMode::TimeLimited(duration) => {
                    if was_off {
                        info!("[{}] @continuous: Enabled (time limit: {}s) - model will continue for up to {:?}",
                            conversation_id, duration.as_secs(), duration);
                        conv.continuous_start = Some(Instant::now());
                    } else {
                        info!("[{}] @continuous: Time limit updated to {}s ({:?})",
                            conversation_id, duration.as_secs(), duration);
                    }
                }
            }

            // Auto-start watch when continuous mode is enabled
            if was_off && !matches!(new_mode, ContinuousMode::Off) {
                if let Some(ref path) = conv.project_path {
                    info!("[{}] @continuous: Auto-starting watch for project: {}", conversation_id, path);
                    let _ = start_watching_project(path).await;
                }

                // Auto-enable swarm mode when continuous mode starts (preserves existing limits)
                if conv.swarm_config.swarm_mode == SwarmMode::Off {
                    conv.swarm_config.swarm_mode = SwarmMode::On;
                    info!("[{}] @continuous: Auto-enabled swarm mode (cap: {})", conversation_id, conv.swarm_config.max_workers);
                }
            }
        }

        // Check for @watch toggle - return early response
        if let Some(watch_enabled) = detect_watch_toggle(msg) {
            let response_text = if watch_enabled {
                if let Some(ref path) = conv.project_path {
                    info!("[{}] @watch: Starting watch for project: {}", conversation_id, path);
                    match start_watching_project(path).await {
                        Ok(_) => format!("[@watch] Now watching: {}\n\nBuild status will be monitored continuously. Use `@verify` to check current status.", path),
                        Err(e) => format!("[@watch] Failed to start watch for {}: {}", path, e),
                    }
                } else {
                    info!("[{}] @watch: No project path detected - cannot start watch", conversation_id);
                    "[@watch] No project path detected.\n\nCannot start watch without a project path. Either:\n- Send a message from a Claude Code session (auto-detected from <env> block)\n- Include `X-Project-Path: /path/to/project` header".to_string()
                }
            } else {
                info!("[{}] @watch: Disabled (note: watchd keeps running, use 'pal watch --stop' to fully stop)", conversation_id);
                "[@watch] Watch disabled for this session.\n\nNote: The watchd daemon continues running. Use `pal watch --stop` to fully stop background monitoring.".to_string()
            };

            // Save conversation state before returning
            state.update_conversation(conv).await;

            return Ok(Json(anthropic::MessagesResponse {
                id: format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                response_type: "message".to_string(),
                role: anthropic::Role::Assistant,
                content: vec![anthropic::ContentBlock::Text { text: response_text }],
                model: "system".to_string(),
                stop_reason: Some(anthropic::StopReason::EndTurn),
                usage: anthropic::Usage { input_tokens: 0, output_tokens: 0 },
            }).into_response());
        }

        // Check for @costs toggle
        if let Some(costs_on) = detect_costs_toggle(msg) {
            let was_on = conv.cost_tracker.enabled;
            conv.cost_tracker.set_enabled(costs_on);
            if costs_on && !was_on {
                info!("[{}] @costs: Enabled - tracking API costs", conversation_id);
            } else if !costs_on && was_on {
                info!("[{}] @costs: Disabled - cost tracking off", conversation_id);
            }
        }

        // Check for @hypermiler= orchestrator override
        if let Some(orchestrator) = detect_hypermiler_override(msg) {
            if let Strategy::Hypermiler(ref mut config) = conv.strategy {
                config.orchestrator_model = Some(orchestrator.clone());
                info!("[{}] @hypermiler: Orchestrator set to {}", conversation_id, orchestrator);
            } else {
                // Enable hypermiler with this orchestrator
                conv.strategy = Strategy::Hypermiler(HypermilerConfig {
                    worker_tier: WorkerTier::Premium,
                    orchestrator_model: Some(orchestrator.clone()),
                });
                info!("[{}] @hypermiler: Enabled with orchestrator {}", conversation_id, orchestrator);
            }
        }

        // Check for @planner= planner model override
        if let Some(planner_model) = detect_planner_model(msg) {
            conv.swarm_config.planner_model = Some(planner_model.clone());
            info!("[{}] @planner: Task planner model set to {}", conversation_id, planner_model);
        }

        // Check for @swarm toggle
        if let Some(swarm_mode) = detect_swarm_mode(msg) {
            let was_on = conv.swarm_config.swarm_mode != SwarmMode::Off;
            conv.swarm_config.swarm_mode = swarm_mode.clone();
            match &swarm_mode {
                SwarmMode::On if !was_on => {
                    info!("[{}] @swarm: ENABLED - Parallel worker spawning active (cap: {})", conversation_id, conv.swarm_config.max_workers);
                    // Note: @swarm=on will trigger planner via detect_spawn_command below
                }
                SwarmMode::TaskLimit(n) => {
                    conv.swarm_config.max_workers = *n;
                    info!("[{}] @swarm: ENABLED with safety cap of {} workers", conversation_id, n);
                }
                SwarmMode::Unlimited => {
                    info!("[{}] @swarm: UNLIMITED - No cap, planner decides task count", conversation_id);
                }
                SwarmMode::Off => {
                    let msg = if was_on {
                        "🔴 **Swarm mode disabled.** Single-model operation restored."
                    } else {
                        "ℹ️ Swarm mode is already off."
                    };
                    info!("[{}] @swarm: Off (was_on={})", conversation_id, was_on);
                    state.update_conversation(conv).await;
                    return Ok(Json(anthropic::MessagesResponse {
                        id: format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                        response_type: "message".to_string(),
                        role: anthropic::Role::Assistant,
                        content: vec![anthropic::ContentBlock::Text { text: msg.to_string() }],
                        model: "system".to_string(),
                        stop_reason: Some(anthropic::StopReason::EndTurn),
                        usage: anthropic::Usage { input_tokens: 0, output_tokens: 0 },
                    }).into_response());
                }
                _ => {}
            }
        }

        // Check for @strategy toggle
        if let Some(new_strategy) = detect_strategy_toggle(msg) {
            conv.strategy = new_strategy.clone();
            info!("[{}] @strategy: Changed to {:?}", conversation_id, new_strategy);
        }

        // Check for @strict toggle
        if let Some(new_strict) = detect_strict_toggle(msg) {
            let was_on = conv.strict_mode == StrictMode::On;
            conv.strict_mode = new_strict.clone();
            match new_strict {
                StrictMode::On if !was_on => {
                    info!("[{}] @strict: ENABLED - Models cannot complete until ALL builds and tests pass", conversation_id);
                }
                StrictMode::Off if was_on => {
                    info!("[{}] @strict: Disabled - Models can complete without passing tests", conversation_id);
                }
                _ => {}
            }
        }

        // Check for @verify command (runs build + tests immediately)
        if detect_verify_command(msg) {
            info!("[{}] @verify: Running verification...", conversation_id);

            let project_path_str = conv.project_path.clone();
            let verification = read_watchd_status(project_path_str.as_deref()).await;

            // Store result in conversation state
            conv.last_verification = Some(verification.clone());

            // Format and return the result
            let response_text = format_verification_status(&verification);

            info!("[{}] @verify: Build={}, Tests={}",
                conversation_id,
                if verification.build_passing { "PASS" } else { "FAIL" },
                if verification.test_passing { "PASS" } else { "FAIL" }
            );

            // Save conversation state
            state.update_conversation(conv).await;

            return Ok(Json(anthropic::MessagesResponse {
                id: format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                response_type: "message".to_string(),
                role: anthropic::Role::Assistant,
                content: vec![anthropic::ContentBlock::Text { text: response_text }],
                model: "system".to_string(),
                stop_reason: Some(anthropic::StopReason::EndTurn),
                usage: anthropic::Usage { input_tokens: 0, output_tokens: 0 },
            }).into_response());
        }

        // Check for @spawn command (trigger swarm execution with streaming output)
        if detect_spawn_command(msg) {
            info!("[{}] @spawn: Triggering streaming planner...", conversation_id);

            // Ensure we have a project path
            let project_path = match &conv.project_path {
                Some(p) => p.clone(),
                None => {
                    let error_text = "[@swarm] No project path detected.\n\nCannot run swarm without a project path. Send from a Claude Code session or use X-Project-Path header.".to_string();
                    state.update_conversation(conv).await;
                    return Ok(Json(anthropic::MessagesResponse {
                        id: format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                        response_type: "message".to_string(),
                        role: anthropic::Role::Assistant,
                        content: vec![anthropic::ContentBlock::Text { text: error_text }],
                        model: "system".to_string(),
                        stop_reason: Some(anthropic::StopReason::EndTurn),
                        usage: anthropic::Usage { input_tokens: 0, output_tokens: 0 },
                    }).into_response());
                }
            };

            // Get planner model from config
            let hypermiler_config = match &conv.strategy {
                Strategy::Hypermiler(config) => config.clone(),
                _ => HypermilerConfig::default(),
            };
            let planner_model = conv.swarm_config.planner_model.clone()
                .or(hypermiler_config.orchestrator_model.clone())
                .unwrap_or_else(|| "devstral-2512".to_string());

            // Clone run_as_user before moving conv
            let run_as_user = conv.run_as_user.clone();

            // Get project status for the prompt
            let build_status = read_watchd_status(Some(&project_path)).await;
            let prompt = build_task_planning_prompt(Some(&project_path), &build_status);

            // Extract the actual user request from the message (everything after @spawn)
            let user_request = msg.split("@spawn").nth(1)
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .unwrap_or("Analyze this project and help with any issues");

            let full_prompt = format!("{}\n\nUser request: {}", prompt, user_request);

            // Save conversation state before streaming
            state.update_conversation(conv).await;

            // Clone state for the streaming function (Arc fields are shared)
            let state_for_stream = state.clone();
            let conv_id_for_stream = conversation_id.clone();

            // Stream the planner output with compact rendering
            let stream = stream_planner_compact(
                &planner_model,
                &full_prompt,
                &project_path,
                run_as_user.as_deref(),
                state_for_stream,
                conv_id_for_stream,
            );

            info!("[{}] @swarm: Starting streaming response", conversation_id);

            // IMPORTANT: Must set Content-Type: text/event-stream for Claude Code to render SSE
            return Ok(Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .header(header::CACHE_CONTROL, "no-cache")
                .body(Body::from_stream(stream))
                .unwrap());
        }

        // Check for @action(s) command to select tasks for execution
        if let Some(selection) = detect_action_command(msg) {
            info!("[{}] @action: Processing selection {:?}", conversation_id, selection);

            // Check if we have pending actions
            if conv.pending_actions.is_empty() {
                let error_text = "[@action] No pending actions available.\n\nRun `@swarm` first to generate actions, then use `@action N` or `@actions 1,2,3` to select.".to_string();
                state.update_conversation(conv).await;
                return Ok(Json(anthropic::MessagesResponse {
                    id: format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                    response_type: "message".to_string(),
                    role: anthropic::Role::Assistant,
                    content: vec![anthropic::ContentBlock::Text { text: error_text }],
                    model: "system".to_string(),
                    stop_reason: Some(anthropic::StopReason::EndTurn),
                    usage: anthropic::Usage { input_tokens: 0, output_tokens: 0 },
                }).into_response());
            }

            // Get selected actions
            let selected: Vec<&PendingAction> = match selection {
                ActionSelection::All => conv.pending_actions.iter().collect(),
                ActionSelection::Indices(indices) => {
                    indices.iter()
                        .filter_map(|&i| conv.pending_actions.iter().find(|a| a.num == i))
                        .collect()
                }
            };

            if selected.is_empty() {
                let error_text = format!(
                    "[@action] No valid actions selected.\n\nAvailable actions: {}\nUse `@action N` where N is an action number.",
                    conv.pending_actions.iter().map(|a| a.num.to_string()).collect::<Vec<_>>().join(", ")
                );
                state.update_conversation(conv).await;
                return Ok(Json(anthropic::MessagesResponse {
                    id: format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                    response_type: "message".to_string(),
                    role: anthropic::Role::Assistant,
                    content: vec![anthropic::ContentBlock::Text { text: error_text }],
                    model: "system".to_string(),
                    stop_reason: Some(anthropic::StopReason::EndTurn),
                    usage: anthropic::Usage { input_tokens: 0, output_tokens: 0 },
                }).into_response());
            }

            // Format selected actions for response
            let mut response_text = format!("🎯 **Selected {} action(s) for execution:**\n\n", selected.len());
            for action in &selected {
                response_text.push_str(&format!("  **{}. {}**\n", action.num, action.label));
                if !action.description.is_empty() {
                    response_text.push_str(&format!("     {}\n", action.description));
                }
                response_text.push('\n');
            }
            response_text.push_str("\n🚀 Ranking and spawning workers...\n");

            info!("[{}] @action: Selected {} actions for execution", conversation_id, selected.len());

            // TODO: Trigger ranking via planner then spawn workers
            // For now, just acknowledge the selection

            state.update_conversation(conv).await;
            return Ok(Json(anthropic::MessagesResponse {
                id: format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                response_type: "message".to_string(),
                role: anthropic::Role::Assistant,
                content: vec![anthropic::ContentBlock::Text { text: response_text }],
                model: "system".to_string(),
                stop_reason: Some(anthropic::StopReason::EndTurn),
                usage: anthropic::Usage { input_tokens: 0, output_tokens: 0 },
            }).into_response());
        }
    }

    // Check continuous mode time limit enforcement
    if let ContinuousMode::TimeLimited(limit) = &conv.continuous_mode {
        if let Some(start_time) = conv.continuous_start {
            let elapsed = start_time.elapsed();
            if elapsed >= *limit {
                // Time limit exceeded - return graceful stop message
                let cost_info = if conv.cost_tracker.enabled {
                    format!("\n- Session cost: ${:.2}", conv.cost_tracker.session_cost)
                } else {
                    String::new()
                };

                let response_text = format!(
                    "[CONTINUOUS] Time limit reached ({:?}). Session summary:\n\
                    - Duration: {:?}\n\
                    - Final status: TIME_LIMIT_EXCEEDED{}\n\n\
                    To continue: @continuous=30m (or @continuous for unlimited)",
                    limit, elapsed, cost_info
                );

                info!("[{}] @continuous: Time limit {:?} exceeded (elapsed: {:?})", conversation_id, limit, elapsed);

                // Disable continuous mode and save state
                conv.continuous_mode = ContinuousMode::Off;
                state.update_conversation(conv).await;

                return Ok(Json(anthropic::MessagesResponse {
                    id: format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                    response_type: "message".to_string(),
                    role: anthropic::Role::Assistant,
                    content: vec![anthropic::ContentBlock::Text { text: response_text }],
                    model: "system".to_string(),
                    stop_reason: Some(anthropic::StopReason::EndTurn),
                    usage: anthropic::Usage { input_tokens: 0, output_tokens: 0 },
                }).into_response());
            }
        }
    }

    // Check for @help command in messages
    if let Some(ref msg) = last_user_message {
        if msg.contains("@help") {
            let strict_status = match &conv.strict_mode {
                StrictMode::Off => "off",
                StrictMode::On => "on (completion blocked until tests pass)",
            };

            let continuous_status = match &conv.continuous_mode {
                ContinuousMode::Off => "off".to_string(),
                ContinuousMode::Unlimited => "on (unlimited)".to_string(),
                ContinuousMode::TimeLimited(d) => format!("on ({:?} limit)", d),
            };

            let strategy_status = match &conv.strategy {
                Strategy::Simple => "simple".to_string(),
                Strategy::Smart(_) => "smart (premium orchestrator)".to_string(),
                Strategy::Premium => "premium".to_string(),
                Strategy::Cheap => "cheap".to_string(),
                Strategy::Free => "free".to_string(),
                Strategy::Local => "local".to_string(),
                Strategy::Airplane => "airplane".to_string(),
                Strategy::Burn => "burn".to_string(),
                Strategy::Hypermiler(config) => {
                    let tier = match config.worker_tier {
                        WorkerTier::Premium => "premium",
                        WorkerTier::Standard => "standard",
                        WorkerTier::Cheap => "cheap",
                        WorkerTier::Free => "free",
                        WorkerTier::Local => "local",
                    };
                    format!("hypermiler (workers: {})", tier)
                }
            };

            let help_text = format!(r#"**Palace Daemon Commands**

**Current Status:**
- Conversation: {}
- Pack: {} ({})
- Continuous: {}
- Strategy: {}
- Strict mode: {}
- Cost tracking: {}
- Session cost: ${:.4}

**Model Switching:**
- `@switch` - List available model packs
- `@switch=mistral` - Switch to Mistral pack (devstral-2512)
- `@switch=glm` - Switch to GLM pack (glm-4.6v)
- `@switch=anthropic` - Switch to Anthropic pack (claude-*)
- `@@modelname` - Hand off to another model mid-response

**Continuous Mode:**
- `@continuous` or `@continuous=on` - Enable unlimited continuous mode
- `@continuous=off` - Disable continuous mode
- `@continuous=30m` - Enable with 30 minute time limit
- `@continuous=2h` - Enable with 2 hour time limit

**Strategy Modes:**
- `@strategy=simple` - Direct model routing (default)
- `@strategy=smart` - Premium orchestrator with delegation
- `@strategy=hypermiler` - Free orchestrator (devstral-2512), premium workers
- `@strategy=hypermiler,cheap` - Free orchestrator, cheap workers
- `@strategy=premium/cheap/free/local` - Restrict to tier

**Cost Tracking:**
- `@costs` or `@costs=on` - Enable cost logging
- `@costs=off` - Disable cost logging

**Strict Mode:**
- `@strict` or `@strict=on` - Block completion until ALL tests pass
- `@strict=off` - Allow completion regardless of test status

**Watch & Verification:**
- `@watch` - Start watching project (auto-starts on @continuous)
- `@unwatch` or `@watch=false` - Stop watching project
- `@verify` - Check current build/test status from watchd
- `X-Project-Path` header overrides auto-detected project path

**Hypermiler Override:**
- `@hypermiler=opus` - Use Opus as orchestrator
- `@hypermiler=devstral` - Use Devstral as orchestrator (default)

**Swarm Mode (parallel workers):**
- `@swarm` or `@swarm=on` - Enable parallel worker spawning
- `@swarm=off` - Disable swarm mode
- `@swarm=unlimited` - Remove safety cap, planner decides everything
- `@swarm=N` - Set safety cap (default 20)
- `@planner=<model>` - Override task planner model (devstral, glm, opus, sonnet, haiku, local)
- `@swarm` or `@spawn` - Run planner to analyze project and generate actions
- `@action N` or `@actions 1,2,3` - Select actions for execution
- `@actions all` - Execute all pending actions
- Note: `@continuous` auto-enables swarm mode
- Note: Models can emit `@@swarm=on` to enable swarm (but not change limits)

**Headers:**
- `X-Conversation-Id: your-id` - Isolate settings per conversation
- `X-Project-Path: /path/to/project` - Set project for @verify and @strict
"#, conversation_id, conv.active_pack.name, conv.active_pack.description, continuous_status, strategy_status,
                strict_status, if conv.cost_tracker.enabled { "on" } else { "off" }, conv.cost_tracker.session_cost);

            info!("[{}] @help: Displaying help menu", conversation_id);

            return Ok(Json(anthropic::MessagesResponse {
                id: format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                response_type: "message".to_string(),
                role: anthropic::Role::Assistant,
                content: vec![anthropic::ContentBlock::Text { text: help_text }],
                model: "system".to_string(),
                stop_reason: Some(anthropic::StopReason::EndTurn),
                usage: anthropic::Usage { input_tokens: 0, output_tokens: 0 },
            }).into_response());
        }
    }

    // Check for @switch command in messages
    if let Some(switch_arg) = extract_switch_command(&request.messages) {
        if switch_arg.is_empty() {
            // @switch with no arg - list available packs
            let pack_list: Vec<String> = state.packs.iter()
                .filter(|(k, v)| *k == &v.name)
                .map(|(_, p)| format!("  {} - {} (opus={}, sonnet={}, haiku={})",
                    p.name, p.description, p.opus, p.sonnet, p.haiku))
                .collect();

            let response_text = format!(
                "**Current model pack:** {} ({})\n\n**Available packs:**\n{}\n\n**Usage:** @switch=glm or @switch=mistral or @switch=anthropic",
                conv.active_pack.name, conv.active_pack.description, pack_list.join("\n")
            );

            return Ok(Json(anthropic::MessagesResponse {
                id: format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                response_type: "message".to_string(),
                role: anthropic::Role::Assistant,
                content: vec![anthropic::ContentBlock::Text { text: response_text }],
                model: conv.active_pack.sonnet.to_string(),
                stop_reason: Some(anthropic::StopReason::EndTurn),
                usage: anthropic::Usage { input_tokens: 0, output_tokens: 0 },
            }).into_response());
        } else {
            // @switch=pack - switch to specified pack SILENTLY
            // Don't return a response - just switch and let the request continue
            if let Some(new_pack) = state.packs.get(switch_arg.as_str()) {
                conv.active_pack = new_pack.clone();

                info!("[{}] @switch: Changed to pack {} (opus={}, sonnet={}, haiku={})",
                    conversation_id, new_pack.name, new_pack.opus, new_pack.sonnet, new_pack.haiku);

                // Persist for restart (global default)
                state.persist_pack(new_pack.name).await;

                // Mark that we switched - need to inject reminder
                did_switch = true;

                // Don't return early - fall through to process the actual request
                // The @switch command was just a directive, not the whole message
            } else {
                // Silently ignore unknown pack names - they might be email addresses
                // or other @ mentions that aren't meant for us
                if switch_arg.contains('.') || switch_arg.contains('@') {
                    info!("@{}: Ignoring (looks like email or domain)", switch_arg);
                } else {
                    info!("@switch={}: Unknown pack, ignoring", switch_arg);
                }
                // Continue processing the request with current pack
            }
        }
    }

    // Map requested model through active pack
    let (mut actual_model, current_pack_name) = {
        let mapped = conv.active_pack.map_model(&request.model);
        // Show "passthrough" when model bypasses pack mapping
        let routing = if mapped == request.model {
            "passthrough".to_string()
        } else {
            format!("pack: {}", conv.active_pack.name)
        };
        info!("Model mapping: {} -> {} ({})", request.model, mapped, routing);
        (mapped.to_string(), conv.active_pack.name.clone())
    };

    // Apply strategy-based model filtering and orchestration
    let (strategy_instructions, _is_hypermiler, _is_smart) = {
        match &conv.strategy {
            Strategy::Hypermiler(config) => {
                // Hypermiler: Force to use cheap orchestrator model
                let orchestrator = select_hypermiler_orchestrator(config.orchestrator_model.as_deref());
                info!("HYPERMILER: Forcing orchestrator model {} (requested: {})", orchestrator, actual_model);
                actual_model = orchestrator;

                // Build worker tier description for instructions
                let worker_desc = match config.worker_tier {
                    WorkerTier::Premium => "Premium models (Claude Opus, GPT-5.1 Codex Max, Gemini 3 Pro via OpenRouter)",
                    WorkerTier::Standard => "Standard models (Claude Sonnet, Devstral via Mistral API)",
                    WorkerTier::Cheap => "Cheap models (Claude Haiku, GLM-4.6v via Z.ai, Devstral via Mistral API)",
                    WorkerTier::Free => "Free/flatrate models (GLM-4.6v via Z.ai, Devstral 2 via Mistral API - free preview)",
                    WorkerTier::Local => "Local models only (Devstral Small 2 24B, gpt-oss-20b, or Devstral 2 quant on 96GB VRAM)",
                };

                let instructions = format!(r#"
HYPERMILER MODE ACTIVE

You are the ORCHESTRATOR running on a FREE model. Your job is to:
1. Handle simple tasks yourself directly
2. Delegate complex tasks to WORKERS using @@switch=modelname

WORKER TIER: {worker_desc}

DELEGATION RULES:
- Simple coding tasks: handle yourself
- Complex architecture decisions: @@switch=claude-opus
- Code review with nuance: @@switch=claude-sonnet
- Bulk simple work: handle yourself or @@switch=glm
- When stuck after 3+ attempts: MUST delegate to premium

Example delegation: "This needs deeper analysis. @@switch=claude-opus"
"#);
                (instructions, true, false)
            }
            Strategy::Smart(config) => {
                // Smart: Premium orchestrator via Claude Code CLI
                // For now, we'll inject instructions - full CLI invocation comes later
                let worker_desc = match config.worker_tier {
                    WorkerTier::Premium => "Premium models (Claude Opus, GPT-5.1 Codex Max, Gemini 3 Pro)",
                    WorkerTier::Standard => "Standard models (Claude Sonnet, Devstral via Mistral API)",
                    WorkerTier::Cheap => "Cheap models (Claude Haiku, GLM-4.6v, Devstral)",
                    WorkerTier::Free => "Free/flatrate models (GLM-4.6v via Z.ai, Devstral 2 via Mistral API)",
                    WorkerTier::Local => "Local models only (Devstral Small 2, gpt-oss-20b)",
                };

                let instructions = format!(r#"
SMART MODE ACTIVE

You are the ORCHESTRATOR (premium model). Your job is to:
1. Analyze task complexity
2. Break into subtasks
3. Assign to optimal worker tier
4. Make judgment calls on quality

WORKER TIER: {worker_desc}

DELEGATION: Use @@switch=modelname to delegate.
RATE LIMITS: Be mindful of rate limits. Batch work when possible.
"#);
                (instructions, false, true)
            }
            strategy => {
                // Other strategies: just filter models
                if !strategy.allows_model(&actual_model) {
                    let tier = ModelTier::from_model_name(&actual_model);
                    if let Some(fallback) = strategy.suggest_fallback(&actual_model) {
                        info!("Strategy {:?}: Model {} (tier {:?}) not allowed, falling back to {}",
                            strategy, actual_model, tier, fallback);
                        actual_model = fallback.to_string();
                    } else {
                        warn!("Strategy {:?}: Model {} (tier {:?}) not allowed but no fallback available",
                            strategy, actual_model, tier);
                    }
                }
                (String::new(), false, false)
            }
        }
    };

    // Update request with actual model
    request.model = actual_model.clone();

    // Inject multi-model capability reminder into system prompt on:
    // Inject handoff system on first message of session or after switch
    // Debug: log message structure to understand Claude Code patterns
    let user_message_count = request.messages.iter()
        .filter(|m| matches!(m.role, anthropic::Role::User))
        .count();
    let assistant_message_count = request.messages.iter()
        .filter(|m| matches!(m.role, anthropic::Role::Assistant))
        .count();
    let total_messages = request.messages.len();
    // Debug: log message roles in order to trace message ordering issues
    let anthropic_role_order: Vec<String> = request.messages.iter()
        .map(|m| format!("{:?}", m.role))
        .collect();
    info!("Message structure: total={}, user={}, assistant={}, did_switch={}, order={:?}",
          total_messages, user_message_count, assistant_message_count, did_switch, anthropic_role_order);

    // Check and manage reminder persistence across turns
    let _reminder_remaining = {
        if did_switch {
            // Reset counter on explicit switch - give model 2 more turns to internalize
            conv.reminder_turns_remaining = 2;
            info!("Switch detected, setting reminder_turns_remaining = 2");
        }
        let current = conv.reminder_turns_remaining;
        if current > 0 {
            conv.reminder_turns_remaining -= 1;
            info!("Reminder turns remaining: {} -> {}", current, conv.reminder_turns_remaining);
        }
        current
    };

    // Inject reminder EVERY MESSAGE
    {
        // Check if continuous mode is active
        let continuous_enabled = !matches!(conv.continuous_mode, ContinuousMode::Off);

        let continuous_section = if continuous_enabled {
            r#"
CONTINUOUS MODE ACTIVE

When you complete a task/goal, output this format at the end:

TASK COMPLETE
confidence:
  [category]: [0-100]
  [category]: [0-100]

Categories should match the task (e.g. code_quality, test_coverage, documentation, performance, correctness, edge_cases).
Scores: 95+ = excellent, 70-94 = good, 50-69 = needs work, <50 = significant issues.
"#
        } else {
            ""
        };

        // Build strategy section for orchestration modes
        let strategy_section = if !strategy_instructions.is_empty() {
            strategy_instructions.clone()
        } else {
            String::new()
        };

        let reminder = format!(r#"

<system-reminder>
MULTI-MODEL SYSTEM

You are {0}, specifically model {1}. Do not switch to yourself - follow the user's instructions. When done, if requested, you can switch with @@switch=OTHER_MODEL. Do NOT @@switch={0} (self) - this will break the user's workflow.

To hand off: put @@switch=modelname in your text output (e.g. @@switch=claude, @@switch=mistral, @@switch=glm).
Do NOT use Skill() or natural language like "passing to X" - only @@switch= works.
{2}{3}</system-reminder>
"#, current_pack_name, actual_model, continuous_section, strategy_section);
        request.system = Some(match &request.system {
            Some(system) => {
                anthropic::SystemPrompt::Text(format!("{}{}", system.to_string(), reminder))
            }
            None => {
                anthropic::SystemPrompt::Text(reminder.trim_start().to_string())
            }
        });
    }

    let provider = Provider::from_model(&actual_model);
    let queue = state.get_queue(provider);

    // Try to acquire a queue slot (limits total concurrent requests when backed off)
    let _permit = queue.semaphore.acquire().await
        .map_err(|_| AppError::QueueFull)?;

    queue.queued.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // RAII guard to decrement queue count on exit
    struct QueueGuard {
        queue: Arc<ProviderQueue>,
    }
    impl Drop for QueueGuard {
        fn drop(&mut self) {
            self.queue.queued.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let _guard = QueueGuard { queue: queue.clone() };

    // Look up model configuration
    let config = state
        .get_model_config(&actual_model)
        .ok_or_else(|| AppError::UnknownModel(actual_model.clone()))?;

    let api_key = env::var(&config.api_key_env)
        .map_err(|_| AppError::MissingApiKey(config.api_key_env.clone()))?;

    // Retry loop with exponential backoff
    let mut attempt = 0;
    let mut last_error = None;
    let mut was_delayed = false;

    loop {
        attempt += 1;

        // Check rate limits - but don't fail, just wait
        match state.rate_limiter.acquire_request(provider) {
            RateLimitResult::Allowed => {}
            RateLimitResult::Limited { wait_ms, reason } => {
                if attempt > MAX_RETRY_ATTEMPTS {
                    warn!("Exceeded max retries after rate limiting: {}", reason);
                    state.rate_limiter.finish_waiting(provider);
                    break;
                }
                was_delayed = true;
                info!("Rate limited (attempt {}), waiting {}ms: {}", attempt, wait_ms, reason);
                tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                state.rate_limiter.finish_waiting(provider);
                continue;
            }
        }

        // Make the actual request
        let result = if config.passthrough {
            // Cap max_tokens for passthrough too
            let mut capped_request = request.clone();
            if capped_request.max_tokens > state.max_output_tokens {
                info!("Capped passthrough max_tokens from {} to {}", capped_request.max_tokens, state.max_output_tokens);
                capped_request.max_tokens = state.max_output_tokens;
            }

            // Strip thinking blocks without signatures (cross-model compatibility)
            // GLM doesn't send signatures, Claude requires them
            strip_invalid_thinking_blocks(&mut capped_request.messages);

            // Sanitize orphaned tool_result blocks (fixes "unexpected tool_use_id" errors)
            sanitize_orphaned_tool_results(&mut capped_request.messages);

            // Sanitize unknown content formats (e.g., GLM's webReader output)
            // This prevents 422 errors when forwarding to backends that don't understand
            // provider-specific content formats
            sanitize_unknown_content(&mut capped_request.messages);

            if request.stream.unwrap_or(false) {
                // Build handoff request for baton passing in passthrough mode
                // Save conversation state before streaming (will be updated during streaming)
                state.update_conversation(conv.clone()).await;

                let handoff_request = Some(HandoffRequest {
                    original_request: request.clone(),
                    http_client: state.http_client.clone(),
                    models: state.models.clone(),
                    max_output_tokens: state.max_output_tokens,
                    continuous_config: state.continuous_config.clone(),
                    original_user_request: last_user_message.clone().unwrap_or_default(),
                    actual_model: actual_model.clone(),
                    project_status: state.project_status.clone(),
                    conversation_id: conversation_id.clone(),
                    conversations: state.conversations.clone(),
                });
                execute_passthrough_streaming(&state, &config.backend_url, &api_key, &capped_request, was_delayed, request_start, handoff_request).await
            } else {
                execute_passthrough_request(&state, &config.backend_url, &api_key, &capped_request, was_delayed, request_start).await
            }
        } else {
            let system_string = request.system.as_ref().map(|s| s.to_string());
            // Use mapper to restore original backend IDs for tool_results
            let (mut openai_messages, _id_mapping) = convert::anthropic_request_to_openai_with_mapper(
                &request.messages,
                system_string.as_deref(),
                Some(&conv.tool_id_mapper),
            );
            // id_mapping maps short_id -> original_id for future use

            // Debug: log message roles in order
            let role_order: Vec<String> = openai_messages.iter()
                .map(|m| format!("{:?}", m.role))
                .collect();
            info!("OpenAI message order: {:?}", role_order);

            // SAFEGUARD 1: Fix orphaned Tool messages.
            // Tool messages can ONLY follow Assistant messages with tool_calls (or other Tool messages).
            // If we find Tool messages without proper context, convert them to User text.
            let mut i = 0;
            let mut in_tool_response_block = false;  // True after seeing Assistant+tool_calls, stays true for Tool messages
            while i < openai_messages.len() {
                let msg = &openai_messages[i];
                match msg.role {
                    openai::Role::Assistant => {
                        in_tool_response_block = msg.tool_calls.as_ref().map(|tc| !tc.is_empty()).unwrap_or(false);
                    }
                    openai::Role::Tool => {
                        if !in_tool_response_block {
                            // Orphaned tool message - convert to User with the tool result as text
                            info!("Message order fix: orphaned Tool message at index {}, converting to User", i);
                            let content = openai_messages[i].content.clone();
                            let tool_id = openai_messages[i].tool_call_id.clone().unwrap_or_default();
                            openai_messages[i] = openai::Message {
                                role: openai::Role::User,
                                content: Some(openai::Content::Text(format!(
                                    "[Tool result for {}]: {}",
                                    tool_id,
                                    match content {
                                        Some(openai::Content::Text(t)) => t,
                                        Some(openai::Content::Parts(_)) => "[complex content]".to_string(),
                                        None => "[empty]".to_string(),
                                    }
                                ))),
                                tool_calls: None,
                                tool_call_id: None,
                                name: None,
                            };
                        }
                        // Stay in tool response block - more Tool messages can follow
                    }
                    _ => {
                        // Any other role ends the tool response block
                        in_tool_response_block = false;
                    }
                }
                i += 1;
            }

            // SAFEGUARD 2: Mistral and some OpenAI-compatible APIs require the last message
            // to be from the user or tool. If the last message is assistant, add a continuation
            // prompt to fix the conversation structure.
            if let Some(last_msg) = openai_messages.last() {
                if matches!(last_msg.role, openai::Role::Assistant) {
                    info!("Message order fix: last message was assistant, adding continuation prompt");
                    openai_messages.push(openai::Message {
                        role: openai::Role::User,
                        content: Some(openai::Content::Text("Continue.".to_string())),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                }
            }

            let openai_tools = request
                .tools
                .as_ref()
                .map(|tools| convert::anthropic_tools_to_openai(tools));

            // Cap max_tokens to prevent runaway generation
            let capped_max_tokens = request.max_tokens.min(state.max_output_tokens);
            if capped_max_tokens < request.max_tokens {
                info!("Capped max_tokens from {} to {}", request.max_tokens, capped_max_tokens);
            }

            let openai_request = openai::ChatCompletionRequest {
                model: config.model_id.clone(),
                messages: openai_messages,
                max_tokens: Some(capped_max_tokens),
                temperature: request.temperature,
                tools: openai_tools,
                stream: request.stream,
            };

            if request.stream.unwrap_or(false) {
                // Build handoff request for baton passing
                // Save conversation state before streaming (will be updated during streaming)
                state.update_conversation(conv.clone()).await;

                let handoff_request = Some(HandoffRequest {
                    original_request: request.clone(),
                    http_client: state.http_client.clone(),
                    models: state.models.clone(),
                    max_output_tokens: state.max_output_tokens,
                    continuous_config: state.continuous_config.clone(),
                    original_user_request: last_user_message.clone().unwrap_or_default(),
                    actual_model: actual_model.clone(),
                    project_status: state.project_status.clone(),
                    conversation_id: conversation_id.clone(),
                    conversations: state.conversations.clone(),
                });
                execute_streaming_request(&state, &config.backend_url, &api_key, openai_request, was_delayed, request_start, handoff_request).await
            } else {
                execute_non_streaming_request(&state, &config.backend_url, &api_key, openai_request, was_delayed, request_start).await
            }
        };

        match result {
            Ok(response) => {
                state.rate_limiter.record_response(provider, 200);
                return Ok(response);
            }
            Err(RequestError::RateLimited { wait_ms }) => {
                // Got a 429 from upstream - record it and retry
                state.rate_limiter.record_response(provider, 429);
                queue.in_backoff.store(true, std::sync::atomic::Ordering::Relaxed);

                if attempt > MAX_RETRY_ATTEMPTS {
                    warn!("Exceeded max retries after {} 429s", attempt);
                    last_error = Some(AppError::Upstream("Rate limit exceeded after retries".to_string()));
                    break;
                }

                was_delayed = true;
                let backoff_ms = wait_ms.max(1000 * (1 << attempt.min(6))); // Exponential backoff
                info!("Got 429 (attempt {}), backing off {}ms", attempt, backoff_ms);
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                continue;
            }
            Err(RequestError::Other(e)) => {
                last_error = Some(e);
                break;
            }
        }
    }

    // Clear backoff flag on exit
    queue.in_backoff.store(false, std::sync::atomic::Ordering::Relaxed);

    Err(last_error.unwrap_or_else(|| AppError::Upstream("Unknown error".to_string())))
}

/// Internal error type that distinguishes 429s from other errors
enum RequestError {
    RateLimited { wait_ms: u64 },
    Other(AppError),
}

/// Execute non-streaming request with retry support
async fn execute_non_streaming_request(
    state: &AppState,
    backend_url: &str,
    api_key: &str,
    request: openai::ChatCompletionRequest,
    was_delayed: bool,
    request_start: Instant,
) -> Result<Response, RequestError> {
    let response = state
        .http_client
        .post(backend_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| RequestError::Other(AppError::Upstream(e.to_string())))?;

    let status = response.status();

    if status == StatusCode::TOO_MANY_REQUESTS {
        // Extract retry-after if present
        let wait_ms = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .map(|s| s * 1000)
            .unwrap_or(5000);
        return Err(RequestError::RateLimited { wait_ms });
    }

    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        error!("Backend error: {} - {}", status, body);
        return Err(RequestError::Other(AppError::Upstream(format!("{}: {}", status, body))));
    }

    let openai_response: openai::ChatCompletionResponse = response
        .json()
        .await
        .map_err(|e| RequestError::Other(AppError::Upstream(e.to_string())))?;

    let choice = openai_response.choices.first()
        .ok_or_else(|| RequestError::Other(AppError::Upstream("No choices in response".to_string())))?;

    let mut content_blocks = convert::openai_message_to_anthropic_content(&choice.message);

    // Inject system reminder if request was delayed
    if was_delayed {
        let delay_secs = request_start.elapsed().as_secs();
        let reminder = format!(
            "<system-reminder>This response was queued due to rate limiting and delivered after {}s delay. The translator buffered your request to avoid failures.</system-reminder>\n\n",
            delay_secs
        );

        // Prepend to first text block or add new one
        if let Some(anthropic::ContentBlock::Text { text }) = content_blocks.first_mut() {
            *text = format!("{}{}", reminder, text);
        } else {
            content_blocks.insert(0, anthropic::ContentBlock::Text { text: reminder });
        }
    }

    let stop_reason = choice
        .finish_reason
        .map(convert::openai_finish_reason_to_anthropic);

    let input_tokens = openai_response.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0);
    let output_tokens = openai_response.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0);

    // Note: Cost tracking for non-streaming requests would require conversation_id
    // For now, cost tracking is only done for streaming requests which have access to conversation state

    let anthropic_response = anthropic::MessagesResponse {
        id: format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
        response_type: "message".to_string(),
        role: anthropic::Role::Assistant,
        content: content_blocks,
        model: request.model,
        stop_reason,
        usage: anthropic::Usage {
            input_tokens,
            output_tokens,
        },
    };

    Ok(Json(anthropic_response).into_response())
}

/// Execute passthrough request with retry support
async fn execute_passthrough_request(
    state: &AppState,
    backend_url: &str,
    api_key: &str,
    request: &anthropic::MessagesRequest,
    was_delayed: bool,
    request_start: Instant,
) -> Result<Response, RequestError> {
    let response = state
        .http_client
        .post(backend_url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| RequestError::Other(AppError::Upstream(e.to_string())))?;

    let status = response.status();

    if status == StatusCode::TOO_MANY_REQUESTS {
        let wait_ms = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .map(|s| s * 1000)
            .unwrap_or(5000);
        return Err(RequestError::RateLimited { wait_ms });
    }

    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        error!("Backend error: {} - {}", status, body);
        return Err(RequestError::Other(AppError::Upstream(format!("{}: {}", status, body))));
    }

    // If delayed, we need to parse and modify the response
    if was_delayed {
        let mut anthropic_response: anthropic::MessagesResponse = response
            .json()
            .await
            .map_err(|e| RequestError::Other(AppError::Upstream(e.to_string())))?;

        // Track costs
        // Note: Cost tracking for non-streaming passthrough requests would require conversation_id

        let delay_secs = request_start.elapsed().as_secs();
        let reminder = format!(
            "<system-reminder>This response was queued due to rate limiting and delivered after {}s delay. The translator buffered your request to avoid failures.</system-reminder>\n\n",
            delay_secs
        );

        if let Some(anthropic::ContentBlock::Text { text }) = anthropic_response.content.first_mut() {
            *text = format!("{}{}", reminder, text);
        } else {
            anthropic_response.content.insert(0, anthropic::ContentBlock::Text { text: reminder });
        }

        return Ok(Json(anthropic_response).into_response());
    }

    // No delay - just forward
    let body = response.bytes().await.map_err(|e| RequestError::Other(AppError::Upstream(e.to_string())))?;

    // Note: Cost tracking for non-streaming passthrough requests would require conversation_id

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap())
}

/// Wrap passthrough stream to detect @@packname patterns in Anthropic format
/// Accumulates text_delta content and checks for switch patterns at message_stop
/// If handoff_request is provided, will chain a follow-up request to the new model
fn wrap_passthrough_stream_for_switch_detection(
    stream: impl futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
    active_pack: Arc<RwLock<ModelPack>>,
    packs: Arc<HashMap<&'static str, ModelPack>>,
    state_file: PathBuf,
    handoff_request: Option<HandoffRequest>,
    reminder_turns: Arc<RwLock<u32>>,
) -> impl futures::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Send + 'static {
    let mut full_response_text = String::new();
    let mut buffer = String::new();
    // Track pending handoff to execute AFTER yielding original chunk
    let mut pending_handoff: Option<(ModelPack, String)> = None;
    // Track pending continuous continuation (pack, continuation_msg, previous_response)
    let mut pending_continuous: Option<(ModelPack, String, String)> = None;
    // Track pending actions display to inject after response
    let mut pending_actions_display: Option<String> = None;

    async_stream::stream! {
        let mut stream = std::pin::pin!(stream);

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(std::io::Error::new(std::io::ErrorKind::Other, e));
                    break;
                }
            };

            // Parse SSE events to extract text_delta content
            let chunk_str = String::from_utf8_lossy(&chunk);
            buffer.push_str(&chunk_str);

            // Process complete SSE events - just parse, don't yield handoff yet
            while let Some(event_end) = buffer.find("\n\n") {
                let event = buffer[..event_end].to_string();
                buffer = buffer[event_end + 2..].to_string();

                // Look for data lines containing text_delta
                for line in event.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        // Try to parse as JSON and extract text_delta
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                            // Check for text_delta in content_block_delta
                            if json.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                                if let Some(delta) = json.get("delta") {
                                    if delta.get("type").and_then(|t| t.as_str()) == Some("text_delta") {
                                        if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                            full_response_text.push_str(text);
                                        }
                                    }
                                }
                            }
                            // Check for message_stop - this is when we check for switches
                            else if json.get("type").and_then(|t| t.as_str()) == Some("message_stop") {
                                // Debug: log that we reached message_stop
                                info!("🛑 message_stop received, checking for handoff in {} chars: '{}'",
                                    full_response_text.len(),
                                    if full_response_text.len() > 100 { &full_response_text[..100] } else { &full_response_text });

                                // Get current pack name for self-call detection
                                let current_pack_name = {
                                    let current = active_pack.read().await;
                                    current.name.to_string()
                                };

                                // Check for @@packname pattern (skips self-calls if another pack follows)
                                if let Some((pack_name, handoff_msg)) = detect_model_switch_with_current(&full_response_text, Some(&current_pack_name)) {
                                    if let Some(new_pack) = packs.get(pack_name.as_str()) {
                                        let is_self_call = new_pack.name == current_pack_name;

                                        if !is_self_call {
                                            // Update active pack
                                            let mut current = active_pack.write().await;
                                            *current = new_pack.clone();

                                            // Set reminder persistence so next 2 turns get the system prompt
                                            *reminder_turns.write().await = 2;

                                            // Persist for restart
                                            if let Some(parent) = state_file.parent() {
                                                let _ = tokio::fs::create_dir_all(parent).await;
                                            }
                                            let _ = tokio::fs::write(&state_file, new_pack.name).await;
                                        }

                                        // Schedule handoff regardless of self-call (model wants to continue with this task)
                                        // Use empty string if no message provided
                                        let msg = handoff_msg.clone().unwrap_or_else(|| "Continue the task.".to_string());
                                        if is_self_call {
                                            info!("@@{}: Self-handoff on {} with message: '{}'",
                                                pack_name, current_pack_name, msg);
                                        } else {
                                            info!("@@{}: Model auto-switched to pack {} with handoff: '{}' (reminder for 2 turns)",
                                                pack_name, new_pack.name, msg);
                                        }
                                        // Schedule handoff for AFTER we yield the original chunk
                                        pending_handoff = Some((new_pack.clone(), msg));
                                    } else {
                                        info!("@@{}: Unknown pack, ignoring", pack_name);
                                    }
                                }

                                // Check for @@swarm=on from model response (can enable but not change limits)
                                if full_response_text.contains("@@swarm=on") || full_response_text.contains("@@swarm ") {
                                    if let Some(ref hr) = handoff_request {
                                        let mut conversations = hr.conversations.write().await;
                                        if let Some(conv) = conversations.get_mut(&hr.conversation_id) {
                                            if conv.swarm_config.swarm_mode == SwarmMode::Off {
                                                conv.swarm_config.swarm_mode = SwarmMode::On;
                                                info!("@@swarm: Model enabled swarm mode (cap: {})", conv.swarm_config.max_workers);
                                            } else {
                                                info!("@@swarm: Swarm already enabled, ignoring (model cannot change limits)");
                                            }
                                        }
                                    }
                                }

                                // Parse actions from response if present
                                if full_response_text.contains("actions:") {
                                    if let Some(ref hr) = handoff_request {
                                        let (_before, parsed_actions, _after) = parse_actions_yaml(&full_response_text);
                                        if !parsed_actions.is_empty() {
                                            let action_count = parsed_actions.len();

                                            // Build formatted display
                                            let mut action_list = String::from("\n\n📋 **Available Actions:**\n\n");
                                            for action in &parsed_actions {
                                                action_list.push_str(&format!("  **{}. {}**\n", action.num, action.label));
                                                if !action.description.is_empty() {
                                                    action_list.push_str(&format!("     {}\n", action.description));
                                                }
                                                action_list.push('\n');
                                            }
                                            action_list.push_str("\n💡 Use `@action N` or `@actions 1,2,3` to select tasks for execution.\n");
                                            pending_actions_display = Some(action_list);

                                            // Store in conversation state
                                            let mut conversations = hr.conversations.write().await;
                                            if let Some(conv) = conversations.get_mut(&hr.conversation_id) {
                                                conv.pending_actions = parsed_actions;
                                                info!("[{}] Parsed {} actions from model response", hr.conversation_id, action_count);
                                            }
                                        }
                                    }
                                }

                                // If no handoff scheduled, check continuous mode
                                if pending_handoff.is_none() {
                                    if let Some(ref hr) = handoff_request {
                                        // Look up conversation state
                                        let conv_state = {
                                            let conversations = hr.conversations.read().await;
                                            conversations.get(&hr.conversation_id).cloned()
                                        };
                                        let continuous_enabled = conv_state.as_ref()
                                            .map(|c| !matches!(c.continuous_mode, ContinuousMode::Off))
                                            .unwrap_or(false);
                                        if continuous_enabled {
                                            info!("CONTINUOUS: No handoff, running verification...");

                                            // Run verification through local Ollama models
                                            let strict = conv_state.as_ref()
                                                .map(|c| c.strict_mode.clone())
                                                .unwrap_or(StrictMode::Off);
                                            match run_continuous_verification(
                                                &hr.http_client,
                                                &hr.continuous_config,
                                                &full_response_text,
                                                &hr.original_user_request,
                                                &hr.actual_model,
                                                hr.project_status.clone(),
                                                &strict,
                                            ).await {
                                                Ok(result) => {
                                                    info!("CONTINUOUS: Verification result: {:?}", result.status);
                                                    if result.status != ContinuousStatus::Done {
                                                        // Schedule continuation with the same pack
                                                        let current_pack = active_pack.read().await.clone();
                                                        let continuation_msg = synthesize_continuation_message(&result);
                                                        info!("CONTINUOUS: Scheduling continuation: '{}'", continuation_msg);
                                                        pending_continuous = Some((current_pack, continuation_msg, full_response_text.clone()));
                                                    } else {
                                                        info!("CONTINUOUS: Task complete, stopping loop");
                                                    }
                                                }
                                                Err(e) => {
                                                    warn!("CONTINUOUS: Verification failed: {}, stopping loop", e);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // FIRST: Pass through the original chunk
            yield Ok(chunk);

            // THEN: Inject formatted action list if we parsed any
            if let Some(action_list) = pending_actions_display.take() {
                // Inject as additional content block
                let escaped = action_list.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
                yield Ok(bytes::Bytes::from(format!(
                    "event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":99,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
                     event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":99,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{}\"}}}}\n\n\
                     event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":99}}\n\n",
                    escaped
                )));
            }

            // THEN: Execute pending handoff if any (takes priority over continuous)
            if let Some((new_pack, msg)) = pending_handoff.take() {
                if let Some(hr) = &handoff_request {
                    info!("🎭 BATON PASS (passthrough): Invoking {} with message: '{}'", new_pack.name, msg);

                    // Build followup messages
                    let mut followup_messages = hr.original_request.messages.clone();

                    // Add the current model's response
                    followup_messages.push(anthropic::Message {
                        role: anthropic::Role::Assistant,
                        content: anthropic::Content::Text(full_response_text.clone()),
                    });

                    // Add the handoff message as a new user turn (clean, no injection)
                    followup_messages.push(anthropic::Message {
                        role: anthropic::Role::User,
                        content: anthropic::Content::Text(msg.clone()),
                    });

                    // Map the model through the new pack
                    let actual_model = new_pack.map_model(&hr.original_request.model);

                    // Look up config for the new model
                    if let Some(config) = hr.models.get(actual_model) {
                        if let Ok(api_key) = env::var(&config.api_key_env) {
                            // New message_start for the handoff response
                            let msg_start = serde_json::json!({
                                "type": "message_start",
                                "message": {
                                    "id": format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [],
                                    "model": actual_model,
                                    "stop_reason": null,
                                    "usage": {"input_tokens": 0, "output_tokens": 0}
                                }
                            });
                            yield Ok(bytes::Bytes::from(format!("event: message_start\ndata: {}\n\n", msg_start)));

                            // Emit colorized @packname indicator as first content block
                            // ANSI: \x1b[1;36m = bold cyan, \x1b[0m = reset
                            let pack_indicator = format!("\x1b[1;36m@{}\x1b[0m ", new_pack.name);
                            let indicator_block_start = serde_json::json!({
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text", "text": ""}
                            });
                            yield Ok(bytes::Bytes::from(format!("event: content_block_start\ndata: {}\n\n", indicator_block_start)));
                            let indicator_delta = serde_json::json!({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": pack_indicator}
                            });
                            yield Ok(bytes::Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", indicator_delta)));

                            // Convert ALL tool blocks to text summaries for cross-model handoff
                            // The receiving model didn't make those tool calls, so its backend would reject them
                            convert_tool_blocks_to_text(&mut followup_messages);
                            strip_invalid_thinking_blocks(&mut followup_messages);

                            // Build the followup request with handoff-specific reminder
                            let handoff_reminder = format!(
                                "\n\n<system-reminder>You have successfully been switched to - you are now {0} (model {1}). Please proceed under that understanding.</system-reminder>",
                                new_pack.name, actual_model
                            );
                            let followup_request = anthropic::MessagesRequest {
                                model: actual_model.to_string(),
                                messages: followup_messages,
                                max_tokens: hr.original_request.max_tokens.min(hr.max_output_tokens),
                                system: {
                                    Some(match &hr.original_request.system {
                                        Some(s) => anthropic::SystemPrompt::Text(format!("{}{}", s.to_string(), handoff_reminder)),
                                        None => anthropic::SystemPrompt::Text(handoff_reminder.trim_start().to_string()),
                                    })
                                },
                                temperature: hr.original_request.temperature,
                                tools: hr.original_request.tools.clone(),
                                stream: Some(true),
                                thinking: hr.original_request.thinking.clone(),
                                extra: Default::default(),
                            };

                            // Make the request to the new model (passthrough mode)
                            if config.passthrough {
                                match hr.http_client
                                    .post(&config.backend_url)
                                    .header("x-api-key", &api_key)
                                    .header("anthropic-version", "2023-06-01")
                                    .header("Content-Type", "application/json")
                                    .json(&followup_request)
                                    .send()
                                    .await
                                {
                                    Ok(response) if response.status().is_success() => {
                                        let mut followup_stream = response.bytes_stream();
                                        let mut buf = String::new();

                                        while let Some(chunk) = followup_stream.next().await {
                                            if let Ok(bytes) = chunk {
                                                let chunk_str = String::from_utf8_lossy(&bytes);
                                                buf.push_str(&chunk_str);

                                                while let Some(end) = buf.find("\n\n") {
                                                    let event = buf[..end].to_string();
                                                    buf = buf[end + 2..].to_string();

                                                    // Skip message_start events (we sent our own)
                                                    if !event.contains("\"type\":\"message_start\"") && !event.contains("\"type\": \"message_start\"") {
                                                        yield Ok(bytes::Bytes::from(format!("{}\n\n", event)));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Ok(response) => {
                                        let status = response.status();
                                        let body = response.text().await.unwrap_or_default();
                                        error!("Handoff request failed: {} - {}", status, body);
                                    }
                                    Err(e) => {
                                        error!("Handoff request error: {}", e);
                                    }
                                }
                            } else {
                                // Translation mode for handoff
                                let system_string = followup_request.system.as_ref().map(|s| s.to_string());
                                let (openai_messages, _id_mapping) = convert::anthropic_request_to_openai(
                                    &followup_request.messages,
                                    system_string.as_deref(),
                                );
                                let openai_tools = followup_request.tools.as_ref()
                                    .map(|tools| convert::anthropic_tools_to_openai(tools));

                                // Debug: log tool IDs in the request
                                for (i, msg) in openai_messages.iter().enumerate() {
                                    if let Some(tcs) = &msg.tool_calls {
                                        for tc in tcs {
                                            info!("HANDOFF msg[{}] tool_call id={}", i, tc.id);
                                        }
                                    }
                                    if let Some(id) = &msg.tool_call_id {
                                        info!("HANDOFF msg[{}] tool_result id={}", i, id);
                                    }
                                }

                                let openai_request = openai::ChatCompletionRequest {
                                    model: config.model_id.clone(),
                                    messages: openai_messages,
                                    max_tokens: Some(followup_request.max_tokens),
                                    temperature: followup_request.temperature,
                                    tools: openai_tools,
                                    stream: Some(true),
                                };

                                match hr.http_client
                                    .post(&config.backend_url)
                                    .header("Authorization", format!("Bearer {}", api_key))
                                    .header("Content-Type", "application/json")
                                    .json(&openai_request)
                                    .send()
                                    .await
                                {
                                    Ok(response) if response.status().is_success() => {
                                        let mut followup_stream = response.bytes_stream();
                                        let mut buf = String::new();
                                        let mut handoff_block_index: i32 = 0; // Start at 0 since we emit indicator first
                                        let mut handoff_text_started = false;

                                        // Emit colorized @packname indicator first (translation mode)
                                        let pack_indicator = format!("\x1b[1;36m@{}\x1b[0m ", new_pack.name);
                                        let indicator_block_start = serde_json::json!({
                                            "type": "content_block_start",
                                            "index": 0,
                                            "content_block": {"type": "text", "text": ""}
                                        });
                                        yield Ok(bytes::Bytes::from(format!("event: content_block_start\ndata: {}\n\n", indicator_block_start)));
                                        let indicator_delta = serde_json::json!({
                                            "type": "content_block_delta",
                                            "index": 0,
                                            "delta": {"type": "text_delta", "text": pack_indicator}
                                        });
                                        yield Ok(bytes::Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", indicator_delta)));

                                        while let Some(chunk) = followup_stream.next().await {
                                            if let Ok(bytes) = chunk {
                                                buf.push_str(&String::from_utf8_lossy(&bytes));

                                                while let Some(line_end) = buf.find("\n\n") {
                                                    let line = buf[..line_end].to_string();
                                                    buf = buf[line_end + 2..].to_string();

                                                    if let Some(data) = line.strip_prefix("data: ") {
                                                        if data.trim() == "[DONE]" { continue; }

                                                        if let Ok(chunk) = serde_json::from_str::<openai::ChatCompletionChunk>(data) {
                                                            for choice in &chunk.choices {
                                                                if let Some(content) = &choice.delta.content {
                                                                    if !content.is_empty() {
                                                                        if !handoff_text_started {
                                                                            handoff_block_index += 1;
                                                                            let block_start = serde_json::json!({
                                                                                "type": "content_block_start",
                                                                                "index": handoff_block_index,
                                                                                "content_block": {"type": "text", "text": ""}
                                                                            });
                                                                            yield Ok(bytes::Bytes::from(format!("event: content_block_start\ndata: {}\n\n", block_start)));
                                                                            handoff_text_started = true;
                                                                        }

                                                                        let delta_event = serde_json::json!({
                                                                            "type": "content_block_delta",
                                                                            "index": handoff_block_index,
                                                                            "delta": {"type": "text_delta", "text": content}
                                                                        });
                                                                        yield Ok(bytes::Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", delta_event)));
                                                                    }
                                                                }

                                                                if choice.finish_reason.is_some() {
                                                                    if handoff_text_started {
                                                                        let block_stop = serde_json::json!({
                                                                            "type": "content_block_stop",
                                                                            "index": handoff_block_index
                                                                        });
                                                                        yield Ok(bytes::Bytes::from(format!("event: content_block_stop\ndata: {}\n\n", block_stop)));
                                                                    }

                                                                    let msg_delta = serde_json::json!({
                                                                        "type": "message_delta",
                                                                        "delta": {"stop_reason": "end_turn"},
                                                                        "usage": {"output_tokens": 0}
                                                                    });
                                                                    yield Ok(bytes::Bytes::from(format!("event: message_delta\ndata: {}\n\n", msg_delta)));

                                                                    let msg_stop = serde_json::json!({"type": "message_stop"});
                                                                    yield Ok(bytes::Bytes::from(format!("event: message_stop\ndata: {}\n\n", msg_stop)));
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Ok(response) => {
                                        let status = response.status();
                                        let body = response.text().await.unwrap_or_default();
                                        error!("Handoff request failed: {} - {}", status, body);
                                    }
                                    Err(e) => {
                                        error!("Handoff request error: {}", e);
                                    }
                                }
                            }
                        } else {
                            error!("Handoff failed: API key {} not set", config.api_key_env);
                        }
                    } else {
                        error!("Handoff failed: model {} not found in registry", actual_model);
                    }
                }
            }

            // Execute pending continuous continuation if any (only if no handoff was executed)
            if let Some((cont_pack, continuation_msg, prev_response)) = pending_continuous.take() {
                if let Some(hr) = &handoff_request {
                    info!("CONTINUOUS: Executing continuation with pack {}", cont_pack.name);

                    // Build followup messages
                    let mut followup_messages = hr.original_request.messages.clone();

                    // Add the model's previous response
                    followup_messages.push(anthropic::Message {
                        role: anthropic::Role::Assistant,
                        content: anthropic::Content::Text(prev_response),
                    });

                    // Add the continuation message as a new user turn
                    followup_messages.push(anthropic::Message {
                        role: anthropic::Role::User,
                        content: anthropic::Content::Text(continuation_msg.clone()),
                    });

                    // Map the model through the pack
                    let actual_model = cont_pack.map_model(&hr.original_request.model);

                    // Look up config for the model
                    if let Some(config) = hr.models.get(actual_model) {
                        if let Ok(api_key) = env::var(&config.api_key_env) {
                            // New message_start for the continuation
                            let msg_start = serde_json::json!({
                                "type": "message_start",
                                "message": {
                                    "id": format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [],
                                    "model": actual_model,
                                    "stop_reason": null,
                                    "usage": {"input_tokens": 0, "output_tokens": 0}
                                }
                            });
                            yield Ok(bytes::Bytes::from(format!("event: message_start\ndata: {}\n\n", msg_start)));

                            // Emit continuation indicator
                            let cont_indicator = "\x1b[1;33m[continuing...]\x1b[0m\n".to_string();
                            let indicator_block_start = serde_json::json!({
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text", "text": ""}
                            });
                            yield Ok(bytes::Bytes::from(format!("event: content_block_start\ndata: {}\n\n", indicator_block_start)));
                            let indicator_delta = serde_json::json!({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": cont_indicator}
                            });
                            yield Ok(bytes::Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", indicator_delta)));

                            // Convert tool blocks for clean continuation
                            convert_tool_blocks_to_text(&mut followup_messages);
                            strip_invalid_thinking_blocks(&mut followup_messages);

                            // Build the followup request
                            let followup_request = anthropic::MessagesRequest {
                                model: actual_model.to_string(),
                                messages: followup_messages,
                                max_tokens: hr.original_request.max_tokens.min(hr.max_output_tokens),
                                system: hr.original_request.system.clone(),
                                temperature: hr.original_request.temperature,
                                tools: hr.original_request.tools.clone(),
                                stream: Some(true),
                                thinking: hr.original_request.thinking.clone(),
                                extra: Default::default(),
                            };

                            // Make the continuation request (passthrough mode for Anthropic)
                            if config.passthrough {
                                match hr.http_client
                                    .post(&config.backend_url)
                                    .header("x-api-key", &api_key)
                                    .header("anthropic-version", "2023-06-01")
                                    .header("Content-Type", "application/json")
                                    .json(&followup_request)
                                    .send()
                                    .await
                                {
                                    Ok(response) if response.status().is_success() => {
                                        let mut followup_stream = response.bytes_stream();

                                        while let Some(chunk) = followup_stream.next().await {
                                            if let Ok(bytes) = chunk {
                                                // Just forward the chunks
                                                yield Ok(bytes);
                                            }
                                        }

                                        // Note: In a full implementation, we'd recursively check
                                        // continuous mode here too. For now, one continuation per turn.
                                    }
                                    Ok(response) => {
                                        let status = response.status();
                                        let body = response.text().await.unwrap_or_default();
                                        error!("Continuation request failed: {} - {}", status, body);
                                    }
                                    Err(e) => {
                                        error!("Continuation request error: {}", e);
                                    }
                                }
                            } else {
                                // For non-passthrough (OpenAI-compatible), we'd need translation
                                // This is more complex - for now log and skip
                                warn!("CONTINUOUS: Non-passthrough continuation not yet implemented");
                            }
                        } else {
                            error!("Continuation failed: API key {} not set", config.api_key_env);
                        }
                    } else {
                        error!("Continuation failed: model {} not found in registry", actual_model);
                    }
                }
            }
        }
    }
}

/// Execute passthrough streaming with retry support
async fn execute_passthrough_streaming(
    state: &AppState,
    backend_url: &str,
    api_key: &str,
    request: &anthropic::MessagesRequest,
    was_delayed: bool,
    request_start: Instant,
    handoff_request: Option<HandoffRequest>,
) -> Result<Response, RequestError> {
    let response = state
        .http_client
        .post(backend_url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| RequestError::Other(AppError::Upstream(e.to_string())))?;

    let status = response.status();

    if status == StatusCode::TOO_MANY_REQUESTS {
        let wait_ms = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .map(|s| s * 1000)
            .unwrap_or(5000);
        return Err(RequestError::RateLimited { wait_ms });
    }

    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        error!("Backend error: {} - {}", status, body);
        return Err(RequestError::Other(AppError::Upstream(format!("{}: {}", status, body))));
    }

    let stream = response.bytes_stream();

    // Get conversation state from handoff request (if available)
    let (active_pack, reminder_turns) = if let Some(ref hr) = handoff_request {
        let conversations = hr.conversations.read().await;
        if let Some(conv) = conversations.get(&hr.conversation_id) {
            (conv.active_pack.clone(), conv.reminder_turns_remaining)
        } else {
            (state.default_pack.clone(), 0)
        }
    } else {
        (state.default_pack.clone(), 0)
    };
    // Wrap in Arc<RwLock<>> for stream processing compatibility
    let active_pack = Arc::new(RwLock::new(active_pack));
    let reminder_turns = Arc::new(RwLock::new(reminder_turns));
    let packs = state.packs.clone();
    let state_file = state.state_file.clone();

    // Wrap stream to detect @@packname patterns in passthrough mode
    let wrapped_stream = wrap_passthrough_stream_for_switch_detection(
        stream,
        active_pack,
        packs,
        state_file,
        handoff_request,
        reminder_turns,
    );

    // If delayed, inject a text delta at the start
    if was_delayed {
        let delay_secs = request_start.elapsed().as_secs();
        let reminder = format!(
            "<system-reminder>This response was queued due to rate limiting and delivered after {}s delay.</system-reminder>\n\n",
            delay_secs
        );

        let prefix_stream = futures::stream::once(async move {
            // Inject a text delta with the reminder
            let delta = serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": reminder}
            });
            Ok::<_, std::io::Error>(bytes::Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", delta)))
        });

        let combined = prefix_stream.chain(wrapped_stream);

        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from_stream(combined))
            .unwrap());
    }

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(wrapped_stream))
        .unwrap())
}

/// Execute streaming request with retry support
async fn execute_streaming_request(
    state: &AppState,
    backend_url: &str,
    api_key: &str,
    request: openai::ChatCompletionRequest,
    was_delayed: bool,
    request_start: Instant,
    handoff_request: Option<HandoffRequest>,
) -> Result<Response, RequestError> {
    let response = state
        .http_client
        .post(backend_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| RequestError::Other(AppError::Upstream(e.to_string())))?;

    let status = response.status();

    if status == StatusCode::TOO_MANY_REQUESTS {
        let wait_ms = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .map(|s| s * 1000)
            .unwrap_or(5000);
        return Err(RequestError::RateLimited { wait_ms });
    }

    if !response.status().is_success() {
        let body = response.text().await.unwrap_or_default();
        error!("Backend error: {} - {}", status, body);
        return Err(RequestError::Other(AppError::Upstream(format!("{}: {}", status, body))));
    }

    let stream = response.bytes_stream();

    // Get conversation state from handoff request (if available)
    let (active_pack, reminder_turns) = if let Some(ref hr) = handoff_request {
        let conversations = hr.conversations.read().await;
        if let Some(conv) = conversations.get(&hr.conversation_id) {
            (conv.active_pack.clone(), conv.reminder_turns_remaining)
        } else {
            (state.default_pack.clone(), 0)
        }
    } else {
        (state.default_pack.clone(), 0)
    };
    // Wrap in Arc<RwLock<>> for stream processing compatibility
    let active_pack = Arc::new(RwLock::new(active_pack));
    let reminder_turns = Arc::new(RwLock::new(reminder_turns));
    let packs = state.packs.clone();
    let state_file = state.state_file.clone();

    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(convert_streaming_response(
            stream,
            request.model,
            was_delayed,
            request_start,
            active_pack,
            packs,
            state_file,
            handoff_request,
            reminder_turns,
        )))
        .unwrap();

    Ok(response)
}

/// Handoff request for baton passing between models
#[derive(Clone)]
struct HandoffRequest {
    /// The original request (needed to reconstruct conversation)
    original_request: anthropic::MessagesRequest,
    /// HTTP client for making follow-up requests
    http_client: reqwest::Client,
    /// Model registry for looking up configs
    models: Arc<HashMap<String, ModelConfig>>,
    /// Maximum output tokens
    max_output_tokens: u32,
    /// Continuous mode config (Ollama endpoints, models)
    continuous_config: Arc<ContinuousConfig>,
    /// Original user request text (for verification context)
    original_user_request: String,
    /// The actual model being used (after pack mapping)
    actual_model: String,
    /// Project build/test status (global)
    project_status: Arc<RwLock<ProjectStatus>>,
    /// Conversation ID for looking up per-conversation state
    conversation_id: String,
    /// Reference to all conversations for looking up state
    conversations: Arc<RwLock<HashMap<String, ConversationState>>>,
}

/// Convert OpenAI streaming chunks to Anthropic format
/// Also detects @@packname patterns in output and switches packs
fn convert_streaming_response(
    stream: impl futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
    model: String,
    was_delayed: bool,
    request_start: Instant,
    active_pack: Arc<RwLock<ModelPack>>,
    packs: Arc<HashMap<&'static str, ModelPack>>,
    state_file: PathBuf,
    handoff_request: Option<HandoffRequest>,
    reminder_turns: Arc<RwLock<u32>>,
) -> impl futures::Stream<Item = Result<String, std::io::Error>> + Send + 'static {
    let mut current_block_index: i32 = -1;
    let mut text_block_started = false;
    let mut thinking_block_started = false;
    let mut tool_blocks: HashMap<usize, ToolBlockState> = HashMap::new();
    // Accumulate full response text to detect @@packname at end
    let mut full_response_text = String::new();
    let mut buffer = String::new();
    let mut injected_reminder = false;

    async_stream::stream! {
        // Send message_start first
        let msg_start = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        });
        yield Ok(format!("event: message_start\ndata: {}\n\n", msg_start));

        let mut stream = std::pin::pin!(stream);

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Ok(format!("event: error\ndata: {{\"type\": \"error\", \"error\": {{\"type\": \"api_error\", \"message\": \"{}\"}}}}\n\n", e));
                    break;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find("\n\n") {
                let line = buffer[..line_end].to_string();
                buffer = buffer[line_end + 2..].to_string();

                if let Some(data) = line.strip_prefix("data: ") {
                    if data.trim() == "[DONE]" {
                        info!("🏁 [DONE] marker received, full_response_text len: {}", full_response_text.len());
                        continue;
                    }

                    if let Ok(chunk) = serde_json::from_str::<openai::ChatCompletionChunk>(data) {
                        for choice in &chunk.choices {
                            let delta = &choice.delta;

                            // Handle Mistral's reasoning_content (thinking) - format like Claude Code
                            if let Some(reasoning) = &delta.reasoning_content {
                                if !reasoning.is_empty() {
                                    if !thinking_block_started {
                                        current_block_index += 1;
                                        // Start with the "∴ Thinking…" header in italic
                                        let block_start = serde_json::json!({
                                            "type": "content_block_start",
                                            "index": current_block_index,
                                            "content_block": {"type": "text", "text": ""}
                                        });
                                        yield Ok(format!("event: content_block_start\ndata: {}\n\n", block_start));

                                        // Emit the thinking header
                                        let header = serde_json::json!({
                                            "type": "content_block_delta",
                                            "index": current_block_index,
                                            "delta": {"type": "text_delta", "text": "*∴ Thinking…*\n\n*"}
                                        });
                                        yield Ok(format!("event: content_block_delta\ndata: {}\n\n", header));
                                        thinking_block_started = true;
                                    }

                                    let delta_event = serde_json::json!({
                                        "type": "content_block_delta",
                                        "index": current_block_index,
                                        "delta": {"type": "text_delta", "text": reasoning}
                                    });
                                    yield Ok(format!("event: content_block_delta\ndata: {}\n\n", delta_event));
                                }
                            }

                            if let Some(content) = &delta.content {
                                if !content.is_empty() {
                                    // Close thinking block if it was open
                                    if thinking_block_started && !text_block_started {
                                        // Close the italic and add separator
                                        let close_thinking = serde_json::json!({
                                            "type": "content_block_delta",
                                            "index": current_block_index,
                                            "delta": {"type": "text_delta", "text": "*\n\n"}
                                        });
                                        yield Ok(format!("event: content_block_delta\ndata: {}\n\n", close_thinking));

                                        // Stop thinking block
                                        let block_stop = serde_json::json!({
                                            "type": "content_block_stop",
                                            "index": current_block_index
                                        });
                                        yield Ok(format!("event: content_block_stop\ndata: {}\n\n", block_stop));
                                    }

                                    if !text_block_started {
                                        current_block_index += 1;
                                        let block_start = serde_json::json!({
                                            "type": "content_block_start",
                                            "index": current_block_index,
                                            "content_block": {"type": "text", "text": ""}
                                        });
                                        yield Ok(format!("event: content_block_start\ndata: {}\n\n", block_start));
                                        text_block_started = true;

                                        // Inject delay reminder at start of text
                                        if was_delayed && !injected_reminder {
                                            let delay_secs = request_start.elapsed().as_secs();
                                            let reminder = format!(
                                                "<system-reminder>Response queued {}s due to rate limiting.</system-reminder>\n\n",
                                                delay_secs
                                            );
                                            let delta_event = serde_json::json!({
                                                "type": "content_block_delta",
                                                "index": current_block_index,
                                                "delta": {"type": "text_delta", "text": reminder}
                                            });
                                            yield Ok(format!("event: content_block_delta\ndata: {}\n\n", delta_event));
                                            injected_reminder = true;
                                        }
                                    }

                                    // Accumulate for @@packname detection
                                    full_response_text.push_str(content);

                                    let delta_event = serde_json::json!({
                                        "type": "content_block_delta",
                                        "index": current_block_index,
                                        "delta": {"type": "text_delta", "text": content}
                                    });
                                    yield Ok(format!("event: content_block_delta\ndata: {}\n\n", delta_event));
                                }
                            }

                            if let Some(tool_calls) = &delta.tool_calls {
                                for tc in tool_calls {
                                    let tc_index = tc.index;

                                    if !tool_blocks.contains_key(&tc_index) {
                                        if text_block_started {
                                            let block_stop = serde_json::json!({
                                                "type": "content_block_stop",
                                                "index": current_block_index
                                            });
                                            yield Ok(format!("event: content_block_stop\ndata: {}\n\n", block_stop));
                                            text_block_started = false;
                                        }

                                        current_block_index += 1;
                                        // Shorten tool ID and record mapping for round-trip
                                        // When Claude sends back tool_result, we'll use the mapper to restore the original backend ID
                                        let raw_tool_id = tc.id.clone().unwrap_or_else(|| {
                                            uuid::Uuid::new_v4().to_string().replace("-", "")[..9].to_string()
                                        });

                                        // Use mapper to shorten ID and record the mapping
                                        let tool_id = if let Some(ref hr) = handoff_request {
                                            let mut conversations = hr.conversations.write().await;
                                            if let Some(conv) = conversations.get_mut(&hr.conversation_id) {
                                                conv.tool_id_mapper.to_short(&raw_tool_id)
                                            } else {
                                                convert::shorten_tool_id(&raw_tool_id)
                                            }
                                        } else {
                                            convert::shorten_tool_id(&raw_tool_id)
                                        };
                                        let tool_name = tc.function.as_ref().and_then(|f| f.name.clone()).unwrap_or_default();

                                        tool_blocks.insert(tc_index, ToolBlockState {
                                            block_index: current_block_index,
                                            id: tool_id.clone(),
                                            name: tool_name.clone(),
                                        });

                                        let block_start = serde_json::json!({
                                            "type": "content_block_start",
                                            "index": current_block_index,
                                            "content_block": {
                                                "type": "tool_use",
                                                "id": tool_id,
                                                "name": tool_name,
                                                "input": {}
                                            }
                                        });
                                        yield Ok(format!("event: content_block_start\ndata: {}\n\n", block_start));
                                    }

                                    if let Some(func) = &tc.function {
                                        if let Some(args) = &func.arguments {
                                            if !args.is_empty() {
                                                let tb = &tool_blocks[&tc_index];
                                                let delta_event = serde_json::json!({
                                                    "type": "content_block_delta",
                                                    "index": tb.block_index,
                                                    "delta": {"type": "input_json_delta", "partial_json": args}
                                                });
                                                yield Ok(format!("event: content_block_delta\ndata: {}\n\n", delta_event));
                                            }
                                        }
                                    }
                                }
                            }

                            if let Some(finish_reason) = &choice.finish_reason {
                                if text_block_started {
                                    let block_stop = serde_json::json!({
                                        "type": "content_block_stop",
                                        "index": current_block_index
                                    });
                                    yield Ok(format!("event: content_block_stop\ndata: {}\n\n", block_stop));
                                }

                                for (_, tb) in &tool_blocks {
                                    let block_stop = serde_json::json!({
                                        "type": "content_block_stop",
                                        "index": tb.block_index
                                    });
                                    yield Ok(format!("event: content_block_stop\ndata: {}\n\n", block_stop));
                                }

                                let stop_reason = match finish_reason {
                                    openai::FinishReason::Stop => "end_turn",
                                    openai::FinishReason::Length => "max_tokens",
                                    openai::FinishReason::ToolCalls => "tool_use",
                                    openai::FinishReason::ContentFilter => "end_turn",
                                };

                                let msg_delta = serde_json::json!({
                                    "type": "message_delta",
                                    "delta": {"stop_reason": stop_reason},
                                    "usage": {"output_tokens": 0}
                                });
                                yield Ok(format!("event: message_delta\ndata: {}\n\n", msg_delta));

                                // Check for @@packname pattern in response to auto-switch
                                // Pattern: @@packname (e.g., @@claude, @@glm, @@mistral)
                                info!("🛑 finish_reason received (translation), checking for handoff in {} chars: '{}'",
                                    full_response_text.len(),
                                    if full_response_text.len() > 100 { &full_response_text[..100] } else { &full_response_text });
                                let mut should_handoff = false;
                                let mut handoff_pack: Option<ModelPack> = None;
                                let mut handoff_message: Option<String> = None;

                                // Get current pack name for self-call detection
                                let current_pack_name = {
                                    let current = active_pack.read().await;
                                    current.name.to_string()
                                };

                                // Check for @@packname pattern (skips self-calls if another pack follows)
                                if let Some((pack_name, handoff_msg)) = detect_model_switch_with_current(&full_response_text, Some(&current_pack_name)) {
                                    if let Some(new_pack) = packs.get(pack_name.as_str()) {
                                        let is_self_call = new_pack.name == current_pack_name;

                                        if !is_self_call {
                                            // Update active pack
                                            let mut current = active_pack.write().await;
                                            *current = new_pack.clone();

                                            // Set reminder persistence so next 2 turns get the system prompt
                                            *reminder_turns.write().await = 2;

                                            // Persist for restart
                                            if let Some(parent) = state_file.parent() {
                                                let _ = tokio::fs::create_dir_all(parent).await;
                                            }
                                            let _ = tokio::fs::write(&state_file, new_pack.name).await;
                                        }

                                        // Schedule handoff regardless of self-call (model wants to continue with this task)
                                        // Use default message if none provided
                                        let msg = handoff_msg.clone().unwrap_or_else(|| "Continue the task.".to_string());
                                        if is_self_call {
                                            info!("@@{}: Self-handoff on {} with message: '{}'",
                                                pack_name, current_pack_name, msg);
                                        } else {
                                            info!("@@{}: Model auto-switched to pack {} with handoff: '{}' (reminder for 2 turns)",
                                                pack_name, new_pack.name, msg);
                                        }
                                        should_handoff = true;
                                        handoff_pack = Some(new_pack.clone());
                                        handoff_message = Some(msg);
                                    } else {
                                        info!("@@{}: Unknown pack, ignoring", pack_name);
                                    }
                                }

                                let msg_stop = serde_json::json!({"type": "message_stop"});
                                yield Ok(format!("event: message_stop\ndata: {}\n\n", msg_stop));

                                // BATON PASSING: If there's a handoff message and request context, call the new model
                                if should_handoff {
                                    if let (Some(hr), Some(new_pack), Some(msg)) = (&handoff_request, handoff_pack, handoff_message) {
                                        info!("🎭 BATON PASS: Invoking {} with message: '{}'", new_pack.name, msg);

                                        // Build the follow-up request with conversation history + this model's response
                                        let mut followup_messages = hr.original_request.messages.clone();

                                        // Add the current model's response (what we just streamed)
                                        followup_messages.push(anthropic::Message {
                                            role: anthropic::Role::Assistant,
                                            content: anthropic::Content::Text(full_response_text.clone()),
                                        });

                                        // Add the handoff message as a new user turn (clean, no injection)
                                        followup_messages.push(anthropic::Message {
                                            role: anthropic::Role::User,
                                            content: anthropic::Content::Text(msg.clone()),
                                        });

                                        // Convert ALL tool blocks to text summaries for cross-model handoff
                                        // The receiving model didn't make those tool calls, so its backend would reject them
                                        convert_tool_blocks_to_text(&mut followup_messages);
                                        strip_invalid_thinking_blocks(&mut followup_messages);

                                        // Map the model through the new pack
                                        let actual_model = new_pack.map_model(&hr.original_request.model);

                                        // Handoff reminder goes in system prompt (same format as passthrough path)
                                        let handoff_instruction = format!(
                                            "\n\n<system-reminder>You have successfully been switched to - you are now {0} (model {1}). Please proceed under that understanding.</system-reminder>",
                                            new_pack.name, actual_model
                                        );

                                        // Look up config for the new model
                                        if let Some(config) = hr.models.get(actual_model) {
                                            if let Ok(api_key) = env::var(&config.api_key_env) {
                                                // Emit a separator to show model switch
                                                yield Ok(format!("\n"));

                                                // New message_start for the handoff response
                                                let msg_start = serde_json::json!({
                                                    "type": "message_start",
                                                    "message": {
                                                        "id": format!("msg_{}", &uuid::Uuid::new_v4().to_string().replace("-", "")[..24]),
                                                        "type": "message",
                                                        "role": "assistant",
                                                        "content": [],
                                                        "model": actual_model,
                                                        "stop_reason": null,
                                                        "usage": {"input_tokens": 0, "output_tokens": 0}
                                                    }
                                                });
                                                yield Ok(format!("event: message_start\ndata: {}\n\n", msg_start));

                                                // Emit colorized @packname indicator as first content block
                                                let pack_indicator = format!("\x1b[1;36m@{}\x1b[0m ", new_pack.name);
                                                let indicator_block_start = serde_json::json!({
                                                    "type": "content_block_start",
                                                    "index": 0,
                                                    "content_block": {"type": "text", "text": ""}
                                                });
                                                yield Ok(format!("event: content_block_start\ndata: {}\n\n", indicator_block_start));
                                                let indicator_delta = serde_json::json!({
                                                    "type": "content_block_delta",
                                                    "index": 0,
                                                    "delta": {"type": "text_delta", "text": pack_indicator}
                                                });
                                                yield Ok(format!("event: content_block_delta\ndata: {}\n\n", indicator_delta));

                                                // Build the followup request - system prompt has general instructions,
                                                // we just append the handoff-specific user-instruction
                                                let followup_request = anthropic::MessagesRequest {
                                                    model: actual_model.to_string(),
                                                    messages: followup_messages,
                                                    max_tokens: hr.original_request.max_tokens.min(hr.max_output_tokens),
                                                    system: {
                                                        Some(match &hr.original_request.system {
                                                            Some(s) => anthropic::SystemPrompt::Text(format!("{}{}", s.to_string(), handoff_instruction)),
                                                            None => anthropic::SystemPrompt::Text(handoff_instruction.trim_start().to_string()),
                                                        })
                                                    },
                                                    temperature: hr.original_request.temperature,
                                                    tools: hr.original_request.tools.clone(),
                                                    stream: Some(true),
                                                    thinking: hr.original_request.thinking.clone(),
                                                    extra: Default::default(),
                                                };

                                                // Make the request to the new model
                                                if config.passthrough {
                                                    // Passthrough mode - just forward
                                                    match hr.http_client
                                                        .post(&config.backend_url)
                                                        .header("x-api-key", &api_key)
                                                        .header("anthropic-version", "2023-06-01")
                                                        .header("Content-Type", "application/json")
                                                        .json(&followup_request)
                                                        .send()
                                                        .await
                                                    {
                                                        Ok(response) if response.status().is_success() => {
                                                            let mut followup_stream = response.bytes_stream();
                                                            let mut buf = String::new();

                                                            while let Some(chunk) = followup_stream.next().await {
                                                                if let Ok(bytes) = chunk {
                                                                    // Forward SSE events, filtering out message_start (we already sent one)
                                                                    let chunk_str = String::from_utf8_lossy(&bytes);
                                                                    buf.push_str(&chunk_str);

                                                                    while let Some(end) = buf.find("\n\n") {
                                                                        let event = buf[..end].to_string();
                                                                        buf = buf[end + 2..].to_string();

                                                                        // Skip message_start events (we sent our own)
                                                                        if !event.contains("\"type\":\"message_start\"") && !event.contains("\"type\": \"message_start\"") {
                                                                            yield Ok(format!("{}\n\n", event));
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        Ok(response) => {
                                                            let status = response.status();
                                                            let body = response.text().await.unwrap_or_default();
                                                            error!("Handoff request failed: {} - {}", status, body);
                                                        }
                                                        Err(e) => {
                                                            error!("Handoff request error: {}", e);
                                                        }
                                                    }
                                                } else {
                                                    // Translation mode - convert to OpenAI format
                                                    // Process inline without recursion to avoid async boxing issues
                                                    let system_string = followup_request.system.as_ref().map(|s| s.to_string());
                                                    let (openai_messages, _id_mapping) = convert::anthropic_request_to_openai(
                                                        &followup_request.messages,
                                                        system_string.as_deref(),
                                                    );
                                                    let openai_tools = followup_request.tools.as_ref()
                                                        .map(|tools| convert::anthropic_tools_to_openai(tools));

                                                    let openai_request = openai::ChatCompletionRequest {
                                                        model: config.model_id.clone(),
                                                        messages: openai_messages,
                                                        max_tokens: Some(followup_request.max_tokens),
                                                        temperature: followup_request.temperature,
                                                        tools: openai_tools,
                                                        stream: Some(true),
                                                    };

                                                    match hr.http_client
                                                        .post(&config.backend_url)
                                                        .header("Authorization", format!("Bearer {}", api_key))
                                                        .header("Content-Type", "application/json")
                                                        .json(&openai_request)
                                                        .send()
                                                        .await
                                                    {
                                                        Ok(response) if response.status().is_success() => {
                                                            // Stream OpenAI response and convert inline
                                                            let mut followup_stream = response.bytes_stream();
                                                            let mut buf = String::new();
                                                            let mut handoff_block_index: i32 = 0; // Start at 0 since we emit indicator first
                                                            let mut handoff_text_started = false;

                                                            // Emit colorized @packname indicator first (translation mode - OpenAI stream)
                                                            let pack_indicator = format!("\x1b[1;36m@{}\x1b[0m ", new_pack.name);
                                                            let indicator_block_start = serde_json::json!({
                                                                "type": "content_block_start",
                                                                "index": 0,
                                                                "content_block": {"type": "text", "text": ""}
                                                            });
                                                            yield Ok(format!("event: content_block_start\ndata: {}\n\n", indicator_block_start));
                                                            let indicator_delta = serde_json::json!({
                                                                "type": "content_block_delta",
                                                                "index": 0,
                                                                "delta": {"type": "text_delta", "text": pack_indicator}
                                                            });
                                                            yield Ok(format!("event: content_block_delta\ndata: {}\n\n", indicator_delta));

                                                            while let Some(chunk) = followup_stream.next().await {
                                                                if let Ok(bytes) = chunk {
                                                                    buf.push_str(&String::from_utf8_lossy(&bytes));

                                                                    while let Some(line_end) = buf.find("\n\n") {
                                                                        let line = buf[..line_end].to_string();
                                                                        buf = buf[line_end + 2..].to_string();

                                                                        if let Some(data) = line.strip_prefix("data: ") {
                                                                            if data.trim() == "[DONE]" { continue; }

                                                                            if let Ok(chunk) = serde_json::from_str::<openai::ChatCompletionChunk>(data) {
                                                                                for choice in &chunk.choices {
                                                                                    if let Some(content) = &choice.delta.content {
                                                                                        if !content.is_empty() {
                                                                                            if !handoff_text_started {
                                                                                                handoff_block_index += 1;
                                                                                                let block_start = serde_json::json!({
                                                                                                    "type": "content_block_start",
                                                                                                    "index": handoff_block_index,
                                                                                                    "content_block": {"type": "text", "text": ""}
                                                                                                });
                                                                                                yield Ok(format!("event: content_block_start\ndata: {}\n\n", block_start));
                                                                                                handoff_text_started = true;
                                                                                            }

                                                                                            let delta_event = serde_json::json!({
                                                                                                "type": "content_block_delta",
                                                                                                "index": handoff_block_index,
                                                                                                "delta": {"type": "text_delta", "text": content}
                                                                                            });
                                                                                            yield Ok(format!("event: content_block_delta\ndata: {}\n\n", delta_event));
                                                                                        }
                                                                                    }

                                                                                    if choice.finish_reason.is_some() {
                                                                                        if handoff_text_started {
                                                                                            let block_stop = serde_json::json!({
                                                                                                "type": "content_block_stop",
                                                                                                "index": handoff_block_index
                                                                                            });
                                                                                            yield Ok(format!("event: content_block_stop\ndata: {}\n\n", block_stop));
                                                                                        }

                                                                                        let msg_delta = serde_json::json!({
                                                                                            "type": "message_delta",
                                                                                            "delta": {"stop_reason": "end_turn"},
                                                                                            "usage": {"output_tokens": 0}
                                                                                        });
                                                                                        yield Ok(format!("event: message_delta\ndata: {}\n\n", msg_delta));

                                                                                        let msg_stop = serde_json::json!({"type": "message_stop"});
                                                                                        yield Ok(format!("event: message_stop\ndata: {}\n\n", msg_stop));
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        Ok(response) => {
                                                            let status = response.status();
                                                            let body = response.text().await.unwrap_or_default();
                                                            error!("Handoff request failed: {} - {}", status, body);
                                                        }
                                                        Err(e) => {
                                                            error!("Handoff request error: {}", e);
                                                        }
                                                    }
                                                }
                                            } else {
                                                error!("Handoff failed: API key {} not set", config.api_key_env);
                                            }
                                        } else {
                                            error!("Handoff failed: model {} not found in registry", actual_model);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Log parse failures - might indicate unexpected finish_reason values
                        if data.contains("finish_reason") {
                            warn!("🔴 Failed to parse chunk with finish_reason - data: {}",
                                if data.len() > 200 { &data[..200] } else { data });
                        }
                    }
                }
            }
        }
    }
}

/// Detect @@packname pattern in model output for auto-switching
/// Returns (pack_name, handoff_message) if found
///
/// Convention: The FIRST valid @@packname triggers the switch, UNLESS it's a self-call
/// and there's another valid pack mentioned later (skip self-calls in favor of real handoffs).
///
/// The handoff_message is EVERYTHING after @@packname (may contain downstream @@ instructions).
///
/// Example: "@@claude Please implement X, then @@mistral for tests"
///   -> switches to claude with message "Please implement X, then @@mistral for tests"
///   -> claude sees the full instruction including the downstream handoff
///
/// Example: "@@mistral - Please add tests... then @@claude for verification" (when on mistral)
///   -> skips @@mistral (self-call), switches to claude with the verification message
///
/// Only matches VALID pack names (glm, claude, mistral, anthropic, devstral, liefstral)
fn detect_model_switch(text: &str) -> Option<(String, Option<String>)> {
    detect_model_switch_with_current(text, None)
}

/// Same as detect_model_switch but aware of current pack to skip self-calls intelligently
fn detect_model_switch_with_current(text: &str, current_pack: Option<&str>) -> Option<(String, Option<String>)> {
    use regex::Regex;

    // Valid pack names (must match ModelPack::all_packs keys)
    const VALID_PACKS: &[&str] = &["glm", "mistral", "devstral", "devstral-small-2", "liefstral", "anthropic", "claude"];

    // Pattern: @@switch=packname OR @@packname (switch= is optional)
    // We'll capture the pack name and then get everything after it as the message
    let re = Regex::new(r"@@(?:switch=)?([a-zA-Z][a-zA-Z0-9_-]*)").ok()?;

    // Collect all valid, non-escaped pack matches
    let mut matches: Vec<(String, usize, usize)> = Vec::new(); // (pack_name, start, end)

    for cap in re.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            let full_match = cap.get(0)?;
            let match_start = full_match.start();

            // Skip if this match is inside an escaped context
            if is_in_escaped_context(text, match_start) {
                continue;
            }

            let pack = m.as_str().to_lowercase();
            if VALID_PACKS.contains(&pack.as_str()) {
                matches.push((pack, match_start, full_match.end()));
            }
        }
    }

    if matches.is_empty() {
        return None;
    }

    // If we know the current pack, check if the first match is a self-call
    // If so, and there's another valid pack after it, use that instead
    if let Some(current) = current_pack {
        let first = &matches[0];
        if first.0 == current && matches.len() > 1 {
            // First match is self-call, use the second match instead
            let second = &matches[1];
            info!("Skipping self-call @@{}, using @@{} instead", first.0, second.0);
            let rest = text[second.2..].trim();
            let message = if rest.is_empty() { None } else { Some(rest.to_string()) };
            return Some((second.0.clone(), message));
        }
    }

    // Use the first match
    let first = &matches[0];
    let rest = text[first.2..].trim();
    let message = if rest.is_empty() { None } else { Some(rest.to_string()) };
    Some((first.0.clone(), message))
}

/// Check if a position in text is inside an escaped context:
/// - Inside backticks: `@@claude`
/// - Inside code blocks: ```...@@claude...```
/// - On a quote line: > @@claude
fn is_in_escaped_context(text: &str, pos: usize) -> bool {
    // Check if on a quote line (line starts with >)
    let line_start = text[..pos].rfind('\n').map(|p| p + 1).unwrap_or(0);
    let line_prefix = &text[line_start..pos];
    if line_prefix.trim_start().starts_with('>') {
        return true;
    }

    // Check if inside a code block (``` ... ```)
    let before = &text[..pos];
    let code_block_opens = before.matches("```").count();
    if code_block_opens % 2 == 1 {
        // Odd number of ``` before us means we're inside a code block
        return true;
    }

    // Check if inside inline backticks (` ... `)
    // Count backticks before position, but exclude triple backticks
    let mut in_backtick = false;
    let mut i = 0;
    let bytes = text.as_bytes();
    while i < bytes.len() && i < pos {
        if i + 2 < bytes.len() && bytes[i] == b'`' && bytes[i+1] == b'`' && bytes[i+2] == b'`' {
            // Skip triple backticks (handled above)
            i += 3;
            continue;
        }
        if bytes[i] == b'`' {
            in_backtick = !in_backtick;
        }
        i += 1;
    }

    in_backtick
}

/// Parse duration from string (e.g., "30s", "5m", "2h", or bare number as seconds)
/// Returns None if parsing fails
fn parse_duration(s: &str) -> Option<Duration> {
    use regex::Regex;

    // Try bare number (seconds)
    if let Ok(secs) = s.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }

    // Try with suffix (30s, 30m, 2h)
    let re = Regex::new(r"^(\d+)(s|m|h)$").ok()?;
    let caps = re.captures(s)?;
    let num: u64 = caps.get(1)?.as_str().parse().ok()?;
    match caps.get(2)?.as_str() {
        "s" => Some(Duration::from_secs(num)),
        "m" => Some(Duration::from_secs(num * 60)),
        "h" => Some(Duration::from_secs(num * 3600)),
        _ => None,
    }
}

/// Detect @continuous toggle commands in user messages
/// Returns Some(ContinuousMode) for various @continuous commands
/// Returns None if no continuous command found
fn detect_continuous_toggle(text: &str) -> Option<ContinuousMode> {
    use regex::Regex;

    // Pattern: @continuous or @continuous=value
    let re = Regex::new(r"@continuous(?:=([^\s]+))?").ok()?;

    for cap in re.captures_iter(text) {
        let full_match = cap.get(0)?;

        // Skip if in escaped context
        if is_in_escaped_context(text, full_match.start()) {
            continue;
        }

        // Check for value after =
        if let Some(value) = cap.get(1) {
            let val = value.as_str();
            return match val.to_lowercase().as_str() {
                "on" | "true" | "yes" | "1" => Some(ContinuousMode::Unlimited),
                "off" | "false" | "no" | "0" => Some(ContinuousMode::Off),
                _ => {
                    // Try parsing as duration
                    if let Some(duration) = parse_duration(val) {
                        Some(ContinuousMode::TimeLimited(duration))
                    } else {
                        None // Unknown value, ignore
                    }
                }
            };
        } else {
            // Bare @continuous enables unlimited mode
            return Some(ContinuousMode::Unlimited);
        }
    }

    None
}

/// Detect @costs toggle commands in user messages
/// Returns Some(true) for @costs or @costs=on (default)
/// Returns Some(false) for @costs=off
/// Returns None if no costs command found
fn detect_costs_toggle(text: &str) -> Option<bool> {
    use regex::Regex;

    // Pattern: @costs or @costs=on/off
    let re = Regex::new(r"@costs(?:=(\w+))?").ok()?;

    for cap in re.captures_iter(text) {
        let full_match = cap.get(0)?;

        // Skip if in escaped context
        if is_in_escaped_context(text, full_match.start()) {
            continue;
        }

        // Check for =on or =off value
        if let Some(value) = cap.get(1) {
            return match value.as_str().to_lowercase().as_str() {
                "on" | "true" | "yes" | "1" => Some(true),
                "off" | "false" | "no" | "0" => Some(false),
                _ => None, // Unknown value, ignore
            };
        } else {
            // Bare @costs toggles on
            return Some(true);
        }
    }

    None
}

/// Detect @strict toggle commands in user messages
/// Returns Some(StrictMode::On) for @strict or @strict=on
/// Returns Some(StrictMode::Off) for @strict=off
/// Returns None if no strict command found
fn detect_strict_toggle(text: &str) -> Option<StrictMode> {
    use regex::Regex;

    // Pattern: @strict or @strict=on/off
    let re = Regex::new(r"@strict(?:=(\w+))?").ok()?;

    for cap in re.captures_iter(text) {
        let full_match = cap.get(0)?;

        // Skip if in escaped context
        if is_in_escaped_context(text, full_match.start()) {
            continue;
        }

        // Check for =on or =off value
        if let Some(value) = cap.get(1) {
            return match value.as_str().to_lowercase().as_str() {
                "on" | "true" | "yes" | "1" => Some(StrictMode::On),
                "off" | "false" | "no" | "0" => Some(StrictMode::Off),
                _ => None, // Unknown value, ignore
            };
        } else {
            // Bare @strict enables strict mode
            return Some(StrictMode::On);
        }
    }

    None
}

/// Detect @watch or @unwatch command in user messages
/// Returns Some(true) for @watch, Some(false) for @watch=false or @unwatch
fn detect_watch_toggle(text: &str) -> Option<bool> {
    use regex::Regex;

    // Check for @unwatch first
    if let Ok(re) = Regex::new(r"@unwatch\b") {
        for mat in re.find_iter(text) {
            if !is_in_escaped_context(text, mat.start()) {
                return Some(false);
            }
        }
    }

    // Pattern: @watch or @watch=on/off
    let re = match Regex::new(r"@watch(?:=(\w+))?") {
        Ok(r) => r,
        Err(_) => return None,
    };

    for cap in re.captures_iter(text) {
        let full_match = cap.get(0)?;
        if is_in_escaped_context(text, full_match.start()) {
            continue;
        }

        if let Some(value) = cap.get(1) {
            return match value.as_str().to_lowercase().as_str() {
                "on" | "true" | "yes" | "1" => Some(true),
                "off" | "false" | "no" | "0" => Some(false),
                _ => None,
            };
        } else {
            return Some(true); // Bare @watch enables watching
        }
    }

    None
}

/// Get watchd credentials from ~/.palace/build-client.env
fn get_watchd_credentials() -> Option<(u16, String)> {
    use std::fs;

    // Try multiple home directories
    let creds_paths = [
        "/root/.palace/build-client.env",
        "/home/wings/.palace/build-client.env",
    ];

    for creds_path in &creds_paths {
        if let Ok(content) = fs::read_to_string(creds_path) {
            let mut port: Option<u16> = None;
            let mut token: Option<String> = None;

            for line in content.lines() {
                if let Some(p) = line.strip_prefix("PALACE_WATCH_PORT=") {
                    port = p.trim().parse().ok();
                } else if let Some(t) = line.strip_prefix("PALACE_WATCH_TOKEN=") {
                    token = Some(t.trim().to_string());
                }
            }

            if let (Some(p), Some(t)) = (port, token) {
                return Some((p, t));
            }
        }
    }

    None
}

/// Start watching a project via watchd API
async fn start_watching_project(project_path: &str) -> Result<(), String> {
    info!("Starting watch for project: {}", project_path);

    // First ensure watchd is running by spawning pal watchd if needed
    let (port, token) = match get_watchd_credentials() {
        Some(creds) => creds,
        None => {
            // Try to start watchd first
            info!("watchd not running, starting it...");
            spawn_watchd().await?;

            // Wait a bit for it to start and write creds
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            get_watchd_credentials()
                .ok_or_else(|| "watchd started but no credentials found".to_string())?
        }
    };

    // Register project with watchd via HTTP API
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/watch", port);

    let payload = serde_json::json!({
        "path": project_path
    });

    match client.post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(data) = response.json::<serde_json::Value>().await {
                    let status = data["status"].as_str().unwrap_or("unknown");
                    if status == "ok" || status == "already_watching" {
                        let types = data["types"].as_array()
                            .map(|arr| arr.iter()
                                .filter_map(|v| v.as_str())
                                .collect::<Vec<_>>()
                                .join(", "))
                            .unwrap_or_else(|| "unknown".to_string());
                        info!("Now watching {} ({})", project_path, types);
                        return Ok(());
                    } else {
                        let error = data["error"].as_str().unwrap_or("unknown error");
                        return Err(format!("watchd error: {}", error));
                    }
                }
                Ok(())
            } else {
                Err(format!("watchd returned {}", response.status()))
            }
        }
        Err(e) => {
            // Connection refused - watchd not running
            if e.is_connect() {
                info!("watchd not responding, starting it...");
                spawn_watchd().await?;
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                // Retry once
                let (port, token) = get_watchd_credentials()
                    .ok_or_else(|| "watchd started but no credentials found".to_string())?;

                let url = format!("http://127.0.0.1:{}/watch", port);
                match client.post(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .header("Content-Type", "application/json")
                    .json(&payload)
                    .send()
                    .await
                {
                    Ok(resp) if resp.status().is_success() => {
                        info!("Now watching {} (after starting watchd)", project_path);
                        Ok(())
                    }
                    Ok(resp) => Err(format!("watchd returned {}", resp.status())),
                    Err(e) => Err(format!("Failed to connect to watchd: {}", e)),
                }
            } else {
                Err(format!("watchd request failed: {}", e))
            }
        }
    }
}

/// Spawn watchd daemon
async fn spawn_watchd() -> Result<(), String> {
    use std::process::Command;

    let pal_paths = [
        "/root/.local/bin/pal",
        "/home/wings/.local/bin/pal",
        "/usr/local/bin/pal",
    ];

    let pal_binary = pal_paths.iter()
        .find(|p| std::path::Path::new(p).exists())
        .map(|s| *s);

    if let Some(pal_path) = pal_binary {
        // pal watch with no path just ensures daemon is running
        let result = Command::new(pal_path)
            .arg("watch")
            .arg("--status")
            .output();

        match result {
            Ok(output) => {
                if output.status.success() || String::from_utf8_lossy(&output.stdout).contains("Daemon") {
                    info!("watchd is now running");
                    Ok(())
                } else {
                    Err(format!("pal watch --status failed: {}", String::from_utf8_lossy(&output.stderr)))
                }
            }
            Err(e) => Err(format!("Failed to run pal: {}", e)),
        }
    } else {
        // Try palace.py directly
        let palace_py = "/mnt/castle/garage/palace-public/palace.py";
        let python = "/mnt/castle/garage/palace-public/.venv/bin/python";

        if std::path::Path::new(palace_py).exists() {
            let result = Command::new(python)
                .arg(palace_py)
                .arg("watchd")
                .spawn();

            match result {
                Ok(_) => {
                    info!("Started palace.py watchd");
                    Ok(())
                }
                Err(e) => Err(format!("Failed to start palace.py watchd: {}", e)),
            }
        } else {
            Err("Could not find pal binary or palace.py".to_string())
        }
    }
}

/// Extract working directory from Claude Code's system prompt <env> block
/// Format: <env>\nWorking directory: /path/to/project\n...\n</env>
fn extract_working_directory(text: &str) -> Option<String> {
    // Look for <env> block
    let env_start = text.find("<env>")?;
    let env_end = text.find("</env>")?;
    if env_end <= env_start {
        return None;
    }

    let env_block = &text[env_start..env_end];

    // Look for "Working directory: /path"
    for line in env_block.lines() {
        if let Some(path) = line.strip_prefix("Working directory: ") {
            let path = path.trim();
            if !path.is_empty() && path.starts_with('/') {
                return Some(path.to_string());
            }
        }
    }

    None
}

/// Get uid for a username by reading /etc/passwd
fn get_uid_for_user(username: &str) -> Option<u32> {
    let passwd = std::fs::read_to_string("/etc/passwd").ok()?;
    for line in passwd.lines() {
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() >= 3 && parts[0] == username {
            return parts[2].parse().ok();
        }
    }
    None
}

/// Get username from project directory ownership
/// Uses sudo stat to find who owns the project path
fn get_user_from_path_owner(path: &str) -> Option<String> {
    use std::process::Command;

    // Use sudo stat to get owner - daemon may not have read access to the directory
    let output = match Command::new("sudo")
        .args(["stat", "-c", "%U", path])
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            warn!("get_user_from_path_owner: Failed to run sudo stat: {}", e);
            return None;
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("get_user_from_path_owner: sudo stat failed for {}: {}", path, stderr);
        return None;
    }

    let owner = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if owner.is_empty() {
        warn!("get_user_from_path_owner: Empty owner for {}", path);
        return None;
    }

    info!("get_user_from_path_owner: {} owned by {}", path, owner);
    Some(owner)
}

/// Detect @verify command in user messages
/// Returns true if @verify is found (triggers immediate build/test verification)
fn detect_verify_command(text: &str) -> bool {
    use regex::Regex;

    let re = match Regex::new(r"@verify\b") {
        Ok(r) => r,
        Err(_) => return false,
    };

    for mat in re.find_iter(text) {
        // Skip if in escaped context
        if !is_in_escaped_context(text, mat.start()) {
            return true;
        }
    }

    false
}

/// Read verification status from watchd API
async fn read_watchd_status(project_path: Option<&str>) -> VerificationStatus {
    // Get watchd credentials
    let Some((port, token)) = get_watchd_credentials() else {
        return VerificationStatus {
            build_passing: false,
            test_passing: false,
            build_errors: vec!["watchd not running (use `@watch` to start)".to_string()],
            test_failures: vec![],
            verified_at: Instant::now(),
        };
    };

    // Query watchd API
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/status", port);

    let response = match client.get(&url)
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return VerificationStatus {
                build_passing: false,
                test_passing: false,
                build_errors: vec![format!("Cannot connect to watchd: {}", e)],
                test_failures: vec![],
                verified_at: Instant::now(),
            };
        }
    };

    if !response.status().is_success() {
        return VerificationStatus {
            build_passing: false,
            test_passing: false,
            build_errors: vec![format!("watchd returned {}", response.status())],
            test_failures: vec![],
            verified_at: Instant::now(),
        };
    }

    let data: serde_json::Value = match response.json().await {
        Ok(d) => d,
        Err(e) => {
            return VerificationStatus {
                build_passing: false,
                test_passing: false,
                build_errors: vec![format!("Invalid watchd response: {}", e)],
                test_failures: vec![],
                verified_at: Instant::now(),
            };
        }
    };

    let projects = match data.get("projects").and_then(|p| p.as_object()) {
        Some(p) => p,
        None => {
            return VerificationStatus {
                build_passing: false,
                test_passing: false,
                build_errors: vec!["No projects being watched (use `@watch` to start)".to_string()],
                test_failures: vec![],
                verified_at: Instant::now(),
            };
        }
    };

    if projects.is_empty() {
        return VerificationStatus {
            build_passing: false,
            test_passing: false,
            build_errors: vec!["No projects being watched (use `@watch` to start)".to_string()],
            test_failures: vec![],
            verified_at: Instant::now(),
        };
    }

    // Aggregate status from all projects (or filter by project_path)
    let mut all_build_passing = true;
    let mut all_test_passing = true;
    let mut all_build_errors = Vec::new();
    let mut all_test_failures = Vec::new();
    let mut found_any = false;

    for (path, status) in projects {
        // If project_path specified, only include matching projects
        if let Some(target) = project_path {
            if !path.contains(target) && !target.contains(path) {
                continue;
            }
        }

        found_any = true;
        let project_name = std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // Parse build status
        if let Some(build) = status.get("build") {
            let build_status = build["status"].as_str().unwrap_or("unknown");
            if build_status != "ok" {
                all_build_passing = false;
                if let Some(error) = build.get("error").and_then(|e| e.as_str()) {
                    all_build_errors.push(format!("[{}] {}", project_name, error.lines().next().unwrap_or(error)));
                } else if let Some(errors) = build["errors"].as_array() {
                    for e in errors.iter().take(3) {
                        if let Some(s) = e.as_str() {
                            all_build_errors.push(format!("[{}] {}", project_name, s.lines().next().unwrap_or(s)));
                        }
                    }
                } else {
                    all_build_errors.push(format!("[{}] Build failed", project_name));
                }
            }
        }

        // Parse test status
        if let Some(tests) = status.get("tests") {
            let test_status = tests["status"].as_str().unwrap_or("unknown");
            if test_status != "ok" && test_status != "pending" {
                all_test_passing = false;
                if let Some(failures) = tests["failures"].as_array() {
                    for f in failures.iter().take(3) {
                        if let Some(s) = f.as_str() {
                            all_test_failures.push(format!("[{}] {}", project_name, s));
                        }
                    }
                }
                let passed = tests["passed"].as_u64().unwrap_or(0);
                let failed = tests["failed"].as_u64().unwrap_or(0);
                if failed > 0 && all_test_failures.is_empty() {
                    all_test_failures.push(format!("[{}] {}/{} tests failed", project_name, failed, passed + failed));
                }
            }
        }
    }

    if !found_any {
        return VerificationStatus {
            build_passing: false,
            test_passing: false,
            build_errors: vec!["Specified project not being watched".to_string()],
            test_failures: vec![],
            verified_at: Instant::now(),
        };
    }

    VerificationStatus {
        build_passing: all_build_passing,
        test_passing: all_test_passing,
        build_errors: all_build_errors,
        test_failures: all_test_failures,
        verified_at: Instant::now(),
    }
}

/// Format verification status for display
fn format_verification_status(status: &VerificationStatus) -> String {
    let mut result = String::new();

    result.push_str("**Verification Results**\n\n");

    // Build status
    if status.build_passing {
        result.push_str("✅ **Build:** PASSING\n");
    } else {
        result.push_str("❌ **Build:** FAILING\n");
        for err in &status.build_errors {
            result.push_str(&format!("   - {}\n", err.lines().next().unwrap_or(err)));
        }
    }

    // Test status
    if status.test_passing {
        result.push_str("✅ **Tests:** PASSING\n");
    } else {
        result.push_str("❌ **Tests:** FAILING\n");
        for failure in &status.test_failures {
            result.push_str(&format!("   - {}\n", failure));
        }
    }

    // Overall verdict
    result.push_str("\n");
    if status.build_passing && status.test_passing {
        result.push_str("**Verdict:** All checks passed! ✅ Ready to complete.\n");
    } else {
        result.push_str("**Verdict:** Must fix failing checks before completion.\n");
    }

    result
}

/// Parse strategy from message text
/// Supports: @strategy=simple|smart|premium|cheap|free|local|airplane|hypermiler|burn
/// Modifiers for smart: @strategy=smart,premium or @strategy=smart,25% or @strategy=smart,25%,global
/// Modifiers for hypermiler: @strategy=hypermiler,cheap or @strategy=hypermiler,free
fn parse_strategy(text: &str) -> Option<Strategy> {
    use regex::Regex;

    // Pattern: @strategy=main,mod1,mod2,...
    let re = Regex::new(r"@strategy=(\w+)(?:,([^\s@]+))?").ok()?;

    let caps = re.captures(text)?;
    let main = caps.get(1)?.as_str().to_lowercase();
    let modifiers: Vec<&str> = caps.get(2)
        .map(|m| m.as_str().split(',').collect())
        .unwrap_or_default();

    match main.as_str() {
        "simple" => Some(Strategy::Simple),
        "smart" => Some(Strategy::Smart(parse_smart_modifiers(&modifiers))),
        "premium" => Some(Strategy::Premium),
        "cheap" => Some(Strategy::Cheap),
        "free" => Some(Strategy::Free),
        "local" => Some(Strategy::Local),
        "airplane" => Some(Strategy::Airplane),
        "hypermiler" => Some(Strategy::Hypermiler(parse_hypermiler_modifiers(&modifiers))),
        "burn" => Some(Strategy::Burn),
        _ => None,
    }
}

/// Parse modifiers for smart strategy
fn parse_smart_modifiers(mods: &[&str]) -> SmartConfig {
    let mut config = SmartConfig::default();

    for m in mods {
        let m_lower = m.to_lowercase();
        match m_lower.as_str() {
            "premium" => config.worker_tier = WorkerTier::Premium,
            "cheap" => config.worker_tier = WorkerTier::Cheap,
            "standard" => config.worker_tier = WorkerTier::Standard,
            "local" | "airplane" => config.worker_tier = WorkerTier::Local,
            s if s.ends_with('%') => {
                if let Ok(n) = s.trim_end_matches('%').parse::<u8>() {
                    config.rate_threshold_percent = n;
                }
            }
            _ => {}
        }
    }

    config
}

/// Parse modifiers for hypermiler strategy
fn parse_hypermiler_modifiers(mods: &[&str]) -> HypermilerConfig {
    let mut config = HypermilerConfig::default();

    for m in mods {
        let m_lower = m.to_lowercase();
        match m_lower.as_str() {
            "premium" => config.worker_tier = WorkerTier::Premium,
            "cheap" => config.worker_tier = WorkerTier::Cheap,
            "free" => config.worker_tier = WorkerTier::Free,
            "standard" => config.worker_tier = WorkerTier::Standard,
            "local" | "airplane" => config.worker_tier = WorkerTier::Local,
            // Orchestrator shortcuts in strategy modifiers
            "glmv" | "glm" => config.orchestrator_model = Some("glm-4v".to_string()),
            "devstral" => config.orchestrator_model = Some("devstral".to_string()),
            _ => {}
        }
    }

    config
}

/// Detect strategy toggle in message and return new strategy if found
fn detect_strategy_toggle(text: &str) -> Option<Strategy> {
    parse_strategy(text)
}

/// Parse hypermiler orchestrator override
/// @hypermiler=glmv -> GLM-4.6V
/// @hypermiler=devstral -> Devstral 2
/// @hypermiler=local -> Best local model
/// @hypermiler=qwen32b -> Specific model
fn parse_hypermiler_orchestrator(text: &str) -> Option<String> {
    use regex::Regex;
    let re = Regex::new(r"@hypermiler=(\w+)").ok()?;
    let caps = re.captures(text)?;
    let orchestrator = caps.get(1)?.as_str().to_lowercase();

    match orchestrator.as_str() {
        "glmv" | "glm" | "glm4" => Some("glm-4v".to_string()),
        "devstral" | "devstral2" => Some("devstral".to_string()),
        "local" => None,  // Auto-select
        "qwen" | "qwen32b" => Some("qwen2.5-coder:32b-instruct-q4_K_M".to_string()),
        "qwen72b" => Some("qwen2.5-coder:72b-instruct-q4_K_M".to_string()),
        "qwen7b" => Some("qwen2.5-coder:7b-instruct-q4_K_M".to_string()),
        other => Some(other.to_string()),  // Allow arbitrary model names
    }
}

/// Detect hypermiler orchestrator override in message
fn detect_hypermiler_override(text: &str) -> Option<String> {
    parse_hypermiler_orchestrator(text)
}

/// Detect @planner=<model> command to override task planner model
/// @planner=devstral -> Use Devstral for task planning
/// @planner=glm -> Use GLM-4.6 for task planning
/// @planner=opus -> Use Claude Opus for task planning
fn detect_planner_model(text: &str) -> Option<String> {
    use regex::Regex;
    let re = Regex::new(r"@planner=(\w+)").ok()?;
    if let Some(caps) = re.captures(text) {
        let model_spec = caps.get(1)?.as_str().to_lowercase();
        // Map aliases to full model names
        let model = match model_spec.as_str() {
            "devstral" | "mistral" => "devstral-2512",
            "glm" | "glmv" => "glm-4.6v",
            "opus" => "claude-opus-4-5-20251101",
            "sonnet" => "claude-sonnet-4-5",
            "haiku" => "claude-haiku-4-5",
            "local" => "devstral-small-2",  // Default local
            other => other,  // Use as-is
        };
        return Some(model.to_string());
    }
    None
}

/// Detect @spawn/@swarm command to trigger planner execution
/// @spawn, @swarm, and @swarm=on all trigger the streaming planner
fn detect_spawn_command(text: &str) -> bool {
    // @spawn always triggers
    if text.contains("@spawn") {
        return true;
    }
    // @swarm=on explicitly triggers the planner
    if text.contains("@swarm=on") {
        return true;
    }
    // @swarm without = also triggers (but @swarm=20, @swarm=off, @swarm=unlimited don't)
    if text.contains("@swarm") {
        // Check if there's no = immediately after @swarm
        for (i, _) in text.match_indices("@swarm") {
            let after = &text[i + 6..]; // "@swarm" is 6 chars
            // If next char is not '=' or there is no next char, it's a trigger
            if after.is_empty() || !after.starts_with('=') {
                return true;
            }
        }
    }
    false
}

/// Detect @action or @actions command to select tasks for execution
/// Returns Some(vec![indices]) if found, None otherwise
/// @action 5 -> Select single action
/// @actions 1,2,3,4 -> Select multiple (comma-separated)
/// @actions 1 2 3 4 -> Select multiple (space-separated)
/// @actions all -> Select all pending actions
fn detect_action_command(text: &str) -> Option<ActionSelection> {
    use regex::Regex;

    // Check for @action(s) all
    if text.contains("@actions all") || text.contains("@action all") {
        return Some(ActionSelection::All);
    }

    // Check for @actions with numbers (comma or space separated)
    let re_actions = Regex::new(r"@actions?\s+([\d,\s]+)").ok()?;
    if let Some(caps) = re_actions.captures(text) {
        let nums_str = caps.get(1)?.as_str();
        let indices: Vec<usize> = nums_str
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if !indices.is_empty() {
            return Some(ActionSelection::Indices(indices));
        }
    }

    None
}

/// Action selection from @action command
#[derive(Debug, Clone)]
enum ActionSelection {
    All,
    Indices(Vec<usize>),
}

/// Parse actions: YAML block from planner output
/// Returns (text_before, actions, text_after)
fn parse_actions_yaml(text: &str) -> (String, Vec<PendingAction>, String) {
    let mut actions = Vec::new();
    let mut text_before = String::new();
    let mut text_after = String::new();

    // Strip [planner] prefixes that the compact renderer adds
    let clean_text: String = text.lines()
        .map(|line| {
            line.strip_prefix("[planner] ")
                .or_else(|| line.strip_prefix("[planner]"))
                .unwrap_or(line)
        })
        .collect::<Vec<_>>()
        .join("\n");
    let text = &clean_text;

    // Find the actions: block
    if let Some(actions_start) = text.find("actions:") {
        text_before = text[..actions_start].trim().to_string();

        let after_actions = &text[actions_start + 8..]; // Skip "actions:"
        let mut in_actions = true;
        let mut current_label = String::new();
        let mut current_desc = String::new();
        let mut action_num = 1; // 1-indexed for user display

        for line in after_actions.lines() {
            let trimmed = line.trim();

            // Check if we've left the actions block (non-indented, non-empty, not starting with -)
            if !line.starts_with(' ') && !line.starts_with('\t') && !trimmed.is_empty() && !trimmed.starts_with('-') {
                // Save any pending action
                if !current_label.is_empty() {
                    actions.push(PendingAction {
                        num: action_num,
                        label: current_label.clone(),
                        description: current_desc.trim().to_string(),
                    });
                    current_label.clear(); // Prevent duplicate at end
                }
                in_actions = false;
                text_after.push_str(line);
                text_after.push('\n');
                continue;
            }

            if !in_actions {
                text_after.push_str(line);
                text_after.push('\n');
                continue;
            }

            // Parse YAML-style list items
            if trimmed.starts_with("- label:") {
                // Save previous action if any
                if !current_label.is_empty() {
                    actions.push(PendingAction {
                        num: action_num,
                        label: current_label.clone(),
                        description: current_desc.trim().to_string(),
                    });
                    action_num += 1;
                }
                current_label = trimmed.strip_prefix("- label:").unwrap_or("").trim().to_string();
                current_desc = String::new();
            } else if trimmed.starts_with("label:") {
                // Save previous action if any
                if !current_label.is_empty() {
                    actions.push(PendingAction {
                        num: action_num,
                        label: current_label.clone(),
                        description: current_desc.trim().to_string(),
                    });
                    action_num += 1;
                }
                current_label = trimmed.strip_prefix("label:").unwrap_or("").trim().to_string();
                current_desc = String::new();
            } else if trimmed.starts_with("description:") {
                current_desc = trimmed.strip_prefix("description:").unwrap_or("").trim().to_string();
            }
        }

        // Don't forget the last action
        if !current_label.is_empty() {
            actions.push(PendingAction {
                num: action_num,
                label: current_label,
                description: current_desc.trim().to_string(),
            });
        }
    } else {
        // No actions block found
        text_before = text.to_string();
    }

    (text_before.trim().to_string(), actions, text_after.trim().to_string())
}

/// Format pending actions as a numbered list for display
fn format_actions_list(actions: &[PendingAction]) -> String {
    let mut output = String::from("\n📋 **Available Actions:**\n\n");
    for action in actions {
        output.push_str(&format!("  **{}. {}**\n", action.num, action.label));
        if !action.description.is_empty() {
            output.push_str(&format!("     {}\n", action.description));
        }
        output.push('\n');
    }
    output.push_str("Use `@action N` or `@actions 1,2,3` to select tasks for execution.\n");
    output
}

/// Detect @swarm command to enable/disable swarm mode
/// @swarm=on -> Enable swarm mode
/// @swarm=off -> Disable swarm mode
/// @swarm=unlimited -> No cap, planner decides
/// @swarm=N -> Enable with N max workers
fn detect_swarm_mode(text: &str) -> Option<SwarmMode> {
    use regex::Regex;
    // Check for @swarm=off first
    if text.contains("@swarm=off") {
        return Some(SwarmMode::Off);
    }
    // Check for @swarm=unlimited
    if text.contains("@swarm=unlimited") {
        return Some(SwarmMode::Unlimited);
    }
    // Check for @swarm=on
    if text.contains("@swarm=on") {
        return Some(SwarmMode::On);
    }
    // Check for @swarm=N (safety cap)
    let re = Regex::new(r"@swarm=(\d+)").ok()?;
    if let Some(caps) = re.captures(text) {
        if let Ok(limit) = caps.get(1)?.as_str().parse::<u32>() {
            return Some(SwarmMode::TaskLimit(limit));
        }
    }
    // Just @swarm with no value = on
    if text.contains("@swarm") && !text.contains("@swarm=") {
        return Some(SwarmMode::On);
    }
    None
}

/// Check if a model is available in Ollama
async fn is_model_available(ollama_url: &str, model: &str) -> bool {
    let client = reqwest::Client::new();
    let url = format!("{}/api/tags", ollama_url);

    match client.get(&url).send().await {
        Ok(resp) => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(models) = json["models"].as_array() {
                    return models.iter().any(|m| {
                        m["name"].as_str()
                            .map(|n| n.contains(model))
                            .unwrap_or(false)
                    });
                }
            }
            false
        }
        Err(_) => false,
    }
}

/// Select orchestrator model for hypermiler mode based on tier
/// Default orchestrator is Devstral 2 (free via Mistral API)
/// User can override with @hypermiler=<tier> or @hypermiler=<model>
fn select_hypermiler_orchestrator(orchestrator_override: Option<&str>) -> String {
    match orchestrator_override {
        // Tier-based selection (@hypermiler=premium, @hypermiler=local, etc.)
        Some("premium") => {
            info!("HYPERMILER: Orchestrator tier=premium, using GPT-5.1 Codex Max");
            "openai/gpt-5.1-codex-max".to_string()
        }
        Some("standard") => {
            info!("HYPERMILER: Orchestrator tier=standard, using Sonnet");
            "claude-sonnet-4-5".to_string()
        }
        Some("cheap") => {
            info!("HYPERMILER: Orchestrator tier=cheap, using Haiku");
            "claude-haiku-4-5".to_string()
        }
        Some("free") => {
            info!("HYPERMILER: Orchestrator tier=free, using Devstral 2");
            "devstral-2512".to_string()
        }
        Some("local") | Some("airplane") => {
            info!("HYPERMILER: Orchestrator tier=local, using Devstral Small 2");
            "devstral-small-2".to_string()
        }
        // Specific model aliases
        Some("glm") | Some("glmv") => {
            info!("HYPERMILER: Orchestrator=GLM-4.6V");
            "glm-4.6v".to_string()
        }
        Some("devstral") => {
            info!("HYPERMILER: Orchestrator=Devstral 2");
            "devstral-2512".to_string()
        }
        Some("smol") | Some("devstral-small") => {
            info!("HYPERMILER: Orchestrator=Devstral Small 2");
            "devstral-small-2".to_string()
        }
        Some("opus") => {
            info!("HYPERMILER: Orchestrator=Claude Opus");
            "claude-opus-4-5-20251101".to_string()
        }
        Some("codex") | Some("gpt5") => {
            info!("HYPERMILER: Orchestrator=GPT-5.1 Codex Max");
            "openai/gpt-5.1-codex-max".to_string()
        }
        Some("gemini") | Some("gemini3") => {
            info!("HYPERMILER: Orchestrator=Gemini 3 Pro");
            "google/gemini-3-pro-preview".to_string()
        }
        // Explicit model name - pass through
        Some(model) => {
            info!("HYPERMILER: Orchestrator={} (explicit)", model);
            model.to_string()
        }
        // DEFAULT: Devstral 2 via Mistral API (free during preview)
        None => {
            info!("HYPERMILER: Orchestrator=Devstral 2 (default, free)");
            "devstral-2512".to_string()
        }
    }
}

/// Run hypermiler orchestration - routes through the translator
/// The translator handles ALL backend routing (Ollama, Mistral, Z.ai, OpenRouter)
async fn run_hypermiler_orchestration(
    task: &str,
    context: &str,
    config: &HypermilerConfig,
    http_client: &reqwest::Client,
) -> Result<String, String> {
    // Select orchestrator model (translator handles the actual backend)
    let model = select_hypermiler_orchestrator(config.orchestrator_model.as_deref());

    info!("HYPERMILER: Orchestrating with {} (workers: {:?})", model, config.worker_tier);

    let prompt = format!(
        "You are a coding task orchestrator. Break this task into steps and assign each to the appropriate model tier.\n\n\
        Worker tiers available: {:?}\n\n\
        CONTEXT:\n{}\n\n\
        TASK:\n{}\n\n\
        For each step, specify:\n\
        1. What to do\n\
        2. Which worker tier should do it\n\
        3. Expected output\n\
        4. Dependencies",
        config.worker_tier, context, task
    );

    // Route through the translator (ourselves!) - it handles all backend routing
    let translator_url = format!(
        "http://127.0.0.1:{}/v1/messages",
        std::env::var("PORT").unwrap_or_else(|_| "19848".to_string())
    );

    let request = serde_json::json!({
        "model": model,
        "max_tokens": 2000,
        "messages": [{
            "role": "user",
            "content": prompt
        }]
    });

    let resp = http_client
        .post(&translator_url)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Translator request failed: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Translator returned {}: {}", status, body));
    }

    let json: serde_json::Value = resp.json().await
        .map_err(|e| format!("Failed to parse translator response: {}", e))?;

    // Extract text from Anthropic-format response
    if let Some(content) = json["content"].as_array() {
        for block in content {
            if block["type"].as_str() == Some("text") {
                if let Some(text) = block["text"].as_str() {
                    return Ok(text.to_string());
                }
            }
        }
    }

    Err("No text content in response".to_string())
}

/// Classification result from verification model
#[derive(Debug, Clone, Copy, PartialEq)]
enum ContinuousStatus {
    /// Task complete, success criteria met
    Done,
    /// Making progress, should continue
    Progress,
    /// Stuck, needs help from stronger model
    Stuck,
    /// Unsure, needs second opinion from medium model
    Unsure,
}

/// Result of verification check
#[derive(Debug)]
struct VerificationResult {
    /// Classification status
    status: ContinuousStatus,
    /// Verification tier that produced this result
    tier: VerificationTier,
    /// Advice from 24b model when stuck (injected into next turn)
    advice: Option<String>,
    /// Summary of progress for continuation message
    progress_summary: Option<String>,
    /// Categories that need work (from parsed scores)
    weak_categories: Vec<(String, u8)>,
}

/// Parsed completion scores from model self-report
#[derive(Debug, Default)]
struct CompletionScores {
    /// Whether "TASK COMPLETE" or similar was found
    has_completion_signal: bool,
    /// Individual category scores (name -> score 0-100)
    scores: Vec<(String, u8)>,
    /// Lowest score across all categories
    min_score: Option<u8>,
}

/// Parse structured completion output from model
/// Expected format:
/// ```
/// TASK COMPLETE
/// confidence:
///   code_quality: 92
///   test_coverage: 85
///   documentation: 70
/// ```
fn parse_completion_scores(text: &str) -> CompletionScores {
    let mut result = CompletionScores::default();

    // Check for completion signal
    let text_upper = text.to_uppercase();
    result.has_completion_signal = text_upper.contains("TASK COMPLETE")
        || text_upper.contains("TASK_COMPLETE")
        || text_upper.contains("COMPLETED SUCCESSFULLY")
        || text_upper.contains("ALL DONE");

    // Parse confidence scores - look for patterns like "category: 85" or "category=85"
    let score_re = regex::Regex::new(r"(?i)([a-z_]+)\s*[:=]\s*(\d{1,3})").ok();

    if let Some(re) = score_re {
        // Only parse scores if we're in a confidence/scores section
        let in_confidence_section = text.to_lowercase().contains("confidence")
            || text.to_lowercase().contains("scores");

        if in_confidence_section || result.has_completion_signal {
            for cap in re.captures_iter(text) {
                if let (Some(name), Some(score_str)) = (cap.get(1), cap.get(2)) {
                    let name = name.as_str().to_lowercase();
                    // Skip common false positives
                    if name == "confidence" || name == "scores" || name == "task" {
                        continue;
                    }
                    if let Ok(score) = score_str.as_str().parse::<u8>() {
                        if score <= 100 {
                            result.scores.push((name, score));
                        }
                    }
                }
            }
        }
    }

    // Calculate min score
    if !result.scores.is_empty() {
        result.min_score = result.scores.iter().map(|(_, s)| *s).min();
    }

    result
}

/// Determine if response looks like it finished naturally (not stuck/interrupted)
fn looks_finished(text: &str) -> bool {
    let text_lower = text.to_lowercase();

    // Positive signals - looks complete
    let complete_signals = [
        "let me know if",
        "feel free to ask",
        "hope this helps",
        "is there anything else",
        "here's the",
        "i've implemented",
        "i've completed",
        "the implementation is",
        "ready for review",
        "all tests pass",
    ];

    // Negative signals - looks incomplete/stuck
    let incomplete_signals = [
        "i'll continue",
        "next, i'll",
        "let me",
        "i need to",
        "working on",
        "in progress",
        "error:",
        "failed:",
        "cannot",
        "unable to",
    ];

    let complete_count = complete_signals.iter().filter(|s| text_lower.contains(*s)).count();
    let incomplete_count = incomplete_signals.iter().filter(|s| text_lower.contains(*s)).count();

    // Consider finished if more complete signals than incomplete, or at least 2 complete signals
    complete_count > incomplete_count || complete_count >= 2
}

/// Select verification tier based on parsed confidence scores
fn select_tier_from_scores(scores: &CompletionScores) -> VerificationTier {
    match scores.min_score {
        Some(s) if s >= 95 => VerificationTier::Quick,   // High confidence - quick verify
        Some(s) if s >= 70 => VerificationTier::Medium,  // Medium - 8b check
        Some(s) if s >= 50 => VerificationTier::Medium,  // Low-medium - 8b check
        Some(_) => VerificationTier::Deep,               // Low (<50) - need 24b help
        None => VerificationTier::Medium,                // No scores - default to 8b
    }
}

/// Get weak categories (score < 70) that need attention
fn get_weak_categories(scores: &CompletionScores) -> Vec<(String, u8)> {
    scores.scores.iter()
        .filter(|(_, s)| *s < 70)
        .cloned()
        .collect()
}

/// Prompt for 3b quick classification
const CLASSIFY_PROMPT_3B: &str = r#"Classify this assistant response. Output ONLY one word: DONE, PROGRESS, STUCK, or UNSURE.

DONE = task complete, success criteria clearly met, builds passing
PROGRESS = making progress, more work needed
STUCK = blocked, errors, build failures, can't proceed
UNSURE = unclear if complete or not

Build Status:
{build_status}

Response:
{response}

Classification:"#;

/// Prompt for 8b second opinion
const CLASSIFY_PROMPT_8B: &str = r#"The quick classifier was unsure. Analyze this response and classify.
Output format:
STATUS: [DONE/PROGRESS/STUCK]
REASON: [one sentence]

Build Status (from project):
{build_status}

Response:
{response}

Original request context:
{request}

Analysis:"#;

/// Prompt for 24b advice when stuck
const ADVICE_PROMPT_24B: &str = r#"The assistant is stuck. Analyze the situation and provide concrete advice to unblock.

Build Status (from project):
{build_status}

Response showing the stuck state:
{response}

Original task:
{request}

Consider the build status when giving advice. If builds are failing, prioritize fixing those errors.
Provide specific, actionable advice (2-3 sentences max):"#;

/// Prompt for generating continuation message
const CONTINUATION_PROMPT: &str = r#"Summarize progress and what should be done next. Be brief (1-2 sentences).

Response so far:
{response}

Summary:"#;

/// Check build status by running cargo build and parsing JSON output
async fn check_build_status() -> ProjectStatus {
    use std::process::Command;
    use std::time::SystemTime;

    let output = match Command::new("cargo")
        .arg("build")
        .arg("--message-format=json")
        .output()
    {
        Ok(out) => out,
        Err(e) => {
            return ProjectStatus {
                build_passing: false,
                build_errors: vec![format!("Failed to run cargo build: {}", e)],
                test_passing: false,
                test_failures: vec![],
                last_build_check: Some(SystemTime::now()),
                last_test_check: None,
            };
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut build_errors = Vec::new();

    // Parse JSON messages from cargo
    for line in stdout.lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if msg["reason"] == "compiler-message" {
                if let Some(message) = msg["message"].as_object() {
                    if message["level"] == "error" {
                        let rendered = message["rendered"].as_str().unwrap_or("");
                        if !rendered.is_empty() {
                            build_errors.push(rendered.to_string());
                        }
                    }
                }
            }
        }
    }

    let build_passing = output.status.success() && build_errors.is_empty();

    ProjectStatus {
        build_passing,
        build_errors,
        test_passing: false,
        test_failures: vec![],
        last_build_check: Some(SystemTime::now()),
        last_test_check: None,
    }
}

/// Check test status by actually RUNNING cargo test (not just compiling)
async fn check_test_status() -> ProjectStatus {
    check_test_status_in_dir(None).await
}

/// Check test status in a specific directory
async fn check_test_status_in_dir(project_dir: Option<&str>) -> ProjectStatus {
    use std::process::Command;
    use std::time::SystemTime;

    // Build the command
    let mut cmd = Command::new("cargo");
    cmd.arg("test");

    // Run in specific directory if provided
    if let Some(dir) = project_dir {
        cmd.current_dir(dir);
    }

    let output = match cmd.output() {
        Ok(out) => out,
        Err(e) => {
            return ProjectStatus {
                build_passing: false,
                build_errors: vec![],
                test_passing: false,
                test_failures: vec![format!("Failed to run cargo test: {}", e)],
                last_build_check: None,
                last_test_check: Some(SystemTime::now()),
            };
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}\n{}", stdout, stderr);

    let mut test_failures = Vec::new();
    let mut in_failures_section = false;
    let mut current_failure = String::new();

    // Parse test output for failures
    // Format: "test result: FAILED. X passed; Y failed; Z ignored"
    // Or look for "failures:" section followed by test names
    for line in combined.lines() {
        // Detect the failures section
        if line.trim() == "failures:" {
            in_failures_section = true;
            continue;
        }

        // End of failures section
        if in_failures_section && (line.starts_with("test result:") || line.trim().is_empty() && !current_failure.is_empty()) {
            if !current_failure.is_empty() {
                test_failures.push(current_failure.trim().to_string());
                current_failure.clear();
            }
            in_failures_section = line.starts_with("failures:");
            continue;
        }

        // Capture failed test names
        if in_failures_section {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                // Skip the "---- test_name stdout ----" separators
                if trimmed.starts_with("----") && trimmed.ends_with("----") {
                    if !current_failure.is_empty() {
                        test_failures.push(current_failure.trim().to_string());
                        current_failure.clear();
                    }
                    // Extract test name from "---- test_name stdout ----"
                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                    if parts.len() >= 2 {
                        current_failure = parts[1].to_string();
                    }
                } else if !trimmed.starts_with("note:") && current_failure.is_empty() {
                    // Test name in the summary list
                    test_failures.push(trimmed.to_string());
                }
            }
        }

        // Also catch explicit FAILED lines
        if line.contains("test ") && line.contains("... FAILED") {
            // Extract test name: "test module::test_name ... FAILED"
            if let Some(test_part) = line.strip_prefix("test ") {
                if let Some(name) = test_part.split(" ... ").next() {
                    let name = name.trim();
                    if !test_failures.contains(&name.to_string()) {
                        test_failures.push(name.to_string());
                    }
                }
            }
        }
    }

    // Check exit status
    let test_passing = output.status.success();

    // If exit status says failed but we didn't parse failures, note that
    if !test_passing && test_failures.is_empty() {
        test_failures.push("Tests failed (check output for details)".to_string());
    }

    ProjectStatus {
        build_passing: false,
        build_errors: vec![],
        test_passing,
        test_failures,
        last_build_check: None,
        last_test_check: Some(SystemTime::now()),
    }
}

/// Update project status by reading from watchd (pal watch daemon)
async fn update_project_status(state: &AppState) {
    let watchd_status = read_watchd_status(None).await;

    let mut status = state.project_status.write().await;
    status.build_passing = watchd_status.build_passing;
    status.build_errors = watchd_status.build_errors;
    status.test_passing = watchd_status.test_passing;
    status.test_failures = watchd_status.test_failures;
    status.last_build_check = Some(std::time::SystemTime::now());
    status.last_test_check = Some(std::time::SystemTime::now());
}

/// Format status for human readability
fn format_status(status: &ProjectStatus) -> String {
    let mut output = String::new();

    if status.build_passing && status.test_passing {
        return "ok".to_string();
    }

    if !status.build_passing {
        output.push_str("Build: FAILING\n");
        for (i, error) in status.build_errors.iter().enumerate() {
            output.push_str(&format!("  Error {}:\n", i + 1));
            for line in error.lines() {
                output.push_str(&format!("    {}\n", line));
            }
        }
    } else {
        output.push_str("Build: PASSING\n");
    }

    if !status.test_passing {
        output.push_str("Tests: FAILING\n");
        for (i, error) in status.test_failures.iter().enumerate() {
            output.push_str(&format!("  Error {}:\n", i + 1));
            for line in error.lines() {
                output.push_str(&format!("    {}\n", line));
            }
        }
    } else {
        output.push_str("Tests: PASSING\n");
    }

    output
}

/// HTTP handler: GET /status/build
async fn get_build_status_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let status = state.project_status.read().await;
    Json(serde_json::json!({
        "build_passing": status.build_passing,
        "build_errors": status.build_errors,
        "last_build_check": status.last_build_check.map(|t| format!("{:?}", t)),
    }))
}

/// HTTP handler: GET /status/tests
async fn get_test_status_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let status = state.project_status.read().await;
    Json(serde_json::json!({
        "test_passing": status.test_passing,
        "test_failures": status.test_failures,
        "last_test_check": status.last_test_check.map(|t| format!("{:?}", t)),
    }))
}

/// HTTP handler: GET /status
async fn get_status_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let status = state.project_status.read().await;
    let formatted = format_status(&status);

    Json(serde_json::json!({
        "build_passing": status.build_passing,
        "build_errors": status.build_errors,
        "test_passing": status.test_passing,
        "test_failures": status.test_failures,
        "last_build_check": status.last_build_check.map(|t| format!("{:?}", t)),
        "last_test_check": status.last_test_check.map(|t| format!("{:?}", t)),
        "formatted": formatted,
    }))
}

/// HTTP handler: POST /status/update
async fn update_status_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    update_project_status(&state).await;

    let status = state.project_status.read().await;
    Json(serde_json::json!({
        "status": "updated",
        "build_passing": status.build_passing,
        "test_passing": status.test_passing,
    }))
}

/// Get build status from `pal build-status` command
/// Returns (status_text, is_available)
async fn get_build_status() -> (String, bool) {
    use std::process::Command;

    // Try to run pal build-status
    match Command::new("pal")
        .arg("build-status")
        .output()
    {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            if output.status.success() {
                let status = stdout.trim().to_string();
                if status.is_empty() {
                    (stderr.trim().to_string(), !stderr.is_empty())
                } else {
                    (status, true)
                }
            } else {
                // Command failed - maybe watchd is down
                if stderr.contains("watchd") || stderr.contains("daemon") || stderr.contains("connection") {
                    (format!("Build status unavailable (watchd daemon not running). Run `pal build-status` manually to check builds.\nError: {}", stderr.trim()), false)
                } else {
                    (format!("Build status check failed: {}", stderr.trim()), false)
                }
            }
        }
        Err(e) => {
            // pal command not found or couldn't execute
            (format!("Build status unavailable (pal command error: {}). Run `cargo test` and `cargo build` manually to verify build status.", e), false)
        }
    }
}

/// Get build status from internal ProjectStatus
/// Returns (status_text, is_available)
async fn get_build_status_from_state(project_status: Arc<RwLock<ProjectStatus>>) -> (String, bool) {
    let status = project_status.read().await;

    // Check if status has been initialized
    if status.last_build_check.is_none() && status.last_test_check.is_none() {
        return ("Build status not yet checked. Trigger update with POST /status/update".to_string(), false);
    }

    let formatted = format_status(&status);
    let _is_ok = status.build_passing && status.test_passing;

    if formatted == "ok" {
        ("ok".to_string(), true)
    } else {
        (formatted, true)
    }
}

/// Call Ollama API with a prompt
async fn call_ollama(
    http_client: &reqwest::Client,
    ollama_url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<String, String> {
    let request_body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "temperature": 0.1,
            "num_predict": max_tokens,
        }
    });

    let url = format!("{}/api/generate", ollama_url);

    let response = http_client
        .post(&url)
        .json(&request_body)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| format!("Ollama request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Ollama returned {}", response.status()));
    }

    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Ollama response: {}", e))?;

    Ok(body["response"].as_str().unwrap_or("").to_string())
}

/// Check if Claude Code CLI is available
async fn is_claude_code_available() -> bool {
    use tokio::process::Command;

    match Command::new("claude")
        .arg("--version")
        .output()
        .await
    {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

/// Check if Codex CLI is available
async fn is_codex_available() -> bool {
    use tokio::process::Command;

    match Command::new("codex")
        .arg("--version")
        .output()
        .await
    {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

/// Select the best available smart orchestrator
async fn select_smart_orchestrator() -> Option<SmartOrchestrator> {
    // Prefer Claude Code as it's more capable
    if is_claude_code_available().await {
        return Some(SmartOrchestrator::ClaudeCode);
    }

    if is_codex_available().await {
        return Some(SmartOrchestrator::Codex);
    }

    None
}

/// Invoke Claude Code CLI for smart orchestration
/// This uses the flatrate Claude Code subscription for ToS-friendly usage
/// Mirrors the approach in palace.py _claude_call method
async fn invoke_claude_code_orchestrator(
    task: &str,
    context: &str,
    timeout_secs: u64,
) -> Result<String, OrchestrationError> {
    use tokio::process::Command;
    use tokio::time::{timeout, Duration};

    info!("SMART: Invoking Claude Code orchestrator for task");

    // Build the prompt for Claude Code (same format as palace.py)
    let full_prompt = format!(
        "You are orchestrating a coding task. Analyze this and provide step-by-step instructions.\n\n\
        CONTEXT:\n{}\n\n\
        TASK:\n{}\n\n\
        Provide a clear action plan with specific steps. For each step, indicate:\n\
        - What model tier should handle it (premium/standard/cheap/local)\n\
        - What the expected output is\n\
        - Any dependencies on previous steps",
        context, task
    );

    let result = timeout(
        Duration::from_secs(timeout_secs),
        async {
            // Use claude -p with --output-format stream-json like palace.py does
            let output = Command::new("claude")
                .args([
                    "-p", &full_prompt,
                    "--model", "claude-sonnet-4-5",  // Sonnet for orchestration (ToS-friendly Max sub)
                    "--max-tokens", "4096",
                    "--output-format", "stream-json",
                ])
                .output()
                .await
                .map_err(OrchestrationError::ProcessSpawn)?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if stderr.contains("rate limit") || stderr.contains("429") {
                    return Err(OrchestrationError::RateLimited);
                }
                return Err(OrchestrationError::OutputParse(
                    format!("Claude Code failed: {}", stderr)
                ));
            }

            // Parse stream-json output to extract text (same as palace.py)
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut output_text = String::new();

            for line in stdout.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(event) = serde_json::from_str::<serde_json::Value>(line) {
                    // Handle assistant message type
                    if event.get("type").and_then(|t| t.as_str()) == Some("assistant") {
                        if let Some(msg) = event.get("message") {
                            if let Some(content) = msg.get("content").and_then(|c| c.as_array()) {
                                for block in content {
                                    if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                            output_text.push_str(text);
                                        }
                                    }
                                }
                            } else if let Some(msg_str) = msg.as_str() {
                                output_text.push_str(msg_str);
                            }
                        }
                    }
                    // Handle content_block_delta type
                    else if event.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                        if let Some(delta) = event.get("delta") {
                            if delta.get("type").and_then(|t| t.as_str()) == Some("text_delta") {
                                if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                    output_text.push_str(text);
                                }
                            }
                        }
                    }
                }
            }

            if output_text.is_empty() {
                // Fallback to raw stdout if parsing failed
                Ok(stdout.to_string())
            } else {
                Ok(output_text)
            }
        }
    ).await;

    match result {
        Ok(inner) => inner,
        Err(_) => Err(OrchestrationError::Timeout),
    }
}

/// Invoke Codex CLI for smart orchestration
/// This uses the OpenAI Codex CLI for ToS-friendly usage
async fn invoke_codex_orchestrator(
    task: &str,
    context: &str,
    timeout_secs: u64,
) -> Result<String, OrchestrationError> {
    use tokio::process::Command;
    use tokio::time::{timeout, Duration};

    info!("SMART: Invoking Codex orchestrator for task");

    let full_prompt = format!(
        "Orchestrate this coding task:\n\nCONTEXT:\n{}\n\nTASK:\n{}\n\n\
        Provide step-by-step instructions with model tier recommendations.",
        context, task
    );

    let result = timeout(
        Duration::from_secs(timeout_secs),
        async {
            let output = Command::new("codex")
                .args([
                    "--approval-mode=full-auto",
                    "--quiet",
                    "-m", &full_prompt,
                ])
                .output()
                .await
                .map_err(OrchestrationError::ProcessSpawn)?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if stderr.contains("rate limit") || stderr.contains("429") {
                    return Err(OrchestrationError::RateLimited);
                }
                return Err(OrchestrationError::OutputParse(
                    format!("Codex failed: {}", stderr)
                ));
            }

            String::from_utf8(output.stdout)
                .map_err(|e| OrchestrationError::OutputParse(e.to_string()))
        }
    ).await;

    match result {
        Ok(inner) => inner,
        Err(_) => Err(OrchestrationError::Timeout),
    }
}

/// Run smart mode orchestration
/// Returns a structured plan for the task
async fn run_smart_orchestration(
    task: &str,
    context: &str,
    config: &SmartConfig,
) -> Result<SmartOrchestrationPlan, OrchestrationError> {
    let orchestrator = select_smart_orchestrator().await
        .ok_or_else(|| OrchestrationError::OutputParse(
            "No smart orchestrator available (install claude or codex CLI)".to_string()
        ))?;

    info!("SMART: Using {:?} as orchestrator", orchestrator);

    let timeout_secs = 120; // 2 minute timeout for orchestration

    let output = match orchestrator {
        SmartOrchestrator::ClaudeCode => {
            invoke_claude_code_orchestrator(task, context, timeout_secs).await?
        }
        SmartOrchestrator::Codex => {
            invoke_codex_orchestrator(task, context, timeout_secs).await?
        }
    };

    // Parse the orchestration output into a plan
    Ok(SmartOrchestrationPlan {
        orchestrator,
        raw_plan: output,
        worker_tier: config.worker_tier.clone(),
    })
}

/// Check if a model can declare completion without premium verification
/// Premium models (Opus, Sonnet, Devstral API, GPT-4) can complete directly
/// Local/cheap models must have completion verified by premium model
fn can_declare_complete(model_name: &str) -> bool {
    let model_lower = model_name.to_lowercase();

    // Premium models CAN complete
    if model_lower.contains("opus") { return true; }
    if model_lower.contains("sonnet") { return true; }
    if model_lower.contains("devstral") && !model_lower.contains("mini") { return true; }
    if model_lower.contains("gpt-4") { return true; }

    // Local/cheap models CANNOT complete
    if model_lower.contains("qwen") { return false; }
    if model_lower.contains("mini") { return false; }
    if model_lower.contains("local") { return false; }
    if model_lower.contains("3b") || model_lower.contains("4b") || model_lower.contains("8b") { return false; }

    // Default: allow (unknown premium model)
    true
}

/// Parse 3b classification response into status
fn parse_3b_classification(text: &str) -> ContinuousStatus {
    let text_upper = text.trim().to_uppercase();

    // Look for the classification word
    if text_upper.contains("DONE") {
        ContinuousStatus::Done
    } else if text_upper.contains("STUCK") {
        ContinuousStatus::Stuck
    } else if text_upper.contains("UNSURE") {
        ContinuousStatus::Unsure
    } else if text_upper.contains("PROGRESS") {
        ContinuousStatus::Progress
    } else {
        // Default to unsure if can't parse
        ContinuousStatus::Unsure
    }
}

/// Parse 8b second opinion into status
fn parse_8b_classification(text: &str) -> ContinuousStatus {
    let text_upper = text.to_uppercase();

    // Look for STATUS: line
    if text_upper.contains("STATUS: DONE") || text_upper.contains("STATUS:DONE") {
        ContinuousStatus::Done
    } else if text_upper.contains("STATUS: STUCK") || text_upper.contains("STATUS:STUCK") {
        ContinuousStatus::Stuck
    } else if text_upper.contains("STATUS: PROGRESS") || text_upper.contains("STATUS:PROGRESS") {
        ContinuousStatus::Progress
    } else {
        // Fallback to keyword detection
        parse_3b_classification(text)
    }
}

/// Run the full continuous mode verification flow
/// Two paths:
/// 1. Model self-reports completion with scores -> parse scores, route by lowest
/// 2. Model just stops without signal -> run 3b->8b->24b chain
async fn run_continuous_verification(
    http_client: &reqwest::Client,
    config: &ContinuousConfig,
    response_text: &str,
    original_request: &str,
    actual_model: &str,
    project_status: Arc<RwLock<ProjectStatus>>,
    strict_mode: &StrictMode,
) -> Result<VerificationResult, String> {
    // Get build status from internal state (fallback to pal build-status if not initialized)
    let (build_status, build_available) = get_build_status_from_state(project_status.clone()).await;
    info!("CONTINUOUS: Build status available: {}", build_available);
    if build_available {
        info!("CONTINUOUS: Build status: {}", build_status.lines().next().unwrap_or("(empty)"));
    } else {
        info!("CONTINUOUS: Build status unavailable - will include manual check instructions");
    }

    // STRICT MODE: If enabled, completion is BLOCKED until all tests pass
    // Always fetch fresh status from watchd (not cached ProjectStatus)
    if *strict_mode == StrictMode::On {
        // Get fresh verification status from watchd API
        let watchd_status = read_watchd_status(None).await;

        // Also update the shared ProjectStatus for other uses
        {
            let mut status = project_status.write().await;
            status.build_passing = watchd_status.build_passing;
            status.build_errors = watchd_status.build_errors.clone();
            status.test_passing = watchd_status.test_passing;
            status.test_failures = watchd_status.test_failures.clone();
            status.last_build_check = Some(std::time::SystemTime::now());
            status.last_test_check = Some(std::time::SystemTime::now());
        }

        if !watchd_status.build_passing || !watchd_status.test_passing {
            info!("CONTINUOUS: STRICT MODE - Build/tests failing, blocking completion");
            let mut issues = vec![];
            if !watchd_status.build_passing {
                issues.extend(watchd_status.build_errors.iter().take(3).cloned());
            }
            if !watchd_status.test_passing {
                issues.extend(watchd_status.test_failures.iter().take(3).cloned());
            }

            return Ok(VerificationResult {
                status: ContinuousStatus::Progress,
                tier: VerificationTier::Quick,
                advice: Some(format!("STRICT MODE: Cannot complete until ALL builds and tests pass.\n\nCurrent issues:\n{}", issues.join("\n"))),
                progress_summary: Some("Build/tests failing - fix required".to_string()),
                weak_categories: vec![("build".to_string(), 0), ("tests".to_string(), 0)],
            });
        }
        info!("CONTINUOUS: STRICT MODE - Build and tests passing, completion allowed");
    }

    // PATH 1: Check for explicit completion with scores
    let scores = parse_completion_scores(response_text);

    if scores.has_completion_signal && !scores.scores.is_empty() {
        // Model self-reported completion with confidence scores
        info!("CONTINUOUS: Model reported TASK COMPLETE with {} scores", scores.scores.len());
        for (cat, score) in &scores.scores {
            info!("  {}: {}", cat, score);
        }

        let weak = get_weak_categories(&scores);
        let tier = select_tier_from_scores(&scores);

        if let Some(min) = scores.min_score {
            if min >= 95 {
                // All scores high - check if model can declare completion
                if !can_declare_complete(actual_model) {
                    // Local model attempted completion - escalate to premium review
                    info!("CONTINUOUS: Local model {} attempted completion - escalating to premium review", actual_model);
                    let prompt = format!("A local model reported TASK COMPLETE with high scores (min={}). Review the work and confirm if it's truly done.\n\nBuild status: {}\n\nScores: {:?}\n\nResponse:\n{}\n\nReply with STATUS: DONE if complete, or STATUS: PROGRESS with specific improvements needed.", min, build_status, scores.scores, response_text);
                    let result = call_ollama(http_client, &config.ollama_url, &config.medium_model, &prompt, 200).await?;
                    let status = parse_8b_classification(&result);

                    if status == ContinuousStatus::Done {
                        info!("CONTINUOUS: Premium model confirmed completion");
                    } else {
                        info!("CONTINUOUS: Premium model requires more work");
                    }

                    return Ok(VerificationResult {
                        status,
                        tier: VerificationTier::Medium,
                        advice: None,
                        progress_summary: if status == ContinuousStatus::Progress {
                            Some("Premium model review required completion".to_string())
                        } else { None },
                        weak_categories: weak,
                    });
                }

                // Premium model or verified local completion - truly done
                info!("CONTINUOUS: All scores >= 95, task complete");
                return Ok(VerificationResult {
                    status: ContinuousStatus::Done,
                    tier,
                    advice: None,
                    progress_summary: None,
                    weak_categories: weak,
                });
            } else if min >= 70 {
                // Good but not perfect - might need polish
                info!("CONTINUOUS: Min score {} (>=70), checking if truly done", min);
                // Quick 3b check to confirm - include build status
                let prompt = format!("Is this task truly complete?\nBuild status: {}\nScores reported: {:?}\nResponse ends with completion signal. Reply DONE or PROGRESS.", build_status, scores.scores);
                let result = call_ollama(http_client, &config.ollama_url, &config.quick_model, &prompt, 20).await?;
                let status = parse_3b_classification(&result);
                return Ok(VerificationResult {
                    status,
                    tier,
                    advice: None,
                    progress_summary: if status == ContinuousStatus::Progress {
                        Some(format!("Weak categories: {:?}", weak))
                    } else { None },
                    weak_categories: weak,
                });
            } else {
                // Low scores - needs more work
                info!("CONTINUOUS: Min score {} (<70), needs more work on {:?}", min, weak);

                // Get advice from appropriate tier model
                let advice = if min < 50 {
                    // Deep trouble - use 24b
                    let prompt = ADVICE_PROMPT_24B
                        .replace("{build_status}", &build_status)
                        .replace("{response}", response_text)
                        .replace("{request}", original_request);
                    Some(call_ollama(http_client, &config.ollama_url, &config.deep_model, &prompt, 300).await?)
                } else {
                    // Medium trouble - use 8b
                    let prompt = format!("Build status: {}\nWeak areas: {:?}. Suggest specific improvements in 1-2 sentences.", build_status, weak);
                    Some(call_ollama(http_client, &config.ollama_url, &config.medium_model, &prompt, 150).await?)
                };

                return Ok(VerificationResult {
                    status: ContinuousStatus::Progress,
                    tier,
                    advice,
                    progress_summary: Some(format!("Focus on: {:?}", weak)),
                    weak_categories: weak,
                });
            }
        }
    }

    // PATH 2: No explicit completion signal - check if looks finished
    if scores.has_completion_signal && scores.scores.is_empty() {
        // Has completion signal but no scores - parse failed, use 8b fallback
        info!("CONTINUOUS: Completion signal found but no scores parsed, using 8b fallback");
        let prompt_8b = CLASSIFY_PROMPT_8B
            .replace("{build_status}", &build_status)
            .replace("{response}", response_text)
            .replace("{request}", original_request);
        let result_8b = call_ollama(http_client, &config.ollama_url, &config.medium_model, &prompt_8b, 100).await?;
        let status = parse_8b_classification(&result_8b);
        info!("CONTINUOUS: 8b fallback classified as {:?}", status);

        return Ok(VerificationResult {
            status,
            tier: VerificationTier::Medium,
            advice: None,
            progress_summary: None,
            weak_categories: vec![],
        });
    }

    // No completion signal at all - check if response looks finished naturally
    if looks_finished(response_text) {
        // Check if model can declare completion
        if !can_declare_complete(actual_model) {
            // Local model appears finished - verify with premium model
            info!("CONTINUOUS: Local model {} appears finished - verifying with premium model", actual_model);
            let prompt = format!("A local model finished without explicit completion signal. Does this work look complete?\n\nBuild status: {}\n\nResponse:\n{}\n\nReply with STATUS: DONE if complete, or STATUS: PROGRESS with what's missing.", build_status, response_text);
            let result = call_ollama(http_client, &config.ollama_url, &config.medium_model, &prompt, 200).await?;
            let status = parse_8b_classification(&result);

            return Ok(VerificationResult {
                status,
                tier: VerificationTier::Medium,
                advice: None,
                progress_summary: if status == ContinuousStatus::Progress {
                    Some("Premium model review suggested more work".to_string())
                } else { None },
                weak_categories: vec![],
            });
        }

        info!("CONTINUOUS: No completion signal but response looks finished");
        return Ok(VerificationResult {
            status: ContinuousStatus::Done,
            tier: VerificationTier::Quick,
            advice: None,
            progress_summary: None,
            weak_categories: vec![],
        });
    }

    // PATH 2b: Model just stopped - run 3b classification chain
    info!("CONTINUOUS: No completion signal, running 3b classification");
    let prompt_3b = CLASSIFY_PROMPT_3B
        .replace("{build_status}", &build_status)
        .replace("{response}", response_text);
    let result_3b = call_ollama(http_client, &config.ollama_url, &config.quick_model, &prompt_3b, 20).await?;
    let status_3b = parse_3b_classification(&result_3b);
    info!("CONTINUOUS: 3b classified as {:?}", status_3b);

    // If UNSURE, get 8b second opinion
    let status = if status_3b == ContinuousStatus::Unsure {
        let prompt_8b = CLASSIFY_PROMPT_8B
            .replace("{build_status}", &build_status)
            .replace("{response}", response_text)
            .replace("{request}", original_request);
        info!("CONTINUOUS: Running 8b second opinion");
        let result_8b = call_ollama(http_client, &config.ollama_url, &config.medium_model, &prompt_8b, 100).await?;
        let status_8b = parse_8b_classification(&result_8b);
        info!("CONTINUOUS: 8b classified as {:?}", status_8b);
        status_8b
    } else {
        status_3b
    };

    // If STUCK, get 24b advice
    let advice = if status == ContinuousStatus::Stuck {
        let prompt_24b = ADVICE_PROMPT_24B
            .replace("{build_status}", &build_status)
            .replace("{response}", response_text)
            .replace("{request}", original_request);
        info!("CONTINUOUS: Getting 24b advice");
        let result_24b = call_ollama(http_client, &config.ollama_url, &config.deep_model, &prompt_24b, 300).await?;
        info!("CONTINUOUS: 24b advice: {}", result_24b.trim());
        Some(result_24b.trim().to_string())
    } else {
        None
    };

    // If PROGRESS, get continuation summary
    let progress_summary = if status == ContinuousStatus::Progress {
        let prompt_cont = CONTINUATION_PROMPT.replace("{response}", response_text);
        let result_cont = call_ollama(http_client, &config.ollama_url, &config.quick_model, &prompt_cont, 100).await?;
        Some(result_cont.trim().to_string())
    } else {
        None
    };

    let tier = match status {
        ContinuousStatus::Done | ContinuousStatus::Progress => VerificationTier::Quick,
        ContinuousStatus::Unsure => VerificationTier::Medium,
        ContinuousStatus::Stuck => VerificationTier::Deep,
    };

    Ok(VerificationResult {
        status,
        tier,
        advice,
        progress_summary,
        weak_categories: vec![],
    })
}

/// Synthesize continuation message to inject as next user turn
fn synthesize_continuation_message(result: &VerificationResult) -> String {
    // Build weak categories hint if available
    let weak_hint = if !result.weak_categories.is_empty() {
        let cats: Vec<String> = result.weak_categories.iter()
            .map(|(name, score)| format!("{} ({})", name, score))
            .collect();
        format!(" Focus on improving: {}.", cats.join(", "))
    } else {
        String::new()
    };

    match result.status {
        ContinuousStatus::Progress => {
            if let Some(ref summary) = result.progress_summary {
                format!("Continue. {}{}", summary, weak_hint)
            } else if !weak_hint.is_empty() {
                format!("Continue working on the task.{}", weak_hint)
            } else {
                "Continue working on the task.".to_string()
            }
        }
        ContinuousStatus::Stuck => {
            if let Some(ref advice) = result.advice {
                format!("You appear stuck. Here's advice: {}{}\n\nTry a different approach based on this.", advice, weak_hint)
            } else {
                format!("You appear stuck.{} Try a different approach.", weak_hint)
            }
        }
        ContinuousStatus::Unsure => {
            format!("Continue if there's more work to do, or summarize what's complete.{}", weak_hint)
        }
        ContinuousStatus::Done => {
            "Task complete.".to_string()
        }
    }
}

// Keep the old function signature for compatibility but mark unused
#[allow(dead_code)]
fn select_verification_tier(response_text: &str) -> VerificationTier {
    let text_lower = response_text.to_lowercase();

    let disaster_signals = [
        "error:", "failed:", "panic", "stack trace", "exception",
        "cannot", "unable to", "i don't know", "i'm not sure",
    ];

    let completion_signals = [
        "complete", "done", "finished", "all tests pass", "successfully",
        "implemented", "works as expected", "ready for review",
    ];

    let disaster_count = disaster_signals.iter()
        .filter(|s| text_lower.contains(*s))
        .count();

    let completion_count = completion_signals.iter()
        .filter(|s| text_lower.contains(*s))
        .count();

    if disaster_count >= 2 {
        VerificationTier::Deep // Disaster - use 24b
    } else if completion_count >= 2 && disaster_count == 0 {
        VerificationTier::Quick // Likely complete - use 3b
    } else {
        VerificationTier::Medium // Uncertain - use 8b
    }
}

#[derive(Debug)]
struct ToolBlockState {
    block_index: i32,
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    name: String,
}

/// Application error types
#[derive(Debug)]
enum AppError {
    UnknownModel(String),
    MissingApiKey(String),
    Upstream(String),
    QueueFull,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::UnknownModel(model) => (
                StatusCode::BAD_REQUEST,
                format!("Unknown model: {}", model),
            ),
            AppError::MissingApiKey(env_var) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Missing API key: {} not set", env_var),
            ),
            AppError::Upstream(msg) => (
                StatusCode::BAD_GATEWAY,
                format!("Upstream error: {}", msg),
            ),
            AppError::QueueFull => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Request queue full - too many pending requests".to_string(),
            ),
        };

        let body = serde_json::json!({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": message
            }
        });

        (status, Json(body)).into_response()
    }
}

// ============================================================================
// SWARM ORCHESTRATION MODULE
// ============================================================================
//
// Implements parallel worker spawning for hypermiler mode.
// When @swarm is enabled, the orchestrator spawns multiple Claude Code sessions
// to work on independent tasks in parallel.

/// A task generated by the planner for swarm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SwarmTask {
    /// Task number/ID
    num: String,
    /// Human-readable task label
    label: String,
    /// Optional description/details
    description: String,
}

/// Information about a running swarm worker
struct SwarmWorker {
    /// Unique worker ID (e.g., "worker-1")
    id: String,
    /// The task this worker is executing
    task: SwarmTask,
    /// The spawned Claude CLI process
    process: std::process::Child,
    /// Model being used by this worker
    model: String,
    /// Accumulated output from this worker
    output_buffer: String,
    /// Whether this worker has finished
    finished: bool,
}

/// Result from a completed swarm worker
#[derive(Debug, Clone)]
struct SwarmWorkerResult {
    /// Worker ID
    id: String,
    /// Task that was executed
    task: SwarmTask,
    /// Exit code from Claude CLI
    exit_code: Option<i32>,
    /// Full output from the worker
    output: String,
    /// Whether task completed successfully
    success: bool,
}

/// Build task planning prompt with project context
fn build_task_planning_prompt(project_path: Option<&str>, build_status: &VerificationStatus) -> String {
    let project_info = if let Some(path) = project_path {
        format!("**Project:** {}\n", path)
    } else {
        String::new()
    };

    let build_info = format!(
        "**Build Status:** {}\n**Test Status:** {}",
        if build_status.build_passing { "✅ PASSING" } else { "❌ FAILING" },
        if build_status.test_passing { "✅ PASSING" } else { "❌ FAILING" }
    );

    let errors = if !build_status.build_errors.is_empty() || !build_status.test_failures.is_empty() {
        let mut err_text = String::new();
        if !build_status.build_errors.is_empty() {
            err_text.push_str("\n**Build Errors:**\n");
            for e in &build_status.build_errors {
                err_text.push_str(&format!("- {}\n", e));
            }
        }
        if !build_status.test_failures.is_empty() {
            err_text.push_str("\n**Test Failures:**\n");
            for f in &build_status.test_failures {
                err_text.push_str(&format!("- {}\n", f));
            }
        }
        err_text
    } else {
        String::new()
    };

    format!(r#"# Task Planning Request

{}{}{}

## Instructions

Analyze this project and determine what work should be done next.

Consider:
- Current build and test status (fix any failures first!)
- What features/fixes are most valuable
- What can be done in parallel vs sequentially

Output a YAML block with parallel tasks to execute:

```yaml
actions:
  - label: Fix the auth module
    description: The auth tests are failing, need to fix the token validation
  - label: Add user profile endpoint
    description: New API endpoint at /api/user/profile
  - label: Update documentation
    description: README needs updating for new features
```

Each task should be:
- Independent (can run in parallel with others)
- Specific and actionable
- Completable in a single Claude Code session

Generate as many tasks as make sense for the project state. Could be 1 task if there's one clear priority, or 10+ if there's lots of independent work to parallelize.
"#, project_info, build_info, errors)
}

/// Parse YAML actions block from planner response
fn parse_tasks_from_response(response: &str) -> Vec<SwarmTask> {
    use regex::Regex;
    let mut tasks = Vec::new();

    // Find YAML code blocks
    let yaml_pattern = Regex::new(r"```(?:yaml|yml)?\s*\n([\s\S]*?)```").ok();
    if yaml_pattern.is_none() {
        return tasks;
    }

    for caps in yaml_pattern.unwrap().captures_iter(response) {
        if let Some(yaml_content) = caps.get(1) {
            // Simple YAML parsing for actions list
            // Looking for "actions:" followed by "- label: ..." items
            let content = yaml_content.as_str();

            // Check if this contains "actions:"
            if !content.contains("actions:") {
                continue;
            }

            // Parse each item (simplified regex-based parsing)
            let item_pattern = Regex::new(r"-\s*label:\s*([^\n]+)(?:\s*description:\s*([^\n]+))?").ok();
            if let Some(re) = item_pattern {
                let mut num = 1;
                for caps in re.captures_iter(content) {
                    let label = caps.get(1)
                        .map(|m| m.as_str().trim().to_string())
                        .unwrap_or_default();
                    let description = caps.get(2)
                        .map(|m| m.as_str().trim().to_string())
                        .unwrap_or_default();

                    if !label.is_empty() {
                        tasks.push(SwarmTask {
                            num: num.to_string(),
                            label,
                            description,
                        });
                        num += 1;
                    }
                }
            }

            // If we found tasks, return them
            if !tasks.is_empty() {
                return tasks;
            }
        }
    }

    tasks
}

/// Get environment variables for spawning a worker with the given provider
fn get_worker_env(provider: &str) -> HashMap<String, String> {
    let mut env: HashMap<String, String> = std::env::vars().collect();

    // Load credentials from /etc/palace/credentials.env
    // Skip ANTHROPIC_API_KEY - that should come from the user's environment
    if let Ok(contents) = std::fs::read_to_string("/etc/palace/credentials.env") {
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();
                // Don't override ANTHROPIC_API_KEY - let user's env take precedence
                if key != "ANTHROPIC_API_KEY" {
                    env.insert(key.to_string(), value.to_string());
                }
            }
        }
    }

    match provider {
        "anthropic" => {
            // Use default Anthropic credentials (from ANTHROPIC_API_KEY)
            env.remove("ANTHROPIC_AUTH_TOKEN");
            env.remove("ANTHROPIC_BASE_URL");
        }
        "z.ai" | "zai" => {
            // Use Z.ai as Anthropic-compatible endpoint
            if let Ok(zai_key) = std::env::var("ZAI_API_KEY") {
                env.insert("ANTHROPIC_AUTH_TOKEN".to_string(), zai_key);
                env.insert("ANTHROPIC_BASE_URL".to_string(), "https://api.z.ai/api/anthropic".to_string());
            }
        }
        "local" | "palace" => {
            // Use local Palace daemon as proxy (non-Claude models)
            env.insert("ANTHROPIC_BASE_URL".to_string(), "http://127.0.0.1:19848".to_string());
        }
        "openrouter" => {
            // Use OpenRouter for model routing
            if let Ok(or_key) = std::env::var("OPENROUTER_API_KEY") {
                env.insert("ANTHROPIC_AUTH_TOKEN".to_string(), or_key);
            }
        }
        _ => {
            // Default: no changes
        }
    }

    env
}

/// Spawn a Claude CLI worker process for a task
fn spawn_worker(
    task: &SwarmTask,
    worker_id: &str,
    model: &str,
    provider: &str,
    project_path: &str,
    run_as_user: Option<&str>,
) -> Result<SwarmWorker, String> {
    use std::process::{Command, Stdio};

    // Build the worker prompt
    let worker_prompt = format!(
        r#"You are worker {} in a parallel swarm.

**YOUR TASK:**
{}: {}

**RULES:**
1. Focus ONLY on this task - do not work on unrelated things
2. When done, output a brief summary of what you accomplished
3. If you encounter blockers, document them clearly
4. Use all available tools (Read, Edit, Bash) to complete the task

Begin working on your assigned task now.
"#,
        worker_id, task.label, task.description
    );

    // Build the system prompt for the worker
    let system_prompt = format!(
        "You are agent {} in a parallel swarm. Complete your assigned task, then stop. Be focused and efficient.",
        worker_id
    );

    // Get environment for this provider
    let env = get_worker_env(provider);

    // Build the Claude CLI command - use sudo -u if run_as_user is specified
    let mut cmd = if let Some(user) = run_as_user {
        let mut c = Command::new("sudo");
        c.args(["-u", user, "claude"]);
        c
    } else {
        Command::new("claude")
    };

    cmd.args([
        "--print",
        "--model", model,
        "--append-system-prompt", &system_prompt,
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--dangerously-skip-permissions",
    ])
    .current_dir(project_path)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .envs(env);

    // Spawn the process
    let mut process = cmd.spawn().map_err(|e| format!("Failed to spawn claude: {}", e))?;

    // Send the initial prompt
    if let Some(ref mut stdin) = process.stdin {
        use std::io::Write;
        let initial_message = serde_json::json!({
            "type": "user",
            "message": {
                "role": "user",
                "content": worker_prompt
            }
        });
        let msg = serde_json::to_string(&initial_message).unwrap_or_default();
        let _ = stdin.write_all(msg.as_bytes());
        let _ = stdin.write_all(b"\n");
        let _ = stdin.flush();
    }

    Ok(SwarmWorker {
        id: worker_id.to_string(),
        task: task.clone(),
        process,
        model: model.to_string(),
        output_buffer: String::new(),
        finished: false,
    })
}

/// Select model for swarm workers based on strategy/tier
fn select_worker_model(worker_tier: &WorkerTier) -> (&'static str, &'static str) {
    match worker_tier {
        WorkerTier::Premium => ("claude-opus-4-5-20251101", "anthropic"),
        WorkerTier::Standard => ("claude-sonnet-4-5", "anthropic"),
        WorkerTier::Cheap => ("claude-haiku-4-5", "anthropic"),
        WorkerTier::Free => ("glm-4.6v", "z.ai"),
        WorkerTier::Local => ("devstral-small-2", "local"),
    }
}

/// Format swarm update for streaming into orchestrator's response
fn format_swarm_update(worker_id: &str, status: &str, detail: &str) -> String {
    format!("\n[{}] {} - {}\n", worker_id, status, detail)
}

/// Render stream-json line to compact format for display
/// Returns None if the line shouldn't produce output (e.g., tool_result, internal msgs)
fn render_stream_json_compact(line: &str, agent_id: &str) -> Option<String> {
    let json: serde_json::Value = serde_json::from_str(line).ok()?;
    let msg_type = json.get("type")?.as_str()?;

    match msg_type {
        "system" => {
            if json.get("subtype").and_then(|s| s.as_str()) == Some("init") {
                let model = json.get("model").and_then(|m| m.as_str()).unwrap_or("unknown");
                Some(format!("[{}] 📡 Model: {}\n", agent_id, model))
            } else {
                None
            }
        }
        "assistant" => {
            let content = json.get("message")?.get("content")?.as_array()?;
            let mut output = String::new();

            for block in content {
                let block_type = block.get("type").and_then(|t| t.as_str())?;

                match block_type {
                    "tool_use" => {
                        let tool_name = block.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                        let input = block.get("input").unwrap_or(&serde_json::Value::Null);

                        let formatted = match tool_name {
                            "Read" => {
                                let path = input.get("file_path").and_then(|p| p.as_str()).unwrap_or("?");
                                let filename = path.split('/').last().unwrap_or(path);
                                format!("[{}] 📖 Reading: {}\n", agent_id, filename)
                            }
                            "Edit" => {
                                let path = input.get("file_path").and_then(|p| p.as_str()).unwrap_or("?");
                                let filename = path.split('/').last().unwrap_or(path);
                                format!("[{}] ✏️  Editing: {}\n", agent_id, filename)
                            }
                            "Write" => {
                                let path = input.get("file_path").and_then(|p| p.as_str()).unwrap_or("?");
                                let filename = path.split('/').last().unwrap_or(path);
                                format!("[{}] 📝 Writing: {}\n", agent_id, filename)
                            }
                            "Bash" => {
                                let cmd = input.get("command").and_then(|c| c.as_str()).unwrap_or("");
                                let truncated: String = cmd.chars().take(50).collect();
                                format!("[{}] 💻 Running: {}...\n", agent_id, truncated)
                            }
                            "Glob" => {
                                let pattern = input.get("pattern").and_then(|p| p.as_str()).unwrap_or("");
                                format!("[{}] 🔍 Finding: {}\n", agent_id, pattern)
                            }
                            "Grep" => {
                                let pattern = input.get("pattern").and_then(|p| p.as_str()).unwrap_or("");
                                format!("[{}] 🔎 Searching: {}\n", agent_id, pattern)
                            }
                            "Task" => {
                                let desc = input.get("description").and_then(|d| d.as_str()).unwrap_or("");
                                format!("[{}] 🚀 Spawning: {}\n", agent_id, desc)
                            }
                            "TodoWrite" => {
                                format!("[{}] 📋 Updating todos\n", agent_id)
                            }
                            _ => {
                                format!("[{}] 🔧 {}\n", agent_id, tool_name)
                            }
                        };
                        output.push_str(&formatted);
                    }
                    "text" => {
                        let text = block.get("text").and_then(|t| t.as_str()).unwrap_or("");
                        if !text.is_empty() {
                            // For text, show it inline (no newline at end to allow streaming)
                            output.push_str(&format!("[{}] {}", agent_id, text));
                        }
                    }
                    _ => {}
                }
            }

            if output.is_empty() { None } else { Some(output) }
        }
        "result" => {
            let is_error = json.get("is_error").and_then(|e| e.as_bool()).unwrap_or(false);
            if is_error {
                let result = json.get("result").and_then(|r| r.as_str()).unwrap_or("Unknown error");
                Some(format!("[{}] ❌ Error: {}\n", agent_id, result))
            } else {
                Some(format!("[{}] ✅ Done\n", agent_id))
            }
        }
        _ => None
    }
}

/// Stream planner output with compact rendering
/// Returns SSE events that can be forwarded to the client
/// Each tool call/output gets its own message block for separate rendering
/// After streaming completes, parses actions and stores them in pending_actions
fn stream_planner_compact(
    planner_model: &str,
    prompt: &str,
    project_path: &str,
    run_as_user: Option<&str>,
    state: AppState,
    conversation_id: String,
) -> impl futures::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Send + 'static {
    use tokio::process::{Command};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use std::process::Stdio;

    let planner_model = planner_model.to_string();
    let prompt = prompt.to_string();
    let project_path = project_path.to_string();
    let run_as_user = run_as_user.map(|s| s.to_string());

    async_stream::stream! {
        let system_prompt = r#"You are a task planner. Analyze the project and help with the requested task.

You have access to all standard Claude Code tools. Use them to explore, understand, and work on the codebase.

Focus on the user's request. Be helpful and thorough."#;

        info!("SPAWN PLANNER: Starting in {} (model: {}, user: {:?})",
              project_path, planner_model, run_as_user);

        // Build command
        let mut cmd = if let Some(ref user) = run_as_user {
            let mut c = Command::new("sudo");
            let escaped_path = project_path.replace('\'', "'\\''");
            let escaped_model = planner_model.replace('\'', "'\\''");
            let escaped_prompt = system_prompt.replace('\'', "'\\''");

            let bash_cmd = format!(
                "export ANTHROPIC_BASE_URL=\"${{ANTHROPIC_BASE_URL:-http://127.0.0.1:19848}}\"; \
                 cd '{}' && exec claude --print --model '{}' --append-system-prompt '{}' --verbose --input-format stream-json --output-format stream-json --dangerously-skip-permissions",
                escaped_path, escaped_model, escaped_prompt
            );
            c.args(["-i", "-u", user, "bash", "-c", &bash_cmd]);
            c
        } else {
            let mut c = Command::new("claude");
            c.args([
                "--print",
                "--model", &planner_model,
                "--append-system-prompt", system_prompt,
                "--verbose",
                "--input-format", "stream-json",
                "--output-format", "stream-json",
                "--dangerously-skip-permissions",
            ])
            .current_dir(&project_path);
            c
        };

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                // Send complete message for error (with event: prefixes for Claude Code)
                let msg_id = format!("spawn_err_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..8].to_string());
                let escaped = format!("[@swarm] Failed to start planner: {}", e).replace('"', "\\\"");
                yield Ok(bytes::Bytes::from(format!(
                    "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"{}\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"planner\",\"stop_reason\":null}}}}\n\n\
                     event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
                     event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{}\"}}}}\n\n\
                     event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
                     event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}}}}\n\n\
                     event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n",
                    msg_id, escaped
                )));
                return;
            }
        };

        // Send the prompt via stdin
        if let Some(mut stdin) = child.stdin.take() {
            let init_msg = serde_json::json!({
                "type": "user",
                "message": {
                    "role": "user",
                    "content": prompt
                }
            });
            let msg_str = format!("{}\n", init_msg.to_string());
            info!("[{}] @swarm: Writing to stdin: {} bytes", conversation_id, msg_str.len());
            let _ = stdin.write_all(msg_str.as_bytes()).await;
            drop(stdin); // Close stdin to signal we're done
            info!("[{}] @swarm: Closed stdin", conversation_id);
        } else {
            warn!("[{}] @swarm: No stdin available!", conversation_id);
        }

        // Read stdout line by line and render compactly
        let stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                let msg_id = format!("spawn_no_stdout_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..8].to_string());
                yield Ok(bytes::Bytes::from(format!(
                    "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"{}\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"planner\",\"stop_reason\":null}}}}\n\n\
                     event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
                     event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"[@swarm] No stdout from planner\"}}}}\n\n\
                     event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
                     event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}}}}\n\n\
                     event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n",
                    msg_id
                )));
                return;
            }
        };

        let mut reader = BufReader::new(stdout).lines();
        let mut block_index = 0;
        let msg_id = format!("spawn_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());
        let mut accumulated_text = String::new(); // Accumulate for action parsing

        // Send initial message_start (with event: prefix for Claude Code)
        info!("[{}] @swarm: Yielding message_start", conversation_id);
        yield Ok(bytes::Bytes::from(format!(
            "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"{}\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"planner\",\"stop_reason\":null}}}}\n\n",
            msg_id
        )));
        info!("[{}] @swarm: message_start yielded", conversation_id);

        while let Ok(Some(line)) = reader.next_line().await {
            info!("[{}] @swarm stdout line: {}", conversation_id, &line[..line.len().min(100)]);

            if line.trim().is_empty() {
                continue;
            }

            // Render compact output
            if let Some(compact) = render_stream_json_compact(&line, "planner") {
                info!("[{}] @swarm rendered: {}", conversation_id, &compact[..compact.len().min(80)]);
                // Accumulate text for action parsing
                accumulated_text.push_str(&compact);
                accumulated_text.push('\n');

                // IMPORTANT: Emit EVERY line as a COMPLETE content block (start, delta, stop)
                // Claude Code only renders when blocks are CLOSED - streaming within a block doesn't display
                let escaped = compact.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
                info!("[{}] @swarm: Yielding block {}", conversation_id, block_index);
                yield Ok(bytes::Bytes::from(format!(
                    "event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
                     event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"text_delta\",\"text\":\"{}\\n\"}}}}\n\n\
                     event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n",
                    block_index, block_index, escaped, block_index
                )));
                info!("[{}] @swarm: Block {} yielded", conversation_id, block_index);
                block_index += 1;
            }
        }

        // Wait for process to finish
        let exit_status = child.wait().await;
        let exit_code = exit_status.map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);

        // Send completion status as final block
        let status_msg = if exit_code == 0 {
            "✅ Planner completed successfully"
        } else {
            "❌ Planner exited with error"
        };
        yield Ok(bytes::Bytes::from(format!(
            "event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
             event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"text_delta\",\"text\":\"{}\"}}}}\n\n\
             event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n",
            block_index, block_index, status_msg, block_index
        )));
        block_index += 1;

        // Parse and store actions from accumulated text
        let (_text_before, parsed_actions, _text_after) = parse_actions_yaml(&accumulated_text);
        if !parsed_actions.is_empty() {
            let action_count = parsed_actions.len();

            // Update state with parsed actions
            {
                let mut conversations = state.conversations.write().await;
                if let Some(conv) = conversations.get_mut(&conversation_id) {
                    conv.pending_actions = parsed_actions.clone();
                    info!("[{}] @swarm: Stored {} pending actions", conversation_id, action_count);
                }
            }

            // Send formatted action list as additional block
            let mut action_list = String::from("\n\n📋 **Available Actions:**\n\n");
            for action in &parsed_actions {
                action_list.push_str(&format!("  **{}. {}**\n", action.num, action.label));
                if !action.description.is_empty() {
                    action_list.push_str(&format!("     {}\n", action.description));
                }
                action_list.push('\n');
            }
            action_list.push_str("\n💡 Use `@action N` or `@actions 1,2,3` to select tasks for execution.\n");

            let escaped = action_list.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
            yield Ok(bytes::Bytes::from(format!(
                "event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
                 event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"text_delta\",\"text\":\"{}\"}}}}\n\n\
                 event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{}}}\n\n",
                block_index, block_index, escaped, block_index
            )));
        }

        // Send message_delta and message_stop
        yield Ok(bytes::Bytes::from(
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n\
             event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        ));
    }
}

/// Call the planner model to generate tasks
/// - Anthropic models (claude-*): uses `claude -p` CLI
/// - Everything else: routes through Palace translator
async fn call_planner_for_tasks(
    planner_model: &str,
    prompt: &str,
    project_path: &str,
    run_as_user: Option<&str>,
) -> Result<Vec<SwarmTask>, String> {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};

    let system_prompt = r#"You are a task planner for a swarm of Claude Code workers.

You receive the current BUILD STATUS and PROJECT PATH. Don't re-run builds - use what's provided.

Your job: Generate INDEPENDENT tasks that workers can execute IN PARALLEL.

Output format - YAML actions block:

```yaml
actions:
  - label: Fix the authentication bug in login.rs
    description: The login function returns early on valid credentials
  - label: Add missing test for user registration
    description: UserRegistration::new() has no test coverage
```

Format requirements:
- ```yaml block with "actions:" key
- Each action: "label:" (required), "description:" (optional)
- Tasks should be independent (parallelizable)"#;

    info!("SWARM PLANNER: Spawning Claude Code agent in {} (model: {}, user: {:?})",
          project_path, planner_model, run_as_user);

    // Build command - use sudo -u with -i to get user's login environment
    let mut cmd = if let Some(user) = run_as_user {
        let mut c = Command::new("sudo");
        // Escape single quotes in args for shell
        let escaped_path = project_path.replace('\'', "'\\''");
        let escaped_model = planner_model.replace('\'', "'\\''");
        let escaped_prompt = system_prompt.replace('\'', "'\\''");

        // Get fallback env vars - only used if user doesn't have them set
        let env = get_worker_env("anthropic");
        // Always use local daemon as fallback - get_worker_env("anthropic") removes ANTHROPIC_BASE_URL
        let anthropic_url = env.get("ANTHROPIC_BASE_URL").map(|s| s.as_str()).unwrap_or("http://127.0.0.1:19848");
        let mistral_key = env.get("MISTRAL_API_KEY").map(|s| s.as_str()).unwrap_or("");
        let zai_key = env.get("ZAI_API_KEY").map(|s| s.as_str()).unwrap_or("");

        // Use ${VAR:-default} so user's env takes precedence
        let bash_cmd = format!(
            "export ANTHROPIC_BASE_URL=\"${{ANTHROPIC_BASE_URL:-{}}}\"; \
             export MISTRAL_API_KEY=\"${{MISTRAL_API_KEY:-{}}}\"; \
             export ZAI_API_KEY=\"${{ZAI_API_KEY:-{}}}\"; \
             cd '{}' && exec claude --print --model '{}' --append-system-prompt '{}' --verbose --input-format stream-json --output-format stream-json --dangerously-skip-permissions",
            anthropic_url, mistral_key, zai_key,
            escaped_path, escaped_model, escaped_prompt
        );
        // -i gets user's login environment (their claude login), -u sets user
        c.args(["-i", "-u", user, "bash", "-c", &bash_cmd]);
        c
    } else {
        let mut c = Command::new("claude");
        c.args([
            "--print",
            "--model", planner_model,
            "--append-system-prompt", system_prompt,
            "--verbose",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
        ])
        .current_dir(project_path);
        c
    };

    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Set up environment
    let env = get_worker_env("anthropic");
    for (key, value) in &env {
        cmd.env(key, value);
    }

    let mut child = cmd.spawn()
        .map_err(|e| format!("Failed to spawn planner: {}", e))?;

    // Send the prompt via stream-json format
    let stdin = child.stdin.as_mut()
        .ok_or("Failed to get stdin")?;

    let init_msg = serde_json::json!({
        "type": "user",
        "message": {
            "role": "user",
            "content": prompt
        }
    });
    writeln!(stdin, "{}", init_msg.to_string())
        .map_err(|e| format!("Failed to write to planner stdin: {}", e))?;
    drop(child.stdin.take()); // Close stdin to signal we're done

    // Read output
    let stdout = child.stdout.take()
        .ok_or("Failed to get stdout")?;
    let reader = BufReader::new(stdout);

    let mut text_content = String::new();
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read planner output: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }

        // Parse stream-json output
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
            // Look for assistant text messages
            if json.get("type").and_then(|t| t.as_str()) == Some("assistant") {
                if let Some(content) = json.get("content").and_then(|c| c.as_str()) {
                    text_content.push_str(content);
                }
            }
            // Also check for content_block_delta (streaming chunks)
            if json.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                if let Some(delta) = json.get("delta") {
                    if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                        text_content.push_str(text);
                    }
                }
            }
        }
    }

    // Wait for process to complete
    let status = child.wait()
        .map_err(|e| format!("Failed to wait for planner: {}", e))?;

    if !status.success() {
        warn!("SWARM PLANNER: Process exited with status: {}", status);
    }

    info!("SWARM PLANNER: Received {} chars of output", text_content.len());
    if text_content.len() < 1000 {
        info!("SWARM PLANNER: Output: {}", text_content);
    } else {
        info!("SWARM PLANNER: Output (truncated): {}...", &text_content[..500]);
    }

    // Parse tasks from YAML actions block
    let tasks = parse_tasks_from_response(&text_content);
    if tasks.is_empty() {
        return Err(format!("Planner did not generate any tasks. Output was: {}",
            if text_content.len() > 200 { &text_content[..200] } else { &text_content }));
    }

    Ok(tasks)
}

/// Call the planner model to generate tasks - direct API version
/// Kept for cases where translator is not available or direct access is needed
#[allow(dead_code)]
async fn call_planner_for_tasks_direct(
    planner_model: &str,
    prompt: &str,
    _state: &AppState,
) -> Result<Vec<SwarmTask>, String> {
    let client = reqwest::Client::new();
    let system_prompt = "You are a task planning assistant. Analyze the project and generate actionable tasks.";

    // Different API formats for different providers
    let text_content = if planner_model.contains("devstral") {
        // Mistral API - OpenAI format
        let mistral_key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| "MISTRAL_API_KEY not set")?;

        let request = serde_json::json!({
            "model": planner_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096
        });

        let response = client
            .post("https://api.mistral.ai/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", mistral_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Mistral request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(format!("Mistral returned {}: {}", status, text));
        }

        let resp: serde_json::Value = response.json().await
            .map_err(|e| format!("Failed to parse Mistral response: {}", e))?;

        resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string()

    } else if planner_model.contains("glm") {
        // Z.ai - Anthropic-compatible format
        let zai_key = std::env::var("ZAI_API_KEY")
            .map_err(|_| "ZAI_API_KEY not set")?;

        let request = anthropic::MessagesRequest {
            model: planner_model.to_string(),
            max_tokens: 4096,
            messages: vec![
                anthropic::Message {
                    role: anthropic::Role::User,
                    content: anthropic::Content::Text(prompt.to_string()),
                },
            ],
            system: Some(anthropic::SystemPrompt::Text(system_prompt.to_string())),
            stream: Some(false),
            temperature: None,
            tools: None,
            thinking: None,
            extra: None,
        };

        let response = client
            .post("https://api.z.ai/api/anthropic/v1/messages")
            .header("Content-Type", "application/json")
            .header("x-api-key", &zai_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Z.ai request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(format!("Z.ai returned {}: {}", status, text));
        }

        let resp: anthropic::MessagesResponse = response.json().await
            .map_err(|e| format!("Failed to parse Z.ai response: {}", e))?;

        resp.content.iter()
            .filter_map(|c| {
                if let anthropic::ContentBlock::Text { text } = c {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n")

    } else {
        // Anthropic - native format
        let anthropic_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY not set")?;

        let request = anthropic::MessagesRequest {
            model: planner_model.to_string(),
            max_tokens: 4096,
            messages: vec![
                anthropic::Message {
                    role: anthropic::Role::User,
                    content: anthropic::Content::Text(prompt.to_string()),
                },
            ],
            system: Some(anthropic::SystemPrompt::Text(system_prompt.to_string())),
            stream: Some(false),
            temperature: None,
            tools: None,
            thinking: None,
            extra: None,
        };

        let response = client
            .post("https://api.anthropic.com/v1/messages")
            .header("Content-Type", "application/json")
            .header("x-api-key", &anthropic_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Anthropic request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(format!("Anthropic returned {}: {}", status, text));
        }

        let resp: anthropic::MessagesResponse = response.json().await
            .map_err(|e| format!("Failed to parse Anthropic response: {}", e))?;

        resp.content.iter()
            .filter_map(|c| {
                if let anthropic::ContentBlock::Text { text } = c {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    // Parse tasks from response
    let tasks = parse_tasks_from_response(&text_content);
    if tasks.is_empty() {
        return Err("Planner did not generate any tasks".to_string());
    }

    Ok(tasks)
}

/// Main swarm orchestration function
/// Called when swarm mode is enabled in hypermiler strategy
async fn run_swarm_orchestration(
    swarm_config: &SwarmConfig,
    hypermiler_config: &HypermilerConfig,
    project_path: &str,
    run_as_user: Option<&str>,
    state: &AppState,
) -> Result<Vec<SwarmWorkerResult>, String> {
    use std::io::{BufRead, BufReader};

    // Step 1: Get project status
    info!("SWARM: Getting project status...");
    let build_status = read_watchd_status(Some(project_path)).await;

    // Step 2: Determine planner model
    let planner_model = swarm_config.planner_model.as_deref()
        .or(hypermiler_config.orchestrator_model.as_deref())
        .unwrap_or("devstral-2512");

    info!("SWARM: Using planner model: {}", planner_model);

    // Step 3: Generate task planning prompt
    let prompt = build_task_planning_prompt(Some(project_path), &build_status);

    // Step 4: Call planner to generate tasks
    info!("SWARM: Calling planner for task generation...");
    let tasks = call_planner_for_tasks(planner_model, &prompt, project_path, run_as_user).await?;

    info!("SWARM: Planner generated {} tasks", tasks.len());
    for task in &tasks {
        info!("  [{}] {}: {}", task.num, task.label, task.description);
    }

    // Step 5: Apply safety cap (unless unlimited)
    let tasks_to_spawn: Vec<_> = match swarm_config.swarm_mode {
        SwarmMode::Unlimited => {
            info!("SWARM: Unlimited mode - spawning all {} tasks", tasks.len());
            tasks
        }
        _ => {
            let max_tasks = swarm_config.max_workers as usize;
            if tasks.len() > max_tasks {
                info!("SWARM: Applying safety cap - {} of {} tasks (use @swarm=unlimited to remove)", max_tasks, tasks.len());
            }
            tasks.into_iter().take(max_tasks).collect()
        }
    };

    // Step 6: Select worker model based on tier
    let (worker_model, worker_provider) = select_worker_model(&hypermiler_config.worker_tier);
    info!("SWARM: Spawning {} workers using {} via {}", tasks_to_spawn.len(), worker_model, worker_provider);

    // Step 7: Spawn workers
    let mut workers: Vec<SwarmWorker> = Vec::new();
    for (idx, task) in tasks_to_spawn.iter().enumerate() {
        let worker_id = format!("worker-{}", idx + 1);
        match spawn_worker(task, &worker_id, worker_model, worker_provider, project_path, run_as_user) {
            Ok(worker) => {
                info!("SWARM: Spawned {} for task: {}", worker_id, task.label);
                workers.push(worker);
            }
            Err(e) => {
                warn!("SWARM: Failed to spawn {}: {}", worker_id, e);
            }
        }
    }

    if workers.is_empty() {
        return Err("Failed to spawn any workers".to_string());
    }

    // Step 8: Monitor workers and collect results
    info!("SWARM: Monitoring {} workers...", workers.len());
    let mut results: Vec<SwarmWorkerResult> = Vec::new();

    // Simple blocking monitor - check each worker's status
    // In production, this would use async I/O with tokio
    loop {
        let mut all_done = true;

        for worker in &mut workers {
            if worker.finished {
                continue;
            }

            // Check if process has exited
            match worker.process.try_wait() {
                Ok(Some(status)) => {
                    // Process exited
                    worker.finished = true;

                    // Read remaining output
                    if let Some(ref mut stdout) = worker.process.stdout.take() {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            if let Ok(line) = line {
                                worker.output_buffer.push_str(&line);
                                worker.output_buffer.push('\n');
                            }
                        }
                    }

                    let exit_code = status.code();
                    let success = exit_code == Some(0);

                    info!("SWARM: {} finished with exit code {:?}", worker.id, exit_code);

                    results.push(SwarmWorkerResult {
                        id: worker.id.clone(),
                        task: worker.task.clone(),
                        exit_code,
                        output: worker.output_buffer.clone(),
                        success,
                    });
                }
                Ok(None) => {
                    // Still running
                    all_done = false;

                    // Read any available output (non-blocking would be better)
                    // For now, skip to avoid blocking
                }
                Err(e) => {
                    warn!("SWARM: Error checking {}: {}", worker.id, e);
                    worker.finished = true;
                }
            }
        }

        if all_done {
            break;
        }

        // Brief sleep to avoid busy-waiting
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    info!("SWARM: All workers finished. {} results collected.", results.len());

    Ok(results)
}

/// Format swarm results as text for the orchestrator response
fn format_swarm_results(results: &[SwarmWorkerResult]) -> String {
    let mut output = String::new();

    output.push_str("\n\n## 🐝 Swarm Execution Complete\n\n");

    let successful = results.iter().filter(|r| r.success).count();
    let failed = results.len() - successful;

    output.push_str(&format!("**Summary:** {} tasks completed, {} failed\n\n", successful, failed));

    for result in results {
        let status_icon = if result.success { "✅" } else { "❌" };
        output.push_str(&format!("### {} [{}] {}\n\n", status_icon, result.id, result.task.label));

        if !result.output.is_empty() {
            // Truncate long outputs
            let output_preview = if result.output.len() > 500 {
                format!("{}...\n\n(output truncated)", &result.output[..500])
            } else {
                result.output.clone()
            };
            output.push_str(&format!("```\n{}\n```\n\n", output_preview));
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== detect_model_switch tests ====================

    #[test]
    fn test_detect_simple_handoff() {
        let result = detect_model_switch("@@claude Please do this task");
        assert!(result.is_some());
        let (pack, msg) = result.unwrap();
        assert_eq!(pack, "claude");
        assert_eq!(msg, Some("Please do this task".to_string()));
    }

    #[test]
    fn test_detect_switch_equals_syntax() {
        // @@switch=packname should work
        let result = detect_model_switch("@@switch=claude Please do this task");
        assert!(result.is_some());
        let (pack, msg) = result.unwrap();
        assert_eq!(pack, "claude");
        assert_eq!(msg, Some("Please do this task".to_string()));

        // @@switch=mistral
        let result = detect_model_switch("@@switch=mistral write tests");
        assert!(result.is_some());
        let (pack, msg) = result.unwrap();
        assert_eq!(pack, "mistral");
        assert_eq!(msg, Some("write tests".to_string()));

        // Both shorthand and switch= should work
        let result = detect_model_switch("@@devstral");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "devstral");

        let result = detect_model_switch("@@switch=devstral");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "devstral");
    }

    #[test]
    fn test_detect_handoff_no_message() {
        let result = detect_model_switch("@@mistral");
        assert!(result.is_some());
        let (pack, msg) = result.unwrap();
        assert_eq!(pack, "mistral");
        assert!(msg.is_none());
    }

    #[test]
    fn test_detect_handoff_preserves_downstream_instructions() {
        // Key test: downstream @@ instructions should be preserved in the message
        let text = "@@claude Please implement X, then @@mistral for tests, then @@glm for review";
        let result = detect_model_switch(text);
        assert!(result.is_some());
        let (pack, msg) = result.unwrap();
        assert_eq!(pack, "claude");
        // The message should contain the downstream instructions
        let message = msg.unwrap();
        assert!(message.contains("@@mistral"));
        assert!(message.contains("@@glm"));
    }

    #[test]
    fn test_detect_all_valid_packs() {
        let packs = vec!["glm", "mistral", "devstral", "liefstral", "anthropic", "claude"];
        for pack_name in packs {
            let text = format!("@@{} do something", pack_name);
            let result = detect_model_switch(&text);
            assert!(result.is_some(), "Should detect pack: {}", pack_name);
            let (pack, _) = result.unwrap();
            assert_eq!(pack, pack_name);
        }
    }

    #[test]
    fn test_detect_invalid_pack_ignored() {
        // Invalid pack names should not trigger a switch
        let result = detect_model_switch("@@foobar Please do this");
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_case_insensitive() {
        let result = detect_model_switch("@@CLAUDE Please do this");
        assert!(result.is_some());
        let (pack, _) = result.unwrap();
        assert_eq!(pack, "claude"); // Should be normalized to lowercase
    }

    #[test]
    fn test_detect_mixed_case() {
        let result = detect_model_switch("@@Mistral Please do this");
        assert!(result.is_some());
        let (pack, _) = result.unwrap();
        assert_eq!(pack, "mistral");
    }

    #[test]
    fn test_detect_first_valid_pack_wins() {
        // If multiple valid packs, first one wins
        let result = detect_model_switch("@@glm do X then @@claude do Y");
        assert!(result.is_some());
        let (pack, msg) = result.unwrap();
        assert_eq!(pack, "glm");
        // Message includes everything after first match
        assert!(msg.unwrap().contains("@@claude"));
    }

    #[test]
    fn test_detect_handoff_in_middle_of_text() {
        let text = "I'll implement hello world first.\n\n@@mistral Can you write tests?";
        let result = detect_model_switch(text);
        assert!(result.is_some());
        let (pack, msg) = result.unwrap();
        assert_eq!(pack, "mistral");
        assert_eq!(msg, Some("Can you write tests?".to_string()));
    }

    #[test]
    fn test_detect_no_handoff_in_plain_text() {
        let result = detect_model_switch("Just some regular text without any handoffs");
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_email_not_matched() {
        // @@ pattern but not a valid pack
        let result = detect_model_switch("Contact me at test@@example.com");
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_escaped_in_backticks() {
        // @@claude inside backticks should NOT trigger
        let result = detect_model_switch("Use `@@claude` to switch models");
        assert!(result.is_none());

        // But @@claude outside backticks SHOULD trigger
        let result = detect_model_switch("Use `syntax` then @@claude to do it");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "claude");
    }

    #[test]
    fn test_detect_user_backtick_example_not_matched() {
        // User providing backticked syntax as an example - should NOT trigger
        // This was the actual bug: "Try using `@@switch=glm` in a message"
        let result = detect_model_switch("Try using `@@switch=glm` in a message");
        assert!(result.is_none(), "backtick-escaped @@switch=glm should not trigger");

        // Multiple backticked examples in same message
        let result = detect_model_switch(
            "Try using `@@switch=glm` in a message, like `@@switch=glm Please write to /tmp/helloworld.py`."
        );
        assert!(result.is_none(), "multiple backtick-escaped examples should not trigger");

        // @@switch= syntax with equals sign inside backticks
        let result = detect_model_switch("The syntax is `@@switch=modelname`");
        assert!(result.is_none(), "@@switch=modelname in backticks should not trigger");
    }

    #[test]
    fn test_detect_escaped_in_code_block() {
        // @@claude inside code blocks should NOT trigger
        let result = detect_model_switch("```\n@@claude\n```");
        assert!(result.is_none());

        // Multi-line code block
        let result = detect_model_switch("Here's how:\n```bash\necho @@mistral\n```\nDone");
        assert!(result.is_none());

        // But after the code block SHOULD trigger
        let result = detect_model_switch("```\ncode\n```\n@@claude do this");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "claude");
    }

    #[test]
    fn test_detect_escaped_in_quote() {
        // @@claude on a quote line should NOT trigger
        let result = detect_model_switch("> @@claude said this");
        assert!(result.is_none());

        // Nested quote
        let result = detect_model_switch(">> @@mistral example");
        assert!(result.is_none());

        // But on a non-quote line SHOULD trigger
        let result = detect_model_switch("> quoted\n@@claude real handoff");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "claude");
    }

    // ==================== Handoff message format tests ====================

    #[test]
    fn test_handoff_message_format() {
        // Test the exact format we send to the receiving model
        let pack_name = "mistral";
        let task = "Write tests for hello.py";

        let formatted = format!(
            "<SUPER_IMPORTANT_USER_SYSTEM_REMINDER>[Multi-model session] You are {}. IMPORTANT: This environment supports model handoffs. Writing @@modelname (e.g. @@claude or @@mistral) in your response WILL switch to that model - this is a real feature, not a hallucination. Use it to hand off tasks.</SUPER_IMPORTANT_USER_SYSTEM_REMINDER>\n\n{}",
            pack_name, task
        );

        // Verify the format is correct
        assert!(formatted.contains("<SUPER_IMPORTANT_USER_SYSTEM_REMINDER>"));
        assert!(formatted.contains("You are mistral"));
        assert!(formatted.contains("not a hallucination"));
    }

    #[test]
    fn test_handoff_message_includes_full_chain() {
        // When claude hands to mistral with instructions to then hand to glm,
        // mistral should see the full chain
        let task = "Write tests, then @@glm for review";
        let pack_name = "mistral";

        let formatted = format!(
            "<SUPER_IMPORTANT_USER_SYSTEM_REMINDER>[Multi-model session] You are {}. IMPORTANT: This environment supports model handoffs. Writing @@modelname (e.g. @@claude or @@mistral) in your response WILL switch to that model - this is a real feature, not a hallucination. Use it to hand off tasks.</SUPER_IMPORTANT_USER_SYSTEM_REMINDER>\n\n{}",
            pack_name, task
        );

        // Mistral should see the @@glm instruction
        assert!(formatted.contains("@@glm"));
    }

    #[test]
    fn test_first_message_reminder_injection() {
        // Test the reminder that gets injected on first message of session
        let reminder = "<SUPER_IMPORTANT_USER_SYSTEM_REMINDER>[Multi-model session] IMPORTANT: This environment supports model handoffs. Writing @@modelname (e.g. @@claude or @@mistral) in your response WILL switch to that model - this is a real feature, not a hallucination. Use it to hand off tasks.</SUPER_IMPORTANT_USER_SYSTEM_REMINDER>\n\n";
        let original = "Hello, please help me";
        let injected = format!("{}{}", reminder, original);

        assert!(injected.contains("<SUPER_IMPORTANT_USER_SYSTEM_REMINDER>"));
        assert!(injected.contains("not a hallucination"));
        assert!(injected.ends_with("Hello, please help me"));
    }

    // ==================== ModelPack tests ====================

    #[test]
    fn test_model_packs_exist() {
        let packs = ModelPack::all_packs();
        assert!(packs.contains_key("glm"));
        assert!(packs.contains_key("mistral"));
        assert!(packs.contains_key("devstral"));
        assert!(packs.contains_key("anthropic"));
        assert!(packs.contains_key("claude"));
    }

    #[test]
    fn test_model_pack_aliases() {
        let packs = ModelPack::all_packs();
        // devstral should be an alias for mistral
        assert_eq!(packs["devstral"].opus, packs["mistral"].opus);
        // claude should be an alias for anthropic
        assert_eq!(packs["claude"].opus, packs["anthropic"].opus);
    }

    #[test]
    fn test_model_pack_mapping() {
        let packs = ModelPack::all_packs();

        // Test GLM pack maps Claude models to GLM
        let glm_pack = &packs["glm"];
        assert_eq!(glm_pack.map_model("claude-opus-4-5-20251101"), "glm-4.6v");
        assert_eq!(glm_pack.map_model("claude-sonnet-4-5"), "glm-4.6");
        assert_eq!(glm_pack.map_model("claude-haiku-4-5"), "glm-4.5-air");

        // Test Mistral pack maps Claude models to Devstral
        let mistral_pack = &packs["mistral"];
        assert_eq!(mistral_pack.map_model("claude-opus-4-5-20251101"), "devstral-2512");
        assert_eq!(mistral_pack.map_model("claude-sonnet-4-5"), "devstral-2512");
        assert_eq!(mistral_pack.map_model("claude-haiku-4-5"), "devstral-mini-2");
    }

    #[test]
    fn test_anthropic_pack_passthrough() {
        let packs = ModelPack::all_packs();
        let anthropic_pack = &packs["anthropic"];

        // Anthropic pack should pass through real Claude model names
        assert_eq!(anthropic_pack.map_model("claude-opus-4-5-20251101"), "claude-opus-4-5-20251101");
        assert_eq!(anthropic_pack.map_model("claude-sonnet-4-5"), "claude-sonnet-4-5");
    }

    // ==================== ModelConfig tests ====================
    // Note: ModelConfig is built in AppState::new(), so we test via AppState

    #[test]
    fn test_devstral_uses_full_fat_model() {
        let state = AppState::new();

        // devstral and devstral-2512 should use full fat model
        assert_eq!(state.models["devstral"].model_id, "devstral-2512");
        assert_eq!(state.models["devstral-2512"].model_id, "devstral-2512");
    }

    #[test]
    fn test_devstral_small_uses_small_model() {
        let state = AppState::new();

        // devstral-small-2 and devstral-mini-2 should use small model
        assert_eq!(state.models["devstral-small-2"].model_id, "labs-devstral-small-2512");
        assert_eq!(state.models["devstral-mini-2"].model_id, "labs-devstral-small-2512");
    }

    #[test]
    fn test_glm_models_are_passthrough() {
        let state = AppState::new();

        assert!(state.models["glm-4.6"].passthrough);
        assert!(state.models["glm-4.6v"].passthrough);
        assert!(state.models["glm-4.5-air"].passthrough);
    }

    #[test]
    fn test_mistral_models_not_passthrough() {
        let state = AppState::new();

        assert!(!state.models["devstral"].passthrough);
        assert!(!state.models["devstral-2512"].passthrough);
        assert!(!state.models["devstral-small-2"].passthrough);
    }

    // ==================== ModelTier tests ====================

    #[test]
    fn test_model_tier_classification() {
        // Premium tier
        assert_eq!(ModelTier::from_model_name("claude-opus-4-5"), ModelTier::Premium);
        assert_eq!(ModelTier::from_model_name("openai/gpt-5.1-codex-max"), ModelTier::Premium);
        assert_eq!(ModelTier::from_model_name("google/gemini-3-pro-preview"), ModelTier::Premium);
        // Standard tier
        assert_eq!(ModelTier::from_model_name("claude-sonnet-4-5"), ModelTier::Standard);
        // Cheap tier
        assert_eq!(ModelTier::from_model_name("claude-haiku-4-5"), ModelTier::Cheap);
        // Free tier (flatrate/preview)
        assert_eq!(ModelTier::from_model_name("glm-4.6"), ModelTier::Free);
        assert_eq!(ModelTier::from_model_name("devstral-2512"), ModelTier::Free);
        // Local tier
        assert_eq!(ModelTier::from_model_name("devstral-small-2"), ModelTier::Local);
        assert_eq!(ModelTier::from_model_name("qwen2.5-coder:32b"), ModelTier::Local);
        assert_eq!(ModelTier::from_model_name("gpt-oss-20b"), ModelTier::Local);
    }

    #[test]
    fn test_model_tier_is_at_or_below() {
        let local = ModelTier::Local;
        let free = ModelTier::Free;
        let cheap = ModelTier::Cheap;
        let standard = ModelTier::Standard;
        let premium = ModelTier::Premium;

        // Local is at or below everything
        assert!(local.is_at_or_below(&local));
        assert!(local.is_at_or_below(&free));
        assert!(local.is_at_or_below(&cheap));
        assert!(local.is_at_or_below(&standard));
        assert!(local.is_at_or_below(&premium));

        // Premium is only at or below itself
        assert!(!premium.is_at_or_below(&local));
        assert!(!premium.is_at_or_below(&free));
        assert!(!premium.is_at_or_below(&cheap));
        assert!(!premium.is_at_or_below(&standard));
        assert!(premium.is_at_or_below(&premium));

        // Cheap is at or below cheap, standard, and premium
        assert!(!cheap.is_at_or_below(&local));
        assert!(!cheap.is_at_or_below(&free));
        assert!(cheap.is_at_or_below(&cheap));
        assert!(cheap.is_at_or_below(&standard));
        assert!(cheap.is_at_or_below(&premium));
    }

    #[test]
    fn test_model_tier_rank() {
        assert_eq!(ModelTier::Local.rank(), 0);
        assert_eq!(ModelTier::Free.rank(), 1);
        assert_eq!(ModelTier::Cheap.rank(), 2);
        assert_eq!(ModelTier::Standard.rank(), 3);
        assert_eq!(ModelTier::Premium.rank(), 4);
    }

    #[test]
    fn test_model_tier_approx_cost() {
        // Costs per 1M output tokens
        assert_eq!(ModelTier::Local.approx_output_cost_per_mtok(), 0.0);      // Local = free
        assert_eq!(ModelTier::Free.approx_output_cost_per_mtok(), 0.0);       // GLM flatrate, Devstral 2 preview
        assert_eq!(ModelTier::Cheap.approx_output_cost_per_mtok(), 5.0);      // Haiku $5/Mtok out
        assert_eq!(ModelTier::Standard.approx_output_cost_per_mtok(), 15.0);  // Sonnet $15/Mtok out
        assert_eq!(ModelTier::Premium.approx_output_cost_per_mtok(), 25.0);   // Opus $25/Mtok out (representative)
    }

    // ==================== Strategy parsing tests ====================

    #[test]
    fn test_parse_strategy_simple() {
        assert_eq!(parse_strategy("@strategy=simple"), Some(Strategy::Simple));
        assert_eq!(parse_strategy("@strategy=SIMPLE"), Some(Strategy::Simple));
    }

    #[test]
    fn test_parse_strategy_smart_default() {
        let result = parse_strategy("@strategy=smart");
        assert!(matches!(result, Some(Strategy::Smart(c)) if c.worker_tier == WorkerTier::Standard));
    }

    #[test]
    fn test_parse_strategy_smart_with_modifiers() {
        let result = parse_strategy("@strategy=smart,premium");
        assert!(matches!(result, Some(Strategy::Smart(c)) if c.worker_tier == WorkerTier::Premium));

        let result = parse_strategy("@strategy=smart,25%");
        assert!(matches!(result, Some(Strategy::Smart(c)) if c.rate_threshold_percent == 25));

        let result = parse_strategy("@strategy=smart,cheap,50%");
        assert!(matches!(result, Some(Strategy::Smart(c)) if c.worker_tier == WorkerTier::Cheap && c.rate_threshold_percent == 50));
    }

    #[test]
    fn test_parse_strategy_hypermiler() {
        let result = parse_strategy("@strategy=hypermiler");
        assert!(matches!(result, Some(Strategy::Hypermiler(c)) if c.worker_tier == WorkerTier::Premium));

        let result = parse_strategy("@strategy=hypermiler,cheap");
        assert!(matches!(result, Some(Strategy::Hypermiler(c)) if c.worker_tier == WorkerTier::Cheap));
    }

    #[test]
    fn test_parse_strategy_tiers() {
        assert_eq!(parse_strategy("@strategy=premium"), Some(Strategy::Premium));
        assert_eq!(parse_strategy("@strategy=cheap"), Some(Strategy::Cheap));
        assert_eq!(parse_strategy("@strategy=free"), Some(Strategy::Free));
        assert_eq!(parse_strategy("@strategy=local"), Some(Strategy::Local));
        assert_eq!(parse_strategy("@strategy=airplane"), Some(Strategy::Airplane));
        assert_eq!(parse_strategy("@strategy=burn"), Some(Strategy::Burn));
    }

    #[test]
    fn test_parse_strategy_not_found() {
        assert_eq!(parse_strategy("no strategy here"), None);
        assert_eq!(parse_strategy("@strategy=invalid"), None);
    }

    // ==================== Strategy model filtering tests ====================

    #[test]
    fn test_strategy_allows_model_simple() {
        // Simple strategy has no restrictions
        let strategy = Strategy::Simple;
        assert!(strategy.allows_model("claude-opus-4-5-20251101"));
        assert!(strategy.allows_model("claude-sonnet-4-5"));
        assert!(strategy.allows_model("claude-haiku-4-5"));
        assert!(strategy.allows_model("devstral-2512"));
        assert!(strategy.allows_model("devstral-small-2"));
    }

    #[test]
    fn test_strategy_allows_model_cheap() {
        // Cheap strategy only allows cheap tier and below
        let strategy = Strategy::Cheap;
        assert!(!strategy.allows_model("claude-opus-4-5-20251101"));  // Premium
        assert!(!strategy.allows_model("claude-sonnet-4-5")); // Standard
        assert!(strategy.allows_model("claude-haiku-4-5"));    // Cheap
        assert!(strategy.allows_model("devstral-2512"));        // Free
        assert!(strategy.allows_model("devstral-small-2"));     // Local
    }

    #[test]
    fn test_strategy_allows_model_free() {
        // Free strategy only allows free tier and below
        let strategy = Strategy::Free;
        assert!(!strategy.allows_model("claude-opus-4-5-20251101"));  // Premium
        assert!(!strategy.allows_model("claude-sonnet-4-5")); // Standard
        assert!(!strategy.allows_model("claude-haiku-4-5"));   // Cheap
        assert!(strategy.allows_model("devstral-2512"));        // Free
        assert!(strategy.allows_model("devstral-small-2"));     // Local
    }

    #[test]
    fn test_strategy_allows_model_local() {
        // Local/airplane strategy only allows local tier
        let strategy = Strategy::Local;
        assert!(!strategy.allows_model("claude-opus-4-5-20251101"));
        assert!(!strategy.allows_model("devstral-2512"));  // Free tier, not local
        assert!(strategy.allows_model("devstral-small-2"));
        assert!(strategy.allows_model("gpt-oss-20b"));

        // Airplane is same as local
        let airplane = Strategy::Airplane;
        assert!(!airplane.allows_model("claude-opus-4-5-20251101"));
        assert!(airplane.allows_model("devstral-small-2"));
    }

    #[test]
    fn test_strategy_suggest_fallback() {
        assert_eq!(Strategy::Premium.suggest_fallback("anything"), Some("claude-opus-4-5-20251101"));
        assert_eq!(Strategy::Cheap.suggest_fallback("claude-opus-4-5-20251101"), Some("glm-4.6v"));
        assert_eq!(Strategy::Free.suggest_fallback("claude-opus-4-5-20251101"), Some("glm-4.6v"));
        assert_eq!(Strategy::Local.suggest_fallback("claude-opus-4-5-20251101"), Some("devstral-small-2"));
        assert_eq!(Strategy::Simple.suggest_fallback("claude-opus-4-5-20251101"), None);
    }
}
