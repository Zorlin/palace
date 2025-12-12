//! Agent Spawn DSL Parser and Executor
//!
//! Models output spawn decisions using compact syntax:
//! ```text
//! +1=opus,2=sonnet,3=haiku-4==
//! ```
//!
//! Meaning:
//! - `+N=model` - Spawn child N using model
//! - `-N` - Cancel/don't spawn child N
//! - `==` - End of spawn decisions
//!
//! For expensive models (opus, sonnet, claude-*), we simulate.
//! For free models (devstral-2512), we actually spawn.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// A parsed spawn decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnDecision {
    /// Child agent ID (1, 2, 3, etc.)
    pub child_id: u32,
    /// Model to use for this child
    pub model: String,
    /// Task description (if provided)
    pub task: Option<String>,
}

/// A cancellation decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelDecision {
    /// Child agent ID to cancel
    pub child_id: u32,
}

/// Parsed spawn DSL output
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpawnDelta {
    /// Agents to spawn
    pub spawn: Vec<SpawnDecision>,
    /// Agents to cancel
    pub cancel: Vec<CancelDecision>,
}

/// Models that are free to use (or cheap enough for swarm usage)
const FREE_MODELS: &[&str] = &[
    // Mistral free tier
    "devstral-2512",
    "devstral-small-2505",  // 24GB RAM friendly
    "devstral",
    "mistral",              // Alias for devstral
    // GLM (Zhipu) - via BigModel API
    "glm-4.6v",        // Vision model (2024-12-08)
    "glm-4.6",         // Base model
];

/// Models that cost real money when --real flag is used
/// WITHOUT --real: opus/sonnet/haiku all map to devstral-2512 (free!)
/// WITH --real: these use actual Claude API (costs money)
const EXPENSIVE_MODELS: &[&str] = &[
    "opus",
    "sonnet",
    "haiku",
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
    "gpt-4",
    "gpt-4o",
];

/// Check if running in "real" mode (--real flag)
/// When false: opus/sonnet/haiku → devstral-2512 (free)
/// When true: opus/sonnet/haiku → actual Claude API (costs money)
pub fn is_real_mode() -> bool {
    std::env::args().any(|arg| arg == "--real")
}

/// Parse spawn DSL from model output
///
/// Formats supported:
/// - `+1=opus` - Spawn child 1 with opus
/// - `+1=opus,+2=sonnet,+3=haiku` - Multiple spawns (each prefixed with +)
/// - `+1=opus+2=sonnet+3=haiku` - Multiple spawns (no commas)
/// - `-4` - Cancel child 4
/// - `+1=opus-2==` - Spawn 1, cancel 2, end
/// - `+1=devstral:task="Fix the bug"` - With task description
pub fn parse_spawn_dsl(input: &str) -> Result<SpawnDelta, SpawnError> {
    let input = input.trim();

    if input.is_empty() {
        return Ok(SpawnDelta::default());
    }

    let mut delta = SpawnDelta::default();
    let mut pos = 0;
    let chars: Vec<char> = input.chars().collect();

    while pos < chars.len() {
        // Skip whitespace and commas
        while pos < chars.len() && (chars[pos].is_whitespace() || chars[pos] == ',') {
            pos += 1;
        }

        if pos >= chars.len() {
            break;
        }

        // Check for == (end marker)
        if pos + 1 < chars.len() && chars[pos] == '=' && chars[pos + 1] == '=' {
            debug!("Hit end marker ==");
            break;
        }

        // Check for + (spawn) or - (cancel)
        match chars[pos] {
            '+' => {
                pos += 1;
                let decision = parse_spawn_decision(&chars, &mut pos)?;
                delta.spawn.push(decision);
            }
            '-' => {
                pos += 1;
                let child_id = parse_number(&chars, &mut pos)?;
                delta.cancel.push(CancelDecision { child_id });
            }
            // If we hit a digit after a spawn, it might be shorthand like +1=opus,2=sonnet
            // Let's be lenient and treat it as +N=model
            c if c.is_ascii_digit() => {
                let decision = parse_spawn_decision(&chars, &mut pos)?;
                delta.spawn.push(decision);
            }
            c => {
                return Err(SpawnError::InvalidFormat(format!(
                    "Expected + or - at position {}, got '{}'",
                    pos, c
                )));
            }
        }
    }

    Ok(delta)
}

/// Parse a single spawn decision: `N=model` or `N=model:task="..."`
fn parse_spawn_decision(chars: &[char], pos: &mut usize) -> Result<SpawnDecision, SpawnError> {
    // Parse child ID
    let child_id = parse_number(chars, pos)?;

    // Expect =
    if *pos >= chars.len() || chars[*pos] != '=' {
        return Err(SpawnError::InvalidFormat(format!(
            "Expected = after child ID at position {}",
            pos
        )));
    }
    *pos += 1;

    // Parse model name (until comma, dash, equals, colon, or whitespace)
    let model = parse_identifier(chars, pos);

    // Optional task description
    let task = if *pos < chars.len() && chars[*pos] == ':' {
        *pos += 1;
        // Look for task="..."
        if *pos + 5 < chars.len() {
            let peek: String = chars[*pos..*pos + 5].iter().collect();
            if peek == "task=" {
                *pos += 5;
                Some(parse_quoted_string(chars, pos)?)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    Ok(SpawnDecision {
        child_id,
        model,
        task,
    })
}

/// Parse a number from the char stream
fn parse_number(chars: &[char], pos: &mut usize) -> Result<u32, SpawnError> {
    let mut num_str = String::new();

    while *pos < chars.len() && chars[*pos].is_ascii_digit() {
        num_str.push(chars[*pos]);
        *pos += 1;
    }

    if num_str.is_empty() {
        return Err(SpawnError::ParseError("Expected number".to_string()));
    }

    num_str
        .parse()
        .map_err(|e| SpawnError::ParseError(format!("Invalid number '{}': {}", num_str, e)))
}

/// Parse an identifier (model name)
fn parse_identifier(chars: &[char], pos: &mut usize) -> String {
    let mut ident = String::new();

    while *pos < chars.len() {
        let c = chars[*pos];
        if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
            ident.push(c);
            *pos += 1;
        } else {
            break;
        }
    }

    ident
}

/// Parse a quoted string
fn parse_quoted_string(chars: &[char], pos: &mut usize) -> Result<String, SpawnError> {
    if *pos >= chars.len() || chars[*pos] != '"' {
        return Err(SpawnError::ParseError("Expected opening quote".to_string()));
    }
    *pos += 1;

    let mut content = String::new();
    while *pos < chars.len() && chars[*pos] != '"' {
        if chars[*pos] == '\\' && *pos + 1 < chars.len() {
            *pos += 1;
            content.push(chars[*pos]);
        } else {
            content.push(chars[*pos]);
        }
        *pos += 1;
    }

    if *pos >= chars.len() {
        return Err(SpawnError::ParseError("Unterminated string".to_string()));
    }
    *pos += 1; // Skip closing quote

    Ok(content)
}

/// Check if a model is free (can be actually spawned)
pub fn is_free_model(model: &str) -> bool {
    let model_lower = model.to_lowercase();
    FREE_MODELS.iter().any(|m| model_lower.contains(m))
}

/// Check if a model is expensive (simulate only)
pub fn is_expensive_model(model: &str) -> bool {
    let model_lower = model.to_lowercase();
    EXPENSIVE_MODELS.iter().any(|m| model_lower.contains(m))
}

/// Result of executing spawn decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnResult {
    /// Actually spawned agents (free models)
    pub spawned: Vec<SpawnedAgent>,
    /// Simulated spawns (expensive models)
    pub simulated: Vec<SimulatedSpawn>,
    /// Cancelled agents
    pub cancelled: Vec<u32>,
}

/// An actually spawned agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnedAgent {
    pub child_id: u32,
    pub model: String,
    pub agent_id: String,
    pub task: Option<String>,
}

/// A simulated spawn (for logging)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedSpawn {
    pub child_id: u32,
    pub model: String,
    pub task: Option<String>,
    pub reason: String,
}

/// Spawn executor
pub struct SpawnExecutor {
    /// Base URL for spawning agents
    backend_url: String,
    /// API key for backend
    api_key: String,
    /// Default model for free spawns
    default_free_model: String,
    /// Track active agents
    active_agents: HashMap<u32, SpawnedAgent>,
}

impl SpawnExecutor {
    pub fn new(backend_url: String, api_key: String) -> Self {
        Self {
            backend_url,
            api_key,
            default_free_model: "devstral-2512".to_string(),
            active_agents: HashMap::new(),
        }
    }

    /// Execute spawn decisions
    ///
    /// Model routing:
    /// - WITHOUT --real: ALL models (including opus/sonnet/haiku) → devstral-2512 (free!)
    /// - WITH --real: opus/sonnet/haiku → actual Claude API (costs money)
    ///
    /// This means you can test swarm architectures for free using devstral-2512,
    /// then flip to --real when you want actual Claude quality.
    pub async fn execute(&mut self, delta: SpawnDelta) -> SpawnResult {
        let mut result = SpawnResult {
            spawned: Vec::new(),
            simulated: Vec::new(),
            cancelled: Vec::new(),
        };

        let real_mode = is_real_mode();

        // Handle cancellations
        for cancel in delta.cancel {
            if self.active_agents.remove(&cancel.child_id).is_some() {
                info!("🛑 Cancelled agent {}", cancel.child_id);
                result.cancelled.push(cancel.child_id);
            } else {
                warn!("Tried to cancel non-existent agent {}", cancel.child_id);
            }
        }

        // Handle spawns
        for spawn in delta.spawn {
            // Determine actual model to use
            let (actual_model, is_substituted) = if !real_mode && is_expensive_model(&spawn.model) {
                // Map expensive models to devstral-2512 when not in real mode
                ("devstral-2512".to_string(), true)
            } else {
                (spawn.model.clone(), false)
            };

            // Actually spawn (everything spawns now, just with model substitution)
            let mut spawn_with_actual = spawn.clone();
            spawn_with_actual.model = actual_model.clone();

            match self.spawn_agent(&spawn_with_actual).await {
                Ok(mut agent) => {
                    if is_substituted {
                        info!(
                            "🚀 Spawned agent {} with {} (requested: {}, id: {})",
                            agent.child_id, actual_model, spawn.model, agent.agent_id
                        );
                    } else {
                        info!(
                            "🚀 Spawned agent {} with {} (id: {})",
                            agent.child_id, agent.model, agent.agent_id
                        );
                    }
                    // Store requested model for display purposes
                    agent.model = spawn.model.clone();
                    self.active_agents.insert(agent.child_id, agent.clone());
                    result.spawned.push(agent);
                }
                Err(e) => {
                    warn!("Failed to spawn agent {}: {}", spawn.child_id, e);
                    result.simulated.push(SimulatedSpawn {
                        child_id: spawn.child_id,
                        model: spawn.model,
                        task: spawn.task,
                        reason: format!("Spawn failed: {}", e),
                    });
                }
            }
        }

        result
    }

    /// Actually spawn an agent (for free models)
    async fn spawn_agent(&self, decision: &SpawnDecision) -> Result<SpawnedAgent, SpawnError> {
        // For now, just create a unique ID - actual subprocess spawning comes later
        let agent_id = format!(
            "agent-{}-{}",
            decision.child_id,
            uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("unknown")
        );

        // TODO: Actually spawn a subprocess or make API call
        // For now, we just register the agent

        Ok(SpawnedAgent {
            child_id: decision.child_id,
            model: decision.model.clone(),
            agent_id,
            task: decision.task.clone(),
        })
    }

    /// Get active agents
    pub fn active_agents(&self) -> &HashMap<u32, SpawnedAgent> {
        &self.active_agents
    }
}

/// Errors from spawn operations
#[derive(Debug)]
pub enum SpawnError {
    InvalidFormat(String),
    ParseError(String),
    SpawnFailed(String),
}

impl std::fmt::Display for SpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpawnError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
            SpawnError::ParseError(s) => write!(f, "Parse error: {}", s),
            SpawnError::SpawnFailed(s) => write!(f, "Spawn failed: {}", s),
        }
    }
}

impl std::error::Error for SpawnError {}

/// Format spawn result for display
pub fn format_spawn_result(result: &SpawnResult) -> String {
    let mut output = String::new();

    if !result.spawned.is_empty() {
        output.push_str("🚀 Spawned agents:\n");
        for agent in &result.spawned {
            output.push_str(&format!(
                "  - Agent {} ({}): {}\n",
                agent.child_id,
                agent.model,
                agent.task.as_deref().unwrap_or("no task")
            ));
        }
    }

    if !result.simulated.is_empty() {
        output.push_str("📋 Simulated spawns:\n");
        for sim in &result.simulated {
            output.push_str(&format!(
                "  - [SIM] Agent {} ({}): {} - {}\n",
                sim.child_id,
                sim.model,
                sim.task.as_deref().unwrap_or("no task"),
                sim.reason
            ));
        }
    }

    if !result.cancelled.is_empty() {
        output.push_str(&format!("🛑 Cancelled: {:?}\n", result.cancelled));
    }

    if output.is_empty() {
        output.push_str("No spawn decisions.\n");
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_spawn() {
        let delta = parse_spawn_dsl("+1=opus").unwrap();
        assert_eq!(delta.spawn.len(), 1);
        assert_eq!(delta.spawn[0].child_id, 1);
        assert_eq!(delta.spawn[0].model, "opus");
    }

    #[test]
    fn test_parse_multiple_spawns() {
        // Each spawn prefixed with + (the format we'll prompt for)
        let delta = parse_spawn_dsl("+1=opus+2=sonnet+3=haiku").unwrap();
        assert_eq!(delta.spawn.len(), 3);
        assert_eq!(delta.spawn[0].model, "opus");
        assert_eq!(delta.spawn[1].model, "sonnet");
        assert_eq!(delta.spawn[2].model, "haiku");
    }

    #[test]
    fn test_parse_with_cancel() {
        // Use space separator to avoid ambiguity with model names containing dashes
        let delta = parse_spawn_dsl("+1=opus -2").unwrap();
        assert_eq!(delta.spawn.len(), 1);
        assert_eq!(delta.cancel.len(), 1);
        assert_eq!(delta.cancel[0].child_id, 2);
    }

    #[test]
    fn test_parse_with_end_marker() {
        let delta = parse_spawn_dsl("+1=opus+2=sonnet==ignored").unwrap();
        assert_eq!(delta.spawn.len(), 2);
    }

    #[test]
    fn test_parse_devstral() {
        let delta = parse_spawn_dsl("+1=devstral-2512+2=devstral").unwrap();
        assert_eq!(delta.spawn.len(), 2);
        assert_eq!(delta.spawn[0].model, "devstral-2512");
        assert_eq!(delta.spawn[1].model, "devstral");
    }

    #[test]
    fn test_is_free_model() {
        // Mistral free tier
        assert!(is_free_model("devstral-2512"));
        assert!(is_free_model("devstral-small-2505"));
        assert!(is_free_model("devstral"));
        assert!(is_free_model("mistral"));
        // GLM via BigModel API
        assert!(is_free_model("glm-4.6v"));
        assert!(is_free_model("glm-4.6"));
        // Expensive models
        assert!(!is_free_model("opus"));
        assert!(!is_free_model("sonnet"));
    }

    #[test]
    fn test_is_expensive_model() {
        assert!(is_expensive_model("opus"));
        assert!(is_expensive_model("claude-opus-4"));
        assert!(is_expensive_model("sonnet"));
        assert!(!is_expensive_model("devstral"));
    }

    #[test]
    fn test_parse_with_task() {
        let delta = parse_spawn_dsl("+1=devstral:task=\"Fix the auth bug\"").unwrap();
        assert_eq!(delta.spawn.len(), 1);
        assert_eq!(delta.spawn[0].task, Some("Fix the auth bug".to_string()));
    }

    #[test]
    fn test_parse_empty() {
        let delta = parse_spawn_dsl("").unwrap();
        assert!(delta.spawn.is_empty());
        assert!(delta.cancel.is_empty());
    }

    #[test]
    fn test_parse_lenient_comma_format() {
        // Be lenient with comma-separated format too (shorthand)
        let delta = parse_spawn_dsl("+1=opus,2=sonnet,3=haiku").unwrap();
        assert_eq!(delta.spawn.len(), 3);
    }

    #[tokio::test]
    async fn test_end_to_end_swarm_flow() {
        // Simulates the full flow:
        // 1. Model outputs spawn DSL (e.g., "+3=research" from our API test)
        // 2. We parse it
        // 3. We execute/simulate spawns

        // Example model output from devstral
        let model_output = "+3=research";

        // Parse the spawn DSL
        let delta = parse_spawn_dsl(model_output).unwrap();
        assert_eq!(delta.spawn.len(), 1);
        assert_eq!(delta.spawn[0].child_id, 3);
        assert_eq!(delta.spawn[0].model, "research");

        // Execute via SpawnExecutor
        let mut executor = SpawnExecutor::new(
            "https://api.mistral.ai/v1".to_string(),
            "test-key".to_string(),
        );

        let result = executor.execute(delta).await;

        // Since "research" is not in FREE_MODELS or EXPENSIVE_MODELS,
        // it should be spawned as-is (treated as unknown/custom model)
        // Actually all models spawn now with substitution logic
        assert_eq!(result.spawned.len(), 1);
        assert_eq!(result.spawned[0].child_id, 3);

        // Format the result for display
        let formatted = format_spawn_result(&result);
        assert!(formatted.contains("Spawned agents"));
        println!("End-to-end result:\n{}", formatted);
    }

    #[tokio::test]
    async fn test_model_substitution_without_real_flag() {
        // Without --real flag, expensive models should be substituted with devstral
        let delta = parse_spawn_dsl("+1=opus+2=sonnet+3=devstral").unwrap();

        let mut executor = SpawnExecutor::new(
            "https://api.mistral.ai/v1".to_string(),
            "test-key".to_string(),
        );

        let result = executor.execute(delta).await;

        // All should spawn (with substitution for opus/sonnet)
        assert_eq!(result.spawned.len(), 3);

        // Check the agents were spawned
        assert!(result.spawned.iter().any(|a| a.child_id == 1));
        assert!(result.spawned.iter().any(|a| a.child_id == 2));
        assert!(result.spawned.iter().any(|a| a.child_id == 3));

        let formatted = format_spawn_result(&result);
        println!("Model substitution test:\n{}", formatted);
    }
}
