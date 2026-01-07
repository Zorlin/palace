//! Scenario System - Ansible for Agents
//!
//! Declarative goal-driven automation for Palace. Scenarios define what you want,
//! an AI supervisor figures out how to achieve it.
//!
//! ## Example Scenario
//!
//! ```yaml
//! scenario:
//!   name: "Implement Classic Asteroids"
//!
//! goals:
//!   - "Working Asteroids game in Rust with WGPU"
//!   - "Ship controls, asteroids, bullets, scoring"
//!
//! constraints:
//!   - "No external game engines"
//!   - "Single binary output"
//!
//! permissions:
//!   allow: [read, write, cargo]
//!   deny: [network, rm_rf]
//!
//! surveys: ubj  # Use Best Judgment
//!
//! project:
//!   path: ~/projects/asteroids-demo
//!   init_if_missing: true
//! ```
//!
//! ## Safety Model
//!
//! On first run, users must acknowledge:
//! - Scenarios run AI agents with real system access
//! - Only run scenarios you trust
//! - Use sandboxing when possible
//! - Prompt injection risks exist
//!
//! After typing "I understand", the safety gate is passed and never nags again.

mod schema;
mod safety;
mod supervisor;
mod handlers;
mod script;
mod corrector;
mod diff;
mod generator;
mod recorder;

pub use schema::{Scenario, ScenarioGoal, PermissionPolicy, SurveyStrategy, ProjectConfig};
pub use safety::SafetyGate;
pub use supervisor::{Supervisor, SupervisorEvent, SupervisorDecision};
pub use handlers::{PermissionHandler, SurveyHandler};
pub use script::{Command, ScriptStep, parse_command, parse_steps};
pub use corrector::{CorrectorState, FeedbackOption, ScenarioCorrector};
pub use diff::{DiffChunk, DiffKind, DiffViewer};
pub use generator::{GeneratorAction, GeneratorState, DynamicQuestion, QuestionOption, ScenarioGenerator, StartOption};
pub use recorder::{RecordedEvent, RecorderConfig, SessionRecorder};

use anyhow::Result;
use std::path::Path;

/// Load a scenario from a YAML file
pub fn load_scenario<P: AsRef<Path>>(path: P) -> Result<Scenario> {
    let content = std::fs::read_to_string(path.as_ref())?;
    let scenario: Scenario = serde_yaml::from_str(&content)?;
    Ok(scenario)
}

/// Run a scenario with the supervisor
pub async fn run_scenario(scenario: Scenario) -> Result<ScenarioResult> {
    // Check safety gate first
    SafetyGate::check()?;

    // Initialize supervisor
    let supervisor = Supervisor::new(&scenario)?;

    // Run the scenario
    supervisor.execute().await
}

/// Result of a scenario execution
#[derive(Debug)]
pub struct ScenarioResult {
    /// Whether all goals were achieved
    pub success: bool,
    /// Goals that were completed
    pub completed_goals: Vec<String>,
    /// Goals that failed or were not reached
    pub failed_goals: Vec<String>,
    /// Captured fixtures during execution
    pub fixtures: Option<CapturedFixtures>,
    /// Screenshots taken during execution
    pub screenshots: Vec<String>,
    /// Summary of what happened
    pub summary: String,
}

/// Fixtures captured during scenario execution
#[derive(Debug, Default)]
pub struct CapturedFixtures {
    pub cards: Vec<crate::fixtures::RecordedCard>,
    pub tool_log: Vec<String>,
    pub thought_log: Vec<String>,
    pub analysis_log: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_scenario() {
        let yaml = r#"
scenario:
  name: "Test Scenario"

goals:
  - "Do something useful"

project:
  path: /tmp/test
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.scenario.name, "Test Scenario");
        assert_eq!(scenario.goals.len(), 1);
    }

    #[test]
    fn test_parse_full_scenario() {
        let yaml = r#"
scenario:
  name: "Full Test"
  description: "A complete test scenario"

goals:
  - "Goal one"
  - "Goal two"

constraints:
  - "No network access"
  - "Single file output"

permissions:
  allow:
    - read
    - write
    - cargo
  deny:
    - network
    - rm_rf

surveys: ubj

project:
  path: ~/projects/test
  init_if_missing: true

capture:
  screenshots: true
  fixtures: true
  output_dir: ./captures
"#;
        let scenario: Scenario = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(scenario.scenario.name, "Full Test");
        assert_eq!(scenario.goals.len(), 2);
        assert_eq!(scenario.constraints.as_ref().unwrap().len(), 2);
        assert!(scenario.permissions.is_some());
        assert_eq!(scenario.surveys, Some(SurveyStrategy::Ubj));
    }
}
