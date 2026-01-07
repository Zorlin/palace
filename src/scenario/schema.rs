//! Scenario YAML schema definitions

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Root scenario document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    /// Scenario metadata
    pub scenario: ScenarioMeta,

    /// High-level goals (AI supervisor figures out how to achieve)
    /// Use this for flexible, adaptive execution
    #[serde(default)]
    pub goals: Vec<String>,

    /// Explicit steps (plain English automation script)
    /// Use this for fast, deterministic execution
    /// Can be mixed with goals - steps execute, goals guide decisions
    #[serde(default)]
    pub steps: Vec<super::script::ScriptStep>,

    /// Constraints the AI must respect
    #[serde(default)]
    pub constraints: Option<Vec<String>>,

    /// Permission policy for auto-approval
    #[serde(default)]
    pub permissions: Option<PermissionPolicy>,

    /// How to handle surveys
    #[serde(default)]
    pub surveys: Option<SurveyStrategy>,

    /// Project configuration
    pub project: ProjectConfig,

    /// Capture configuration
    #[serde(default)]
    pub capture: Option<CaptureConfig>,

    /// OKRs for measuring success
    #[serde(default)]
    pub okrs: Option<Vec<Okr>>,

    /// Hard requirements that must be met
    #[serde(default)]
    pub requirements: Option<Vec<String>>,
}

impl Scenario {
    /// Check if this scenario uses script mode (has steps)
    pub fn is_scripted(&self) -> bool {
        !self.steps.is_empty()
    }

    /// Check if this scenario uses goal mode (has goals, no steps)
    pub fn is_goal_driven(&self) -> bool {
        !self.goals.is_empty() && self.steps.is_empty()
    }

    /// Check if this scenario is mixed mode (both steps and goals)
    pub fn is_mixed(&self) -> bool {
        !self.steps.is_empty() && !self.goals.is_empty()
    }
}

/// Scenario metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioMeta {
    /// Human-readable name
    pub name: String,

    /// Optional description
    #[serde(default)]
    pub description: Option<String>,

    /// Version of this scenario
    #[serde(default = "default_version")]
    pub version: String,

    /// Author
    #[serde(default)]
    pub author: Option<String>,

    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_version() -> String {
    "1.0".to_string()
}

/// A single goal (can be simple string or structured)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScenarioGoal {
    /// Simple goal as a string
    Simple(String),
    /// Structured goal with details
    Structured {
        description: String,
        #[serde(default)]
        priority: Option<u32>,
        #[serde(default)]
        acceptance_criteria: Vec<String>,
    },
}

impl ScenarioGoal {
    pub fn description(&self) -> &str {
        match self {
            ScenarioGoal::Simple(s) => s,
            ScenarioGoal::Structured { description, .. } => description,
        }
    }
}

/// Permission policy for auto-approval
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PermissionPolicy {
    /// Permissions to automatically allow
    #[serde(default)]
    pub allow: Vec<PermissionType>,

    /// Permissions to automatically deny
    #[serde(default)]
    pub deny: Vec<PermissionType>,

    /// Ask supervisor for anything not in allow/deny
    #[serde(default = "default_ask_unknown")]
    pub ask_unknown: bool,

    /// Maximum number of auto-approvals before requiring human review
    #[serde(default)]
    pub max_auto_approvals: Option<u32>,
}

fn default_ask_unknown() -> bool {
    true
}

/// Types of permissions that can be auto-handled
#[derive(Debug, Clone, Hash, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PermissionType {
    /// Read files
    Read,
    /// Write/edit files
    Write,
    /// Create new files
    Create,
    /// Delete files
    Delete,
    /// Run cargo commands
    Cargo,
    /// Run npm/yarn commands
    Npm,
    /// Run git commands
    Git,
    /// Network access
    Network,
    /// Dangerous rm -rf style commands
    RmRf,
    /// Execute arbitrary bash
    Bash,
    /// Safe bash (no rm, no network, etc)
    BashSafe,
    /// Any command matching a pattern
    #[serde(rename = "pattern")]
    Pattern(String),
}

/// Strategy for handling surveys
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SurveyStrategy {
    /// Use Best Judgment - AI supervisor decides
    Ubj,
    /// Always pick first option
    First,
    /// Always pick last option
    Last,
    /// Ask user (pause scenario)
    Ask,
    /// Skip surveys if possible
    Skip,
    /// Custom strategy with rules
    Custom(Vec<SurveyRule>),
}

/// Custom rule for survey handling
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SurveyRule {
    /// Pattern to match in survey question
    pub pattern: String,
    /// Response to give
    pub response: SurveyResponse,
}

/// Possible survey responses
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SurveyResponse {
    /// Select option by index (0-based)
    Index(usize),
    /// Select option containing this text
    Contains(String),
    /// Let supervisor decide
    Ubj,
    /// Ask user
    Ask,
}

/// Project configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    /// Path to the project
    pub path: String,

    /// Initialize if missing (create directory)
    #[serde(default)]
    pub init_if_missing: bool,

    /// Git clone URL if project should be cloned
    #[serde(default)]
    pub git_clone: Option<String>,

    /// Branch to checkout
    #[serde(default)]
    pub branch: Option<String>,

    /// Template to use for initialization
    #[serde(default)]
    pub template: Option<String>,
}

impl ProjectConfig {
    /// Expand ~ and environment variables in path
    pub fn expanded_path(&self) -> PathBuf {
        let path = shellexpand::tilde(&self.path);
        PathBuf::from(path.as_ref())
    }
}

/// Configuration for capturing outputs
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CaptureConfig {
    /// Capture screenshots during execution
    #[serde(default)]
    pub screenshots: bool,

    /// Capture fixture data
    #[serde(default)]
    pub fixtures: bool,

    /// Generate animated SVG of session
    #[serde(default)]
    pub animated_svg: bool,

    /// Output directory for captures
    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    /// Take screenshot at specific events
    #[serde(default)]
    pub screenshot_on: Vec<CaptureEvent>,
}

fn default_output_dir() -> String {
    "./captures".to_string()
}

/// Events that can trigger captures
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CaptureEvent {
    /// When cards are displayed
    CardsReady,
    /// Before execution starts
    BeforeExecution,
    /// After execution completes
    AfterExecution,
    /// On any error
    OnError,
    /// On survey prompt
    OnSurvey,
    /// On permission prompt
    OnPermission,
    /// At regular intervals (seconds)
    Interval(u32),
}

/// OKR (Objectives and Key Results) for measuring success
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Okr {
    /// Objective description
    pub objective: String,

    /// Key results to measure
    pub key_results: Vec<KeyResult>,
}

/// A measurable key result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyResult {
    /// Description of the key result
    pub description: String,

    /// How to verify this result
    #[serde(default)]
    pub verification: Option<VerificationMethod>,
}

/// How to verify a key result was achieved
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationMethod {
    /// File exists at path
    FileExists(String),
    /// File contains text
    FileContains { path: String, text: String },
    /// Command succeeds (exit code 0)
    CommandSucceeds(String),
    /// Command output contains text
    CommandOutputContains { command: String, text: String },
    /// Ask supervisor to verify
    SupervisorVerify,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_type_serde() {
        let yaml = "read";
        let perm: PermissionType = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(perm, PermissionType::Read);
    }

    #[test]
    fn test_survey_strategy_ubj() {
        let yaml = "ubj";
        let strategy: SurveyStrategy = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(strategy, SurveyStrategy::Ubj);
    }

    #[test]
    fn test_project_path_expansion() {
        let config = ProjectConfig {
            path: "~/projects/test".to_string(),
            init_if_missing: false,
            git_clone: None,
            branch: None,
            template: None,
        };
        let expanded = config.expanded_path();
        assert!(!expanded.to_string_lossy().contains('~'));
    }
}
