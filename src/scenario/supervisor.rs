//! AI Supervisor for scenario execution
//!
//! The supervisor watches the Palace UI and makes decisions:
//! - When to click buttons/select options
//! - How to handle surveys (UBJ = Use Best Judgment)
//! - Whether to approve permission requests
//! - When goals have been achieved

use super::schema::{Scenario, PermissionPolicy, SurveyStrategy};
use super::{ScenarioResult, CapturedFixtures};
use crate::fixtures::FixtureCapture;
use anyhow::Result;
use tokio::sync::mpsc;

/// Events the supervisor can observe
#[derive(Debug, Clone)]
pub enum SupervisorEvent {
    /// Palace UI state changed
    StateChanged {
        state: String,
        details: serde_json::Value,
    },

    /// Cards are ready to view
    CardsReady {
        count: usize,
        cards: Vec<CardInfo>,
    },

    /// A survey prompt appeared
    SurveyPrompt {
        question: String,
        options: Vec<SurveyOption>,
        multi_select: bool,
    },

    /// A permission request appeared
    PermissionRequest {
        command: String,
        risk_level: RiskLevel,
    },

    /// Execution progress update
    ExecutionProgress {
        current_step: usize,
        total_steps: usize,
        current_action: String,
    },

    /// Execution completed
    ExecutionComplete {
        success: bool,
        summary: String,
    },

    /// An error occurred
    Error {
        message: String,
        recoverable: bool,
    },

    /// Tool call observed (for logging)
    ToolCall {
        tool: String,
        args: String,
        timestamp: u64,
    },

    /// AI thought/commentary (for logging)
    Thought {
        content: String,
        timestamp: u64,
    },
}

/// Simplified card info for supervisor
#[derive(Debug, Clone)]
pub struct CardInfo {
    pub id: usize,
    pub title: String,
    pub category: String,
    pub description: String,
}

/// Survey option info
#[derive(Debug, Clone)]
pub struct SurveyOption {
    pub label: String,
    pub description: Option<String>,
}

/// Risk level for permission requests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,      // Read files, safe commands
    Medium,   // Write files, cargo/npm
    High,     // Delete files, network
    Critical, // System commands, rm -rf
}

/// Decisions the supervisor can make
#[derive(Debug, Clone)]
pub enum SupervisorDecision {
    /// Select cards by indices
    SelectCards(Vec<usize>),

    /// Start execution with selected cards
    StartExecution,

    /// Answer a survey
    AnswerSurvey {
        /// Selected option indices
        selections: Vec<usize>,
    },

    /// Approve a permission request
    ApprovePermission,

    /// Deny a permission request
    DenyPermission,

    /// Wait for more events
    Wait,

    /// Abort the scenario
    Abort { reason: String },

    /// Take a screenshot
    CaptureScreenshot { name: String },

    /// Navigate to a different state
    Navigate { target: String },
}

/// The AI supervisor that drives scenario execution
pub struct Supervisor {
    /// The scenario being executed
    scenario: Scenario,

    /// Permission policy
    permission_policy: PermissionPolicy,

    /// Survey strategy
    survey_strategy: SurveyStrategy,

    /// Fixture capture
    fixture_capture: FixtureCapture,

    /// Screenshots captured
    screenshots: Vec<String>,

    /// Count of auto-approvals (for limits)
    auto_approval_count: u32,

    /// Event receiver
    event_rx: Option<mpsc::Receiver<SupervisorEvent>>,

    /// Decision sender
    decision_tx: Option<mpsc::Sender<SupervisorDecision>>,
}

impl Supervisor {
    /// Create a new supervisor for a scenario
    pub fn new(scenario: &Scenario) -> Result<Self> {
        let permission_policy = scenario.permissions.clone().unwrap_or_default();
        let survey_strategy = scenario.surveys.clone().unwrap_or(SurveyStrategy::Ubj);

        Ok(Self {
            scenario: scenario.clone(),
            permission_policy,
            survey_strategy,
            fixture_capture: FixtureCapture::new(),
            screenshots: Vec::new(),
            auto_approval_count: 0,
            event_rx: None,
            decision_tx: None,
        })
    }

    /// Execute the scenario
    ///
    /// This is the main entry point. It:
    /// 1. Launches Palace on the project
    /// 2. Watches for events
    /// 3. Makes decisions based on goals
    /// 4. Captures fixtures/screenshots
    /// 5. Returns results
    pub async fn execute(self) -> Result<ScenarioResult> {
        // TODO: Connect to Palace and start watching
        // For now, return a placeholder result

        tracing::info!("Supervisor starting scenario: {}", self.scenario.scenario.name);

        // Log goals
        for (i, goal) in self.scenario.goals.iter().enumerate() {
            tracing::info!("  Goal {}: {}", i + 1, goal);
        }

        // Log constraints
        if let Some(ref constraints) = self.scenario.constraints {
            for constraint in constraints {
                tracing::info!("  Constraint: {}", constraint);
            }
        }

        // Expand project path
        let project_path = self.scenario.project.expanded_path();
        tracing::info!("Project path: {}", project_path.display());

        // Check if project exists
        if !project_path.exists() {
            if self.scenario.project.init_if_missing {
                tracing::info!("Creating project directory...");
                std::fs::create_dir_all(&project_path)?;
            } else {
                return Ok(ScenarioResult {
                    success: false,
                    completed_goals: vec![],
                    failed_goals: self.scenario.goals.clone(),
                    fixtures: None,
                    screenshots: vec![],
                    summary: format!("Project path does not exist: {}", project_path.display()),
                });
            }
        }

        // TODO: Actually launch Palace and drive the UI
        // This requires integrating with the event system

        Ok(ScenarioResult {
            success: false,
            completed_goals: vec![],
            failed_goals: self.scenario.goals.clone(),
            fixtures: Some(CapturedFixtures::default()),
            screenshots: self.screenshots,
            summary: "Supervisor execution not yet implemented".to_string(),
        })
    }

    /// Handle an incoming event
    pub async fn handle_event(&mut self, event: SupervisorEvent) -> SupervisorDecision {
        match event {
            SupervisorEvent::CardsReady { cards, .. } => {
                self.handle_cards_ready(cards).await
            }
            SupervisorEvent::SurveyPrompt { question, options, multi_select } => {
                self.handle_survey(question, options, multi_select).await
            }
            SupervisorEvent::PermissionRequest { command, risk_level } => {
                self.handle_permission(command, risk_level).await
            }
            SupervisorEvent::ToolCall { tool, args, timestamp } => {
                self.fixture_capture.add_tool(format!("[{}] {} {}", timestamp, tool, args));
                SupervisorDecision::Wait
            }
            SupervisorEvent::Thought { content, timestamp: _ } => {
                self.fixture_capture.add_thought(content);
                SupervisorDecision::Wait
            }
            SupervisorEvent::ExecutionComplete { success, summary } => {
                if success {
                    tracing::info!("Execution completed successfully: {}", summary);
                } else {
                    tracing::warn!("Execution failed: {}", summary);
                }
                SupervisorDecision::Wait // Let the main loop handle completion
            }
            SupervisorEvent::Error { message, recoverable } => {
                if recoverable {
                    tracing::warn!("Recoverable error: {}", message);
                    SupervisorDecision::Wait
                } else {
                    SupervisorDecision::Abort { reason: message }
                }
            }
            _ => SupervisorDecision::Wait,
        }
    }

    /// Handle cards being ready - decide which to select
    async fn handle_cards_ready(&mut self, cards: Vec<CardInfo>) -> SupervisorDecision {
        // Record cards in fixtures
        for card in &cards {
            self.fixture_capture.cards.push(crate::fixtures::RecordedCard {
                id: card.id,
                title: card.title.clone(),
                category: card.category.clone(),
                description: card.description.clone(),
                selected: false,
            });
        }

        // TODO: Use AI to decide which cards align with goals
        // For now, select all cards
        let indices: Vec<usize> = (0..cards.len()).collect();

        tracing::info!("Selecting {} cards", indices.len());
        SupervisorDecision::SelectCards(indices)
    }

    /// Handle a survey prompt - decide how to answer
    async fn handle_survey(
        &mut self,
        question: String,
        options: Vec<SurveyOption>,
        multi_select: bool,
    ) -> SupervisorDecision {
        match &self.survey_strategy {
            SurveyStrategy::Ubj => {
                // Use Best Judgment - call AI to decide
                self.ubj_survey(&question, &options, multi_select).await
            }
            SurveyStrategy::First => {
                SupervisorDecision::AnswerSurvey { selections: vec![0] }
            }
            SurveyStrategy::Last => {
                SupervisorDecision::AnswerSurvey {
                    selections: vec![options.len().saturating_sub(1)],
                }
            }
            SurveyStrategy::Ask => {
                // TODO: Pause and ask user
                tracing::warn!("Survey requires user input (not yet implemented)");
                SupervisorDecision::Wait
            }
            SurveyStrategy::Skip => {
                // Try to skip if possible
                SupervisorDecision::Wait
            }
            SurveyStrategy::Custom(rules) => {
                self.custom_survey(&question, &options, rules).await
            }
        }
    }

    /// Use Best Judgment for survey
    async fn ubj_survey(
        &self,
        question: &str,
        options: &[SurveyOption],
        _multi_select: bool,
    ) -> SupervisorDecision {
        // TODO: Call AI to make decision based on goals
        // For now, use simple heuristics

        let question_lower = question.to_lowercase();

        // Look for common patterns
        if question_lower.contains("continue") || question_lower.contains("proceed") {
            // Usually want to continue
            return SupervisorDecision::AnswerSurvey { selections: vec![0] };
        }

        if question_lower.contains("cancel") || question_lower.contains("abort") {
            // Usually want to continue, not cancel
            // Find option that doesn't say cancel
            for (i, opt) in options.iter().enumerate() {
                if !opt.label.to_lowercase().contains("cancel") {
                    return SupervisorDecision::AnswerSurvey { selections: vec![i] };
                }
            }
        }

        // Default: pick first option
        tracing::info!("UBJ defaulting to first option for: {}", question);
        SupervisorDecision::AnswerSurvey { selections: vec![0] }
    }

    /// Apply custom survey rules
    async fn custom_survey(
        &self,
        question: &str,
        options: &[SurveyOption],
        rules: &[super::schema::SurveyRule],
    ) -> SupervisorDecision {
        for rule in rules {
            if question.to_lowercase().contains(&rule.pattern.to_lowercase()) {
                match &rule.response {
                    super::schema::SurveyResponse::Index(i) => {
                        return SupervisorDecision::AnswerSurvey { selections: vec![*i] };
                    }
                    super::schema::SurveyResponse::Contains(text) => {
                        for (i, opt) in options.iter().enumerate() {
                            if opt.label.to_lowercase().contains(&text.to_lowercase()) {
                                return SupervisorDecision::AnswerSurvey { selections: vec![i] };
                            }
                        }
                    }
                    super::schema::SurveyResponse::Ubj => {
                        return self.ubj_survey(question, options, false).await;
                    }
                    super::schema::SurveyResponse::Ask => {
                        tracing::warn!("Survey rule requests user input (not yet implemented)");
                        return SupervisorDecision::Wait;
                    }
                }
            }
        }

        // No rule matched, use UBJ
        self.ubj_survey(question, options, false).await
    }

    /// Handle a permission request
    async fn handle_permission(&mut self, command: String, risk_level: RiskLevel) -> SupervisorDecision {
        

        // Check deny list first
        for denied in &self.permission_policy.deny {
            if self.permission_matches(&command, denied, risk_level) {
                tracing::warn!("Permission denied by policy: {}", command);
                return SupervisorDecision::DenyPermission;
            }
        }

        // Check allow list
        for allowed in &self.permission_policy.allow {
            if self.permission_matches(&command, allowed, risk_level) {
                // Check auto-approval limit
                if let Some(max) = self.permission_policy.max_auto_approvals {
                    if self.auto_approval_count >= max {
                        tracing::warn!("Auto-approval limit reached, denying: {}", command);
                        return SupervisorDecision::DenyPermission;
                    }
                }
                self.auto_approval_count += 1;
                tracing::info!("Permission auto-approved: {}", command);
                return SupervisorDecision::ApprovePermission;
            }
        }

        // Not in allow or deny - check ask_unknown
        if self.permission_policy.ask_unknown {
            // TODO: Call AI to decide
            tracing::info!("Permission requires AI decision: {}", command);

            // For now, approve low/medium risk, deny high/critical
            match risk_level {
                RiskLevel::Low | RiskLevel::Medium => {
                    tracing::info!("Auto-approving {} risk permission",
                        if risk_level == RiskLevel::Low { "low" } else { "medium" });
                    SupervisorDecision::ApprovePermission
                }
                RiskLevel::High | RiskLevel::Critical => {
                    tracing::warn!("Denying {} risk permission",
                        if risk_level == RiskLevel::High { "high" } else { "critical" });
                    SupervisorDecision::DenyPermission
                }
            }
        } else {
            // Default deny
            SupervisorDecision::DenyPermission
        }
    }

    /// Check if a command matches a permission type
    fn permission_matches(&self, command: &str, perm: &super::schema::PermissionType, risk: RiskLevel) -> bool {
        use super::schema::PermissionType;

        match perm {
            PermissionType::Read => command.contains("Read") || command.contains("read"),
            PermissionType::Write => command.contains("Write") || command.contains("Edit"),
            PermissionType::Create => command.contains("Create") || command.contains("create"),
            PermissionType::Delete => command.contains("Delete") || command.contains("delete"),
            PermissionType::Cargo => command.starts_with("cargo "),
            PermissionType::Npm => command.starts_with("npm ") || command.starts_with("yarn "),
            PermissionType::Git => command.starts_with("git "),
            PermissionType::Network => command.contains("curl") || command.contains("wget") || command.contains("http"),
            PermissionType::RmRf => command.contains("rm -rf") || command.contains("rm -r"),
            PermissionType::Bash => command.contains("bash") || command.contains("sh -c"),
            PermissionType::BashSafe => {
                (command.contains("bash") || command.contains("sh -c"))
                    && !command.contains("rm ")
                    && !command.contains("curl")
                    && !command.contains("wget")
                    && risk != RiskLevel::Critical
            }
            PermissionType::Pattern(pat) => {
                // Simple glob-style matching
                if pat.starts_with('*') && pat.ends_with('*') {
                    command.contains(&pat[1..pat.len()-1])
                } else if pat.starts_with('*') {
                    command.ends_with(&pat[1..])
                } else if pat.ends_with('*') {
                    command.starts_with(&pat[..pat.len()-1])
                } else {
                    command == pat
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::Low != RiskLevel::Critical);
    }

    #[test]
    fn test_permission_pattern_matching() {
        use super::super::schema::PermissionType;

        let scenario = Scenario {
            scenario: super::super::schema::ScenarioMeta {
                name: "test".to_string(),
                description: None,
                version: "1.0".to_string(),
                author: None,
                tags: vec![],
            },
            goals: vec!["test".to_string()],
            steps: vec![],
            constraints: None,
            permissions: None,
            surveys: None,
            project: super::super::schema::ProjectConfig {
                path: "/tmp".to_string(),
                init_if_missing: false,
                git_clone: None,
                branch: None,
                template: None,
            },
            capture: None,
            okrs: None,
            requirements: None,
        };

        let supervisor = Supervisor::new(&scenario).unwrap();

        // Test cargo matching
        assert!(supervisor.permission_matches("cargo build", &PermissionType::Cargo, RiskLevel::Low));
        assert!(!supervisor.permission_matches("npm install", &PermissionType::Cargo, RiskLevel::Low));

        // Test pattern matching
        let pattern = PermissionType::Pattern("cargo *".to_string());
        assert!(supervisor.permission_matches("cargo build", &pattern, RiskLevel::Low));
        assert!(supervisor.permission_matches("cargo test", &pattern, RiskLevel::Low));
    }
}
