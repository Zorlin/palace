//! Session Recorder - Record Palace sessions as scenario files
//!
//! Records high-level user actions during a Palace session and exports
//! them as scenario YAML files with semantic waits.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::schema::{
    CaptureConfig, PermissionPolicy, PermissionType, ProjectConfig, Scenario, ScenarioMeta,
    SurveyStrategy,
};
use super::script::ScriptStep;

/// A snapshot of a suggestion card (for AI context capture)
#[derive(Debug, Clone)]
pub struct CardSnapshot {
    pub title: String,
    pub category: String,
    pub description: String,
}

/// Answer to a survey question
#[derive(Debug, Clone)]
pub struct SurveyAnswer {
    pub question: String,
    pub selected_index: usize,
    pub selected_label: String,
    pub all_options: Vec<String>,
}

/// High-level events recorded during a session
#[derive(Debug, Clone)]
pub enum RecordedEvent {
    // High-level actions only (no keystrokes)
    ProjectOpened { path: PathBuf },
    CardsReceived { count: usize, cards: Option<Vec<CardSnapshot>> },
    CardsSelected { indices: Vec<usize> },
    ExecutionStarted,
    PermissionGranted { command: String },
    PermissionDenied { command: String },
    SurveyAnswered { question: String, selection: usize, options: Vec<String> },
    ExecutionCompleted { success: bool },
    Screenshot { filename: String },

    // For AI context capture (optional)
    AiThought { content: String },
    ToolCall { tool: String, args: String },

    // Checkpoint marker
    Checkpoint { filename: String },
}

/// Configuration for the recorder
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    /// Include AI responses/thoughts in output
    pub include_ai_context: bool,
    /// Directory for checkpoint files
    pub checkpoint_dir: PathBuf,
    /// Output file path
    pub output_path: Option<PathBuf>,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            include_ai_context: false,
            checkpoint_dir: PathBuf::from("/tmp/palace-recording"),
            output_path: None,
        }
    }
}

/// Session recorder that captures high-level actions
#[derive(Debug, Clone)]
pub struct SessionRecorder {
    /// When recording started
    pub start_time: Instant,
    /// Project being recorded
    pub project_path: PathBuf,
    /// All recorded events
    pub events: Vec<RecordedEvent>,

    /// Cards seen during session
    pub cards_seen: Vec<CardSnapshot>,
    /// Permissions that were granted
    pub permissions_granted: Vec<String>,
    /// Surveys that were answered
    pub surveys_answered: Vec<SurveyAnswer>,

    /// Checkpointing
    last_checkpoint: Instant,
    checkpoint_dir: PathBuf,

    /// Configuration
    include_ai_context: bool,
    output_path: Option<PathBuf>,
}

impl SessionRecorder {
    /// Checkpoint interval (5 minutes)
    const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(5 * 60);

    /// Create a new session recorder
    pub fn new(project_path: PathBuf, config: RecorderConfig) -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            project_path,
            events: Vec::new(),
            cards_seen: Vec::new(),
            permissions_granted: Vec::new(),
            surveys_answered: Vec::new(),
            last_checkpoint: now,
            checkpoint_dir: config.checkpoint_dir,
            include_ai_context: config.include_ai_context,
            output_path: config.output_path,
        }
    }

    /// Record an event
    pub fn record(&mut self, event: RecordedEvent) {
        // Check if we need to checkpoint based on time
        self.maybe_time_checkpoint();

        // Check if this is a significant event that should trigger checkpoint
        self.maybe_event_checkpoint(&event);

        self.events.push(event);
    }

    /// Record a project being opened
    pub fn record_project_opened(&mut self, path: PathBuf) {
        self.record(RecordedEvent::ProjectOpened { path });
    }

    /// Record cards being received
    pub fn record_cards_received(&mut self, count: usize, cards: Option<Vec<CardSnapshot>>) {
        if let Some(ref c) = cards {
            self.cards_seen.extend(c.iter().cloned());
        }
        self.record(RecordedEvent::CardsReceived { count, cards });
    }

    /// Record cards being selected
    pub fn record_cards_selected(&mut self, indices: Vec<usize>) {
        self.record(RecordedEvent::CardsSelected { indices });
    }

    /// Record execution starting
    pub fn record_execution_started(&mut self) {
        self.record(RecordedEvent::ExecutionStarted);
    }

    /// Record a permission being granted
    pub fn record_permission_granted(&mut self, command: String) {
        self.permissions_granted.push(command.clone());
        self.record(RecordedEvent::PermissionGranted { command });
    }

    /// Record a permission being denied
    pub fn record_permission_denied(&mut self, command: String) {
        self.record(RecordedEvent::PermissionDenied { command });
    }

    /// Record a survey being answered
    pub fn record_survey_answered(
        &mut self,
        question: String,
        selection: usize,
        options: Vec<String>,
    ) {
        let selected_label = options.get(selection).cloned().unwrap_or_default();
        self.surveys_answered.push(SurveyAnswer {
            question: question.clone(),
            selected_index: selection,
            selected_label,
            all_options: options.clone(),
        });
        self.record(RecordedEvent::SurveyAnswered {
            question,
            selection,
            options,
        });
    }

    /// Record execution completing
    pub fn record_execution_completed(&mut self, success: bool) {
        self.record(RecordedEvent::ExecutionCompleted { success });
    }

    /// Record a screenshot being taken
    pub fn record_screenshot(&mut self, filename: String) {
        self.record(RecordedEvent::Screenshot { filename });
    }

    /// Record an AI thought (if include_ai_context is enabled)
    pub fn record_ai_thought(&mut self, content: String) {
        if self.include_ai_context {
            self.record(RecordedEvent::AiThought { content });
        }
    }

    /// Record a tool call (if include_ai_context is enabled)
    pub fn record_tool_call(&mut self, tool: String, args: String) {
        if self.include_ai_context {
            self.record(RecordedEvent::ToolCall { tool, args });
        }
    }

    /// Check if we should checkpoint based on time
    fn maybe_time_checkpoint(&mut self) {
        let elapsed = self.last_checkpoint.elapsed();
        if elapsed >= Self::CHECKPOINT_INTERVAL {
            self.save_checkpoint();
        }
    }

    /// Check if this event should trigger a checkpoint
    fn maybe_event_checkpoint(&mut self, event: &RecordedEvent) {
        match event {
            RecordedEvent::ExecutionStarted
            | RecordedEvent::ExecutionCompleted { .. }
            | RecordedEvent::CardsSelected { .. } => {
                self.save_checkpoint();
            }
            _ => {}
        }
    }

    /// Save a checkpoint
    fn save_checkpoint(&mut self) {
        let filename = format!(
            "checkpoint-{}.yml",
            chrono::Utc::now().format("%Y%m%d-%H%M%S")
        );
        let path = self.checkpoint_dir.join(&filename);

        // Ensure directory exists
        if let Err(e) = std::fs::create_dir_all(&self.checkpoint_dir) {
            tracing::warn!("Failed to create checkpoint directory: {}", e);
            return;
        }

        match self.export_scenario() {
            Ok(scenario) => {
                if let Ok(yaml) = serde_yaml::to_string(&scenario) {
                    if let Err(e) = std::fs::write(&path, yaml) {
                        tracing::warn!("Failed to write checkpoint: {}", e);
                    } else {
                        tracing::debug!("Checkpoint saved: {}", path.display());
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to export scenario for checkpoint: {}", e);
            }
        }

        self.events
            .push(RecordedEvent::Checkpoint { filename });
        self.last_checkpoint = Instant::now();
    }

    /// Get elapsed recording time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Export the recorded session as a Scenario
    pub fn export_scenario(&self) -> anyhow::Result<Scenario> {
        let mut steps = Vec::new();

        // Convert events to semantic steps (no raw delays)
        for event in &self.events {
            match event {
                RecordedEvent::ProjectOpened { path } => {
                    steps.push(format!("open project {}", path.display()));
                }

                // SEMANTIC WAIT: "wait for cards" instead of "wait 3.2s"
                RecordedEvent::CardsReceived { count, cards } => {
                    steps.push("wait for cards".into());
                    if self.include_ai_context {
                        if let Some(cards) = cards {
                            steps.push(format!(
                                "# Received {} cards: {}",
                                count,
                                cards
                                    .iter()
                                    .map(|c| c.title.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ));
                        }
                    }
                }

                RecordedEvent::CardsSelected { indices } => {
                    if indices.len() == self.cards_seen.len() && !indices.is_empty() {
                        steps.push("select all cards".into());
                    } else if !indices.is_empty() {
                        steps.push(format!(
                            "select cards {}",
                            indices
                                .iter()
                                .map(|i| (i + 1).to_string()) // 1-indexed for user
                                .collect::<Vec<_>>()
                                .join(", ")
                        ));
                    }
                }

                RecordedEvent::ExecutionStarted => {
                    steps.push("execute".into());
                }

                // SEMANTIC WAIT: "wait for completion" instead of timing
                RecordedEvent::ExecutionCompleted { success } => {
                    steps.push("wait for completion".into());
                    if !success {
                        steps.push("# Note: execution failed".into());
                    }
                }

                RecordedEvent::PermissionGranted { command } => {
                    steps.push(format!("# Permission granted: {}", command));
                }

                RecordedEvent::PermissionDenied { command } => {
                    steps.push(format!("# Permission denied: {}", command));
                }

                RecordedEvent::SurveyAnswered {
                    question,
                    selection,
                    options,
                } => {
                    let label = options.get(*selection).cloned().unwrap_or_default();
                    steps.push(format!("# Survey: {} -> {}", question, label));
                }

                RecordedEvent::Screenshot { filename } => {
                    steps.push(format!("screenshot \"{}\"", filename));
                }

                RecordedEvent::AiThought { content } if self.include_ai_context => {
                    steps.push(format!("# AI: {}", content.lines().next().unwrap_or("")));
                }

                RecordedEvent::ToolCall { tool, args } if self.include_ai_context => {
                    steps.push(format!("# Tool: {} {}", tool, args));
                }

                RecordedEvent::Checkpoint { .. } => {
                    // Don't include checkpoint markers in output
                }

                _ => {}
            }
        }

        // Convert steps to ScriptStep enum
        let script_steps: Vec<ScriptStep> = steps.into_iter().map(ScriptStep::Simple).collect();

        Ok(Scenario {
            scenario: ScenarioMeta {
                name: format!(
                    "Recorded: {}",
                    self.project_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("session")
                ),
                description: Some(format!(
                    "Recorded session on {}",
                    chrono::Local::now().format("%Y-%m-%d %H:%M")
                )),
                version: "1.0".into(),
                author: None,
                tags: vec!["recorded".into()],
            },
            goals: vec![], // Recorded sessions are step-based, not goal-based
            steps: script_steps,
            permissions: Some(PermissionPolicy {
                allow: self.infer_permissions(),
                deny: vec![],
                ask_unknown: true,
                max_auto_approvals: None,
            }),
            surveys: Some(SurveyStrategy::Ubj),
            project: ProjectConfig {
                path: self.project_path.to_string_lossy().into(),
                init_if_missing: false,
                git_clone: None,
                branch: None,
                template: None,
            },
            capture: Some(self.build_capture_config()),
            okrs: None,
            requirements: None,
            constraints: None,
        })
    }

    /// Infer permissions from granted commands
    fn infer_permissions(&self) -> Vec<PermissionType> {
        let mut perms = HashSet::new();
        perms.insert(PermissionType::Read);
        perms.insert(PermissionType::Write);

        for cmd in &self.permissions_granted {
            if cmd.starts_with("cargo ") {
                perms.insert(PermissionType::Cargo);
            } else if cmd.starts_with("git ") {
                perms.insert(PermissionType::Git);
            } else if cmd.starts_with("npm ") || cmd.starts_with("yarn ") {
                perms.insert(PermissionType::Npm);
            } else if cmd.contains("curl") || cmd.contains("wget") || cmd.contains("http") {
                perms.insert(PermissionType::Network);
            }
        }

        perms.into_iter().collect()
    }

    /// Build capture config based on what was recorded
    fn build_capture_config(&self) -> CaptureConfig {
        let has_screenshots = self
            .events
            .iter()
            .any(|e| matches!(e, RecordedEvent::Screenshot { .. }));

        CaptureConfig {
            screenshots: has_screenshots,
            fixtures: self.include_ai_context,
            animated_svg: false,
            output_dir: "./captures".into(),
            screenshot_on: vec![],
        }
    }

    /// Save the scenario to a file
    pub fn save(&self, output_path: &std::path::Path) -> anyhow::Result<()> {
        let scenario = self.export_scenario()?;
        let yaml = serde_yaml::to_string(&scenario)?;
        std::fs::write(output_path, yaml)?;
        Ok(())
    }

    /// Get output path (either configured or default)
    pub fn get_output_path(&self) -> PathBuf {
        self.output_path.clone().unwrap_or_else(|| {
            let name = self
                .project_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("session");
            PathBuf::from(format!("recorded-{}.yml", name))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recorder_basic_flow() {
        let mut recorder = SessionRecorder::new(
            PathBuf::from("/tmp/test-project"),
            RecorderConfig::default(),
        );

        recorder.record_project_opened(PathBuf::from("/tmp/test-project"));
        recorder.record_cards_received(3, None);
        recorder.record_cards_selected(vec![0, 1, 2]);
        recorder.record_execution_started();
        recorder.record_execution_completed(true);

        let scenario = recorder.export_scenario().unwrap();
        assert_eq!(scenario.steps.len(), 5);
        assert!(scenario.scenario.name.contains("Recorded"));
    }

    #[test]
    fn test_semantic_waits() {
        let mut recorder = SessionRecorder::new(
            PathBuf::from("/tmp/test"),
            RecorderConfig::default(),
        );

        recorder.record_cards_received(5, None);
        recorder.record_execution_completed(true);

        let scenario = recorder.export_scenario().unwrap();

        // Check that we use semantic waits
        let step_strings: Vec<&str> = scenario
            .steps
            .iter()
            .map(|s| match s {
                ScriptStep::Simple(text) => text.as_str(),
                ScriptStep::Structured(_) => "",
            })
            .collect();

        assert!(step_strings.contains(&"wait for cards"));
        assert!(step_strings.contains(&"wait for completion"));
    }

    #[test]
    fn test_permission_inference() {
        let mut recorder = SessionRecorder::new(
            PathBuf::from("/tmp/test"),
            RecorderConfig::default(),
        );

        recorder.record_permission_granted("cargo build".into());
        recorder.record_permission_granted("git commit -m 'test'".into());

        let scenario = recorder.export_scenario().unwrap();
        let perms = scenario.permissions.unwrap();

        assert!(perms.allow.contains(&PermissionType::Cargo));
        assert!(perms.allow.contains(&PermissionType::Git));
    }
}
