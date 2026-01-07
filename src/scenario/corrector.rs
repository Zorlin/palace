//! Scenario Corrector - LLM-powered YAML correction with iterative feedback
//!
//! Sends broken/incorrect scenario files to Claude for correction,
//! then displays the diff for user approval with optional feedback loop.

use super::diff::{DiffChunk, DiffViewer};
use super::schema::Scenario;
use anyhow::Result;
use std::path::PathBuf;

/// State machine for the correction workflow
#[derive(Debug)]
pub enum CorrectorState {
    /// Loading/parsing the input file
    Loading {
        input_path: PathBuf,
    },

    /// Sending to LLM for correction
    Correcting {
        input_path: PathBuf,
        original: String,
        parse_error: Option<String>,
    },

    /// Showing diff viewer for approval
    DiffViewer {
        input_path: PathBuf,
        viewer: DiffViewer,
    },

    /// User is providing feedback on a specific chunk
    Feedback {
        input_path: PathBuf,
        viewer: DiffViewer,
        highlighted_chunk: usize,
        feedback_options: Vec<FeedbackOption>,
        selected_option: usize,
        custom_feedback: Option<String>,
        custom_cursor: usize,
    },

    /// Waiting for LLM to refine based on feedback
    Refining {
        input_path: PathBuf,
        viewer: DiffViewer,
        feedback: String,
    },

    /// Correction accepted, ready to save
    Accepted {
        input_path: PathBuf,
        corrected: String,
        output_path: Option<PathBuf>,
    },

    /// Correction cancelled
    Cancelled,

    /// Error state
    Error {
        message: String,
    },
}

/// A feedback option for the user to select
#[derive(Debug, Clone)]
pub struct FeedbackOption {
    /// Short label for gamepad selection
    pub label: String,
    /// Detailed feedback to send to LLM
    pub feedback_text: String,
}

impl FeedbackOption {
    pub fn new(label: impl Into<String>, feedback_text: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            feedback_text: feedback_text.into(),
        }
    }
}

/// Default feedback options (before LLM generates context-specific ones)
pub fn default_feedback_options() -> Vec<FeedbackOption> {
    vec![
        FeedbackOption::new(
            "Keep original",
            "Keep the original content for this section - do not change it.",
        ),
        FeedbackOption::new(
            "Wrong fix",
            "This correction is incorrect. Please try a different approach.",
        ),
        FeedbackOption::new(
            "Missing context",
            "The correction doesn't account for the surrounding context.",
        ),
        FeedbackOption::new(
            "Different intent",
            "My intent was different than what you assumed. Let me explain.",
        ),
        FeedbackOption::new("Custom feedback...", ""),
    ]
}

/// The scenario corrector
#[derive(Debug)]
pub struct ScenarioCorrector {
    /// Current state
    pub state: CorrectorState,
    /// Output path (if specified)
    pub output_path: Option<PathBuf>,
    /// Whether to auto-accept without showing UI
    pub auto_accept: bool,
}

impl ScenarioCorrector {
    /// Create a new corrector for an input file
    pub fn new(input_path: PathBuf, output_path: Option<PathBuf>, auto_accept: bool) -> Self {
        Self {
            state: CorrectorState::Loading { input_path },
            output_path,
            auto_accept,
        }
    }

    /// Load and parse the input file
    pub fn load(&mut self) -> Result<()> {
        let input_path = match &self.state {
            CorrectorState::Loading { input_path } => input_path.clone(),
            _ => return Ok(()),
        };

        // Read the file
        let content = std::fs::read_to_string(&input_path)?;

        // Try to parse as YAML
        let parse_error = match serde_yaml::from_str::<Scenario>(&content) {
            Ok(_) => None,
            Err(e) => Some(e.to_string()),
        };

        self.state = CorrectorState::Correcting {
            input_path,
            original: content,
            parse_error,
        };

        Ok(())
    }

    /// Get the correction prompt for the LLM
    pub fn get_correction_prompt(&self) -> Option<String> {
        match &self.state {
            CorrectorState::Correcting {
                original,
                parse_error,
                ..
            } => {
                let error_context = parse_error
                    .as_ref()
                    .map(|e| format!("Parse error: {}", e))
                    .unwrap_or_else(|| "No parse error, but may have semantic issues.".to_string());

                Some(format!(
                    r#"{}

Given this scenario file (which may have errors):
```yaml
{}
```

{}

Return the corrected YAML in a code block, followed by an explanation of what you changed and why.

For each change, explain briefly what was wrong and how you fixed it.
"#,
                    CORRECTION_PROMPT_HEADER, original, error_context
                ))
            }
            CorrectorState::Refining {
                viewer, feedback, ..
            } => Some(format!(
                r#"{}

Original scenario:
```yaml
{}
```

Your previous correction:
```yaml
{}
```

User feedback: {}

Please provide an updated correction that addresses the feedback.
Return the corrected YAML in a code block, followed by an explanation.
"#,
                CORRECTION_PROMPT_HEADER, viewer.original, viewer.corrected, feedback
            )),
            _ => None,
        }
    }

    /// Apply a correction from the LLM
    pub fn apply_correction(&mut self, corrected: String, explanation: String) {
        match &self.state {
            CorrectorState::Correcting {
                input_path,
                original,
                ..
            } => {
                if self.auto_accept {
                    self.state = CorrectorState::Accepted {
                        input_path: input_path.clone(),
                        corrected,
                        output_path: self.output_path.clone(),
                    };
                } else {
                    let viewer = DiffViewer::new(original.clone(), corrected, explanation);
                    self.state = CorrectorState::DiffViewer {
                        input_path: input_path.clone(),
                        viewer,
                    };
                }
            }
            CorrectorState::Refining {
                input_path,
                viewer: old_viewer,
                ..
            } => {
                let mut viewer =
                    DiffViewer::new(old_viewer.original.clone(), corrected, explanation);
                viewer.iteration = old_viewer.iteration + 1;

                self.state = CorrectorState::DiffViewer {
                    input_path: input_path.clone(),
                    viewer,
                };
            }
            _ => {}
        }
    }

    /// Accept the current correction
    pub fn accept(&mut self) {
        if let CorrectorState::DiffViewer { input_path, viewer } = &self.state {
            self.state = CorrectorState::Accepted {
                input_path: input_path.clone(),
                corrected: viewer.corrected.clone(),
                output_path: self.output_path.clone(),
            };
        }
    }

    /// Start providing feedback on a chunk
    pub fn start_feedback(&mut self) {
        if let CorrectorState::DiffViewer { input_path, viewer } = &mut self.state {
            let highlighted_chunk = viewer.cursor_chunk;

            self.state = CorrectorState::Feedback {
                input_path: input_path.clone(),
                viewer: std::mem::replace(
                    viewer,
                    DiffViewer::new(String::new(), String::new(), String::new()),
                ),
                highlighted_chunk,
                feedback_options: default_feedback_options(),
                selected_option: 0,
                custom_feedback: None,
                custom_cursor: 0,
            };
        }
    }

    /// Navigate feedback options
    pub fn navigate_feedback(&mut self, direction: i32) {
        if let CorrectorState::Feedback {
            selected_option,
            feedback_options,
            ..
        } = &mut self.state
        {
            let new_idx = (*selected_option as i32 + direction)
                .max(0)
                .min(feedback_options.len().saturating_sub(1) as i32)
                as usize;
            *selected_option = new_idx;
        }
    }

    /// Select the current feedback option
    pub fn select_feedback(&mut self) {
        if let CorrectorState::Feedback {
            input_path,
            viewer,
            feedback_options,
            selected_option,
            custom_feedback,
            ..
        } = &mut self.state
        {
            let option = &feedback_options[*selected_option];

            if option.label == "Custom feedback..." {
                // Enable custom feedback input
                *custom_feedback = Some(String::new());
            } else {
                // Submit the feedback
                let feedback = option.feedback_text.clone();
                self.state = CorrectorState::Refining {
                    input_path: input_path.clone(),
                    viewer: std::mem::replace(
                        viewer,
                        DiffViewer::new(String::new(), String::new(), String::new()),
                    ),
                    feedback,
                };
            }
        }
    }

    /// Submit custom feedback
    pub fn submit_custom_feedback(&mut self) {
        if let CorrectorState::Feedback {
            input_path,
            viewer,
            custom_feedback: Some(feedback),
            ..
        } = &mut self.state
        {
            if !feedback.is_empty() {
                self.state = CorrectorState::Refining {
                    input_path: input_path.clone(),
                    viewer: std::mem::replace(
                        viewer,
                        DiffViewer::new(String::new(), String::new(), String::new()),
                    ),
                    feedback: feedback.clone(),
                };
            }
        }
    }

    /// Cancel feedback and return to diff viewer
    pub fn cancel_feedback(&mut self) {
        if let CorrectorState::Feedback {
            input_path, viewer, ..
        } = &mut self.state
        {
            self.state = CorrectorState::DiffViewer {
                input_path: input_path.clone(),
                viewer: std::mem::replace(
                    viewer,
                    DiffViewer::new(String::new(), String::new(), String::new()),
                ),
            };
        }
    }

    /// Cancel the entire correction
    pub fn cancel(&mut self) {
        self.state = CorrectorState::Cancelled;
    }

    /// Save the corrected file
    pub fn save(&self) -> Result<PathBuf> {
        match &self.state {
            CorrectorState::Accepted {
                input_path,
                corrected,
                output_path,
            } => {
                let path = output_path.clone().unwrap_or_else(|| input_path.clone());
                std::fs::write(&path, corrected)?;
                Ok(path)
            }
            _ => anyhow::bail!("Cannot save - correction not accepted"),
        }
    }

    /// Get the current diff viewer (if in that state)
    pub fn viewer(&self) -> Option<&DiffViewer> {
        match &self.state {
            CorrectorState::DiffViewer { viewer, .. } => Some(viewer),
            CorrectorState::Feedback { viewer, .. } => Some(viewer),
            _ => None,
        }
    }

    /// Get mutable diff viewer
    pub fn viewer_mut(&mut self) -> Option<&mut DiffViewer> {
        match &mut self.state {
            CorrectorState::DiffViewer { viewer, .. } => Some(viewer),
            CorrectorState::Feedback { viewer, .. } => Some(viewer),
            _ => None,
        }
    }
}

/// Header for the correction prompt
const CORRECTION_PROMPT_HEADER: &str = r#"You are fixing a Palace scenario file. Palace scenarios use YAML format with:
- scenario: metadata (name, description, version, author, tags)
- goals: list of objectives (for AI-driven execution)
- steps: list of plain English commands (for scripted execution)
- permissions: allow/deny lists for auto-approval
- surveys: response strategy (ubj, first, last, ask, skip)
- project: path and init settings
- capture: screenshot/fixture configuration
- okrs: success criteria
- requirements: hard requirements
- constraints: execution constraints

VALID STEP COMMANDS:
- open project <path>
- click "<target>"
- select all cards / select cards 1, 3, 5 / select cards 1-5
- deselect card <n> / deselect all
- execute / start execution
- wait for cards / wait for completion / wait <N> seconds
- screenshot "<filename>"
- approve all [cargo|network|...] permissions
- deny [network|rm_rf|...] permissions
- use best judgment
- log "<message>"

VALID PERMISSION TYPES:
read, write, create, delete, cargo, npm, git, network, rm_rf, bash, bash_safe

VALID SURVEY STRATEGIES:
ubj, first, last, ask, skip"#;

/// Parse the LLM response to extract corrected YAML and explanation
pub fn parse_correction_response(response: &str) -> Option<(String, String)> {
    // Look for YAML code block
    let yaml_start = response.find("```yaml").or_else(|| response.find("```yml"))?;
    let yaml_content_start = response[yaml_start..].find('\n')? + yaml_start + 1;
    let yaml_end = response[yaml_content_start..].find("```")? + yaml_content_start;

    let yaml = response[yaml_content_start..yaml_end].trim().to_string();

    // Everything after the code block is explanation
    let explanation_start = yaml_end + 3;
    let explanation = response[explanation_start..].trim().to_string();

    Some((yaml, explanation))
}

/// Generate context-specific feedback options for a chunk
pub fn generate_feedback_options_prompt(chunk: &DiffChunk) -> String {
    format!(
        r#"The user is reviewing a diff correction and wants to provide feedback.

Original chunk (lines {}-{}):
```
{}
```

Corrected chunk:
```
{}
```

Generate 4-5 feedback options the user might want to give about this specific change.
Each option should be a concise label (2-4 words) and a detailed feedback message.

Return JSON array:
[
  {{"label": "short label", "feedback_text": "detailed feedback for the LLM"}},
  ...
]

Include options like:
- Keep the original (don't change)
- Specific issues with the correction
- Alternative approaches
- Missing context or requirements
"#,
        chunk.line_start_original,
        chunk.line_start_original + chunk.original_lines.len().saturating_sub(1),
        chunk.original_lines.join("\n"),
        chunk.corrected_lines.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_correction_response() {
        let response = r#"Here is the corrected scenario:

```yaml
scenario:
  name: "Test"
  version: "1.0"
```

I fixed the indentation and added the required version field.
"#;

        let (yaml, explanation) = parse_correction_response(response).unwrap();
        assert!(yaml.contains("scenario:"));
        assert!(yaml.contains("name: \"Test\""));
        assert!(explanation.contains("indentation"));
    }

    #[test]
    fn test_corrector_state_flow() {
        let corrector = ScenarioCorrector::new(
            PathBuf::from("/tmp/test.yml"),
            None,
            false,
        );

        assert!(matches!(corrector.state, CorrectorState::Loading { .. }));
    }

    #[test]
    fn test_default_feedback_options() {
        let options = default_feedback_options();
        assert!(options.len() >= 4);
        assert!(options.iter().any(|o| o.label.contains("original")));
        assert!(options.iter().any(|o| o.label.contains("Custom")));
    }
}
