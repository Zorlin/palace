//! Plain English scripting for scenarios
//!
//! Modern AutoHotKey/SCAR-style automation with natural language commands.
//! Scripts handle the 99% that's deterministic, AI supervisor handles the rest.
//!
//! ## Example Commands
//!
//! ```text
//! open project ~/projects/asteroids
//! click "Suggest Tasks"
//! wait for cards
//! select all cards
//! select cards 1, 3, 5
//! deselect card 2
//! execute
//! wait for completion
//! screenshot "result.png"
//! approve all cargo permissions
//! deny network permissions
//! use best judgment on surveys
//! press F2
//! wait 5 seconds
//! if state is "executing" then wait for completion
//! ```

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// A parsed script command
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScriptStep {
    /// Simple string command (parsed at runtime)
    Simple(String),
    /// Structured command with explicit fields
    Structured(StructuredStep),
}

/// Structured step with explicit action and parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredStep {
    pub action: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub value: Option<serde_yaml::Value>,
    #[serde(default)]
    pub timeout: Option<String>,
    #[serde(default)]
    pub condition: Option<String>,
}

/// Parsed command ready for execution
#[derive(Debug, Clone)]
pub enum Command {
    // Navigation
    OpenProject { path: String },
    Click { target: String },
    Press { key: String },

    // Card operations (batch - fast!)
    SelectAllCards,
    SelectCards { indices: Vec<usize> },
    DeselectCard { index: usize },
    DeselectAllCards,

    // Execution
    Execute,
    StartAnalysis,

    // Synchronization
    WaitForCards { min_count: Option<usize>, timeout_secs: Option<u32> },
    WaitForState { state: String, timeout_secs: Option<u32> },
    WaitForCompletion { timeout_secs: Option<u32> },
    Wait { seconds: f32 },

    // Capture
    Screenshot { filename: String },
    CaptureFixtures { filename: String },

    // Permission rules (batch)
    ApproveAllPermissions { pattern: Option<String> },
    DenyAllPermissions { pattern: Option<String> },
    ApprovePermission { pattern: String },
    DenyPermission { pattern: String },

    // Survey handling
    UseBestJudgment,
    AnswerSurvey { strategy: String },

    // Control flow
    If { condition: String, then: Box<Command> },
    Loop { count: usize, body: Vec<Command> },

    // Edit mode
    EnterEditMode,
    ExitEditMode,
    DragPanel { panel: String, to: (f32, f32) },

    // Utility
    Log { message: String },
    AssertState { state: String },
    Fail { message: String },
}

/// Parse a plain English command string
pub fn parse_command(input: &str) -> Result<Command> {
    let input = input.trim();
    let lower = input.to_lowercase();

    // Open project
    if lower.starts_with("open project ") {
        let path = input[13..].trim().to_string();
        return Ok(Command::OpenProject { path });
    }

    // Click
    if lower.starts_with("click ") {
        let target = input[6..].trim().trim_matches('"').to_string();
        return Ok(Command::Click { target });
    }

    // Press key
    if lower.starts_with("press ") {
        let key = input[6..].trim().to_string();
        return Ok(Command::Press { key });
    }

    // Select all cards
    if lower == "select all cards" || lower == "select all" {
        return Ok(Command::SelectAllCards);
    }

    // Deselect all cards
    if lower == "deselect all cards" || lower == "deselect all" || lower == "clear selection" {
        return Ok(Command::DeselectAllCards);
    }

    // Select cards 1, 3, 5
    if lower.starts_with("select cards ") || lower.starts_with("select card ") {
        let nums_str = if lower.starts_with("select cards ") {
            &input[13..]
        } else {
            &input[12..]
        };
        let indices = parse_number_list(nums_str)?;
        return Ok(Command::SelectCards { indices });
    }

    // Deselect card 2
    if lower.starts_with("deselect card ") {
        let num_str = input[14..].trim();
        let index: usize = num_str.parse()
            .map_err(|_| anyhow::anyhow!("Invalid card number: {}", num_str))?;
        return Ok(Command::DeselectCard { index });
    }

    // Execute
    if lower == "execute" || lower == "start execution" || lower == "run" {
        return Ok(Command::Execute);
    }

    // Start analysis
    if lower == "analyze" || lower == "start analysis" || lower == "run analysis" {
        return Ok(Command::StartAnalysis);
    }

    // Wait for cards
    if lower.starts_with("wait for cards") {
        let rest = &lower[14..].trim();
        let min_count = if rest.starts_with("count >=") || rest.starts_with(">= ") {
            let num_part = rest.trim_start_matches("count >=").trim_start_matches(">= ").trim();
            num_part.parse().ok()
        } else if rest.is_empty() {
            None
        } else {
            rest.parse().ok()
        };
        return Ok(Command::WaitForCards { min_count, timeout_secs: None });
    }

    // Wait for state
    if lower.starts_with("wait for state ") {
        let state = input[15..].trim().trim_matches('"').to_string();
        return Ok(Command::WaitForState { state, timeout_secs: None });
    }

    // Wait for completion
    if lower == "wait for completion" || lower == "wait until complete" || lower == "wait until done" {
        return Ok(Command::WaitForCompletion { timeout_secs: None });
    }

    // Wait N seconds
    if lower.starts_with("wait ") {
        let rest = &lower[5..];
        if let Some(secs) = parse_duration_seconds(rest) {
            return Ok(Command::Wait { seconds: secs });
        }
    }

    // Screenshot
    if lower.starts_with("screenshot ") || lower.starts_with("take screenshot ") {
        let filename = if lower.starts_with("take screenshot ") {
            input[16..].trim()
        } else {
            input[11..].trim()
        }.trim_matches('"').to_string();
        return Ok(Command::Screenshot { filename });
    }

    // Capture fixtures
    if lower.starts_with("capture fixtures") || lower.starts_with("save fixtures") {
        let filename = input.split_whitespace().last()
            .map(|s| s.trim_matches('"').to_string())
            .unwrap_or_else(|| "fixtures.json".to_string());
        return Ok(Command::CaptureFixtures { filename });
    }

    // Approve all permissions (with optional pattern)
    if lower.starts_with("approve all") {
        let pattern = if lower.contains("permission") {
            let rest = lower.replace("approve all", "").replace("permissions", "").replace("permission", "").trim().to_string();
            if rest.is_empty() { None } else { Some(rest) }
        } else {
            let rest = lower[11..].trim().to_string();
            if rest.is_empty() { None } else { Some(rest) }
        };
        return Ok(Command::ApproveAllPermissions { pattern });
    }

    // Deny all permissions
    if lower.starts_with("deny all") {
        let pattern = if lower.contains("permission") {
            let rest = lower.replace("deny all", "").replace("permissions", "").replace("permission", "").trim().to_string();
            if rest.is_empty() { None } else { Some(rest) }
        } else {
            let rest = lower[8..].trim().to_string();
            if rest.is_empty() { None } else { Some(rest) }
        };
        return Ok(Command::DenyAllPermissions { pattern });
    }

    // Approve specific permission
    if lower.starts_with("approve ") && (lower.contains("permission") || lower.contains("cargo") || lower.contains("read") || lower.contains("write")) {
        let pattern = input[8..].trim().replace(" permissions", "").replace(" permission", "");
        return Ok(Command::ApprovePermission { pattern });
    }

    // Deny specific permission
    if lower.starts_with("deny ") && (lower.contains("permission") || lower.contains("network") || lower.contains("delete")) {
        let pattern = input[5..].trim().replace(" permissions", "").replace(" permission", "");
        return Ok(Command::DenyPermission { pattern });
    }

    // Use best judgment
    if lower.contains("best judgment") || lower.contains("best judgement") || lower == "ubj" {
        return Ok(Command::UseBestJudgment);
    }

    // Edit mode
    if lower == "enter edit mode" || lower == "edit mode" || lower == "press f2" {
        return Ok(Command::EnterEditMode);
    }
    if lower == "exit edit mode" || lower == "leave edit mode" {
        return Ok(Command::ExitEditMode);
    }

    // Log
    if lower.starts_with("log ") || lower.starts_with("print ") || lower.starts_with("echo ") {
        let message = input.splitn(2, ' ').nth(1).unwrap_or("").trim_matches('"').to_string();
        return Ok(Command::Log { message });
    }

    // Assert state
    if lower.starts_with("assert state ") || lower.starts_with("expect state ") {
        let state = input.splitn(3, ' ').nth(2).unwrap_or("").trim_matches('"').to_string();
        return Ok(Command::AssertState { state });
    }

    // Fail
    if lower.starts_with("fail ") || lower.starts_with("error ") {
        let message = input.splitn(2, ' ').nth(1).unwrap_or("").trim_matches('"').to_string();
        return Ok(Command::Fail { message });
    }

    bail!("Unknown command: {}", input)
}

/// Parse a comma/space separated list of numbers
fn parse_number_list(input: &str) -> Result<Vec<usize>> {
    let mut result = Vec::new();

    for part in input.split(|c| c == ',' || c == ' ') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        // Handle ranges like "1-5"
        if part.contains('-') && !part.starts_with('-') {
            let mut parts = part.split('-');
            let start: usize = parts.next().unwrap().trim().parse()?;
            let end: usize = parts.next().unwrap().trim().parse()?;
            for i in start..=end {
                result.push(i);
            }
        } else {
            result.push(part.parse()?);
        }
    }

    Ok(result)
}

/// Parse a duration string to seconds
fn parse_duration_seconds(input: &str) -> Option<f32> {
    let input = input.trim().to_lowercase();

    // Check longer suffixes first to avoid partial matches
    // e.g., "ms" before "s", "minutes" before "min" before "m"
    if let Some(s) = input.strip_suffix("ms") {
        return s.trim().parse::<f32>().ok().map(|ms| ms / 1000.0);
    }
    if let Some(s) = input.strip_suffix("seconds") {
        return s.trim().parse().ok();
    }
    if let Some(s) = input.strip_suffix("second") {
        return s.trim().parse().ok();
    }
    if let Some(s) = input.strip_suffix("secs") {
        return s.trim().parse().ok();
    }
    if let Some(s) = input.strip_suffix("sec") {
        return s.trim().parse().ok();
    }
    if let Some(s) = input.strip_suffix('s') {
        return s.trim().parse().ok();
    }
    if let Some(s) = input.strip_suffix("minutes") {
        return s.trim().parse::<f32>().ok().map(|m| m * 60.0);
    }
    if let Some(s) = input.strip_suffix("minute") {
        return s.trim().parse::<f32>().ok().map(|m| m * 60.0);
    }
    if let Some(s) = input.strip_suffix("min") {
        return s.trim().parse::<f32>().ok().map(|m| m * 60.0);
    }
    if let Some(s) = input.strip_suffix('m') {
        return s.trim().parse::<f32>().ok().map(|m| m * 60.0);
    }

    // Try parsing as plain number (assume seconds)
    input.parse().ok()
}

/// Parse all steps from a scenario
pub fn parse_steps(steps: &[ScriptStep]) -> Result<Vec<Command>> {
    let mut commands = Vec::new();

    for step in steps {
        match step {
            ScriptStep::Simple(s) => {
                commands.push(parse_command(s)?);
            }
            ScriptStep::Structured(s) => {
                commands.push(parse_structured_step(s)?);
            }
        }
    }

    Ok(commands)
}

/// Parse a structured step
fn parse_structured_step(step: &StructuredStep) -> Result<Command> {
    let action = step.action.to_lowercase();

    match action.as_str() {
        "click" => {
            let target = step.target.clone().unwrap_or_default();
            Ok(Command::Click { target })
        }
        "wait" => {
            if let Some(ref target) = step.target {
                if target == "cards" {
                    return Ok(Command::WaitForCards { min_count: None, timeout_secs: None });
                }
                if target == "completion" {
                    return Ok(Command::WaitForCompletion { timeout_secs: None });
                }
                return Ok(Command::WaitForState { state: target.clone(), timeout_secs: None });
            }
            if let Some(ref val) = step.value {
                if let Some(secs) = val.as_f64() {
                    return Ok(Command::Wait { seconds: secs as f32 });
                }
            }
            Ok(Command::Wait { seconds: 1.0 })
        }
        "screenshot" => {
            let filename = step.target.clone().unwrap_or_else(|| "screenshot.png".to_string());
            Ok(Command::Screenshot { filename })
        }
        "select" => {
            if let Some(ref target) = step.target {
                if target == "all" || target == "all cards" {
                    return Ok(Command::SelectAllCards);
                }
            }
            if let Some(ref val) = step.value {
                if let Some(arr) = val.as_sequence() {
                    let indices: Vec<usize> = arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect();
                    return Ok(Command::SelectCards { indices });
                }
            }
            Ok(Command::SelectAllCards)
        }
        "execute" | "run" => Ok(Command::Execute),
        "analyze" | "analysis" => Ok(Command::StartAnalysis),
        _ => {
            // Try parsing as plain text
            parse_command(&step.action)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_open_project() {
        let cmd = parse_command("open project ~/projects/test").unwrap();
        assert!(matches!(cmd, Command::OpenProject { path } if path == "~/projects/test"));
    }

    #[test]
    fn test_parse_click() {
        let cmd = parse_command("click \"Suggest Tasks\"").unwrap();
        assert!(matches!(cmd, Command::Click { target } if target == "Suggest Tasks"));
    }

    #[test]
    fn test_parse_select_all() {
        let cmd = parse_command("select all cards").unwrap();
        assert!(matches!(cmd, Command::SelectAllCards));
    }

    #[test]
    fn test_parse_select_specific() {
        let cmd = parse_command("select cards 1, 3, 5").unwrap();
        if let Command::SelectCards { indices } = cmd {
            assert_eq!(indices, vec![1, 3, 5]);
        } else {
            panic!("Wrong command type");
        }
    }

    #[test]
    fn test_parse_select_range() {
        let cmd = parse_command("select cards 1-5").unwrap();
        if let Command::SelectCards { indices } = cmd {
            assert_eq!(indices, vec![1, 2, 3, 4, 5]);
        } else {
            panic!("Wrong command type");
        }
    }

    #[test]
    fn test_parse_wait_seconds() {
        let cmd = parse_command("wait 5 seconds").unwrap();
        assert!(matches!(cmd, Command::Wait { seconds } if (seconds - 5.0).abs() < 0.01));

        let cmd = parse_command("wait 500ms").unwrap();
        assert!(matches!(cmd, Command::Wait { seconds } if (seconds - 0.5).abs() < 0.01));
    }

    #[test]
    fn test_parse_screenshot() {
        let cmd = parse_command("screenshot \"result.png\"").unwrap();
        assert!(matches!(cmd, Command::Screenshot { filename } if filename == "result.png"));
    }

    #[test]
    fn test_parse_approve_permissions() {
        let cmd = parse_command("approve all cargo permissions").unwrap();
        assert!(matches!(cmd, Command::ApproveAllPermissions { pattern: Some(p) } if p.contains("cargo")));

        let cmd = parse_command("approve all permissions").unwrap();
        assert!(matches!(cmd, Command::ApproveAllPermissions { pattern: None }));
    }

    #[test]
    fn test_parse_ubj() {
        let cmd = parse_command("use best judgment on surveys").unwrap();
        assert!(matches!(cmd, Command::UseBestJudgment));

        let cmd = parse_command("ubj").unwrap();
        assert!(matches!(cmd, Command::UseBestJudgment));
    }

    #[test]
    fn test_parse_execute() {
        let cmd = parse_command("execute").unwrap();
        assert!(matches!(cmd, Command::Execute));

        let cmd = parse_command("start execution").unwrap();
        assert!(matches!(cmd, Command::Execute));
    }

    #[test]
    fn test_number_list_parsing() {
        assert_eq!(parse_number_list("1, 2, 3").unwrap(), vec![1, 2, 3]);
        assert_eq!(parse_number_list("1 2 3").unwrap(), vec![1, 2, 3]);
        assert_eq!(parse_number_list("1-3").unwrap(), vec![1, 2, 3]);
        assert_eq!(parse_number_list("1, 3-5, 7").unwrap(), vec![1, 3, 4, 5, 7]);
    }
}
