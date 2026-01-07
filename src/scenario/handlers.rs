//! Handlers for automated permission and survey responses

use super::schema::{PermissionPolicy, PermissionType, SurveyStrategy};
use super::supervisor::RiskLevel;

/// Handler for permission requests
pub struct PermissionHandler {
    policy: PermissionPolicy,
    approval_count: u32,
}

impl PermissionHandler {
    pub fn new(policy: PermissionPolicy) -> Self {
        Self {
            policy,
            approval_count: 0,
        }
    }

    /// Classify a command's risk level
    pub fn classify_risk(command: &str) -> RiskLevel {
        let cmd_lower = command.to_lowercase();

        // Critical: destructive or system-level
        if cmd_lower.contains("rm -rf")
            || cmd_lower.contains("rm -r /")
            || cmd_lower.contains("sudo")
            || cmd_lower.contains("chmod 777")
            || cmd_lower.contains("mkfs")
            || cmd_lower.contains("> /dev/")
        {
            return RiskLevel::Critical;
        }

        // High: network, deletion, system modification
        if cmd_lower.contains("curl")
            || cmd_lower.contains("wget")
            || cmd_lower.contains("rm -r")
            || cmd_lower.contains("delete")
            || cmd_lower.starts_with("rm ")
            || cmd_lower.contains("pip install")
            || cmd_lower.contains("npm install -g")
        {
            return RiskLevel::High;
        }

        // Medium: writing, building, local installs
        if cmd_lower.contains("write")
            || cmd_lower.contains("edit")
            || cmd_lower.contains("create")
            || cmd_lower.starts_with("cargo ")
            || cmd_lower.starts_with("npm ")
            || cmd_lower.starts_with("go ")
            || cmd_lower.starts_with("git ")
        {
            return RiskLevel::Medium;
        }

        // Low: reading, listing, checking
        RiskLevel::Low
    }

    /// Check if a permission should be auto-approved
    pub fn should_approve(&mut self, command: &str) -> Option<bool> {
        let _risk = Self::classify_risk(command);

        // Check deny list first (always takes precedence)
        for denied in &self.policy.deny {
            if self.matches(command, denied) {
                return Some(false);
            }
        }

        // Check allow list
        for allowed in &self.policy.allow {
            if self.matches(command, allowed) {
                // Check approval limit
                if let Some(max) = self.policy.max_auto_approvals {
                    if self.approval_count >= max {
                        return Some(false);
                    }
                }
                self.approval_count += 1;
                return Some(true);
            }
        }

        // Not explicitly allowed or denied
        if self.policy.ask_unknown {
            // Return None to indicate supervisor should decide
            None
        } else {
            // Default deny for unknown
            Some(false)
        }
    }

    fn matches(&self, command: &str, perm: &PermissionType) -> bool {
        match perm {
            PermissionType::Read => {
                command.to_lowercase().contains("read")
                    || command.starts_with("cat ")
                    || command.starts_with("head ")
                    || command.starts_with("tail ")
                    || command.starts_with("ls ")
            }
            PermissionType::Write => {
                command.to_lowercase().contains("write")
                    || command.to_lowercase().contains("edit")
            }
            PermissionType::Create => {
                command.to_lowercase().contains("create")
                    || command.starts_with("touch ")
                    || command.starts_with("mkdir ")
            }
            PermissionType::Delete => {
                command.to_lowercase().contains("delete")
                    || command.starts_with("rm ")
            }
            PermissionType::Cargo => command.starts_with("cargo "),
            PermissionType::Npm => {
                command.starts_with("npm ")
                    || command.starts_with("yarn ")
                    || command.starts_with("pnpm ")
            }
            PermissionType::Git => command.starts_with("git "),
            PermissionType::Network => {
                command.contains("curl")
                    || command.contains("wget")
                    || command.contains("fetch")
                    || command.contains("http://")
                    || command.contains("https://")
            }
            PermissionType::RmRf => {
                command.contains("rm -rf") || command.contains("rm -r ")
            }
            PermissionType::Bash => {
                command.starts_with("bash ")
                    || command.starts_with("sh ")
                    || command.contains("sh -c")
            }
            PermissionType::BashSafe => {
                let is_bash = command.starts_with("bash ")
                    || command.starts_with("sh ")
                    || command.contains("sh -c");
                let is_dangerous = command.contains("rm ")
                    || command.contains("curl")
                    || command.contains("wget")
                    || command.contains("sudo");
                is_bash && !is_dangerous
            }
            PermissionType::Pattern(pattern) => {
                glob_match(pattern, command)
            }
        }
    }
}

/// Simple glob pattern matching
fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if pattern.starts_with('*') && pattern.ends_with('*') {
        let inner = &pattern[1..pattern.len() - 1];
        return text.contains(inner);
    }

    if pattern.starts_with('*') {
        let suffix = &pattern[1..];
        return text.ends_with(suffix);
    }

    if pattern.ends_with('*') {
        let prefix = &pattern[..pattern.len() - 1];
        return text.starts_with(prefix);
    }

    pattern == text
}

/// Handler for survey responses
pub struct SurveyHandler {
    strategy: SurveyStrategy,
}

impl SurveyHandler {
    pub fn new(strategy: SurveyStrategy) -> Self {
        Self { strategy }
    }

    /// Get automatic response for a survey, if applicable
    ///
    /// Returns Some(indices) for automatic response, None if supervisor should decide
    pub fn auto_respond(
        &self,
        question: &str,
        options: &[String],
        _multi_select: bool,
    ) -> Option<Vec<usize>> {
        match &self.strategy {
            SurveyStrategy::First => Some(vec![0]),
            SurveyStrategy::Last => Some(vec![options.len().saturating_sub(1)]),
            SurveyStrategy::Skip => None, // Can't auto-skip, supervisor decides
            SurveyStrategy::Ask => None,   // Explicitly requires human
            SurveyStrategy::Ubj => None,   // Supervisor decides
            SurveyStrategy::Custom(rules) => {
                for rule in rules {
                    if question.to_lowercase().contains(&rule.pattern.to_lowercase()) {
                        match &rule.response {
                            super::schema::SurveyResponse::Index(i) => {
                                return Some(vec![*i]);
                            }
                            super::schema::SurveyResponse::Contains(text) => {
                                for (i, opt) in options.iter().enumerate() {
                                    if opt.to_lowercase().contains(&text.to_lowercase()) {
                                        return Some(vec![i]);
                                    }
                                }
                            }
                            super::schema::SurveyResponse::Ubj => return None,
                            super::schema::SurveyResponse::Ask => return None,
                        }
                    }
                }
                None // No rule matched
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_classification() {
        assert_eq!(PermissionHandler::classify_risk("cat file.txt"), RiskLevel::Low);
        assert_eq!(PermissionHandler::classify_risk("cargo build"), RiskLevel::Medium);
        assert_eq!(PermissionHandler::classify_risk("curl https://example.com"), RiskLevel::High);
        assert_eq!(PermissionHandler::classify_risk("rm -rf /"), RiskLevel::Critical);
    }

    #[test]
    fn test_glob_match() {
        assert!(glob_match("cargo *", "cargo build"));
        assert!(glob_match("cargo *", "cargo test --release"));
        assert!(!glob_match("cargo *", "npm install"));

        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.rs", "main.go"));

        assert!(glob_match("*test*", "run_test_suite"));
        assert!(glob_match("*test*", "testing"));
    }

    #[test]
    fn test_permission_handler() {
        let policy = PermissionPolicy {
            allow: vec![PermissionType::Read, PermissionType::Cargo],
            deny: vec![PermissionType::RmRf],
            ask_unknown: true,
            max_auto_approvals: Some(10),
        };

        let mut handler = PermissionHandler::new(policy);

        // Should approve read
        assert_eq!(handler.should_approve("Read file.txt"), Some(true));

        // Should approve cargo
        assert_eq!(handler.should_approve("cargo build"), Some(true));

        // Should deny rm -rf even though bash might be allowed
        assert_eq!(handler.should_approve("rm -rf /tmp/test"), Some(false));

        // Unknown should return None (ask supervisor)
        assert_eq!(handler.should_approve("unknown command"), None);
    }

    #[test]
    fn test_survey_handler_first() {
        let handler = SurveyHandler::new(SurveyStrategy::First);
        let options = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        assert_eq!(
            handler.auto_respond("Pick one", &options, false),
            Some(vec![0])
        );
    }

    #[test]
    fn test_survey_handler_last() {
        let handler = SurveyHandler::new(SurveyStrategy::Last);
        let options = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        assert_eq!(
            handler.auto_respond("Pick one", &options, false),
            Some(vec![2])
        );
    }

    #[test]
    fn test_survey_handler_ubj() {
        let handler = SurveyHandler::new(SurveyStrategy::Ubj);
        let options = vec!["A".to_string(), "B".to_string()];

        // UBJ should return None to let supervisor decide
        assert_eq!(handler.auto_respond("Pick one", &options, false), None);
    }
}
