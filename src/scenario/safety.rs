//! Safety gate for scenario execution
//!
//! On first run, users must acknowledge the risks of running AI-driven scenarios.
//! After acknowledgment, the gate is passed and never nags again.

use anyhow::{bail, Result};
use std::io::{self, Write};
use std::path::PathBuf;

const SAFETY_WARNING: &str = r#"
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ⚠️  SCENARIO SAFETY WARNING ⚠️                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Scenarios run AI agents with REAL SYSTEM ACCESS.                            ║
║                                                                              ║
║  • The AI supervisor will read, write, and execute code                      ║
║  • It will make autonomous decisions based on the scenario goals             ║
║  • Prompt injection risks exist in untrusted scenarios                       ║
║  • The AI may exhibit emergent behavior not explicitly programmed            ║
║                                                                              ║
║  RECOMMENDATIONS:                                                            ║
║  • Only run scenarios you have personally reviewed and trust                 ║
║  • Run in a sandbox/VM when possible (Docker, Firecracker, etc.)             ║
║  • Start with restrictive permission policies                                ║
║  • Monitor execution, especially on first runs                               ║
║                                                                              ║
║  This warning will only appear once.                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

"#;

/// Safety gate that must be passed before running scenarios
pub struct SafetyGate;

impl SafetyGate {
    /// Check if the safety gate has been passed
    ///
    /// If not, shows the warning and asks for acknowledgment.
    /// After acknowledgment, creates a marker file so we don't ask again.
    pub fn check() -> Result<()> {
        if Self::is_acknowledged() {
            return Ok(());
        }

        Self::show_warning_and_prompt()
    }

    /// Check if the user has previously acknowledged the warning
    fn is_acknowledged() -> bool {
        Self::marker_path().exists()
    }

    /// Get the path to the acknowledgment marker file
    fn marker_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("palace")
            .join(".scenario-safety-acknowledged")
    }

    /// Show the warning and prompt for acknowledgment
    fn show_warning_and_prompt() -> Result<()> {
        // Print the warning
        eprint!("{}", SAFETY_WARNING);

        // Prompt for acknowledgment
        eprint!("Type 'I understand' to continue: ");
        io::stderr().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let input = input.trim();

        if input.eq_ignore_ascii_case("I understand") || input.eq_ignore_ascii_case("i understand") {
            // Create marker file
            Self::create_marker()?;
            eprintln!("\n✓ Safety acknowledgment recorded. This warning will not appear again.\n");
            Ok(())
        } else {
            bail!("Safety acknowledgment required. Scenario execution cancelled.");
        }
    }

    /// Create the marker file to record acknowledgment
    fn create_marker() -> Result<()> {
        let path = Self::marker_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, format!(
            "Safety warning acknowledged at: {}\n\
             User typed 'I understand' to acknowledge risks of scenario execution.\n",
            chrono_lite::Utc::now(),
        ).as_bytes())?;
        Ok(())
    }

    /// Reset the safety gate (for testing or if user wants to see warning again)
    #[allow(dead_code)]
    pub fn reset() -> Result<()> {
        let path = Self::marker_path();
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    /// Check gate in non-interactive mode (fails if not already acknowledged)
    pub fn check_non_interactive() -> Result<()> {
        if Self::is_acknowledged() {
            Ok(())
        } else {
            bail!(
                "Scenario safety warning not yet acknowledged.\n\
                 Run `palace scenario --acknowledge` interactively first,\n\
                 or run any scenario interactively to see the warning."
            );
        }
    }
}

/// Simple timestamp helper (avoids chrono dependency)
mod chrono_lite {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub struct Utc;

    impl Utc {
        pub fn now() -> String {
            let duration = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default();
            let secs = duration.as_secs();

            // Simple ISO-8601 approximation
            let days = secs / 86400;
            let years = 1970 + days / 365;
            let remaining_days = days % 365;
            let months = remaining_days / 30 + 1;
            let day = remaining_days % 30 + 1;

            let day_secs = secs % 86400;
            let hours = day_secs / 3600;
            let minutes = (day_secs % 3600) / 60;
            let seconds = day_secs % 60;

            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                years, months, day, hours, minutes, seconds
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_path_is_reasonable() {
        let path = SafetyGate::marker_path();
        let path_str = path.to_string_lossy();
        assert!(path_str.contains("palace") || path_str.contains(".scenario"));
    }
}
