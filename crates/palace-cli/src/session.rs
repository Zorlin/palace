//! Session management for Palace CLI
//!
//! Saves and restores RHSI session state for resumable workflows.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Session state for RHSI loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session ID
    pub id: String,

    /// When the session was created
    pub created_at: DateTime<Utc>,

    /// When the session was last updated
    pub updated_at: DateTime<Utc>,

    /// Current iteration number
    pub iteration: u32,

    /// Pending actions from Claude
    pub pending_actions: Vec<PendingAction>,

    /// Project root path
    pub project_root: PathBuf,

    /// Last prompt sent
    pub last_prompt: Option<String>,

    /// Session metadata
    pub metadata: SessionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingAction {
    pub label: String,
    pub description: Option<String>,
    pub modifiers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionMetadata {
    /// Model used
    pub model: Option<String>,

    /// Whether turbo mode was used
    pub turbo_mode: bool,

    /// Total cost so far (if tracking enabled)
    pub total_cost: Option<f64>,

    /// Number of completed actions
    pub completed_actions: u32,
}

impl Session {
    /// Create a new session
    pub fn new(project_root: PathBuf) -> Self {
        let id = generate_session_id();
        let now = Utc::now();

        Self {
            id,
            created_at: now,
            updated_at: now,
            iteration: 0,
            pending_actions: Vec::new(),
            project_root,
            last_prompt: None,
            metadata: SessionMetadata::default(),
        }
    }

    /// Get the session storage directory
    fn sessions_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Could not find home directory")?;
        let dir = home.join(".palace").join("sessions");
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    /// Save the session to disk
    pub fn save(&self) -> Result<()> {
        let dir = Self::sessions_dir()?;
        let path = dir.join(format!("{}.json", self.id));
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load a session by ID
    pub fn load(id: &str) -> Result<Option<Self>> {
        let dir = Self::sessions_dir()?;
        let path = dir.join(format!("{}.json", id));

        if !path.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(path)?;
        let session: Self = serde_json::from_str(&content)?;
        Ok(Some(session))
    }

    /// List all sessions
    pub fn list_all() -> Result<Vec<SessionSummary>> {
        let dir = Self::sessions_dir()?;
        let mut summaries = Vec::new();

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map_or(false, |e| e == "json") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(session) = serde_json::from_str::<Session>(&content) {
                        summaries.push(SessionSummary {
                            id: session.id,
                            created_at: session.created_at,
                            updated_at: session.updated_at,
                            iteration: session.iteration,
                            pending_count: session.pending_actions.len(),
                            project_name: session
                                .project_root
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| "unknown".to_string()),
                        });
                    }
                }
            }
        }

        // Sort by most recent first
        summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

        Ok(summaries)
    }

    /// Delete a session
    pub fn delete(id: &str) -> Result<bool> {
        let dir = Self::sessions_dir()?;
        let path = dir.join(format!("{}.json", id));

        if path.exists() {
            std::fs::remove_file(path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Increment iteration and update timestamp
    pub fn next_iteration(&mut self) {
        self.iteration += 1;
        self.updated_at = Utc::now();
    }
}

/// Summary of a session for listing
#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub iteration: u32,
    pub pending_count: usize,
    pub project_name: String,
}

/// Generate a unique session ID
fn generate_session_id() -> String {
    use chrono::Datelike;
    use chrono::Timelike;

    let now = Utc::now();
    let random: u32 = rand_simple();

    format!(
        "{:04}{:02}{:02}-{:02}{:02}-{:04x}",
        now.year(),
        now.month(),
        now.day(),
        now.hour(),
        now.minute(),
        random % 0xFFFF
    )
}

/// Simple random number without extra dependencies
fn rand_simple() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    (duration.subsec_nanos() ^ (duration.as_secs() as u32)) % 0xFFFFFFFF
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use tempfile::tempdir;

    #[test]
    fn test_session_new() {
        let session = Session::new(PathBuf::from("/test/project"));
        assert!(!session.id.is_empty());
        assert_eq!(session.iteration, 0);
        assert!(session.pending_actions.is_empty());
    }

    #[test]
    fn test_session_id_format() {
        let id = generate_session_id();
        // Should be like "20251211-1530-a1b2"
        assert!(id.len() > 10);
        assert!(id.contains('-'));
    }

    #[test]
    fn test_rand_simple() {
        let a = rand_simple();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let b = rand_simple();
        // Not guaranteed to be different, but usually will be
        assert!(a != 0 || b != 0);
    }
}
