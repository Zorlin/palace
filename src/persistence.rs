//! Persistence layer using ReDB for tasks, permissions, and user preferences

use anyhow::{Context, Result};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::path::PathBuf;

// Table definitions
const TASKS: TableDefinition<u64, &[u8]> = TableDefinition::new("tasks");
// Key format: "project_path:command_prefix" -> bool
const PROJECT_PERMISSIONS: TableDefinition<&str, bool> = TableDefinition::new("project_permissions");
const USER_PREFS: TableDefinition<&str, &[u8]> = TableDefinition::new("user_prefs");

/// Task status for tracking execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Complete,
    Failed,
}

impl Default for TaskStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// A persistent task that can be saved/loaded from the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: u64,
    pub title: String,
    pub category: String,
    pub description: String,
    pub command: Option<String>,
    pub status: TaskStatus,
    pub progress: f32,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub project_path: String,
    pub archived: bool,
}

impl Task {
    pub fn new(id: u64, title: String, project_path: String) -> Self {
        Self {
            id,
            title,
            category: String::new(),
            description: String::new(),
            command: None,
            status: TaskStatus::Pending,
            progress: 0.0,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            completed_at: None,
            project_path,
            archived: false,
        }
    }

    /// Create a task from a suggestion card
    pub fn from_suggestion(
        id: u64,
        title: &str,
        category: &str,
        description: &str,
        command: Option<&str>,
        project_path: &str,
    ) -> Self {
        Self {
            id,
            title: title.to_string(),
            category: category.to_string(),
            description: description.to_string(),
            command: command.map(|s| s.to_string()),
            status: TaskStatus::Pending,
            progress: 0.0,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            completed_at: None,
            project_path: project_path.to_string(),
            archived: false,
        }
    }
}

/// Palace database wrapper for ReDB
pub struct PalaceDB {
    db: Database,
}

impl PalaceDB {
    /// Open or create the Palace database
    pub fn open() -> Result<Self> {
        let path = Self::db_path()?;

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create database directory: {:?}", parent))?;
        }

        let db = Database::create(&path)
            .with_context(|| format!("Failed to create/open database at {:?}", path))?;

        Ok(Self { db })
    }

    /// Get the database file path
    fn db_path() -> Result<PathBuf> {
        let data_dir = dirs::data_dir()
            .or_else(|| dirs::home_dir().map(|h| h.join(".local/share")))
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(data_dir.join("palace").join("palace.redb"))
    }

    // ========== Task Operations ==========

    /// Save a task (insert or update)
    pub fn save_task(&self, task: &Task) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(TASKS)?;
            let bytes = serde_json::to_vec(task)?;
            table.insert(task.id, bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Load a task by ID
    pub fn load_task(&self, id: u64) -> Result<Option<Task>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(TASKS)?;
        if let Some(guard) = table.get(id)? {
            let bytes: &[u8] = guard.value();
            let task: Task = serde_json::from_slice(bytes)?;
            Ok(Some(task))
        } else {
            Ok(None)
        }
    }

    /// Load all current (non-archived) tasks for a project
    pub fn load_current_tasks(&self, project_path: &str) -> Result<Vec<Task>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(TASKS)?;
        let mut tasks = Vec::new();

        for result in table.iter()? {
            let (_, guard) = result?;
            let bytes: &[u8] = guard.value();
            let task: Task = serde_json::from_slice(bytes)?;
            if !task.archived && task.project_path == project_path {
                tasks.push(task);
            }
        }

        // Sort by creation time (newest first)
        tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(tasks)
    }

    /// Load all archived tasks
    pub fn load_archived_tasks(&self) -> Result<Vec<Task>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(TASKS)?;
        let mut tasks = Vec::new();

        for result in table.iter()? {
            let (_, guard) = result?;
            let bytes: &[u8] = guard.value();
            let task: Task = serde_json::from_slice(bytes)?;
            if task.archived {
                tasks.push(task);
            }
        }

        // Sort by creation time (newest first)
        tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(tasks)
    }

    /// Archive a task
    pub fn archive_task(&self, id: u64) -> Result<()> {
        if let Some(mut task) = self.load_task(id)? {
            task.archived = true;
            self.save_task(&task)?;
        }
        Ok(())
    }

    /// Unarchive a task (restore from archive)
    pub fn unarchive_task(&self, id: u64) -> Result<()> {
        if let Some(mut task) = self.load_task(id)? {
            task.archived = false;
            self.save_task(&task)?;
        }
        Ok(())
    }

    /// Delete a task permanently
    pub fn delete_task(&self, id: u64) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(TASKS)?;
            table.remove(id)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Generate the next task ID
    pub fn next_task_id(&self) -> Result<u64> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(TASKS)?;
        let mut max_id = 0u64;
        for result in table.iter()? {
            let (key_guard, _) = result?;
            let key: u64 = key_guard.value();
            if key > max_id {
                max_id = key;
            }
        }
        Ok(max_id + 1)
    }

    // ========== Permission Operations (per-project) ==========

    /// Build permission key from project path and command prefix
    fn permission_key(project_path: &str, prefix: &str) -> String {
        format!("{}:{}", project_path, prefix)
    }

    /// Check if a command prefix is approved for a specific project
    pub fn is_prefix_approved(&self, project_path: &str, prefix: &str) -> Result<bool> {
        let key = Self::permission_key(project_path, prefix);
        let read_txn = self.db.begin_read()?;
        match read_txn.open_table(PROJECT_PERMISSIONS) {
            Ok(table) => {
                if let Some(value) = table.get(key.as_str())? {
                    Ok(value.value())
                } else {
                    Ok(false)
                }
            }
            Err(_) => Ok(false), // Table doesn't exist yet
        }
    }

    /// Check if a command matches any approved prefix for a project
    pub fn is_command_approved(&self, project_path: &str, command: &str) -> Result<bool> {
        let prefixes = self.get_approved_prefixes(project_path)?;
        for prefix in prefixes {
            if command.starts_with(&prefix) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Approve a command prefix for a specific project
    pub fn approve_prefix(&self, project_path: &str, prefix: &str) -> Result<()> {
        let key = Self::permission_key(project_path, prefix);
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(PROJECT_PERMISSIONS)?;
            table.insert(key.as_str(), true)?;
        }
        write_txn.commit()?;
        tracing::info!("Approved '{}' for project {}", prefix, project_path);
        Ok(())
    }

    /// Revoke approval for a command prefix in a project
    pub fn revoke_prefix(&self, project_path: &str, prefix: &str) -> Result<()> {
        let key = Self::permission_key(project_path, prefix);
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(PROJECT_PERMISSIONS)?;
            table.remove(key.as_str())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Get all approved command prefixes for a project
    pub fn get_approved_prefixes(&self, project_path: &str) -> Result<Vec<String>> {
        let prefix_start = format!("{}:", project_path);
        let read_txn = self.db.begin_read()?;
        match read_txn.open_table(PROJECT_PERMISSIONS) {
            Ok(table) => {
                let mut prefixes = Vec::new();
                for result in table.iter()? {
                    let (key_guard, value_guard) = result?;
                    let approved: bool = value_guard.value();
                    if approved {
                        let key: &str = key_guard.value();
                        // Extract prefix from "project_path:prefix"
                        if key.starts_with(&prefix_start) {
                            let cmd_prefix = &key[prefix_start.len()..];
                            prefixes.push(cmd_prefix.to_string());
                        }
                    }
                }
                Ok(prefixes)
            }
            Err(_) => Ok(Vec::new()), // Table doesn't exist yet
        }
    }

    /// Get all approved permissions across all projects (for debugging/admin)
    pub fn get_all_permissions(&self) -> Result<Vec<(String, String)>> {
        let read_txn = self.db.begin_read()?;
        match read_txn.open_table(PROJECT_PERMISSIONS) {
            Ok(table) => {
                let mut perms = Vec::new();
                for result in table.iter()? {
                    let (key_guard, value_guard) = result?;
                    let approved: bool = value_guard.value();
                    if approved {
                        let key: &str = key_guard.value();
                        if let Some(idx) = key.find(':') {
                            let project = &key[..idx];
                            let prefix = &key[idx + 1..];
                            perms.push((project.to_string(), prefix.to_string()));
                        }
                    }
                }
                Ok(perms)
            }
            Err(_) => Ok(Vec::new()),
        }
    }

    // ========== User Preferences ==========

    /// Get a user preference by key
    pub fn get_pref<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let read_txn = self.db.begin_read()?;
        match read_txn.open_table(USER_PREFS) {
            Ok(table) => {
                if let Some(guard) = table.get(key)? {
                    let bytes: &[u8] = guard.value();
                    let pref: T = serde_json::from_slice(bytes)?;
                    Ok(Some(pref))
                } else {
                    Ok(None)
                }
            }
            Err(_) => Ok(None), // Table doesn't exist yet
        }
    }

    /// Set a user preference
    pub fn set_pref<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(USER_PREFS)?;
            let bytes = serde_json::to_vec(value)?;
            table.insert(key, bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Delete a user preference
    pub fn delete_pref(&self, key: &str) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(USER_PREFS)?;
            table.remove(key)?;
        }
        write_txn.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_db() -> PalaceDB {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.redb");
        let db = Database::create(path).unwrap();
        PalaceDB { db }
    }

    #[test]
    fn test_task_crud() {
        let db = test_db();

        // Create and save a task
        let task = Task::new(1, "Test task".to_string(), "/test/project".to_string());
        db.save_task(&task).unwrap();

        // Load it back
        let loaded = db.load_task(1).unwrap().unwrap();
        assert_eq!(loaded.title, "Test task");
        assert_eq!(loaded.project_path, "/test/project");
        assert!(!loaded.archived);

        // Archive it
        db.archive_task(1).unwrap();
        let archived = db.load_task(1).unwrap().unwrap();
        assert!(archived.archived);

        // Delete it
        db.delete_task(1).unwrap();
        assert!(db.load_task(1).unwrap().is_none());
    }

    #[test]
    fn test_permissions() {
        let db = test_db();
        let project = "/test/project";

        // Initially not approved
        assert!(!db.is_prefix_approved(project, "cargo build").unwrap());

        // Approve it
        db.approve_prefix(project, "cargo build").unwrap();
        assert!(db.is_prefix_approved(project, "cargo build").unwrap());

        // Check command matching
        assert!(db.is_command_approved(project, "cargo build --release").unwrap());
        assert!(!db.is_command_approved(project, "rm -rf /").unwrap());

        // Check list
        let prefixes = db.get_approved_prefixes(project).unwrap();
        assert!(prefixes.contains(&"cargo build".to_string()));

        // Different project should NOT have this permission
        assert!(!db.is_prefix_approved("/other/project", "cargo build").unwrap());

        // Revoke it
        db.revoke_prefix(project, "cargo build").unwrap();
        assert!(!db.is_prefix_approved(project, "cargo build").unwrap());
    }

    #[test]
    fn test_user_prefs() {
        let db = test_db();

        // Set a preference
        db.set_pref("dark_mode", &true).unwrap();
        db.set_pref("ui_scale", &1.5f32).unwrap();

        // Get them back
        let dark_mode: bool = db.get_pref("dark_mode").unwrap().unwrap();
        assert!(dark_mode);

        let ui_scale: f32 = db.get_pref("ui_scale").unwrap().unwrap();
        assert!((ui_scale - 1.5).abs() < 0.001);

        // Delete one
        db.delete_pref("dark_mode").unwrap();
        assert!(db.get_pref::<bool>("dark_mode").unwrap().is_none());
    }
}
