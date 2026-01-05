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

    // Helper to initialize tables in test DB
    fn init_test_db(db: &PalaceDB) {
        let write_txn = db.db.begin_write().unwrap();
        {
            // Create all tables by opening them
            let _ = write_txn.open_table(TASKS);
            let _ = write_txn.open_table(PROJECT_PERMISSIONS);
            let _ = write_txn.open_table(USER_PREFS);
        }
        write_txn.commit().unwrap();
    }

    // ============== TaskStatus Tests ==============
    #[test]
    fn test_task_status_default() {
        let status = TaskStatus::default();
        assert_eq!(status, TaskStatus::Pending);
    }

    #[test]
    fn test_task_status_equality() {
        assert_eq!(TaskStatus::Pending, TaskStatus::Pending);
        assert_eq!(TaskStatus::Running, TaskStatus::Running);
        assert_eq!(TaskStatus::Complete, TaskStatus::Complete);
        assert_eq!(TaskStatus::Failed, TaskStatus::Failed);

        assert_ne!(TaskStatus::Pending, TaskStatus::Running);
        assert_ne!(TaskStatus::Complete, TaskStatus::Failed);
    }

    // ============== Task Tests ==============
    #[test]
    fn test_task_new() {
        let task = Task::new(1, "Test task".to_string(), "/test/project".to_string());

        assert_eq!(task.id, 1);
        assert_eq!(task.title, "Test task");
        assert_eq!(task.project_path, "/test/project");
        assert!(task.category.is_empty());
        assert!(task.description.is_empty());
        assert!(task.command.is_none());
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.progress, 0.0);
        assert!(task.completed_at.is_none());
        assert!(!task.archived);
    }

    #[test]
    fn test_task_from_suggestion() {
        let task = Task::from_suggestion(
            1,
            "Fix bug",
            "fix",
            "Fix the critical bug in main",
            Some("cargo fix"),
            "/project",
        );

        assert_eq!(task.id, 1);
        assert_eq!(task.title, "Fix bug");
        assert_eq!(task.category, "fix");
        assert_eq!(task.description, "Fix the critical bug in main");
        assert_eq!(task.command, Some("cargo fix".to_string()));
        assert_eq!(task.project_path, "/project");
        assert_eq!(task.status, TaskStatus::Pending);
    }

    #[test]
    fn test_task_from_suggestion_no_command() {
        let task = Task::from_suggestion(
            2,
            "Read docs",
            "docs",
            "Read the documentation",
            None,
            "/project",
        );

        assert!(task.command.is_none());
    }

    // ============== Task CRUD Tests ==============
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
    fn test_task_update() {
        let db = test_db();

        // Create and save
        let mut task = Task::new(1, "Original title".to_string(), "/project".to_string());
        db.save_task(&task).unwrap();

        // Modify and save
        task.title = "Updated title".to_string();
        task.status = TaskStatus::Complete;
        db.save_task(&task).unwrap();

        // Load and verify
        let loaded = db.load_task(1).unwrap().unwrap();
        assert_eq!(loaded.title, "Updated title");
        assert_eq!(loaded.status, TaskStatus::Complete);
    }

    #[test]
    fn test_load_nonexistent_task() {
        let db = test_db();
        let result = db.load_task(999);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_load_current_tasks_empty() {
        let db = test_db();
        let tasks = db.load_current_tasks("/test/project").unwrap();
        assert!(tasks.is_empty());
    }

    #[test]
    fn test_load_current_tasks_filters_by_project() {
        let db = test_db();

        // Add tasks for different projects
        let task1 = Task::new(1, "Task 1".to_string(), "/project1".to_string());
        let task2 = Task::new(2, "Task 2".to_string(), "/project2".to_string());
        let task3 = Task::new(3, "Task 3".to_string(), "/project1".to_string());

        db.save_task(&task1).unwrap();
        db.save_task(&task2).unwrap();
        db.save_task(&task3).unwrap();

        // Load tasks for project1
        let tasks = db.load_current_tasks("/project1").unwrap();
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().any(|t| t.id == 1));
        assert!(tasks.iter().any(|t| t.id == 3));
        assert!(!tasks.iter().any(|t| t.id == 2));
    }

    #[test]
    fn test_load_current_tasks_excludes_archived() {
        let db = test_db();

        let task1 = Task::new(1, "Task 1".to_string(), "/project".to_string());
        let task2 = Task::new(2, "Task 2".to_string(), "/project".to_string());

        db.save_task(&task1).unwrap();
        db.save_task(&task2).unwrap();

        // Archive one task
        db.archive_task(1).unwrap();

        let tasks = db.load_current_tasks("/project").unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, 2);
    }

    #[test]
    fn test_load_current_tasks_sorting() {
        let db = test_db();

        // Create tasks with different timestamps
        let mut task1 = Task::new(1, "Task 1".to_string(), "/project".to_string());
        task1.created_at = 1000;

        let mut task2 = Task::new(2, "Task 2".to_string(), "/project".to_string());
        task2.created_at = 3000;

        let mut task3 = Task::new(3, "Task 3".to_string(), "/project".to_string());
        task3.created_at = 2000;

        db.save_task(&task1).unwrap();
        db.save_task(&task2).unwrap();
        db.save_task(&task3).unwrap();

        let tasks = db.load_current_tasks("/project").unwrap();
        assert_eq!(tasks.len(), 3);
        // Should be sorted newest first
        assert_eq!(tasks[0].id, 2); // created_at = 3000
        assert_eq!(tasks[1].id, 3); // created_at = 2000
        assert_eq!(tasks[2].id, 1); // created_at = 1000
    }

    #[test]
    fn test_load_archived_tasks() {
        let db = test_db();

        let task1 = Task::new(1, "Task 1".to_string(), "/project".to_string());
        let task2 = Task::new(2, "Task 2".to_string(), "/project".to_string());

        db.save_task(&task1).unwrap();
        db.save_task(&task2).unwrap();

        // Archive both
        db.archive_task(1).unwrap();
        db.archive_task(2).unwrap();

        let archived = db.load_archived_tasks().unwrap();
        assert_eq!(archived.len(), 2);
        assert!(archived.iter().all(|t| t.archived));
    }

    #[test]
    fn test_unarchive_task() {
        let db = test_db();

        let task = Task::new(1, "Task".to_string(), "/project".to_string());
        db.save_task(&task).unwrap();

        db.archive_task(1).unwrap();
        let loaded = db.load_task(1).unwrap().unwrap();
        assert!(loaded.archived);

        db.unarchive_task(1).unwrap();
        let loaded = db.load_task(1).unwrap().unwrap();
        assert!(!loaded.archived);
    }

    #[test]
    fn test_next_task_id_empty_db() {
        let db = test_db();
        let next_id = db.next_task_id().unwrap();
        assert_eq!(next_id, 1);
    }

    #[test]
    fn test_next_task_id_existing_tasks() {
        let db = test_db();

        let task1 = Task::new(1, "Task 1".to_string(), "/project".to_string());
        let task2 = Task::new(5, "Task 2".to_string(), "/project".to_string());
        let task3 = Task::new(3, "Task 3".to_string(), "/project".to_string());

        db.save_task(&task1).unwrap();
        db.save_task(&task2).unwrap();
        db.save_task(&task3).unwrap();

        let next_id = db.next_task_id().unwrap();
        assert_eq!(next_id, 6); // max_id (5) + 1
    }

    // ============== Permission Tests ==============
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
    fn test_permission_key_format() {
        let key = PalaceDB::permission_key("/my/project", "cargo build");
        assert_eq!(key, "/my/project:cargo build");
    }

    #[test]
    fn test_multiple_permissions_same_project() {
        let db = test_db();
        let project = "/test/project";

        db.approve_prefix(project, "cargo build").unwrap();
        db.approve_prefix(project, "cargo test").unwrap();
        db.approve_prefix(project, "git push").unwrap();

        let prefixes = db.get_approved_prefixes(project).unwrap();
        assert_eq!(prefixes.len(), 3);
        assert!(prefixes.contains(&"cargo build".to_string()));
        assert!(prefixes.contains(&"cargo test".to_string()));
        assert!(prefixes.contains(&"git push".to_string()));
    }

    #[test]
    fn test_command_matching_with_prefixes() {
        let db = test_db();
        let project = "/test/project";

        // Approve "cargo"
        db.approve_prefix(project, "cargo").unwrap();

        // All these should match
        assert!(db.is_command_approved(project, "cargo build").unwrap());
        assert!(db.is_command_approved(project, "cargo test").unwrap());
        assert!(db.is_command_approved(project, "cargo run --bin foo").unwrap());

        // These should not match
        assert!(!db.is_command_approved(project, "git status").unwrap());
        // Note: "cargox build" starts with "cargo" so it will match
        // This is expected behavior - prefix matching works on starts_with
    }

    #[test]
    fn test_get_approved_prefixes_empty() {
        let db = test_db();
        let prefixes = db.get_approved_prefixes("/nonexistent").unwrap();
        assert!(prefixes.is_empty());
    }

    #[test]
    fn test_get_all_permissions() {
        let db = test_db();

        db.approve_prefix("/project1", "cargo build").unwrap();
        db.approve_prefix("/project1", "cargo test").unwrap();
        db.approve_prefix("/project2", "npm install").unwrap();

        let perms = db.get_all_permissions().unwrap();
        assert_eq!(perms.len(), 3);

        assert!(perms.contains(&("/project1".to_string(), "cargo build".to_string())));
        assert!(perms.contains(&("/project1".to_string(), "cargo test".to_string())));
        assert!(perms.contains(&("/project2".to_string(), "npm install".to_string())));
    }

    #[test]
    fn test_revoke_nonexistent_permission() {
        let db = test_db();
        // Should not error even if permission doesn't exist
        let result = db.revoke_prefix("/project", "nonexistent command");
        assert!(result.is_ok());
    }

    // ============== User Preferences Tests ==============
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

    #[test]
    fn test_pref_update() {
        let db = test_db();

        db.set_pref("counter", &42i32).unwrap();
        let value: i32 = db.get_pref("counter").unwrap().unwrap();
        assert_eq!(value, 42);

        // Update
        db.set_pref("counter", &100i32).unwrap();
        let value: i32 = db.get_pref("counter").unwrap().unwrap();
        assert_eq!(value, 100);
    }

    #[test]
    fn test_get_nonexistent_pref() {
        let db = test_db();
        let result: Option<String> = db.get_pref("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_pref_complex_types() {
        use serde_json::json;

        let db = test_db();

        // Test with JSON value
        let value = json!({
            "name": "test",
            "items": vec![1, 2, 3]
        });

        db.set_pref("complex", &value).unwrap();
        let loaded: serde_json::Value = db.get_pref("complex").unwrap().unwrap();
        assert_eq!(loaded, value);
    }

    #[test]
    fn test_pref_string() {
        let db = test_db();

        db.set_pref("api_key", &"secret-key-123").unwrap();
        let key: String = db.get_pref("api_key").unwrap().unwrap();
        assert_eq!(key, "secret-key-123");
    }

    #[test]
    fn test_pref_option() {
        let db = test_db();

        db.set_pref("optional", &Some(42i32)).unwrap();
        let value: Option<i32> = db.get_pref("optional").unwrap().unwrap();
        assert_eq!(value, Some(42));

        db.set_pref("none", &Option::<i32>::None).unwrap();
        let value: Option<i32> = db.get_pref("none").unwrap().unwrap();
        assert_eq!(value, None);
    }

    #[test]
    fn test_delete_nonexistent_pref() {
        let db = test_db();
        // Should not error
        let result = db.delete_pref("nonexistent");
        assert!(result.is_ok());
    }

    // ============== Integration Tests ==============
    #[test]
    fn test_task_workflow() {
        let db = test_db();

        // Create a task from suggestion
        let task = Task::from_suggestion(
            1,
            "Run tests",
            "test",
            "Run the full test suite",
            Some("cargo test"),
            "/myproject",
        );
        db.save_task(&task).unwrap();

        // Verify it's in current tasks
        let current = db.load_current_tasks("/myproject").unwrap();
        assert_eq!(current.len(), 1);

        // Simulate execution
        let mut loaded = db.load_task(1).unwrap().unwrap();
        loaded.status = TaskStatus::Running;
        db.save_task(&loaded).unwrap();

        loaded.status = TaskStatus::Complete;
        loaded.completed_at = Some(12345);
        db.save_task(&loaded).unwrap();

        // Verify completion
        let final_task = db.load_task(1).unwrap().unwrap();
        assert_eq!(final_task.status, TaskStatus::Complete);
        assert_eq!(final_task.completed_at, Some(12345));

        // Archive it
        db.archive_task(1).unwrap();
        let current = db.load_current_tasks("/myproject").unwrap();
        assert!(current.is_empty());

        let archived = db.load_archived_tasks().unwrap();
        assert_eq!(archived.len(), 1);
    }

    #[test]
    fn test_permission_workflow() {
        let db = test_db();

        // First time: check permission (not approved)
        assert!(!db.is_command_approved("/project", "cargo build --release").unwrap());

        // User approves "cargo" prefix
        db.approve_prefix("/project", "cargo").unwrap();

        // Now all cargo commands are approved
        assert!(db.is_command_approved("/project", "cargo build").unwrap());
        assert!(db.is_command_approved("/project", "cargo test").unwrap());
        assert!(db.is_command_approved("/project", "cargo build --release").unwrap());

        // But other commands still need approval
        assert!(!db.is_command_approved("/project", "git push").unwrap());

        // Check all permissions
        let all = db.get_all_permissions().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0], ("/project".to_string(), "cargo".to_string()));

        // Later, user revokes
        db.revoke_prefix("/project", "cargo").unwrap();
        assert!(!db.is_command_approved("/project", "cargo build").unwrap());
    }
}
