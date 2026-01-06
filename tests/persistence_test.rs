//! Integration tests for Palace persistence layer
//!
//! Tests database operations, task persistence, and permission management.

use tempfile::TempDir;

use palace::persistence::{PalaceDB, Task, TaskStatus};

// ============== Test Utilities ==============

/// Creates a test database in a temporary directory
fn create_test_db() -> (PalaceDB, TempDir) {
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("test.redb");

    // Create the database
    let db = Database::create(&db_path).expect("Failed to create test database");

    let palace_db = PalaceDB { db };

    // Initialize tables by opening them (creates if not exists)
    let write_txn = palace_db.db.begin_write().unwrap();

    // Define tables (must match persistence.rs)
    let tasks_def = redb::TableDefinition::new("tasks");
    let permissions_def = redb::TableDefinition::new("project_permissions");
    let prefs_def = redb::TableDefinition::new("user_prefs");

    {
        let _ = write_txn.open_table(tasks_def);
        let _ = write_txn.open_table(permissions_def);
        let _ = write_txn.open_table(prefs_def);
    }
    write_txn.commit().unwrap();

    (palace_db, dir)
}

/// Note: Table definitions match those in persistence.rs
/// They are redefined here since they're private
use redb::{Database, TableDefinition};

const TASKS: TableDefinition<u64, &[u8]> = TableDefinition::new("tasks");
const PROJECT_PERMISSIONS: TableDefinition<&str, bool> = TableDefinition::new("project_permissions");
const USER_PREFS: TableDefinition<&str, &[u8]> = TableDefinition::new("user_prefs");

// ============== Task Lifecycle Integration Tests ==============

#[test]
fn test_task_full_lifecycle() {
    let (db, _dir) = create_test_db();

    // Create a new task
    let task = Task::new(1, "Implement feature X".to_string(), "/test/project".to_string());

    // Save it
    db.save_task(&task).expect("Failed to save task");

    // Load it back
    let loaded = db
        .load_task(1)
        .expect("Failed to load task")
        .expect("Task not found");

    assert_eq!(loaded.id, 1);
    assert_eq!(loaded.title, "Implement feature X");
    assert_eq!(loaded.status, TaskStatus::Pending);
    assert!(!loaded.archived);

    // Update to running
    let mut loaded = loaded;
    loaded.status = TaskStatus::Running;
    db.save_task(&loaded).expect("Failed to update task");

    let running = db
        .load_task(1)
        .expect("Failed to load task")
        .expect("Task not found");
    assert_eq!(running.status, TaskStatus::Running);

    // Complete it
    let mut running = running;
    running.status = TaskStatus::Complete;
    running.completed_at = Some(12345);
    db.save_task(&running).expect("Failed to update task");

    let completed = db
        .load_task(1)
        .expect("Failed to load task")
        .expect("Task not found");
    assert_eq!(completed.status, TaskStatus::Complete);
    assert_eq!(completed.completed_at, Some(12345));

    // Archive it
    db.archive_task(1).expect("Failed to archive task");

    let archived = db
        .load_task(1)
        .expect("Failed to load task")
        .expect("Task not found");
    assert!(archived.archived);

    // Delete it
    db.delete_task(1).expect("Failed to delete task");

    let deleted = db.load_task(1).expect("Failed to load task");
    assert!(deleted.is_none(), "Task should be deleted");
}

#[test]
fn test_task_from_suggestion_workflow() {
    let (db, _dir) = create_test_db();

    // Create task from suggestion
    let task = Task::from_suggestion(
        1,
        "Fix memory leak",
        "fix",
        "Fix the memory leak in the parser",
        Some("cargo fix"),
        "/myproject",
    );

    db.save_task(&task).expect("Failed to save task");

    // Verify all fields were saved
    let loaded = db
        .load_task(1)
        .expect("Failed to load task")
        .expect("Task not found");

    assert_eq!(loaded.title, "Fix memory leak");
    assert_eq!(loaded.category, "fix");
    assert_eq!(loaded.description, "Fix the memory leak in the parser");
    assert_eq!(loaded.command, Some("cargo fix".to_string()));
    assert_eq!(loaded.project_path, "/myproject");
    assert_eq!(loaded.status, TaskStatus::Pending);
    assert!(!loaded.archived);
}

#[test]
fn test_multiple_tasks_project_filtering() {
    let (db, _dir) = create_test_db();

    // Create tasks for different projects
    let task1 = Task::new(1, "Project A task 1".to_string(), "/project/a".to_string());
    let task2 = Task::new(2, "Project A task 2".to_string(), "/project/a".to_string());
    let task3 = Task::new(3, "Project B task 1".to_string(), "/project/b".to_string());

    db.save_task(&task1).expect("Failed to save task");
    db.save_task(&task2).expect("Failed to save task");
    db.save_task(&task3).expect("Failed to save task");

    // Load tasks for project A
    let project_a_tasks = db
        .load_current_tasks("/project/a")
        .expect("Failed to load tasks");

    assert_eq!(project_a_tasks.len(), 2);
    assert!(project_a_tasks.iter().all(|t| t.project_path == "/project/a"));
    assert!(project_a_tasks.iter().any(|t| t.id == 1));
    assert!(project_a_tasks.iter().any(|t| t.id == 2));

    // Load tasks for project B
    let project_b_tasks = db
        .load_current_tasks("/project/b")
        .expect("Failed to load tasks");

    assert_eq!(project_b_tasks.len(), 1);
    assert_eq!(project_b_tasks[0].id, 3);
}

#[test]
fn test_task_archiving_workflow() {
    let (db, _dir) = create_test_db();

    // Create multiple tasks
    for i in 1..=5 {
        let task = Task::new(i, format!("Task {}", i), "/project".to_string());
        db.save_task(&task).expect("Failed to save task");
    }

    // Verify all are current
    let current = db.load_current_tasks("/project").expect("Failed to load tasks");
    assert_eq!(current.len(), 5);

    // Archive some tasks
    db.archive_task(2).expect("Failed to archive");
    db.archive_task(4).expect("Failed to archive");

    // Check current tasks (should be 1, 3, 5)
    let current = db.load_current_tasks("/project").expect("Failed to load tasks");
    assert_eq!(current.len(), 3);
    let current_ids: Vec<_> = current.iter().map(|t| t.id).collect();
    assert_eq!(current_ids, vec![1, 3, 5]);

    // Check archived tasks
    let archived = db.load_archived_tasks().expect("Failed to load archived");
    assert_eq!(archived.len(), 2);
    let archived_ids: Vec<_> = archived.iter().map(|t| t.id).collect();
    assert_eq!(archived_ids, vec![2, 4]);

    // Unarchive one
    db.unarchive_task(2).expect("Failed to unarchive");

    let current = db.load_current_tasks("/project").expect("Failed to load tasks");
    assert_eq!(current.len(), 4);
}

#[test]
fn test_task_id_generation() {
    let (db, _dir) = create_test_db();

    // Empty DB should start at ID 1
    let next_id = db.next_task_id().expect("Failed to get next ID");
    assert_eq!(next_id, 1);

    // Add some tasks with non-sequential IDs
    let task1 = Task::new(1, "Task 1".to_string(), "/project".to_string());
    let task5 = Task::new(5, "Task 5".to_string(), "/project".to_string());
    let task3 = Task::new(3, "Task 3".to_string(), "/project".to_string());

    db.save_task(&task1).expect("Failed to save task");
    db.save_task(&task5).expect("Failed to save task");
    db.save_task(&task3).expect("Failed to save task");

    // Next ID should be max + 1
    let next_id = db.next_task_id().expect("Failed to get next ID");
    assert_eq!(next_id, 6, "Should be max_id (5) + 1");
}

#[test]
fn test_task_sorting_by_creation_time() {
    let (db, _dir) = create_test_db();

    // Create tasks with different timestamps
    let mut task1 = Task::new(1, "Task 1".to_string(), "/project".to_string());
    task1.created_at = 1000;

    let mut task2 = Task::new(2, "Task 2".to_string(), "/project".to_string());
    task2.created_at = 3000;

    let mut task3 = Task::new(3, "Task 3".to_string(), "/project".to_string());
    task3.created_at = 2000;

    db.save_task(&task1).expect("Failed to save task");
    db.save_task(&task2).expect("Failed to save task");
    db.save_task(&task3).expect("Failed to save task");

    // Load should return sorted by creation time (newest first)
    let tasks = db.load_current_tasks("/project").expect("Failed to load tasks");

    assert_eq!(tasks.len(), 3);
    assert_eq!(tasks[0].id, 2, "Task 2 should be first (created_at=3000)");
    assert_eq!(tasks[1].id, 3, "Task 3 should be second (created_at=2000)");
    assert_eq!(tasks[2].id, 1, "Task 1 should be third (created_at=1000)");
}

#[test]
fn test_task_update_preserves_all_fields() {
    let (db, _dir) = create_test_db();

    // Create a comprehensive task
    let mut task = Task::new(1, "Original title".to_string(), "/project".to_string());
    task.category = "fix".to_string();
    task.description = "Original description".to_string();
    task.command = Some("cargo build".to_string());
    task.progress = 0.5;

    db.save_task(&task).expect("Failed to save task");

    // Update multiple fields
    task.title = "Updated title".to_string();
    task.status = TaskStatus::Complete;
    task.progress = 1.0;
    task.completed_at = Some(99999);

    db.save_task(&task).expect("Failed to update task");

    // Verify all fields are preserved
    let loaded = db
        .load_task(1)
        .expect("Failed to load task")
        .expect("Task not found");

    assert_eq!(loaded.title, "Updated title");
    assert_eq!(loaded.category, "fix", "Category should be preserved");
    assert_eq!(loaded.description, "Original description", "Description should be preserved");
    assert_eq!(loaded.command, Some("cargo build".to_string()), "Command should be preserved");
    assert_eq!(loaded.status, TaskStatus::Complete);
    assert_eq!(loaded.progress, 1.0);
    assert_eq!(loaded.completed_at, Some(99999));
}

// ============== Permission Management Integration Tests ==============

#[test]
fn test_permission_approval_workflow() {
    let (db, _dir) = create_test_db();

    let project = "/test/project";

    // Initially no permissions
    assert!(!db.is_prefix_approved(project, "cargo build").unwrap());

    // Approve a prefix
    db.approve_prefix(project, "cargo build").unwrap();
    assert!(db.is_prefix_approved(project, "cargo build").unwrap());

    // Check command matching
    assert!(db
        .is_command_approved(project, "cargo build --release")
        .unwrap());

    // Revoke it
    db.revoke_prefix(project, "cargo build").unwrap();
    assert!(!db.is_prefix_approved(project, "cargo build").unwrap());
}

#[test]
fn test_permission_prefix_matching() {
    let (db, _dir) = create_test_db();

    let project = "/test/project";

    // Approve "cargo" prefix
    db.approve_prefix(project, "cargo").unwrap();

    // All cargo commands should match
    assert!(db.is_command_approved(project, "cargo build").unwrap());
    assert!(db.is_command_approved(project, "cargo test").unwrap());
    assert!(db.is_command_approved(project, "cargo run --bin foo").unwrap());
    assert!(db.is_command_approved(project, "cargo build --release -j 4").unwrap());

    // Non-cargo commands should not match
    assert!(!db.is_command_approved(project, "git status").unwrap());
    assert!(!db.is_command_approved(project, "npm install").unwrap());
}

#[test]
fn test_permission_project_isolation() {
    let (db, _dir) = create_test_db();

    // Approve for project A
    db.approve_prefix("/project/a", "cargo build").unwrap();

    // Should be approved for project A
    assert!(db
        .is_command_approved("/project/a", "cargo build")
        .unwrap());

    // Should NOT be approved for project B
    assert!(!db
        .is_command_approved("/project/b", "cargo build")
        .unwrap());

    // Approve different command for project B
    db.approve_prefix("/project/b", "npm install").unwrap();

    // Project A should still only have cargo
    assert!(db
        .is_command_approved("/project/a", "cargo build")
        .unwrap());
    assert!(!db
        .is_command_approved("/project/a", "npm install")
        .unwrap());

    // Project B should only have npm
    assert!(!db
        .is_command_approved("/project/b", "cargo build")
        .unwrap());
    assert!(db
        .is_command_approved("/project/b", "npm install")
        .unwrap());
}

#[test]
fn test_multiple_permissions_same_project() {
    let (db, _dir) = create_test_db();

    let project = "/test/project";

    // Approve multiple prefixes
    db.approve_prefix(project, "cargo build").unwrap();
    db.approve_prefix(project, "cargo test").unwrap();
    db.approve_prefix(project, "git push").unwrap();
    db.approve_prefix(project, "npm install").unwrap();

    // Get all approved prefixes
    let prefixes = db.get_approved_prefixes(project).expect("Failed to get prefixes");

    assert_eq!(prefixes.len(), 4);
    assert!(prefixes.contains(&"cargo build".to_string()));
    assert!(prefixes.contains(&"cargo test".to_string()));
    assert!(prefixes.contains(&"git push".to_string()));
    assert!(prefixes.contains(&"npm install".to_string()));

    // All should be approved
    assert!(db.is_command_approved(project, "cargo build").unwrap());
    assert!(db.is_command_approved(project, "cargo test").unwrap());
    assert!(db.is_command_approved(project, "git push").unwrap());
    assert!(db.is_command_approved(project, "npm install").unwrap());
}

#[test]
fn test_permission_key_format() {
    // Test the internal key format
    let key = palace::persistence::PalaceDB::permission_key("/my/project", "cargo build");
    assert_eq!(key, "/my/project:cargo build");

    let key2 = palace::persistence::PalaceDB::permission_key("/project", "cmd");
    assert_eq!(key2, "/project:cmd");
}

#[test]
fn test_get_all_permissions() {
    let (db, _dir) = create_test_db();

    // Set up permissions for multiple projects
    db.approve_prefix("/project/a", "cargo build").unwrap();
    db.approve_prefix("/project/a", "cargo test").unwrap();
    db.approve_prefix("/project/b", "npm install").unwrap();
    db.approve_prefix("/project/c", "git push").unwrap();

    // Get all permissions
    let all = db.get_all_permissions().expect("Failed to get all permissions");

    assert_eq!(all.len(), 4);
    assert!(all.contains(&(
        "/project/a".to_string(),
        "cargo build".to_string()
    )));
    assert!(all.contains(&(
        "/project/a".to_string(),
        "cargo test".to_string()
    )));
    assert!(all.contains(&(
        "/project/b".to_string(),
        "npm install".to_string()
    )));
    assert!(all.contains(&(
        "/project/c".to_string(),
        "git push".to_string()
    )));
}

#[test]
fn test_revoke_nonexistent_permission() {
    let (db, _dir) = create_test_db();

    // Revoking a non-existent permission should not error
    let result = db.revoke_prefix("/project", "nonexistent command");
    assert!(result.is_ok(), "Revoking non-existent permission should succeed");
}

#[test]
fn test_permission_persistence() {
    let (db, _dir) = create_test_db();

    let project = "/test/project";

    // Approve a permission
    db.approve_prefix(project, "cargo build").unwrap();

    // It should persist
    assert!(db.is_prefix_approved(project, "cargo build").unwrap());

    // Even if we query it multiple times
    assert!(db.is_prefix_approved(project, "cargo build").unwrap());
    assert!(db.is_prefix_approved(project, "cargo build").unwrap());

    // Commands should match
    assert!(db
        .is_command_approved(project, "cargo build --release")
        .unwrap());
    assert!(db
        .is_command_approved(project, "cargo build --verbose")
        .unwrap());
}

// ============== User Preferences Integration Tests ==============

#[test]
fn test_user_preferences_crud() {
    let (db, _dir) = create_test_db();

    // Set a simple preference
    db.set_pref("dark_mode", &true).unwrap();

    // Get it back
    let dark_mode: bool = db.get_pref("dark_mode").unwrap().expect("Pref not found");
    assert!(dark_mode);

    // Update it
    db.set_pref("dark_mode", &false).unwrap();
    let dark_mode: bool = db.get_pref("dark_mode").unwrap().expect("Pref not found");
    assert!(!dark_mode);

    // Delete it
    db.delete_pref("dark_mode").unwrap();
    let dark_mode: Option<bool> = db.get_pref("dark_mode").unwrap();
    assert!(dark_mode.is_none());
}

#[test]
fn test_user_preferences_different_types() {
    let (db, _dir) = create_test_db();

    // Boolean
    db.set_pref("bool_val", &true).unwrap();
    let bool_val: bool = db.get_pref("bool_val").unwrap().unwrap();
    assert!(bool_val);

    // Integer
    db.set_pref("int_val", &42i32).unwrap();
    let int_val: i32 = db.get_pref("int_val").unwrap().unwrap();
    assert_eq!(int_val, 42);

    // Float
    db.set_pref("float_val", &3.14f64).unwrap();
    let float_val: f64 = db.get_pref("float_val").unwrap().unwrap();
    assert!((float_val - 3.14).abs() < 0.001);

    // String
    db.set_pref("string_val", &"hello, world").unwrap();
    let string_val: String = db.get_pref("string_val").unwrap().unwrap();
    assert_eq!(string_val, "hello, world");
}

#[test]
fn test_user_preferences_complex_types() {
    use serde_json::json;

    let (db, _dir) = create_test_db();

    // JSON value
    let config = json!({
        "api_key": "secret",
        "endpoints": ["endpoint1", "endpoint2"],
        "settings": {
            "timeout": 30,
            "retries": 3
        }
    });

    db.set_pref("config", &config).unwrap();

    let loaded: serde_json::Value = db.get_pref("config").unwrap().unwrap();
    assert_eq!(loaded, config);

    // Vec
    let items = vec!["item1", "item2", "item3"];
    db.set_pref("items", &items).unwrap();

    let loaded: Vec<String> = db.get_pref("items").unwrap().unwrap();
    assert_eq!(loaded, items);
}

#[test]
fn test_user_preferences_option_types() {
    let (db, _dir) = create_test_db();

    // Some
    db.set_pref("some_value", &Some(42i32)).unwrap();
    let value: Option<i32> = db.get_pref("some_value").unwrap().unwrap();
    assert_eq!(value, Some(42));

    // None
    db.set_pref("none_value", &Option::<i32>::None).unwrap();
    let value: Option<i32> = db.get_pref("none_value").unwrap().unwrap();
    assert_eq!(value, None);
}

#[test]
fn test_user_preferences_isolation() {
    let (db, _dir) = create_test_db();

    // Set multiple preferences
    db.set_pref("pref1", &1).unwrap();
    db.set_pref("pref2", &2).unwrap();
    db.set_pref("pref3", &3).unwrap();

    // Each should be independent
    assert_eq!(db.get_pref::<i32>("pref1").unwrap().unwrap(), 1);
    assert_eq!(db.get_pref::<i32>("pref2").unwrap().unwrap(), 2);
    assert_eq!(db.get_pref::<i32>("pref3").unwrap().unwrap(), 3);

    // Deleting one shouldn't affect others
    db.delete_pref("pref2").unwrap();

    assert!(db.get_pref::<i32>("pref1").unwrap().is_some());
    assert!(db.get_pref::<i32>("pref2").unwrap().is_none());
    assert!(db.get_pref::<i32>("pref3").unwrap().is_some());
}

#[test]
fn test_user_preferences_update_same_key() {
    let (db, _dir) = create_test_db();

    // Set initial value
    db.set_pref("counter", &0i32).unwrap();
    let value: i32 = db.get_pref("counter").unwrap().unwrap();
    assert_eq!(value, 0);

    // Update multiple times
    for i in 1..=10 {
        db.set_pref("counter", &i).unwrap();
    }

    // Final value should be 10
    let value: i32 = db.get_pref("counter").unwrap().unwrap();
    assert_eq!(value, 10);
}

#[test]
fn test_delete_nonexistent_preference() {
    let (db, _dir) = create_test_db();

    // Deleting a non-existent pref should not error
    let result = db.delete_pref("nonexistent");
    assert!(result.is_ok());
}

// ============== Integration Workflow Tests ==============

#[test]
fn test_full_task_permission_workflow() {
    let (db, _dir) = create_test_db();

    let project = "/myproject";

    // User starts working on a project
    // AI suggests running a command that needs approval
    assert!(!db.is_command_approved(project, "cargo build").unwrap());

    // User approves the command
    db.approve_prefix(project, "cargo").unwrap();

    // Now all cargo commands are approved
    assert!(db.is_command_approved(project, "cargo build").unwrap());
    assert!(db.is_command_approved(project, "cargo test").unwrap());

    // Task is created from the suggestion
    let task = Task::from_suggestion(
        1,
        "Build the project",
        "build",
        "Run cargo build to compile",
        Some("cargo build --release"),
        project,
    );

    db.save_task(&task).expect("Failed to save task");

    // Task can be loaded
    let loaded = db.load_task(1).unwrap().unwrap();
    assert_eq!(loaded.title, "Build the project");

    // Task appears in current tasks
    let current = db.load_current_tasks(project).unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].id, 1);

    // Simulate execution
    let mut loaded = loaded;
    loaded.status = TaskStatus::Running;
    db.save_task(&loaded).unwrap();

    let mut running = db.load_task(1).unwrap().unwrap();
    running.status = TaskStatus::Complete;
    running.completed_at = Some(12345);
    db.save_task(&running).unwrap();

    // Task is complete
    let completed = db.load_task(1).unwrap().unwrap();
    assert_eq!(completed.status, TaskStatus::Complete);

    // Archive completed task
    db.archive_task(1).unwrap();

    let current = db.load_current_tasks(project).unwrap();
    assert!(current.is_empty(), "Completed task should be archived");

    let archived = db.load_archived_tasks().unwrap();
    assert_eq!(archived.len(), 1);
}

#[test]
fn test_multi_project_permissions_workflow() {
    let (db, _dir) = create_test_db();

    // User works on multiple projects
    let project_a = "/projects/rust-app";
    let project_b = "/projects/node-app";

    // Different permissions for each project
    db.approve_prefix(project_a, "cargo build").unwrap();
    db.approve_prefix(project_a, "cargo test").unwrap();
    db.approve_prefix(project_b, "npm install").unwrap();
    db.approve_prefix(project_b, "npm test").unwrap();

    // Verify isolation
    assert!(db
        .is_command_approved(project_a, "cargo build")
        .unwrap());
    assert!(!db
        .is_command_approved(project_a, "npm install")
        .unwrap());

    assert!(db
        .is_command_approved(project_b, "npm install")
        .unwrap());
    assert!(!db
        .is_command_approved(project_b, "cargo build")
        .unwrap());

    // Create tasks for each project
    let task_a = Task::from_suggestion(
        1,
        "Build Rust app",
        "build",
        "Compile the Rust application",
        Some("cargo build --release"),
        project_a,
    );

    let task_b = Task::from_suggestion(
        2,
        "Install Node dependencies",
        "build",
        "Install npm packages",
        Some("npm install"),
        project_b,
    );

    db.save_task(&task_a).unwrap();
    db.save_task(&task_b).unwrap();

    // Load tasks per project
    let tasks_a = db.load_current_tasks(project_a).unwrap();
    assert_eq!(tasks_a.len(), 1);
    assert_eq!(tasks_a[0].title, "Build Rust app");

    let tasks_b = db.load_current_tasks(project_b).unwrap();
    assert_eq!(tasks_b.len(), 1);
    assert_eq!(tasks_b[0].title, "Install Node dependencies");
}
