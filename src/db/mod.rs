use anyhow::Result;
use redb::{Database as RedbDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::path::Path;

const TASKS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("tasks");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: u64,
    /// The action label from Claude's YAML (short, <60 chars)
    pub label: String,
    /// Description of what this task involves (can be detailed)
    #[serde(default)]
    pub description: String,
    /// Estimated time to complete (e.g., "5 min", "30 min", "2 hours")
    #[serde(default)]
    pub time_estimate: Option<String>,
    /// Complexity level (e.g., "trivial", "simple", "moderate", "complex")
    #[serde(default)]
    pub complexity: Option<String>,
    /// Files likely to be affected
    #[serde(default)]
    pub affected_files: Vec<String>,
    /// When this task was suggested (unix timestamp)
    pub created_at: u64,
    /// Whether user has selected this for execution
    pub selected: bool,
    /// Whether this has been executed
    pub completed: bool,
}

pub struct Database {
    db: RedbDatabase,
    /// In-memory cache of tasks
    tasks: Vec<Task>,
    /// Current cursor position
    cursor: usize,
    /// Next ID for new tasks
    next_id: u64,
}

impl Database {
    pub fn open(project_path: &Path) -> Result<Self> {
        let db_path = project_path.join(".palace").join("palace.redb");

        // Ensure .palace directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let db = RedbDatabase::create(&db_path)?;

        // Initialize table
        {
            let write_txn = db.begin_write()?;
            let _ = write_txn.open_table(TASKS_TABLE);
            write_txn.commit()?;
        }

        // Load tasks
        let (tasks, next_id) = Self::load_tasks(&db)?;

        Ok(Self {
            db,
            tasks,
            cursor: 0,
            next_id,
        })
    }

    fn load_tasks(db: &RedbDatabase) -> Result<(Vec<Task>, u64)> {
        let read_txn = db.begin_read()?;
        let table = read_txn.open_table(TASKS_TABLE)?;

        let mut tasks = Vec::new();
        let mut max_id = 0u64;

        for result in table.iter()? {
            let (key, value) = result?;
            let id = key.value();
            let task: Task = serde_json::from_slice(value.value())?;
            max_id = max_id.max(id);
            tasks.push(task);
        }

        // Sort by ID
        tasks.sort_by_key(|t| t.id);

        Ok((tasks, max_id + 1))
    }

    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn select_next(&mut self) {
        if !self.tasks.is_empty() && self.cursor < self.tasks.len() - 1 {
            self.cursor += 1;
        }
    }

    pub fn select_previous(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    pub fn select_index(&mut self, index: usize) {
        if index < self.tasks.len() {
            self.cursor = index;
        }
    }

    /// Jump up by N items (Ctrl+Up)
    pub fn jump_up(&mut self, n: usize) {
        self.cursor = self.cursor.saturating_sub(n);
    }

    /// Jump down by N items (Ctrl+Down)
    pub fn jump_down(&mut self, n: usize) {
        if !self.tasks.is_empty() {
            self.cursor = (self.cursor + n).min(self.tasks.len() - 1);
        }
    }

    /// Page up by visible height
    pub fn page_up(&mut self, page_size: usize) {
        self.cursor = self.cursor.saturating_sub(page_size);
    }

    /// Page down by visible height
    pub fn page_down(&mut self, page_size: usize) {
        if !self.tasks.is_empty() {
            self.cursor = (self.cursor + page_size).min(self.tasks.len() - 1);
        }
    }

    /// Jump to first item
    pub fn jump_to_start(&mut self) {
        self.cursor = 0;
    }

    /// Jump to last item
    pub fn jump_to_end(&mut self) {
        if !self.tasks.is_empty() {
            self.cursor = self.tasks.len() - 1;
        }
    }

    /// Select current and move up (Shift+Up)
    pub fn select_and_move_up(&mut self) {
        if let Some(task) = self.tasks.get_mut(self.cursor) {
            task.selected = true;
        }
        if self.cursor > 0 {
            self.cursor -= 1;
            if let Some(task) = self.tasks.get_mut(self.cursor) {
                task.selected = true;
            }
        }
        let _ = self.save_all();
    }

    /// Select current and move down (Shift+Down)
    pub fn select_and_move_down(&mut self) {
        if let Some(task) = self.tasks.get_mut(self.cursor) {
            task.selected = true;
        }
        if !self.tasks.is_empty() && self.cursor < self.tasks.len() - 1 {
            self.cursor += 1;
            if let Some(task) = self.tasks.get_mut(self.cursor) {
                task.selected = true;
            }
        }
        let _ = self.save_all();
    }

    /// Select range from current cursor to target index (Shift+click)
    pub fn select_range_to(&mut self, target: usize) {
        let target = target.min(self.tasks.len().saturating_sub(1));
        let (start, end) = if self.cursor <= target {
            (self.cursor, target)
        } else {
            (target, self.cursor)
        };

        for i in start..=end {
            if let Some(task) = self.tasks.get_mut(i) {
                task.selected = true;
            }
        }
        self.cursor = target;
        let _ = self.save_all();
    }

    pub fn toggle_current_selection(&mut self) {
        if let Some(task) = self.tasks.get_mut(self.cursor) {
            task.selected = !task.selected;
        }
        if let Some(task) = self.tasks.get(self.cursor) {
            let _ = self.save_task(task.clone());
        }
    }

    /// Toggle all selected - if any unselected, select all; otherwise deselect all
    pub fn toggle_select_all(&mut self) {
        let all_selected = self.tasks.iter().all(|t| t.selected);
        for task in &mut self.tasks {
            task.selected = !all_selected;
        }
        let _ = self.save_all();
    }

    /// Deselect all tasks
    pub fn deselect_all(&mut self) {
        for task in &mut self.tasks {
            task.selected = false;
        }
        let _ = self.save_all();
    }

    pub fn has_selection(&self) -> bool {
        self.tasks.iter().any(|t| t.selected)
    }

    pub fn selected_tasks(&self) -> Vec<&Task> {
        self.tasks.iter().filter(|t| t.selected).collect()
    }

    pub fn add_task(
        &mut self,
        label: String,
        description: String,
        time_estimate: Option<String>,
        complexity: Option<String>,
        affected_files: Vec<String>,
    ) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;

        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let task = Task {
            id,
            label,
            description,
            time_estimate,
            complexity,
            affected_files,
            created_at,
            selected: false,
            completed: false,
        };

        self.save_task(task.clone())?;
        // Insert at beginning so newest are first
        self.tasks.insert(0, task);
        // Adjust cursor to stay on same item
        if self.cursor < self.tasks.len() - 1 {
            self.cursor += 1;
        }

        Ok(id)
    }

    /// Check if a task with this label already exists (avoid duplicates)
    pub fn has_task_with_label(&self, label: &str) -> bool {
        self.tasks.iter().any(|t| t.label == label)
    }

    /// Get count of incomplete tasks
    pub fn pending_count(&self) -> usize {
        self.tasks.iter().filter(|t| !t.completed).count()
    }

    pub fn delete_selected(&mut self) -> Result<()> {
        let ids_to_delete: Vec<u64> = self.tasks.iter().filter(|t| t.selected).map(|t| t.id).collect();

        if ids_to_delete.is_empty() {
            return Ok(());
        }

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(TASKS_TABLE)?;
            for id in &ids_to_delete {
                table.remove(*id)?;
            }
        }
        write_txn.commit()?;

        self.tasks.retain(|t| !t.selected);

        // Adjust cursor if needed
        if self.cursor >= self.tasks.len() && !self.tasks.is_empty() {
            self.cursor = self.tasks.len() - 1;
        }

        Ok(())
    }

    fn save_task(&self, task: Task) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(TASKS_TABLE)?;
            let data = serde_json::to_vec(&task)?;
            table.insert(task.id, data.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn save_all(&self) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(TASKS_TABLE)?;
            for task in &self.tasks {
                let data = serde_json::to_vec(task)?;
                table.insert(task.id, data.as_slice())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }
}
