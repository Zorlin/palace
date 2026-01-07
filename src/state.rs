#![allow(dead_code)]
use std::path::PathBuf;

/// Display configuration (persisted)
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DisplaySettings {
    /// Primary display ID (raw monitor name)
    pub primary: Option<String>,
    /// Enabled display IDs (raw monitor names)
    pub enabled: Vec<String>,
}

/// Application settings (persisted)
#[derive(Debug, Clone)]
pub struct Settings {
    /// Dark mode enabled (true = dark, false = light)
    pub dark_mode: bool,
    /// UI scale setting (None = auto-detect, Some = user override)
    pub ui_scale: Option<f32>,
    /// Display configuration
    pub display: DisplaySettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            dark_mode: true, // Default to dark mode (OLED-friendly)
            ui_scale: None,  // Auto-detect by default
            display: DisplaySettings::default(),
        }
    }
}

/// Layout mode based on display aspect ratio
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayoutMode {
    /// Standard layouts: 16:9, 21:9
    #[default]
    Standard,
    /// Ultrawide layout: 32:9 (aspect ratio > 2.5)
    Ultrawide,
}

impl LayoutMode {
    /// Detect layout mode from screen dimensions
    pub fn from_dimensions(width: u32, height: u32) -> Self {
        let aspect = width as f32 / height as f32;
        if aspect > 2.5 {
            LayoutMode::Ultrawide
        } else {
            LayoutMode::Standard
        }
    }

    /// Is this an ultrawide display?
    pub fn is_ultrawide(&self) -> bool {
        matches!(self, LayoutMode::Ultrawide)
    }
}

/// Main menu items (opened with Start button)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MainMenuItem {
    Resume,
    Settings,
    Exit,
}

impl MainMenuItem {
    pub fn all() -> &'static [MainMenuItem] {
        &[MainMenuItem::Resume, MainMenuItem::Settings, MainMenuItem::Exit]
    }

    pub fn label(&self) -> &'static str {
        match self {
            MainMenuItem::Resume => "Resume",
            MainMenuItem::Settings => "Settings",
            MainMenuItem::Exit => "Exit",
        }
    }
}

/// Settings menu items
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsItem {
    Display,
    DarkMode,
    UiScale,
}

impl SettingsItem {
    pub fn all() -> &'static [SettingsItem] {
        &[SettingsItem::Display, SettingsItem::DarkMode, SettingsItem::UiScale]
    }

    pub fn label(&self) -> &'static str {
        match self {
            SettingsItem::Display => "Display",
            SettingsItem::DarkMode => "Dark Mode",
            SettingsItem::UiScale => "UI Scale",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            SettingsItem::Display => "Configure monitors and extend across displays",
            SettingsItem::DarkMode => "Toggle dark/light theme",
            SettingsItem::UiScale => "Adjust interface size",
        }
    }
}

/// UI Scale options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiScaleOption {
    Auto,
    Scale50,
    Scale100,
    Scale150,
    Scale200,
    Scale300,
    Scale400,
}

impl UiScaleOption {
    pub fn all() -> &'static [UiScaleOption] {
        &[
            UiScaleOption::Auto,
            UiScaleOption::Scale50,
            UiScaleOption::Scale100,
            UiScaleOption::Scale150,
            UiScaleOption::Scale200,
            UiScaleOption::Scale300,
            UiScaleOption::Scale400,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            UiScaleOption::Auto => "Auto-detect",
            UiScaleOption::Scale50 => "50%",
            UiScaleOption::Scale100 => "100%",
            UiScaleOption::Scale150 => "150%",
            UiScaleOption::Scale200 => "200%",
            UiScaleOption::Scale300 => "300%",
            UiScaleOption::Scale400 => "400%",
        }
    }

    /// Get the scale value (None for Auto)
    pub fn value(&self) -> Option<f32> {
        match self {
            UiScaleOption::Auto => None,
            UiScaleOption::Scale50 => Some(0.5),
            UiScaleOption::Scale100 => Some(1.0),
            UiScaleOption::Scale150 => Some(1.5),
            UiScaleOption::Scale200 => Some(2.0),
            UiScaleOption::Scale300 => Some(3.0),
            UiScaleOption::Scale400 => Some(4.0),
        }
    }

    /// Get the option from a settings value (None = Auto)
    pub fn from_setting(scale: Option<f32>) -> Self {
        match scale {
            None => UiScaleOption::Auto,
            Some(s) if s < 0.75 => UiScaleOption::Scale50,
            Some(s) if s < 1.25 => UiScaleOption::Scale100,
            Some(s) if s < 1.75 => UiScaleOption::Scale150,
            Some(s) if s < 2.5 => UiScaleOption::Scale200,
            Some(s) if s < 3.5 => UiScaleOption::Scale300,
            Some(_) => UiScaleOption::Scale400,
        }
    }
}

/// Menu actions available in ProjectView
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectAction {
    StartPalaceLoop,
    Build,
    Run,
    ViewGitHistory,
}

impl ProjectAction {
    pub fn all() -> &'static [ProjectAction] {
        &[
            ProjectAction::StartPalaceLoop,
            ProjectAction::Build,
            ProjectAction::Run,
            ProjectAction::ViewGitHistory,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            ProjectAction::StartPalaceLoop => "Start Palace Loop",
            ProjectAction::Build => "Build",
            ProjectAction::Run => "Run",
            ProjectAction::ViewGitHistory => "View Git History",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            ProjectAction::StartPalaceLoop => "Start an AI-assisted development session",
            ProjectAction::Build => "Compile the project",
            ProjectAction::Run => "Execute the project",
            ProjectAction::ViewGitHistory => "Browse commit history",
        }
    }
}

/// A streaming suggestion card that fills in as data arrives
#[derive(Debug, Clone)]
pub struct SuggestionCard {
    /// Unique ID for this suggestion
    pub id: usize,
    /// Title (streams in)
    pub title: String,
    /// Category: fix, test, build, refactor, docs
    pub category: String,
    /// Description (streams in)
    pub description: String,
    /// Optional command
    pub command: Option<String>,
    /// Is this card selected for execution?
    pub selected: bool,
    /// Is this card still streaming (incomplete)?
    pub streaming: bool,
}

impl SuggestionCard {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            title: String::new(),
            category: String::new(),
            description: String::new(),
            command: None,
            selected: false,
            streaming: true,
        }
    }

    /// Get the color based on category
    pub fn color(&self) -> [f32; 4] {
        match self.category.as_str() {
            "fix" => [1.0, 0.4, 0.4, 1.0],       // Red
            "test" => [0.4, 0.8, 1.0, 1.0],      // Cyan
            "build" => [1.0, 0.7, 0.2, 1.0],     // Orange
            "refactor" => [0.7, 0.5, 1.0, 1.0],  // Purple
            "docs" => [0.5, 0.9, 0.5, 1.0],      // Green
            _ => [0.5, 0.5, 0.6, 1.0],           // Gray
        }
    }
}

/// A survey option for AskUserQuestion
#[derive(Debug, Clone)]
pub struct SurveyOption {
    /// Display label for this option
    pub label: String,
    /// Description explaining this option
    pub description: String,
}

impl SurveyOption {
    pub fn new(label: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: description.into(),
        }
    }
}

/// Response from a survey (sent back to caller)
#[derive(Debug, Clone)]
pub enum SurveyResponse {
    /// User selected specific option(s)
    Selected(Vec<usize>),
    /// User provided custom text
    Custom(String),
    /// User cancelled
    Cancelled,
}

/// Options for the project context menu
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectContextOption {
    /// Remove project from list (doesn't delete files)
    Remove,
    /// Change the assigned language
    ChangeLanguage,
    /// Archive/unarchive the project
    ToggleArchive,
    /// Cancel
    Cancel,
}

impl ProjectContextOption {
    pub fn all() -> &'static [ProjectContextOption] {
        &[
            ProjectContextOption::Remove,
            ProjectContextOption::ChangeLanguage,
            ProjectContextOption::ToggleArchive,
            ProjectContextOption::Cancel,
        ]
    }

    pub fn label(&self, is_archived: bool) -> &'static str {
        match self {
            ProjectContextOption::Remove => "Remove from list",
            ProjectContextOption::ChangeLanguage => "Change language",
            ProjectContextOption::ToggleArchive => {
                if is_archived { "Unarchive" } else { "Archive" }
            }
            ProjectContextOption::Cancel => "Cancel",
        }
    }
}

/// Current application state
#[derive(Debug, Clone)]
pub enum AppState {
    /// Project chooser screen - shown when launched from $HOME
    ProjectChooser {
        selected_index: usize,
        /// Show archived projects (All Projects view)
        show_archived: bool,
    },
    /// Active project view - shown when a project is selected
    ProjectView {
        project_path: PathBuf,
        selected_action: usize,
    },
    /// Palace Loop - AI suggestions streaming in as cards
    PalaceLoop {
        project_path: PathBuf,
        /// Cards with streaming suggestions
        cards: Vec<SuggestionCard>,
        /// Currently focused card index (keyboard/gamepad)
        focused_index: usize,
        /// Currently hovered card index (mouse)
        hovered_index: Option<usize>,
        /// Is the AI still generating suggestions?
        generating: bool,
        /// Tool call currently being displayed
        current_tool: Option<String>,
        /// Tool calls log (left side) - most recent first
        tool_log: Vec<String>,
        /// AI thoughts/commentary log (right side) - most recent first
        thought_log: Vec<String>,
        /// Scroll offset for logs (right thumbstick)
        log_scroll_offset: usize,
        /// Scroll offset for focused card detail panel (right thumbstick)
        detail_scroll_offset: f32,
        /// Max scroll for current focused card (calculated from content height)
        detail_max_scroll: f32,
        /// Scroll offset for card grid (mouse wheel / navigation)
        card_scroll_offset: f32,
    },
    /// Main menu - opened with Start button (Resume, Settings, Exit)
    MainMenu {
        selected_item: usize,
        /// Previous state to return to
        previous_state: Box<AppState>,
    },
    /// Settings submenu - opened from Main menu
    SettingsMenu {
        selected_item: usize,
        /// Previous state (MainMenu) to return to
        previous_state: Box<AppState>,
    },
    /// UI Scale submenu - opened from Settings menu
    UiScaleMenu {
        selected_item: usize,
        /// Previous state (SettingsMenu) to return to
        previous_state: Box<AppState>,
        /// Current user scale override (None = auto)
        user_scale_override: Option<f32>,
    },
    /// Permission modal - AI is waiting for user approval
    PermissionModal {
        /// The command requesting permission
        command: String,
        /// Command prefix for "always" approval
        command_prefix: String,
        /// Selected choice (0=Yes once, 1=Yes always, 2=No)
        selected_choice: usize,
        /// Previous state to return to
        previous_state: Box<AppState>,
    },
    /// Execute modal - options for executing selected cards
    ExecuteModal {
        /// Selected execution option
        selected_option: usize,
        /// Previous state to return to
        previous_state: Box<AppState>,
    },
    /// Add card menu - shown when "+" card is selected
    AddCardMenu {
        /// Selected option (0=Generate More, 1=Custom Task, 2=Cancel)
        selected_option: usize,
        /// Previous state to return to
        previous_state: Box<AppState>,
    },
    /// Custom task input - text entry for custom task (name and/or description)
    CustomTaskInput {
        /// Task name (title)
        name: String,
        /// Task description
        description: String,
        /// Which field is currently being edited (0=name, 1=description)
        active_field: usize,
        /// Cursor position in active field
        cursor: usize,
        /// Previous state to return to
        previous_state: Box<AppState>,
    },
    /// Project context menu - shown when X pressed on a project
    ProjectContextMenu {
        /// Index of the project being edited
        project_index: usize,
        /// Selected menu option
        selected_option: usize,
        /// Previous state to return to
        previous_state: Box<AppState>,
    },
    /// Language selector for a project
    LanguageSelector {
        /// Index of the project being edited
        project_index: usize,
        /// Selected language index
        selected_index: usize,
        /// Available languages
        languages: Vec<String>,
        /// Previous state to return to
        previous_state: Box<AppState>,
    },
    /// Executing - Claude/Z.ai is running the selected tasks
    Executing {
        /// Project path
        project_path: PathBuf,
        /// Cards being executed (selected subset)
        executing_cards: Vec<SuggestionCard>,
        /// All cards from the deck (for quest log view)
        all_cards: Vec<SuggestionCard>,
        /// Current execution status
        status: ExecutionStatus,
        /// Per-task status (synced from Claude's task_update tool)
        task_statuses: Vec<TaskStatus>,
        /// Left column: tool calls with timestamps (format: "[HH:MM:SS] icon action")
        tool_log: Vec<String>,
        /// Right column: Claude's commentary/thoughts with timestamps
        thought_log: Vec<String>,
        /// Scroll offset for logs (pixels)
        log_scroll_offset: f32,
        /// Which executor is running (Claude, ZAi, etc)
        executor: ExecuteOption,
        /// Previous state to return to (PalaceLoop)
        previous_state: Box<AppState>,
        /// Quest log view visible (Select toggles)
        quest_log_visible: bool,
        /// Focused card index in quest log view
        quest_log_focus: usize,
        /// Total tokens used in this execution
        tokens_used: u64,
        /// Whether a request is currently in flight
        request_active: bool,
    },
    /// Survey/Question UI - for Claude's AskUserQuestion tool
    Survey {
        /// The question being asked
        question: String,
        /// Short header label (e.g., "Auth method")
        header: String,
        /// Available options (up to 4 + custom)
        options: Vec<SurveyOption>,
        /// Currently focused option index
        focused_index: usize,
        /// Custom input text (when "Other" option is selected)
        custom_input: String,
        /// Whether custom input is active
        custom_active: bool,
        /// Allow multiple selections
        multi_select: bool,
        /// Selected option indices (for multi-select)
        selected_indices: Vec<usize>,
        /// Use quick-select face buttons (disabled when user navigates with dpad)
        use_quick_select: bool,
        /// Scroll offset for long option lists
        scroll_offset: usize,
        /// Previous state to return to
        previous_state: Box<AppState>,
        /// Channel to send response back
        response_tx: Option<std::sync::mpsc::Sender<SurveyResponse>>,
    },
    /// Multi-display dialog - shown when a new monitor is detected or from Settings
    /// Each monitor has an enable/disable toggle. Primary can be set on any enabled monitor.
    /// Gamepad: A = toggle enabled, X = set as primary
    /// Keyboard: Space = toggle enabled, P = set as primary
    /// Mouse: Click toggle or "Set Primary" button
    MultiDisplayDialog {
        /// Which monitor in the list is focused (for keyboard nav)
        focus_index: usize,
        /// Available monitors with enabled/primary state
        options: Vec<DisplayOption>,
        /// Remember this choice for future sessions
        remember_choice: bool,
        /// Which row is focused: 0=monitors, 1=remember toggle, 2=apply button
        focused_row: usize,
        /// Primary pill drag state: (source_monitor_index, cursor_x, cursor_y)
        primary_pill_drag: Option<(usize, f32, f32)>,
        /// Previous state to return to
        previous_state: Box<AppState>,
    },
    /// Simple dialog for newly connected monitor (hot-plug)
    /// Much simpler than MultiDisplayDialog - just "Extend to this monitor? Yes/No/Never"
    NewMonitorDialog {
        /// Name of the newly connected monitor
        monitor_name: String,
        /// Resolution of the monitor
        resolution: String,
        /// Focused option: 0=Yes, 1=No, 2=Never ask again
        focus_index: usize,
        /// Previous state to return to
        previous_state: Box<AppState>,
    },

    // ============== Scenario System States ==============

    /// Scenario diff viewer - shows corrections side-by-side
    /// Used by `palace scenario correct`
    ScenarioDiffViewer {
        /// Input file path
        input_path: PathBuf,
        /// Output file path
        output_path: Option<PathBuf>,
        /// The diff viewer state (handles scrolling, chunks, etc.)
        viewer: Box<crate::scenario::DiffViewer>,
        /// Whether we're in feedback mode
        feedback_mode: bool,
        /// Feedback options (generated by LLM)
        feedback_options: Vec<crate::scenario::FeedbackOption>,
        /// Selected feedback option
        selected_feedback: usize,
        /// Custom feedback text (when entering custom)
        custom_feedback: Option<String>,
        /// Cursor position in custom feedback
        custom_cursor: usize,
    },

    /// Scenario generator - recursive survey wizard
    /// Used by `palace scenario generate`
    ScenarioGenerator {
        /// The generator state machine
        generator: Box<crate::scenario::ScenarioGenerator>,
        /// Output file path
        output_path: Option<PathBuf>,
    },

    /// Recording mode - captures session as scenario
    /// Used by `palace scenario record`
    Recording {
        /// The underlying app state (what's actually being displayed)
        inner_state: Box<AppState>,
        /// The session recorder
        recorder: Box<crate::scenario::SessionRecorder>,
        /// Output file path
        output_path: Option<PathBuf>,
        /// Whether recording is paused
        paused: bool,
    },
}

/// Display option for multi-display dialog
#[derive(Debug, Clone)]
pub struct DisplayOption {
    /// Display name (e.g., "Monitor 1", "LG Ultrawide")
    pub name: String,
    /// Display identifier (for winit)
    pub id: String,
    /// Resolution string (e.g., "3440x1440")
    pub resolution: String,
    /// Is this monitor enabled (Palace will display on it)?
    pub enabled: bool,
    /// Is this the primary display?
    pub is_primary: bool,
}

impl DisplayOption {
    /// Create a new display option. enabled and is_primary are set separately.
    pub fn new(name: impl Into<String>, id: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            name: name.into(),
            id: id.into(),
            resolution: format!("{}x{}", width, height),
            enabled: false,
            is_primary: false,
        }
    }

    /// Builder: set enabled state
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Builder: set as primary (also enables)
    pub fn with_primary(mut self) -> Self {
        self.is_primary = true;
        self.enabled = true; // Primary must be enabled
        self
    }

    /// Label for display in the dialog
    pub fn label(&self) -> String {
        let status = if self.is_primary {
            "[Primary]"
        } else if self.enabled {
            "[Enabled]"
        } else {
            ""
        };
        if status.is_empty() {
            format!("{} ({})", self.name, self.resolution)
        } else {
            format!("{} ({}) {}", self.name, self.resolution, status)
        }
    }
}

/// Options for the "+" card menu (add new cards)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddCardOption {
    /// Generate more AI suggestions
    GenerateMore,
    /// Interview user then generate targeted suggestions
    InterviewMe,
    /// Enter custom task manually
    CustomTask,
    /// Cancel and go back
    Cancel,
}

impl AddCardOption {
    pub fn all() -> &'static [AddCardOption] {
        &[
            AddCardOption::GenerateMore,
            AddCardOption::InterviewMe,
            AddCardOption::CustomTask,
            AddCardOption::Cancel,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            AddCardOption::GenerateMore => "Generate more suggestions",
            AddCardOption::InterviewMe => "Interview me, then generate",
            AddCardOption::CustomTask => "Add custom task",
            AddCardOption::Cancel => "Cancel",
        }
    }
}

/// Execution options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecuteOption {
    /// Run with Claude Code CLI
    Claude,
    /// Run with Z.ai API
    ZAi,
    /// Run with Z.ai in turbo mode (streams between agents)
    ZAiTurbo,
}

impl ExecuteOption {
    pub fn all() -> &'static [ExecuteOption] {
        &[
            ExecuteOption::Claude,
            ExecuteOption::ZAi,
            ExecuteOption::ZAiTurbo,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            ExecuteOption::Claude => "Run with Claude",
            ExecuteOption::ZAi => "Run with Z.ai",
            ExecuteOption::ZAiTurbo => "Run with Z.ai (turbo)",
        }
    }
}

/// Execution status
#[derive(Debug, Clone)]
pub enum ExecutionStatus {
    /// Waiting to start
    Pending,
    /// Currently running
    Running {
        /// Current card index being executed
        current_card: usize,
        /// Total cards to execute
        total_cards: usize,
    },
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed(String),
    /// Cancelled by user
    Cancelled,
}

impl ExecutionStatus {
    pub fn is_running(&self) -> bool {
        matches!(self, ExecutionStatus::Running { .. })
    }

    pub fn is_done(&self) -> bool {
        matches!(self, ExecutionStatus::Completed | ExecutionStatus::Failed(_) | ExecutionStatus::Cancelled)
    }
}

/// Per-task execution status (for quest log badges)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TaskStatus {
    /// Task not yet started
    #[default]
    Pending,
    /// Task is currently being executed
    InProgress,
    /// Task completed successfully (by AI)
    Completed,
    /// Task verified by user
    Verified,
    /// Task blocked - needs attention
    Blocked,
    /// Task needs user review
    NeedsReview,
}

impl TaskStatus {
    /// Badge text for display
    pub fn badge_text(&self) -> &'static str {
        match self {
            TaskStatus::Pending => "PENDING",
            TaskStatus::InProgress => "IN PROGRESS",
            TaskStatus::Completed => "DONE",
            TaskStatus::Verified => "VERIFIED",
            TaskStatus::Blocked => "BLOCKED",
            TaskStatus::NeedsReview => "REVIEW",
        }
    }

    /// Badge color [R, G, B, A]
    pub fn badge_color(&self) -> [f32; 4] {
        match self {
            TaskStatus::Pending => [0.5, 0.5, 0.5, 0.7],       // Gray
            TaskStatus::InProgress => [0.4, 0.7, 1.0, 0.9],    // Blue
            TaskStatus::Completed => [0.2, 1.0, 0.4, 0.9],     // Green
            TaskStatus::Verified => [0.7, 0.4, 1.0, 0.9],      // Purple
            TaskStatus::Blocked => [1.0, 0.4, 0.3, 0.9],       // Red
            TaskStatus::NeedsReview => [1.0, 0.7, 0.2, 0.9],   // Orange/Yellow
        }
    }
}

/// Permission choice options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionChoice {
    YesOnce,
    YesAlways,
    No,
    SuggestElse,
}

impl PermissionChoice {
    pub fn all() -> &'static [PermissionChoice] {
        &[
            PermissionChoice::YesOnce,
            PermissionChoice::YesAlways,
            PermissionChoice::No,
            PermissionChoice::SuggestElse,
        ]
    }

    pub fn label(&self, prefix: &str) -> String {
        match self {
            PermissionChoice::YesOnce => "Yes (once)".to_string(),
            PermissionChoice::YesAlways => format!("Yes (always for '{}')", prefix),
            PermissionChoice::No => "No".to_string(),
            PermissionChoice::SuggestElse => "Suggest something else".to_string(),
        }
    }

    /// Short label for glyph hints
    pub fn short_label(&self) -> &'static str {
        match self {
            PermissionChoice::YesOnce => "Yes",
            PermissionChoice::YesAlways => "Always",
            PermissionChoice::No => "No",
            PermissionChoice::SuggestElse => "Suggest",
        }
    }
}

/// Response to a permission request (from UI to AI)
#[derive(Debug, Clone)]
pub enum PermissionResponse {
    /// Approved (run the command)
    Approved,
    /// Approved and remember for this prefix
    ApprovedAlways(String),
    /// Denied (don't run)
    Denied,
    /// Suggest alternatives (fork conversation, ask Z.ai for alternatives)
    SuggestElse {
        /// The original command that was rejected
        original_command: String,
    },
}

impl PermissionResponse {
    /// Whether this response approves the action
    pub fn is_approved(&self) -> bool {
        matches!(self, PermissionResponse::Approved | PermissionResponse::ApprovedAlways(_))
    }
}

impl AppState {
    pub fn project_chooser() -> Self {
        Self::ProjectChooser { selected_index: 0, show_archived: false }
    }

    pub fn project_view(path: PathBuf) -> Self {
        Self::ProjectView {
            project_path: path,
            selected_action: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============== Settings Tests ==============
    #[test]
    fn test_settings_default() {
        let settings = Settings::default();
        assert!(settings.dark_mode, "Default should be dark mode");
        assert!(settings.ui_scale.is_none(), "Default UI scale should be None (auto)");
    }

    // ============== MainMenuItem Tests ==============
    #[test]
    fn test_main_menu_items() {
        let items = MainMenuItem::all();
        assert_eq!(items.len(), 3, "Should have 3 main menu items");

        assert_eq!(MainMenuItem::Resume.label(), "Resume");
        assert_eq!(MainMenuItem::Settings.label(), "Settings");
        assert_eq!(MainMenuItem::Exit.label(), "Exit");
    }

    // ============== SettingsItem Tests ==============
    #[test]
    fn test_settings_items() {
        let items = SettingsItem::all();
        assert_eq!(items.len(), 3, "Should have 3 settings items");

        assert_eq!(SettingsItem::Display.label(), "Display");
        assert_eq!(SettingsItem::DarkMode.label(), "Dark Mode");
        assert_eq!(SettingsItem::UiScale.label(), "UI Scale");
    }

    // ============== UiScaleOption Tests ==============
    #[test]
    fn test_ui_scale_options() {
        let options = UiScaleOption::all();
        assert_eq!(options.len(), 7, "Should have 7 scale options");
    }

    #[test]
    fn test_ui_scale_labels() {
        assert_eq!(UiScaleOption::Auto.label(), "Auto-detect");
        assert_eq!(UiScaleOption::Scale50.label(), "50%");
        assert_eq!(UiScaleOption::Scale100.label(), "100%");
        assert_eq!(UiScaleOption::Scale150.label(), "150%");
        assert_eq!(UiScaleOption::Scale200.label(), "200%");
        assert_eq!(UiScaleOption::Scale300.label(), "300%");
        assert_eq!(UiScaleOption::Scale400.label(), "400%");
    }

    #[test]
    fn test_ui_scale_values() {
        assert_eq!(UiScaleOption::Auto.value(), None);
        assert_eq!(UiScaleOption::Scale50.value(), Some(0.5));
        assert_eq!(UiScaleOption::Scale100.value(), Some(1.0));
        assert_eq!(UiScaleOption::Scale150.value(), Some(1.5));
        assert_eq!(UiScaleOption::Scale200.value(), Some(2.0));
        assert_eq!(UiScaleOption::Scale300.value(), Some(3.0));
        assert_eq!(UiScaleOption::Scale400.value(), Some(4.0));
    }

    #[test]
    fn test_ui_scale_from_setting() {
        // None -> Auto
        assert_eq!(UiScaleOption::from_setting(None), UiScaleOption::Auto);

        // Specific values
        assert_eq!(UiScaleOption::from_setting(Some(0.5)), UiScaleOption::Scale50);
        assert_eq!(UiScaleOption::from_setting(Some(1.0)), UiScaleOption::Scale100);
        assert_eq!(UiScaleOption::from_setting(Some(1.5)), UiScaleOption::Scale150);
        assert_eq!(UiScaleOption::from_setting(Some(2.0)), UiScaleOption::Scale200);
        assert_eq!(UiScaleOption::from_setting(Some(3.0)), UiScaleOption::Scale300);

        // Boundary tests - should map to nearest option
        assert_eq!(UiScaleOption::from_setting(Some(0.6)), UiScaleOption::Scale50);
        assert_eq!(UiScaleOption::from_setting(Some(0.74)), UiScaleOption::Scale50);
        assert_eq!(UiScaleOption::from_setting(Some(0.75)), UiScaleOption::Scale100);
        assert_eq!(UiScaleOption::from_setting(Some(1.24)), UiScaleOption::Scale100);
        assert_eq!(UiScaleOption::from_setting(Some(1.25)), UiScaleOption::Scale150);
        assert_eq!(UiScaleOption::from_setting(Some(1.74)), UiScaleOption::Scale150);
        assert_eq!(UiScaleOption::from_setting(Some(1.75)), UiScaleOption::Scale200);
        assert_eq!(UiScaleOption::from_setting(Some(2.49)), UiScaleOption::Scale200);
        assert_eq!(UiScaleOption::from_setting(Some(2.5)), UiScaleOption::Scale300);
        assert_eq!(UiScaleOption::from_setting(Some(3.49)), UiScaleOption::Scale300);
        assert_eq!(UiScaleOption::from_setting(Some(3.5)), UiScaleOption::Scale400);
        assert_eq!(UiScaleOption::from_setting(Some(5.0)), UiScaleOption::Scale400);
    }

    // ============== ProjectAction Tests ==============
    #[test]
    fn test_project_actions() {
        let actions = ProjectAction::all();
        assert_eq!(actions.len(), 4, "Should have 4 project actions");
    }

    #[test]
    fn test_project_action_labels() {
        assert_eq!(ProjectAction::StartPalaceLoop.label(), "Start Palace Loop");
        assert_eq!(ProjectAction::Build.label(), "Build");
        assert_eq!(ProjectAction::Run.label(), "Run");
        assert_eq!(ProjectAction::ViewGitHistory.label(), "View Git History");
    }

    #[test]
    fn test_project_action_descriptions() {
        assert!(ProjectAction::StartPalaceLoop.description().contains("AI-assisted"));
        assert!(ProjectAction::Build.description().to_lowercase().contains("compile"));
        assert!(ProjectAction::Run.description().to_lowercase().contains("execute"));
        assert!(ProjectAction::ViewGitHistory.description().to_lowercase().contains("commit"));
    }

    // ============== SuggestionCard Tests ==============
    #[test]
    fn test_suggestion_card_new() {
        let card = SuggestionCard::new(42);
        assert_eq!(card.id, 42);
        assert!(card.title.is_empty());
        assert!(card.category.is_empty());
        assert!(card.description.is_empty());
        assert!(card.command.is_none());
        assert!(!card.selected);
        assert!(card.streaming);
    }

    #[test]
    fn test_suggestion_card_colors() {
        let mut card = SuggestionCard::new(0);

        // Test each category color
        card.category = "fix".to_string();
        assert_eq!(card.color(), [1.0, 0.4, 0.4, 1.0]);

        card.category = "test".to_string();
        assert_eq!(card.color(), [0.4, 0.8, 1.0, 1.0]);

        card.category = "build".to_string();
        assert_eq!(card.color(), [1.0, 0.7, 0.2, 1.0]);

        card.category = "refactor".to_string();
        assert_eq!(card.color(), [0.7, 0.5, 1.0, 1.0]);

        card.category = "docs".to_string();
        assert_eq!(card.color(), [0.5, 0.9, 0.5, 1.0]);

        card.category = "unknown".to_string();
        assert_eq!(card.color(), [0.5, 0.5, 0.6, 1.0]);
    }

    // ============== SurveyOption Tests ==============
    #[test]
    fn test_survey_option_new() {
        let option = SurveyOption::new("Test Label", "Test Description");
        assert_eq!(option.label, "Test Label");
        assert_eq!(option.description, "Test Description");
    }

    #[test]
    fn test_survey_option_with_string_conversions() {
        let option = SurveyOption::new(String::from("Label"), String::from("Desc"));
        assert_eq!(option.label, "Label");
        assert_eq!(option.description, "Desc");
    }

    // ============== ExecuteOption Tests ==============
    #[test]
    fn test_execute_options() {
        let options = ExecuteOption::all();
        assert_eq!(options.len(), 3, "Should have 3 execute options");
    }

    #[test]
    fn test_execute_option_labels() {
        assert_eq!(ExecuteOption::Claude.label(), "Run with Claude");
        assert_eq!(ExecuteOption::ZAi.label(), "Run with Z.ai");
        assert_eq!(ExecuteOption::ZAiTurbo.label(), "Run with Z.ai (turbo)");
    }

    // ============== ExecutionStatus Tests ==============
    #[test]
    fn test_execution_status_pending() {
        let status = ExecutionStatus::Pending;
        assert!(!status.is_running());
        assert!(!status.is_done());
    }

    #[test]
    fn test_execution_status_running() {
        let status = ExecutionStatus::Running {
            current_card: 2,
            total_cards: 5,
        };
        assert!(status.is_running());
        assert!(!status.is_done());
    }

    #[test]
    fn test_execution_status_completed() {
        let status = ExecutionStatus::Completed;
        assert!(!status.is_running());
        assert!(status.is_done());
    }

    #[test]
    fn test_execution_status_failed() {
        let status = ExecutionStatus::Failed("Test error".to_string());
        assert!(!status.is_running());
        assert!(status.is_done());
    }

    #[test]
    fn test_execution_status_cancelled() {
        let status = ExecutionStatus::Cancelled;
        assert!(!status.is_running());
        assert!(status.is_done());
    }

    // ============== PermissionChoice Tests ==============
    #[test]
    fn test_permission_choices() {
        let choices = PermissionChoice::all();
        assert_eq!(choices.len(), 4, "Should have 4 permission choices");
    }

    #[test]
    fn test_permission_choice_labels() {
        assert_eq!(PermissionChoice::YesOnce.label("prefix"), "Yes (once)");
        assert_eq!(
            PermissionChoice::YesAlways.label("cargo"),
            "Yes (always for 'cargo')"
        );
        assert_eq!(PermissionChoice::No.label("prefix"), "No");
        assert_eq!(
            PermissionChoice::SuggestElse.label("prefix"),
            "Suggest something else"
        );
    }

    #[test]
    fn test_permission_choice_short_labels() {
        assert_eq!(PermissionChoice::YesOnce.short_label(), "Yes");
        assert_eq!(PermissionChoice::YesAlways.short_label(), "Always");
        assert_eq!(PermissionChoice::No.short_label(), "No");
        assert_eq!(PermissionChoice::SuggestElse.short_label(), "Suggest");
    }

    // ============== PermissionResponse Tests ==============
    #[test]
    fn test_permission_response_approved() {
        let response = PermissionResponse::Approved;
        assert!(response.is_approved());
    }

    #[test]
    fn test_permission_response_approved_always() {
        let response = PermissionResponse::ApprovedAlways("cargo".to_string());
        assert!(response.is_approved());
    }

    #[test]
    fn test_permission_response_denied() {
        let response = PermissionResponse::Denied;
        assert!(!response.is_approved());
    }

    #[test]
    fn test_permission_response_suggest_else() {
        let response = PermissionResponse::SuggestElse {
            original_command: "rm -rf /".to_string(),
        };
        assert!(!response.is_approved());
    }

    // ============== AppState Tests ==============
    #[test]
    fn test_app_state_project_chooser() {
        let state = AppState::project_chooser();
        match state {
            AppState::ProjectChooser { selected_index, .. } => {
                assert_eq!(selected_index, 0);
            }
            _ => panic!("Expected ProjectChooser state"),
        }
    }

    #[test]
    fn test_app_state_project_view() {
        let path = PathBuf::from("/test/project");
        let state = AppState::project_view(path.clone());
        match state {
            AppState::ProjectView {
                project_path,
                selected_action,
            } => {
                assert_eq!(project_path, path);
                assert_eq!(selected_action, 0);
            }
            _ => panic!("Expected ProjectView state"),
        }
    }

    #[test]
    fn test_survey_response_variants() {
        // Selected variant
        let response = SurveyResponse::Selected(vec![0, 2]);
        match response {
            SurveyResponse::Selected(indices) => {
                assert_eq!(indices, vec![0, 2]);
            }
            _ => panic!("Expected Selected variant"),
        }

        // Custom variant
        let response = SurveyResponse::Custom("custom input".to_string());
        match response {
            SurveyResponse::Custom(text) => {
                assert_eq!(text, "custom input");
            }
            _ => panic!("Expected Custom variant"),
        }

        // Cancelled variant
        let response = SurveyResponse::Cancelled;
        match response {
            SurveyResponse::Cancelled => {}
            _ => panic!("Expected Cancelled variant"),
        }
    }
}
