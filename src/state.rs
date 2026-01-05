use std::path::PathBuf;

/// Application settings (persisted)
#[derive(Debug, Clone)]
pub struct Settings {
    /// Dark mode enabled (true = dark, false = light)
    pub dark_mode: bool,
    /// UI scale setting (None = auto-detect, Some = user override)
    pub ui_scale: Option<f32>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            dark_mode: true, // Default to dark mode (OLED-friendly)
            ui_scale: None,  // Auto-detect by default
        }
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
    DarkMode,
    UiScale,
}

impl SettingsItem {
    pub fn all() -> &'static [SettingsItem] {
        &[SettingsItem::DarkMode, SettingsItem::UiScale]
    }

    pub fn label(&self) -> &'static str {
        match self {
            SettingsItem::DarkMode => "Dark Mode",
            SettingsItem::UiScale => "UI Scale",
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

/// Current application state
#[derive(Debug, Clone)]
pub enum AppState {
    /// Project chooser screen - shown when launched from $HOME
    ProjectChooser {
        selected_index: usize,
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
        /// Currently focused card index
        focused_index: usize,
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
        Self::ProjectChooser { selected_index: 0 }
    }

    pub fn project_view(path: PathBuf) -> Self {
        Self::ProjectView {
            project_path: path,
            selected_action: 0,
        }
    }
}
