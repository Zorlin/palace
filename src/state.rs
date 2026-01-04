use std::path::PathBuf;

/// Application settings (persisted)
#[derive(Debug, Clone)]
pub struct Settings {
    /// Dark mode enabled (true = dark, false = light)
    pub dark_mode: bool,
    /// UI scale factor (1.0 = 100%, 1.5 = 150%, etc.)
    pub ui_scale: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            dark_mode: true, // Default to dark mode (OLED-friendly)
            ui_scale: 1.0,
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
