//! Palace - GPU-native recursive hierarchical self-improvement environment
//!
//! This library exposes the public API for integration testing.

// Public modules for testing
pub mod state;
pub mod persistence;
pub mod scenario;
pub mod fixtures;

// Re-export commonly used types from state
pub use state::{
    AppState, DisplayOption, ExecuteOption, ExecutionStatus, MainMenuItem, PermissionChoice,
    PermissionResponse, ProjectAction, SettingsItem, SuggestionCard, SurveyOption,
    SurveyResponse, TaskStatus, UiScaleOption,
};

// Re-export from persistence
pub use persistence::{PalaceDB, Task};

// Re-export scenario types
pub use scenario::{
    DiffViewer, DiffKind, DiffChunk, FeedbackOption, ScenarioCorrector, CorrectorState,
    ScenarioGenerator, GeneratorState, SessionRecorder, RecorderConfig, RecordedEvent,
};

// Note: app module cannot be included in lib due to winit/wgpu dependencies
// The app.rs tests are embedded in the module itself using #[cfg(test)]
