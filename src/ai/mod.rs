// AI backend module - Anthropic SDK and OpenCode HTTP API

pub mod client;
pub mod streaming;

pub use client::{AnthropicClient, TaskSuggestion};
pub use streaming::{
    DialogueOption, ExecutionEvent, Executor, PermissionRequest, PermissionResponse, UserResponse,
};
