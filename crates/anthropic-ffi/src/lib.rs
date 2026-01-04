use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::sync::Once;

/// Event type constants (must match Go side)
pub const EVENT_TEXT: c_int = 0;
pub const EVENT_TOOL_USE_START: c_int = 1;
pub const EVENT_TOOL_USE_INPUT: c_int = 2;
pub const EVENT_TOOL_USE_END: c_int = 3;
pub const EVENT_THINKING: c_int = 4;
pub const EVENT_DONE: c_int = 5;
pub const EVENT_ERROR: c_int = 6;
pub const EVENT_TOOL_RESULT: c_int = 7;

/// Callback type for streaming events
/// event_type: one of EVENT_* constants
/// data: event-specific data (text chunk, tool name, JSON input, etc.)
pub type StreamCallback = extern "C" fn(event_type: c_int, data: *const c_char);

/// Callback type for tool execution
/// Returns a C string with the tool result (caller must free)
pub type ToolExecutor = extern "C" fn(tool_name: *const c_char, tool_input: *const c_char) -> *mut c_char;

#[link(name = "anthropic")]
extern "C" {
    fn anthropic_init_with_base(api_key: *const c_char, base_url: *const c_char) -> i32;
    fn anthropic_message(
        model: *const c_char,
        system_prompt: *const c_char,
        user_message: *const c_char,
        max_tokens: i32,
    ) -> *mut c_char;
    fn anthropic_message_stream(
        model: *const c_char,
        system_prompt: *const c_char,
        user_message: *const c_char,
        max_tokens: i32,
        callback: StreamCallback,
    );
    fn anthropic_agentic_loop(
        model: *const c_char,
        system_prompt: *const c_char,
        user_message: *const c_char,
        max_tokens: i32,
        tools_json: *const c_char,
        callback: StreamCallback,
        tool_executor: ToolExecutor,
    );
    fn anthropic_free_string(s: *mut c_char);
}

static INIT: Once = Once::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageResponse {
    pub success: bool,
    pub content: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub usage: Usage,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: i64,
    pub output_tokens: i64,
}

#[derive(Debug, thiserror::Error)]
pub enum AnthropicError {
    #[error("Failed to initialize client: {0}")]
    InitError(String),
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Invalid string: {0}")]
    StringError(#[from] std::ffi::NulError),
    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Anthropic API client using Go SDK via FFI
pub struct Client {
    _private: (),
}

impl Client {
    /// Create a new client with the given API key
    /// If api_key is None, checks environment variables in order:
    /// ANTHROPIC_API_KEY, ZAI_API_KEY, OPENAI_API_KEY
    pub fn new(api_key: Option<&str>) -> Result<Self, AnthropicError> {
        let mut result = Ok(());

        INIT.call_once(|| {
            // Determine API key and base URL from env
            let (key, base_url) = if let Some(k) = api_key {
                (Some(k.to_string()), std::env::var("ANTHROPIC_BASE_URL").ok())
            } else if let Ok(k) = std::env::var("ANTHROPIC_API_KEY") {
                (Some(k), std::env::var("ANTHROPIC_BASE_URL").ok())
            } else if let Ok(k) = std::env::var("ZAI_API_KEY") {
                let base = std::env::var("ZAI_BASE_URL")
                    .unwrap_or_else(|_| "https://api.z.ai/api/anthropic".to_string());
                (Some(k), Some(base))
            } else if let Ok(k) = std::env::var("OPENAI_API_KEY") {
                (Some(k), std::env::var("OPENAI_API_BASE").ok())
            } else {
                (None, None)
            };

            let key_cstr = key
                .as_ref()
                .map(|s| CString::new(s.as_str()).unwrap())
                .unwrap_or_else(|| CString::new("").unwrap());

            let base_cstr = base_url
                .as_ref()
                .map(|s| CString::new(s.as_str()).unwrap());

            let base_ptr = base_cstr
                .as_ref()
                .map(|s| s.as_ptr())
                .unwrap_or(std::ptr::null());

            let ret = unsafe { anthropic_init_with_base(key_cstr.as_ptr(), base_ptr) };
            if ret != 0 {
                result = Err(AnthropicError::InitError(
                    "Failed to initialize Anthropic client. Check API key.".to_string(),
                ));
            }
        });

        result?;
        Ok(Self { _private: () })
    }

    /// Send a message to Claude
    pub fn message(
        &self,
        model: &str,
        system_prompt: Option<&str>,
        user_message: &str,
        max_tokens: i32,
    ) -> Result<MessageResponse, AnthropicError> {
        let model_c = CString::new(model)?;
        let system_c = CString::new(system_prompt.unwrap_or(""))?;
        let user_c = CString::new(user_message)?;

        let response_ptr = unsafe {
            anthropic_message(
                model_c.as_ptr(),
                system_c.as_ptr(),
                user_c.as_ptr(),
                max_tokens,
            )
        };

        if response_ptr.is_null() {
            return Err(AnthropicError::ApiError("Null response from API".to_string()));
        }

        let response_str = unsafe {
            let s = CStr::from_ptr(response_ptr).to_string_lossy().into_owned();
            anthropic_free_string(response_ptr);
            s
        };

        let response: MessageResponse = serde_json::from_str(&response_str)?;

        if !response.success {
            return Err(AnthropicError::ApiError(
                response.error.unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }

        Ok(response)
    }

    /// Send a streaming message to Claude
    /// The callback is called for each text chunk, then once with is_done=true
    pub fn message_stream<F>(
        &self,
        model: &str,
        system_prompt: Option<&str>,
        user_message: &str,
        max_tokens: i32,
        mut on_chunk: F,
    ) -> Result<(), AnthropicError>
    where
        F: FnMut(StreamEvent) + 'static,
    {
        use std::sync::{Arc, Mutex};

        let model_c = CString::new(model)?;
        let system_c = CString::new(system_prompt.unwrap_or(""))?;
        let user_c = CString::new(user_message)?;

        // Store callback in thread-local storage for FFI access
        let callback_box: Arc<Mutex<Option<Box<dyn FnMut(StreamEvent)>>>> =
            Arc::new(Mutex::new(Some(Box::new(on_chunk))));

        // Store in thread-local for the callback to access
        STREAM_CALLBACK.with(|cell| {
            *cell.borrow_mut() = Some(callback_box.clone());
        });

        unsafe {
            anthropic_message_stream(
                model_c.as_ptr(),
                system_c.as_ptr(),
                user_c.as_ptr(),
                max_tokens,
                stream_callback_handler,
            );
        }

        // Clean up
        STREAM_CALLBACK.with(|cell| {
            *cell.borrow_mut() = None;
        });

        Ok(())
    }

    /// Run an agentic loop with tool execution
    /// The tool_executor is called for each tool use, returns the result
    pub fn agentic_loop<F, T>(
        &self,
        model: &str,
        system_prompt: Option<&str>,
        user_message: &str,
        max_tokens: i32,
        tools_json: &str,
        on_event: F,
        tool_executor: T,
    ) -> Result<(), AnthropicError>
    where
        F: FnMut(StreamEvent) + 'static,
        T: Fn(&str, &str) -> String + 'static,
    {
        let model_c = CString::new(model)?;
        let system_c = CString::new(system_prompt.unwrap_or(""))?;
        let user_c = CString::new(user_message)?;
        let tools_c = CString::new(tools_json)?;

        // Store callbacks in thread-local storage
        let event_box: Arc<Mutex<Option<Box<dyn FnMut(StreamEvent)>>>> =
            Arc::new(Mutex::new(Some(Box::new(on_event))));
        let tool_box: Arc<Mutex<Option<Box<dyn Fn(&str, &str) -> String>>>> =
            Arc::new(Mutex::new(Some(Box::new(tool_executor))));

        STREAM_CALLBACK.with(|cell| {
            *cell.borrow_mut() = Some(event_box.clone());
        });
        TOOL_EXECUTOR.with(|cell| {
            *cell.borrow_mut() = Some(tool_box.clone());
        });

        unsafe {
            anthropic_agentic_loop(
                model_c.as_ptr(),
                system_c.as_ptr(),
                user_c.as_ptr(),
                max_tokens,
                tools_c.as_ptr(),
                stream_callback_handler,
                tool_executor_handler,
            );
        }

        // Clean up
        STREAM_CALLBACK.with(|cell| {
            *cell.borrow_mut() = None;
        });
        TOOL_EXECUTOR.with(|cell| {
            *cell.borrow_mut() = None;
        });

        Ok(())
    }
}

/// Stream event types
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Text chunk received
    Text(String),
    /// Tool use started (contains "id:name" or just "name")
    ToolUseStart(String),
    /// Tool input JSON chunk
    ToolUseInput(String),
    /// Tool use ended
    ToolUseEnd,
    /// Thinking text
    Thinking(String),
    /// Stream completed successfully
    Done,
    /// Error occurred
    Error(String),
    /// Tool result (truncated preview)
    ToolResult(String),
}

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

thread_local! {
    static STREAM_CALLBACK: RefCell<Option<Arc<Mutex<Option<Box<dyn FnMut(StreamEvent)>>>>>> = RefCell::new(None);
    static TOOL_EXECUTOR: RefCell<Option<Arc<Mutex<Option<Box<dyn Fn(&str, &str) -> String>>>>>> = RefCell::new(None);
}

/// FFI callback handler for tool execution
extern "C" fn tool_executor_handler(tool_name: *const c_char, tool_input: *const c_char) -> *mut c_char {
    let name = if !tool_name.is_null() {
        unsafe { CStr::from_ptr(tool_name).to_string_lossy().into_owned() }
    } else {
        return std::ptr::null_mut();
    };

    let input = if !tool_input.is_null() {
        unsafe { CStr::from_ptr(tool_input).to_string_lossy().into_owned() }
    } else {
        String::new()
    };

    let result = TOOL_EXECUTOR.with(|cell| {
        if let Some(ref executor_arc) = *cell.borrow() {
            if let Ok(guard) = executor_arc.lock() {
                if let Some(ref executor) = *guard {
                    return executor(&name, &input);
                }
            }
        }
        format!("Error: Tool executor not available")
    });

    // Return as C string (Go will free it)
    CString::new(result).map(|s| s.into_raw()).unwrap_or(std::ptr::null_mut())
}

/// FFI callback handler that forwards to the Rust closure
extern "C" fn stream_callback_handler(event_type: c_int, data: *const c_char) {
    STREAM_CALLBACK.with(|cell| {
        if let Some(ref callback_arc) = *cell.borrow() {
            if let Ok(mut guard) = callback_arc.lock() {
                if let Some(ref mut callback) = *guard {
                    let data_str = if !data.is_null() {
                        unsafe { CStr::from_ptr(data).to_string_lossy().into_owned() }
                    } else {
                        String::new()
                    };

                    let event = match event_type {
                        EVENT_TEXT => StreamEvent::Text(data_str),
                        EVENT_TOOL_USE_START => StreamEvent::ToolUseStart(data_str),
                        EVENT_TOOL_USE_INPUT => StreamEvent::ToolUseInput(data_str),
                        EVENT_TOOL_USE_END => StreamEvent::ToolUseEnd,
                        EVENT_THINKING => StreamEvent::Thinking(data_str),
                        EVENT_DONE => StreamEvent::Done,
                        EVENT_ERROR => StreamEvent::Error(data_str),
                        EVENT_TOOL_RESULT => StreamEvent::ToolResult(data_str),
                        _ => return, // Unknown event type
                    };

                    callback(event);
                }
            }
        }
    });
}

/// Model constants
pub mod models {
    pub const CLAUDE_OPUS_4_5: &str = "claude-opus-4-5-20251101";
    pub const CLAUDE_SONNET_4_5: &str = "claude-sonnet-4-5";
    pub const CLAUDE_HAIKU_4_5: &str = "claude-haiku-4-5";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_constants() {
        assert!(models::CLAUDE_SONNET_4_5.contains("sonnet"));
    }
}
