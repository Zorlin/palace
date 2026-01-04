use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Once;

#[link(name = "anthropic")]
extern "C" {
    fn anthropic_init_with_base(api_key: *const c_char, base_url: *const c_char) -> i32;
    fn anthropic_message(
        model: *const c_char,
        system_prompt: *const c_char,
        user_message: *const c_char,
        max_tokens: i32,
    ) -> *mut c_char;
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
