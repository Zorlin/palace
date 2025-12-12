//! 1b Classifier Integration
//!
//! A tiny local model (runs on CPU, ~1GB RAM) that manages context windows.
//! The classifier takes context block summaries and outputs add/remove instructions.
//!
//! Input format:
//! ```text
//! [1] Current file: palace.py (router code)
//! [2] Recent error: TypeError in streaming
//! [3] User goal: Fix streaming bug
//! [4] Previous fix attempt: Added model lookup
//! [5] Unrelated: Yesterday's kubernetes work
//! ```
//!
//! Output format:
//! ```text
//! ++1,2,3,4--5
//! ```

use crate::delta::Delta;
use serde::{Deserialize, Serialize};

/// Classifier configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    /// Model name (e.g., "phi-2", "qwen2.5-1.5b")
    pub model: String,
    /// Ollama host URL
    pub ollama_host: String,
    /// Maximum tokens for response
    pub max_tokens: u32,
    /// Temperature (low for consistency)
    pub temperature: f32,
    /// System prompt for the classifier
    pub system_prompt: String,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            model: std::env::var("CLASSIFIER_MODEL")
                .unwrap_or_else(|_| "hf.co/unsloth/Qwen3-4B-GGUF".to_string()),
            // Use the Ollama instance at 10.7.1.135
            ollama_host: std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://10.7.1.135:11434".to_string()),
            max_tokens: 50,
            temperature: 0.1,
            system_prompt: CLASSIFIER_SYSTEM_PROMPT.to_string(),
        }
    }
}

/// Default system prompt for the context classifier
pub const CLASSIFIER_SYSTEM_PROMPT: &str = r#"You are a context manager for an AI assistant. Your ONLY job is to decide which context blocks should be active.

You will receive a list of context blocks with IDs and summaries. Each block has a type and description.

Based on relevance to the current task, output which blocks to ADD or REMOVE from the active context.

RULES:
1. Keep blocks that are directly relevant to the current task
2. Remove blocks that are unrelated or outdated
3. Prefer recent errors, current files, and user goals
4. Remove unrelated conversations or completed tasks

OUTPUT FORMAT: Only output a delta string like: ++1,2,3--4,5
- ++N adds block N to active context
- --N removes block N from active context
- Multiple IDs separated by commas
- No spaces, no explanation

EXAMPLES:
Input: [1]* CurrentFile: main.rs, [2] Error: TypeError, [3] Unrelated: old kubernetes work
Output: ++1,2--3

Input: [1]* UserGoal: Fix auth bug, [2] TestOutput: All passing, [3] Related: auth module
Output: ++1,3--2
"#;

/// Classifier client for interacting with local 1b model
pub struct ClassifierClient {
    config: ClassifierConfig,
}

impl ClassifierClient {
    /// Create a new classifier client
    pub fn new(config: ClassifierConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default_client() -> Self {
        Self::new(ClassifierConfig::default())
    }

    /// Classify context blocks and return a delta
    pub async fn classify(&self, input: &str) -> Result<Delta, ClassifierError> {
        let response = self.call_ollama(input).await?;
        crate::delta::parse_delta(&response).map_err(|e| ClassifierError::ParseError(e.to_string()))
    }

    /// Call Ollama API for classification
    async fn call_ollama(&self, input: &str) -> Result<String, ClassifierError> {
        let client = reqwest::Client::new();

        let request_body = serde_json::json!({
            "model": self.config.model,
            "prompt": input,
            "system": self.config.system_prompt,
            "stream": false,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        });

        let response = client
            .post(format!("{}/api/generate", self.config.ollama_host))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| ClassifierError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(ClassifierError::ApiError(format!(
                "Ollama returned status: {}",
                response.status()
            )));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ClassifierError::ParseError(e.to_string()))?;

        body["response"]
            .as_str()
            .map(|s| s.trim().to_string())
            .ok_or_else(|| ClassifierError::ParseError("No response field in Ollama output".to_string()))
    }

    /// Batch classify - run multiple classifications in parallel
    pub async fn batch_classify(&self, inputs: Vec<String>) -> Vec<Result<Delta, ClassifierError>> {
        let futures: Vec<_> = inputs.iter().map(|input| self.classify(input)).collect();
        futures::future::join_all(futures).await
    }
}

/// Errors from the classifier
#[derive(Debug)]
pub enum ClassifierError {
    NetworkError(String),
    ApiError(String),
    ParseError(String),
    ModelNotFound(String),
}

impl std::fmt::Display for ClassifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClassifierError::NetworkError(s) => write!(f, "Network error: {}", s),
            ClassifierError::ApiError(s) => write!(f, "API error: {}", s),
            ClassifierError::ParseError(s) => write!(f, "Parse error: {}", s),
            ClassifierError::ModelNotFound(s) => write!(f, "Model not found: {}", s),
        }
    }
}

impl std::error::Error for ClassifierError {}

/// Mock classifier for testing (no network calls)
pub struct MockClassifier {
    /// Canned responses for testing
    responses: Vec<String>,
    current: std::sync::atomic::AtomicUsize,
}

impl MockClassifier {
    /// Create a mock classifier with predefined responses
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            current: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Get next delta (cycles through responses)
    pub fn classify(&self, _input: &str) -> Result<Delta, ClassifierError> {
        let idx = self.current.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let response = &self.responses[idx % self.responses.len()];
        crate::delta::parse_delta(response).map_err(|e| ClassifierError::ParseError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ClassifierConfig::default();
        assert_eq!(config.model, "hf.co/unsloth/Qwen3-4B-GGUF");
        assert!(config.temperature < 0.5); // Low for consistency
    }

    #[test]
    fn test_mock_classifier() {
        let mock = MockClassifier::new(vec!["++1,2--3".to_string(), "++4--5,6".to_string()]);

        let delta1 = mock.classify("test input 1").unwrap();
        assert_eq!(delta1.add, vec![1, 2]);
        assert_eq!(delta1.remove, vec![3]);

        let delta2 = mock.classify("test input 2").unwrap();
        assert_eq!(delta2.add, vec![4]);
        assert_eq!(delta2.remove, vec![5, 6]);

        // Cycles back
        let delta3 = mock.classify("test input 3").unwrap();
        assert_eq!(delta3.add, vec![1, 2]);
    }
}
