//! API Translator: Anthropic format → OpenAI format → Backend → Anthropic format
//!
//! Handles streaming translation for Claude Code compatibility with Mistral/Ollama backends.

use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response, Sse},
    Json,
};
use futures::stream::{self, Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Bidirectional ID mapping for tool calls across providers
#[derive(Default)]
pub struct ToolIdMap {
    /// source_id -> target_id
    forward: HashMap<String, String>,
    /// target_id -> source_id
    reverse: HashMap<String, String>,
}

impl ToolIdMap {
    /// Store a bidirectional mapping
    pub fn insert(&mut self, source_id: String, target_id: String) {
        self.reverse.insert(target_id.clone(), source_id.clone());
        self.forward.insert(source_id, target_id);
    }

    /// Get target ID from source ID
    pub fn get_target(&self, source_id: &str) -> Option<&String> {
        self.forward.get(source_id)
    }

    /// Get source ID from target ID (reverse lookup)
    pub fn get_source(&self, target_id: &str) -> Option<&String> {
        self.reverse.get(target_id)
    }
}

/// Shared state for the translator
#[derive(Clone)]
pub struct TranslatorState {
    pub client: Client,
    pub backend_url: String,
    pub backend_api_key: String,
    pub backend_model: String,
    pub model_registry: Arc<RwLock<ModelRegistry>>,
    /// Tool ID mapping for consistent round-trips (anthropic <-> backend)
    pub tool_id_map: Arc<RwLock<ToolIdMap>>,
}

/// Model registry for multi-model routing
#[derive(Default)]
pub struct ModelRegistry {
    pub models: std::collections::HashMap<String, ModelConfig>,
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub model_id: String,
    pub base_url: String,
    pub api_key_env: String,
    pub format: ApiFormat,
}

#[derive(Clone, Debug, Default)]
pub enum ApiFormat {
    #[default]
    OpenAI,
    Anthropic,
}

// ============================================================================
// Anthropic Request/Response types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum SystemPrompt {
    String(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    String(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: Option<String>,
        is_error: Option<bool>,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

// ============================================================================
// OpenAI Request/Response types
// ============================================================================

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OpenAIFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIToolDef,
}

#[derive(Debug, Serialize)]
struct OpenAIToolDef {
    name: String,
    description: String,
    parameters: Value,
}

// ============================================================================
// Translation Logic
// ============================================================================

impl TranslatorState {
    pub fn new(backend_url: String, backend_api_key: String, backend_model: String) -> Self {
        Self {
            client: Client::new(),
            backend_url,
            backend_api_key,
            backend_model,
            model_registry: Arc::new(RwLock::new(ModelRegistry::default())),
            tool_id_map: Arc::new(RwLock::new(ToolIdMap::default())),
        }
    }

    /// Generate a stable backend-compatible ID from an Anthropic tool ID
    /// Mistral/OpenAI need 9-char alphanumeric IDs
    fn get_or_create_backend_id(&self, anthropic_id: &str) -> String {
        // Check if we already have a mapping
        {
            let map = self.tool_id_map.read();
            if let Some(backend_id) = map.get_target(anthropic_id) {
                return backend_id.clone();
            }
        }

        // Generate a deterministic 9-char ID using simple hash
        // This ensures the same anthropic_id always maps to the same backend_id
        let mut hash: u64 = 0;
        for (i, byte) in anthropic_id.bytes().enumerate() {
            hash = hash.wrapping_add((byte as u64).wrapping_mul(31_u64.wrapping_pow(i as u32)));
        }
        let backend_id = format!("{:09x}", hash % 0xFFFFFFFFFF).chars().take(9).collect::<String>();

        // Store the bidirectional mapping
        {
            let mut map = self.tool_id_map.write();
            map.insert(anthropic_id.to_string(), backend_id.clone());
        }

        backend_id
    }

    /// Look up the original anthropic ID from a backend ID (reverse lookup)
    fn get_anthropic_id(&self, backend_id: &str) -> Option<String> {
        let map = self.tool_id_map.read();
        map.get_source(backend_id).cloned()
    }

    /// Translate Anthropic request to OpenAI format
    fn translate_request(&self, req: &AnthropicRequest) -> OpenAIRequest {
        let mut messages = Vec::new();

        // Handle system prompt
        if let Some(system) = &req.system {
            let system_text = match system {
                SystemPrompt::String(s) => s.clone(),
                SystemPrompt::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|b| b.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(system_text),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Translate messages
        for msg in &req.messages {
            match &msg.content {
                MessageContent::String(s) => {
                    messages.push(OpenAIMessage {
                        role: msg.role.clone(),
                        content: Some(s.clone()),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                MessageContent::Blocks(blocks) => {
                    let mut text_parts = Vec::new();
                    let mut tool_calls = Vec::new();
                    let mut tool_results = Vec::new();

                    for block in blocks {
                        match block {
                            ContentBlock::Text { text } => {
                                text_parts.push(text.clone());
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                // Use bidirectional mapping for consistent ID round-trips
                                let backend_id = self.get_or_create_backend_id(id);

                                tool_calls.push(OpenAIToolCall {
                                    id: backend_id,
                                    call_type: "function".to_string(),
                                    function: OpenAIFunction {
                                        name: name.clone(),
                                        arguments: serde_json::to_string(input).unwrap_or_default(),
                                    },
                                });
                            }
                            ContentBlock::ToolResult { tool_use_id, content, .. } => {
                                tool_results.push((tool_use_id.clone(), content.clone()));
                            }
                        }
                    }

                    // Add assistant message with text and/or tool calls
                    if msg.role == "assistant" {
                        let content = if text_parts.is_empty() {
                            None
                        } else {
                            Some(text_parts.join("\n"))
                        };

                        messages.push(OpenAIMessage {
                            role: "assistant".to_string(),
                            content,
                            tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                            tool_call_id: None,
                        });
                    } else if msg.role == "user" {
                        // User messages: text content or tool results
                        if !text_parts.is_empty() {
                            messages.push(OpenAIMessage {
                                role: "user".to_string(),
                                content: Some(text_parts.join("\n")),
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        }

                        // Tool results become "tool" role messages
                        for (tool_id, content) in tool_results {
                            // Look up the backend ID that corresponds to this anthropic ID
                            let backend_id = self.get_or_create_backend_id(&tool_id);
                            messages.push(OpenAIMessage {
                                role: "tool".to_string(),
                                content,
                                tool_calls: None,
                                tool_call_id: Some(backend_id),
                            });
                        }
                    }
                }
            }
        }

        // Translate tools
        let tools = req.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| OpenAITool {
                    tool_type: "function".to_string(),
                    function: OpenAIToolDef {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.input_schema.clone(),
                    },
                })
                .collect()
        });

        OpenAIRequest {
            model: self.backend_model.clone(),
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: req.stream.unwrap_or(true),
            tools,
            tool_choice: if req.tools.is_some() { Some("auto".to_string()) } else { None },
        }
    }
}

// ============================================================================
// SSE Event types for Anthropic streaming format
// ============================================================================

#[derive(Debug, Serialize)]
struct MessageStartEvent {
    #[serde(rename = "type")]
    event_type: String,
    message: MessageInfo,
}

#[derive(Debug, Serialize)]
struct MessageInfo {
    id: String,
    #[serde(rename = "type")]
    msg_type: String,
    role: String,
    content: Vec<Value>,
    model: String,
    stop_reason: Option<String>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ContentBlockStart {
    #[serde(rename = "type")]
    event_type: String,
    index: usize,
    content_block: ContentBlockInfo,
}

#[derive(Debug, Serialize)]
struct ContentBlockInfo {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<Value>,
}

#[derive(Debug, Serialize)]
struct ContentBlockDelta {
    #[serde(rename = "type")]
    event_type: String,
    index: usize,
    delta: DeltaInfo,
}

#[derive(Debug, Serialize)]
struct DeltaInfo {
    #[serde(rename = "type")]
    delta_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    partial_json: Option<String>,
}

#[derive(Debug, Serialize)]
struct ContentBlockStop {
    #[serde(rename = "type")]
    event_type: String,
    index: usize,
}

#[derive(Debug, Serialize)]
struct MessageDelta {
    #[serde(rename = "type")]
    event_type: String,
    delta: MessageDeltaInfo,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct MessageDeltaInfo {
    stop_reason: String,
}

#[derive(Debug, Serialize)]
struct MessageStop {
    #[serde(rename = "type")]
    event_type: String,
}

// ============================================================================
// Handler
// ============================================================================

/// Main handler for /v1/messages endpoint
pub async fn handle_messages(
    State(state): State<Arc<TranslatorState>>,
    Json(req): Json<AnthropicRequest>,
) -> Response {
    let stream = req.stream.unwrap_or(true);

    tracing::info!("📨 /v1/messages request: model={}, stream={}, messages={}",
        req.model, stream, req.messages.len());

    if stream {
        handle_streaming(state, req).await
    } else {
        handle_non_streaming(state, req).await
    }
}

async fn handle_non_streaming(
    state: Arc<TranslatorState>,
    req: AnthropicRequest,
) -> Response {
    let openai_req = state.translate_request(&req);

    let response = match state.client
        .post(format!("{}/chat/completions", state.backend_url))
        .header("Authorization", format!("Bearer {}", state.backend_api_key))
        .header("Content-Type", "application/json")
        .json(&openai_req)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (StatusCode::BAD_GATEWAY, Json(json!({
                "type": "error",
                "error": {"type": "api_error", "message": e.to_string()}
            }))).into_response();
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return (StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY), Json(json!({
            "type": "error",
            "error": {"type": "api_error", "message": body}
        }))).into_response();
    }

    let openai_response: Value = match response.json().await {
        Ok(v) => v,
        Err(e) => {
            return (StatusCode::BAD_GATEWAY, Json(json!({
                "type": "error",
                "error": {"type": "api_error", "message": e.to_string()}
            }))).into_response();
        }
    };

    // Translate OpenAI response to Anthropic format
    let anthropic_response = translate_openai_response(&openai_response, &state.backend_model, &state);
    Json(anthropic_response).into_response()
}

fn translate_openai_response(openai: &Value, model: &str, state: &TranslatorState) -> Value {
    let choice = &openai["choices"][0];
    let message = &choice["message"];
    let finish_reason = choice["finish_reason"].as_str().unwrap_or("end_turn");

    let mut content = Vec::new();

    // Add text content
    if let Some(text) = message["content"].as_str() {
        if !text.is_empty() {
            content.push(json!({"type": "text", "text": text}));
        }
    }

    // Add tool calls - map backend IDs to anthropic format and store mapping
    if let Some(tool_calls) = message["tool_calls"].as_array() {
        for tc in tool_calls {
            let backend_id = tc["id"].as_str().unwrap_or("");
            // Generate anthropic-style ID and store bidirectional mapping
            let anthropic_id = format!("toolu_{}", Uuid::new_v4().to_string().replace("-", "")[..24].to_string());

            // Store the mapping: anthropic_id <-> backend_id
            // This allows tool_result with anthropic_id to find the backend_id
            {
                let mut map = state.tool_id_map.write();
                map.insert(anthropic_id.clone(), backend_id.to_string());
            }

            content.push(json!({
                "type": "tool_use",
                "id": anthropic_id,
                "name": tc["function"]["name"],
                "input": serde_json::from_str::<Value>(
                    tc["function"]["arguments"].as_str().unwrap_or("{}")
                ).unwrap_or(json!({}))
            }));
        }
    }

    let stop_reason = match finish_reason {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    };

    json!({
        "id": format!("msg_{}", Uuid::new_v4().to_string().replace("-", "")[..24].to_string()),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": openai["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            "output_tokens": openai["usage"]["completion_tokens"].as_u64().unwrap_or(0)
        }
    })
}

async fn handle_streaming(
    state: Arc<TranslatorState>,
    req: AnthropicRequest,
) -> Response {
    let openai_req = state.translate_request(&req);
    let model = state.backend_model.clone();
    let msg_id = format!("msg_{}", &Uuid::new_v4().to_string().replace("-", "")[..24]);

    // Build SSE response
    let stream = async_stream::stream! {
        // Send message_start immediately
        let start = MessageStartEvent {
            event_type: "message_start".to_string(),
            message: MessageInfo {
                id: msg_id.clone(),
                msg_type: "message".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                model: model.clone(),
                stop_reason: None,
                usage: Usage { input_tokens: 0, output_tokens: 0 },
            },
        };
        yield Ok::<_, std::convert::Infallible>(
            axum::response::sse::Event::default()
                .event("message_start")
                .data(serde_json::to_string(&start).unwrap())
        );

        // Make request to backend
        let response = match state.client
            .post(format!("{}/chat/completions", state.backend_url))
            .header("Authorization", format!("Bearer {}", state.backend_api_key))
            .header("Content-Type", "application/json")
            .json(&openai_req)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let error = json!({"type": "error", "error": {"type": "api_error", "message": e.to_string()}});
                yield Ok(axum::response::sse::Event::default()
                    .event("error")
                    .data(serde_json::to_string(&error).unwrap()));
                return;
            }
        };

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            tracing::error!("Backend error: {}", body);
            let error = json!({"type": "error", "error": {"type": "api_error", "message": body}});
            yield Ok(axum::response::sse::Event::default()
                .event("error")
                .data(serde_json::to_string(&error).unwrap()));
            return;
        }

        // Process streaming response
        let mut current_block_index: i32 = -1;
        let mut text_block_started = false;
        let mut tool_blocks: std::collections::HashMap<usize, (usize, String)> = std::collections::HashMap::new();
        let mut finish_reason = "end_turn".to_string();

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!("Stream error: {}", e);
                    break;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete lines
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..];
                if data == "[DONE]" {
                    break;
                }

                let chunk: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let choice = &chunk["choices"][0];
                let delta = &choice["delta"];

                if let Some(fr) = choice["finish_reason"].as_str() {
                    finish_reason = fr.to_string();
                }

                // Handle text content
                if let Some(content) = delta["content"].as_str() {
                    if !content.is_empty() {
                        // Start text block if needed
                        if !text_block_started {
                            current_block_index += 1;
                            let block_start = ContentBlockStart {
                                event_type: "content_block_start".to_string(),
                                index: current_block_index as usize,
                                content_block: ContentBlockInfo {
                                    block_type: "text".to_string(),
                                    text: Some("".to_string()),
                                    id: None,
                                    name: None,
                                    input: None,
                                },
                            };
                            yield Ok(axum::response::sse::Event::default()
                                .event("content_block_start")
                                .data(serde_json::to_string(&block_start).unwrap()));
                            text_block_started = true;
                        }

                        // Send text delta
                        let delta_event = ContentBlockDelta {
                            event_type: "content_block_delta".to_string(),
                            index: current_block_index as usize,
                            delta: DeltaInfo {
                                delta_type: "text_delta".to_string(),
                                text: Some(content.to_string()),
                                partial_json: None,
                            },
                        };
                        yield Ok(axum::response::sse::Event::default()
                            .event("content_block_delta")
                            .data(serde_json::to_string(&delta_event).unwrap()));
                    }
                }

                // Handle tool calls
                if let Some(tool_calls) = delta["tool_calls"].as_array() {
                    for tc in tool_calls {
                        let tc_index = tc["index"].as_u64().unwrap_or(0) as usize;
                        let tc_id = tc["id"].as_str();
                        let func = &tc["function"];
                        let func_name = func["name"].as_str();
                        let func_args = func["arguments"].as_str().unwrap_or("");

                        if !tool_blocks.contains_key(&tc_index) {
                            // Close text block if open
                            if text_block_started {
                                let stop = ContentBlockStop {
                                    event_type: "content_block_stop".to_string(),
                                    index: current_block_index as usize,
                                };
                                yield Ok(axum::response::sse::Event::default()
                                    .event("content_block_stop")
                                    .data(serde_json::to_string(&stop).unwrap()));
                                text_block_started = false;
                            }

                            // Start new tool_use block
                            current_block_index += 1;
                            let anthropic_id = format!("toolu_{}", &Uuid::new_v4().to_string().replace("-", "")[..24]);

                            // Store bidirectional mapping: anthropic_id <-> backend_id
                            // This allows tool_result with anthropic_id to find the backend_id
                            if let Some(backend_id) = tc_id {
                                let mut map = state.tool_id_map.write();
                                map.insert(anthropic_id.clone(), backend_id.to_string());
                            }

                            tool_blocks.insert(tc_index, (current_block_index as usize, anthropic_id.clone()));

                            let block_start = ContentBlockStart {
                                event_type: "content_block_start".to_string(),
                                index: current_block_index as usize,
                                content_block: ContentBlockInfo {
                                    block_type: "tool_use".to_string(),
                                    text: None,
                                    id: Some(anthropic_id),
                                    name: func_name.map(|s| s.to_string()),
                                    input: Some(json!({})),
                                },
                            };
                            yield Ok(axum::response::sse::Event::default()
                                .event("content_block_start")
                                .data(serde_json::to_string(&block_start).unwrap()));
                        }

                        // Send argument delta
                        if !func_args.is_empty() {
                            if let Some((block_idx, _)) = tool_blocks.get(&tc_index) {
                                let delta_event = ContentBlockDelta {
                                    event_type: "content_block_delta".to_string(),
                                    index: *block_idx,
                                    delta: DeltaInfo {
                                        delta_type: "input_json_delta".to_string(),
                                        text: None,
                                        partial_json: Some(func_args.to_string()),
                                    },
                                };
                                yield Ok(axum::response::sse::Event::default()
                                    .event("content_block_delta")
                                    .data(serde_json::to_string(&delta_event).unwrap()));
                            }
                        }
                    }
                }
            }
        }

        // Close any open text block
        if text_block_started {
            let stop = ContentBlockStop {
                event_type: "content_block_stop".to_string(),
                index: current_block_index as usize,
            };
            yield Ok(axum::response::sse::Event::default()
                .event("content_block_stop")
                .data(serde_json::to_string(&stop).unwrap()));
        }

        // Close any open tool blocks
        for (_, (block_idx, _)) in &tool_blocks {
            let stop = ContentBlockStop {
                event_type: "content_block_stop".to_string(),
                index: *block_idx,
            };
            yield Ok(axum::response::sse::Event::default()
                .event("content_block_stop")
                .data(serde_json::to_string(&stop).unwrap()));
        }

        // Determine stop reason
        let stop_reason = match finish_reason.as_str() {
            "tool_calls" => "tool_use",
            "length" => "max_tokens",
            _ => "end_turn",
        };

        // Send message_delta
        let msg_delta = MessageDelta {
            event_type: "message_delta".to_string(),
            delta: MessageDeltaInfo {
                stop_reason: stop_reason.to_string(),
            },
            usage: Usage { input_tokens: 0, output_tokens: 0 },
        };
        yield Ok(axum::response::sse::Event::default()
            .event("message_delta")
            .data(serde_json::to_string(&msg_delta).unwrap()));

        // Send message_stop
        let msg_stop = MessageStop {
            event_type: "message_stop".to_string(),
        };
        yield Ok(axum::response::sse::Event::default()
            .event("message_stop")
            .data(serde_json::to_string(&msg_stop).unwrap()));
    };

    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}
