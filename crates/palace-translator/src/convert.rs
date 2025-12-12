//! Conversion functions between Anthropic and OpenAI formats

use palace_core::{anthropic, openai};
use std::collections::HashMap;

/// Check if a tool ID is already in Mistral-compatible format (9 alphanumeric chars)
pub fn is_valid_mistral_id(id: &str) -> bool {
    id.len() == 9 && id.chars().all(|c| c.is_ascii_alphanumeric())
}

/// Normalize a tool ID for Mistral compatibility.
/// - If already valid (9 alphanumeric), return as-is
/// - Otherwise, generate a deterministic 9-char hash
pub fn shorten_tool_id(original_id: &str) -> String {
    // If already valid, don't modify
    if is_valid_mistral_id(original_id) {
        return original_id.to_string();
    }

    // Use a simple hash-based approach to generate a 9-char alphanumeric ID
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    original_id.hash(&mut hasher);
    let hash = hasher.finish();

    // Convert to base62 (a-zA-Z0-9)
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut result = String::with_capacity(9);
    let mut n = hash;
    for _ in 0..9 {
        result.push(CHARS[(n % 62) as usize] as char);
        n /= 62;
    }
    result
}

/// Bidirectional tool ID mapper for translating between Anthropic and OpenAI formats.
///
/// When converting Anthropic→OpenAI:
/// - Long Claude IDs (toolu_...) get shortened to 9-char
/// - Short IDs pass through unchanged
/// - Mapping is recorded for reverse lookup
///
/// When converting OpenAI→Anthropic:
/// - If we have a mapping, restore original long ID
/// - Otherwise use ID as-is
#[derive(Default, Clone, Debug)]
pub struct ToolIdMapper {
    /// Maps short_id -> original_id (for restoring backend IDs)
    short_to_long: HashMap<String, String>,
    /// Maps original_id -> short_id (for consistent shortening)
    long_to_short: HashMap<String, String>,
}

impl ToolIdMapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a short ID for a long ID
    pub fn to_short(&mut self, original_id: &str) -> String {
        // If already short/valid, return as-is
        if is_valid_mistral_id(original_id) {
            return original_id.to_string();
        }

        // Check cache
        if let Some(short) = self.long_to_short.get(original_id) {
            return short.clone();
        }

        // Generate and cache
        let short = shorten_tool_id(original_id);
        self.long_to_short.insert(original_id.to_string(), short.clone());
        self.short_to_long.insert(short.clone(), original_id.to_string());
        short
    }

    /// Get the original long ID if we have a mapping.
    /// If not found AND the ID is too long for Mistral, shorten it.
    /// This handles Claude-generated tool IDs (toolu_xxx) that we never saw before.
    pub fn to_long(&self, short_id: &str) -> String {
        if let Some(long_id) = self.short_to_long.get(short_id) {
            return long_id.clone();
        }
        // ID not in our mapper - if it's already Mistral-compatible, use as-is
        // Otherwise shorten it (handles Claude's own toolu_xxx IDs)
        if is_valid_mistral_id(short_id) {
            short_id.to_string()
        } else {
            shorten_tool_id(short_id)
        }
    }

    /// Register a mapping (used when we see a new ID from OpenAI response)
    pub fn register(&mut self, short_id: String, long_id: String) {
        if short_id != long_id {
            self.short_to_long.insert(short_id.clone(), long_id.clone());
            self.long_to_short.insert(long_id, short_id);
        }
    }
}

// ============================================================================
// Anthropic -> OpenAI conversions
// ============================================================================

/// Convert a full Anthropic request to OpenAI messages
/// Prepends system message if provided
/// Returns (messages, id_mapping) where id_mapping maps short IDs back to original IDs
pub fn anthropic_request_to_openai(
    messages: &[anthropic::Message],
    system: Option<&str>,
) -> (Vec<openai::Message>, HashMap<String, String>) {
    anthropic_request_to_openai_with_mapper(messages, system, None)
}

/// Convert a full Anthropic request to OpenAI messages, using an optional mapper
/// for proper round-trip ID translation.
///
/// When a mapper is provided:
/// - tool_use IDs are shortened AND recorded in the mapper
/// - tool_result IDs are looked up in the mapper to restore the original backend ID
///
/// This ensures that when Claude sends back a tool_result with the ID we gave it,
/// we can restore the original backend ID that the backend expects.
pub fn anthropic_request_to_openai_with_mapper(
    messages: &[anthropic::Message],
    system: Option<&str>,
    mapper: Option<&ToolIdMapper>,
) -> (Vec<openai::Message>, HashMap<String, String>) {
    let mut result = Vec::new();
    let mut id_mapping: HashMap<String, String> = HashMap::new();

    // Add system message first if present
    if let Some(sys) = system {
        result.push(openai::Message {
            role: openai::Role::System,
            content: Some(openai::Content::Text(sys.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    // Convert each message, collecting ID mappings
    for msg in messages {
        let (msgs, mapping) = anthropic_message_to_openai_messages_with_mapper(msg, mapper);
        result.extend(msgs);
        id_mapping.extend(mapping);
    }

    (result, id_mapping)
}

/// Convert a single Anthropic message to OpenAI messages with optional mapper
fn anthropic_message_to_openai_messages_with_mapper(
    msg: &anthropic::Message,
    mapper: Option<&ToolIdMapper>,
) -> (Vec<openai::Message>, HashMap<String, String>) {
    let mut id_mapping: HashMap<String, String> = HashMap::new();

    match &msg.content {
        anthropic::Content::Text(text) => {
            (vec![openai::Message {
                role: anthropic_role_to_openai(msg.role),
                content: Some(openai::Content::Text(text.clone())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }], id_mapping)
        }
        anthropic::Content::Blocks(blocks) => {
            let mut messages = Vec::new();
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();
            let mut tool_results = Vec::new();

            for block in blocks {
                match block {
                    anthropic::ContentBlock::Text { text } => {
                        text_parts.push(text.clone());
                    }
                    anthropic::ContentBlock::Thinking { thinking, .. } => {
                        text_parts.push(format!("<thinking>{}</thinking>", thinking));
                    }
                    anthropic::ContentBlock::ToolUse { id, name, input } => {
                        let short_id = shorten_tool_id(id);
                        id_mapping.insert(short_id.clone(), id.clone());
                        tool_calls.push(openai::ToolCall {
                            id: short_id,
                            call_type: Some("function".to_string()),
                            function: openai::FunctionCall {
                                name: name.clone(),
                                arguments: serde_json::to_string(input).unwrap_or_default(),
                            },
                            index: None,
                        });
                    }
                    anthropic::ContentBlock::ToolResult { tool_use_id, content, .. } => {
                        // Use mapper to restore original backend ID if available
                        let backend_id = if let Some(m) = mapper {
                            m.to_long(tool_use_id)
                        } else {
                            shorten_tool_id(tool_use_id)
                        };
                        tool_results.push((backend_id, tool_result_to_string(content)));
                    }
                    anthropic::ContentBlock::Image { .. } => {
                        // TODO: Handle image conversion
                    }
                }
            }

            // If we have tool results, each becomes a separate message
            if !tool_results.is_empty() {
                for (backend_id, content) in tool_results {
                    messages.push(openai::Message {
                        role: openai::Role::Tool,
                        content: Some(openai::Content::Text(content)),
                        tool_calls: None,
                        tool_call_id: Some(backend_id),
                        name: None,
                    });
                }
            } else {
                // Regular message with optional text and tool calls
                let content = if text_parts.is_empty() {
                    None
                } else {
                    Some(openai::Content::Text(text_parts.join("\n\n")))
                };

                let tool_calls_opt = if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                };

                messages.push(openai::Message {
                    role: anthropic_role_to_openai(msg.role),
                    content,
                    tool_calls: tool_calls_opt,
                    tool_call_id: None,
                    name: None,
                });
            }

            (messages, id_mapping)
        }
        anthropic::Content::Unknown(v) => {
            let text = if let Some(s) = v.as_str() {
                s.to_string()
            } else {
                format!("[Provider-specific content: {}]",
                    serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string()))
            };
            (vec![openai::Message {
                role: anthropic_role_to_openai(msg.role),
                content: Some(openai::Content::Text(text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }], id_mapping)
        }
    }
}

/// Convert a single Anthropic message to one or more OpenAI messages
/// Multiple tool_results become multiple OpenAI messages
pub fn anthropic_message_to_openai_messages(msg: &anthropic::Message) -> Vec<openai::Message> {
    anthropic_message_to_openai_messages_with_mapping(msg).0
}

/// Convert a single Anthropic message to OpenAI messages WITH ID mapping
/// Returns (messages, id_mapping) where id_mapping maps short_id -> original_id
pub fn anthropic_message_to_openai_messages_with_mapping(msg: &anthropic::Message) -> (Vec<openai::Message>, HashMap<String, String>) {
    let mut id_mapping: HashMap<String, String> = HashMap::new();

    match &msg.content {
        anthropic::Content::Text(text) => {
            (vec![openai::Message {
                role: anthropic_role_to_openai(msg.role),
                content: Some(openai::Content::Text(text.clone())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }], id_mapping)
        }
        anthropic::Content::Blocks(blocks) => {
            let mut messages = Vec::new();
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();
            let mut tool_results = Vec::new();

            for block in blocks {
                match block {
                    anthropic::ContentBlock::Text { text } => {
                        text_parts.push(text.clone());
                    }
                    anthropic::ContentBlock::Thinking { thinking, .. } => {
                        // Include thinking blocks as text prefixed with marker
                        text_parts.push(format!("<thinking>{}</thinking>", thinking));
                    }
                    anthropic::ContentBlock::ToolUse { id, name, input } => {
                        // Shorten the ID for OpenAI/Mistral compatibility
                        let short_id = shorten_tool_id(id);
                        id_mapping.insert(short_id.clone(), id.clone());
                        tool_calls.push(openai::ToolCall {
                            id: short_id,
                            call_type: Some("function".to_string()),
                            function: openai::FunctionCall {
                                name: name.clone(),
                                arguments: serde_json::to_string(input).unwrap_or_default(),
                            },
                            index: None,
                        });
                    }
                    anthropic::ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        // Use shortened ID for tool results too
                        let short_id = shorten_tool_id(tool_use_id);
                        id_mapping.insert(short_id.clone(), tool_use_id.clone());
                        tool_results.push((short_id, tool_result_to_string(content)));
                    }
                    anthropic::ContentBlock::Image { .. } => {
                        // TODO: Handle image conversion
                    }
                }
            }

            // If we have tool results, each becomes a separate message
            if !tool_results.is_empty() {
                for (short_id, content) in tool_results {
                    messages.push(openai::Message {
                        role: openai::Role::Tool,
                        content: Some(openai::Content::Text(content)),
                        tool_calls: None,
                        tool_call_id: Some(short_id),
                        name: None,
                    });
                }
            } else {
                // Regular message with optional text and tool calls
                let content = if text_parts.is_empty() {
                    None
                } else {
                    // Join with newlines so thinking blocks are separated from text
                    Some(openai::Content::Text(text_parts.join("\n\n")))
                };

                let tool_calls_opt = if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                };

                messages.push(openai::Message {
                    role: anthropic_role_to_openai(msg.role),
                    content,
                    tool_calls: tool_calls_opt,
                    tool_call_id: None,
                    name: None,
                });
            }

            (messages, id_mapping)
        }
        anthropic::Content::Unknown(v) => {
            // Unknown content (e.g., GLM's webReader output) - convert to text
            let text = if let Some(s) = v.as_str() {
                s.to_string()
            } else {
                format!("[Provider-specific content: {}]",
                    serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string()))
            };
            (vec![openai::Message {
                role: anthropic_role_to_openai(msg.role),
                content: Some(openai::Content::Text(text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }], id_mapping)
        }
    }
}

/// Convert a single Anthropic message to a single OpenAI message
/// For simple cases where we know there's only one message
pub fn anthropic_message_to_openai(msg: &anthropic::Message) -> openai::Message {
    anthropic_message_to_openai_messages(msg)
        .into_iter()
        .next()
        .unwrap_or(openai::Message {
            role: anthropic_role_to_openai(msg.role),
            content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        })
}

fn anthropic_role_to_openai(role: anthropic::Role) -> openai::Role {
    match role {
        anthropic::Role::User => openai::Role::User,
        anthropic::Role::Assistant => openai::Role::Assistant,
    }
}

fn tool_result_to_string(content: &anthropic::ToolResultContent) -> String {
    match content {
        anthropic::ToolResultContent::Text(s) => s.clone(),
        anthropic::ToolResultContent::Blocks(blocks) => {
            blocks
                .iter()
                .filter_map(|b| match b {
                    anthropic::ToolResultBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        }
    }
}

/// Convert Anthropic tools to OpenAI format
pub fn anthropic_tools_to_openai(tools: &[anthropic::Tool]) -> Vec<openai::Tool> {
    tools
        .iter()
        .map(|t| openai::Tool {
            tool_type: "function".to_string(),
            function: openai::FunctionDef {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect()
}

// ============================================================================
// OpenAI -> Anthropic conversions
// ============================================================================

/// Convert OpenAI message content to Anthropic content blocks
pub fn openai_message_to_anthropic_content(msg: &openai::Message) -> Vec<anthropic::ContentBlock> {
    openai_message_to_anthropic_content_with_mapper(msg, None).0
}

/// Convert OpenAI message content to Anthropic content blocks, with optional ID mapping
/// Returns (blocks, id_mapping) where id_mapping maps shortened_id -> original_backend_id
pub fn openai_message_to_anthropic_content_with_mapper(
    msg: &openai::Message,
    mut mapper: Option<&mut ToolIdMapper>,
) -> (Vec<anthropic::ContentBlock>, HashMap<String, String>) {
    let mut blocks = Vec::new();
    let mut id_mapping = HashMap::new();

    // Add text content if present and non-empty
    if let Some(content) = &msg.content {
        let text = match content {
            openai::Content::Text(s) => s.clone(),
            openai::Content::Parts(parts) => {
                parts
                    .iter()
                    .filter_map(|p| match p {
                        openai::ContentPart::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("")
            }
        };
        if !text.is_empty() {
            blocks.push(anthropic::ContentBlock::Text { text });
        }
    }

    // Add tool calls as tool_use blocks
    if let Some(tool_calls) = &msg.tool_calls {
        for tc in tool_calls {
            let input: serde_json::Value =
                serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));

            // Shorten ID and store mapping for round-trip
            let backend_id = tc.id.clone();
            let short_id = match mapper.as_mut() {
                Some(m) => m.to_short(&backend_id),
                None => shorten_tool_id(&backend_id),
            };

            // Store the mapping so we can restore original backend ID later
            if short_id != backend_id {
                id_mapping.insert(short_id.clone(), backend_id);
            }

            blocks.push(anthropic::ContentBlock::ToolUse {
                id: short_id,
                name: tc.function.name.clone(),
                input,
            });
        }
    }

    (blocks, id_mapping)
}

/// Convert OpenAI finish reason to Anthropic stop reason
pub fn openai_finish_reason_to_anthropic(reason: openai::FinishReason) -> anthropic::StopReason {
    match reason {
        openai::FinishReason::Stop => anthropic::StopReason::EndTurn,
        openai::FinishReason::Length => anthropic::StopReason::MaxTokens,
        openai::FinishReason::ToolCalls => anthropic::StopReason::ToolUse,
        openai::FinishReason::ContentFilter => anthropic::StopReason::EndTurn,
    }
}

// ============================================================================
// Streaming conversions
// ============================================================================

/// Convert an OpenAI chunk delta to an Anthropic content delta
pub fn openai_chunk_delta_to_anthropic(delta: &openai::ChunkDelta) -> Option<anthropic::ContentDelta> {
    if let Some(content) = &delta.content {
        if !content.is_empty() {
            return Some(anthropic::ContentDelta::TextDelta {
                text: content.clone(),
            });
        }
    }
    None
}

/// Extract tool call start info from a chunk (index, id, name)
pub fn extract_tool_call_start(delta: &openai::ChunkDelta) -> Option<(usize, String, String)> {
    if let Some(tool_calls) = &delta.tool_calls {
        for tc in tool_calls {
            if let (Some(id), Some(func)) = (&tc.id, &tc.function) {
                if let Some(name) = &func.name {
                    return Some((tc.index, id.clone(), name.clone()));
                }
            }
        }
    }
    None
}

/// Extract tool call arguments from a chunk for a specific index
pub fn extract_tool_call_args(delta: &openai::ChunkDelta, index: usize) -> Option<String> {
    if let Some(tool_calls) = &delta.tool_calls {
        for tc in tool_calls {
            if tc.index == index {
                if let Some(func) = &tc.function {
                    if let Some(args) = &func.arguments {
                        return Some(args.clone());
                    }
                }
            }
        }
    }
    None
}
