//! Integration tests for the translator

#[cfg(test)]
mod strip_thinking_tests {
    use palace_core::anthropic::{Content, ContentBlock, Message, Role};

    /// Strip thinking blocks that lack valid signatures (cross-model compatibility)
    fn strip_invalid_thinking_blocks(messages: &mut Vec<Message>) {
        for msg in messages.iter_mut() {
            if let Content::Blocks(blocks) = &mut msg.content {
                blocks.retain(|block| {
                    match block {
                        ContentBlock::Thinking { signature, .. } => !signature.is_empty(),
                        _ => true,
                    }
                });
            }
        }
    }

    #[test]
    fn test_glm_thinking_block_gets_stripped() {
        // Simulate conversation history with GLM's thinking block (no signature)
        let mut messages = vec![
            Message {
                role: Role::User,
                content: Content::Text("Hello".to_string()),
            },
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![
                    ContentBlock::Thinking {
                        thinking: "Let me think about this...".to_string(),
                        signature: "".to_string(), // GLM doesn't send signatures
                    },
                    ContentBlock::Text {
                        text: "Hello! How can I help?".to_string(),
                    },
                ]),
            },
            Message {
                role: Role::User,
                content: Content::Text("Switch to Claude".to_string()),
            },
        ];

        strip_invalid_thinking_blocks(&mut messages);

        // The thinking block should be stripped
        if let Content::Blocks(blocks) = &messages[1].content {
            assert_eq!(blocks.len(), 1, "Should only have text block left");
            assert!(matches!(blocks[0], ContentBlock::Text { .. }));
        } else {
            panic!("Expected Blocks content");
        }
    }

    #[test]
    fn test_claude_thinking_block_preserved() {
        // Simulate conversation with Claude's thinking block (has signature)
        let mut messages = vec![
            Message {
                role: Role::User,
                content: Content::Text("Hello".to_string()),
            },
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![
                    ContentBlock::Thinking {
                        thinking: "Let me think about this...".to_string(),
                        signature: "ErUFD3PwT...valid_signature".to_string(),
                    },
                    ContentBlock::Text {
                        text: "Hello! How can I help?".to_string(),
                    },
                ]),
            },
        ];

        strip_invalid_thinking_blocks(&mut messages);

        // The thinking block should be preserved
        if let Content::Blocks(blocks) = &messages[1].content {
            assert_eq!(blocks.len(), 2, "Both blocks should remain");
            assert!(matches!(blocks[0], ContentBlock::Thinking { .. }));
        } else {
            panic!("Expected Blocks content");
        }
    }

    #[test]
    fn test_deserialize_glm_thinking_without_signature() {
        use serde_json::json;

        // This is what GLM sends back - thinking without signature
        let glm_response = json!({
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "The user wants to know about Rust."
                },
                {
                    "type": "text",
                    "text": "Rust is a systems programming language."
                }
            ]
        });

        let msg: Message = serde_json::from_value(glm_response).expect("Should deserialize");
        
        if let Content::Blocks(blocks) = &msg.content {
            assert_eq!(blocks.len(), 2);
            
            // First block should be Thinking with empty signature
            match &blocks[0] {
                ContentBlock::Thinking { thinking, signature } => {
                    assert_eq!(thinking, "The user wants to know about Rust.");
                    assert_eq!(signature, "", "Signature should default to empty");
                }
                _ => panic!("Expected Thinking block"),
            }
        } else {
            panic!("Expected Blocks content");
        }
    }

    #[test]
    fn test_full_flow_glm_to_claude() {
        use serde_json::json;

        // 1. Deserialize a GLM response (no signature)
        let glm_response = json!({
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Processing..."
                },
                {
                    "type": "text", 
                    "text": "Here's my answer"
                }
            ]
        });

        let msg: Message = serde_json::from_value(glm_response).expect("Should deserialize");
        let mut messages = vec![msg];

        // 2. Strip invalid thinking blocks before sending to Claude
        strip_invalid_thinking_blocks(&mut messages);

        // 3. Serialize for Claude - should not have thinking block
        let serialized = serde_json::to_value(&messages[0]).expect("Should serialize");
        let content = serialized.get("content").expect("Should have content");
        let blocks = content.as_array().expect("Content should be array");
        
        assert_eq!(blocks.len(), 1, "Only text block should remain");
        assert_eq!(blocks[0].get("type").unwrap(), "text");
    }
}

    #[test]
    fn test_actual_claude_code_payload() {
        use serde_json::json;
        use palace_core::anthropic::MessagesRequest;
        
        // This is what Claude Code actually sends when switching from GLM to Claude
        // The history contains GLM's thinking blocks
        let payload = json!({
            "model": "claude-opus-4-5-20251101",
            "max_tokens": 16000,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello @switch=glm GLM"
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "The user is greeting me and asking me to switch to GLM mode."
                        },
                        {
                            "type": "text",
                            "text": "Hello! I'm here as GLM. What can I help you with?"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": "Now @switch=claude Claude please"
                }
            ],
            "stream": true
        });
        
        // 1. Deserialize the request
        let request: MessagesRequest = serde_json::from_value(payload).expect("Should deserialize");
        
        // Check we got the thinking block
        if let palace_core::anthropic::Content::Blocks(blocks) = &request.messages[1].content {
            assert_eq!(blocks.len(), 2, "Should have 2 blocks before strip");
            match &blocks[0] {
                palace_core::anthropic::ContentBlock::Thinking { signature, .. } => {
                    assert_eq!(signature, "", "GLM thinking should have empty signature");
                }
                _ => panic!("Expected Thinking block"),
            }
        }
        
        // 2. Clone and strip (like the daemon does)
        let mut capped_request = request.clone();
        
        for msg in capped_request.messages.iter_mut() {
            if let palace_core::anthropic::Content::Blocks(blocks) = &mut msg.content {
                blocks.retain(|block| {
                    match block {
                        palace_core::anthropic::ContentBlock::Thinking { signature, .. } => !signature.is_empty(),
                        _ => true,
                    }
                });
            }
        }
        
        // 3. Verify thinking was stripped
        if let palace_core::anthropic::Content::Blocks(blocks) = &capped_request.messages[1].content {
            assert_eq!(blocks.len(), 1, "Should only have text block after strip");
            assert!(matches!(blocks[0], palace_core::anthropic::ContentBlock::Text { .. }));
        }
        
        // 4. Serialize and verify no thinking block
        let serialized = serde_json::to_value(&capped_request).expect("Should serialize");
        let msg1_content = &serialized["messages"][1]["content"];
        let blocks = msg1_content.as_array().expect("Should be array");
        assert_eq!(blocks.len(), 1, "Serialized should only have 1 block");
        assert_eq!(blocks[0]["type"], "text");
    }

    #[test]
    fn test_thinking_and_text_blocks_have_separator() {
        // When converting Anthropic -> OpenAI, thinking and text blocks
        // should be separated with newlines, not concatenated directly.
        // This prevents "testing.From" (no space) issues.
        use crate::convert::anthropic_message_to_openai;
        use palace_core::anthropic::{Content, ContentBlock, Message, Role};
        use palace_core::openai;

        let msg = Message {
            role: Role::Assistant,
            content: Content::Blocks(vec![
                ContentBlock::Thinking {
                    thinking: "The user is testing.".to_string(),
                    signature: "sig".to_string(),
                },
                ContentBlock::Text {
                    text: "From my perspective, everything is fine.".to_string(),
                },
            ]),
        };

        let openai_msg = anthropic_message_to_openai(&msg);

        if let Some(openai::Content::Text(text)) = &openai_msg.content {
            // Should have separator between thinking and text
            assert!(
                !text.contains("testing.From"),
                "Thinking and text should be separated, not concatenated. Got: {}",
                text
            );
            assert!(
                text.contains("testing.") && text.contains("From my perspective"),
                "Both thinking and text content should be present. Got: {}",
                text
            );
        } else {
            panic!("Expected Text content, got {:?}", openai_msg.content);
        }
    }

    #[test]
    fn test_claude_code_injects_thinking_blocks() {
        // Claude Code may inject thinking blocks into history even when
        // the model (like GLM) didn't produce them. These won't have signatures.
        use serde_json::json;
        use palace_core::anthropic::MessagesRequest;

        // Simulate: User had extended thinking ON, talked to GLM, now switching to Claude
        // Claude Code injects a thinking block reconstruction
        let payload = json!({
            "model": "claude-opus-4-5-20251101",
            "max_tokens": 16000,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 10000
            },
            "messages": [
                {
                    "role": "user",
                    "content": "Hello GLM"
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Claude Code reconstructed this thinking block"
                            // NO signature field at all - Claude Code doesn't have one to inject
                        },
                        {
                            "type": "text",
                            "text": "Hello! I'm GLM."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": "@switch=claude Hi Claude!"
                }
            ],
            "stream": true
        });

        let request: MessagesRequest = serde_json::from_value(payload)
            .expect("Should deserialize - this is what Claude Code sends");

        // Verify thinking block exists with empty signature
        if let palace_core::anthropic::Content::Blocks(blocks) = &request.messages[1].content {
            match &blocks[0] {
                palace_core::anthropic::ContentBlock::Thinking { signature, .. } => {
                    assert!(signature.is_empty(), "Signature should be empty/default");
                }
                _ => panic!("Expected Thinking block"),
            }
        }

        // Now strip and verify
        let mut capped = request.clone();
        for msg in capped.messages.iter_mut() {
            if let palace_core::anthropic::Content::Blocks(blocks) = &mut msg.content {
                blocks.retain(|block| {
                    match block {
                        palace_core::anthropic::ContentBlock::Thinking { signature, .. } => !signature.is_empty(),
                        _ => true,
                    }
                });
            }
        }

        if let palace_core::anthropic::Content::Blocks(blocks) = &capped.messages[1].content {
            assert_eq!(blocks.len(), 1, "Thinking should be stripped");
        }
    }

#[cfg(test)]
mod tool_id_translation_tests {
    use crate::convert::{
        is_valid_mistral_id, shorten_tool_id, ToolIdMapper,
        anthropic_request_to_openai,
    };
    use palace_core::anthropic::{Content, ContentBlock, Message, Role, ToolResultContent};
    use palace_core::openai;

    // ============================================================================
    // Basic ID validation tests
    // ============================================================================

    #[test]
    fn test_valid_mistral_ids() {
        // Valid: exactly 9 alphanumeric chars
        assert!(is_valid_mistral_id("abc123def"));
        assert!(is_valid_mistral_id("ABCDEFGHI"));
        assert!(is_valid_mistral_id("123456789"));
        assert!(is_valid_mistral_id("aB3dE6gH9"));
    }

    #[test]
    fn test_invalid_mistral_ids() {
        // Too short
        assert!(!is_valid_mistral_id("abc12345"));
        // Too long
        assert!(!is_valid_mistral_id("abc1234567"));
        // Contains non-alphanumeric
        assert!(!is_valid_mistral_id("abc-12345"));
        assert!(!is_valid_mistral_id("abc_12345"));
        // Claude format (toolu_...)
        assert!(!is_valid_mistral_id("toolu_01JDFz8xGYXBaLj4gENsQDvj"));
    }

    // ============================================================================
    // shorten_tool_id tests
    // ============================================================================

    #[test]
    fn test_shorten_valid_id_unchanged() {
        // Already valid IDs should pass through unchanged
        let valid_id = "abc123def";
        assert_eq!(shorten_tool_id(valid_id), valid_id);
    }

    #[test]
    fn test_shorten_claude_id() {
        // Claude IDs (toolu_...) should be shortened
        let claude_id = "toolu_01JDFz8xGYXBaLj4gENsQDvj";
        let short = shorten_tool_id(claude_id);

        assert_eq!(short.len(), 9, "Should be exactly 9 chars");
        assert!(short.chars().all(|c| c.is_ascii_alphanumeric()),
            "Should be alphanumeric only: {}", short);
    }

    #[test]
    fn test_shorten_deterministic() {
        // Same input should always produce same output (deterministic hash)
        let claude_id = "toolu_01JDFz8xGYXBaLj4gENsQDvj";
        let short1 = shorten_tool_id(claude_id);
        let short2 = shorten_tool_id(claude_id);

        assert_eq!(short1, short2, "Shortening should be deterministic");
    }

    #[test]
    fn test_shorten_different_inputs_different_outputs() {
        // Different inputs should produce different outputs
        let id1 = "toolu_aaaaaaaaaaaaaaaaaaaa";
        let id2 = "toolu_bbbbbbbbbbbbbbbbbbbb";

        let short1 = shorten_tool_id(id1);
        let short2 = shorten_tool_id(id2);

        assert_ne!(short1, short2, "Different inputs should hash differently");
    }

    // ============================================================================
    // ToolIdMapper tests
    // ============================================================================

    #[test]
    fn test_mapper_to_short_valid_passthrough() {
        let mut mapper = ToolIdMapper::new();
        let valid_id = "abc123def";

        let result = mapper.to_short(valid_id);
        assert_eq!(result, valid_id, "Valid ID should pass through");
    }

    #[test]
    fn test_mapper_to_short_claude_id() {
        let mut mapper = ToolIdMapper::new();
        let claude_id = "toolu_01JDFz8xGYXBaLj4gENsQDvj";

        let short = mapper.to_short(claude_id);
        assert_eq!(short.len(), 9);
        assert!(is_valid_mistral_id(&short));
    }

    #[test]
    fn test_mapper_to_long_restores_original() {
        let mut mapper = ToolIdMapper::new();
        let claude_id = "toolu_01JDFz8xGYXBaLj4gENsQDvj";

        // Shorten it
        let short = mapper.to_short(claude_id);

        // Restore it
        let restored = mapper.to_long(&short);
        assert_eq!(restored, claude_id, "Should restore original long ID");
    }

    #[test]
    fn test_mapper_to_long_unknown_id() {
        let mapper = ToolIdMapper::new();
        let unknown_id = "xyz789abc";

        // Unknown IDs should pass through
        let result = mapper.to_long(unknown_id);
        assert_eq!(result, unknown_id, "Unknown ID should pass through");
    }

    #[test]
    fn test_mapper_caches_consistently() {
        let mut mapper = ToolIdMapper::new();
        let claude_id = "toolu_01JDFz8xGYXBaLj4gENsQDvj";

        // Call to_short multiple times
        let short1 = mapper.to_short(claude_id);
        let short2 = mapper.to_short(claude_id);

        assert_eq!(short1, short2, "Should return cached value");
    }

    #[test]
    fn test_mapper_register_manual_mapping() {
        let mut mapper = ToolIdMapper::new();

        // Manually register a mapping
        mapper.register("xyz789abc".to_string(), "toolu_custom_id".to_string());

        // Should be able to restore it
        let restored = mapper.to_long("xyz789abc");
        assert_eq!(restored, "toolu_custom_id");
    }

    // ============================================================================
    // Full conversation flow tests
    // ============================================================================

    #[test]
    fn test_anthropic_to_openai_tool_use_ids_shortened() {
        // Simulate Claude's tool_use with long ID
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![
                    ContentBlock::ToolUse {
                        id: "toolu_01JDFz8xGYXBaLj4gENsQDvj".to_string(),
                        name: "Read".to_string(),
                        input: serde_json::json!({"file_path": "/test"}),
                    },
                ]),
            },
        ];

        let (openai_msgs, id_mapping) = anthropic_request_to_openai(&messages, None);

        // Should have the message
        assert_eq!(openai_msgs.len(), 1);

        // Should have tool_calls with shortened ID
        let tool_calls = openai_msgs[0].tool_calls.as_ref().expect("Should have tool_calls");
        assert_eq!(tool_calls.len(), 1);

        let tool_id = &tool_calls[0].id;
        assert_eq!(tool_id.len(), 9, "ID should be shortened to 9 chars");
        assert!(is_valid_mistral_id(tool_id), "ID should be Mistral-compatible");

        // Mapping should exist
        assert!(id_mapping.contains_key(tool_id), "Mapping should be recorded");
        assert_eq!(
            id_mapping.get(tool_id).unwrap(),
            "toolu_01JDFz8xGYXBaLj4gENsQDvj",
            "Mapping should point to original"
        );
    }

    #[test]
    fn test_anthropic_to_openai_tool_result_ids_shortened() {
        // Simulate user message with tool_result referencing Claude's ID
        let messages = vec![
            Message {
                role: Role::User,
                content: Content::Blocks(vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "toolu_01JDFz8xGYXBaLj4gENsQDvj".to_string(),
                        content: ToolResultContent::Text("file contents".to_string()),
                        is_error: Some(false),
                    },
                ]),
            },
        ];

        let (openai_msgs, id_mapping) = anthropic_request_to_openai(&messages, None);

        // Should produce a Tool role message
        assert_eq!(openai_msgs.len(), 1);
        assert_eq!(openai_msgs[0].role, openai::Role::Tool);

        // ID should be shortened
        let tool_call_id = openai_msgs[0].tool_call_id.as_ref().expect("Should have tool_call_id");
        assert_eq!(tool_call_id.len(), 9, "ID should be shortened");
        assert!(is_valid_mistral_id(tool_call_id));
    }

    #[test]
    fn test_tool_use_and_result_ids_match() {
        // Full flow: assistant tool_use -> user tool_result
        // IDs must match after shortening
        let claude_tool_id = "toolu_01JDFz8xGYXBaLj4gENsQDvj";

        let messages = vec![
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![
                    ContentBlock::ToolUse {
                        id: claude_tool_id.to_string(),
                        name: "Read".to_string(),
                        input: serde_json::json!({"file_path": "/test"}),
                    },
                ]),
            },
            Message {
                role: Role::User,
                content: Content::Blocks(vec![
                    ContentBlock::ToolResult {
                        tool_use_id: claude_tool_id.to_string(),
                        content: ToolResultContent::Text("file contents".to_string()),
                        is_error: Some(false),
                    },
                ]),
            },
        ];

        let (openai_msgs, _) = anthropic_request_to_openai(&messages, None);

        // Get the IDs from both messages
        let tool_use_id = &openai_msgs[0]
            .tool_calls.as_ref().unwrap()[0].id;
        let tool_result_id = openai_msgs[1]
            .tool_call_id.as_ref().unwrap();

        assert_eq!(
            tool_use_id, tool_result_id,
            "tool_use ID and tool_result ID must match after shortening"
        );
    }

    #[test]
    fn test_already_short_ids_preserved() {
        // If IDs are already Mistral-compatible (from a previous Mistral turn),
        // they should pass through unchanged
        let mistral_id = "xyz789abc";

        let messages = vec![
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![
                    ContentBlock::ToolUse {
                        id: mistral_id.to_string(),
                        name: "Read".to_string(),
                        input: serde_json::json!({"file_path": "/test"}),
                    },
                ]),
            },
            Message {
                role: Role::User,
                content: Content::Blocks(vec![
                    ContentBlock::ToolResult {
                        tool_use_id: mistral_id.to_string(),
                        content: ToolResultContent::Text("ok".to_string()),
                        is_error: Some(false),
                    },
                ]),
            },
        ];

        let (openai_msgs, _) = anthropic_request_to_openai(&messages, None);

        // IDs should be unchanged
        let tool_use_id = &openai_msgs[0].tool_calls.as_ref().unwrap()[0].id;
        let tool_result_id = openai_msgs[1].tool_call_id.as_ref().unwrap();

        assert_eq!(tool_use_id, mistral_id, "Valid ID should pass through");
        assert_eq!(tool_result_id, mistral_id, "Valid ID should pass through");
    }

    #[test]
    fn test_mixed_ids_in_conversation() {
        // Conversation with both Claude IDs (from handoff history) and Mistral IDs
        let claude_id = "toolu_01JDFz8xGYXBaLj4gENsQDvj";
        let mistral_id = "xyz789abc";

        let messages = vec![
            // Old Claude tool call (from history before handoff)
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![
                    ContentBlock::ToolUse {
                        id: claude_id.to_string(),
                        name: "Read".to_string(),
                        input: serde_json::json!({"path": "/old"}),
                    },
                ]),
            },
            Message {
                role: Role::User,
                content: Content::Blocks(vec![
                    ContentBlock::ToolResult {
                        tool_use_id: claude_id.to_string(),
                        content: ToolResultContent::Text("old result".to_string()),
                        is_error: Some(false),
                    },
                ]),
            },
            // Recent Mistral tool call (after handoff)
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![
                    ContentBlock::ToolUse {
                        id: mistral_id.to_string(),
                        name: "Bash".to_string(),
                        input: serde_json::json!({"cmd": "ls"}),
                    },
                ]),
            },
            Message {
                role: Role::User,
                content: Content::Blocks(vec![
                    ContentBlock::ToolResult {
                        tool_use_id: mistral_id.to_string(),
                        content: ToolResultContent::Text("new result".to_string()),
                        is_error: Some(false),
                    },
                ]),
            },
        ];

        let (openai_msgs, _) = anthropic_request_to_openai(&messages, None);

        // Claude ID pair should match (shortened)
        let claude_use = &openai_msgs[0].tool_calls.as_ref().unwrap()[0].id;
        let claude_result = openai_msgs[1].tool_call_id.as_ref().unwrap();
        assert_eq!(claude_use, claude_result, "Claude ID pair must match");
        assert_eq!(claude_use.len(), 9, "Claude ID should be shortened");

        // Mistral ID pair should match (unchanged)
        let mistral_use = &openai_msgs[2].tool_calls.as_ref().unwrap()[0].id;
        let mistral_result = openai_msgs[3].tool_call_id.as_ref().unwrap();
        assert_eq!(mistral_use, mistral_result, "Mistral ID pair must match");
        assert_eq!(mistral_use, mistral_id, "Mistral ID should be unchanged");
    }
}
