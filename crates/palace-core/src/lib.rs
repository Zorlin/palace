//! Palace Core - Shared types for API translation
//!
//! This crate defines the message formats for both Anthropic and OpenAI APIs,
//! enabling bidirectional translation between them.

pub mod anthropic;
pub mod openai;
pub mod error;
pub mod rate_limit;

pub use error::TranslatorError;
pub use rate_limit::{Provider, RateLimiter, RateLimitResult, RateLimitStats};

#[cfg(test)]
mod tests {
    use super::anthropic::*;
    use serde_json::json;

    #[test]
    fn test_thinking_block_without_signature_deserializes() {
        let glm_content = json!([
            {
                "type": "thinking",
                "thinking": "Let me think about this..."
            },
            {
                "type": "text",
                "text": "Here's my response"
            }
        ]);
        
        let blocks: Vec<ContentBlock> = serde_json::from_value(glm_content).expect("Should deserialize");
        assert_eq!(blocks.len(), 2);
        
        match &blocks[0] {
            ContentBlock::Thinking { thinking, signature } => {
                assert_eq!(thinking, "Let me think about this...");
                assert_eq!(signature, ""); // Should default to empty
            }
            _ => panic!("Expected Thinking block"),
        }
    }
    
    #[test]
    fn test_thinking_block_with_signature_deserializes() {
        let claude_content = json!([
            {
                "type": "thinking",
                "thinking": "Let me think...",
                "signature": "abc123"
            }
        ]);
        
        let blocks: Vec<ContentBlock> = serde_json::from_value(claude_content).expect("Should deserialize");
        
        match &blocks[0] {
            ContentBlock::Thinking { thinking, signature } => {
                assert_eq!(thinking, "Let me think...");
                assert_eq!(signature, "abc123");
            }
            _ => panic!("Expected Thinking block"),
        }
    }
    
    #[test]
    fn test_strip_thinking_without_signature() {
        let mut messages = vec![
            Message {
                role: Role::Assistant,
                content: Content::Blocks(vec![
                    ContentBlock::Thinking {
                        thinking: "GLM thinking".to_string(),
                        signature: "".to_string(), // Empty = from GLM
                    },
                    ContentBlock::Text {
                        text: "Response".to_string(),
                    },
                ]),
            },
        ];
        
        // Strip logic
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
        
        // Should only have Text block left
        if let Content::Blocks(blocks) = &messages[0].content {
            assert_eq!(blocks.len(), 1);
            assert!(matches!(blocks[0], ContentBlock::Text { .. }));
        } else {
            panic!("Expected Blocks");
        }
    }
}
