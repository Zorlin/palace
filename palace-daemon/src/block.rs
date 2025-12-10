//! Context block management
//!
//! This module handles the creation and management of context blocks.

use crate::{BlockId, BlockMeta, BlockType};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Builder for creating context blocks with fluent API
#[derive(Debug, Default)]
pub struct BlockBuilder {
    summary: Option<String>,
    block_type: Option<BlockType>,
    content: Option<BlockContent>,
    channels: Vec<String>,
}

/// Content source for a block
#[derive(Debug, Clone)]
pub enum BlockContent {
    /// Content from a file path
    File(PathBuf),
    /// Content from raw string (will be written to temp file)
    Raw(String),
    /// Content from bytes (will be written to temp file)
    Bytes(Vec<u8>),
}

impl BlockBuilder {
    /// Create a new block builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the summary for classifier input
    pub fn summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Set the block type
    pub fn block_type(mut self, block_type: BlockType) -> Self {
        self.block_type = Some(block_type);
        self
    }

    /// Set content from file
    pub fn file(mut self, path: impl Into<PathBuf>) -> Self {
        self.content = Some(BlockContent::File(path.into()));
        self
    }

    /// Set content from raw string
    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(BlockContent::Raw(content.into()));
        self
    }

    /// Set content from bytes
    pub fn bytes(mut self, bytes: Vec<u8>) -> Self {
        self.content = Some(BlockContent::Bytes(bytes));
        self
    }

    /// Set channels for this block
    pub fn channels(mut self, channels: &[&str]) -> Self {
        self.channels = channels.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add a single channel
    pub fn channel(mut self, channel: impl Into<String>) -> Self {
        self.channels.push(channel.into());
        self
    }

    /// Build the block configuration
    pub fn build(self) -> Result<BlockConfig, &'static str> {
        Ok(BlockConfig {
            summary: self.summary.ok_or("summary is required")?,
            block_type: self.block_type.ok_or("block_type is required")?,
            content: self.content.ok_or("content is required")?,
            channels: self.channels,
        })
    }
}

/// Configuration for creating a new block
#[derive(Debug, Clone)]
pub struct BlockConfig {
    pub summary: String,
    pub block_type: BlockType,
    pub content: BlockContent,
    pub channels: Vec<String>,
}

/// Snapshot of a block at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockSnapshot {
    pub id: BlockId,
    pub summary: String,
    pub block_type: BlockType,
    pub content_hash: String,
    pub size: u64,
    pub timestamp: u64,
    pub channels: Vec<String>,
}

impl From<&BlockMeta> for BlockSnapshot {
    fn from(meta: &BlockMeta) -> Self {
        BlockSnapshot {
            id: meta.id,
            summary: meta.summary.clone(),
            block_type: meta.block_type,
            content_hash: String::new(), // TODO: compute hash
            size: meta.size,
            timestamp: meta.modified,
            channels: meta.channels.clone(),
        }
    }
}

/// Block update event
#[derive(Debug, Clone)]
pub enum BlockEvent {
    /// Block was registered
    Registered(BlockId),
    /// Block was activated
    Activated(BlockId),
    /// Block was deactivated
    Deactivated(BlockId),
    /// Block content was modified
    Modified(BlockId),
    /// Block was removed
    Removed(BlockId),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_builder() {
        let config = BlockBuilder::new()
            .summary("Test block")
            .block_type(BlockType::CurrentFile)
            .content("hello world")
            .build()
            .unwrap();

        assert_eq!(config.summary, "Test block");
        assert_eq!(config.block_type, BlockType::CurrentFile);
    }

    #[test]
    fn test_block_builder_file() {
        let config = BlockBuilder::new()
            .summary("File block")
            .block_type(BlockType::BuildStatus)
            .file("/tmp/test.txt")
            .build()
            .unwrap();

        matches!(config.content, BlockContent::File(_));
    }

    #[test]
    fn test_block_builder_missing_fields() {
        let result = BlockBuilder::new().summary("Test").build();
        assert!(result.is_err());
    }
}
