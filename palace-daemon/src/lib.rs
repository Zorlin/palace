//! Palace Daemon Library
//!
//! High-performance core for Palace:
//! - API Translator (Anthropic ↔ OpenAI format)
//! - Context Cache (swarm consciousness)
//! - Agent Coordination
//!
//! Key design principles:
//! - Active context IDs stay in L0 cache (just integers)
//! - Content is memory-mapped, loaded on demand
//! - fsnotify watches for changes (build status, test results, etc.)
//! - Near-zero cost when idle, instant when needed
//!
//! The 1b classifier outputs simple add/remove instructions:
//! ```text
//! ++1,2,3,4--5,6,7,8
//! ```
//!
//! This gets parsed and applied to the active set atomically.

pub mod translator;
pub mod delta;
pub mod block;
pub mod timeline;
pub mod classifier;
pub mod spawn;

use dashmap::DashMap;
use memmap2::Mmap;
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// A context block ID - just a u32 for L0 cache efficiency
pub type BlockId = u32;

/// Context block metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockMeta {
    /// Unique block ID
    pub id: BlockId,
    /// Short summary for classifier input
    pub summary: String,
    /// Block type for categorization
    pub block_type: BlockType,
    /// Path to full content file (memory-mapped on demand)
    pub content_path: PathBuf,
    /// Size in bytes
    pub size: u64,
    /// Last modified timestamp
    pub modified: u64,
    /// Is this block currently active?
    pub active: bool,
    /// Context channels this block belongs to (coding, security, writing, etc.)
    pub channels: Vec<String>,
}

/// Block types for context categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlockType {
    /// Current file being worked on
    CurrentFile,
    /// Recent error or warning
    Error,
    /// User's stated goal
    UserGoal,
    /// Previous attempt or approach
    PreviousAttempt,
    /// Build/test status
    BuildStatus,
    /// Test output
    TestOutput,
    /// Related but not immediately relevant
    Related,
    /// Unrelated context (candidate for removal)
    Unrelated,
    /// Model response from chain
    ModelResponse,
    /// Tool call result
    ToolResult,
}

/// Infer default channels from block type
pub fn infer_channels_from_type(block_type: BlockType) -> Vec<String> {
    match block_type {
        BlockType::CurrentFile => vec!["coding".into()],
        BlockType::Error => vec!["coding".into(), "debugging".into()],
        BlockType::UserGoal => vec!["general".into()],
        BlockType::PreviousAttempt => vec!["coding".into(), "debugging".into()],
        BlockType::BuildStatus => vec!["coding".into(), "testing".into()],
        BlockType::TestOutput => vec!["testing".into()],
        BlockType::Related => vec!["general".into()],
        BlockType::Unrelated => vec![],
        BlockType::ModelResponse => vec!["general".into()],
        BlockType::ToolResult => vec!["coding".into()],
    }
}

/// The hot cache - stores active context IDs and provides instant access
pub struct ContextCache {
    /// Active block IDs - tiny, fits in L0 cache
    active_ids: Arc<RwLock<HashSet<BlockId>>>,

    /// Block metadata registry
    blocks: Arc<DashMap<BlockId, BlockMeta>>,

    /// Memory-mapped content files (loaded on demand)
    content_cache: Arc<DashMap<BlockId, Mmap>>,

    /// File system watcher
    watcher: Option<RecommendedWatcher>,

    /// Event channel for watcher notifications
    event_tx: Option<mpsc::UnboundedSender<Event>>,

    /// Next available block ID
    next_id: Arc<RwLock<BlockId>>,

    /// Base directory for content files
    content_dir: PathBuf,
}

impl ContextCache {
    /// Create a new context cache
    pub fn new(content_dir: PathBuf) -> Self {
        Self {
            active_ids: Arc::new(RwLock::new(HashSet::new())),
            blocks: Arc::new(DashMap::new()),
            content_cache: Arc::new(DashMap::new()),
            watcher: None,
            event_tx: None,
            next_id: Arc::new(RwLock::new(1)),
            content_dir,
        }
    }

    /// Initialize filesystem watcher
    pub fn init_watcher(&mut self) -> Result<mpsc::UnboundedReceiver<Event>, notify::Error> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.event_tx = Some(tx.clone());

        let watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    if tx.send(event).is_err() {
                        error!("Failed to send filesystem event");
                    }
                }
                Err(e) => error!("Filesystem watch error: {:?}", e),
            }
        })?;

        self.watcher = Some(watcher);
        Ok(rx)
    }

    /// Watch a path for changes
    pub fn watch(&mut self, path: &PathBuf) -> Result<(), notify::Error> {
        if let Some(ref mut watcher) = self.watcher {
            watcher.watch(path.as_ref(), RecursiveMode::Recursive)?;
            info!("Watching path: {:?}", path);
        } else {
            warn!("Watcher not initialized, call init_watcher() first");
        }
        Ok(())
    }

    /// Register a new context block
    pub fn register_block(&self, summary: String, block_type: BlockType, content_path: PathBuf) -> BlockId {
        self.register_block_with_channels(summary, block_type, content_path, vec![])
    }

    /// Register a new context block with specific channels
    pub fn register_block_with_channels(
        &self,
        summary: String,
        block_type: BlockType,
        content_path: PathBuf,
        channels: Vec<String>,
    ) -> BlockId {
        let id = {
            let mut next_id = self.next_id.write();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let size = std::fs::metadata(&content_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let modified = std::fs::metadata(&content_path)
            .and_then(|m| m.modified())
            .map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs())
            .unwrap_or(0);

        // Auto-infer channels from block type if not provided
        let channels = if channels.is_empty() {
            infer_channels_from_type(block_type)
        } else {
            channels
        };

        let meta = BlockMeta {
            id,
            summary,
            block_type,
            content_path,
            size,
            modified,
            active: false,
            channels: channels.clone(),
        };

        debug!("Registered block {}: {:?} channels={:?}", id, block_type, channels);
        self.blocks.insert(id, meta);
        id
    }

    /// Get the current active block IDs (instant - L0 cache)
    #[inline]
    pub fn get_active_ids(&self) -> Vec<BlockId> {
        self.active_ids.read().iter().copied().collect()
    }

    /// Check if a block is active (instant - L0 cache)
    #[inline]
    pub fn is_active(&self, id: BlockId) -> bool {
        self.active_ids.read().contains(&id)
    }

    /// Apply a delta from the 1b classifier: ++1,2,3--4,5,6
    pub fn apply_delta(&self, delta_str: &str) -> Result<(), DeltaError> {
        let delta = delta::parse_delta(delta_str)?;

        let add_count = delta.add.len();
        let remove_count = delta.remove.len();

        let mut active = self.active_ids.write();

        // Add new blocks
        for id in delta.add {
            if self.blocks.contains_key(&id) {
                active.insert(id);
                if let Some(mut meta) = self.blocks.get_mut(&id) {
                    meta.active = true;
                }
                debug!("Activated block {}", id);
            } else {
                warn!("Tried to activate unknown block {}", id);
            }
        }

        // Remove blocks
        for id in delta.remove {
            active.remove(&id);
            if let Some(mut meta) = self.blocks.get_mut(&id) {
                meta.active = false;
            }
            // Also evict from content cache
            self.content_cache.remove(&id);
            debug!("Deactivated block {}", id);
        }

        info!("Delta applied: +{} -{}, active count: {}",
              add_count, remove_count, active.len());

        Ok(())
    }

    /// Get content for a block (memory-mapped, lazy loaded)
    pub fn get_content(&self, id: BlockId) -> Option<Vec<u8>> {
        // Check cache first
        if let Some(mmap) = self.content_cache.get(&id) {
            return Some(mmap.to_vec());
        }

        // Load from disk and cache
        let meta = self.blocks.get(&id)?;
        let file = File::open(&meta.content_path).ok()?;
        let mmap = unsafe { Mmap::map(&file).ok()? };
        let content = mmap.to_vec();
        self.content_cache.insert(id, mmap);

        Some(content)
    }

    /// Get content as string
    pub fn get_content_str(&self, id: BlockId) -> Option<String> {
        self.get_content(id)
            .and_then(|bytes| String::from_utf8(bytes).ok())
    }

    /// Get all active content concatenated
    pub fn get_active_context(&self) -> String {
        self.get_active_context_filtered(&[])
    }

    /// Get active content filtered by channels
    pub fn get_active_context_filtered(&self, channels: &[&str]) -> String {
        let ids = self.get_active_ids();
        let mut context = String::new();

        for id in ids {
            if let Some(meta) = self.blocks.get(&id) {
                // Filter by channels if specified
                if !channels.is_empty() {
                    let has_channel = meta.channels.iter().any(|c| channels.contains(&c.as_str()));
                    if !has_channel {
                        continue;
                    }
                }

                context.push_str(&format!("[{}] {}: {} [{}]\n",
                    id,
                    format!("{:?}", meta.block_type),
                    meta.summary,
                    meta.channels.join(",")
                ));
                if let Some(content) = self.get_content_str(id) {
                    context.push_str(&content);
                    context.push_str("\n\n");
                }
            }
        }

        context
    }

    /// Get blocks that match specific channels
    pub fn get_blocks_by_channel(&self, channel: &str) -> Vec<BlockId> {
        self.blocks
            .iter()
            .filter(|entry| entry.channels.contains(&channel.to_string()))
            .map(|entry| entry.id)
            .collect()
    }

    /// Add channels to an existing block
    pub fn add_channels(&self, id: BlockId, new_channels: &[String]) {
        if let Some(mut meta) = self.blocks.get_mut(&id) {
            for channel in new_channels {
                if !meta.channels.contains(channel) {
                    meta.channels.push(channel.clone());
                }
            }
        }
    }

    /// Remove channels from an existing block
    pub fn remove_channels(&self, id: BlockId, channels_to_remove: &[String]) {
        if let Some(mut meta) = self.blocks.get_mut(&id) {
            meta.channels.retain(|c| !channels_to_remove.contains(c));
        }
    }

    /// Generate classifier input - just summaries with IDs
    pub fn get_classifier_input(&self) -> String {
        let mut input = String::new();

        for entry in self.blocks.iter() {
            let meta = entry.value();
            let status = if meta.active { "*" } else { " " };
            input.push_str(&format!("[{}]{} {:?}: {} [{}]\n",
                meta.id,
                status,
                meta.block_type,
                meta.summary,
                meta.channels.join(",")
            ));
        }

        input
    }

    /// Get block count statistics
    pub fn stats(&self) -> CacheStats {
        let active_count = self.active_ids.read().len();
        let total_blocks = self.blocks.len();
        let cached_content = self.content_cache.len();

        CacheStats {
            active_blocks: active_count,
            total_blocks,
            cached_content,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub active_blocks: usize,
    pub total_blocks: usize,
    pub cached_content: usize,
}

/// Delta parsing error
#[derive(Debug)]
pub enum DeltaError {
    ParseError(String),
    InvalidFormat(String),
}

impl std::fmt::Display for DeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeltaError::ParseError(s) => write!(f, "Parse error: {}", s),
            DeltaError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
        }
    }
}

impl std::error::Error for DeltaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_activate() {
        let dir = std::env::temp_dir().join("palace_test_1");
        std::fs::create_dir_all(&dir).unwrap();
        let content_path = dir.join("test.txt");
        std::fs::write(&content_path, "test content").unwrap();

        let cache = ContextCache::new(dir.clone());

        let id = cache.register_block(
            "Test file".to_string(),
            BlockType::CurrentFile,
            content_path,
        );

        assert_eq!(id, 1);
        assert!(!cache.is_active(id));

        cache.apply_delta("++1").unwrap();
        assert!(cache.is_active(id));

        cache.apply_delta("--1").unwrap();
        assert!(!cache.is_active(id));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_get_content() {
        let dir = std::env::temp_dir().join("palace_test_2");
        std::fs::create_dir_all(&dir).unwrap();
        let content_path = dir.join("test.txt");
        std::fs::write(&content_path, "hello world").unwrap();

        let cache = ContextCache::new(dir.clone());
        let id = cache.register_block(
            "Test file".to_string(),
            BlockType::CurrentFile,
            content_path,
        );

        let content = cache.get_content_str(id).unwrap();
        assert_eq!(content, "hello world");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_delta_parsing() {
        let dir = std::env::temp_dir().join("palace_test_3");
        std::fs::create_dir_all(&dir).unwrap();
        let cache = ContextCache::new(dir.clone());

        // Register some blocks
        for i in 1..=8 {
            let path = dir.join(format!("block{}.txt", i));
            std::fs::write(&path, format!("content {}", i)).unwrap();
            cache.register_block(format!("Block {}", i), BlockType::Related, path);
        }

        // Apply delta
        cache.apply_delta("++1,2,3,4--5,6,7,8").unwrap();

        let active = cache.get_active_ids();
        assert_eq!(active.len(), 4);
        assert!(active.contains(&1));
        assert!(active.contains(&2));
        assert!(active.contains(&3));
        assert!(active.contains(&4));
        assert!(!active.contains(&5));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_classifier_input() {
        let dir = std::env::temp_dir().join("palace_test_4");
        std::fs::create_dir_all(&dir).unwrap();
        let cache = ContextCache::new(dir.clone());

        let path1 = dir.join("file1.txt");
        let path2 = dir.join("file2.txt");
        std::fs::write(&path1, "content 1").unwrap();
        std::fs::write(&path2, "content 2").unwrap();

        cache.register_block("Current file: main.rs".to_string(), BlockType::CurrentFile, path1);
        cache.register_block("Build status: PASSING".to_string(), BlockType::BuildStatus, path2);

        cache.apply_delta("++1").unwrap();

        let input = cache.get_classifier_input();
        assert!(input.contains("[1]*"));
        assert!(input.contains("[2] "));
        assert!(input.contains("CurrentFile"));
        assert!(input.contains("BuildStatus"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
