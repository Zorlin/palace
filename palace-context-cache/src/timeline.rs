//! DAG Timeline for swarm consciousness
//!
//! Stores events, decisions, and context snapshots in a directed acyclic graph.
//! The display window slides - you're not losing history, you're focusing it.
//!
//! ```text
//! ┌─────┐     ┌─────┐     ┌─────┐
//! │ E1  │────▶│ E2  │────▶│ E3  │  Main timeline
//! └─────┘     └─────┘     └──┬──┘
//!                            │
//!                     ┌──────┴──────┐
//!                     ▼             ▼
//!                 ┌─────┐       ┌─────┐
//!                 │ E4  │       │ E5  │  Branches (exploration)
//!                 └─────┘       └─────┘
//! ```

use crate::BlockId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique event ID
pub type EventId = u64;

/// A timeline event in the DAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Unique event ID
    pub id: EventId,
    /// Parent event(s) - multiple parents for merges
    pub parents: Vec<EventId>,
    /// Timestamp (unix epoch)
    pub timestamp: u64,
    /// Event type and payload
    pub event_type: EventType,
    /// Active context blocks at this point
    pub active_context: Vec<BlockId>,
    /// Optional branch name for exploration
    pub branch: Option<String>,
}

/// Types of timeline events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    /// User message
    UserMessage {
        content: String,
        model_target: Option<String>,
    },
    /// Model response
    ModelResponse {
        model: String,
        content: String,
        tokens_used: Option<u32>,
    },
    /// Tool call
    ToolCall {
        tool_name: String,
        arguments: String,
    },
    /// Tool result
    ToolResult {
        tool_name: String,
        result: String,
        success: bool,
    },
    /// Context delta applied
    ContextDelta {
        delta: String,
        classifier_model: Option<String>,
    },
    /// Decision point
    Decision {
        description: String,
        chosen_option: String,
        alternatives: Vec<String>,
    },
    /// Branch created
    BranchCreated {
        branch_name: String,
        reason: String,
    },
    /// Branch merged
    BranchMerged {
        branch_name: String,
        into: EventId,
    },
    /// Checkpoint for recovery
    Checkpoint {
        description: String,
    },
    /// Model delegation
    Delegation {
        from_model: String,
        to_model: String,
        delegation_type: DelegationType,
    },
}

/// Types of model delegation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DelegationType {
    /// Full handoff - delegate takes over
    Delegate,
    /// Single-turn consultation
    Ask,
    /// Chain mention - continue conversation
    Mention,
}

/// The DAG timeline storage
pub struct Timeline {
    /// All events by ID
    events: HashMap<EventId, TimelineEvent>,
    /// Next event ID
    next_id: EventId,
    /// Head events (latest on each branch)
    heads: HashMap<String, EventId>,
    /// Main branch head
    main_head: Option<EventId>,
}

impl Timeline {
    /// Create a new empty timeline
    pub fn new() -> Self {
        Self {
            events: HashMap::new(),
            next_id: 1,
            heads: HashMap::new(),
            main_head: None,
        }
    }

    /// Append an event to the main timeline
    pub fn append(&mut self, event_type: EventType, active_context: Vec<BlockId>) -> EventId {
        let id = self.next_id;
        self.next_id += 1;

        let parents = self.main_head.map(|h| vec![h]).unwrap_or_default();

        let event = TimelineEvent {
            id,
            parents,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            event_type,
            active_context,
            branch: None,
        };

        self.events.insert(id, event);
        self.main_head = Some(id);
        id
    }

    /// Create a branch from the current head
    pub fn create_branch(&mut self, name: &str, reason: &str) -> EventId {
        let branch_event = self.append(
            EventType::BranchCreated {
                branch_name: name.to_string(),
                reason: reason.to_string(),
            },
            vec![],
        );

        self.heads.insert(name.to_string(), branch_event);
        branch_event
    }

    /// Append an event to a specific branch
    pub fn append_to_branch(
        &mut self,
        branch: &str,
        event_type: EventType,
        active_context: Vec<BlockId>,
    ) -> Option<EventId> {
        let parent_id = *self.heads.get(branch)?;

        let id = self.next_id;
        self.next_id += 1;

        let event = TimelineEvent {
            id,
            parents: vec![parent_id],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            event_type,
            active_context,
            branch: Some(branch.to_string()),
        };

        self.events.insert(id, event);
        self.heads.insert(branch.to_string(), id);
        Some(id)
    }

    /// Merge a branch into main
    pub fn merge_branch(&mut self, branch: &str) -> Option<EventId> {
        let branch_head = *self.heads.get(branch)?;
        let main_head = self.main_head?;

        let id = self.next_id;
        self.next_id += 1;

        let event = TimelineEvent {
            id,
            parents: vec![main_head, branch_head],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            event_type: EventType::BranchMerged {
                branch_name: branch.to_string(),
                into: main_head,
            },
            active_context: vec![],
            branch: None,
        };

        self.events.insert(id, event);
        self.main_head = Some(id);
        self.heads.remove(branch);
        Some(id)
    }

    /// Get an event by ID
    pub fn get(&self, id: EventId) -> Option<&TimelineEvent> {
        self.events.get(&id)
    }

    /// Get the main head event
    pub fn head(&self) -> Option<&TimelineEvent> {
        self.main_head.and_then(|id| self.events.get(&id))
    }

    /// Get recent events from main timeline (sliding window)
    pub fn recent(&self, count: usize) -> Vec<&TimelineEvent> {
        let mut current = self.main_head;
        let mut events = Vec::with_capacity(count);

        while let Some(id) = current {
            if events.len() >= count {
                break;
            }
            if let Some(event) = self.events.get(&id) {
                events.push(event);
                current = event.parents.first().copied();
            } else {
                break;
            }
        }

        events
    }

    /// Get all events (for export)
    pub fn all_events(&self) -> impl Iterator<Item = &TimelineEvent> {
        self.events.values()
    }

    /// Get event count
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if timeline is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get all branch names
    pub fn branches(&self) -> Vec<&str> {
        self.heads.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_events() {
        let mut timeline = Timeline::new();

        let e1 = timeline.append(
            EventType::UserMessage {
                content: "Hello".to_string(),
                model_target: None,
            },
            vec![1, 2, 3],
        );

        let e2 = timeline.append(
            EventType::ModelResponse {
                model: "claude".to_string(),
                content: "Hi there!".to_string(),
                tokens_used: Some(10),
            },
            vec![1, 2, 3],
        );

        assert_eq!(timeline.len(), 2);
        let empty_vec: Vec<EventId> = vec![];
        assert_eq!(timeline.get(e1).unwrap().parents, empty_vec);
        assert_eq!(timeline.get(e2).unwrap().parents, vec![e1]);
    }

    #[test]
    fn test_branching() {
        let mut timeline = Timeline::new();

        timeline.append(
            EventType::UserMessage {
                content: "Start".to_string(),
                model_target: None,
            },
            vec![],
        );

        timeline.create_branch("exploration", "Testing alternative");

        timeline.append_to_branch(
            "exploration",
            EventType::UserMessage {
                content: "Branch work".to_string(),
                model_target: None,
            },
            vec![],
        );

        assert!(timeline.branches().contains(&"exploration"));
        assert_eq!(timeline.len(), 3);
    }

    #[test]
    fn test_recent_window() {
        let mut timeline = Timeline::new();

        for i in 0..10 {
            timeline.append(
                EventType::UserMessage {
                    content: format!("Message {}", i),
                    model_target: None,
                },
                vec![],
            );
        }

        let recent = timeline.recent(3);
        assert_eq!(recent.len(), 3);

        // Most recent first
        if let EventType::UserMessage { content, .. } = &recent[0].event_type {
            assert_eq!(content, "Message 9");
        } else {
            panic!("Expected UserMessage");
        }
    }

    #[test]
    fn test_delegation_event() {
        let mut timeline = Timeline::new();

        timeline.append(
            EventType::Delegation {
                from_model: "claude".to_string(),
                to_model: "mistral".to_string(),
                delegation_type: DelegationType::Delegate,
            },
            vec![1, 2],
        );

        let event = timeline.head().unwrap();
        if let EventType::Delegation { delegation_type, .. } = &event.event_type {
            assert!(matches!(delegation_type, DelegationType::Delegate));
        } else {
            panic!("Expected Delegation event");
        }
    }
}
