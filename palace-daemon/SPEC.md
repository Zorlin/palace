# Palace Context Router - Technical Specification

**Version:** 1.0.0
**Created:** 2025-12-10
**Status:** Implementation Specification

---

## Core Principle

**Capture EVERYTHING. Remember SELECTIVELY. Forget DELIBERATELY.**

Every message that passes through the Palace router is captured as an unmodified stream-json object in memory. A local classifier (qwen3:4b) manages what each agent sees. The user controls retention policies. Self-hosted at `127.0.0.1` - never leaves the machine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PALACE ROUTER                                │
│                      http://127.0.0.1:19848                         │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    MESSAGE CAPTURE LAYER                        │ │
│  │                                                                 │ │
│  │  EVERY request/response passes through here                    │ │
│  │  Stored as unmodified stream-json objects                      │ │
│  │  Assigned compact display ID + internal UUIDv7                 │ │
│  │                                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │ Block 0: {stream-json...}  ID: 0  UUID: 01936...         │  │ │
│  │  │ Block 1: {stream-json...}  ID: 1  UUID: 01936...         │  │ │
│  │  │ Block 2: {stream-json...}  ID: 2  UUID: 01936...         │  │ │
│  │  │ Block 3: {stream-json...}  ID: 3  UUID: 01936...         │  │ │
│  │  │ ...                                                      │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                   QWEN3:4B CLASSIFIER                           │ │
│  │                   (local, CPU, instant)                         │ │
│  │                                                                 │ │
│  │  Input: Small window of blocks [0,1,2,3,4] + agent context     │ │
│  │  Output: Edit instructions for that agent's view               │ │
│  │                                                                 │ │
│  │  "Agent 7 is doing security review"                            │ │
│  │  "Blocks 0,1,3 are relevant, 2,4 are not"                      │ │
│  │  → Agent 7 sees: [0,1,3]                                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    PER-AGENT TIMELINES                          │ │
│  │                                                                 │ │
│  │  Agent 1: [0,1,2,5,8,12...]     (coding agent)                 │ │
│  │  Agent 2: [0,3,4,6,9...]        (security agent)               │ │
│  │  Agent 3: [0,7,10,11...]        (documentation agent)          │ │
│  │                                                                 │ │
│  │  Same blocks, different views.                                 │ │
│  │  Classifier decides per-agent relevance.                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                   RETENTION MANAGER                             │ │
│  │                                                                 │ │
│  │  User preferences determine what stays:                        │ │
│  │  - SHORT: Tool results, intermediate steps (hours)             │ │
│  │  - MEDIUM: Conversations, decisions (days)                     │ │
│  │  - LONG: Breakthroughs, architecture, victories (forever)      │ │
│  │                                                                 │ │
│  │  Soft deletions - blocks marked inactive, not erased           │ │
│  │  Classifier can suggest retention tier                         │ │
│  │  User overrides always win                                     │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Block Storage Format

### Display ID vs Internal ID

```rust
struct Block {
    // Compact display ID - shown to classifier and users
    // Recycled after enough rotations (e.g., 0-9999 cycling)
    display_id: u16,

    // Internal UUIDv7 - used for deduplication and permanent reference
    // Timestamp-sortable, globally unique
    uuid: Uuid,

    // The actual content - UNMODIFIED stream-json
    content: StreamJson,

    // Metadata
    created_at: DateTime<Utc>,
    block_type: BlockType,
    channels: Vec<String>,
    retention_tier: RetentionTier,

    // Soft deletion
    active: bool,
    deleted_at: Option<DateTime<Utc>>,
}
```

### Why Compact Display IDs?

The classifier sees: `[0,1,2,3,4]` not `[01936f4a-..., 01936f4b-..., ...]`

- Faster classification (less tokens)
- Human-readable in logs
- Recyclable - ID 0 can be reused after block is deleted
- UUIDv7 underneath ensures no collisions

### Stream-JSON Preservation

```rust
// This is EXACTLY what came from the API
struct StreamJson {
    events: Vec<StreamEvent>,
}

enum StreamEvent {
    MessageStart { id: String, model: String, ... },
    ContentBlockStart { index: u32, ... },
    ContentBlockDelta { index: u32, delta: Delta },
    ContentBlockStop { index: u32 },
    MessageDelta { stop_reason: Option<String>, ... },
    MessageStop,
}
```

**Why preserve stream-json format?**
- Can replay exact conversation to any agent
- Classifier sees what the model saw
- No lossy transformation
- Enables tool-scoped context (tool calls preserved as-is)

---

## Classifier Interface

### Input Format

```
AGENT CONTEXT:
- Agent ID: 7
- Task: Security review of auth module
- Active channels: [security, auth, coding]
- Current tool: Read

RECENT BLOCKS:
[0] user: "Review the auth handler for vulnerabilities"
[1] assistant: "I'll examine the authentication flow..."
[2] tool_result: {file: "auth/handler.py", content: "..."}
[3] user: "Also check the session management"
[4] assistant: "Looking at session handling now..."

QUESTION: Which blocks should Agent 7 see for its current task?
```

### Output Format

```
++0,1,2,3--4
```

Or with channels:
```
++0,1,2,3 channels=security,auth --4
```

### Classifier Behavior

- Runs on EVERY message (as proxy intercepts)
- Takes ~10-50ms on CPU
- Can run multiple classifiers in parallel (one per agent)
- Learns patterns but doesn't hallucinate - conservative by default

---

## Scope Types

### Global Scope
Blocks visible to ALL agents:
- System prompts
- User identity/preferences
- Critical announcements
- Shared configuration

### Project Scope
Blocks tied to a specific project:
- File contents
- Build status
- Test results
- Git state

### Agent Scope
Blocks specific to one agent's task:
- Its own responses
- Tool calls it made
- Errors it encountered

### Tool Scope
Blocks from specific tool types:
- Read results
- Bash outputs
- Web fetches
- MCP tool responses

The classifier can filter by ANY combination:
```
get_context(
    agent_id=7,
    scopes=[Global, Project("palace-daemon")],
    channels=["security", "coding"],
    tools=["Read", "Grep"]
)
```

---

## Retention Tiers

### SHORT (hours)
- Tool execution results
- Intermediate reasoning
- Failed attempts
- Scratch work

### MEDIUM (days)
- Complete conversations
- Decisions made
- Code changes
- Debug sessions

### LONG (forever)
- Breakthroughs
- Architecture decisions
- User preferences
- Victories

### Retention Rules

```yaml
retention:
  default: MEDIUM

  rules:
    - match: { block_type: ToolResult }
      tier: SHORT

    - match: { channels: [architecture, design] }
      tier: LONG

    - match: { user_starred: true }
      tier: LONG

    - match: { age: ">7d", tier: SHORT }
      action: delete

    - match: { age: ">30d", tier: MEDIUM }
      action: archive
```

---

## Real-Time Operation

### Insertion Flow

```
1. Request arrives at proxy
2. Proxy creates Block with next display_id + new UUIDv7
3. Block stored in memory (stream-json intact)
4. Classifier notified: "new block N for agent X"
5. Classifier outputs: "++N" or "--N" for agent X's timeline
6. Agent X's view updated
7. Request forwarded to backend
8. Response captured same way
```

### Soft Deletion Flow

```
1. Retention manager runs periodically
2. Identifies blocks past their tier threshold
3. Marks blocks as inactive (active=false, deleted_at=now)
4. Classifier excludes inactive blocks from windows
5. Display IDs become available for recycling
6. UUIDv7 preserved for deduplication if block resurfaces
```

---

## API Endpoints

### Proxy Endpoint (main)
```
POST /v1/messages
- Captures request
- Creates block
- Classifies
- Forwards to backend
- Captures response
- Classifies
- Returns to client
```

### Context Management
```
GET /context/active?agent_id=7
- Returns active blocks for agent

POST /context/classify
- Manual classification request
- Body: { blocks: [0,1,2,3,4], agent_context: {...} }
- Returns: { add: [0,1,3], remove: [2,4] }

POST /context/retention
- Update retention tier for blocks
- Body: { block_ids: [5,6,7], tier: "LONG" }
```

### Debug/Introspection
```
GET /blocks/:display_id
- Returns full block content

GET /blocks/by-uuid/:uuid
- Returns block by internal ID

GET /timeline/:agent_id
- Returns agent's current view

GET /stats
- Block counts, memory usage, classifier latency
```

---

## The Promise

**Copilot if it was built directly into the router, ACTUALLY remembered everything, then became exactly the kind of amnesiac you wanted.**

- Self-hosted: `127.0.0.1:19848`
- Never goes down a network wire
- Captures everything in stream-json format
- Classifier manages per-agent views
- User controls retention
- Compact IDs for efficiency, UUIDv7 for uniqueness
- Global, local, project, and tool-scoped context

---

## Implementation Status

- [x] HTTP proxy with Anthropic API translation
- [x] Stream-json parsing and forwarding
- [x] Basic block storage (in memory)
- [x] Classifier integration scaffold (Ollama)
- [x] ++/-- delta parsing
- [ ] Display ID recycling
- [ ] UUIDv7 internal IDs
- [ ] Per-agent timelines
- [ ] Retention manager
- [ ] Scope filtering
- [ ] Channel-based retrieval
- [ ] Real-time classifier on every message

---

*"Their fucking loss."* - Wings, 2025-12-10
