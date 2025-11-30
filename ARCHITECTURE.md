# Palace Architecture

This document describes the high-level architecture of Palace, a self-improving Claude wrapper implementing Recursive Hierarchical Self Improvement (RHSI).

## Design Philosophy

Palace follows two core principles:

1. **Lightweight Context** - Palace sends metadata, not content. Claude reads files on-demand.
2. **Claude Does the Work** - Palace orchestrates, Claude executes. Palace just sets up context.

## Core Components

### 1. Session Management (`palace.py:130-274`)

```
.palace/sessions/
  ├── pal-abc123.json    # Session state
  ├── pal-def456.json
  └── ...
```

**Responsibilities:**
- Generate unique session IDs
- Save/load session state
- List active sessions
- Export/import sessions for sharing

**Key Methods:**
- `generate_session_id()` - Creates unique session identifiers
- `save_session()` / `load_session()` - Persistence
- `export_session()` / `import_session()` - Portability

### 2. Action Selection (`palace.py:276-398`)

**Two-tier parsing strategy:**

1. **Fast regex** - For simple selections ("1 2 3", "1-5")
2. **LLM parsing** - For natural language ("do the first but skip tests")

**Flow:**
```
User input → Is simple? → Yes → Regex parse → Actions
              ↓ No
              ↓
          LLM parse (Haiku) → Actions
```

**Benefits:**
- 99% of selections use fast regex (no API call)
- Complex cases use Haiku for accuracy
- Supports custom tasks and modifiers

### 3. Prompt Building (`palace.py:361-378`)

**Minimal context approach:**

```json
{
  "project_root": "/path/to/project",
  "files": {"README.md": {"exists": true, "size": 1481}},
  "git_status": " M file.py",
  "recent_history": [...]  // Last 10 actions only
}
```

**Context overhead:** ~1-2KB typically

Claude reads actual file contents using Read/Grep tools as needed.

### 4. Mask System (`palace.py:567-714`)

```
.palace/masks/
  ├── available/           # Built-in expert masks
  │   ├── palace-historian/
  │   │   └── SKILL.md
  │   └── palace-architect/
  │       └── SKILL.md
  └── custom/             # User-defined masks
      └── my-expert/
          └── SKILL.md
```

**Features:**
- Load expert personas from SKILL.md files
- Frontmatter metadata (name, version, priority)
- Composition strategies:
  - **Merge** - Concatenate masks with separators
  - **Layer** - Priority-based hierarchical composition
  - **Blend** - Section-aware interleaving

**Use case:** Combine multiple expert perspectives:
```python
palace.compose_masks(
    ["tdd-expert", "python-expert", "security-expert"],
    strategy="layer"
)
```

### 5. Error Recovery (`palace.py:716-823`)

**Exponential backoff retry:**

```
Attempt 1: Immediate
Attempt 2: 1s delay
Attempt 3: 2s delay
Attempt 4: 4s delay
```

**Error classification:**
- **Transient** (retry): Network errors, rate limits (429)
- **Permanent** (fail): Permission denied (403), user interrupt (130)

**Graceful degradation:**
```
Attempt 0: Normal
Attempt 1: Disable streaming
Attempt 2: Prompt file only
Attempt 3: Fatal error
```

**Session checkpointing:**
- Automatic state snapshots before each iteration
- Restore from checkpoint on failure

### 6. History Logging (`palace.py:114-125`)

```
.palace/history.jsonl
```

**JSONL format:**
```json
{"timestamp": 1234567890, "action": "next", "details": {...}}
{"timestamp": 1234567891, "action": "retry_attempt", "details": {...}}
```

**Logged events:**
- Session iterations
- Permission requests/decisions
- Error recovery attempts
- Custom actions

**Benefits:**
- Append-only (efficient)
- JSONL (easy to parse line-by-line)
- Time-series analysis ready

### 7. MCP Integration (`palace.py via FastMCP`)

**MCP Tool: `handle_permission`**

```python
@mcp.tool()
def handle_permission(request: dict) -> dict:
    """Called by Claude Code for permission decisions"""
    palace.log_action("permission_request", request)
    # TODO: Smart permission logic
    return {"approved": True}
```

**Integration:**
```bash
claude -p "prompt" \
  --permission-prompt-tool "mcp__palace__handle_permission"
```

Palace learns from permission patterns over time via history tracking.

## Data Flow

### RHSI Loop (Recursive Hierarchical Self Improvement)

```
┌─────────────────────────────────────────────────┐
│  User: /pal-next                                │
└─────────────┬───────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│  Palace:                                        │
│  1. Gather context (git status, files, history)│
│  2. Build prompt (with optional mask)           │
│  3. Write to .palace/current_prompt.md          │
│  4. Invoke: claude -p "$(cat prompt.md)"        │
└─────────────┬───────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│  Claude:                                        │
│  1. Read prompt file                            │
│  2. Analyze project state                       │
│  3. Suggest actions (ACTIONS: section)          │
│  4. Execute selected actions                    │
│  5. Report results                              │
└─────────────┬───────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│  Palace:                                        │
│  1. Log action to history                       │
│  2. Update session state                        │
└─────────────┬───────────────────────────────────┘
              ↓
          (Loop continues)
```

### Action Selection Flow

```
User input: "do 1 and 3 but skip tests"
              ↓
Is simple? → No (contains "but", natural language)
              ↓
LLM Parser (Haiku) analyzes:
  Available actions: [
    {num: "1", label: "Write tests", ...},
    {num: "2", label: "Update docs", ...},
    {num: "3", label: "Deploy", ...}
  ]
              ↓
Returns: {
  selected_numbers: ["1", "3"],
  modifiers: ["skip tests"],
  is_custom_task: false
}
              ↓
Build action list with modifiers attached
              ↓
Return to user for execution
```

## File Structure

```
palace/
├── palace.py               # Core implementation (2235 lines)
├── requirements.txt        # Dependencies
├── .palace/               # Runtime data
│   ├── config.json        # Project config
│   ├── history.jsonl      # Action log
│   ├── sessions/          # Saved sessions
│   ├── masks/             # Expert personas
│   │   ├── available/
│   │   └── custom/
│   ├── skills/            # Command safety skills
│   └── current_prompt.md  # Latest prompt
├── tests/                 # Test suite (114 tests)
│   ├── test_core.py
│   ├── test_error_recovery.py
│   ├── test_mask_system.py
│   ├── test_mcp.py
│   ├── test_integration.py
│   └── ...
├── .github/workflows/     # CI/CD
│   ├── ci.yml
│   ├── release.yml
│   └── code-quality.yml
└── docs/
    ├── README.md
    ├── SPEC.md
    ├── QUICKSTART.md
    ├── ARCHITECTURE.md (this file)
    ├── CLAUDE.md          # Integration guide
    └── CONTRIBUTING.md
```

## Key Design Decisions

### 1. Why JSONL for history?

**Pros:**
- Append-only (fast writes)
- Line-by-line parsing (memory efficient)
- No need to read entire file to add entry
- Works well with tail/grep for quick analysis

**Cons:**
- Can't seek to middle efficiently
- No atomic updates (but we only append)

**Trade-off:** Optimized for write-heavy workload (logging) over read-heavy.

### 2. Why two-tier action parsing?

**Problem:** Natural language parsing requires LLM call (slow, costs $).

**Solution:**
- Detect if input is "simple" (just numbers/ranges)
- Use regex for simple cases (99% of inputs)
- Fall back to Haiku for complex natural language

**Result:** 99% of selections are instant, complex cases still work.

### 3. Why separate masks from prompts?

**Problem:** Different tasks need different expertise.

**Solution:**
- Masks are reusable expert personas
- Prompts are task-specific
- Composition allows combining multiple experts

**Benefit:** Mix and match expertise without duplicating knowledge.

### 4. Why MCP for permissions?

**Problem:** Claude needs permission for destructive operations.

**Solution:**
- MCP tool allows Claude to call back to Palace
- Palace logs all permission requests
- Over time, Palace learns what to auto-approve

**Future:** Smart permission system based on historical patterns.

## Performance Characteristics

| Operation | Time | API Calls | Context Used |
|-----------|------|-----------|--------------|
| Initialize | <10ms | 0 | 0 |
| Gather context | <50ms | 0 | ~1-2KB |
| Build prompt | <10ms | 0 | ~1-2KB |
| Simple selection | <1ms | 0 | 0 |
| Complex selection | ~500ms | 1 (Haiku) | ~200 tokens |
| Invoke Claude | ~2-30s | 1 (Sonnet) | Varies |
| Log action | <10ms | 0 | 0 |

**Total RHSI iteration:** ~2-30s (dominated by Claude invocation)

## Extension Points

### 1. Custom Masks

Add expert personas:

```bash
mkdir -p .palace/masks/custom/my-expert
cat > .palace/masks/custom/my-expert/SKILL.md <<EOF
---
name: my-expert
version: 1.0
priority: 5
---

# My Domain Expert

## Core Expertise
- Domain-specific knowledge
- Best practices
EOF
```

### 2. Custom Commands

Add to `palace.py`:

```python
def cmd_mycmd(self, args):
    """My custom command"""
    # Implementation

# In main():
commands['mycmd'] = palace.cmd_mycmd
```

### 3. Permission Logic

Customize MCP handler:

```python
@mcp.tool()
def handle_permission(request: dict) -> dict:
    tool = request.get("tool_name")

    # Auto-approve safe tools
    if tool in ["Read", "Grep", "Glob"]:
        return {"approved": True}

    # Require confirmation for writes
    if tool == "Write":
        return {
            "approved": False,
            "reason": "Write requires user confirmation"
        }
```

## Testing Strategy

**114 tests covering:**

- Core functionality (48 tests)
- Error recovery (16 tests)
- Integration scenarios (11 tests)
- Mask system (20 tests)
- MCP integration (5 tests)
- Mode detection (5 tests)
- Prompt building (8 tests)

**Test philosophy:**
- TDD: Tests written first
- Modular: Each test is independent
- Fast: All tests run in <1s
- Isolated: Uses temp directories

## Future Directions

### 1. Smart Permission System
- Learn from permission history
- Auto-approve safe patterns
- Flag unusual requests

### 2. Mask Marketplace
- Share expert masks with community
- Version control for masks
- Dependency resolution

### 3. Performance Profiling
- Track iteration times
- Identify bottlenecks
- Optimize context gathering

### 4. Multi-Agent Orchestration
- Parallel task execution
- Agent specialization
- Result aggregation

### 5. Interactive Mode Improvements
- Better action presentation
- Rich CLI output
- Progress visualization

---

**Key Takeaway:** Palace is a thin, efficient orchestration layer that maximizes Claude's capabilities while minimizing context overhead and complexity.
