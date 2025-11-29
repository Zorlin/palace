# Palace → Claude Code Integration

This document explains how Palace integrates with Claude Code to enable Recursive Hierarchical Self Improvement (RHSI).

## 🥇 Golden Rule #1: TDD (Test-Driven Development)

**EVERY feature, EVERY change, EVERY improvement MUST have tests.**

- Tests are written FIRST, implementation follows
- Tests are modular and build up incrementally
- Use pytest for all testing
- Tests define the spec - they ARE the documentation
- No pull request without tests
- No commit to main without green tests

This principle is encoded into Palace's DNA and applies to:
- Palace itself
- Any projects Palace builds
- Any code Claude writes through Palace
- Community contributions

**Test coverage is not optional - it's mandatory.**

## Permission Handling via MCP-Compliant Tool

Palace implements an MCP-compliant permission handler that Claude Code CLI invokes directly.

### Current Status: MCP Tool Registration Required

**Error you're seeing:**
```
Error: MCP tool python3 /home/wings/projects/palace/palace.py permissions (passed via --permission-prompt-tool) not found. Available MCP tools: none
```

**What this means:**
- Palace is passing `--permission-prompt-tool "python3 /path/to/palace.py permissions"` to Claude CLI
- Claude Code expects this to be a registered MCP tool NAME, not a command
- We need to register Palace as an MCP tool with Claude Code first

**What needs to happen:**
1. Figure out how to register MCP tools with Claude Code CLI
2. Register `palace-permissions` as an MCP tool that runs `python3 palace.py permissions`
3. Update Palace to use the registered tool name

**DO NOT remove the `--permission-prompt-tool` flag - this is a REQUIRED feature for RHSI safety.**

### How It Will Work (Once MCP Registration is Solved)

When you run `python3 palace.py next` in interactive mode, Palace invokes Claude with:

```bash
claude -p "prompt" \
  --permission-prompt-tool "palace-permissions"
```

The `--permission-prompt-tool` flag tells Claude Code to use Palace's permission handler for all permission requests during that session.

### The Permission Handler

`palace.py permissions` acts as an MCP-compliant tool:

1. **Receives requests** via stdin (stream-json format)
2. **Logs the request** to `.palace/history.jsonl`
3. **Approves/denies** based on Palace's permission logic
4. **Returns response** via stdout (stream-json format)

**This is NOT optional** - the permission system is core to Palace's ability to autonomously improve itself while maintaining safety.

### Learning from Permissions

Over time, Palace will:
- Track which permissions are frequently requested
- Identify patterns in RHSI loops
- Learn what should be auto-approved for efficiency
- Flag unusual requests for review

This creates a feedback loop where Palace becomes smarter about what changes are safe during self-improvement.

### Next Steps to Fix MCP Registration

1. Research Claude Code CLI MCP tool registration
2. Find where MCP tools are configured (likely `~/.claude-code/` somewhere)
3. Add Palace to that configuration
4. Test that `--permission-prompt-tool` works
5. Document the setup process here

## Overview

Palace is **not** a replacement for Claude - it's a thin orchestration layer that:
1. Gathers minimal context about the project state
2. Generates focused prompts for Claude
3. Lets Claude use its full capabilities to do the actual work

## The Flow

```
User types: /pal-next
    ↓
Claude Code executes: python3 palace.py next
    ↓
Palace gathers lightweight context
    ↓
Palace writes .palace/current_prompt.md
    ↓
Palace outputs: "CLAUDE: Please read the prompt file above and provide your analysis"
    ↓
Claude reads .palace/current_prompt.md
    ↓
Claude analyzes, decides, and executes using all available tools
    ↓
Claude can call Palace again via bash if needed (e.g., python3 palace.py scaffold)
```

## Lightweight Context Passing

Palace is designed to be **context efficient**. It does NOT dump entire files or massive logs.

### What Palace Provides

```json
{
  "project_root": "/path/to/project",
  "palace_version": "0.1.0",
  "files": {
    "README.md": {"exists": true, "size": 1481},
    "SPEC.md": {"exists": true, "size": 2058}
  },
  "git_status": " M .gitignore\n M README.md\n?? .palace/",
  "config": {...},
  "recent_history": [
    {"timestamp": 1234567890, "action": "next", "details": {...}},
    ...last 10 actions
  ]
}
```

**Total overhead: ~1-2KB typically**

### What Palace Does NOT Provide

- ❌ Full file contents (Claude reads files as needed)
- ❌ Complete git history (only current status)
- ❌ Massive logs (only last 10 actions)
- ❌ Redundant information (Claude has tools to explore)

## Claude's Responsibilities

When you (Claude) receive a Palace prompt, you should:

1. **Read the prompt file** - It contains the task and minimal context
2. **Assess what you need** - Use Glob/Grep/Read to explore further
3. **Make decisions** - You have the full context of the conversation
4. **Execute** - Use all your tools (Read, Write, Edit, Bash, etc.)
5. **Log if needed** - Palace will log your actions to history

## Example: /pal-next Workflow

### User Action
```
/pal-next
```

### Palace Output
```
🏛️  Palace - Invoking Claude for next step analysis...

📝 Context prepared at: .palace/current_prompt.md

PROJECT STATE:

Existing files:
  ✓ README.md
  ✓ SPEC.md

Git status:
  M README.md
  ?? new_file.py

────────────────────────────────────────────────────────────

🤖 CLAUDE: Please read the prompt file above and provide your analysis.
```

### Claude's Response Process

1. **Read the prompt**
   ```
   Read .palace/current_prompt.md
   ```

2. **Understand the ask**
   - The prompt says: "Analyze this project and suggest what to do next"
   - Context shows: README.md modified, new_file.py untracked

3. **Gather additional context as needed**
   ```
   Read README.md (to see what was changed)
   Read new_file.py (to understand what was added)
   Read SPEC.md (if it exists, to understand project goals)
   ```

4. **Analyze and decide**
   - Modified README probably documents new_file.py
   - Files are untracked → likely should commit
   - Check if tests exist for new_file.py

5. **Provide actionable suggestion**
   ```
   Based on the project state:

   Next Action: Write tests for new_file.py

   Why: You've added new_file.py and documented it in README.md,
   but there are no tests yet. Following TDD principles, we should
   test the new functionality.

   How to execute:
   1. Create tests/test_new_file.py
   2. Write test cases for the main functions
   3. Run: python3 palace.py test
   4. Commit when tests pass
   ```

6. **Optional: Execute immediately**

   If the action is clear and you have permission, just do it:
   ```
   Let me create tests for new_file.py...
   [Uses Write tool to create tests/test_new_file.py]
   [Uses Bash to run tests]
   [Provides results]
   ```

## Slash Commands

### /pal-next
**Purpose:** Ask Claude what to do next (RHSI core)

**Behavior:**
- Analyzes project state
- Suggests next logical action
- Claude decides and executes

**Use when:**
- You're stuck and want guidance
- Starting a work session
- After completing a task

### /pal-new <name>
**Purpose:** Create a new project

**Behavior:**
- Claude asks clarifying questions
- Sets up project structure
- Initializes Palace
- Creates foundational files

### /pal-scaffold
**Purpose:** Scaffold current project with best practices

**Behavior:**
- Detects project type
- Creates missing structure (tests/, docs/, etc.)
- Adds config files
- Sets up tooling

### /pal-test
**Purpose:** Run project tests

**Behavior:**
- Detects test framework
- Runs tests
- Analyzes results
- Suggests fixes if failures

## Context Efficiency Principles

1. **Metadata Over Content**
   - File sizes, not file contents
   - Git status, not full diff
   - Recent actions, not full history

2. **On-Demand Detail**
   - Palace gives you pointers
   - You (Claude) read what you need
   - No wasteful pre-loading

3. **Incremental History**
   - Only last N actions logged
   - Enough to show patterns
   - Not enough to bloat context

4. **Smart Defaults**
   - 10 history entries (not 1000)
   - Git status (not git log --all)
   - File list (not find -exec cat)

## Advanced: Bidirectional Communication

Palace can call Claude, and Claude can call Palace:

### Claude → Palace
```bash
python3 palace.py scaffold
python3 palace.py test
python3 palace.py next
```

### Palace → Claude
Creates prompts in .palace/current_prompt.md for Claude to read

This creates a feedback loop:
```
User → /pal-next → Palace → Claude → Executes → Logs action
                     ↑                              ↓
                     └──────── /pal-next ───────────┘
```

## Token Budget

Typical Palace invocation overhead:
- Context JSON: ~500-1000 tokens
- Prompt text: ~200-300 tokens
- **Total: ~700-1300 tokens**

This leaves ~99%+ of your context for:
- Reading actual project files
- Analyzing code
- Planning solutions
- Executing tasks

## Best Practices for Claude

1. **Always read the prompt file** - Don't assume what Palace wants
2. **Explore judiciously** - Only read files you need
3. **Be decisive** - Palace asks you to suggest AND execute
4. **Log important actions** - Palace tracks this for learning
5. **Use your tools** - Palace is minimal so you can be maximal
6. **Think recursively** - Your suggestions feed back into Palace

## The RHSI Loop

```
1. User: /pal-next
2. Palace: "Here's the project state, what should we do?"
3. Claude: Analyzes, suggests, executes
4. Palace: Logs the action
5. User: /pal-next (again)
6. Palace: "Given history, what's next?"
7. Claude: Builds on previous action
8. ... (improvement accelerates)
```

Over time:
- Palace's history grows richer
- Suggestions become more contextual
- Patterns emerge and are reused
- The system learns what works

This is **Recursive Hierarchical Self Improvement**.

## Example Session

```
$ python3 palace.py install
✅ Installed Palace commands to Claude Code

$ cd my-project
$ claude

> /pal-next

🏛️  Palace - Invoking Claude for next step analysis...
📝 Context prepared at: .palace/current_prompt.md

🤖 CLAUDE: [reads prompt, analyzes, suggests]
"I recommend adding tests. Let me create them..."
[creates tests/test_main.py]
[runs tests]
✅ Tests passing

> /pal-next

🏛️  Palace - Invoking Claude for next step analysis...
[Palace now knows tests were just added]

🤖 CLAUDE: [reads prompt with test history]
"Tests are green, README exists. Next: document the API..."
[documents API in README.md]
✅ Documentation complete

> /pal-next

🏛️  Palace - Invoking Claude for next step analysis...

🤖 CLAUDE: [reads prompt]
"Everything looks good. Ready to commit?"
[creates git commit]
✅ Committed
```

Each iteration builds on the last. Palace remembers. Claude improves.

---

**Remember:** Palace is YOUR tool. It organizes context so you can focus on what you do best - analyze, decide, create, and improve.
