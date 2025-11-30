# Palace User Guide

Complete guide to using Palace for Recursive Hierarchical Self Improvement (RHSI).

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Daily Workflow](#daily-workflow)
4. [Advanced Features](#advanced-features)
5. [Tips and Tricks](#tips-and-tricks)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### What You'll Learn

By the end of this guide, you'll be able to:

- Set up Palace in any project
- Use `/pal-next` to get intelligent suggestions
- Manage sessions for complex workflows
- Leverage masks for specialized knowledge
- Optimize your development workflow with RHSI

### First Time Setup

1. **Install Palace**

   ```bash
   cd palace
   source .venv/bin/activate
   python palace.py install
   ```

2. **Verify Installation**

   ```bash
   claude
   > /help
   # Should see /pal-* commands listed
   ```

3. **Initialize Your Project**

   ```bash
   cd ~/your-project
   python palace.py init
   ```

## Core Concepts

### The RHSI Loop

**Recursive Hierarchical Self Improvement** means each iteration builds on previous work:

```
┌─────────────────────────────────────┐
│ User runs: /pal-next                │
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Palace analyzes:                    │
│ • Git status                        │
│ • Recent history                    │
│ • Project files                     │
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Claude suggests actions:            │
│ 1. Write tests for new feature      │
│ 2. Update documentation             │
│ 3. Run type checker                 │
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ User selects: 1                     │
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Claude executes:                    │
│ • Creates test files                │
│ • Runs tests                        │
│ • Reports results                   │
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Palace logs action to history       │
└───────────┬─────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ User runs: /pal-next again          │
│ (Palace now knows tests were added) │
└─────────────────────────────────────┘
```

### Context Efficiency

Palace is lightweight by design:

| What Palace Sends | What Palace Doesn't Send |
|-------------------|-------------------------|
| File sizes        | File contents           |
| Git status        | Full git history        |
| Last 10 actions   | All history             |
| Project metadata  | Package dependencies    |

**Total overhead: ~1-2KB**

This leaves ~99% of context budget for Claude to read files as needed.

### Sessions

Sessions track multi-step workflows:

- **Session ID**: e.g., `pal-abc123`
- **Iteration**: Current step number
- **Pending Actions**: Queue of selected actions
- **Context**: Snapshot of project state

Sessions enable:
- Resuming work after interruption
- Sharing workflows with team
- Analyzing improvement patterns

## Daily Workflow

### Morning: Start a New Session

```bash
$ cd my-project
$ claude

> /pal-next
```

Palace analyzes overnight changes and suggests what to tackle first.

### During Development: Continuous Improvement

After making changes:

```bash
> /pal-next
```

Palace sees your recent work and suggests logical next steps:

```
ACTIONS:
1. Add tests for the new auth flow
2. Update API documentation
3. Run integration tests
4. Deploy to staging

Select: 1

[Claude creates comprehensive tests]

> /pal-next

ACTIONS:
1. Review test coverage report
2. Update API documentation (from before)
3. Run integration tests (from before)

Select: 2 3

[Claude updates docs and runs tests in sequence]
```

### End of Day: Session Cleanup

```bash
# Export your work session
python palace.py export pal-abc123 -o today.json

# Clean up old sessions
python palace.py cleanup --days 7
```

## Advanced Features

### Action Selection Patterns

#### Simple Numeric

```
> 1           # Select action 1
> 1 2 3       # Select multiple
> 1-5         # Select range
> 1,3-5,7     # Mixed format
```

#### With Modifiers

```
> 1 2 (use TypeScript)
> 3 (follow TDD strictly)
> 1-3 (prioritize performance)
```

#### Natural Language

Uses LLM to parse complex selections:

```
> do the first and third
> all of them except documentation
> 1 but skip the integration tests
```

#### Custom Tasks

If no numbers are provided, treated as custom task:

```
> refactor the authentication system
> optimize database queries
> add dark mode support
```

### Session Management

#### List Sessions

```bash
python palace.py sessions
```

Output:
```
📋 Saved Sessions:

  • pal-abc123
    Iteration: 5 | Pending: 2 actions | Updated: 2024-01-15 14:30

  • pal-def456
    Iteration: 3 | Pending: 0 actions | Updated: 2024-01-15 09:15
```

#### Resume Session

```bash
# Resume and select actions
python palace.py next --resume pal-abc123 --select "1,2"

# Resume with natural language
python palace.py next --resume pal-abc123 --select "do 1 but skip tests"
```

#### Export/Import Sessions

```bash
# Export for sharing
python palace.py export pal-abc123 -o workflow.json

# Import teammate's workflow
python palace.py import workflow.json
```

Export includes:
- Full session state
- Iteration history
- Action queue
- Project context

### Mask System

Masks are expert personas that guide Palace's behavior.

#### Available Masks

```bash
# List installed masks
ls .palace/masks/available/

# Example output:
# palace-historian/  - Focuses on documentation and knowledge management
# palace-tester/     - Emphasizes comprehensive testing
# palace-architect/  - System design and architecture
```

#### Using Masks

```bash
# Use single mask
/pal-next --mask palace-researcher

# Compose multiple masks
/pal-next --masks "palace-tester,palace-architect" --strategy merge
```

Composition strategies:

- **merge**: Concatenate masks in order
- **layer**: Apply by priority (frontmatter field)
- **blend**: Interleave sections from each mask

#### Creating Custom Masks

```bash
mkdir -p .palace/masks/custom/my-expert
```

Create `.palace/masks/custom/my-expert/SKILL.md`:

```markdown
---
priority: 10
description: Expert in my domain
---

# My Expert Mask

## Expertise

You are an expert in [domain]. When analyzing code, focus on:

1. [Specific aspect 1]
2. [Specific aspect 2]
3. [Specific aspect 3]

## Approach

Always prioritize [key principle].

## Workflow

When suggesting actions:
- Start with [first step type]
- Then move to [second step type]
- Finally, ensure [quality check]
```

### Error Recovery

Palace handles failures gracefully:

#### Retry with Backoff

```
🏛️  Invoking Claude...
❌ Error: Rate limit exceeded

⏳ Retry 1/3 in 1s...
⏳ Retry 2/3 in 2s...
✅ Success on retry 2
```

#### Graceful Degradation

On repeated failures, Palace falls back to:

1. Normal mode (streaming)
2. No-stream mode
3. Prompt file only
4. Fatal error (exits gracefully)

#### Session Checkpointing

Palace checkpoints session state before each iteration:

```bash
# If Claude crashes mid-iteration, resume from checkpoint
python palace.py next --resume pal-abc123
```

### User Steering (ESC-ESC)

Interrupt Claude mid-execution to provide guidance:

```
[Claude is working...]

Press ESC ESC

⏸️  ──────────────────────────────────────────────
   PALACE PAUSED - Enter steering command
   (Press Enter to resume, /abort to stop)
   ──────────────────────────────────────────────

🎯 Steer: Use more descriptive variable names

▶️  Resuming with steering...
```

Steering is logged and applied to the current task.

## Tips and Tricks

### Optimize for RHSI

**Start small, iterate fast:**

```
> /pal-next
Select: 1    # Do ONE thing well

> /pal-next  # Immediately check what's next
Select: 1    # Keep momentum
```

Rather than:

```
> /pal-next
Select: 1-10  # Trying to do everything at once
```

### Leverage History

Palace remembers. Use this:

```
# Morning
> /pal-next
"Yesterday you added auth. Today: test it, document it, deploy it."

# After each task
> /pal-next
"Tests pass. Next: docs, then deploy."
```

### Use Modifiers for Context

```
> 1 2 (reference the ADR in docs/architecture/)
> 3 (follow the pattern from src/users/)
> 4 (use the same approach as PR #123)
```

Modifiers are passed to Claude, giving extra context.

### Combine with Git Workflow

```bash
# Feature branch workflow
git checkout -b feature-x
/pal-next  # "Add feature X"
/pal-next  # "Test feature X"
/pal-next  # "Document feature X"
git commit -am "feat: X"

# Main branch
git checkout main
/pal-next  # "Review feature branches, prepare release"
```

### Create Project-Specific Masks

For long-term projects:

```bash
# masks/custom/myapp-expert/SKILL.md
```

Include:
- Architecture diagrams
- Coding conventions
- Testing patterns
- Deployment checklist

Palace will suggest actions that align with your project norms.

## Troubleshooting

### Issue: Actions not showing in menu

**Symptom**: Claude responds, but no `ACTIONS:` section appears.

**Solution**: Claude needs to format output correctly.

```bash
# Remind Claude in your next message:
> Please end your response with:

ACTIONS:
1. Action one
2. Action two
```

### Issue: Session corrupted

**Symptom**: `load_session` returns None or errors.

**Solution**:

```bash
# Try loading session file directly
cat .palace/sessions/pal-abc123.json

# If corrupted, delete it
rm .palace/sessions/pal-abc123.json

# Or clean up all sessions
python palace.py cleanup --all
```

### Issue: History growing too large

**Symptom**: Context includes too much history.

**Solution**:

```bash
# Trim history to last 100 entries
python palace.py cleanup --history --keep-history 100

# Or adjust in code (default is 10):
# palace.py line 103: context["recent_history"][-10:]
```

### Issue: MCP permissions not working

**Symptom**: Palace permission tool not found.

**Solution**:

```bash
# Re-register MCP server
uv run mcp install palace.py

# Verify registration
claude mcp list | grep palace

# Test permission handling
python palace.py permissions < test_request.json
```

### Issue: Commands not found

**Symptom**: `/pal-next` not recognized in Claude Code.

**Solution**:

```bash
# Reinstall commands
python palace.py install

# Verify installation
ls ~/.claude/commands/pal-*.md

# If still not working, check Claude Code version
claude --version  # Should be recent version
```

### Issue: Slow context gathering

**Symptom**: `gather_context()` takes a long time.

**Solution**:

Context should be fast. If slow:

1. Check git repo size: `du -sh .git/`
2. Large repo? Git status can be slow.
3. Consider: `git gc` to optimize repo.

Palace is designed to be lightweight - if context gathering is slow, something is unusual.

### Issue: Claude suggests same actions repeatedly

**Symptom**: Each `/pal-next` suggests the same thing.

**Possible causes**:

1. **Actions not being executed**: Make sure you select and Claude completes them.
2. **History not logging**: Check `.palace/history.jsonl` is being written.
3. **Git state unchanged**: Commit your work so Palace sees progress.

**Solution**:

```bash
# Check history is being written
tail -5 .palace/history.jsonl

# Commit work to show progress
git add .
git commit -m "Work in progress"

# Run /pal-next again
```

---

## Summary

Palace is designed to get out of your way:

1. **Initialize once**: `python palace.py init`
2. **Ask repeatedly**: `/pal-next`
3. **Let it learn**: Palace adapts to your workflow

The more you use RHSI, the smarter Palace becomes at suggesting what to do next.

**Ready to master Palace?** Try running `/pal-next` three times in a row on a real project. Watch how suggestions evolve.

🏛️ **Structure determines action.**
