# Palace Quick Start Guide

Get up and running with Palace in 5 minutes.

## What is Palace?

Palace is a self-improving agent wrapper for Claude Code that enables **Recursive Hierarchical Self Improvement (RHSI)**.

Instead of manually planning your next steps, Palace analyzes your project state and suggests what to do next - then Claude executes it. Over time, Palace learns from your workflow patterns and becomes smarter about what to suggest.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **uv** package manager (recommended) or pip
- **Claude Code CLI** installed ([installation guide](https://github.com/anthropics/claude-code))
- **Anthropic API key** set in your environment

## Installation

### 1. Clone and Install Palace

```bash
# Clone the repository
git clone https://github.com/yourusername/palace.git
cd palace

# Install dependencies with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 3. Install Palace Commands

```bash
# Install Palace as a slash command in Claude Code
python palace.py install
```

This registers `/pal-*` commands in Claude Code.

### 4. (Optional) Install as MCP Server

To enable Palace's permission handling tools:

```bash
# Install Palace as an MCP server
uv run mcp install palace.py

# For Claude Code CLI (global)
claude mcp add palace --scope user \
  $(pwd)/.venv/bin/python \
  $(pwd)/palace.py
```

## Your First Palace Session

### Initialize a Project

Navigate to any project and initialize Palace:

```bash
cd ~/my-project
python palace.py init
```

This creates `.palace/` directory with configuration and history tracking.

### Ask Palace What's Next

The core of Palace is the `/pal-next` command:

```bash
# Start Claude Code
claude

# In Claude Code, run:
> /pal-next
```

**What happens:**

1. Palace analyzes your project (git status, files, recent history)
2. Generates a focused prompt for Claude
3. Claude reads the prompt and suggests next actions
4. You can select actions to execute immediately

### Example Session

```
$ claude

> /pal-next

🏛️  Palace - Invoking Claude for next step analysis...
📝 Context prepared at: .palace/current_prompt.md

[Claude analyzes your project]

Based on your project state, here are suggested next actions:

ACTIONS:
1. Write tests for new_file.py
   You added new_file.py but it lacks test coverage.

2. Update README with new functionality
   Document the changes you made.

3. Run type checker (mypy)
   Ensure type safety before committing.

Select actions: 1

[Claude creates tests, runs them, reports results]
✅ Tests created and passing
```

## Core Commands

### Project Commands

```bash
# Suggest next steps (the main RHSI loop)
/pal-next

# Create a new project with scaffolding
/pal-new my-app

# Add best practices to existing project
/pal-scaffold

# Run tests
/pal-test
```

### Session Management

```bash
# List saved sessions
python palace.py sessions

# Resume a previous session
python palace.py next --resume SESSION_ID

# Clean up old sessions
python palace.py cleanup --days 30
```

### Mask System

Masks are expert personas that guide Palace's behavior:

```bash
# Use a mask for specialized workflows
/pal-next --mask palace-researcher

# List available masks
python palace.py masks list

# Create a custom mask
mkdir -p .palace/masks/custom/my-expert
echo "# My Expert Mask..." > .palace/masks/custom/my-expert/SKILL.md
```

## The RHSI Loop

Palace implements **Recursive Hierarchical Self Improvement**:

```
User: /pal-next
  ↓
Palace: Analyzes project → Suggests actions
  ↓
Claude: Executes selected actions
  ↓
Palace: Logs results to history
  ↓
User: /pal-next (again)
  ↓
Palace: Suggests next actions based on what was just done
  ↓
... (improvement accelerates)
```

Each iteration builds on the last. Palace remembers what you've done and suggests logical next steps.

## Key Features

### 1. Context Efficiency

Palace is designed to be lightweight:

- Sends **metadata**, not full file contents (~1-2KB context)
- Claude reads files on-demand using Read/Grep tools
- Only logs recent actions (configurable limit)
- Git status, not full history

### 2. Test-Driven Development

Palace enforces TDD by default:

- Every change requires tests
- Tests are written first
- No commits without green tests
- Automated test running

### 3. Error Recovery

Built-in resilience for Claude CLI invocations:

- Retry with exponential backoff for transient errors
- Graceful degradation on repeated failures
- Session checkpointing for recovery
- Smart error classification

### 4. Action Selection

Flexible ways to select actions:

```
# Simple numbers
> 1 2 3

# Ranges
> 1-5

# With modifiers
> 1 2 (use TypeScript)

# Natural language (uses LLM)
> do the first and third but skip tests
```

## Configuration

Palace stores configuration in `.palace/config.json`:

```json
{
  "name": "my-project",
  "version": "0.1.0",
  "palace_version": "0.1.0",
  "initialized": true
}
```

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY="your-api-key"

# Optional
PALACE_LOG_LEVEL="INFO"
PALACE_MAX_HISTORY=50
```

## Advanced Usage

### Custom Action Selection

Use natural language to select and modify actions:

```
> do 1 and 3 but skip the integration tests
> all of them except documentation
> refactor the auth system (custom task)
```

### Session Resumption

Resume a previous session with selected actions:

```bash
# List sessions
python palace.py sessions

# Resume with specific actions
python palace.py next --resume pal-abc123 --select "1,3-5"
```

### Permission System

When running Claude Code in permission mode:

```bash
# Claude will call Palace's MCP tool for permission handling
claude -p "your prompt" \
  --permission-prompt-tool "mcp__palace__handle_permission"
```

Palace tracks permissions in `.palace/history.jsonl` for learning over time.

## Troubleshooting

### Palace commands not found in Claude Code

```bash
# Reinstall
python palace.py install

# Verify installation
cat ~/.claude/commands/pal-next.md
```

### MCP server not working

```bash
# Check MCP registration
claude mcp list

# Should see: palace

# Reinstall if needed
uv run mcp install palace.py
```

### Tests failing

```bash
# Run tests manually to see detailed output
source .venv/bin/activate
pytest tests/ -v

# Run specific test
pytest tests/test_core.py::TestConfigManagement -v
```

### Session state issues

```bash
# Clean up corrupted sessions
python palace.py cleanup --all

# Start fresh
rm -rf .palace/sessions/*
```

## Next Steps

Now that you're up and running:

1. **Read [SPEC.md](SPEC.md)** - Understand Palace's architecture
2. **Explore masks** - Check `.palace/masks/available/` for expert personas
3. **Try RHSI** - Run `/pal-next` multiple times in a row to see improvement
4. **Contribute** - Read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/palace/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/palace/discussions)
- **Documentation**: See [docs/](docs/) directory

## Philosophy

Palace follows two golden rules:

### 🥇 Golden Rule #1: TDD

**Every feature, every change, every improvement MUST have tests.**

- Tests are written FIRST
- Tests define the spec
- No commits without green tests

### 🥈 Golden Rule #2: Don't Be Prescriptive

**Offer options, don't dictate solutions.**

- Present multiple valid paths
- The user decides what to do
- Never restrict outputs artificially

---

**Ready to let Palace guide your workflow?**

```bash
cd your-project
python palace.py init
claude
> /pal-next
```

Let the recursive improvement begin! 🏛️
