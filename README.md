# Palace 🏛️

**Structure determines action.**

Palace is a self-improving agent orchestration layer for Claude Code that enables **Recursive Hierarchical Self Improvement (RHSI)**.

Instead of manually planning your next steps, ask Palace. It analyzes your project state, suggests intelligent actions, and learns from your workflow patterns over time.

## What is RHSI?

```
You: /pal-next
  ↓
Palace: Analyzes project → Suggests actions
  ↓
Claude: Executes your selection
  ↓
Palace: Logs results
  ↓
You: /pal-next (again)
  ↓
Palace: Suggests next steps based on what was just done
  ↓
... improvement accelerates
```

Each iteration builds on the last. Palace remembers what you've done and suggests logical next steps.

## Features

- **🎯 Intelligent Suggestions**: Analyzes git status, recent history, and project files to suggest what to do next
- **📦 Lightweight Context**: Sends metadata (1-2KB), not full files - leaves 99% of context for Claude to explore
- **🔄 Session Management**: Resume complex workflows, export/import sessions, track progress
- **🎭 Mask System**: Apply expert personas for specialized workflows (testing, security, architecture)
- **⚡ Error Recovery**: Automatic retry with backoff, graceful degradation, session checkpointing
- **🧪 Test-Driven**: Enforces TDD by default - every change requires tests
- **🔒 Strict Mode**: Validates tests before completion - ensures code quality (use `--yolo` to disable)
- **🎨 Flexible Selection**: Choose actions with numbers, ranges, modifiers, or natural language

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/yourusername/palace.git
cd palace

# Install dependencies (using uv recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Install Palace commands into Claude Code
python palace.py install
```

### Your First Session

```bash
# Initialize in your project
cd ~/my-project
python palace.py init

# Start Claude Code
claude

# Ask Palace what to do next
> /pal-next
```

**What happens:**

1. Palace analyzes your project state
2. Claude suggests multiple actionable next steps
3. You select which actions to execute
4. Claude performs the work
5. Palace logs the results
6. Repeat `/pal-next` - suggestions get smarter

## Core Commands

| Command | Description |
|---------|-------------|
| `pal next` | Suggest next steps (RHSI core) |
| `pal new <name>` | Create a new project |
| `pal scaffold` | Add best practices to existing project |
| `pal test` | Run project tests |
| `pal sessions` | List saved sessions |
| `pal cleanup` | Clean up old sessions |
| `pal export <id>` | Export session for sharing |
| `pal import <file>` | Import a session |
| `pal analyze` | Self-analysis of Palace metrics |
| `pal install` | Install Palace into Claude Code |

## Strict Mode 🔒

Palace enforces **strict mode** by default, ensuring all tests pass before a session completes.

### How It Works

1. **During Execution**: Claude can freely write and edit files
2. **At Completion**: Palace detects which tests are affected by modified files
3. **Validation**: Runs only the relevant tests (not the entire suite)
4. **Enforcement**: Session cannot complete until all affected tests pass

### Usage

```bash
# Default: Strict mode enabled
pal next

# Disable strict mode (YOLO mode)
pal next --yolo

# Also works with all commands
pal scaffold --yolo
```

### Why Strict Mode?

- **Prevents broken code**: Catches test failures immediately
- **Fast feedback**: Runs only affected tests, not the full suite
- **TDD enforcement**: Aligns with Golden Rule #1
- **CI/CD ready**: Code that completes in strict mode is deploy-ready

### YOLO Mode

Use `--yolo` to disable test validation:
- For rapid prototyping
- When tests don't exist yet
- For exploration and experimentation
- When you know what you're doing

**Note**: YOLO mode logs a warning to history for audit purposes.

## Provider Overrides 💎💰

Control which AI models Palace uses with provider override flags.

### Default Behavior

- **Normal mode** (`pal next`): Uses Claude Sonnet 4.5 (high quality)
- **Turbo mode** (`pal next -t`): Uses GLM-4.6 (cost-efficient)

### Override Flags

```bash
# Use Claude models even in turbo mode (higher quality, higher cost)
pal next -t --claude

# Use GLM even in normal mode (lower cost, faster)
pal next --glm

# Combine with other flags
pal next -t --claude --yolo  # Quality + speed
```

### When to Use

**`--claude` (ENGAGE mode)**
- You want max quality in turbo mode
- Complex reasoning tasks need Claude
- "I want to spend money for better results"

**`--glm` (Economy mode)**
- Cost savings in normal mode
- Simple tasks don't need Claude
- Fast iteration on straightforward work

Both flags work with all commands that invoke Claude.

## Documentation

- **[Quick Start](QUICKSTART.md)** - Get up and running in 5 minutes
- **[User Guide](docs/USER_GUIDE.md)** - Complete guide to using Palace
- **[Examples](docs/EXAMPLES.md)** - Real-world workflow examples
- **[Specification](SPEC.md)** - Technical specification and architecture
- **[Mask System](docs/MASK_SYSTEM.md)** - Expert personas and composition
- **[Error Recovery](docs/ERROR_RECOVERY.md)** - Resilience and retry strategies
- **[Architecture](ARCHITECTURE.md)** - System design and implementation
- **[Roadmap](ROADMAP.md)** - Future plans and features
- **[Contributing](CONTRIBUTING.md)** - How to contribute

## Philosophy

Palace follows two golden rules:

### 🥇 Golden Rule #1: TDD

**Every feature, every change, every improvement MUST have tests.**

- Tests are written FIRST
- Tests define the spec
- No commits without green tests
- Test coverage is mandatory, not optional

### 🥈 Golden Rule #2: Don't Be Prescriptive

**Offer options, don't dictate solutions.**

- Present multiple valid paths forward
- The user decides what to do
- Never restrict outputs artificially
- Let the user steer - you're a tool, not a boss

## Example Workflows

### Starting a New Feature

```bash
$ claude
> /pal-next

ACTIONS:
1. Write tests for user registration
2. Implement registration endpoint
3. Update API documentation

Select: 1

[Claude writes comprehensive tests]

> /pal-next

ACTIONS:
1. Implement registration endpoint (tests exist)
2. Add input validation
3. Update API documentation

Select: 1 2

[Claude implements to pass tests]
```

### Team Collaboration

```bash
# Export your workflow
$ python palace.py export pal-abc123 -o feature-workflow.json

# Teammate imports and continues
$ python palace.py import feature-workflow.json
$ python palace.py next --resume pal-xyz789
```

### Advanced Selection

```bash
# Simple numbers
> 1 2 3

# Ranges
> 1-5

# With modifiers
> 1 2 (use TypeScript and follow TDD)

# Natural language (uses LLM)
> do the first and third but skip the integration tests
```

## How Palace Works

Palace is a **thin orchestration layer** - it doesn't replace Claude, it enhances it:

1. **Gathers lightweight context** (file metadata, git status, recent actions)
2. **Generates focused prompts** for Claude to analyze
3. **Lets Claude use full capabilities** to explore and execute
4. **Logs actions to history** for learning and iteration
5. **Suggests smarter next steps** based on what was just done

Total overhead: ~1-2KB of context, leaving 99%+ for Claude to work.

## Architecture

```
┌─────────────────────────────────────────┐
│ User                                    │
│   ↓ /pal-next                           │
└──────────┬──────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ Palace (Orchestration)                  │
│ • Gather context (metadata only)        │
│ • Build prompt                          │
│ • Invoke Claude Code CLI                │
└──────────┬──────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ Claude (Execution)                      │
│ • Read files as needed                  │
│ • Analyze and plan                      │
│ • Use tools (Edit, Write, Bash, etc.)   │
│ • Suggest next actions                  │
└──────────┬──────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ Palace (Logging & Learning)             │
│ • Log actions to history                │
│ • Save session state                    │
│ • Learn patterns over time              │
└─────────────────────────────────────────┘
```

## MCP Integration

Palace requires MCP server registration to handle permissions during RHSI loops. **This is required for `pal next` to work properly.**

### Option 1: Manual Configuration (Recommended)

Add Palace to your `~/.claude.json` under the `mcpServers` key:

```json
{
  "mcpServers": {
    "palace": {
      "type": "stdio",
      "command": "/path/to/palace/.venv/bin/python",
      "args": [
        "/path/to/palace/palace.py"
      ],
      "env": {}
    }
  }
}
```

Replace `/path/to/palace` with your actual Palace installation path.

### Option 2: CLI Registration

```bash
# Install as MCP server (Claude Desktop)
uv run mcp install palace.py

# For Claude Code CLI
claude mcp add palace --scope user \
  $(pwd)/.venv/bin/python \
  $(pwd)/palace.py
```

### Troubleshooting

If `pal next` shows no output or immediately displays a prompt without any Claude response, the MCP server is likely not configured. Verify by checking:

```bash
jq '.mcpServers.palace' ~/.claude.json
```

If this returns `null`, Palace MCP is not registered.

Palace's `handle_permission` tool uses Haiku to assess safety of commands during RHSI loops.

## Requirements

- **Python 3.10+**
- **Claude Code CLI** ([install guide](https://github.com/anthropics/claude-code))
- **Anthropic API key**
- **uv** (recommended) or pip

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. **Write tests first** (TDD is mandatory)
4. Ensure all tests pass: `pytest tests/ -v`
5. Follow [Semantic Line Breaks](https://sembr.org/)
6. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

Palace is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for details.

## Links

- **Issues**: [GitHub Issues](https://github.com/yourusername/palace/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/palace/discussions)
- **Claude Code**: [anthropics/claude-code](https://github.com/anthropics/claude-code)

---

**Ready to let Palace guide your workflow?**

```bash
cd your-project
python palace.py init
claude
> /pal-next
```

Let the recursive improvement begin! 🏛️
