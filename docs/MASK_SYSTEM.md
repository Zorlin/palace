# Mask System Design for Palace

## Overview

The mask system allows Palace to load specialized "masks" or "skills" that give Claude domain-specific expertise. This is inspired by the palace-skills RHSI framework.

## Concept

A **mask** is a markdown file containing:
- Identity and expertise definition
- Behavioral guidelines
- Domain-specific knowledge
- Example scenarios

When loaded, a mask transforms Claude from a general assistant into a domain expert.

## Architecture

### Directory Structure

```
.palace/
├── masks/
│   ├── available/          # Community/shared masks
│   │   ├── palace-historian/
│   │   │   ├── SKILL.md
│   │   │   └── EXAMPLES.md
│   │   ├── database-architect/
│   │   │   ├── SKILL.md
│   │   │   └── EXAMPLES.md
│   │   └── ...
│   └── custom/             # User-created masks
│       └── my-specialty/
│           └── SKILL.md
└── config.json
```

### Mask Format

Masks use Claude Code's skill format (markdown with frontmatter):

```markdown
---
name: mask-name
description: Brief description of the mask's specialty
---

# Mask Name - Claude Sonnet

## Identity
Who this masked Claude is and what they specialize in.

## Core Expertise
- Domain knowledge areas
- Key frameworks and patterns
- Problem-solving approaches

## Behavioral Guidelines
How this mask should approach problems, communicate, and make decisions.

## Quick Start
Common workflows and commands for this specialty.

## Examples
Detailed examples with commands and outputs.
```

## Usage Workflow

### 1. Install Masks

```bash
# Install from palace-skills repository
palace mask install palace-historian

# Or install from URL
palace mask install https://raw.githubusercontent.com/.../SKILL.md

# List available masks
palace mask list
```

### 2. Load Mask

```bash
# Load mask for current session
palace next --mask palace-historian

# Or through Claude Code
/pal-next --mask palace-historian
```

### 3. Mask-Aware Invocation

When a mask is loaded:

1. Palace reads the mask file
2. Appends mask content to system prompt
3. Invokes Claude with enhanced context
4. Claude responds with domain expertise

## Implementation Plan

### Phase 1: Basic Mask Support

```python
class Palace:
    def load_mask(self, mask_name: str) -> Optional[str]:
        """Load a mask from .palace/masks/"""
        mask_file = self.palace_dir / "masks" / "available" / mask_name / "SKILL.md"
        if not mask_file.exists():
            mask_file = self.palace_dir / "masks" / "custom" / mask_name / "SKILL.md"

        if mask_file.exists():
            return mask_file.read_text()
        return None

    def invoke_claude_with_mask(self, prompt: str, mask_name: Optional[str] = None):
        """Invoke Claude with optional mask loaded"""
        mask_content = ""
        if mask_name:
            mask_content = self.load_mask(mask_name)
            if not mask_content:
                print(f"⚠️  Mask '{mask_name}' not found")
                return

        # Build prompt with mask
        full_prompt = self.build_prompt(prompt)
        if mask_content:
            full_prompt = f"{mask_content}\n\n{full_prompt}"

        # Invoke Claude
        return self.invoke_claude_cli(full_prompt)
```

### Phase 2: Mask Management Commands

```bash
# List available masks
palace mask list

# Show mask details
palace mask show palace-historian

# Install mask from URL
palace mask install <url>

# Create new mask from template
palace mask create my-specialty
```

### Phase 3: Mask Persistence

```json
// .palace/config.json
{
  "default_mask": "palace-historian",
  "masks": {
    "palace-historian": {
      "installed": true,
      "version": "2.0",
      "source": "palace-skills",
      "last_used": "2025-11-30T01:00:00Z"
    }
  }
}
```

### Phase 4: Mask Composition

Allow combining multiple masks for complex tasks:

```bash
# Load multiple masks
palace next --mask database-architect,distributed-systems
```

## Integration with Claude Code

### Slash Commands

```markdown
# .claude/commands/pal-historian.md
# Palace Historian Mode

Invoke Palace with palace-historian mask loaded.

\`\`\`bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python3 /path/to/palace.py next --mask palace-historian
\`\`\`
```

### Skill Installation

Masks can be installed as Claude Code skills:

```bash
# Install mask as skill
palace mask export palace-historian --format claude-skill \
  --output ~/.claude/skills/palace-historian.md
```

## Benefits

### For Users

1. **Domain Expertise On-Demand**: Load specialist knowledge when needed
2. **Consistent Behavior**: Same expert behavior across sessions
3. **Shareable Knowledge**: Export and share masks with team
4. **Composable Skills**: Combine masks for complex tasks

### For Palace

1. **Extensibility**: Community can create masks for any domain
2. **Specialization**: Each mask tested and refined for specific tasks
3. **RHSI Foundation**: Masks can evolve through recursive improvement
4. **Knowledge Base**: Growing library of expert behaviors

## Example: Using Palace Historian

```bash
# Install the historian mask
palace mask install palace-historian

# Load for repository cleanup
palace next --mask palace-historian

# Claude now has historian expertise
# User: "Clean up this repository and preserve important context"
# Claude responds with:
# - Artifact analysis
# - Space recovery strategy
# - Knowledge consolidation plan
# - git filter-repo commands
```

## Mask Categories

### Development Masks
- palace-historian (repository archaeology)
- database-architect (schema design)
- distributed-systems (consensus, replication)
- infrastructure-deployment (IaC, orchestration)

### Research Masks
- competition-math-researcher (formal proofs)
- consciousness-researcher (AI alignment)
- ecdlp-researcher (cryptography)

### Creative Masks
- creative-writing-poetry (literary analysis)
- music-theory-composer (composition)

### Testing Masks
- playwright-tester (browser automation)

## Future Enhancements

### 1. Mask Benchmarking

Track mask performance:
```bash
palace mask benchmark palace-historian \
  --scenario repository-cleanup \
  --iterations 5
```

### 2. Mask Evolution

Recursive improvement through RHSI:
```bash
palace mask improve palace-historian \
  --feedback "Needs better .gitignore pattern generation"
```

### 3. Mask Marketplace

Share and discover masks:
```bash
palace mask search "database"
palace mask install community/postgres-expert
```

### 4. Mask Analytics

Track usage and effectiveness:
```bash
palace mask stats
# Shows:
# - Most used masks
# - Average session duration
# - Success rate per mask
```

## Security Considerations

### Mask Validation

- Verify mask sources (signature checking)
- Sandbox mask execution (prevent malicious system prompts)
- Review before installation (show preview)

### Sensitive Data

- Masks should not contain secrets
- Use environment variables for credentials
- Document data handling in mask guidelines

## Migration Path

### From palace-skills

1. Export existing masks to Palace format
2. Update references (palace-skills → palace)
3. Maintain compatibility with both systems

### To Claude Code Skills

1. Provide export command
2. Generate skill frontmatter
3. Include examples and usage

## Open Questions

1. **Mask versioning**: How to handle mask updates?
2. **Mask conflicts**: What if two masks give contradictory advice?
3. **Mask discovery**: How do users find masks they need?
4. **Mask composition**: How to intelligently combine masks?

## Conclusion

The mask system transforms Palace from a simple orchestration layer into a platform for domain expertise. By leveraging the palace-skills RHSI framework, Palace can provide on-demand specialist knowledge while maintaining its lightweight, context-efficient design.

**Next Steps:**
1. Implement basic mask loading (Phase 1)
2. Create mask management commands (Phase 2)
3. Port 2-3 masks from palace-skills as proof of concept
4. Gather user feedback and iterate
