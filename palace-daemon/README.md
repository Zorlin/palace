# palace-daemon (Standalone - DEPRECATED)

**DEPRECATED**: This standalone implementation is superseded by the canonical workspace crates at `/crates/palace-daemon/`.

## Status

This directory contains a **reference implementation** of the swarm consciousness / context cache features that have NOT yet been ported to the canonical workspace structure.

### Canonical Location (USE THIS)

```
crates/
├── palace-core/       # Shared types (Anthropic, OpenAI formats)
├── palace-translator/ # API format conversion (the library crate)
├── palace-daemon/     # HTTP server binary (uses palace-translator)
└── palace-cli/        # Future CLI tooling
```

**Build the canonical binary:**
```bash
cargo build --release -p palace-daemon
```

### What's Here (Reference Only)

This standalone has features NOT yet in canonical:

| Module | Description | Port Status |
|--------|-------------|-------------|
| `lib.rs` | ContextCache with memory-mapped files, fsnotify | NOT PORTED |
| `block.rs` | Context block management (BlockBuilder, BlockContent) | NOT PORTED |
| `classifier.rs` | 1b classifier integration (qwen3:4b via Ollama) | NOT PORTED |
| `delta.rs` | Delta parsing for ++/-- context protocol | NOT PORTED |
| `spawn.rs` | Agent spawn DSL parser (+1=opus,2=sonnet syntax) | NOT PORTED |
| `timeline.rs` | DAG timeline for swarm consciousness | NOT PORTED |
| `translator.rs` | API translation (PORTED to palace-translator crate) | PORTED |
| `bin/daemon.rs` | HTTP server entry point (PORTED to crates/palace-daemon) | PORTED |

### SPEC.md

See `SPEC.md` for the full context router / swarm consciousness architecture specification. This describes the vision for:

- Block storage with compact display IDs + UUIDv7
- Per-agent timelines with classifier-managed views
- Retention tiers (SHORT/MEDIUM/LONG)
- Scope filtering (global, project, agent, tool)

### Future Work

These features should be ported to the canonical crates when ready:

1. **palace-context crate** - ContextCache, blocks, fsnotify
2. **palace-classifier crate** - 1b classifier, delta parsing
3. **palace-swarm crate** - Spawn DSL, timeline DAG

Until then, this code serves as reference for the architecture.

---

**DO NOT RUN THIS VERSION** - Use `cargo build --release -p palace-daemon` to build the canonical binary.
