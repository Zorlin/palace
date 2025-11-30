A hot-reloading, self-improving Claude wrapper that utilises
the full power of the Claude SDK, enabling Claude to gain capabilities
and learn from its mistakes through interactions with the user.

This builds both individual and collective intelligence - you can keep your own personal Masks, or use community masks to share knowledge and achieve goals. Contribute to the open source community to help improve Claude's capabilities and make it more accessible to everyone.

Palace is an experimental implementation of Recursive Hierarchical Self Improvement (RHSI) technology, which aims to create a self-improving system that can learn and adapt to new situations and environments. The system is designed to run on modest hardware, with support for local inference and decision-making.

# Basics
Palace starts with a simple self improvement loop -
a command installed in Claude that runs with the following instruction:

### Bootstrapping
Palace is designed to be used to build and reason about Palace itself, creating a harmonious feedback loop. Other than initial guidance and previous experiments, Palace will be written almost entirely using and inside Palace itself.

"Please read the entire Palace codebase so far including the synthesis documents, and suggest some concrete improvements to build it towards its goals. Aim to maintain compatibility with the following three targets:

1. Claude Code directly (plugins work via calling Palace via bash)
2. Claude Agent SDK
3. Claude Code without Palace (skills work without it)

In this early version, we suggest that you implement the following basic commands as Claude Commands:

Creating a project:
- `pal next` - Suggest what to do next in the project
- `pal new` - Create a project
- `pal scaffold` - Scaffold a project with best practices
- `pal test` - Test the project

Which will be installable with `python3 palace.py install`
Pay particularly close attention to "pal next" as a basic command (python3 palace.py next in our initial version) - looped correctly, this will unlock RHSI quickly."

# Rules
