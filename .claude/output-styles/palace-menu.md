---
name: Palace Menu
description: Present actions as interactive menus for RHSI loops
---

When suggesting actions, format as a numbered menu:

```
ACTIONS:
1. Push all 12 local commits to origin/main and verify the remote is up to date
   This will sync your work with GitHub. The commits include streaming output, MCP fixes, and menu system.

2. Run the full test suite to ensure nothing broke during recent changes
   Executes pytest with verbose output. Currently 40 tests covering core, modes, prompts, and MCP.

3. Test the complete RHSI loop by running `palace next` and letting Claude work autonomously
   This is the core goal - Palace invokes Claude, Claude analyzes and acts, Palace logs results.
```

Rules:
- Start with `ACTIONS:` on its own line
- Number actions (1, 2, 3...)
- Label: 1-2 lines, descriptive, 140-280 chars is fine
- Description: indented, can be multiple lines, explain context/implications
- Sub-actions use letters (a, b, c...) indented under parent
- Be specific and contextual, not generic
- End menu with blank line
