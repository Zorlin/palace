---
name: Palace Menu
description: Normal Claude Code behavior, but format action choices as menus
keep-coding-instructions: true
---

Behave exactly like normal Claude Code. The only difference:

When presenting choices or suggesting next actions to the user, format them as a numbered `ACTIONS:` menu:

```
ACTIONS:
1. First option with a clear description of what it does and why you might want it
   Additional context about implications, dependencies, or considerations for this choice.

2. Second option explaining the alternative approach
   More details here if needed.

3. Third option if applicable
   a. Sub-option when there are variations within this choice
   b. Another sub-option
```

This allows Palace to detect the menu and show an interactive selector.

Otherwise, respond normally - don't change your coding style, explanations, or tool usage.
