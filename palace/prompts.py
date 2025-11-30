"""
Prompt building functionality for Palace
"""

import json
from typing import Dict, Any, Optional, List


class PromptBuilder:
    """Handles prompt construction for Claude invocations"""

    def build_prompt(self, task_prompt: str, context: Dict[str, Any] = None) -> str:
        """Build a complete prompt with context for Claude"""
        # Allow subclasses to provide gather_context if context not provided
        if context is None and hasattr(self, 'gather_context'):
            context = self.gather_context()
        elif context is None:
            context = {}

        prompt_parts = [
            "# Palace Request\n",
            f"{task_prompt}\n",
            "\n## Project Context\n",
            f"```json\n{json.dumps(context, indent=2)}\n```\n",
            "\n## Instructions\n",
            "You are operating within Palace, a self-improving Claude wrapper.\n",
            "Use all your available tools to complete this task.\n",
            "When done, you can call Palace commands via bash if needed.\n"
        ]

        return "".join(prompt_parts)

    def build_prompt_with_steering(self, task_prompt: str, steering: str = None,
                                    context: Dict[str, Any] = None) -> str:
        """
        Build prompt with optional user steering context.

        If steering is provided, adds a prominent section for user direction.
        """
        base_prompt = self.build_prompt(task_prompt, context)

        if not steering:
            return base_prompt

        steering_section = f"""
## 🎯 USER STEERING

The user has provided this direction - prioritize it:

> {steering}

Take this into account as you continue the task.

"""
        # Insert steering after the task description
        parts = base_prompt.split("## Project Context", 1)
        if len(parts) == 2:
            return parts[0] + steering_section + "## Project Context" + parts[1]

        # Fallback: prepend steering
        return steering_section + base_prompt

    def _build_action_prompt(self, actions: List[dict]) -> str:
        """Build a prompt from selected actions"""
        if len(actions) == 1:
            action = actions[0]
            if action.get("_custom"):
                task_desc = action.get("label", "")
                return f"""Execute this task: {task_desc}

After completing, suggest possible next actions. Include an ACTIONS: section with multiple options."""
            else:
                task_desc = action.get("label", "")
                task_detail = action.get("description", "")
                modifiers = action.get("_modifiers", [])
                mod_text = f"\nModifications: {', '.join(modifiers)}" if modifiers else ""
                return f"""Execute this action: {task_desc}
{f"Details: {task_detail}" if task_detail else ""}{mod_text}

After completing, suggest possible next actions. Include an ACTIONS: section with multiple options."""
        else:
            tasks = []
            for a in actions:
                task = f"- {a.get('label', '')}"
                mods = a.get("_modifiers", [])
                if mods:
                    task += f" (modifications: {', '.join(mods)})"
                tasks.append(task)
            return f"""Execute these actions in order:
{chr(10).join(tasks)}

After completing all, suggest possible next actions. Include an ACTIONS: section with multiple options."""
