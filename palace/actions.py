"""
Action selection parsing for Palace
"""

import re
import json
from typing import List, Dict, Tuple


class ActionParser:
    """Handles parsing of user action selections"""

    def _is_simple_selection(self, selection: str) -> bool:
        """
        Check if selection is simple enough for regex parsing.

        Simple selections are:
        - Pure numbers: "1", "1 2 3", "1,2,3"
        - Ranges: "1-5", "2 - 4"
        - Numbers with parenthetical modifiers: "1 2 (use TDD)"

        Complex selections require LLM:
        - Natural language: "do the first and third"
        - Word modifiers: "1 but skip tests"
        - Ambiguous: "all of them except docs"
        """
        # Remove parenthetical content for simplicity check
        clean = re.sub(r'\([^)]+\)', '', selection).strip()

        # Check if remaining is only numbers, spaces, commas, dashes
        return bool(re.match(r'^[\d,\s\-]+$', clean))

    def _parse_selection_with_llm(self, selection: str, actions: List[dict]) -> dict:
        """
        Use Haiku to parse complex natural language selection.

        Returns dict with:
        - selected_numbers: List of action numbers to select
        - modifiers: List of modifier strings
        - is_custom_task: Whether this is a custom task (not selecting from menu)
        - custom_task: The custom task description if is_custom_task
        """
        import anthropic

        # Build action context for the LLM
        action_context = "\n".join([
            f"{a.get('num')}. {a.get('label')}: {a.get('description', '')}"
            for a in actions
        ])

        system_prompt = """You parse user input selecting from a menu of actions.

Return JSON with these fields:
- selected_numbers: array of action number strings the user wants (e.g., ["1", "3"])
- modifiers: array of instruction strings the user added (e.g., ["skip tests", "use TypeScript"])
- is_custom_task: true if user is describing a NEW task not in the menu
- custom_task: the custom task description if is_custom_task is true, else null

Examples:
- "1 2 3" → {"selected_numbers": ["1", "2", "3"], "modifiers": [], "is_custom_task": false, "custom_task": null}
- "do the first and third but skip tests" → {"selected_numbers": ["1", "3"], "modifiers": ["skip tests"], "is_custom_task": false, "custom_task": null}
- "all of them except documentation" → {"selected_numbers": ["1", "2", "3", "5"], "modifiers": ["except documentation"], "is_custom_task": false, "custom_task": null}
- "refactor the auth system" → {"selected_numbers": [], "modifiers": [], "is_custom_task": true, "custom_task": "refactor the auth system"}

IMPORTANT: Return ONLY valid JSON, no explanation."""

        prompt = f"""Available actions:
{action_context}

User selection: {selection}

Parse this and return JSON."""

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text.strip()

        # Parse JSON from response
        if "{" in response_text and "}" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            return json.loads(response_text[json_start:json_end])

        # Fallback if parsing fails
        return {
            "selected_numbers": [],
            "modifiers": [],
            "is_custom_task": True,
            "custom_task": selection
        }

    def _parse_simple_selection(self, selection: str) -> Tuple[List[str], List[str]]:
        """
        Fast regex-based parsing for simple numeric selections.

        Returns (numbers, modifiers) tuple.
        """
        modifiers = []

        # Extract parenthetical modifiers
        paren_match = re.search(r'\(([^)]+)\)', selection)
        if paren_match:
            modifiers.append(paren_match.group(1).strip())
            selection = re.sub(r'\([^)]+\)', '', selection)

        # Collect all numbers
        numbers = set()

        # Handle ranges
        range_matches = re.findall(r'(\d+)\s*-\s*(\d+)', selection)
        for start, end in range_matches:
            try:
                for i in range(int(start), int(end) + 1):
                    numbers.add(str(i))
            except:
                pass

        # Standalone numbers
        selection_no_ranges = re.sub(r'\d+\s*-\s*\d+', '', selection)
        standalone = re.findall(r'\b(\d+)\b', selection_no_ranges)
        for num in standalone:
            numbers.add(num)

        return sorted(numbers, key=lambda x: int(x)), modifiers

    def parse_action_selection(self, selection: str, actions: List[dict]) -> List[dict]:
        """
        Parse action selection string into list of actions.

        Supports:
        - Numeric: "1 2 3" or "1,2,3" or "1-5" or "1,3-5,7"
        - With modifiers: "1 2 3 (use the palace-skills repo as base)"
        - Natural language: "do 5 but skip the tests" (uses LLM)

        Returns list of selected actions, potentially modified by instructions.
        """
        selection = selection.strip()
        if not selection:
            return []

        # Check if simple enough for fast regex parsing
        if self._is_simple_selection(selection):
            numbers, modifiers = self._parse_simple_selection(selection)
        else:
            # Use LLM for complex natural language
            try:
                llm_result = self._parse_selection_with_llm(selection, actions)

                # Handle custom task
                if llm_result.get("is_custom_task"):
                    task = llm_result.get("custom_task", selection)
                    return [{"num": "c", "label": task, "description": "Custom task", "_custom": True}]

                numbers = llm_result.get("selected_numbers", [])
                modifiers = llm_result.get("modifiers", [])
            except Exception:
                # Fallback to regex on LLM error
                numbers, modifiers = self._parse_simple_selection(selection)
                # Check for text that might be a custom task
                text_without_nums = re.sub(r'[\d,\s\-]+', '', selection).strip()
                text_without_nums = re.sub(r'\([^)]+\)', '', text_without_nums).strip()
                if not numbers and text_without_nums:
                    return [{"num": "c", "label": selection, "description": "Custom task", "_custom": True}]

        # Build selected actions list
        selected = []
        for num in numbers:
            for a in actions:
                if a.get("num") == num:
                    action_copy = dict(a)
                    selected.append(action_copy)
                    break

        # Attach modifiers
        if modifiers:
            for a in selected:
                a["_modifiers"] = modifiers

        # Handle case where no actions matched but there's text
        if not selected:
            text_without_nums = re.sub(r'[\d,\s\-]+', '', selection).strip()
            text_without_nums = re.sub(r'\([^)]+\)', '', text_without_nums).strip()
            if text_without_nums:
                return [{"num": "c", "label": selection, "description": "Custom task", "_custom": True}]

        return selected
