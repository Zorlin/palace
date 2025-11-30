#!/usr/bin/env python3
"""
Palace - A self-improving Claude wrapper
Recursive Hierarchical Self Improvement (RHSI) for Claude

Palace is NOT a replacement for Claude - it's an orchestration layer.
Every command invokes Claude with context, letting Claude use its full power.

This file serves dual purposes:
1. CLI tool for invoking Claude (python3 palace.py next)
2. MCP server providing tools to Claude (uv run mcp install palace.py)
"""

import sys
import os
import json
import subprocess
import re
import shutil
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

VERSION = "0.1.0"

def is_interactive() -> bool:
    """Detect if Palace is running interactively (can invoke Claude) vs from within Claude Code"""
    # Check if we're in a CI environment or being called from Claude Code
    if os.getenv("CI") or os.getenv("CLAUDE_CODE_SESSION"):
        return False
    # Check if stdin is a TTY (terminal)
    return sys.stdin.isatty()

class Palace:
    """Palace orchestration layer - coordinates Claude invocations"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.palace_dir = self.project_root / ".palace"
        self.config_file = self.palace_dir / "config.json"

    def ensure_palace_dir(self):
        """Ensure .palace directory exists"""
        self.palace_dir.mkdir(exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        """Load Palace configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}

    def save_config(self, config: Dict[str, Any]):
        """Save Palace configuration"""
        self.ensure_palace_dir()
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def gather_context(self) -> Dict[str, Any]:
        """Gather project context for Claude"""
        context = {
            "project_root": str(self.project_root),
            "palace_version": VERSION,
            "files": {},
            "git_status": None,
            "config": self.load_config()
        }

        # Check for important files
        important_files = ["README.md", "SPEC.md", "ROADMAP.md", "package.json",
                          "requirements.txt", "Cargo.toml", "go.mod"]

        for filename in important_files:
            filepath = self.project_root / filename
            if filepath.exists():
                context["files"][filename] = {
                    "exists": True,
                    "size": filepath.stat().st_size
                }

        # Get git status if in a repo
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                context["git_status"] = result.stdout
        except:
            pass

        # Load history
        history_file = self.palace_dir / "history.jsonl"
        if history_file.exists():
            context["recent_history"] = []
            with open(history_file, 'r') as f:
                for line in f:
                    if line.strip():
                        context["recent_history"].append(json.loads(line))
            context["recent_history"] = context["recent_history"][-10:]  # Last 10

        return context

    def log_action(self, action: str, details: Dict[str, Any] = None):
        """Log action to history"""
        self.ensure_palace_dir()
        history_file = self.palace_dir / "history.jsonl"

        entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details or {}
        }

        with open(history_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    # ========================================================================
    # Session Management
    # ========================================================================

    def _generate_session_id(self) -> str:
        """Generate a short, memorable session ID"""
        # Use short UUID prefix + timestamp suffix for uniqueness
        return f"pal-{uuid.uuid4().hex[:6]}"

    def _get_sessions_dir(self) -> Path:
        """Get the sessions directory"""
        sessions_dir = self.palace_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return sessions_dir

    def save_session(self, session_id: str, state: Dict[str, Any]):
        """Save session state for later resumption"""
        sessions_dir = self._get_sessions_dir()
        session_file = sessions_dir / f"{session_id}.json"

        state["session_id"] = session_id
        state["updated_at"] = time.time()

        with open(session_file, 'w') as f:
            json.dump(state, f, indent=2)

        return session_file

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a saved session state"""
        sessions_dir = self._get_sessions_dir()
        session_file = sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            return None

        with open(session_file, 'r') as f:
            return json.load(f)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions"""
        sessions_dir = self._get_sessions_dir()
        sessions = []

        for session_file in sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    state = json.load(f)
                    sessions.append({
                        "session_id": state.get("session_id", session_file.stem),
                        "updated_at": state.get("updated_at"),
                        "iteration": state.get("iteration", 0),
                        "pending_actions": len(state.get("pending_actions", []))
                    })
            except:
                pass

        return sorted(sessions, key=lambda s: s.get("updated_at", 0), reverse=True)

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
            model="claude-haiku-4-20250514",
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

    def build_prompt(self, task_prompt: str, context: Dict[str, Any] = None) -> str:
        """Build a complete prompt with context for Claude"""
        full_context = context or self.gather_context()

        prompt_parts = [
            "# Palace Request\n",
            f"{task_prompt}\n",
            "\n## Project Context\n",
            f"```json\n{json.dumps(full_context, indent=2)}\n```\n",
            "\n## Instructions\n",
            "You are operating within Palace, a self-improving Claude wrapper.\n",
            "Use all your available tools to complete this task.\n",
            "When done, you can call Palace commands via bash if needed.\n"
        ]

        return "".join(prompt_parts)

    # ========================================================================
    # Mask System
    # ========================================================================

    def load_mask(self, mask_name: str) -> Optional[str]:
        """
        Load a mask from .palace/masks/

        Searches in order:
        1. .palace/masks/available/{mask_name}/SKILL.md
        2. .palace/masks/custom/{mask_name}/SKILL.md

        Returns mask content or None if not found.
        """
        # Try available masks first
        mask_file = self.palace_dir / "masks" / "available" / mask_name / "SKILL.md"
        if mask_file.exists():
            return mask_file.read_text()

        # Try custom masks
        mask_file = self.palace_dir / "masks" / "custom" / mask_name / "SKILL.md"
        if mask_file.exists():
            return mask_file.read_text()

        return None

    def get_mask_metadata(self, mask_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from mask frontmatter.

        Parses YAML frontmatter if present, otherwise returns minimal metadata.
        """
        content = self.load_mask(mask_name)
        if not content:
            return None

        metadata = {"name": mask_name}

        # Check for frontmatter (--- at start)
        if content.startswith("---"):
            try:
                # Extract frontmatter block
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter_text = parts[1].strip()
                    # Simple YAML-like parsing (key: value)
                    for line in frontmatter_text.split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            metadata[key.strip()] = value.strip()
            except:
                pass

        return metadata

    def list_masks(self) -> List[Dict[str, Any]]:
        """
        List all available masks.

        Returns list of dicts with:
        - name: mask name
        - type: "available" or "custom"
        - path: full path to mask
        """
        masks = []
        masks_dir = self.palace_dir / "masks"

        if not masks_dir.exists():
            return masks

        # List available masks
        available_dir = masks_dir / "available"
        if available_dir.exists():
            for mask_dir in available_dir.iterdir():
                if mask_dir.is_dir() and (mask_dir / "SKILL.md").exists():
                    masks.append({
                        "name": mask_dir.name,
                        "type": "available",
                        "path": str(mask_dir / "SKILL.md")
                    })

        # List custom masks
        custom_dir = masks_dir / "custom"
        if custom_dir.exists():
            for mask_dir in custom_dir.iterdir():
                if mask_dir.is_dir() and (mask_dir / "SKILL.md").exists():
                    masks.append({
                        "name": mask_dir.name,
                        "type": "custom",
                        "path": str(mask_dir / "SKILL.md")
                    })

        return masks

    def build_prompt_with_mask(self, task_prompt: str, mask_name: Optional[str] = None,
                               context: Dict[str, Any] = None) -> Optional[str]:
        """
        Build prompt with optional mask loaded.

        If mask_name is provided, prepends mask content to the prompt.
        Returns None if mask not found.
        """
        # Load mask if specified
        mask_content = None
        if mask_name:
            mask_content = self.load_mask(mask_name)
            if not mask_content:
                return None

        # Build base prompt
        base_prompt = self.build_prompt(task_prompt, context)

        # Prepend mask content if loaded
        if mask_content:
            return f"{mask_content}\n\n{base_prompt}"

        return base_prompt

    # ========================================================================
    # Error Recovery
    # ========================================================================

    def is_transient_error(self, exit_code: int, error_msg: str = "") -> bool:
        """
        Determine if an error is transient (retryable) or permanent.

        Transient errors:
        - Network timeouts
        - Rate limits (429)
        - Temporary file locks

        Permanent errors:
        - Permission denied (403)
        - User interrupts (Ctrl-C, exit code 130)
        - Invalid syntax
        """
        # User interrupt
        if exit_code == 130:
            return False

        # Permission errors
        if exit_code == 403:
            return False

        # Rate limits are transient
        if exit_code == 429:
            return True

        # Check error message for transient indicators
        error_lower = error_msg.lower()
        transient_keywords = ["timeout", "network", "connection", "temporary", "rate limit"]
        permanent_keywords = ["permission denied", "forbidden", "interrupt", "syntax"]

        if any(kw in error_lower for kw in permanent_keywords):
            return False

        if any(kw in error_lower for kw in transient_keywords):
            return True

        # Default: assume non-zero exit codes from Claude are retryable
        return exit_code != 0

    def checkpoint_session(self, session_id: str, state: Dict[str, Any]):
        """
        Checkpoint session state for recovery.

        Similar to save_session but adds checkpoint_at timestamp.
        """
        state["checkpoint_at"] = time.time()
        self.save_session(session_id, state)

    def log_retry_attempt(self, attempt: int, max_retries: int, error: str, wait_time: float):
        """Log retry attempt to history"""
        self.log_action("retry_attempt", {
            "attempt": attempt,
            "max_retries": max_retries,
            "error": error,
            "wait_time": wait_time
        })

    def log_error_recovery(self, error_type: str, recovered: bool, attempts: int):
        """Log error recovery outcome to history"""
        self.log_action("error_recovery", {
            "error_type": error_type,
            "recovered": recovered,
            "attempts": attempts
        })

    def get_degradation_mode(self, attempt: int) -> str:
        """
        Get degradation mode based on failure attempt number.

        Returns:
        - "retry": Normal retry
        - "no-stream": Disable streaming
        - "prompt-file": Fall back to prompt file only
        - "fatal": Fatal error, exit
        """
        if attempt == 0:
            return "retry"
        elif attempt == 1:
            return "no-stream"
        elif attempt == 2:
            return "prompt-file"
        else:
            return "fatal"

    def invoke_with_retry(self, prompt: str, max_retries: int = 3) -> Tuple[int, Optional[List[dict]]]:
        """
        Invoke Claude with retry and exponential backoff.

        For transient failures (network, rate limits), retries with
        exponential backoff: 1s, 2s, 4s, etc.

        Returns (exit_code, actions) tuple.
        """
        for attempt in range(max_retries):
            try:
                exit_code, actions = self.invoke_claude_cli(prompt)

                # Success
                if exit_code == 0:
                    if attempt > 0:
                        # Log successful recovery
                        self.log_error_recovery("transient_error", recovered=True, attempts=attempt + 1)
                    return exit_code, actions

                # Check if error is transient
                if not self.is_transient_error(exit_code):
                    # Permanent error, don't retry
                    self.log_error_recovery("permanent_error", recovered=False, attempts=attempt + 1)
                    return exit_code, actions

                # Transient error - retry with backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, ...
                    self.log_retry_attempt(attempt + 1, max_retries, f"Exit code {exit_code}", wait_time)
                    print(f"⏳ Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                    time.sleep(wait_time)

            except Exception as e:
                if attempt == max_retries - 1:
                    self.log_error_recovery("exception", recovered=False, attempts=max_retries)
                    raise

                wait_time = 2 ** attempt
                self.log_retry_attempt(attempt + 1, max_retries, str(e), wait_time)
                print(f"⚠️  Error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)

        # All retries exhausted
        self.log_error_recovery("exhausted_retries", recovered=False, attempts=max_retries)
        return 1, None

    def invoke_claude_cli(self, prompt: str) -> Tuple[int, Optional[List[dict]]]:
        """
        Invoke Claude Code CLI directly (interactive mode)

        Returns tuple of (exit_code, selected_actions)
        - exit_code: 0 on success, non-zero on error
        - selected_actions: list of action dicts if user selected any, None otherwise
        """
        # Menu format instructions for action selection
        menu_prompt = """IMPORTANT: End your response with suggested next actions using this EXACT format:

ACTIONS:
1. First option with description
   Additional context if needed.

2. Second option
   More details here.

The "ACTIONS:" header is required (exact spelling with colon) - it triggers the interactive menu system."""

        cmd = [
            "claude",
            "-p", prompt,
            "--model", "claude-sonnet-4-5",
            "--append-system-prompt", menu_prompt,
            "--verbose",
            "--include-partial-messages",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--dangerously-skip-permissions"  # Auto-approve for RHSI loops
        ]

        print("🏛️  Palace - Invoking Claude...")
        print()

        try:
            # Stream output and parse JSON for succinct display
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            selected_action = self._process_stream_output(process.stdout)
            process.wait()
            return process.returncode, selected_action

        except FileNotFoundError:
            print("❌ Claude Code CLI not found. Make sure 'claude' is in your PATH.")
            print("   Install from: https://code.claude.com/")
            return 1, None
        except Exception as e:
            print(f"❌ Error invoking Claude: {e}")
            return 1, None

    def _parse_actions_menu(self, text: str) -> List[dict]:
        """Parse actions from text - supports multiple formats"""
        actions = []

        # Try multiple action header patterns
        action_section = None
        patterns = [
            ("ACTIONS:", 1),
            ("## Next Action:", 0),  # Single action format
            ("**Next Action:**", 0),
            ("Next steps:", 1),
            ("Suggested actions:", 1),
        ]

        for pattern, _ in patterns:
            if pattern in text:
                action_section = text.split(pattern, 1)[1].strip()
                break

        # Also detect numbered lists anywhere in the text
        if not action_section:
            # Look for numbered list patterns like "1. Do something"
            numbered = re.findall(r'^\s*(\d+)\.\s+\*?\*?([^\n*]+)', text, re.MULTILINE)
            if numbered:
                for num, label in numbered[:7]:  # Max 7 actions
                    actions.append({
                        "num": num,
                        "label": label.strip(),
                        "description": "",
                        "subactions": []
                    })
                return actions
            return []

        lines = action_section.split("\n")
        current_action = None

        for line in lines:
            # Empty line ends menu
            if not line.strip() and current_action:
                actions.append(current_action)
                current_action = None
                continue

            # Numbered action: "1. Label text here"
            match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if match:
                if current_action:
                    actions.append(current_action)
                current_action = {
                    "num": match.group(1),
                    "label": match.group(2),
                    "description": "",
                    "subactions": []
                }
                continue

            # Sub-action: "   a. Sub label"
            match = re.match(r'^\s+([a-z])\.\s+(.+)$', line)
            if match and current_action:
                current_action["subactions"].append({
                    "letter": match.group(1),
                    "label": match.group(2)
                })
                continue

            # Description line (indented, no number/letter)
            if line.startswith("   ") and current_action:
                if current_action["description"]:
                    current_action["description"] += " " + line.strip()
                else:
                    current_action["description"] = line.strip()

        if current_action:
            actions.append(current_action)

        return actions

    def _format_action_choice(self, action: dict, width: int = 100) -> str:
        """Format action for display with truncated description"""
        num = action.get("num", "?")
        label = action.get("label", "")
        desc = action.get("description", "")

        # Build the display string
        prefix = f"{num}. {label}"
        if desc:
            # Truncate description to fit width
            remaining = width - len(prefix) - 3  # " - " separator
            if remaining > 20:
                truncated = desc[:remaining] if len(desc) <= remaining else desc[:remaining-1] + "…"
                return f"{prefix} - {truncated}"
        return prefix

    def _show_action_menu(self, actions: List[dict]) -> Optional[List[dict]]:
        """Show rich interactive menu with multi-select, return selected actions or None to exit"""
        try:
            import questionary
            from questionary import Style
        except ImportError:
            # Fallback to simple numbered input
            return self._show_simple_menu(actions)

        # Get terminal width for formatting
        try:
            import shutil
            term_width = shutil.get_terminal_size().columns - 10
        except:
            term_width = 90

        # Custom style for the menu
        custom_style = Style([
            ('qmark', 'fg:yellow bold'),
            ('question', 'fg:cyan bold'),
            ('pointer', 'fg:cyan bold'),
            ('highlighted', 'fg:cyan bold'),
            ('selected', 'fg:green'),
            ('separator', 'fg:gray'),
            ('instruction', 'fg:gray italic'),
        ])

        # Build choices for checkbox menu
        choices = []
        for a in actions:
            display = self._format_action_choice(a, term_width)
            choices.append(questionary.Choice(display, value=a))

        print()
        print("💡 Select next action(s):")
        print("   KB: ↑/↓ navigate | Space select | Enter run | Esc/q cancel")
        print()

        try:
            selected = questionary.checkbox(
                "",
                choices=choices,
                style=custom_style,
                instruction="",
            ).ask()
        except KeyboardInterrupt:
            return None

        if selected is None or len(selected) == 0:
            return None

        return selected

    def _show_simple_menu(self, actions: List[dict]) -> Optional[List[dict]]:
        """Fallback menu with rich formatting"""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.prompt import Prompt

            console = Console()
            console.print()
            console.print("[bold cyan]💡 Select action(s):[/bold cyan]")
            console.print("[dim]   Numbers (space or comma separated), 0 to exit, or type custom task[/dim]")
            console.print()

            # Create a nice table
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Num", style="bold yellow", width=4)
            table.add_column("Action", style="white")

            for a in actions:
                # Truncate long labels
                label = a['label'][:80] + "..." if len(a['label']) > 80 else a['label']
                table.add_row(f"{a['num']}.", label)

            table.add_row("[dim]0.[/dim]", "[dim]Exit loop[/dim]")
            console.print(table)
            console.print()

            choice = Prompt.ask("[bold green]>[/bold green]")

        except ImportError:
            # Plain fallback if rich not available
            print("\n💡 Select action(s) (space/comma separated, 0 to exit, or type custom task):")
            for a in actions:
                print(f"  {a['num']}. {a['label'][:80]}")
            print("  0. Exit loop")
            choice = input("\n> ").strip()

        if not choice or choice == "0" or choice.lower() in ("q", "quit", "exit"):
            return None

        # Parse the selection using the smart parser
        return self.parse_action_selection(choice, actions)

    def _prompt_custom_task(self) -> Optional[str]:
        """Prompt user for custom task when no actions available"""
        try:
            import questionary
            from questionary import Style

            custom_style = Style([
                ('qmark', 'fg:yellow bold'),
                ('question', 'fg:cyan bold'),
            ])

            print()
            task = questionary.text(
                "💬 No actions detected. Enter your next task (or 'q' to quit):",
                style=custom_style,
            ).ask()

            if task and task.lower() not in ("q", "quit", "exit"):
                return task
            return None

        except (ImportError, KeyboardInterrupt):
            task = input("\n💬 No actions detected. Enter your next task (or 'q' to quit): ").strip()
            if task and task.lower() not in ("q", "quit", "exit"):
                return task
            return None

    def _process_stream_output(self, stream):
        """Process streaming JSON output and display succinct progress"""
        text_by_msg = {}  # Track text per message ID
        full_text = ""  # Buffer all text for menu detection
        seen_tools = set()
        current_line_len = 0  # Track chars on current line for clearing

        for line in stream:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
                msg_type = msg.get("type", "")

                if msg_type == "system" and msg.get("subtype") == "init":
                    model = msg.get("model", "unknown")
                    print(f"📡 Model: {model}")
                    print()

                elif msg_type == "assistant":
                    message = msg.get("message", {})
                    msg_id = message.get("id", "")
                    content = message.get("content", [])

                    for block in content:
                        if block.get("type") == "text":
                            text = block.get("text", "")
                            prev_text = text_by_msg.get(msg_id, "")
                            if text and len(text) > len(prev_text):
                                new_text = text[len(prev_text):]
                                print(new_text, end="", flush=True)
                                text_by_msg[msg_id] = text
                                full_text += new_text  # Buffer for menu detection
                                # Track if we're mid-line
                                if "\n" in new_text:
                                    current_line_len = len(new_text.split("\n")[-1])
                                else:
                                    current_line_len += len(new_text)

                        elif block.get("type") == "tool_use":
                            tool_id = block.get("id", "")
                            if tool_id in seen_tools:
                                continue
                            seen_tools.add(tool_id)

                            # If mid-line, just move to new line (don't reprint - text is already visible)
                            if current_line_len > 0:
                                print()  # New line
                                current_line_len = 0

                            tool_name = block.get("name", "unknown")
                            tool_input = block.get("input", {})

                            if tool_name == "Read":
                                path = tool_input.get("file_path", "?").split("/")[-1]
                                print(f"📖 Reading: {path}")
                            elif tool_name == "Edit":
                                path = tool_input.get("file_path", "?").split("/")[-1]
                                print(f"✏️  Editing: {path}")
                            elif tool_name == "Write":
                                path = tool_input.get("file_path", "?").split("/")[-1]
                                print(f"📝 Writing: {path}")
                            elif tool_name == "Bash":
                                cmd = tool_input.get("command", "")[:50]
                                print(f"💻 Running: {cmd}...")
                            elif tool_name == "Glob":
                                print(f"🔍 Finding: {tool_input.get('pattern', '')}")
                            elif tool_name == "Grep":
                                print(f"🔎 Searching: {tool_input.get('pattern', '')}")
                            else:
                                print(f"🔧 {tool_name}")

                elif msg_type == "result":
                    print("\n✅ Done")

            except (json.JSONDecodeError, Exception):
                pass

        print()

        # Check for action menu and show selector
        actions = self._parse_actions_menu(full_text)

        # Always show the steering prompt
        return self._show_steering_prompt(actions)

    def _show_steering_prompt(self, actions: List[dict]) -> Optional[List[dict]]:
        """Show persistent steering prompt - works with or without detected actions"""
        try:
            import questionary
            from questionary import Style

            custom_style = Style([
                ('qmark', 'fg:yellow bold'),
                ('question', 'fg:cyan bold'),
                ('pointer', 'fg:cyan bold'),
                ('highlighted', 'fg:cyan bold'),
                ('selected', 'fg:green'),
            ])

            if actions:
                # Show action menu with text input option
                print()
                print("💡 Select action(s) or type custom task:")
                print("   KB: ↑/↓ navigate | Space select | Enter run | Tab for text input")
                print()

                # Get terminal width
                try:
                    import shutil
                    term_width = shutil.get_terminal_size().columns - 10
                except:
                    term_width = 90

                choices = []
                for a in actions:
                    display = self._format_action_choice(a, term_width)
                    choices.append(questionary.Choice(display, value=a))

                # Add custom input option
                choices.append(questionary.Choice("📝 Type custom task...", value={"_text_input": True}))

                selected = questionary.checkbox(
                    "",
                    choices=choices,
                    style=custom_style,
                    instruction="",
                ).ask()

                if selected is None:
                    return None

                # Check if user wants text input
                if any(s.get("_text_input") for s in selected):
                    custom = questionary.text(
                        ">",
                        style=custom_style,
                    ).ask()
                    if custom and custom.strip():
                        return [{"num": "c", "label": custom.strip(), "description": "Custom task", "_custom": True}]
                    return None

                if len(selected) > 0:
                    if len(selected) == 1:
                        print(f"\n🎯 Selected: {selected[0].get('label', '')}")
                    else:
                        print(f"\n🎯 Selected {len(selected)} actions:")
                        for s in selected:
                            print(f"   • {s.get('label', '')}")
                    return selected
                return None

            else:
                # No actions - just show text prompt
                print()
                custom = questionary.text(
                    "> ",
                    style=custom_style,
                ).ask()

                if custom and custom.strip() and custom.lower() not in ("q", "quit", "exit"):
                    print(f"\n🎯 Task: {custom.strip()}")
                    return [{"num": "c", "label": custom.strip(), "description": "Custom task", "_custom": True}]
                return None

        except (ImportError, KeyboardInterrupt):
            # Fallback without questionary
            if actions:
                return self._show_simple_menu(actions)
            else:
                print()
                custom = input("> ").strip()
                if custom and custom.lower() not in ("q", "quit", "exit"):
                    return [{"num": "c", "label": custom, "description": "Custom task", "_custom": True}]
                return None

    def invoke_claude(self, prompt: str, context: Dict[str, Any] = None) -> Tuple[Any, Optional[List[dict]]]:
        """
        Invoke Claude with a prompt and context.

        Behavior depends on mode:
        - Interactive: Calls Claude Code CLI directly, returns (exit_code, selected_actions)
        - Non-interactive: Saves prompt file, returns (prompt_file, None)
        """
        full_prompt = self.build_prompt(prompt, context)
        full_context = context or self.gather_context()

        # Always save the prompt file for reference
        prompt_file = self.palace_dir / "current_prompt.md"
        self.ensure_palace_dir()

        with open(prompt_file, 'w') as f:
            f.write(full_prompt)

        if is_interactive():
            # Interactive mode: invoke Claude directly
            return self.invoke_claude_cli(full_prompt)
        else:
            # Non-interactive mode: output context for Claude to read
            return prompt_file, None

    def cmd_next(self, args):
        """
        Ask Claude: What should we do next?

        This is the KEY to RHSI - Claude analyzes the project state and suggests
        the next action. Over time, these suggestions become smarter as Palace
        learns from history.

        Supports:
        - Interactive mode: continuous loop with action selection
        - Non-interactive mode: saves session for later resumption
        - --resume SESSION_ID: resume a paused session
        - --select "1,2,3" or "do 1 but skip tests": select actions
        """
        # Check for resume mode
        resume_id = getattr(args, 'resume', None)
        selection = getattr(args, 'select', None)

        session_id = None
        iteration = 0
        pending_actions = []

        if resume_id:
            # Resume existing session
            session = self.load_session(resume_id)
            if not session:
                print(f"❌ Session '{resume_id}' not found")
                print("\nAvailable sessions:")
                for s in self.list_sessions()[:5]:
                    print(f"  • {s['session_id']} (iteration {s['iteration']}, {s['pending_actions']} pending)")
                return

            session_id = resume_id
            iteration = session.get("iteration", 0)
            pending_actions = session.get("pending_actions", [])
            print(f"🔄 Resuming session: {session_id}")
            print(f"   Iteration: {iteration}, Pending actions: {len(pending_actions)}")

            if selection and pending_actions:
                # Apply selection to pending actions
                selected = self.parse_action_selection(selection, pending_actions)
                if selected:
                    pending_actions = selected
                    print(f"   Selected: {len(selected)} action(s)")
                    for a in selected:
                        label = a.get("label", "")
                        mods = a.get("_modifiers", [])
                        if mods:
                            print(f"   • {label} (modifiers: {', '.join(mods)})")
                        else:
                            print(f"   • {label}")
        else:
            # New session
            session_id = self._generate_session_id()

        # Initial prompt for first iteration
        initial_prompt = """Analyze this project and suggest possible next actions.

Consider what exists, what's in progress, and what could come next.
Check SPEC.md and ROADMAP.md if they exist.

Provide as many options as you see fit - there may be many valid paths forward.
Be concrete and actionable. The user will select which action(s) to execute."""

        current_prompt = initial_prompt

        # If resuming with selected actions, build prompt from them
        if pending_actions and (resume_id or selection):
            current_prompt = self._build_action_prompt(pending_actions)

        while True:
            iteration += 1
            context = self.gather_context()

            if not is_interactive():
                # Non-interactive mode: invoke Claude, save session, output for continuation
                result, _ = self.invoke_claude(current_prompt, context)

                # Parse any actions from the response for session state
                # (The actions will be in the prompt file)
                self._show_non_interactive_output(result, context, session_id, iteration)

                self.log_action("next", {
                    "session_id": session_id,
                    "iteration": iteration,
                    "context_file": str(result)
                })
                return

            # Interactive mode: continuous loop
            print(f"\n{'═' * 60}")
            print(f"🔄 RHSI Loop - Session: {session_id} | Iteration {iteration}")
            print(f"{'═' * 60}\n")

            exit_code, selected_actions = self.invoke_claude(current_prompt, context)

            self.log_action("next", {
                "session_id": session_id,
                "iteration": iteration,
                "exit_code": exit_code,
                "selected_actions": [a.get("label") for a in (selected_actions or [])]
            })

            if exit_code != 0:
                print(f"\n⚠️  Claude exited with code {exit_code}")

            if not selected_actions:
                print("\n👋 Exiting RHSI loop.")
                break

            # Build prompt for next iteration based on selected actions
            current_prompt = self._build_action_prompt(selected_actions)
            print()

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

    def _show_non_interactive_output(self, result, context: Dict[str, Any], session_id: str = None, iteration: int = 1):
        """Show output for non-interactive mode and save session state"""
        print("🏛️  Palace - Invoking Claude for next step analysis...")
        print()
        if session_id:
            print(f"📋 Session ID: {session_id}")
        print(f"📝 Context prepared at: {result}")
        print()
        print("─" * 60)
        print()

        print("PROJECT STATE:")
        if context["files"]:
            print("\nExisting files:")
            for filename, info in context["files"].items():
                print(f"  ✓ {filename}")

        if context.get("git_status"):
            print("\nGit status:")
            for line in context["git_status"].strip().split('\n')[:5]:
                print(f"  {line}")

        if context.get("recent_history"):
            print(f"\nRecent actions: {len(context['recent_history'])} logged")

        print()
        print("─" * 60)
        print()
        print("🤖 CLAUDE: Please analyze the project and suggest next actions.")
        print()
        print("After Claude responds with ACTIONS:, you can resume with:")
        print(f"   python3 palace.py next --resume {session_id} --select \"1,2,3\"")
        print(f"   python3 palace.py next --resume {session_id} --select \"do 1 but skip tests\"")
        print()

        # Save initial session state (pending_actions will be populated when Claude responds)
        if session_id:
            self.save_session(session_id, {
                "iteration": iteration,
                "pending_actions": [],  # Will be populated from Claude's response
                "context": context,
                "prompt_file": str(result)
            })

    def cmd_new(self, args):
        """Ask Claude to create a new project"""
        project_name = args.name if hasattr(args, 'name') and args.name else None

        if not project_name:
            project_name = input("Project name: ").strip()
            if not project_name:
                print("❌ Project name required")
                return

        prompt = f"""Create a new project called '{project_name}'.

Steps:
1. Create the project directory
2. Initialize Palace (.palace/config.json)
3. Create initial files (README.md, etc.)
4. Set up version control if appropriate
5. Create a SPEC.md outlining what this project should do

Ask the user questions if you need more information about what this project should do.
Then execute the setup using your available tools."""

        prompt_file = self.invoke_claude(prompt)

        print(f"🏛️  Palace - Creating new project: {project_name}")
        print(f"📝 Prompt ready at: {prompt_file}")
        print()
        print("🤖 CLAUDE: Please read the prompt above and create the project.")

        self.log_action("new", {"project_name": project_name})

    def cmd_scaffold(self, args):
        """Ask Claude to scaffold the project"""

        prompt = """Scaffold this project with best practices.

Analyze:
1. What type of project is this? (Python, Node.js, Rust, etc.)
2. What files/structure are already present?
3. What's missing for a well-structured project?

Then create:
- Appropriate directory structure
- Configuration files (if not present)
- Test setup
- Documentation templates
- Any language-specific best practices

Use your judgment and knowledge of best practices for this project type.
Ask questions if needed to clarify the project type or goals."""

        context = self.gather_context()
        prompt_file = self.invoke_claude(prompt, context)

        print("🏛️  Palace - Scaffolding project...")
        print(f"📝 Prompt ready at: {prompt_file}")
        print()
        print("🤖 CLAUDE: Please analyze the project and create appropriate scaffolding.")

        self.log_action("scaffold")

    def cmd_test(self, args):
        """Ask Claude to run tests"""

        prompt = """Run tests for this project.

1. Detect what testing framework is being used
2. Run the tests
3. Analyze the results
4. If there are failures, suggest fixes

If no tests exist yet, suggest setting up a testing framework appropriate
for this project type."""

        context = self.gather_context()
        prompt_file = self.invoke_claude(prompt, context)

        print("🏛️  Palace - Running tests...")
        print(f"📝 Prompt ready at: {prompt_file}")
        print()
        print("🤖 CLAUDE: Please run the project tests and analyze results.")

        self.log_action("test")

    def cmd_install(self, args):
        """Install Palace commands, dependencies, and output style into Claude Code"""

        print("🏛️  Palace - Installing to Claude Code")
        print()

        # Install Python dependencies using uv (preferred) or pip
        print("📦 Installing dependencies...")
        deps = ["questionary", "rich", "anthropic", "mcp"]

        # Check for palace venv or use uv
        palace_venv = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
        uv_path = Path.home() / ".local" / "bin" / "uv"

        try:
            if uv_path.exists():
                # Use uv pip install
                cmd = [str(uv_path), "pip", "install", "--quiet"] + deps
                result = subprocess.run(cmd, capture_output=True, text=True)
            elif palace_venv.exists():
                # Use palace venv pip
                cmd = [str(palace_venv), "-m", "pip", "install", "--quiet"] + deps
                result = subprocess.run(cmd, capture_output=True, text=True)
            else:
                # Fallback to system pip (may fail on managed environments)
                cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + deps
                result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Installed: {', '.join(deps)}")
            else:
                print(f"⚠️  Install failed. Try: uv pip install {' '.join(deps)}")
        except Exception as e:
            print(f"⚠️  Could not install dependencies: {e}")
        print()

        # Find Claude Code config directory
        claude_dir = Path.home() / ".claude"
        if not claude_dir.exists():
            claude_dir = Path.home() / ".config" / "claude"

        commands_dir = claude_dir / "commands"
        styles_dir = claude_dir / "output-styles"

        commands_dir.mkdir(parents=True, exist_ok=True)
        styles_dir.mkdir(parents=True, exist_ok=True)

        # Get absolute path to palace.py and output style
        palace_path = Path(__file__).resolve()
        palace_dir = palace_path.parent

        # Install output style
        style_src = palace_dir / ".claude" / "output-styles" / "palace-menu.md"
        style_dst = styles_dir / "palace-menu.md"
        if style_src.exists():
            shutil.copy(style_src, style_dst)
            print(f"✅ Installed output style: palace-menu")

        commands = {
            "pal-next.md": f"""# Palace Next
Invoke Palace to suggest what to do next in the project.

Palace will analyze the project state and ask Claude to suggest the next logical step.
This is the core of Recursive Hierarchical Self Improvement (RHSI).

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python3 {palace_path} next
```

After running this, read the generated prompt file and provide your analysis.
""",
            "pal-new.md": f"""# Palace New
Create a new project with Palace.

```bash
python3 {palace_path} new "$@"
```

After running this, read the generated prompt and create the project structure.
""",
            "pal-scaffold.md": f"""# Palace Scaffold
Scaffold the current project with best practices.

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python3 {palace_path} scaffold
```

After running this, read the generated prompt and scaffold the project.
""",
            "pal-test.md": f"""# Palace Test
Run tests for the current project.

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python3 {palace_path} test
```

After running this, read the generated prompt and run the tests.
""",
        }

        installed = []
        for filename, content in commands.items():
            filepath = commands_dir / filename
            with open(filepath, 'w') as f:
                f.write(content)
            installed.append(filename.replace('.md', ''))

        print("✅ Installed Palace commands:")
        for cmd in installed:
            print(f"   • /{cmd}")

        print()
        print("🎉 Palace is now integrated with Claude Code!")
        print()
        print("Output style 'palace-menu' enables action menus.")
        print("Try: /pal-next")

    def cmd_init(self, args):
        """Initialize Palace in current directory"""
        print("🏛️  Palace - Initializing in current directory")
        print()

        if self.palace_dir.exists():
            print("⚠️  Palace already initialized here")
            return

        self.ensure_palace_dir()

        config = {
            "name": self.project_root.name,
            "version": "0.1.0",
            "palace_version": VERSION,
            "initialized": True
        }

        self.save_config(config)

        print("✅ Palace initialized!")
        print(f"📁 Created {self.palace_dir}")
        print()
        print("Next steps:")
        print("  1. Run 'python3 palace.py next' to see what to do next")
        print("  2. Or run 'python3 palace.py install' to add Palace to Claude Code")

    def cmd_sessions(self, args):
        """List saved sessions"""
        sessions = self.list_sessions()

        if not sessions:
            print("📋 No saved sessions found")
            print()
            print("Sessions are created when running 'palace next' in non-interactive mode.")
            return

        print("📋 Saved Sessions:")
        print()
        for s in sessions:
            session_id = s.get("session_id", "?")
            iteration = s.get("iteration", 0)
            pending = s.get("pending_actions", 0)
            updated = s.get("updated_at")

            # Format timestamp
            if updated:
                from datetime import datetime
                dt = datetime.fromtimestamp(updated)
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            else:
                time_str = "unknown"

            print(f"  • {session_id}")
            print(f"    Iteration: {iteration} | Pending: {pending} actions | Updated: {time_str}")
            print()

        print("Resume with:")
        print('  python3 palace.py next --resume SESSION_ID --select "1,2,3"')
        print()

    def cmd_cleanup(self, args):
        """Cleanup old sessions and history"""
        import datetime

        # Determine cleanup mode
        all_sessions = getattr(args, 'all', False)
        days = getattr(args, 'days', 30)

        sessions_dir = self._get_sessions_dir()

        if all_sessions:
            # Delete all sessions
            count = 0
            for session_file in sessions_dir.glob("*.json"):
                session_file.unlink()
                count += 1
            print(f"🗑️  Deleted {count} session(s)")

        else:
            # Delete old sessions
            cutoff = time.time() - (days * 24 * 60 * 60)
            count = 0

            for session_file in sessions_dir.glob("*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session = json.load(f)
                    updated_at = session.get("updated_at", 0)

                    if updated_at < cutoff:
                        session_file.unlink()
                        count += 1
                except:
                    pass

            print(f"🗑️  Deleted {count} session(s) older than {days} days")

        # Optionally trim history
        if getattr(args, 'history', False):
            history_file = self.palace_dir / "history.jsonl"
            if history_file.exists():
                # Keep only last N entries
                keep_lines = getattr(args, 'keep_history', 1000)

                lines = history_file.read_text().strip().split('\n')
                if len(lines) > keep_lines:
                    trimmed = lines[-keep_lines:]
                    history_file.write_text('\n'.join(trimmed) + '\n')
                    print(f"📝 Trimmed history to {keep_lines} entries (removed {len(lines) - keep_lines})")

        print("✅ Cleanup complete")

    def cmd_permissions(self, args):
        """
        Handle permission prompts from Claude Code

        This is called by Claude when it needs to request permissions.
        Palace can implement custom permission logic here.
        """
        # For now, pass through to default behavior
        # In the future, Palace could track permissions, learn from them, etc.

        # Read permission request from stdin (stream-json format)
        if not sys.stdin.isatty():
            try:
                request = json.load(sys.stdin)
                # Log the permission request
                self.log_action("permission_request", {"request": request})

                # For now, output approval (in the future, add logic here)
                response = {"approved": True}
                json.dump(response, sys.stdout)
                sys.stdout.flush()
            except Exception as e:
                # On error, deny permission
                response = {"approved": False, "reason": str(e)}
                json.dump(response, sys.stdout)
                sys.stdout.flush()
        else:
            print("❌ Permissions command should be called by Claude Code, not directly")
            sys.exit(1)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Palace - Claude orchestration for RHSI",
        epilog="Palace doesn't do the work - Claude does. Palace just sets up context."
    )

    parser.add_argument('--version', action='version', version=f'Palace {VERSION}')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Commands that invoke Claude
    parser_next = subparsers.add_parser('next', help='Ask Claude what to do next (RHSI core)')
    parser_next.add_argument('--resume', '-r', metavar='SESSION_ID',
                             help='Resume a paused session')
    parser_next.add_argument('--select', '-s', metavar='SELECTION',
                             help='Select actions: "1,2,3" or "do 1 but skip tests"')

    parser_new = subparsers.add_parser('new', help='Ask Claude to create a new project')
    parser_new.add_argument('name', nargs='?', help='Project name')

    subparsers.add_parser('scaffold', help='Ask Claude to scaffold the project')
    subparsers.add_parser('test', help='Ask Claude to run tests')

    # Utility commands
    subparsers.add_parser('install', help='Install Palace commands to Claude Code')
    subparsers.add_parser('init', help='Initialize Palace in current directory')
    subparsers.add_parser('sessions', help='List saved sessions')

    parser_cleanup = subparsers.add_parser('cleanup', help='Cleanup old sessions and history')
    parser_cleanup.add_argument('--all', action='store_true', help='Delete all sessions')
    parser_cleanup.add_argument('--days', type=int, default=30, help='Delete sessions older than N days')
    parser_cleanup.add_argument('--history', action='store_true', help='Trim history log')
    parser_cleanup.add_argument('--keep-history', type=int, default=1000, help='Keep last N history entries')

    subparsers.add_parser('permissions', help='Handle Claude Code permission requests (internal)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    palace = Palace()

    commands = {
        'next': palace.cmd_next,
        'new': palace.cmd_new,
        'scaffold': palace.cmd_scaffold,
        'test': palace.cmd_test,
        'install': palace.cmd_install,
        'init': palace.cmd_init,
        'sessions': palace.cmd_sessions,
        'cleanup': palace.cmd_cleanup,
        'permissions': palace.cmd_permissions,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()

# ============================================================================
# MCP Server Integration
# ============================================================================
# Palace is an MCP server providing the handle_permission tool.
#
# Install globally with:
#   claude mcp add palace --scope user \
#     /path/to/palace/.venv/bin/python \
#     /path/to/palace/palace.py

_DEFAULT_COMMAND_SAFETY_SKILL = """# Command Safety Assessment

You are a safety check for autonomous coding loops. The user has opted into this mode.

## Philosophy: Productive Caution

Most dev operations are fine. Use context to decide.

## THINK TWICE about:
- Mass deletions (rm -rf, find -delete, DROP TABLE)
- Force operations (git push --force, overwriting without backup)
- System-level changes (modifying /etc, ~/.bashrc, system packages)
- Credential access (reading .env, secrets, API keys)
- Network operations to unfamiliar hosts
- Database migrations that drop or truncate
- Changing permissions broadly (chmod -R 777)
- Operations outside the project directory

## Context matters:
- `rm -rf node_modules` = routine cleanup, approve
- `rm -rf /` = catastrophic, deny
- `git push --force feature-branch` = probably intentional, approve
- `git push --force main` = risky, think twice
- `pip install pytest` = dev dependency, approve
- `apt install nginx` = system change, think twice
- Writing to project files = normal dev work, approve
- Writing to ~/.ssh/ = sensitive, think twice

## Decision
If the operation makes sense in a development context, approve it.
If it seems out of place or potentially destructive, deny with explanation.

## Response Format
Respond with ONLY a JSON object:
{"approved": true, "reason": "brief explanation"}
or
{"approved": false, "reason": "brief explanation"}
"""

try:
    from mcp.server.fastmcp import FastMCP

    # Create MCP server instance
    mcp = FastMCP("Palace")

    def _assess_permission_safety(tool_name: str, tool_input: dict) -> dict:
        """
        Use Haiku to assess whether a permission request is safe.

        Loads the command-safety skill from .palace/skills/command-safety.md
        which users can customize to train the safety assessment.
        """
        try:
            import anthropic

            # Load the command-safety skill
            palace = Palace()
            skill_path = palace.palace_dir / "skills" / "command-safety.md"

            if not skill_path.exists():
                # Create default skill file
                skill_path.parent.mkdir(parents=True, exist_ok=True)
                skill_path.write_text(_DEFAULT_COMMAND_SAFETY_SKILL)

            skill_content = skill_path.read_text()

            # Build the prompt for Haiku
            prompt = f"""Tool: {tool_name}
Input: {json.dumps(tool_input, indent=2)}

Based on the safety guidelines, should this operation be approved?"""

            # Call Haiku for fast safety assessment
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-haiku-4-20250514",
                max_tokens=256,
                system=skill_content,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse the response
            response_text = response.content[0].text.strip()

            # Try to extract JSON from the response
            if "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                result = json.loads(response_text[json_start:json_end])
                return {
                    "approved": result.get("approved", True),
                    "reason": result.get("reason", "Assessed by Haiku")
                }

            # Fallback: approve if we can't parse
            return {"approved": True, "reason": "Could not parse safety response"}

        except Exception as e:
            # On any error, approve but log the issue
            return {"approved": True, "reason": f"Safety check error: {str(e)[:100]}"}

    @mcp.tool()
    def handle_permission(
        tool_name: str = "",
        input: dict = None,
        tool_use_id: str = "",
        **kwargs
    ) -> dict:
        """
        Handle permission requests from Claude during RHSI loops.

        Claude Code sends permission requests with tool_name, input, and tool_use_id.
        Uses Haiku to assess safety based on the trainable command-safety skill.

        Returns:
            dict with 'approved' (bool) and optional 'reason' (str)
        """
        # Initialize Palace instance to access logging
        palace = Palace()

        # Log the permission request
        request_data = {
            "tool_name": tool_name,
            "input": input or {},
            "tool_use_id": tool_use_id,
            **kwargs
        }
        palace.log_action("permission_request", {"request": request_data})

        # Assess safety using Haiku and the command-safety skill
        result = _assess_permission_safety(tool_name, input or {})

        # Log the decision
        palace.log_action("permission_decision", {
            "tool_name": tool_name,
            "approved": result.get("approved"),
            "reason": result.get("reason")
        })

        return result

    MCP_AVAILABLE = True
except ImportError:
    # MCP not installed - server functionality unavailable
    # This is fine for CLI-only usage
    MCP_AVAILABLE = False

if __name__ == "__main__":
    # If no arguments, run as MCP server (for Claude Code integration)
    # If arguments provided, run as CLI
    if len(sys.argv) == 1 and MCP_AVAILABLE:
        # No arguments - start MCP server
        mcp.run()
    else:
        main()
