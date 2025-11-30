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

# Load credentials from ~/.palace/credentials.env
def _load_credentials():
    creds_file = Path.home() / ".palace" / "credentials.env"
    if creds_file.exists():
        for line in creds_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

_load_credentials()

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

    def __init__(self, strict_mode: bool = True, force_claude: bool = False, force_glm: bool = False) -> None:
        self.project_root = Path.cwd()
        self.palace_dir = self.project_root / ".palace"
        self.config_file = self.palace_dir / "config.json"
        self.strict_mode = strict_mode
        self.modified_files = set()  # Track files modified during execution
        self.force_claude = force_claude  # Use Claude even in turbo mode
        self.force_glm = force_glm  # Use GLM even in normal mode

    def ensure_palace_dir(self) -> None:
        """Ensure .palace directory exists"""
        self.palace_dir.mkdir(exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        """Load Palace configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}

    def save_config(self, config: Dict[str, Any]) -> None:
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

    def log_action(self, action: str, details: Optional[Dict[str, Any]] = None) -> None:
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

    def save_session(self, session_id: str, state: Dict[str, Any]) -> Path:
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

        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

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

    def export_session(self, session_id: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Export a session to a portable JSON file.

        Includes full session state, history, and metadata for sharing or backup.

        Args:
            session_id: The session ID to export
            output_path: Optional output file path (defaults to current dir)

        Returns path to exported file or None if session not found.
        """
        session = self.load_session(session_id)
        if not session:
            return None

        # Build export bundle
        export_data = {
            "version": "1.0",
            "exported_at": time.time(),
            "palace_version": VERSION,
            "session": session
        }

        # Add relevant history entries for this session
        history_entries = []
        history_file = self.palace_dir / "history.jsonl"
        if history_file.exists():
            for line in history_file.read_text().strip().split('\n'):
                try:
                    entry = json.loads(line)
                    # Include entries from this session
                    if entry.get("details", {}).get("session_id") == session_id:
                        history_entries.append(entry)
                except:
                    pass

        export_data["history"] = history_entries

        # Determine output path
        if not output_path:
            output_path = f"{session_id}_export.json"

        # Write export file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        return output_path

    def import_session(self, import_path: str, new_session_id: Optional[str] = None) -> Optional[str]:
        """
        Import a session from an exported JSON file.

        Args:
            import_path: Path to the exported session file
            new_session_id: Optional new session ID (generates one if not provided)

        Returns the imported session ID or None if import failed.
        """
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)

            # Validate export format
            if "session" not in import_data:
                return None

            session = import_data["session"]

            # Generate new session ID if not provided
            if not new_session_id:
                new_session_id = self._generate_session_id()

            # Update session ID
            session["session_id"] = new_session_id
            session["imported_at"] = time.time()
            session["imported_from"] = import_path

            # Save imported session
            self.save_session(new_session_id, session)

            # Import history entries if present
            if "history" in import_data and import_data["history"]:
                self.ensure_palace_dir()
                history_file = self.palace_dir / "history.jsonl"
                with open(history_file, 'a') as f:
                    for entry in import_data["history"]:
                        # Update session_id in history entries
                        if "details" in entry and isinstance(entry["details"], dict):
                            entry["details"]["session_id"] = new_session_id
                        f.write(json.dumps(entry) + '\n')

            return new_session_id

        except Exception as e:
            return None

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

    def compose_masks(self, mask_names: List[str], strategy: str = "merge") -> Optional[str]:
        """
        Compose multiple masks together.

        Strategies:
        - "merge": Concatenate masks in order with separators
        - "layer": Apply masks hierarchically (later masks override earlier)
        - "blend": Interleave sections from each mask

        Returns composed mask content or None if any mask not found.
        """
        if not mask_names:
            return None

        # Load all masks
        mask_contents = []
        for name in mask_names:
            content = self.load_mask(name)
            if not content:
                return None  # Fail if any mask is missing
            mask_contents.append((name, content))

        if strategy == "merge":
            # Simple concatenation with clear separators
            parts = []
            for name, content in mask_contents:
                parts.append(f"# Mask: {name}")
                parts.append(content)
                parts.append("")  # Blank line separator
            return "\n".join(parts)

        elif strategy == "layer":
            # Later masks override earlier ones
            # Use frontmatter priority field if available
            layered_parts = []
            for name, content in mask_contents:
                metadata = self.get_mask_metadata(name)
                priority = int(metadata.get("priority", 0)) if metadata else 0
                layered_parts.append((priority, name, content))

            # Sort by priority (lower = higher precedence)
            layered_parts.sort(key=lambda x: x[0])

            # Build layered content
            result = []
            for _, name, content in layered_parts:
                result.append(f"# Layer: {name}")
                result.append(content)
                result.append("")
            return "\n".join(result)

        elif strategy == "blend":
            # Interleave sections from each mask
            # Split each mask into sections (by headers)
            sections = []
            for name, content in mask_contents:
                mask_sections = self._split_mask_into_sections(content)
                sections.extend([(name, s) for s in mask_sections])

            # Interleave sections
            blended = []
            for name, section in sections:
                blended.append(f"<!-- From {name} -->")
                blended.append(section)
                blended.append("")
            return "\n".join(blended)

        return None

    def _split_mask_into_sections(self, content: str) -> List[str]:
        """
        Split mask content into sections by markdown headers.

        Returns list of section strings.
        """
        sections = []
        current_section = []

        for line in content.split("\n"):
            if line.startswith("#") and current_section:
                # Start new section
                sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        # Add final section
        if current_section:
            sections.append("\n".join(current_section))

        return sections

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

    def build_prompt_with_masks(self, task_prompt: str, mask_names: List[str],
                                strategy: str = "merge",
                                context: Dict[str, Any] = None) -> Optional[str]:
        """
        Build prompt with multiple masks composed together.

        Args:
            task_prompt: The task description
            mask_names: List of mask names to compose
            strategy: Composition strategy ("merge", "layer", "blend")
            context: Optional context dict

        Returns composed prompt or None if any mask not found.
        """
        if not mask_names:
            return self.build_prompt(task_prompt, context)

        # Compose masks
        composed_content = self.compose_masks(mask_names, strategy)
        if not composed_content:
            return None

        # Build base prompt
        base_prompt = self.build_prompt(task_prompt, context)

        # Prepend composed mask content
        return f"{composed_content}\n\n{base_prompt}"

    # ========================================================================
    # User Steering (ESC-ESC Interrupt)
    # ========================================================================

    def _setup_escape_handler(self):
        """
        Initialize escape sequence detection state.

        Call this before starting stream processing to enable ESC-ESC detection.
        """
        self._last_esc_time = None
        self._escape_timeout = 0.5  # 500ms window for double-tap

    def _check_escape_sequence(self, char: str) -> Optional[str]:
        """
        Check if character is part of ESC-ESC sequence.

        Returns:
        - "first_esc": First ESC detected, waiting for second
        - "interrupt": ESC-ESC sequence completed, trigger interrupt
        - None: Not an ESC key
        """
        if char != chr(27):  # ESC character
            return None

        current_time = time.time()

        if self._last_esc_time is None:
            # First ESC
            self._last_esc_time = current_time
            return "first_esc"

        # Check if within timeout window
        elapsed = current_time - self._last_esc_time
        if elapsed <= self._escape_timeout:
            # Double-tap detected!
            self._last_esc_time = None
            return "interrupt"
        else:
            # Too slow, reset and treat as new first ESC
            self._last_esc_time = current_time
            return "first_esc"

    def _check_for_escape(self) -> bool:
        """
        Non-blocking check for ESC keypress.

        Returns True if ESC-ESC was detected.
        Uses select() to check stdin without blocking.
        """
        import select
        import sys
        import tty
        import termios
        import fcntl
        import os

        # Only works in TTY
        if not sys.stdin.isatty():
            return False

        fd = sys.stdin.fileno()

        try:
            # Save current terminal settings and flags
            old_settings = termios.tcgetattr(fd)
            old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)

            try:
                # Set non-blocking
                fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)

                # Set terminal to cbreak mode (char by char, no echo)
                new_settings = termios.tcgetattr(fd)
                new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO)
                new_settings[6][termios.VMIN] = 0
                new_settings[6][termios.VTIME] = 0
                termios.tcsetattr(fd, termios.TCSANOW, new_settings)

                # Try to read
                try:
                    char = os.read(fd, 1)
                    if char == b'\x1b':  # ESC
                        result = self._check_escape_sequence(chr(27))
                        return result == "interrupt"
                except (BlockingIOError, OSError):
                    pass

            finally:
                # Restore terminal settings
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)

        except Exception:
            pass

        return False

    def _handle_user_interrupt(self) -> Optional[Dict[str, Any]]:
        """
        Handle user interrupt (ESC-ESC).

        Shows steering prompt and returns user input.

        Returns dict with:
        - action: "steer", "resume", or "abort"
        - steering: User's steering text (if action is "steer")

        Returns None if user cancels (empty input).
        """
        print("\n")
        print("⏸️  " + "─" * 50)
        print("   PALACE PAUSED - Enter steering command")
        print("   (Press Enter to resume, /abort to stop)")
        print("   " + "─" * 50)
        print()

        try:
            steering = input("🎯 Steer: ").strip()

            if not steering:
                # Empty input = resume normally
                print("▶️  Resuming...")
                return {"action": "resume"}

            if steering.lower() == "/abort":
                print("🛑 Aborting session...")
                return {"action": "abort"}

            # Log the steering
            self.log_steering(steering)

            print(f"✅ Steering applied: {steering[:50]}...")
            return {"action": "steer", "steering": steering}

        except (EOFError, KeyboardInterrupt):
            print("\n▶️  Resuming...")
            return {"action": "resume"}

    def log_steering(self, steering: str):
        """Log user steering to history for learning"""
        self.log_action("user_steering", {"steering": steering})

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

    # ========================================================================
    # Multi-Provider System
    # ========================================================================

    def get_provider_config(self) -> Dict[str, Any]:
        """
        Load provider configuration.

        Checks (in order, later overrides earlier):
        1. Built-in defaults
        2. Global ~/.palace/providers.json
        3. Project .palace/providers.json

        Returns merged config.
        """
        defaults = {
            "default_provider": "anthropic",
            "providers": {
                "anthropic": {
                    "base_url": "https://api.anthropic.com",
                    "format": "anthropic",
                    "api_key_env": "ANTHROPIC_API_KEY"
                },
                "z.ai": {
                    "base_url": "https://api.z.ai/api/anthropic",
                    "format": "anthropic",
                    "api_key_env": "ZAI_API_KEY"
                },
                "openrouter": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "format": "openai",
                    "api_key_env": "OPENROUTER_API_KEY"
                }
            },
            "model_aliases": {
                "opus": {"provider": "anthropic", "model": "claude-opus-4-5"},
                "sonnet": {"provider": "anthropic", "model": "claude-sonnet-4-5"},
                "haiku": {"provider": "anthropic", "model": "claude-haiku-4-5"},
                "glm": {"provider": "z.ai", "model": "glm-4.6"},
                "glm-fast": {"provider": "z.ai", "model": "glm-4-flash"}
            }
        }

        # Load from ~/.palace/providers.json
        config_path = Path.home() / ".palace" / "providers.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                # Merge user config with defaults
                for key in ["providers", "model_aliases"]:
                    if key in user_config:
                        defaults[key].update(user_config[key])
                if "default_provider" in user_config:
                    defaults["default_provider"] = user_config["default_provider"]
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Failed to load {config_path}: {e}")

        return defaults

    def resolve_model(self, model_or_alias: str) -> Tuple[str, str]:
        """
        Resolve model alias to (provider, model) tuple.

        If alias not found, assumes Anthropic provider with model name as-is.
        """
        config = self.get_provider_config()
        aliases = config.get("model_aliases", {})

        if model_or_alias in aliases:
            alias_config = aliases[model_or_alias]
            return alias_config["provider"], alias_config["model"]

        # Not an alias, assume Anthropic
        return "anthropic", model_or_alias

    def build_api_request(self, provider: str, model: str, messages: List[dict],
                          system: str = None, **kwargs) -> Dict[str, Any]:
        """Build API request in provider's native format"""
        config = self.get_provider_config()
        provider_config = config["providers"].get(provider, {})
        fmt = provider_config.get("format", "anthropic")

        if fmt == "anthropic":
            request = {"model": model, "messages": messages}
            if system:
                request["system"] = system
            request.update(kwargs)
            return request
        else:
            # OpenAI format
            return self._build_openai_request(model, messages, system, **kwargs)

    def _build_openai_request(self, model: str, messages: List[dict],
                               system: str = None, **kwargs) -> Dict[str, Any]:
        """Build request in OpenAI/OpenRouter format"""
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages)

        request = {"model": model, "messages": openai_messages}
        # Translate max_tokens to max_completion_tokens if needed
        if "max_tokens" in kwargs:
            request["max_completion_tokens"] = kwargs.pop("max_tokens")
        request.update(kwargs)
        return request

    def translate_to_openai_format(self, messages: List[dict], system: str = None) -> List[dict]:
        """Translate Anthropic messages to OpenAI format"""
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})

        for msg in messages:
            openai_messages.append({
                "role": msg["role"],
                "content": msg.get("content", "")
            })

        return openai_messages

    def translate_tool_use_to_openai(self, anthropic_content: List[dict]) -> Dict[str, Any]:
        """Translate Anthropic tool_use blocks to OpenAI function_call format"""
        text_parts = []
        tool_calls = []

        for block in anthropic_content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}))
                    }
                })

        result = {"role": "assistant", "content": " ".join(text_parts) if text_parts else None}
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    def translate_openai_to_anthropic(self, openai_response: Dict[str, Any]) -> Dict[str, Any]:
        """Translate OpenAI response to Anthropic format"""
        choice = openai_response.get("choices", [{}])[0]
        message = choice.get("message", {})

        content = []

        # Text content
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})

        # Tool calls
        for tool_call in message.get("tool_calls", []):
            func = tool_call.get("function", {})
            try:
                arguments = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}

            content.append({
                "type": "tool_use",
                "id": tool_call.get("id", ""),
                "name": func.get("name", ""),
                "input": arguments
            })

        return {"role": "assistant", "content": content}

    def invoke_provider(self, provider: str, model: str, messages: List[dict],
                        system: str = None, **kwargs) -> Dict[str, Any]:
        """
        Invoke any configured provider.

        Handles format translation automatically based on provider config.
        """
        import requests

        config = self.get_provider_config()
        provider_config = config["providers"].get(provider)

        if not provider_config:
            raise ValueError(f"Unknown provider: {provider}")

        fmt = provider_config.get("format", "anthropic")
        base_url = provider_config.get("base_url", "")
        api_key_env = provider_config.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "")

        if fmt == "anthropic":
            # Use Anthropic SDK
            import anthropic
            client = anthropic.Anthropic(
                api_key=api_key if api_key else None,
                base_url=base_url if base_url and "anthropic.com" not in base_url else None
            )
            response = client.messages.create(
                model=model,
                messages=messages,
                system=system or "",
                max_tokens=kwargs.get("max_tokens", 4096)
            )
            return {"content": [{"type": "text", "text": c.text} for c in response.content if hasattr(c, 'text')]}

        else:
            # OpenAI-compatible API (OpenRouter, etc.)
            openai_messages = self.translate_to_openai_format(messages, system)
            request_body = {
                "model": model,
                "messages": openai_messages,
                "max_tokens": kwargs.get("max_tokens", 4096)
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # OpenRouter-specific headers
            if "openrouter" in provider.lower():
                headers["HTTP-Referer"] = "https://github.com/anthropics/palace"
                headers["X-Title"] = "Palace RHSI"

            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=request_body
            )
            response.raise_for_status()
            return self.translate_openai_to_anthropic(response.json())

    # ========================================================================
    # Benchmarking System
    # ========================================================================

    def get_benchmark_tasks(self) -> List[Dict[str, Any]]:
        """Get standard benchmark tasks for model evaluation"""
        return [
            {
                "name": "code_generation",
                "prompt": "Write a Python function that calculates the nth Fibonacci number using memoization.",
                "expected_capabilities": ["code_quality", "correctness", "efficiency"]
            },
            {
                "name": "code_analysis",
                "prompt": "Analyze this code and identify potential bugs:\n\ndef divide(a, b):\n    return a / b",
                "expected_capabilities": ["bug_detection", "edge_cases"]
            },
            {
                "name": "refactoring",
                "prompt": "Refactor this code to be more readable:\n\ndef f(x):return[i for i in range(x)if i%2==0]",
                "expected_capabilities": ["readability", "best_practices"]
            },
            {
                "name": "natural_language",
                "prompt": "Explain how a binary search tree works to a beginner programmer.",
                "expected_capabilities": ["clarity", "accuracy", "pedagogy"]
            }
        ]

    def run_benchmark(self, model_alias: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single benchmark task on a model"""
        import time

        provider, model = self.resolve_model(model_alias)

        start_time = time.time()
        try:
            response = self.invoke_provider(
                provider=provider,
                model=model,
                messages=[{"role": "user", "content": task["prompt"]}],
                max_tokens=2048
            )
            latency_ms = (time.time() - start_time) * 1000

            response_text = ""
            for block in response.get("content", []):
                if block.get("type") == "text":
                    response_text += block.get("text", "")

            return {
                "model": model_alias,
                "provider": provider,
                "actual_model": model,
                "task": task["name"],
                "latency_ms": latency_ms,
                "response": response_text,
                "success": True
            }

        except Exception as e:
            return {
                "model": model_alias,
                "task": task["name"],
                "latency_ms": (time.time() - start_time) * 1000,
                "response": None,
                "success": False,
                "error": str(e)
            }

    def judge_benchmark_result(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Judge a benchmark result using Opus as the judge.

        Returns score (0-10) and reasoning.
        """
        judge_prompt = f"""You are evaluating an AI model's response to a coding task.

TASK: {task['prompt']}

EXPECTED CAPABILITIES: {', '.join(task['expected_capabilities'])}

MODEL RESPONSE:
{response}

Score this response from 0-10 based on:
- Correctness and accuracy
- Code quality (if applicable)
- Clarity and helpfulness
- Addressing all aspects of the task

Respond with JSON only:
{{"score": <0-10>, "reasoning": "<brief explanation>"}}"""

        try:
            result = self.invoke_provider(
                provider="anthropic",
                model="claude-opus-4-5",
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=512
            )

            response_text = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    response_text += block.get("text", "")

            # Parse JSON from response
            if "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                return json.loads(response_text[json_start:json_end])

        except Exception as e:
            pass

        return {"score": 0, "reasoning": f"Judge failed: {str(e)}"}

    def run_full_benchmark(self, model_aliases: List[str] = None) -> Dict[str, Any]:
        """
        Run full benchmark suite across multiple models.

        Returns comparative results.
        """
        if model_aliases is None:
            model_aliases = ["opus", "sonnet", "haiku"]

        tasks = self.get_benchmark_tasks()
        results = {"models": {}, "summary": {}}

        for alias in model_aliases:
            results["models"][alias] = {"tasks": [], "total_score": 0, "avg_latency_ms": 0}

            total_latency = 0
            for task in tasks:
                print(f"  Benchmarking {alias} on {task['name']}...")
                result = self.run_benchmark(alias, task)

                if result["success"]:
                    score = self.judge_benchmark_result(task, result["response"])
                    result["score"] = score.get("score", 0)
                    result["reasoning"] = score.get("reasoning", "")
                    results["models"][alias]["total_score"] += result["score"]
                else:
                    result["score"] = 0

                total_latency += result["latency_ms"]
                results["models"][alias]["tasks"].append(result)

            results["models"][alias]["avg_latency_ms"] = total_latency / len(tasks)

        # Summary ranking
        rankings = sorted(
            [(alias, data["total_score"], data["avg_latency_ms"])
             for alias, data in results["models"].items()],
            key=lambda x: (-x[1], x[2])  # Higher score, lower latency
        )
        results["summary"]["rankings"] = [
            {"model": r[0], "total_score": r[1], "avg_latency_ms": r[2]}
            for r in rankings
        ]

        return results

    # ========================================================================
    # Turbo Mode (Swarm Execution)
    # ========================================================================

    def rank_tasks_by_model(self, tasks: List[dict]) -> Dict[str, Dict[str, Any]]:
        """
        Use Opus/GLM to rank tasks and assign optimal models.

        Returns dict mapping task_num -> {model, reasoning, task}
        """
        task_descriptions = "\n".join([
            f"{t.get('num')}. {t.get('label')}: {t.get('description', '')}"
            for t in tasks
        ])

        ranking_prompt = f"""Analyze these tasks and assign the optimal model for each:

TASKS:
{task_descriptions}

AVAILABLE MODELS:
- opus: Complex reasoning, architecture, security-critical code
- sonnet: Medium complexity, refactoring, feature implementation
- haiku: Simple tasks, tests, documentation, quick fixes

Assign each task to the most suitable model. Consider:
- Task complexity
- Need for deep reasoning vs speed
- Risk level (security, data integrity)

Return JSON only:
{{
  "assignments": [
    {{"task_num": "1", "model": "haiku", "reasoning": "Simple test writing"}},
    ...
  ]
}}"""

        try:
            # Use GLM via Z.ai for ranking - need enough tokens for large task lists
            response = self.invoke_provider(
                provider="z.ai",
                model="glm-4.6",
                messages=[{"role": "user", "content": ranking_prompt}],
                max_tokens=4096
            )

            response_text = ""
            for block in response.get("content", []):
                if block.get("type") == "text":
                    response_text += block.get("text", "")

            # Extract JSON - handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            # Parse JSON
            if "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]

                # Try to fix common JSON issues
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try fixing trailing commas
                    import re
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    result = json.loads(json_str)

                assignments = {}
                for a in result.get("assignments", []):
                    task_num = a.get("task_num")
                    # Find original task
                    original_task = next((t for t in tasks if t.get("num") == task_num), None)
                    assignments[task_num] = {
                        "model": a.get("model", "sonnet"),
                        "reasoning": a.get("reasoning", ""),
                        "task": original_task
                    }
                return assignments

        except Exception as e:
            print(f"⚠️  Ranking failed: {e}")

        # Fallback: assign all to sonnet
        return {
            t.get("num"): {"model": "sonnet", "reasoning": "Default", "task": t}
            for t in tasks
        }

    def _evaluate_continuation_strategy(self, next_tasks: List[str], iteration: int) -> Dict[str, Any]:
        """
        Evaluate whether to auto-continue turbo mode or present options to user.

        Args:
            next_tasks: List of next task descriptions
            iteration: Current RHSI iteration number

        Returns:
            {
                "strategy": "auto_continue" | "present_options",
                "reason": "explanation",
                "confidence": 0.0-1.0
            }
        """
        import anthropic

        if not next_tasks:
            return {
                "strategy": "present_options",
                "reason": "No specific tasks identified - user input needed",
                "confidence": 1.0
            }

        # Get recent history to check for rehashes
        history_context = ""
        history_file = self.palace_dir / "history.jsonl"
        if history_file.exists():
            recent_actions = []
            with open(history_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            if entry.get("action") in ["turbo_complete", "next"]:
                                recent_actions.append(entry)
                        except:
                            pass
            # Last 5 actions
            for action in recent_actions[-5:]:
                history_context += f"- {action.get('action')}: {action.get('details', {})}\n"

        prompt = f"""Evaluate continuation strategy for an RHSI turbo mode loop.

Current iteration: {iteration}

Next tasks identified:
{chr(10).join(f"- {t}" for t in next_tasks)}

Recent history:
{history_context if history_context else "No recent history"}

Decision criteria:
1. **Auto-continue** if:
   - Tasks are obvious completions of previous work
   - Tasks are fixing known issues from last iteration
   - No novel strategic decisions required
   - High confidence the right path is clear

2. **Present options** if:
   - Tasks represent new strategic directions
   - Multiple valid approaches exist
   - User input would materially improve outcome
   - Tasks require clarification or prioritization

Reply with JSON only:
{{"strategy": "auto_continue" or "present_options", "reason": "1-sentence explanation", "confidence": 0.0-1.0}}"""

        try:
            # Use GLM for quick evaluation
            zai_key = os.environ.get("ZAI_API_KEY", "")
            if zai_key:
                client = anthropic.Anthropic(
                    api_key=zai_key,
                    base_url="https://api.z.ai/api/anthropic"
                )
            else:
                client = anthropic.Anthropic()

            response = client.messages.create(
                model="glm-4.6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Parse JSON
            if "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                result = json.loads(response_text[json_start:json_end])

                # Validate and return
                if result.get("strategy") in ["auto_continue", "present_options"]:
                    return result

            # Fallback
            return {
                "strategy": "present_options",
                "reason": "Unable to evaluate - defaulting to user input",
                "confidence": 0.5
            }

        except Exception as e:
            # On error, default to presenting options (safer)
            return {
                "strategy": "present_options",
                "reason": f"Evaluation error: {str(e)[:30]}",
                "confidence": 0.0
            }

    def evaluate_turbo_completion(self) -> Dict[str, Any]:
        """
        Evaluate if turbo mode goals are complete using Opus.

        Reads README.md and specs, evaluates codebase state, decides if done.
        Returns: {complete: bool, reason: str, next_tasks: list}
        """
        import anthropic

        # Gather context
        readme_content = ""
        spec_content = ""

        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            readme_content = readme_path.read_text()[:4000]  # Limit size

        spec_path = self.project_root / "SPEC.md"
        if spec_path.exists():
            spec_content = spec_path.read_text()[:2000]

        # Get file listing
        try:
            import subprocess
            files = subprocess.run(
                ["find", ".", "-type", "f", "-name", "*.py", "-o", "-name", "*.md", "-o", "-name", "*.toml"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5
            ).stdout[:2000]
        except:
            files = ""

        prompt = f"""Evaluate if the project goals are complete.

README.md:
{readme_content}

{f"SPEC.md:{chr(10)}{spec_content}" if spec_content else ""}

Current files:
{files}

Based on the README requirements, is the project complete?

Reply with JSON only:
{{"complete": true/false, "reason": "brief explanation", "next_tasks": ["task1", "task2"] if not complete else []}}"""

        try:
            # Use Z.ai for evaluation (same as turbo mode agents)
            zai_key = os.environ.get("ZAI_API_KEY", "")
            if zai_key:
                client = anthropic.Anthropic(
                    api_key=zai_key,
                    base_url="https://api.z.ai/api/anthropic"
                )
            else:
                client = anthropic.Anthropic()

            response = client.messages.create(
                model="glm-4.6",  # Use GLM for cheap evaluation
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Parse JSON from response
            if "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                return json.loads(response_text[json_start:json_end])

            return {"complete": True, "reason": "Evaluation complete", "next_tasks": []}

        except Exception as e:
            # On error, assume complete to avoid infinite loops
            return {"complete": True, "reason": f"Evaluation error: {str(e)[:50]}", "next_tasks": []}

    def build_swarm_prompt(self, task: str, agent_id: str, base_context: str = "") -> str:
        """Build simple task prompt for swarm agent"""
        prompt_parts = [
            f"# Task: {task}",
            "",
            "Execute this task fully. Use your tools to read, write, edit files and run commands.",
            "Do NOT just plan - actually DO the work.",
            "",
        ]

        if base_context:
            prompt_parts.extend(["## Project Context", base_context, ""])

        return "\n".join(prompt_parts)

    def spawn_swarm(self, assignments: Dict[str, Dict[str, Any]],
                    base_prompt: str) -> Dict[str, Any]:
        """
        Spawn parallel Claude CLI processes for swarm execution.
        """
        processes = {}
        config = self.get_provider_config()

        for task_num, assignment in assignments.items():
            model_alias = assignment.get("model", "sonnet")  # Keep for effort labeling
            task = assignment.get("task", {})
            task_label = task.get("label", f"Task {task_num}")

            # TURBO MODE: Default to GLM-4.6 via Z.ai for cheap parallel execution
            # Override with --claude flag to use Claude models for higher quality
            # The model_alias (haiku/sonnet/opus) indicates ranked effort level
            if self.force_claude:
                # Use Claude models via Anthropic (higher cost, higher quality)
                provider = "anthropic"
                model_map = {
                    "haiku": "claude-3-5-haiku-20241022",
                    "sonnet": "claude-sonnet-4-5",
                    "opus": "claude-3-opus-20240229"
                }
                model = model_map.get(model_alias, "claude-sonnet-4-5")
            else:
                # Use GLM for cost efficiency (default turbo behavior)
                provider = "z.ai"
                model = "glm-4.6"
            provider_config = config["providers"].get(provider, {})

            agent_id = f"{model_alias}-{task_num}"

            # Build task prompt
            prompt = self.build_swarm_prompt(
                task=f"{task_label}: {task.get('description', '')}",
                agent_id=agent_id,
                base_context=base_prompt
            )

            # Prepare environment with provider credentials
            env = os.environ.copy()
            api_key_env = provider_config.get("api_key_env", "ANTHROPIC_API_KEY")
            base_url = provider_config.get("base_url", "")

            # Route to correct provider API
            if provider == "anthropic":
                # Use default Anthropic - unset any overrides
                env.pop("ANTHROPIC_AUTH_TOKEN", None)
                env.pop("ANTHROPIC_BASE_URL", None)
            elif provider == "z.ai":
                # Z.ai uses Anthropic format with different endpoint
                zai_key = os.environ.get("ZAI_API_KEY", "")
                if zai_key:
                    env["ANTHROPIC_AUTH_TOKEN"] = zai_key
                    env["ANTHROPIC_BASE_URL"] = "https://api.z.ai/api/anthropic"
            elif provider == "openrouter":
                # OpenRouter needs translation - TODO: set up translator proxy
                openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
                if openrouter_key:
                    env["ANTHROPIC_AUTH_TOKEN"] = openrouter_key
                    # TODO: Need translator proxy URL for OpenRouter
                    # env["ANTHROPIC_BASE_URL"] = "http://localhost:8080/openrouter"
            elif base_url and "anthropic.com" not in base_url:
                env["ANTHROPIC_BASE_URL"] = base_url

            # System prompt for swarm agent
            swarm_system = f"""You are agent {agent_id} in a parallel swarm.

When your task is complete: output a summary, then stop.

If you verify another agent's task is complete (tests pass), output:
[VERIFIED: agent-id]
This tells Palace to stop that agent."""

            # Build CLI command - bidirectional streaming JSON
            cmd = [
                "claude",
                "--model", model,
                "--append-system-prompt", swarm_system,
                "--verbose",
                "--input-format", "stream-json",
                "--output-format", "stream-json",
                "--permission-prompt-tool", "mcp__palace__handle_permission",
            ]

            # Debug: show what we're running and which API
            api_target = "Anthropic"
            if provider == "z.ai":
                api_target = "Z.ai"
            elif provider == "openrouter":
                api_target = "OpenRouter"
            print(f"   🚀 {agent_id}: {model} via {api_target}")

            try:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=self.project_root,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )

                # Send initial prompt via streaming JSON
                initial_message = {
                    "type": "user",
                    "message": {"role": "user", "content": prompt}
                }
                process.stdin.write(json.dumps(initial_message) + "\n")
                process.stdin.flush()

                processes[task_num] = {
                    "process": process,
                    "agent_id": agent_id,
                    "model": model_alias,
                    "task": task_label,
                }

            except Exception as e:
                print(f"❌ Failed to spawn {agent_id}: {e}")

        return processes

    def monitor_swarm(self, processes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor swarm processes and interleave their output.

        Returns results when all processes complete.
        """
        import select
        import threading
        from datetime import datetime

        results = {}
        active = dict(processes)

        # Print date header
        print(f"\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🐝 Swarm active: {len(active)} agents")
        print("─" * 50)

        # Track output buffers and seen tool IDs (to avoid duplicates)
        buffers = {num: "" for num in active}
        seen_tools = set()

        # Track which agents have marked themselves as "Done"
        # Done agents don't receive interleaved input until USER gives new input
        # This prevents the "long tail" problem of agents triggering each other
        done_agents = set()

        # Track agent confidence (based on activity patterns)
        # confidence: 1.0 = highly confident, 0.0 = stuck
        agent_confidence = {num: 1.0 for num in active}
        agent_last_action = {num: time.time() for num in active}
        agent_action_count = {num: 0 for num in active}

        def confidence_timestamp(task_num: str) -> str:
            """Return colored timestamp based on agent confidence"""
            conf = agent_confidence.get(task_num, 0.5)
            ts = datetime.now().strftime("%H:%M:%S")
            # ANSI colors: bright green → green → yellow → orange → red
            if conf >= 0.9:
                return f"\033[92m[{ts}]\033[0m"  # Bright green - highly confident
            elif conf >= 0.7:
                return f"\033[32m[{ts}]\033[0m"  # Dark green - very confident
            elif conf >= 0.5:
                return f"\033[33m[{ts}]\033[0m"  # Yellow - confident
            elif conf >= 0.3:
                return f"\033[38;5;208m[{ts}]\033[0m"  # Orange - uncertain
            else:
                return f"\033[91m[{ts}]\033[0m"  # Red - stuck/worried

        def update_confidence(task_num: str, action_type: str):
            """Update confidence based on action patterns"""
            now = time.time()
            elapsed = now - agent_last_action[task_num]
            agent_last_action[task_num] = now
            agent_action_count[task_num] += 1

            # Increase confidence on productive actions
            if action_type in ("tool_use", "text"):
                agent_confidence[task_num] = min(1.0, agent_confidence[task_num] + 0.1)
            # Decrease if long gap between actions (stuck)
            if elapsed > 10:
                agent_confidence[task_num] = max(0.0, agent_confidence[task_num] - 0.2)
            elif elapsed > 5:
                agent_confidence[task_num] = max(0.0, agent_confidence[task_num] - 0.1)

        while active:
            # Debug: show we're still monitoring
            # print(f"  [monitoring {len(active)} active agents]", end="\r", flush=True)
            for task_num, info in list(active.items()):
                process = info["process"]
                agent_id = info["agent_id"]

                # Check if process finished
                if process.poll() is not None:
                    # Read remaining output
                    remaining = process.stdout.read()
                    buffers[task_num] += remaining

                    # Calculate duration
                    duration = time.time() - agent_last_action.get(task_num, time.time())

                    results[task_num] = {
                        "agent": agent_id,
                        "exit_code": process.returncode,
                        "output": buffers[task_num]
                    }

                    # Debug: show exit code and output snippet if failed
                    if process.returncode != 0:
                        print(f"❌ {agent_id} failed (exit {process.returncode})")
                        # Show first 500 chars of output for debugging
                        output_preview = buffers[task_num][:500]
                        if output_preview:
                            print(f"   Output: {output_preview}")
                    else:
                        was_done = task_num in done_agents
                        status = "✅" if was_done else "⏱️"
                        print(f"{status} {agent_id} finished (exit 0, {'completed task' if was_done else 'no result msg'})")

                    # Only delete if still in active (may have been removed by peer verification)
                    if task_num in active:
                        del active[task_num]
                    continue

                # Skip processing if agent already marked done
                # (still in active dict waiting for process to exit, but no more output processing)
                if task_num in done_agents:
                    time.sleep(0.01)
                    continue

                # Read available output (non-blocking)
                try:
                    # Use select for non-blocking read
                    readable, _, _ = select.select([process.stdout], [], [], 0.1)
                    if readable:
                        line = process.stdout.readline()
                        if line:
                            buffers[task_num] += line

                            # Parse and show progress - SAME FORMAT as _process_stream_output
                            try:
                                msg = json.loads(line)
                                msg_type = msg.get("type")

                                if msg_type == "system" and msg.get("subtype") == "init":
                                    model = msg.get("model", "unknown")
                                    ts = confidence_timestamp(task_num)
                                    print(f"{ts}[{agent_id}] 📡 Model: {model}")

                                elif msg_type == "assistant":
                                    content = msg.get("message", {}).get("content", [])
                                    for block in content:
                                        if block.get("type") == "tool_use":
                                            tool_id = block.get("id", "")
                                            if tool_id in seen_tools:
                                                continue
                                            seen_tools.add(tool_id)

                                            update_confidence(task_num, "tool_use")
                                            ts = confidence_timestamp(task_num)
                                            tool_name = block.get("name", "unknown")
                                            tool_input = block.get("input", {})

                                            # Same formatting as _process_stream_output
                                            if tool_name == "Read":
                                                path = tool_input.get("file_path", "?").split("/")[-1]
                                                print(f"{ts}[{agent_id}] 📖 Reading: {path}")
                                            elif tool_name == "Edit":
                                                path = tool_input.get("file_path", "?").split("/")[-1]
                                                print(f"{ts}[{agent_id}] ✏️  Editing: {path}")
                                            elif tool_name == "Write":
                                                path = tool_input.get("file_path", "?").split("/")[-1]
                                                print(f"{ts}[{agent_id}] 📝 Writing: {path}")
                                            elif tool_name == "Bash":
                                                cmd = tool_input.get("command", "")[:50]
                                                print(f"{ts}[{agent_id}] 💻 Running: {cmd}...")
                                            elif tool_name == "Glob":
                                                print(f"{ts}[{agent_id}] 🔍 Finding: {tool_input.get('pattern', '')}")
                                            elif tool_name == "Grep":
                                                print(f"{ts}[{agent_id}] 🔎 Searching: {tool_input.get('pattern', '')}")
                                            else:
                                                print(f"{ts}[{agent_id}] 🔧 {tool_name}")

                                        elif block.get("type") == "text":
                                            text = block.get("text", "")
                                            if text:
                                                update_confidence(task_num, "text")
                                                ts = confidence_timestamp(task_num)
                                                print(f"{ts}[{agent_id}] {text}", end="", flush=True)

                                                # Check for peer verification
                                                import re
                                                verify_match = re.search(r'\[VERIFIED:\s*(\S+)\]', text)
                                                if verify_match:
                                                    target_agent = verify_match.group(1)
                                                    # Find and stop the verified agent
                                                    for other_num, other_info in list(active.items()):
                                                        if other_info["agent_id"] == target_agent and other_num not in done_agents:
                                                            print(f"\n🔒 {target_agent} verified complete by {agent_id}")
                                                            done_agents.add(other_num)
                                                            # Tell agent to stop
                                                            try:
                                                                stop_msg = {"type": "user", "message": {"role": "user", "content": "Your task has been verified complete. Stop now."}}
                                                                other_info["process"].stdin.write(json.dumps(stop_msg) + "\n")
                                                                other_info["process"].stdin.flush()
                                                                other_info["process"].stdin.close()
                                                                other_info["process"].terminate()
                                                            except:
                                                                pass
                                                            results[other_num] = {
                                                                "agent": target_agent,
                                                                "exit_code": 0,
                                                                "output": buffers.get(other_num, ""),
                                                                "completed": True,
                                                                "verified_by": agent_id
                                                            }
                                                            # Only delete if still in active
                                                            if other_num in active:
                                                                del active[other_num]

                                elif msg_type == "result":
                                    ts = confidence_timestamp(task_num)
                                    print(f"{ts}[{agent_id}] ✅ Done")
                                    # Mark agent as done - fully stop it NOW
                                    done_agents.add(task_num)
                                    # Kill the process immediately - don't wait for cleanup
                                    try:
                                        process.stdin.close()
                                        process.terminate()
                                    except:
                                        pass
                                    # Record result and remove from active
                                    results[task_num] = {
                                        "agent": agent_id,
                                        "exit_code": 0,
                                        "output": buffers[task_num],
                                        "completed": True
                                    }
                                    # Only delete if still in active (may have been removed by peer verification)
                                    if task_num in active:
                                        del active[task_num]
                                    break  # Stop processing this agent

                                # Forward to active (non-done) agents - convert output format to input format
                                # Output: {"type": "assistant", "message": {"content": [...]}}
                                # Input:  {"type": "user", "message": {"role": "user", "content": "..."}}
                                # Skip forwarding if sender is done (prevents long-tail triggering)
                                if msg_type == "assistant" and task_num not in done_agents:
                                    # Convert Anthropic content array to text for forwarding
                                    content = msg.get("message", {}).get("content", [])
                                    converted = self._convert_output_to_input(agent_id, content)
                                    if converted:
                                        context_msg = {
                                            "type": "user",
                                            "message": {"role": "user", "content": converted}
                                        }
                                        context_line = json.dumps(context_msg) + "\n"
                                        for other_num, other_info in active.items():
                                            # Skip: self, done agents
                                            if other_num != task_num and other_num not in done_agents:
                                                try:
                                                    other_info["process"].stdin.write(context_line)
                                                    other_info["process"].stdin.flush()
                                                except (BrokenPipeError, OSError):
                                                    pass

                            except json.JSONDecodeError:
                                pass

                except Exception:
                    pass

            time.sleep(0.05)

        print("─" * 50)
        print(f"🐝 Swarm complete: {len(results)} tasks finished")

        return results

    def _convert_output_to_input(self, agent_id: str, content: List[dict]) -> Optional[str]:
        """
        Convert Anthropic output content to input format for interleaving.

        Takes content array from assistant message and converts to string
        that can be sent as user message content to other agents.
        """
        parts = []

        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
            elif block_type == "tool_use":
                # Format tool use as JSON for other agents to parse
                tool_info = {
                    "tool": block.get("name"),
                    "id": block.get("id"),
                    "input": block.get("input", {})
                }
                parts.append(f"[TOOL_USE: {json.dumps(tool_info)}]")

        if parts:
            return f"[{agent_id}] " + " ".join(parts)
        return None

    def run_turbo_mode(self, tasks: List[dict], base_context: str = "") -> Dict[str, Any]:
        """
        Run turbo mode: rank tasks, spawn swarm, monitor.

        Returns results from all swarm agents.
        """
        print("\n🚀 TURBO MODE ACTIVATED")
        if self.force_claude:
            print("💎 Using Claude models (high quality, higher cost)")
        else:
            print("💰 Using GLM-4.6 (cost-efficient, fast)")
        print("─" * 50)

        # Step 1: Rank tasks by model
        print("📊 Ranking tasks by optimal model...")
        assignments = self.rank_tasks_by_model(tasks)

        for num, a in assignments.items():
            print(f"   {num}. {a['task'].get('label', 'Task')} → {a['model']} ({a['reasoning']})")

        print()

        # Step 2: Spawn swarm
        print("🐝 Spawning parallel agents...")
        processes = self.spawn_swarm(assignments, base_context)

        # Step 3: Monitor and interleave
        results = self.monitor_swarm(processes)

        return {
            "assignments": assignments,
            "results": results
        }

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

        # Determine which model to use
        if self.force_glm:
            # Use GLM for cost savings (requires Z.ai API key)
            model = "glm-4.6"
            # Set up environment for Z.ai
            env = os.environ.copy()
            zai_key = os.environ.get("ZAI_API_KEY", "")
            if zai_key:
                env["ANTHROPIC_AUTH_TOKEN"] = zai_key
                env["ANTHROPIC_BASE_URL"] = "https://api.z.ai/api/anthropic"
        else:
            # Use Claude Sonnet (default)
            model = "claude-sonnet-4-5"
            env = None  # Use default environment

        cmd = [
            "claude",
            "--model", model,
            "--append-system-prompt", menu_prompt,
            "--verbose",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--permission-prompt-tool", "mcp__palace__handle_permission",
        ]

        print("🏛️  Palace - Invoking Claude...")
        if not self.strict_mode:
            print("⚡ YOLO mode active - test validation disabled")
        if self.force_glm:
            print("💰 GLM mode active - using GLM-4.6 for cost savings")
        print()

        try:
            # Bidirectional streaming JSON
            popen_args = {
                "cwd": self.project_root,
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True
            }
            if env is not None:
                popen_args["env"] = env

            process = subprocess.Popen(cmd, **popen_args)

            # Send initial prompt via streaming JSON
            initial_message = {
                "type": "user",
                "message": {"role": "user", "content": prompt}
            }
            process.stdin.write(json.dumps(initial_message) + "\n")
            process.stdin.flush()
            process.stdin.close()  # Signal end of input

            selected_action = self._process_stream_output(process.stdout, process)
            process.wait()

            # COMPLETION HOOK: Validate tests in strict mode
            if self.strict_mode:
                modified_files_path = self.palace_dir / "modified_files.json"
                modified_files = set()

                if modified_files_path.exists():
                    try:
                        with open(modified_files_path, 'r') as f:
                            modified_files = set(json.load(f))
                    except Exception:
                        pass  # Ignore read errors

                if modified_files:
                    print("\n🔒 Strict mode: Validating tests for modified files...")
                    test_files = self.detect_affected_tests(modified_files)

                    if test_files:
                        print(f"📝 Running {len(test_files)} test file(s)...")
                        tests_passed = self.run_test_subset(test_files)

                        if not tests_passed:
                            print("\n❌ Strict mode: Tests must pass before completion")
                            print("   Modified files:", ", ".join(sorted(modified_files)))
                            print("   Fix the failing tests to complete this session.")
                            return 1, selected_action  # Non-zero exit code

                        print("✅ All tests passed!")

                        # Clear modified files after successful test run
                        try:
                            modified_files_path.unlink()
                        except Exception:
                            pass
                    else:
                        print("⚠️  No tests found for modified files (consider adding tests)")

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

    def _process_stream_output(self, stream, process=None):
        """Process streaming JSON output and display succinct progress"""
        import threading

        text_by_msg = {}  # Track text per message ID
        full_text = ""  # Buffer all text for menu detection
        seen_tools = set()
        current_line_len = 0  # Track chars on current line for clearing

        # Start escape monitoring thread
        stop_escape_monitor = threading.Event()
        escape_detected = threading.Event()
        self._setup_escape_handler()

        def monitor_escape():
            while not stop_escape_monitor.is_set():
                if self._check_for_escape():
                    escape_detected.set()
                    break
                time.sleep(0.05)  # Check 20 times per second

        escape_thread = threading.Thread(target=monitor_escape, daemon=True)
        escape_thread.start()

        user_steering = None

        for line in stream:
            # Check for ESC-ESC interrupt
            if escape_detected.is_set():
                stop_escape_monitor.set()
                result = self._handle_user_interrupt()
                if result and result.get("action") == "abort":
                    # Kill the subprocess
                    if process:
                        process.terminate()
                    return None
                elif result and result.get("action") == "steer":
                    user_steering = result.get("steering")
                    # Continue processing but remember steering for next iteration
                # Reset and continue
                escape_detected.clear()
                self._setup_escape_handler()
                escape_thread = threading.Thread(target=monitor_escape, daemon=True)
                escape_thread.start()
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

        # Stop escape monitoring
        stop_escape_monitor.set()

        print()

        # Check for action menu and show selector
        actions = self._parse_actions_menu(full_text)

        # Always show the steering prompt
        result = self._show_steering_prompt(actions)

        # If user provided steering during execution, attach it
        if user_steering and result:
            for action in result:
                if "_modifiers" not in action:
                    action["_modifiers"] = []
                action["_modifiers"].append(f"USER STEERING: {user_steering}")

        return result

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
        turbo_mode = getattr(args, 'turbo', False)
        guidance = ' '.join(getattr(args, 'guidance', []))

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
        if guidance:
            initial_prompt = f"""Analyze this project and suggest possible next actions.

USER GUIDANCE: {guidance}

Focus your suggestions on what the user has asked for above.
Check SPEC.md and ROADMAP.md if they exist for context.

Provide as many options as you see fit - there may be many valid paths forward.
Be concrete and actionable. The user will select which action(s) to execute."""
        else:
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

            # Turbo mode: parallel swarm execution
            if turbo_mode:
                # Convert selected actions to task format for turbo mode
                tasks = []
                for i, action in enumerate(selected_actions):
                    tasks.append({
                        "num": str(i + 1),
                        "label": action.get("label", f"Task {i + 1}"),
                        "description": action.get("description", "")
                    })

                # Run turbo mode with all selected actions in parallel
                result = self.run_turbo_mode(tasks, json.dumps(context))

                self.log_action("turbo_complete", {
                    "session_id": session_id,
                    "tasks": len(tasks),
                    "results": {k: v.get("exit_code", -1) for k, v in result.get("results", {}).items()}
                })

                print("\n✅ Swarm round complete!")

                # Evaluate if we should continue or stop
                evaluation = self.evaluate_turbo_completion()

                if evaluation.get("complete"):
                    print(f"\n🏆 {evaluation.get('reason', 'Goals achieved!')}")
                    break
                else:
                    # Not complete - intelligently decide next steps
                    print(f"\n🔄 {evaluation.get('reason', 'More work needed...')}")
                    next_tasks = evaluation.get("next_tasks", [])

                    # Evaluate if we should present options or auto-continue
                    continuation = self._evaluate_continuation_strategy(next_tasks, iteration)

                    if continuation["strategy"] == "auto_continue":
                        print(f"🤖 {continuation['reason']}")
                        print("🚀 Auto-continuing turbo mode...")
                        # Convert tasks to actions and continue automatically
                        selected_actions = [{"label": t, "description": ""} for t in next_tasks]
                        current_prompt = self._build_action_prompt(selected_actions)
                        continue
                    elif continuation["strategy"] == "present_options":
                        print(f"💡 {continuation['reason']}")
                        print("📋 Generating options for user selection...")
                        # Generate and present action menu to user
                        # Break out to normal RHSI loop with action menu
                        current_prompt = "Evaluate the current state and suggest next actions. Include an ACTIONS: section."
                        break  # Exit turbo loop, present menu to user
                    else:
                        # Fallback: let Claude decide
                        current_prompt = "Evaluate the current state and suggest next actions. Include an ACTIONS: section."
                        continue

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

    def detect_affected_tests(self, modified_files: set) -> set:
        """
        Detect which test files should run based on modified files.

        Uses pytest --collect-only to map files to tests.
        Returns set of test file paths.
        """
        if not modified_files:
            return set()

        test_files = set()
        tests_dir = self.project_root / "tests"

        if not tests_dir.exists():
            return set()

        # For each modified file, try to find corresponding test file
        for modified_file in modified_files:
            try:
                file_path = Path(modified_file)

                # Skip if file is already a test
                if file_path.parts and file_path.parts[0] == "tests":
                    test_files.add(str(file_path))
                    continue

                # Try to find test file by naming convention
                # e.g., palace.py -> test_palace.py or test_core.py
                filename = file_path.stem

                # Look for test files that might test this module
                possible_patterns = [
                    f"test_{filename}.py",
                    f"test_*{filename}*.py",
                    f"test_core.py",  # Core tests likely test main module
                ]

                for pattern in possible_patterns:
                    matching_tests = list(tests_dir.glob(f"**/{pattern}"))
                    for test_file in matching_tests:
                        test_files.add(str(test_file.relative_to(self.project_root)))

            except Exception as e:
                # If detection fails, skip this file
                print(f"⚠️  Could not detect tests for {modified_file}: {e}")
                continue

        # If no specific tests found, run all tests (safer in strict mode)
        if not test_files and modified_files:
            all_tests = list(tests_dir.glob("**test_*.py"))
            test_files = {str(t.relative_to(self.project_root)) for t in all_tests}

        return test_files

    def run_test_subset(self, test_files: set = None) -> bool:
        """
        Run a subset of tests.

        Args:
            test_files: Set of test file paths to run. If None, runs all tests.

        Returns:
            True if tests pass, False otherwise
        """
        if not test_files:
            # Run all tests
            cmd = ["python3", "-m", "pytest", "tests/", "-x", "--tb=short"]
        else:
            # Run specific test files
            cmd = ["python3", "-m", "pytest", "-x", "--tb=short"] + list(test_files)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"\n❌ Tests failed:\n{result.stdout}\n{result.stderr}")
                return False

            return True

        except Exception as e:
            print(f"⚠️  Error running tests: {e}")
            return False

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

        palace_dir = Path(__file__).resolve().parent
        palace_venv = palace_dir / ".venv"
        uv_path = Path.home() / ".local" / "bin" / "uv"

        # Step 1: Ensure uv is installed
        if not uv_path.exists():
            # Check if uv is in PATH
            uv_in_path = shutil.which("uv")
            if uv_in_path:
                uv_path = Path(uv_in_path)
            else:
                print("📦 Installing uv (Python package manager)...")
                try:
                    result = subprocess.run(
                        ["curl", "-LsSf", "https://astral.sh/uv/install.sh"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        # Pipe to sh
                        install_result = subprocess.run(
                            ["sh"], input=result.stdout,
                            capture_output=True, text=True
                        )
                        if install_result.returncode == 0:
                            print("✅ uv installed")
                            uv_path = Path.home() / ".local" / "bin" / "uv"
                        else:
                            print(f"⚠️  uv install failed: {install_result.stderr}")
                            print("   Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh")
                            return
                    else:
                        print("⚠️  Could not download uv installer")
                        print("   Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh")
                        return
                except FileNotFoundError:
                    print("⚠️  curl not found. Install uv manually:")
                    print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
                    return

        # Step 2: Create venv and sync dependencies
        print("📦 Setting up Python environment...")
        try:
            # uv sync creates venv and installs from pyproject.toml
            result = subprocess.run(
                [str(uv_path), "sync"],
                cwd=palace_dir,
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("✅ Dependencies installed")
            else:
                print(f"⚠️  uv sync failed: {result.stderr}")
                # Fallback: try creating venv and pip install
                if not palace_venv.exists():
                    subprocess.run([str(uv_path), "venv"], cwd=palace_dir, capture_output=True)
                deps = ["questionary", "rich", "anthropic", "mcp", "json-stream", "pytest"]
                subprocess.run(
                    [str(uv_path), "pip", "install"] + deps,
                    cwd=palace_dir, capture_output=True
                )
                print("✅ Dependencies installed (fallback)")
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

        # Install 'pal' CLI alias
        self._install_pal_alias(palace_path)

        print()
        print("🎉 Palace is now integrated with Claude Code!")
        print()
        print("Output style 'palace-menu' enables action menus.")
        print("Try: /pal-next or just: pal next")

    def _install_pal_alias(self, palace_path: Path):
        """Install 'pal' command alias to ~/.local/bin"""
        bin_dir = Path.home() / ".local" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)

        pal_script = bin_dir / "pal"

        # Always use the venv python (created by install)
        palace_venv_python = palace_path.parent / ".venv" / "bin" / "python"

        script_content = f'''#!/bin/bash
# Palace CLI - installed by 'palace.py install'
# Usage: pal <command> [args]
#   pal next      - Suggest next step
#   pal next <guidance> - Suggest with focus (e.g., "pal next focus on testing")
#   pal new       - Create new project
#   pal scaffold  - Scaffold project
#   pal test      - Run tests
#   pal install   - Install/update Palace
#   pal sessions  - List sessions

exec "{palace_venv_python}" "{palace_path}" "$@"
'''

        with open(pal_script, 'w') as f:
            f.write(script_content)

        # Make executable
        pal_script.chmod(0o755)

        print(f"✅ Installed 'pal' command: {pal_script}")

        # Check if ~/.local/bin is in PATH
        path_dirs = os.environ.get("PATH", "").split(":")
        if str(bin_dir) not in path_dirs:
            print()
            print("⚠️  Add ~/.local/bin to your PATH:")
            print('   export PATH="$HOME/.local/bin:$PATH"')
            print("   (Add to ~/.bashrc or ~/.zshrc for persistence)")

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

    def cmd_export(self, args):
        """Export a session to a file"""
        session_id = args.session_id
        output = getattr(args, 'output', None)

        result = self.export_session(session_id, output)
        if result:
            print(f"📦 Exported session {session_id}")
            print(f"   → {result}")
        else:
            print(f"❌ Session {session_id} not found")

    def cmd_import(self, args):
        """Import a session from a file"""
        import_path = args.file
        new_id = getattr(args, 'session_id', None)

        result = self.import_session(import_path, new_id)
        if result:
            print(f"📥 Imported session from {import_path}")
            print(f"   → New session ID: {result}")
            print()
            print("Resume with:")
            print(f'  python3 palace.py next --resume {result}')
        else:
            print(f"❌ Failed to import session from {import_path}")

    def cmd_analyze(self, args):
        """Self-analysis: Palace analyzes itself"""
        print("🔍 Palace Self-Analysis")
        print()
        print("Analyzing Palace's own codebase...")
        print()

        # Gather metrics about Palace itself
        metrics = {}

        # Code metrics
        palace_file = Path(__file__)
        if palace_file.exists():
            lines = palace_file.read_text().split('\n')
            metrics['total_lines'] = len(lines)
            metrics['code_lines'] = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
            metrics['comment_lines'] = len([l for l in lines if l.strip().startswith('#')])

        # Test coverage
        test_files = list(Path('tests').glob('test_*.py'))
        metrics['test_files'] = len(test_files)
        metrics['test_lines'] = sum(len(f.read_text().split('\n')) for f in test_files if f.exists())

        # Session data
        sessions = self.list_sessions()
        metrics['total_sessions'] = len(sessions)

        # History depth
        history_file = self.palace_dir / "history.jsonl"
        if history_file.exists():
            metrics['history_entries'] = len(history_file.read_text().strip().split('\n'))

        # Mask system
        masks = self.list_masks()
        metrics['available_masks'] = len([m for m in masks if m['type'] == 'available'])
        metrics['custom_masks'] = len([m for m in masks if m['type'] == 'custom'])

        # Display metrics
        print("📊 Code Metrics:")
        print(f"   Total lines: {metrics.get('total_lines', 0)}")
        print(f"   Code lines: {metrics.get('code_lines', 0)}")
        print(f"   Comment lines: {metrics.get('comment_lines', 0)}")
        print(f"   Test files: {metrics.get('test_files', 0)}")
        print(f"   Test lines: {metrics.get('test_lines', 0)}")
        print()

        print("📋 Session Data:")
        print(f"   Total sessions: {metrics.get('total_sessions', 0)}")
        print(f"   History entries: {metrics.get('history_entries', 0)}")
        print()

        print("🎭 Mask System:")
        print(f"   Available masks: {metrics.get('available_masks', 0)}")
        print(f"   Custom masks: {metrics.get('custom_masks', 0)}")
        print()

        # Analyze recent history for patterns
        if history_file.exists():
            print("📈 Recent Activity Patterns:")
            action_types = {}
            for line in history_file.read_text().strip().split('\n')[-50:]:
                try:
                    entry = json.loads(line)
                    action = entry.get('action', 'unknown')
                    action_types[action] = action_types.get(action, 0) + 1
                except:
                    pass

            for action, count in sorted(action_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {action}: {count}")
            print()

        # Suggestions for improvement
        print("💡 Suggestions:")
        suggestions = []

        code_to_test_ratio = metrics.get('code_lines', 0) / max(metrics.get('test_lines', 1), 1)
        if code_to_test_ratio > 1.5:
            suggestions.append("Consider adding more tests (current ratio: {:.1f}:1)".format(code_to_test_ratio))

        if metrics.get('total_sessions', 0) > 10:
            suggestions.append("Run 'palace cleanup' to remove old sessions")

        if metrics.get('custom_masks', 0) == 0:
            suggestions.append("Create custom masks for project-specific expertise")

        if not suggestions:
            suggestions.append("Palace is in good shape!")

        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        print()

        # Save analysis
        analysis = {
            "timestamp": time.time(),
            "metrics": metrics,
            "suggestions": suggestions
        }

        analysis_file = self.palace_dir / "self_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"📝 Analysis saved to {analysis_file}")

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
    parser.add_argument('--strict', action='store_true', default=True,
                        help='Strict mode: Enforce test validation (default)')
    parser.add_argument('--yolo', action='store_true',
                        help='YOLO mode: Skip all test validation (--no-strict)')
    parser.add_argument('--claude', action='store_true',
                        help='Use Claude models even in turbo mode (higher quality, higher cost)')
    parser.add_argument('--glm', action='store_true',
                        help='Use GLM model even in normal mode (lower cost, faster)')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Commands that invoke Claude
    parser_next = subparsers.add_parser('next', help='Ask Claude what to do next (RHSI core)')
    parser_next.add_argument('guidance', nargs='*', default=[],
                             help='Optional guidance to focus suggestions (e.g., "focus on testing")')
    parser_next.add_argument('--resume', '-r', metavar='SESSION_ID',
                             help='Resume a paused session')
    parser_next.add_argument('--select', '-s', metavar='SELECTION',
                             help='Select actions: "1,2,3" or "do 1 but skip tests"')
    parser_next.add_argument('--turbo', '-t', action='store_true',
                             help='Turbo mode: parallel swarm execution with model-task ranking')

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

    parser_export = subparsers.add_parser('export', help='Export a session to a file')
    parser_export.add_argument('session_id', help='Session ID to export')
    parser_export.add_argument('--output', '-o', help='Output file path (default: SESSION_ID_export.json)')

    parser_import = subparsers.add_parser('import', help='Import a session from a file')
    parser_import.add_argument('file', help='Path to exported session file')
    parser_import.add_argument('--session-id', help='New session ID (default: auto-generate)')

    subparsers.add_parser('analyze', help='Self-analysis: Palace analyzes itself')

    subparsers.add_parser('permissions', help='Handle Claude Code permission requests (internal)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Determine strict mode: --yolo disables, otherwise default to --strict
    strict_mode = not args.yolo if hasattr(args, 'yolo') else args.strict

    # Determine provider overrides
    force_claude = getattr(args, 'claude', False)
    force_glm = getattr(args, 'glm', False)

    palace = Palace(strict_mode=strict_mode, force_claude=force_claude, force_glm=force_glm)

    commands = {
        'next': palace.cmd_next,
        'new': palace.cmd_new,
        'scaffold': palace.cmd_scaffold,
        'test': palace.cmd_test,
        'install': palace.cmd_install,
        'init': palace.cmd_init,
        'sessions': palace.cmd_sessions,
        'cleanup': palace.cmd_cleanup,
        'export': palace.cmd_export,
        'import': palace.cmd_import,
        'analyze': palace.cmd_analyze,
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
                model="claude-haiku-4-5",
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
                approved = result.get("approved", True)
                reason = result.get("reason", "Assessed by Haiku")
                # Return Claude Code expected format
                if approved:
                    return {"behavior": "allow", "updatedInput": tool_input}
                else:
                    return {"behavior": "deny", "message": reason}

            # Fallback: approve if we can't parse
            return {"behavior": "allow", "updatedInput": tool_input}

        except Exception as e:
            # On any error, approve but log the issue
            return {"behavior": "allow", "updatedInput": tool_input}

    @mcp.tool()
    def handle_permission(
        tool_name: str = "",
        input: dict = None,
        tool_use_id: str = ""
    ) -> dict:
        """
        Handle permission requests from Claude during RHSI loops.

        Claude Code sends permission requests with tool_name, input, and tool_use_id.
        Uses Haiku to assess safety based on the trainable command-safety skill.

        Returns:
            dict with 'behavior' ("allow"/"deny") and 'updatedInput' or 'message'
        """
        # Initialize Palace instance to access logging
        palace = Palace()
        tool_input = input or {}

        # Log the permission request
        request_data = {
            "tool_name": tool_name,
            "input": tool_input,
            "tool_use_id": tool_use_id
        }
        palace.log_action("permission_request", {"request": request_data})

        # Assess safety using Haiku and the command-safety skill
        result = _assess_permission_safety(tool_name, tool_input)

        # Track file modifications in strict mode
        if result.get("behavior") == "allow" and tool_name in ["Write", "Edit"]:
            file_path = tool_input.get("file_path", "")
            if file_path:
                # Store modified file in session tracker
                modified_files_path = palace.palace_dir / "modified_files.json"
                try:
                    if modified_files_path.exists():
                        with open(modified_files_path, 'r') as f:
                            modified_files = set(json.load(f))
                    else:
                        modified_files = set()

                    modified_files.add(file_path)

                    palace.ensure_palace_dir()
                    with open(modified_files_path, 'w') as f:
                        json.dump(list(modified_files), f)
                except Exception as e:
                    # Don't fail on tracking errors
                    print(f"⚠️  Could not track modified file: {e}")

        # Log the decision
        palace.log_action("permission_decision", {
            "tool_name": tool_name,
            "behavior": result.get("behavior"),
            "message": result.get("message", "")
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
