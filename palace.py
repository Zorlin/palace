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

    def parse_action_selection(self, selection: str, actions: List[dict]) -> List[dict]:
        """
        Parse action selection string into list of actions.

        Supports:
        - Numeric: "0,1,2,3,5,10" or "1-5" or "1,3-5,7"
        - Natural language: "do 5 but skip the tests" (parsed by Claude)

        Returns list of selected actions, potentially modified by natural language instructions.
        """
        selected = []
        modifiers = []

        # Check if it's purely numeric selection
        numeric_pattern = r'^[\d,\s\-]+$'
        if re.match(numeric_pattern, selection.strip()):
            # Parse numeric selection
            parts = selection.replace(" ", "").split(",")
            for part in parts:
                if "-" in part:
                    # Range: "1-5"
                    try:
                        start, end = part.split("-")
                        for i in range(int(start), int(end) + 1):
                            for a in actions:
                                if a.get("num") == str(i):
                                    selected.append(a)
                                    break
                    except:
                        pass
                else:
                    # Single number
                    for a in actions:
                        if a.get("num") == part:
                            selected.append(a)
                            break
        else:
            # Natural language selection - extract numbers and modifiers
            # Pattern: "do 1,2,3 but don't do X" or "5 but with Y"
            numbers = re.findall(r'\b(\d+)\b', selection)
            for num in numbers:
                for a in actions:
                    if a.get("num") == num:
                        selected.append(a)
                        break

            # Extract modifiers (text after "but", "except", "with", etc.)
            modifier_patterns = [
                r'\bbut\s+(.+)',
                r'\bexcept\s+(.+)',
                r'\bwith\s+(.+)',
                r'\bwithout\s+(.+)',
                r'\bskip\s+(.+)',
            ]
            for pattern in modifier_patterns:
                match = re.search(pattern, selection, re.IGNORECASE)
                if match:
                    modifiers.append(match.group(1).strip())

            # Store modifiers in each selected action
            if modifiers:
                for a in selected:
                    a["_modifiers"] = modifiers

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

    def invoke_claude_cli(self, prompt: str) -> Tuple[int, Optional[List[dict]]]:
        """
        Invoke Claude Code CLI directly (interactive mode)

        Returns tuple of (exit_code, selected_actions)
        - exit_code: 0 on success, non-zero on error
        - selected_actions: list of action dicts if user selected any, None otherwise
        """
        # Menu format instructions for action selection
        menu_prompt = """When presenting choices or suggesting next actions, format as:

ACTIONS:
1. First option with description
   Additional context if needed.

2. Second option
   More details here.

Use numbered items, with optional indented descriptions."""

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
        """Parse ACTIONS: menu from text, return list of action dicts"""
        if "ACTIONS:" not in text:
            return []

        actions = []
        lines = text.split("ACTIONS:", 1)[1].strip().split("\n")
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
        """Simple fallback menu without questionary"""
        print("\n💡 Select action(s) (comma-separated, 0 to exit, or type custom task):")
        for a in actions:
            print(f"  {a['num']}. {a['label']}")
        print("  0. Exit loop")

        choice = input("\nEnter number(s) or custom task: ").strip()

        if choice == "0" or choice.lower() in ("q", "quit", "exit"):
            return None

        # Check if it's a custom task (not a number)
        if choice and not choice.replace(",", "").replace(" ", "").isdigit():
            return [{"num": "c", "label": choice, "description": "Custom task", "_custom": True}]

        # Parse comma-separated numbers
        selected = []
        for num in choice.split(","):
            num = num.strip()
            for a in actions:
                if a["num"] == num:
                    selected.append(a)
                    break

        return selected if selected else None

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
        if actions:
            selected = self._show_action_menu(actions)
            if selected and len(selected) > 0:
                # Show what was selected
                if len(selected) == 1:
                    label = selected[0].get("label", "")
                    print(f"\n🎯 Selected: {label}")
                else:
                    print(f"\n🎯 Selected {len(selected)} actions:")
                    for s in selected:
                        print(f"   • {s.get('label', '')}")
                return selected
            return None  # User cancelled

        # No actions detected - prompt for custom task
        custom_task = self._prompt_custom_task()
        if custom_task:
            return [{"num": "c", "label": custom_task, "description": "Custom task", "_custom": True}]
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
        initial_prompt = """Analyze this project and suggest what to do next.

Consider:
1. What exists already (check the files context)
2. What's in progress (check git status and recent history)
3. What should come next in a logical development flow
4. The project's SPEC.md and ROADMAP.md if they exist

Provide:
- A clear next action to take
- Why this is the logical next step
- How to execute it (specific commands or steps)

Be concrete and actionable. This is part of a Recursive Hierarchical Self Improvement loop,
so your suggestion will be used to actually advance the project."""

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

After completing, suggest what to do next. Include an ACTIONS: section with your recommendations."""
            else:
                task_desc = action.get("label", "")
                task_detail = action.get("description", "")
                modifiers = action.get("_modifiers", [])
                mod_text = f"\nModifications: {', '.join(modifiers)}" if modifiers else ""
                return f"""Execute this action: {task_desc}
{f"Details: {task_detail}" if task_detail else ""}{mod_text}

After completing, suggest what to do next. Include an ACTIONS: section with your recommendations."""
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

After completing all, suggest what to do next. Include an ACTIONS: section with your recommendations."""

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
        """Install Palace commands and output style into Claude Code"""

        print("🏛️  Palace - Installing to Claude Code")
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

try:
    from mcp.server.fastmcp import FastMCP

    # Create MCP server instance
    mcp = FastMCP("Palace")

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
        We auto-approve all requests for RHSI autonomous operation.

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

        # Auto-approve all requests for RHSI loops
        # TODO: Add smart permission logic based on learning from history
        return {"approved": True}

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
