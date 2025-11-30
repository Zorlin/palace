"""
Stream processing and output formatting for Palace
"""

import json
import re
import subprocess
from typing import Optional, List, Dict, Any, Tuple


class StreamProcessor:
    """Handles Claude CLI stream processing and output formatting"""

    def invoke_claude_cli(self, prompt: str, degradation_mode: str = "retry") -> Tuple[int, Optional[List[dict]]]:
        """
        Invoke Claude CLI with a prompt.

        Processes streaming JSON output and extracts ACTIONS sections.

        Returns: (exit_code, actions_list)
        """
        # Write prompt to file
        if hasattr(self, 'palace_dir'):
            prompt_file = self.palace_dir / "current_prompt.md"
            prompt_file.parent.mkdir(parents=True, exist_ok=True)
            with open(prompt_file, 'w') as f:
                f.write(prompt)

        # Build claude command
        cmd = [
            "claude",
            "-p", prompt,
            "--output-format", "stream-json",
            "--permission-prompt-tool", "mcp__palace__handle_permission"
        ]

        try:
            # Run Claude CLI
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Process streaming output
            full_text_buffer = ""
            actions = []

            for line in proc.stdout:
                try:
                    event = json.loads(line)

                    # Handle different event types
                    if event.get("type") == "assistant":
                        message = event.get("message", {})
                        for block in message.get("content", []):
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                full_text_buffer += text
                                print(text, end="", flush=True)

                    elif event.get("type") == "result":
                        # Final result
                        pass

                except json.JSONDecodeError:
                    # Non-JSON line, print as-is
                    print(line, end="")

            proc.wait()

            # Extract ACTIONS from buffer
            if full_text_buffer:
                actions = self._extract_actions_from_text(full_text_buffer)

            return proc.returncode, actions

        except Exception as e:
            print(f"❌ Error invoking Claude: {e}")
            return 1, None

    def _extract_actions_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        Extract ACTIONS section from Claude's response.

        Looks for:
        ACTIONS:
        1. Label - description
        2. Another - more details
        ...

        Returns list of action dicts with num, label, description.
        """
        actions = []

        # Find ACTIONS: section
        actions_match = re.search(r'ACTIONS?:\s*\n((?:\d+\..*\n?)+)', text, re.IGNORECASE)
        if not actions_match:
            return actions

        actions_text = actions_match.group(1)

        # Parse numbered items
        for match in re.finditer(r'(\d+)\.\s*(.+?)(?:\s*-\s*(.+?))?(?=\n\d+\.|\n*$)', actions_text, re.DOTALL):
            num = match.group(1)
            label = match.group(2).strip()
            description = match.group(3).strip() if match.group(3) else ""

            actions.append({
                "num": num,
                "label": label,
                "description": description
            })

        return actions

    def _format_action_menu(self, actions: List[dict]) -> str:
        """
        Format actions as an interactive menu.

        Returns formatted menu string.
        """
        if not actions:
            return ""

        menu = ["\n╔═══════════════════════════════════════════════════════════════╗"]
        menu.append("║  NEXT ACTIONS                                                  ║")
        menu.append("╠═══════════════════════════════════════════════════════════════╣")

        for action in actions:
            num = action.get("num", "?")
            label = action.get("label", "")
            desc = action.get("description", "")

            # Truncate if too long
            display = f"{num}. {label}"
            if len(display) > 58:
                display = display[:55] + "..."

            menu.append(f"║  {display:<60} ║")
            if desc:
                desc_line = f"    {desc}"
                if len(desc_line) > 58:
                    desc_line = desc_line[:55] + "..."
                menu.append(f"║  {desc_line:<60} ║")

        menu.append("╚═══════════════════════════════════════════════════════════════╝")
        menu.append("\nSelect: <numbers> (e.g. '1 2 3' or '1-5')")
        menu.append("        or: describe a custom task")
        menu.append("        or: /quit to exit\n")

        return "\n".join(menu)

    def display_actions_menu(self, actions: List[dict]) -> str:
        """Display actions menu and get user selection"""
        menu = self._format_action_menu(actions)
        print(menu)

        try:
            selection = input("👉 ").strip()
            return selection
        except (EOFError, KeyboardInterrupt):
            return "/quit"

    def _display_context_info(self, context: Dict[str, Any]):
        """Display minimal context info to user"""
        print("\n📂 PROJECT STATE:\n")

        # Show important files
        if context.get("files"):
            print("Existing files:")
            for filename, info in context["files"].items():
                if info.get("exists"):
                    size = info.get("size", 0)
                    print(f"  ✓ {filename} ({size} bytes)")

        # Show git status
        if context.get("git_status"):
            status = context["git_status"].strip()
            if status:
                print("\nGit status:")
                for line in status.split("\n")[:5]:  # Max 5 lines
                    print(f"  {line}")
                if len(status.split("\n")) > 5:
                    print("  ...")

        print()
        print("─" * 60)
        print()
