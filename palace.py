#!/usr/bin/env python3
"""
Palace - A self-improving Claude wrapper
Recursive Hierarchical Self Improvement (RHSI) for Claude

Palace is NOT a replacement for Claude - it's an orchestration layer.
Every command invokes Claude with context, letting Claude use its full power.
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

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

        import time
        entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details or {}
        }

        with open(history_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

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

    def invoke_claude_cli(self, prompt: str) -> int:
        """
        Invoke Claude Code CLI directly (interactive mode)

        Returns the exit code from Claude
        """
        palace_path = Path(__file__).resolve()

        cmd = [
            "claude",
            "-p", prompt,
            "--verbose",
            "--system-prompt",
            "--include-partial-messages",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--permission-prompt-tool", f"python3 {palace_path} permissions"
        ]

        print("🏛️  Palace - Invoking Claude Code CLI...")
        print()

        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode
        except FileNotFoundError:
            print("❌ Claude Code CLI not found. Make sure 'claude' is in your PATH.")
            print("   Install from: https://code.claude.com/")
            return 1
        except Exception as e:
            print(f"❌ Error invoking Claude: {e}")
            return 1

    def invoke_claude(self, prompt: str, context: Dict[str, Any] = None):
        """
        Invoke Claude with a prompt and context.

        Behavior depends on mode:
        - Interactive: Calls Claude Code CLI directly
        - Non-interactive: Saves prompt file and outputs instructions
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
            return prompt_file

    def cmd_next(self, args):
        """
        Ask Claude: What should we do next?

        This is the KEY to RHSI - Claude analyzes the project state and suggests
        the next action. Over time, these suggestions become smarter as Palace
        learns from history.
        """
        context = self.gather_context()

        prompt = """Analyze this project and suggest what to do next.

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

        result = self.invoke_claude(prompt, context)

        if not is_interactive():
            # Non-interactive mode: output context for Claude
            print("🏛️  Palace - Invoking Claude for next step analysis...")
            print()
            print(f"📝 Context prepared at: {result}")
            print()
            print("Now invoking Claude to analyze and suggest next steps...")
            print()
            print("─" * 60)
            print()

            # Show the context to user
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
            print("🤖 CLAUDE: Please read the prompt file above and provide your analysis.")
            print()

        self.log_action("next", {"context_file": str(result) if not is_interactive() else "claude_cli"})

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
        """Install Palace commands into Claude Code"""

        print("🏛️  Palace - Installing to Claude Code")
        print()

        claude_code_dir = Path.home() / ".claude-code" / "commands"

        if not claude_code_dir.exists():
            claude_code_dir = Path.home() / ".config" / "claude-code" / "commands"

        if not claude_code_dir.exists():
            print("❌ Could not find Claude Code commands directory")
            return

        claude_code_dir.mkdir(parents=True, exist_ok=True)

        # Get absolute path to palace.py
        palace_path = Path(__file__).resolve()

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
            filepath = claude_code_dir / filename
            with open(filepath, 'w') as f:
                f.write(content)
            installed.append(filename.replace('.md', ''))

        print("✅ Installed Palace commands to Claude Code:")
        for cmd in installed:
            print(f"   • /{cmd}")

        print()
        print("🎉 Palace is now integrated with Claude Code!")
        print()
        print("These commands invoke Palace, which prepares context and prompts")
        print("for YOU (Claude) to analyze and execute using your full capabilities.")
        print()
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
    subparsers.add_parser('next', help='Ask Claude what to do next (RHSI core)')

    parser_new = subparsers.add_parser('new', help='Ask Claude to create a new project')
    parser_new.add_argument('name', nargs='?', help='Project name')

    subparsers.add_parser('scaffold', help='Ask Claude to scaffold the project')
    subparsers.add_parser('test', help='Ask Claude to run tests')

    # Utility commands
    subparsers.add_parser('install', help='Install Palace commands to Claude Code')
    subparsers.add_parser('init', help='Initialize Palace in current directory')
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
        'permissions': palace.cmd_permissions,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
