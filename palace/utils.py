"""
Utility functions for Palace
"""

import sys
import os


def is_interactive() -> bool:
    """Detect if Palace is running interactively (can invoke Claude) vs from within Claude Code"""
    # Check if we're in a CI environment or being called from Claude Code
    if os.getenv("CI") or os.getenv("CLAUDE_CODE_SESSION"):
        return False
    # Check if stdin is a TTY (terminal)
    return sys.stdin.isatty()


def _load_credentials():
    """Load credentials from ~/.palace/credentials.env"""
    from pathlib import Path

    creds_file = Path.home() / ".palace" / "credentials.env"
    if creds_file.exists():
        for line in creds_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
