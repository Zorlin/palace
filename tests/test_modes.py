"""
Mode detection tests

Tests for:
- Interactive vs non-interactive mode detection
- Environment variable handling
- TTY detection
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import is_interactive


class TestModeDetection:
    """Test interactive vs non-interactive mode detection"""

    def test_is_interactive_with_ci_env(self):
        """is_interactive returns False when CI env var is set"""
        with patch.dict(os.environ, {"CI": "true"}):
            assert is_interactive() is False

    def test_is_interactive_with_claude_code_session(self):
        """is_interactive returns False when CLAUDE_CODE_SESSION is set"""
        with patch.dict(os.environ, {"CLAUDE_CODE_SESSION": "true"}):
            assert is_interactive() is False

    @patch('sys.stdin')
    def test_is_interactive_with_tty(self, mock_stdin):
        """is_interactive returns True when stdin is a TTY"""
        mock_stdin.isatty.return_value = True
        with patch.dict(os.environ, {}, clear=True):
            assert is_interactive() is True

    @patch('sys.stdin')
    def test_is_interactive_without_tty(self, mock_stdin):
        """is_interactive returns False when stdin is not a TTY"""
        mock_stdin.isatty.return_value = False
        with patch.dict(os.environ, {}, clear=True):
            assert is_interactive() is False

    def test_is_interactive_ci_overrides_tty(self):
        """CI env var overrides TTY detection"""
        with patch.dict(os.environ, {"CI": "true"}):
            with patch('sys.stdin') as mock_stdin:
                mock_stdin.isatty.return_value = True
                assert is_interactive() is False
