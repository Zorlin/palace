"""
Prompt building and Claude invocation tests

Tests for:
- Prompt construction
- Context injection
- File writing
"""

import pytest
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace


class TestPromptBuilding:
    """Test prompt construction with context"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_build_prompt_includes_task(self, temp_palace):
        """build_prompt includes the task prompt"""
        task = "This is a test task"
        prompt = temp_palace.build_prompt(task)

        assert task in prompt

    def test_build_prompt_includes_context(self, temp_palace):
        """build_prompt includes project context"""
        task = "Test task"
        context = {"test_key": "test_value"}
        prompt = temp_palace.build_prompt(task, context)

        assert "test_key" in prompt
        assert "test_value" in prompt

    def test_build_prompt_has_structure(self, temp_palace):
        """build_prompt creates structured prompt"""
        prompt = temp_palace.build_prompt("Task")

        assert "# Palace Request" in prompt
        assert "## Project Context" in prompt
        assert "## Instructions" in prompt

    def test_build_prompt_context_is_json(self, temp_palace):
        """build_prompt includes context as valid JSON"""
        task = "Test"
        context = {"key": "value", "number": 123}
        prompt = temp_palace.build_prompt(task, context)

        # Extract the JSON block
        assert "```json" in prompt
        assert "```" in prompt

        # Verify it's valid JSON in the prompt
        start = prompt.find("```json\n") + 8
        end = prompt.find("\n```\n", start)
        json_str = prompt[start:end]
        parsed = json.loads(json_str)

        assert parsed["key"] == "value"
        assert parsed["number"] == 123

    def test_build_prompt_mentions_palace(self, temp_palace):
        """build_prompt identifies Palace in instructions"""
        prompt = temp_palace.build_prompt("Task")

        assert "Palace" in prompt
        assert "self-improving" in prompt.lower()


class TestPromptFileWriting:
    """Test prompt file creation and management"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_invoke_claude_creates_prompt_file(self, temp_palace):
        """invoke_claude creates current_prompt.md file"""
        prompt_file = temp_palace.palace_dir / "current_prompt.md"
        assert not prompt_file.exists()

        # Mock non-interactive mode
        with patch('palace.is_interactive', return_value=False):
            temp_palace.invoke_claude("Test task")

        assert prompt_file.exists()

    def test_prompt_file_has_content(self, temp_palace):
        """Prompt file contains the expected content"""
        task = "Analyze the project"

        with patch('palace.is_interactive', return_value=False):
            temp_palace.invoke_claude(task)

        prompt_file = temp_palace.palace_dir / "current_prompt.md"
        content = prompt_file.read_text()

        assert task in content
        assert "# Palace Request" in content

    def test_prompt_file_overwritten_on_new_invoke(self, temp_palace):
        """New invocations overwrite the prompt file"""
        with patch('palace.is_interactive', return_value=False):
            temp_palace.invoke_claude("First task")
            temp_palace.invoke_claude("Second task")

        prompt_file = temp_palace.palace_dir / "current_prompt.md"
        content = prompt_file.read_text()

        assert "Second task" in content
        assert "First task" not in content


# Import patch for mocking
from unittest.mock import patch
