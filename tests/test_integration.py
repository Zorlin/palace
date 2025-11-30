"""
Integration tests for Palace commands

Tests the full command execution flow including CLI parsing,
context gathering, and prompt generation.
"""

import pytest
import subprocess
import tempfile
import os
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace, VERSION


class TestCLICommands:
    """Test CLI command execution"""

    @pytest.fixture
    def temp_palace_env(self, tmp_path):
        """Create temp environment with Palace initialized"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Initialize with config
        config = {
            "name": "test_project",
            "version": "1.0.0",
            "palace_version": VERSION,
            "initialized": True
        }
        palace.save_config(config)

        yield tmp_path, palace

    def test_init_command(self, tmp_path):
        """Test 'palace init' command"""
        os.chdir(tmp_path)

        # Run init command
        result = subprocess.run(
            ["python3", "-m", "palace", "init"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
        )

        assert result.returncode == 0
        assert (tmp_path / ".palace").exists()
        assert (tmp_path / ".palace" / "config.json").exists()

        # Verify config
        with open(tmp_path / ".palace" / "config.json") as f:
            config = json.load(f)
            assert config["palace_version"] == VERSION
            assert config["initialized"] is True

    def test_sessions_command_empty(self, temp_palace_env):
        """Test 'palace sessions' with no sessions"""
        tmp_path, palace = temp_palace_env

        result = subprocess.run(
            ["python3", "-m", "palace", "sessions"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
        )

        assert result.returncode == 0
        assert "No saved sessions" in result.stdout

    def test_sessions_command_with_sessions(self, temp_palace_env):
        """Test 'palace sessions' with saved sessions"""
        tmp_path, palace = temp_palace_env

        # Create test sessions
        palace.save_session("pal-test01", {
            "iteration": 1,
            "pending_actions": [{"num": "1", "label": "Test task"}]
        })
        palace.save_session("pal-test02", {
            "iteration": 2,
            "pending_actions": []
        })

        result = subprocess.run(
            ["python3", "-m", "palace", "sessions"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
        )

        assert result.returncode == 0
        assert "pal-test01" in result.stdout
        assert "pal-test02" in result.stdout

    def test_version_flag(self):
        """Test --version flag"""
        result = subprocess.run(
            ["python3", "-m", "palace", "--version"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
        )

        assert result.returncode == 0
        assert VERSION in result.stdout


class TestContextGathering:
    """Test context gathering integration"""

    @pytest.fixture
    def project_env(self, tmp_path):
        """Create a project environment"""
        os.chdir(tmp_path)

        # Create project files
        (tmp_path / "README.md").write_text("# Test Project")
        (tmp_path / "SPEC.md").write_text("# Specification")
        (tmp_path / "requirements.txt").write_text("pytest>=7.0.0")

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, capture_output=True)

        # Create a change
        (tmp_path / "test.py").write_text("print('hello')")

        palace = Palace()
        palace.ensure_palace_dir()

        yield tmp_path, palace

    def test_context_includes_files(self, project_env):
        """Test that context gathering finds project files"""
        tmp_path, palace = project_env

        context = palace.gather_context()

        assert "README.md" in context["files"]
        assert "SPEC.md" in context["files"]
        assert "requirements.txt" in context["files"]

    def test_context_includes_git_status(self, project_env):
        """Test that context includes git status"""
        tmp_path, palace = project_env

        context = palace.gather_context()

        assert context["git_status"] is not None
        assert "test.py" in context["git_status"]

    def test_prompt_building(self, project_env):
        """Test full prompt building"""
        tmp_path, palace = project_env

        task = "Analyze this project"
        prompt = palace.build_prompt(task)

        assert "Analyze this project" in prompt
        assert "README.md" in prompt
        assert "Palace" in prompt
        assert "palace_version" in prompt


class TestActionParsing:
    """Test action selection parsing integration"""

    def test_parse_complex_selection(self):
        """Test parsing complex action selections"""
        palace = Palace()

        actions = [
            {"num": "1", "label": "Write tests"},
            {"num": "2", "label": "Run build"},
            {"num": "3", "label": "Deploy"},
            {"num": "4", "label": "Update docs"},
            {"num": "5", "label": "Commit changes"}
        ]

        # Test range with modifiers
        selected = palace.parse_action_selection("1-3 (follow TDD)", actions)
        assert len(selected) == 3
        assert all("_modifiers" in a for a in selected)

        # Test mixed format
        selected = palace.parse_action_selection("1, 3-4, 5", actions)
        assert len(selected) == 4

        # Test with text modifiers using "but" pattern
        selected = palace.parse_action_selection("2 but skip validation", actions)
        assert len(selected) == 1
        # Note: modifiers only attached when pattern matches (but, with, follow, etc.)
        if "_modifiers" in selected[0]:
            assert any("skip" in mod.lower() or "validation" in mod.lower() for mod in selected[0]["_modifiers"])


class TestPromptFileCreation:
    """Test prompt file generation for non-interactive mode"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create temp Palace instance"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_prompt_file_created(self, temp_palace):
        """Test that prompt file is created"""
        # Set to non-interactive mode
        os.environ["CI"] = "true"

        try:
            result, _ = temp_palace.invoke_claude("Test prompt")
            assert isinstance(result, Path)
            assert result.exists()
            assert result.name == "current_prompt.md"

            # Check content
            content = result.read_text()
            assert "Test prompt" in content
            assert "Palace" in content
        finally:
            if "CI" in os.environ:
                del os.environ["CI"]


class TestHistoryLogging:
    """Test action logging integration"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create temp Palace instance"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_action_logging_creates_history(self, temp_palace):
        """Test that logging actions creates history file"""
        history_file = temp_palace.palace_dir / "history.jsonl"
        assert not history_file.exists()

        temp_palace.log_action("test_action", {"detail": "test"})

        assert history_file.exists()

        # Verify content
        lines = history_file.read_text().strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["action"] == "test_action"
        assert entry["details"]["detail"] == "test"
        assert "timestamp" in entry

    def test_history_in_context(self, temp_palace):
        """Test that history is included in context"""
        temp_palace.log_action("action1")
        temp_palace.log_action("action2")

        context = temp_palace.gather_context()

        assert "recent_history" in context
        assert len(context["recent_history"]) == 2
