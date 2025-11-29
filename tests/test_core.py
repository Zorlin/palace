"""
Core Palace functionality tests

Tests for:
- Palace initialization
- Configuration management
- Directory structure
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path to import palace
sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace, VERSION


class TestPalaceInit:
    """Test Palace initialization and basic setup"""

    def test_palace_creates_instance(self):
        """Palace can be instantiated"""
        palace = Palace()
        assert palace is not None
        assert palace.project_root == Path.cwd()

    def test_palace_dir_path(self):
        """Palace sets correct .palace directory path"""
        palace = Palace()
        expected = Path.cwd() / ".palace"
        assert palace.palace_dir == expected

    def test_config_file_path(self):
        """Palace sets correct config file path"""
        palace = Palace()
        expected = Path.cwd() / ".palace" / "config.json"
        assert palace.config_file == expected


class TestConfigManagement:
    """Test configuration loading and saving"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        yield palace
        # Cleanup happens automatically with tmp_path

    def test_load_config_empty_when_no_file(self, temp_palace):
        """Loading config returns empty dict when no file exists"""
        config = temp_palace.load_config()
        assert config == {}

    def test_save_config_creates_directory(self, temp_palace):
        """Saving config creates .palace directory"""
        config = {"test": "value"}
        temp_palace.save_config(config)
        assert temp_palace.palace_dir.exists()
        assert temp_palace.palace_dir.is_dir()

    def test_save_and_load_config(self, temp_palace):
        """Config can be saved and loaded"""
        test_config = {
            "name": "test_project",
            "version": "1.0.0",
            "palace_version": VERSION
        }
        temp_palace.save_config(test_config)
        loaded_config = temp_palace.load_config()
        assert loaded_config == test_config

    def test_config_file_is_valid_json(self, temp_palace):
        """Saved config file is valid JSON"""
        config = {"key": "value"}
        temp_palace.save_config(config)

        with open(temp_palace.config_file, 'r') as f:
            loaded = json.load(f)

        assert loaded == config


class TestDirectoryManagement:
    """Test directory creation and management"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        yield palace

    def test_ensure_palace_dir_creates_dir(self, temp_palace):
        """ensure_palace_dir creates .palace directory"""
        assert not temp_palace.palace_dir.exists()
        temp_palace.ensure_palace_dir()
        assert temp_palace.palace_dir.exists()
        assert temp_palace.palace_dir.is_dir()

    def test_ensure_palace_dir_idempotent(self, temp_palace):
        """ensure_palace_dir can be called multiple times safely"""
        temp_palace.ensure_palace_dir()
        temp_palace.ensure_palace_dir()
        assert temp_palace.palace_dir.exists()


class TestContextGathering:
    """Test project context gathering"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance with some files"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Create some test files
        (tmp_path / "README.md").write_text("# Test Project")
        (tmp_path / "SPEC.md").write_text("# Specification")

        yield palace

    def test_gather_context_structure(self, temp_palace):
        """gather_context returns dict with expected keys"""
        context = temp_palace.gather_context()

        assert "project_root" in context
        assert "palace_version" in context
        assert "files" in context
        assert "git_status" in context
        assert "config" in context

    def test_gather_context_detects_files(self, temp_palace):
        """gather_context detects existing project files"""
        context = temp_palace.gather_context()

        assert "README.md" in context["files"]
        assert "SPEC.md" in context["files"]
        assert context["files"]["README.md"]["exists"] is True
        assert context["files"]["SPEC.md"]["exists"] is True

    def test_gather_context_includes_file_sizes(self, temp_palace):
        """gather_context includes file size information"""
        context = temp_palace.gather_context()

        assert "size" in context["files"]["README.md"]
        assert context["files"]["README.md"]["size"] > 0

    def test_gather_context_version(self, temp_palace):
        """gather_context includes Palace version"""
        context = temp_palace.gather_context()
        assert context["palace_version"] == VERSION


class TestHistoryLogging:
    """Test action history logging"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_log_action_creates_file(self, temp_palace):
        """log_action creates history file"""
        history_file = temp_palace.palace_dir / "history.jsonl"
        assert not history_file.exists()

        temp_palace.log_action("test_action")
        assert history_file.exists()

    def test_log_action_appends_entries(self, temp_palace):
        """log_action appends to history file"""
        temp_palace.log_action("action1")
        temp_palace.log_action("action2")

        history_file = temp_palace.palace_dir / "history.jsonl"
        lines = history_file.read_text().strip().split('\n')

        assert len(lines) == 2

    def test_log_action_valid_json(self, temp_palace):
        """log_action writes valid JSON entries"""
        temp_palace.log_action("test", {"detail": "value"})

        history_file = temp_palace.palace_dir / "history.jsonl"
        line = history_file.read_text().strip()
        entry = json.loads(line)

        assert entry["action"] == "test"
        assert entry["details"]["detail"] == "value"
        assert "timestamp" in entry

    def test_gather_context_includes_history(self, temp_palace):
        """gather_context includes recent history"""
        temp_palace.log_action("action1", {"detail": "test1"})
        temp_palace.log_action("action2", {"detail": "test2"})

        context = temp_palace.gather_context()

        assert "recent_history" in context
        assert len(context["recent_history"]) == 2
        assert context["recent_history"][0]["action"] == "action1"
        assert context["recent_history"][1]["action"] == "action2"
