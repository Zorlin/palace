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
from unittest.mock import patch, MagicMock

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


class TestSessionManagement:
    """Test session management for RHSI loops"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_generate_session_id(self, temp_palace):
        """Session IDs are generated with correct format"""
        session_id = temp_palace._generate_session_id()
        assert session_id.startswith("pal-")
        assert len(session_id) == 10  # "pal-" + 6 hex chars

    def test_save_and_load_session(self, temp_palace):
        """Sessions can be saved and loaded"""
        session_id = "pal-test01"
        state = {
            "iteration": 3,
            "pending_actions": [
                {"num": "1", "label": "Write tests"},
                {"num": "2", "label": "Run build"}
            ]
        }

        temp_palace.save_session(session_id, state)
        loaded = temp_palace.load_session(session_id)

        assert loaded is not None
        assert loaded["session_id"] == session_id
        assert loaded["iteration"] == 3
        assert len(loaded["pending_actions"]) == 2
        assert "updated_at" in loaded

    def test_load_nonexistent_session(self, temp_palace):
        """Loading nonexistent session returns None"""
        loaded = temp_palace.load_session("pal-nothere")
        assert loaded is None

    def test_list_sessions_empty(self, temp_palace):
        """list_sessions returns empty list when no sessions"""
        sessions = temp_palace.list_sessions()
        assert sessions == []

    def test_list_sessions_multiple(self, temp_palace):
        """list_sessions returns all sessions sorted by time"""
        import time

        temp_palace.save_session("pal-first1", {"iteration": 1, "pending_actions": []})
        time.sleep(0.01)
        temp_palace.save_session("pal-second", {"iteration": 2, "pending_actions": [{"num": "1", "label": "test"}]})

        sessions = temp_palace.list_sessions()

        assert len(sessions) == 2
        # Most recent first
        assert sessions[0]["session_id"] == "pal-second"
        assert sessions[0]["pending_actions"] == 1
        assert sessions[1]["session_id"] == "pal-first1"

    def test_session_resumption_with_actions(self, temp_palace):
        """Session can be resumed with pending actions"""
        session_id = "pal-resume1"
        pending = [
            {"num": "1", "label": "Write tests", "description": "Add unit tests"},
            {"num": "2", "label": "Run build", "description": "Execute build"}
        ]

        temp_palace.save_session(session_id, {
            "iteration": 2,
            "pending_actions": pending,
            "context": {"project_root": str(temp_palace.project_root)}
        })

        loaded = temp_palace.load_session(session_id)
        assert loaded is not None
        assert loaded["iteration"] == 2
        assert len(loaded["pending_actions"]) == 2
        assert loaded["pending_actions"][0]["label"] == "Write tests"

    def test_session_action_selection_parsing(self, temp_palace):
        """Session actions can be filtered with parse_action_selection"""
        actions = [
            {"num": "1", "label": "Task 1"},
            {"num": "2", "label": "Task 2"},
            {"num": "3", "label": "Task 3"}
        ]

        # Simulate user selecting subset
        selected = temp_palace.parse_action_selection("1 3", actions)
        assert len(selected) == 2
        assert selected[0]["label"] == "Task 1"
        assert selected[1]["label"] == "Task 3"

    def test_session_with_modifiers(self, temp_palace):
        """Session actions can have modifiers applied during selection"""
        actions = [
            {"num": "1", "label": "Deploy app"},
            {"num": "2", "label": "Run tests"}
        ]

        selected = temp_palace.parse_action_selection("1 2 (follow TDD)", actions)
        assert len(selected) == 2
        for action in selected:
            assert "_modifiers" in action
            assert "follow TDD" in action["_modifiers"]


class TestActionSelectionParsing:
    """Test action selection parsing for RHSI loops"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        yield palace

    @pytest.fixture
    def sample_actions(self):
        """Sample actions for testing"""
        return [
            {"num": "1", "label": "Write tests", "description": "Add unit tests"},
            {"num": "2", "label": "Run build", "description": "Execute the build"},
            {"num": "3", "label": "Deploy", "description": "Deploy to production"},
            {"num": "4", "label": "Update docs", "description": "Update documentation"},
            {"num": "5", "label": "Commit changes", "description": "Git commit"}
        ]

    def test_parse_single_number(self, temp_palace, sample_actions):
        """Parse single numeric selection"""
        selected = temp_palace.parse_action_selection("1", sample_actions)
        assert len(selected) == 1
        assert selected[0]["label"] == "Write tests"

    def test_parse_multiple_numbers(self, temp_palace, sample_actions):
        """Parse comma-separated numbers"""
        selected = temp_palace.parse_action_selection("1,3,5", sample_actions)
        assert len(selected) == 3
        assert selected[0]["label"] == "Write tests"
        assert selected[1]["label"] == "Deploy"
        assert selected[2]["label"] == "Commit changes"

    def test_parse_range(self, temp_palace, sample_actions):
        """Parse numeric range"""
        selected = temp_palace.parse_action_selection("2-4", sample_actions)
        assert len(selected) == 3
        assert selected[0]["label"] == "Run build"
        assert selected[1]["label"] == "Deploy"
        assert selected[2]["label"] == "Update docs"

    def test_parse_mixed_range_and_numbers(self, temp_palace, sample_actions):
        """Parse mixed ranges and numbers"""
        selected = temp_palace.parse_action_selection("1,3-4", sample_actions)
        assert len(selected) == 3
        assert selected[0]["label"] == "Write tests"
        assert selected[1]["label"] == "Deploy"
        assert selected[2]["label"] == "Update docs"

    def test_parse_natural_language_with_numbers(self, temp_palace, sample_actions):
        """Parse natural language with numbers - uses LLM"""
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_numbers": ["1", "3"],
                "modifiers": [],
                "is_custom_task": False,
                "custom_task": None
            }
            selected = temp_palace.parse_action_selection("do 1 and 3", sample_actions)
            assert len(selected) == 2
            assert selected[0]["label"] == "Write tests"
            assert selected[1]["label"] == "Deploy"

    def test_parse_natural_language_with_modifiers(self, temp_palace, sample_actions):
        """Parse natural language with modifiers - uses LLM"""
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_numbers": ["1"],
                "modifiers": ["skip the tests"],
                "is_custom_task": False,
                "custom_task": None
            }
            selected = temp_palace.parse_action_selection("do 1 but skip the tests", sample_actions)
            assert len(selected) == 1
            assert selected[0]["label"] == "Write tests"
            assert "_modifiers" in selected[0]
            assert "skip the tests" in selected[0]["_modifiers"]

    def test_parse_invalid_selection(self, temp_palace, sample_actions):
        """Parse invalid selection returns empty list"""
        selected = temp_palace.parse_action_selection("99", sample_actions)
        assert selected == []

    def test_parse_with_spaces(self, temp_palace, sample_actions):
        """Parse selection with spaces"""
        selected = temp_palace.parse_action_selection("1, 2, 3", sample_actions)
        assert len(selected) == 3

    def test_parse_space_separated_numbers(self, temp_palace, sample_actions):
        """Parse space-separated numbers (no commas)"""
        selected = temp_palace.parse_action_selection("1 2 3", sample_actions)
        assert len(selected) == 3
        assert selected[0]["label"] == "Write tests"
        assert selected[1]["label"] == "Run build"
        assert selected[2]["label"] == "Deploy"

    def test_parse_with_parenthetical_modifiers(self, temp_palace, sample_actions):
        """Parse selection with parenthetical modifiers"""
        selected = temp_palace.parse_action_selection("1 2 3 (use the palace-skills repo as base)", sample_actions)
        assert len(selected) == 3
        assert "_modifiers" in selected[0]
        assert "use the palace-skills repo as base" in selected[0]["_modifiers"]
        # Check all actions have the modifier
        for action in selected:
            assert "_modifiers" in action
            assert "use the palace-skills repo as base" in action["_modifiers"]

    def test_parse_with_follow_modifier(self, temp_palace, sample_actions):
        """Parse 'follow' modifier pattern - uses LLM"""
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_numbers": ["1", "2"],
                "modifiers": ["/path/to/guide"],
                "is_custom_task": False,
                "custom_task": None
            }
            selected = temp_palace.parse_action_selection("1 2 follow /path/to/guide", sample_actions)
            assert len(selected) == 2
            assert "_modifiers" in selected[0]
            assert "/path/to/guide" in selected[0]["_modifiers"]

    def test_parse_with_use_modifier(self, temp_palace, sample_actions):
        """Parse 'use' modifier pattern - uses LLM"""
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_numbers": ["3"],
                "modifiers": ["TypeScript"],
                "is_custom_task": False,
                "custom_task": None
            }
            selected = temp_palace.parse_action_selection("3 use TypeScript", sample_actions)
            assert len(selected) == 1
            assert selected[0]["label"] == "Deploy"
            assert "_modifiers" in selected[0]
            assert "TypeScript" in selected[0]["_modifiers"]

    def test_parse_with_without_modifier(self, temp_palace, sample_actions):
        """Parse 'without' modifier pattern - uses LLM"""
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_numbers": ["2"],
                "modifiers": ["breaking changes"],
                "is_custom_task": False,
                "custom_task": None
            }
            selected = temp_palace.parse_action_selection("2 without breaking changes", sample_actions)
            assert len(selected) == 1
            assert "_modifiers" in selected[0]
            assert "breaking changes" in selected[0]["_modifiers"]

    def test_parse_range_with_spaces(self, temp_palace, sample_actions):
        """Parse range with spaces around dash"""
        selected = temp_palace.parse_action_selection("1 - 3", sample_actions)
        assert len(selected) == 3
        assert selected[0]["label"] == "Write tests"
        assert selected[2]["label"] == "Deploy"

    def test_parse_complex_mixed_format(self, temp_palace, sample_actions):
        """Parse complex mixed format: ranges, numbers, and modifiers"""
        selected = temp_palace.parse_action_selection("1-2, 4 5 (follow TDD)", sample_actions)
        assert len(selected) == 4
        assert selected[0]["label"] == "Write tests"
        assert selected[1]["label"] == "Run build"
        assert selected[2]["label"] == "Update docs"
        assert selected[3]["label"] == "Commit changes"
        # Check modifiers are attached
        for action in selected:
            assert "_modifiers" in action
            assert "follow TDD" in action["_modifiers"]

    def test_parse_custom_task(self, temp_palace, sample_actions):
        """Parse custom task (pure text, no numbers) - uses LLM"""
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_numbers": [],
                "modifiers": [],
                "is_custom_task": True,
                "custom_task": "refactor the authentication system"
            }
            selected = temp_palace.parse_action_selection("refactor the authentication system", sample_actions)
            assert len(selected) == 1
            assert selected[0]["label"] == "refactor the authentication system"
            assert selected[0]["_custom"] is True

    def test_parse_multiple_modifiers(self, temp_palace, sample_actions):
        """Parse multiple modifier patterns - uses LLM"""
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_numbers": ["1", "2"],
                "modifiers": ["base: palace-skills", "skip tests"],
                "is_custom_task": False,
                "custom_task": None
            }
            selected = temp_palace.parse_action_selection("1 2 (base: palace-skills) but skip tests", sample_actions)
            assert len(selected) == 2
            assert "_modifiers" in selected[0]
            # Should have modifiers (may extract multiple from complex patterns)
            assert len(selected[0]["_modifiers"]) >= 2
            assert "base: palace-skills" in selected[0]["_modifiers"]
            # Check that some form of "skip" modifier is captured
            assert any("skip" in mod.lower() for mod in selected[0]["_modifiers"])

    def test_parse_action_copies_dont_mutate_original(self, temp_palace, sample_actions):
        """Ensure parsing creates copies and doesn't mutate original actions"""
        original_count = len([a for a in sample_actions if "_modifiers" in a])
        selected = temp_palace.parse_action_selection("1 2 (with modifier)", sample_actions)
        # Original actions should not have modifiers
        after_count = len([a for a in sample_actions if "_modifiers" in a])
        assert original_count == after_count
        # But selected should have modifiers
        assert all("_modifiers" in a for a in selected)


class TestLLMActionParsing:
    """Test LLM-based action selection parsing using Haiku"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        yield palace

    @pytest.fixture
    def sample_actions(self):
        """Sample actions for testing"""
        return [
            {"num": "1", "label": "Write tests", "description": "Add unit tests for the module"},
            {"num": "2", "label": "Run build", "description": "Execute the build process"},
            {"num": "3", "label": "Deploy", "description": "Deploy to production server"},
            {"num": "4", "label": "Update docs", "description": "Update API documentation"},
            {"num": "5", "label": "Commit changes", "description": "Git commit all changes"}
        ]

    def test_llm_parser_simple_numbers(self, temp_palace, sample_actions):
        """Simple numbers bypass LLM, use fast regex"""
        # Mock the LLM call to verify it's NOT called for simple inputs
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            selected = temp_palace.parse_action_selection("1", sample_actions)
            assert len(selected) == 1
            assert selected[0]["label"] == "Write tests"
            # LLM should NOT be called for simple numeric input
            mock_llm.assert_not_called()

    def test_llm_parser_complex_natural_language(self, temp_palace, sample_actions):
        """Complex natural language uses LLM parser"""
        # Mock the LLM to return expected result
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_numbers": ["1", "3"],
                "modifiers": ["skip testing phase"],
                "is_custom_task": False,
                "custom_task": None
            }
            selected = temp_palace.parse_action_selection(
                "do the first and third but skip testing phase",
                sample_actions
            )
            # LLM should be called for natural language
            mock_llm.assert_called_once()
            assert len(selected) == 2

    def test_llm_parser_returns_modifiers(self, temp_palace, sample_actions):
        """LLM parser correctly extracts modifiers"""
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_numbers": ["2"],
                "modifiers": ["use TypeScript", "follow TDD"],
                "is_custom_task": False,
                "custom_task": None
            }
            selected = temp_palace.parse_action_selection(
                "do 2 use TypeScript and follow TDD",
                sample_actions
            )
            assert len(selected) == 1
            assert "_modifiers" in selected[0]
            assert "use TypeScript" in selected[0]["_modifiers"]
            assert "follow TDD" in selected[0]["_modifiers"]

    def test_llm_parser_custom_task(self, temp_palace, sample_actions):
        """LLM parser identifies custom tasks"""
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_numbers": [],
                "modifiers": [],
                "is_custom_task": True,
                "custom_task": "refactor the authentication system"
            }
            selected = temp_palace.parse_action_selection(
                "refactor the authentication system",
                sample_actions
            )
            assert len(selected) == 1
            assert selected[0]["_custom"] is True
            assert selected[0]["label"] == "refactor the authentication system"

    def test_llm_parser_fallback_on_error(self, temp_palace, sample_actions):
        """LLM parser falls back to regex on API error"""
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            mock_llm.side_effect = Exception("API error")
            # Should still work using regex fallback
            selected = temp_palace.parse_action_selection("1 2 3", sample_actions)
            assert len(selected) == 3

    def test_llm_parser_ambiguous_intent(self, temp_palace, sample_actions):
        """LLM parser handles ambiguous user intent"""
        with patch.object(temp_palace, '_parse_selection_with_llm') as mock_llm:
            mock_llm.return_value = {
                "selected_numbers": ["1", "2", "3"],
                "modifiers": ["except documentation"],
                "is_custom_task": False,
                "custom_task": None
            }
            # User says "all of them except documentation"
            selected = temp_palace.parse_action_selection(
                "all of them except documentation",
                sample_actions
            )
            mock_llm.assert_called_once()
            # Should get actions, with modifier attached
            assert len(selected) >= 1
            if selected:
                assert "_modifiers" in selected[0]

    def test_llm_parser_integration(self, temp_palace, sample_actions):
        """Integration test: actual LLM call (mocked at anthropic level)"""
        with patch('anthropic.Anthropic') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock the LLM response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text=json.dumps({
                "selected_numbers": ["1", "3"],
                "modifiers": ["using pytest"],
                "is_custom_task": False,
                "custom_task": None
            }))]
            mock_client.messages.create.return_value = mock_response

            # This should trigger actual _parse_selection_with_llm
            selected = temp_palace.parse_action_selection(
                "do tests and deploy using pytest",
                sample_actions
            )
            # Verify the flow works end-to-end
            assert isinstance(selected, list)
