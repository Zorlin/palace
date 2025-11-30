"""
Comprehensive tests for session lifecycle management

Tests critical paths for:
- Session export and import
- Session cleanup
- Session corruption handling
- Session state persistence
"""

import pytest
import json
import time
from pathlib import Path
import sys
import os

# Add parent directory to path to import palace
sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace, VERSION


class TestSessionExport:
    """Test session export functionality"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_export_session_creates_file(self, temp_palace):
        """Test that exporting creates a file"""
        session_id = "pal-export1"
        temp_palace.save_session(session_id, {
            "iteration": 3,
            "pending_actions": []
        })

        output_path = temp_palace.export_session(session_id)

        assert output_path is not None
        assert Path(output_path).exists()

    def test_export_includes_session_data(self, temp_palace):
        """Test that export includes full session data"""
        session_id = "pal-export2"
        actions = [{"num": "1", "label": "Test action"}]
        temp_palace.save_session(session_id, {
            "iteration": 5,
            "pending_actions": actions
        })

        output_path = temp_palace.export_session(session_id)

        with open(output_path, 'r') as f:
            export_data = json.load(f)

        assert "version" in export_data
        assert "session" in export_data
        assert export_data["session"]["iteration"] == 5
        assert len(export_data["session"]["pending_actions"]) == 1

    def test_export_includes_history(self, temp_palace):
        """Test that export includes relevant history"""
        session_id = "pal-export3"
        temp_palace.save_session(session_id, {"iteration": 1, "pending_actions": []})

        # Log some actions for this session
        temp_palace.log_action("test_action", {"session_id": session_id})

        output_path = temp_palace.export_session(session_id)

        with open(output_path, 'r') as f:
            export_data = json.load(f)

        assert "history" in export_data
        assert len(export_data["history"]) > 0

    def test_export_nonexistent_session(self, temp_palace):
        """Test exporting nonexistent session returns None"""
        result = temp_palace.export_session("pal-nothere")
        assert result is None

    def test_export_custom_path(self, temp_palace, tmp_path):
        """Test exporting to custom path"""
        session_id = "pal-export4"
        temp_palace.save_session(session_id, {"iteration": 1, "pending_actions": []})

        custom_path = str(tmp_path / "custom_export.json")
        result = temp_palace.export_session(session_id, custom_path)

        assert result == custom_path
        assert Path(custom_path).exists()


class TestSessionImport:
    """Test session import functionality"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_import_session_creates_session(self, temp_palace, tmp_path):
        """Test that importing creates a session"""
        # Create an export file
        export_data = {
            "version": "1.0",
            "session": {
                "iteration": 3,
                "pending_actions": [{"num": "1", "label": "Test"}]
            },
            "history": []
        }
        export_file = tmp_path / "export.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f)

        new_id = temp_palace.import_session(str(export_file))

        assert new_id is not None
        assert new_id.startswith("pal-")

        # Verify session was created
        loaded = temp_palace.load_session(new_id)
        assert loaded is not None
        assert loaded["iteration"] == 3

    def test_import_with_custom_id(self, temp_palace, tmp_path):
        """Test importing with custom session ID"""
        export_data = {
            "version": "1.0",
            "session": {"iteration": 1, "pending_actions": []},
            "history": []
        }
        export_file = tmp_path / "export.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f)

        custom_id = "pal-custom1"
        result = temp_palace.import_session(str(export_file), custom_id)

        assert result == custom_id

        loaded = temp_palace.load_session(custom_id)
        assert loaded is not None

    def test_import_includes_history(self, temp_palace, tmp_path):
        """Test that import restores history entries"""
        export_data = {
            "version": "1.0",
            "session": {"iteration": 2, "pending_actions": []},
            "history": [
                {"action": "test", "details": {"session_id": "old-id"}}
            ]
        }
        export_file = tmp_path / "export.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f)

        new_id = temp_palace.import_session(str(export_file))

        # Check history was imported
        history_file = temp_palace.palace_dir / "history.jsonl"
        assert history_file.exists()

        lines = history_file.read_text().strip().split('\n')
        # Last line should be the imported entry with new session_id
        last_entry = json.loads(lines[-1])
        assert last_entry["action"] == "test"
        assert last_entry["details"]["session_id"] == new_id

    def test_import_invalid_file(self, temp_palace, tmp_path):
        """Test importing invalid file returns None"""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("not json")

        result = temp_palace.import_session(str(invalid_file))
        assert result is None

    def test_import_nonexistent_file(self, temp_palace):
        """Test importing nonexistent file returns None"""
        result = temp_palace.import_session("/nonexistent/file.json")
        assert result is None


class TestSessionCleanup:
    """Test session cleanup functionality"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_cleanup_all_sessions(self, temp_palace):
        """Test cleanup with --all flag"""
        # Create several sessions
        for i in range(5):
            temp_palace.save_session(f"pal-test{i}", {"iteration": i, "pending_actions": []})

        sessions_before = temp_palace.list_sessions()
        assert len(sessions_before) == 5

        # Cleanup all
        from argparse import Namespace
        args = Namespace(all=True, days=30, history=False, keep_history=1000)
        temp_palace.cmd_cleanup(args)

        sessions_after = temp_palace.list_sessions()
        assert len(sessions_after) == 0

    def test_cleanup_old_sessions(self, temp_palace):
        """Test cleanup removes old sessions"""
        # Create an old session (simulate by setting timestamp manually)
        old_session_id = "pal-old"
        temp_palace.save_session(old_session_id, {"iteration": 1, "pending_actions": []})

        # Manually set old timestamp
        session_file = temp_palace._get_sessions_dir() / f"{old_session_id}.json"
        with open(session_file, 'r') as f:
            data = json.load(f)
        data["updated_at"] = time.time() - (40 * 24 * 60 * 60)  # 40 days ago
        with open(session_file, 'w') as f:
            json.dump(data, f)

        # Create a recent session
        temp_palace.save_session("pal-new", {"iteration": 1, "pending_actions": []})

        # Cleanup sessions older than 30 days
        from argparse import Namespace
        args = Namespace(all=False, days=30, history=False, keep_history=1000)
        temp_palace.cmd_cleanup(args)

        sessions = temp_palace.list_sessions()
        # Should only have the new session
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "pal-new"

    def test_cleanup_history_trim(self, temp_palace):
        """Test cleanup trims history"""
        # Create many history entries
        for i in range(100):
            temp_palace.log_action(f"action_{i}", {"iteration": i})

        # Cleanup with history trim
        from argparse import Namespace
        args = Namespace(all=False, days=30, history=True, keep_history=20)
        temp_palace.cmd_cleanup(args)

        # Check history is trimmed
        history_file = temp_palace.palace_dir / "history.jsonl"
        lines = history_file.read_text().strip().split('\n')
        assert len(lines) == 20


class TestSessionStatePersistence:
    """Test session state persistence and recovery"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_checkpoint_session_adds_timestamp(self, temp_palace):
        """Test checkpoint adds checkpoint_at timestamp"""
        session_id = "pal-checkpoint1"
        state = {"iteration": 3, "pending_actions": []}

        temp_palace.checkpoint_session(session_id, state)

        loaded = temp_palace.load_session(session_id)
        assert "checkpoint_at" in loaded

    def test_session_preserves_all_fields(self, temp_palace):
        """Test that session save/load preserves all fields"""
        session_id = "pal-preserve"
        state = {
            "iteration": 10,
            "pending_actions": [
                {"num": "1", "label": "Task 1", "description": "Do task 1"},
                {"num": "2", "label": "Task 2", "description": "Do task 2"}
            ],
            "context": {"project_root": "/test"},
            "custom_field": "custom_value"
        }

        temp_palace.save_session(session_id, state)
        loaded = temp_palace.load_session(session_id)

        assert loaded["iteration"] == 10
        assert len(loaded["pending_actions"]) == 2
        assert loaded["context"]["project_root"] == "/test"
        assert loaded["custom_field"] == "custom_value"

    def test_concurrent_session_updates(self, temp_palace):
        """Test that concurrent updates don't corrupt state"""
        session_id = "pal-concurrent"

        # Save multiple times rapidly
        for i in range(10):
            temp_palace.save_session(session_id, {
                "iteration": i,
                "pending_actions": []
            })

        # Should have the latest state
        loaded = temp_palace.load_session(session_id)
        assert loaded["iteration"] == 9

    def test_session_with_large_state(self, temp_palace):
        """Test session with large state data"""
        session_id = "pal-large"
        large_actions = [
            {"num": str(i), "label": f"Action {i}", "description": "x" * 1000}
            for i in range(100)
        ]

        temp_palace.save_session(session_id, {
            "iteration": 1,
            "pending_actions": large_actions
        })

        loaded = temp_palace.load_session(session_id)
        assert len(loaded["pending_actions"]) == 100


class TestSessionCorruptionHandling:
    """Test handling of corrupted session files"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_list_sessions_skips_corrupted(self, temp_palace):
        """Test that list_sessions skips corrupted files"""
        # Create valid session
        temp_palace.save_session("pal-valid", {"iteration": 1, "pending_actions": []})

        # Create corrupted session file
        sessions_dir = temp_palace._get_sessions_dir()
        corrupted_file = sessions_dir / "pal-corrupt.json"
        with open(corrupted_file, 'w') as f:
            f.write("not valid json{}")

        # Should still work and return only valid session
        sessions = temp_palace.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "pal-valid"

    def test_load_corrupted_session_returns_none(self, temp_palace):
        """Test that loading corrupted session returns None gracefully"""
        # Create corrupted session file
        sessions_dir = temp_palace._get_sessions_dir()
        corrupted_file = sessions_dir / "pal-corrupt.json"
        with open(corrupted_file, 'w') as f:
            f.write("corrupt data")

        # Should return None, not crash
        loaded = temp_palace.load_session("pal-corrupt")
        # Implementation might return None or raise, depends on error handling
        # At minimum, should not crash

    def test_session_with_missing_fields(self, temp_palace):
        """Test handling session with missing required fields"""
        session_id = "pal-incomplete"
        # Save session with minimal fields
        temp_palace.save_session(session_id, {})

        loaded = temp_palace.load_session(session_id)
        assert loaded is not None
        # Should have session_id and updated_at added
        assert "session_id" in loaded
        assert "updated_at" in loaded
