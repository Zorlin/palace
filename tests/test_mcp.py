"""
Tests for Palace MCP server functionality

Tests MCP tool registration, permission handling, and integration.
"""

import pytest
import json
from pathlib import Path
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path to import palace
sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace


class TestMCPPermissionHandling:
    """Test MCP permission handler tool"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_permission_request_logs_to_history(self, temp_palace):
        """Permission requests are logged"""
        temp_palace.log_action("permission_request", {
            "tool_name": "Bash",
            "input": {"command": "ls"}
        })

        history_file = temp_palace.palace_dir / "history.jsonl"
        assert history_file.exists()

        lines = history_file.read_text().strip().split('\n')
        entry = json.loads(lines[-1])

        assert entry["action"] == "permission_request"
        assert entry["details"]["tool_name"] == "Bash"


class TestSafetySkillSystem:
    """Test command safety skill loading"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance with safety skill"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_safety_skill_directory_exists(self, temp_palace):
        """Skills directory can be created"""
        skills_dir = temp_palace.palace_dir / "skills"
        skills_dir.mkdir(exist_ok=True)
        assert skills_dir.exists()

    def test_custom_safety_skill_can_be_written(self, temp_palace):
        """Custom safety skills can be created"""
        skill_file = temp_palace.palace_dir / "skills" / "command-safety.md"
        skill_file.parent.mkdir(exist_ok=True)
        skill_file.write_text("# Custom Safety\nTest content")

        assert skill_file.exists()
        content = skill_file.read_text()
        assert "Custom Safety" in content


class TestPermissionLogging:
    """Test permission request and decision logging"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_log_permission_decision(self, temp_palace):
        """Permission decisions are logged"""
        temp_palace.log_action("permission_decision", {
            "tool_name": "Bash",
            "approved": True,
            "reason": "Safe operation"
        })

        history_file = temp_palace.palace_dir / "history.jsonl"
        lines = history_file.read_text().strip().split('\n')
        entry = json.loads(lines[-1])

        assert entry["action"] == "permission_decision"
        assert entry["details"]["approved"] is True

    def test_permission_history_tracking(self, temp_palace):
        """Track permission history over time"""
        for i in range(5):
            temp_palace.log_action("permission_request", {
                "tool_name": "Bash",
                "command": f"test_{i}"
            })

        context = temp_palace.gather_context()
        assert "recent_history" in context

        permission_requests = [
            h for h in context["recent_history"]
            if h["action"] == "permission_request"
        ]
        assert len(permission_requests) == 5
