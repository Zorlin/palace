"""
MCP server integration tests

Tests for:
- handle_permission tool functionality
- Permission logging
- MCP availability detection
"""

import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace, MCP_AVAILABLE


class TestMCPAvailability:
    """Test MCP server availability detection"""

    def test_mcp_available_flag_exists(self):
        """MCP_AVAILABLE flag should be defined"""
        # Just verify it's a boolean
        assert isinstance(MCP_AVAILABLE, bool)

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not installed")
    def test_mcp_module_imports_when_available(self):
        """When MCP is available, FastMCP should be importable"""
        from mcp.server.fastmcp import FastMCP
        assert FastMCP is not None


class TestPermissionHandling:
    """Test permission handling functionality"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_permission_logging(self, temp_palace):
        """Permission requests should be logged to history"""
        request_data = {
            "tool_name": "Write",
            "input": {"file_path": "/test/file.py"},
            "tool_use_id": "test-123"
        }

        temp_palace.log_action("permission_request", {"request": request_data})

        history_file = temp_palace.palace_dir / "history.jsonl"
        assert history_file.exists()

        content = history_file.read_text()
        entry = json.loads(content.strip())

        assert entry["action"] == "permission_request"
        assert entry["details"]["request"]["tool_name"] == "Write"
        assert entry["details"]["request"]["tool_use_id"] == "test-123"

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not installed")
    def test_handle_permission_returns_approval(self, tmp_path):
        """handle_permission should return approval dict"""
        os.chdir(tmp_path)

        # Import and test the actual MCP tool
        from palace import handle_permission

        result = handle_permission(
            tool_name="Read",
            input={"file_path": "/test/file.py"},
            tool_use_id="test-456"
        )

        assert result["approved"] is True
        assert isinstance(result, dict)

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not installed")
    def test_handle_permission_logs_action(self, tmp_path):
        """handle_permission should log the request to history"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        from palace import handle_permission

        handle_permission(
            tool_name="Bash",
            input={"command": "ls -la"},
            tool_use_id="test-789"
        )

        history_file = palace.palace_dir / "history.jsonl"
        assert history_file.exists()

        # Find the permission_request entry
        content = history_file.read_text().strip().split('\n')
        found = False
        for line in content:
            entry = json.loads(line)
            if entry["action"] == "permission_request":
                assert entry["details"]["request"]["tool_name"] == "Bash"
                assert entry["details"]["request"]["tool_use_id"] == "test-789"
                found = True
                break

        assert found, "Permission request not found in history"

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not installed")
    def test_handle_permission_with_kwargs(self, tmp_path):
        """handle_permission should handle extra kwargs"""
        os.chdir(tmp_path)

        from palace import handle_permission

        result = handle_permission(
            tool_name="Edit",
            input={"file_path": "/test/file.py", "old_string": "foo"},
            tool_use_id="test-abc",
            extra_param="should be handled"
        )

        assert result["approved"] is True

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not installed")
    def test_handle_permission_with_none_input(self, tmp_path):
        """handle_permission should handle None input"""
        os.chdir(tmp_path)

        from palace import handle_permission

        result = handle_permission(
            tool_name="Read",
            input=None,
            tool_use_id="test-def"
        )

        assert result["approved"] is True


class TestMCPServerSetup:
    """Test MCP server configuration"""

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not installed")
    def test_mcp_instance_exists(self):
        """MCP server instance should be created when available"""
        from palace import mcp
        assert mcp is not None
        assert mcp.name == "Palace"

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not installed")
    def test_handle_permission_is_registered(self):
        """handle_permission should be registered as an MCP tool"""
        from palace import mcp, handle_permission

        # The function should be decorated and available
        assert callable(handle_permission)


class TestPermissionRequestStructure:
    """Test permission request data handling"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_various_tool_types(self, temp_palace):
        """Permission logging works for various tool types"""
        tools = [
            {"tool_name": "Read", "input": {"file_path": "/a.txt"}},
            {"tool_name": "Write", "input": {"file_path": "/b.txt", "content": "test"}},
            {"tool_name": "Edit", "input": {"file_path": "/c.txt", "old_string": "a", "new_string": "b"}},
            {"tool_name": "Bash", "input": {"command": "echo hi"}},
            {"tool_name": "Glob", "input": {"pattern": "**/*.py"}},
            {"tool_name": "Grep", "input": {"pattern": "TODO"}},
        ]

        for tool in tools:
            temp_palace.log_action("permission_request", {"request": tool})

        history_file = temp_palace.palace_dir / "history.jsonl"
        lines = history_file.read_text().strip().split('\n')

        assert len(lines) == 6

        # Verify each tool was logged
        logged_tools = [json.loads(l)["details"]["request"]["tool_name"] for l in lines]
        assert "Read" in logged_tools
        assert "Write" in logged_tools
        assert "Edit" in logged_tools
        assert "Bash" in logged_tools
        assert "Glob" in logged_tools
        assert "Grep" in logged_tools


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not installed")
class TestSafetyAssessment:
    """Test Haiku-based safety assessment"""

    @pytest.fixture
    def temp_palace_with_skill(self, tmp_path):
        """Create Palace with command-safety skill"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Create skills directory and command-safety skill
        skills_dir = palace.palace_dir / "skills"
        skills_dir.mkdir(exist_ok=True)
        skill_file = skills_dir / "command-safety.md"
        skill_file.write_text("""# Test Safety Skill
Respond with JSON: {"approved": true, "reason": "test"}""")

        yield palace

    def test_safety_assessment_creates_default_skill(self, tmp_path):
        """Without skill file, should create default and use it"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        from palace import _assess_permission_safety

        skill_path = palace.palace_dir / "skills" / "command-safety.md"
        assert not skill_path.exists()

        # Mock anthropic to avoid real API call
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"approved": true, "reason": "Safe operation"}')]

        with patch('anthropic.Anthropic') as mock_client:
            mock_client.return_value.messages.create.return_value = mock_response
            result = _assess_permission_safety("Read", {"file_path": "/test.py"})

        # Skill file should now exist
        assert skill_path.exists()
        assert "THINK TWICE" in skill_path.read_text()
        assert result["approved"] is True

    def test_safety_assessment_with_skill(self, temp_palace_with_skill):
        """With skill file, should call Haiku (mocked)"""
        from palace import _assess_permission_safety

        # Mock anthropic client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"approved": true, "reason": "Safe read operation"}')]

        with patch('anthropic.Anthropic') as mock_client:
            mock_client.return_value.messages.create.return_value = mock_response

            result = _assess_permission_safety("Read", {"file_path": "/test.py"})

            assert result["approved"] is True
            assert "Safe read operation" in result["reason"]

    def test_safety_assessment_deny(self, temp_palace_with_skill):
        """Should handle denial responses"""
        from palace import _assess_permission_safety

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"approved": false, "reason": "Dangerous operation"}')]

        with patch('anthropic.Anthropic') as mock_client:
            mock_client.return_value.messages.create.return_value = mock_response

            result = _assess_permission_safety("Bash", {"command": "rm -rf /"})

            assert result["approved"] is False
            assert "Dangerous" in result["reason"]

    def test_safety_assessment_error_fallback(self, temp_palace_with_skill):
        """On API error, should approve with error explanation"""
        from palace import _assess_permission_safety

        with patch('anthropic.Anthropic') as mock_client:
            mock_client.return_value.messages.create.side_effect = Exception("API error")

            result = _assess_permission_safety("Read", {"file_path": "/test.py"})

            assert result["approved"] is True
            assert "error" in result["reason"].lower()

    def test_permission_decision_logged(self, tmp_path):
        """Permission decisions should be logged"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        from palace import handle_permission

        # Mock the safety assessment
        with patch('palace._assess_permission_safety') as mock_assess:
            mock_assess.return_value = {"approved": True, "reason": "Test approval"}

            handle_permission(
                tool_name="Read",
                input={"file_path": "/test.py"},
                tool_use_id="test-123"
            )

        # Check both logs exist
        history_file = palace.palace_dir / "history.jsonl"
        lines = history_file.read_text().strip().split('\n')

        actions = [json.loads(l)["action"] for l in lines]
        assert "permission_request" in actions
        assert "permission_decision" in actions
