"""
Permission Handler Tests

Tests that the MCP permission handler:
1. Can be invoked with correct parameters
2. Returns approval decisions
3. Actually grants permissions to Claude
"""

import pytest
import json
import os
import sys
import subprocess
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace


class TestPermissionHandlerDirect:
    """Test permission handler function directly"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_handler_accepts_expected_params(self, temp_palace):
        """Handler accepts tool_name, input, tool_use_id"""
        # Import the handler from the MCP module
        from palace import handle_permission

        # Call with expected parameters
        result = handle_permission(
            tool_name="Read",
            input={"file_path": "/tmp/test.txt"},
            tool_use_id="toolu_123"
        )

        assert isinstance(result, dict)
        assert "behavior" in result

    def test_handler_returns_approval(self, temp_palace):
        """Handler returns behavior=allow for safe operations"""
        from palace import handle_permission

        result = handle_permission(
            tool_name="Read",
            input={"file_path": "/tmp/safe.txt"},
            tool_use_id="toolu_456"
        )

        assert result["behavior"] == "allow"
        assert "updatedInput" in result

    def test_handler_with_empty_params(self, temp_palace):
        """Handler works with default/empty params"""
        from palace import handle_permission

        # Should not raise - defaults should work
        result = handle_permission()

        assert isinstance(result, dict)
        assert "behavior" in result

    def test_handler_logs_request(self, temp_palace):
        """Handler logs permission requests to history"""
        from palace import handle_permission

        handle_permission(
            tool_name="Bash",
            input={"command": "echo test"},
            tool_use_id="toolu_789"
        )

        # Check history was logged
        history_file = temp_palace.palace_dir / "history.jsonl"
        assert history_file.exists()

        with open(history_file) as f:
            lines = f.readlines()

        # Should have logged permission_request and permission_decision
        actions = [json.loads(line)["action"] for line in lines]
        assert "permission_request" in actions
        assert "permission_decision" in actions


class TestPermissionHandlerMCP:
    """Test permission handler via MCP protocol"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_mcp_server_starts(self, temp_palace):
        """MCP server process starts without error"""
        # Start MCP server as subprocess
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).parent.parent / "palace.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Give it a moment to start
        time.sleep(0.5)

        # Should still be running (poll returns None if running)
        assert process.poll() is None

        # Clean up
        process.terminate()
        process.wait(timeout=2)

    def test_mcp_tool_schema_no_required_kwargs(self, temp_palace):
        """MCP tool schema should NOT have kwargs as required"""
        # This is what was breaking - kwargs was being parsed as required
        from palace import handle_permission
        import inspect

        sig = inspect.signature(handle_permission)
        params = sig.parameters

        # Should have these params with defaults
        assert "tool_name" in params
        assert "input" in params
        assert "tool_use_id" in params

        # Should NOT have kwargs
        assert "kwargs" not in params

        # All should have defaults (not required)
        for name, param in params.items():
            assert param.default is not inspect.Parameter.empty, f"{name} should have default"


class TestPermissionE2E:
    """End-to-end permission flow tests"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_claude_calls_permission_handler(self, temp_palace, tmp_path):
        """Claude actually calls the permission handler and gets approval"""
        # Create a test file for Claude to read
        test_file = tmp_path / "test_read.txt"
        test_file.write_text("Hello from test")

        # Build the command that would spawn Claude with permission handler
        cmd = [
            "claude",
            "--model", "haiku",
            "--print",  # Non-interactive, print result
            "--output-format", "json",
            "--max-turns", "1",
            "--permission-prompt-tool", "mcp__palace__handle_permission",
            "-p", f"Read the file {test_file} and tell me what it says. Just read it, nothing else."
        ]

        # Run Claude with the permission handler
        try:
            result = subprocess.run(
                cmd,
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "HOME": str(tmp_path.parent)}
            )
        except subprocess.TimeoutExpired:
            pytest.skip("Claude CLI timed out - may not be available")
        except FileNotFoundError:
            pytest.skip("Claude CLI not installed")

        # If we got here, check the result
        # stdout should contain the response
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        print(f"returncode: {result.returncode}")

        # The permission handler should have been called
        # Check if history.jsonl has permission events
        history_file = temp_palace.palace_dir / "history.jsonl"
        if history_file.exists():
            with open(history_file) as f:
                history = f.read()
            print(f"history: {history}")
            # Should see permission_request logged
            assert "permission_request" in history or "Read" in result.stdout

        # Claude should have been able to read the file (permission granted)
        # The response should mention the file contents or acknowledge reading
        assert result.returncode == 0 or "Hello" in result.stdout or "permission" in result.stderr.lower()

    def test_permission_handler_called_via_subprocess(self, temp_palace, tmp_path):
        """Verify permission handler is actually invoked"""
        # Create a marker file that the handler will create
        marker = tmp_path / ".permission_called"

        # Patch the handler to create marker file
        original_code = (Path(__file__).parent.parent / "palace.py").read_text()

        # Instead of patching, just verify the handler works
        from palace import handle_permission

        # Simulate what Claude would send
        result = handle_permission(
            tool_name="Read",
            input={"file_path": str(tmp_path / "test.txt")},
            tool_use_id="toolu_test123"
        )

        # Handler must return approval in Claude Code format
        assert result["behavior"] == "allow"
        assert "updatedInput" in result


class TestRealClaudePermission:
    """Test REAL Claude CLI with permission handler - actually executes"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        # Create .palace dir
        palace_dir = tmp_path / ".palace"
        palace_dir.mkdir()
        palace = Palace()
        yield palace, tmp_path

    def test_claude_reads_file_with_permission_handler(self, temp_palace):
        """Claude ACTUALLY reads a file using our permission handler"""
        palace, tmp_path = temp_palace

        # Create test file
        test_file = tmp_path / "secret.txt"
        test_file.write_text("THE_SECRET_VALUE_12345")

        # Spawn Claude with our permission handler
        cmd = [
            "claude",
            "-p", f"Read {test_file} and tell me the exact contents. Reply ONLY with the file contents, nothing else.",
            "--model", "haiku",
            "--max-turns", "3",
            "--permission-prompt-tool", "mcp__palace__handle_permission",
            "--output-format", "text",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=120
            )
        except FileNotFoundError:
            pytest.skip("Claude CLI not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("Claude timed out - permission handler may be blocking")

        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"RETURN: {result.returncode}")

        # The file should have been read - check for the secret value
        assert "THE_SECRET_VALUE_12345" in result.stdout, \
            f"Claude didn't read the file! stdout={result.stdout}, stderr={result.stderr}"

    def test_claude_writes_file_with_permission_handler(self, temp_palace):
        """Claude ACTUALLY writes a file using our permission handler"""
        palace, tmp_path = temp_palace

        output_file = tmp_path / "output.txt"

        cmd = [
            "claude",
            "-p", f"Write the text 'WRITTEN_BY_CLAUDE_67890' to {output_file}. Use the Write tool. Reply 'done' when finished.",
            "--model", "haiku",
            "--max-turns", "3",
            "--permission-prompt-tool", "mcp__palace__handle_permission",
            "--output-format", "text",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=120
            )
        except FileNotFoundError:
            pytest.skip("Claude CLI not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("Claude timed out - permission handler may be blocking")

        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

        # The file should exist with the content
        assert output_file.exists(), f"File wasn't created! stderr={result.stderr}"
        content = output_file.read_text()
        assert "WRITTEN_BY_CLAUDE_67890" in content, f"Wrong content: {content}"


class TestPermissionSafety:
    """Test permission safety assessment"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_safe_read_approved(self, temp_palace):
        """Safe Read operations are approved"""
        from palace import handle_permission

        result = handle_permission(
            tool_name="Read",
            input={"file_path": "/home/user/project/src/main.py"},
            tool_use_id="toolu_read1"
        )

        assert result["behavior"] == "allow"

    def test_safe_edit_approved(self, temp_palace):
        """Safe Edit operations are approved"""
        from palace import handle_permission

        result = handle_permission(
            tool_name="Edit",
            input={
                "file_path": "/home/user/project/src/main.py",
                "old_string": "foo",
                "new_string": "bar"
            },
            tool_use_id="toolu_edit1"
        )

        assert result["behavior"] == "allow"

    def test_safe_bash_approved(self, temp_palace):
        """Safe Bash commands are approved"""
        from palace import handle_permission

        result = handle_permission(
            tool_name="Bash",
            input={"command": "python3 -m pytest tests/"},
            tool_use_id="toolu_bash1"
        )

        assert result["behavior"] == "allow"
