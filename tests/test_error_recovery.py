"""
Tests for Palace error recovery functionality

Tests retry logic, backoff, and graceful degradation.
"""

import pytest
import json
import time
from pathlib import Path
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path to import palace
sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace


class TestRetryLogic:
    """Test retry with exponential backoff"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_invoke_with_retry_success_first_try(self, temp_palace):
        """Successful invocation on first try doesn't retry"""
        with patch.object(temp_palace, 'invoke_claude_cli') as mock_invoke:
            mock_invoke.return_value = (0, [])

            exit_code, actions = temp_palace.invoke_with_retry("test prompt")

            assert exit_code == 0
            assert mock_invoke.call_count == 1

    def test_invoke_with_retry_transient_failure(self, temp_palace):
        """Transient failure triggers retry"""
        with patch.object(temp_palace, 'invoke_claude_cli') as mock_invoke:
            # Fail twice, then succeed
            mock_invoke.side_effect = [
                (1, None),  # First attempt fails
                (1, None),  # Second attempt fails
                (0, [])     # Third attempt succeeds
            ]

            with patch('time.sleep'):  # Mock sleep to speed up test
                exit_code, actions = temp_palace.invoke_with_retry("test prompt", max_retries=3)

            assert exit_code == 0
            assert mock_invoke.call_count == 3

    def test_invoke_with_retry_all_failures(self, temp_palace):
        """All retries exhausted returns failure"""
        with patch.object(temp_palace, 'invoke_claude_cli') as mock_invoke:
            mock_invoke.return_value = (1, None)

            with patch('time.sleep'):
                exit_code, actions = temp_palace.invoke_with_retry("test prompt", max_retries=3)

            assert exit_code == 1
            assert actions is None
            assert mock_invoke.call_count == 3

    def test_invoke_with_retry_exponential_backoff(self, temp_palace):
        """Verify exponential backoff timing"""
        with patch.object(temp_palace, 'invoke_claude_cli') as mock_invoke:
            mock_invoke.return_value = (1, None)

            with patch('time.sleep') as mock_sleep:
                temp_palace.invoke_with_retry("test prompt", max_retries=4)

            # Check sleep calls for exponential pattern
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert len(sleep_calls) == 3  # 3 retries = 3 sleeps
            assert sleep_calls == [1, 2, 4]  # 2^0, 2^1, 2^2


class TestErrorClassification:
    """Test error type classification"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_is_transient_error_network(self, temp_palace):
        """Network errors are transient"""
        assert temp_palace.is_transient_error(1, "Connection timeout") is True
        assert temp_palace.is_transient_error(1, "Network error") is True

    def test_is_transient_error_rate_limit(self, temp_palace):
        """Rate limits are transient"""
        assert temp_palace.is_transient_error(429, "Rate limit exceeded") is True

    def test_is_permanent_error_permission(self, temp_palace):
        """Permission errors are permanent"""
        assert temp_palace.is_transient_error(403, "Permission denied") is False

    def test_is_permanent_error_user_interrupt(self, temp_palace):
        """User interrupts are permanent"""
        assert temp_palace.is_transient_error(130, "KeyboardInterrupt") is False


class TestSessionCheckpointing:
    """Test session checkpointing for recovery"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_checkpoint_session_before_iteration(self, temp_palace):
        """Checkpoint session state before iteration"""
        session_id = "pal-test01"
        state = {
            "iteration": 1,
            "current_prompt": "Test prompt",
            "pending_actions": []
        }

        temp_palace.checkpoint_session(session_id, state)

        # Verify checkpoint was saved
        loaded = temp_palace.load_session(session_id)
        assert loaded is not None
        assert loaded["iteration"] == 1
        assert loaded["current_prompt"] == "Test prompt"

    def test_checkpoint_includes_timestamp(self, temp_palace):
        """Checkpoint includes timestamp"""
        session_id = "pal-test02"
        state = {"iteration": 1}

        before = time.time()
        temp_palace.checkpoint_session(session_id, state)
        after = time.time()

        loaded = temp_palace.load_session(session_id)
        checkpoint_time = loaded.get("checkpoint_at", 0)

        assert before <= checkpoint_time <= after

    def test_restore_from_checkpoint(self, temp_palace):
        """Restore session from checkpoint after failure"""
        session_id = "pal-restore"
        state = {
            "iteration": 3,
            "pending_actions": [{"num": "1", "label": "Task"}],
            "current_prompt": "Continue from checkpoint"
        }

        temp_palace.checkpoint_session(session_id, state)

        # Simulate failure and restoration
        restored = temp_palace.load_session(session_id)
        assert restored["iteration"] == 3
        assert len(restored["pending_actions"]) == 1


class TestErrorLogging:
    """Test error logging for analysis"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_log_retry_attempt(self, temp_palace):
        """Log retry attempts to history"""
        temp_palace.log_retry_attempt(
            attempt=1,
            max_retries=3,
            error="Connection timeout",
            wait_time=2
        )

        # Verify logged to history
        history_file = temp_palace.palace_dir / "history.jsonl"
        assert history_file.exists()

        lines = history_file.read_text().strip().split('\n')
        entry = json.loads(lines[-1])

        assert entry["action"] == "retry_attempt"
        assert entry["details"]["attempt"] == 1
        assert entry["details"]["max_retries"] == 3

    def test_log_error_recovery_success(self, temp_palace):
        """Log successful error recovery"""
        temp_palace.log_error_recovery(
            error_type="network_timeout",
            recovered=True,
            attempts=2
        )

        history_file = temp_palace.palace_dir / "history.jsonl"
        lines = history_file.read_text().strip().split('\n')
        entry = json.loads(lines[-1])

        assert entry["action"] == "error_recovery"
        assert entry["details"]["recovered"] is True
        assert entry["details"]["attempts"] == 2

    def test_log_error_recovery_failure(self, temp_palace):
        """Log failed error recovery"""
        temp_palace.log_error_recovery(
            error_type="permission_denied",
            recovered=False,
            attempts=3
        )

        history_file = temp_palace.palace_dir / "history.jsonl"
        lines = history_file.read_text().strip().split('\n')
        entry = json.loads(lines[-1])

        assert entry["action"] == "error_recovery"
        assert entry["details"]["recovered"] is False


class TestGracefulDegradation:
    """Test graceful degradation on repeated failures"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_degradation_mode_detection(self, temp_palace):
        """Detect when degradation is needed"""
        # First failure: no degradation
        mode = temp_palace.get_degradation_mode(attempt=0)
        assert mode == "retry"

        # Second failure: disable streaming
        mode = temp_palace.get_degradation_mode(attempt=1)
        assert mode == "no-stream"

        # Third failure: prompt file only
        mode = temp_palace.get_degradation_mode(attempt=2)
        assert mode == "prompt-file"

        # Fourth failure: fatal
        mode = temp_palace.get_degradation_mode(attempt=3)
        assert mode == "fatal"
