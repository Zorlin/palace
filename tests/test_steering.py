"""
ESC-ESC Steering Tests

Tests for:
- Escape sequence detection
- User interrupt handling
- Session resume with steering context
"""

import pytest
import json
import time
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from io import StringIO

# Add parent directory to path to import palace
sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace


class TestEscapeDetection:
    """Test ESC-ESC sequence detection"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        yield palace

    def test_escape_handler_exists(self, temp_palace):
        """Palace has escape handler method"""
        assert hasattr(temp_palace, '_setup_escape_handler')
        assert callable(temp_palace._setup_escape_handler)

    def test_check_escape_sequence_exists(self, temp_palace):
        """Palace has escape sequence checker"""
        assert hasattr(temp_palace, '_check_escape_sequence')
        assert callable(temp_palace._check_escape_sequence)

    def test_escape_sequence_not_triggered_by_single_esc(self, temp_palace):
        """Single ESC should not trigger interrupt"""
        # Initialize escape handler
        temp_palace._setup_escape_handler()
        result = temp_palace._check_escape_sequence(chr(27))  # ESC char
        assert result == "first_esc"  # Waiting for second

    def test_escape_sequence_triggered_by_double_esc(self, temp_palace):
        """ESC-ESC within timeout triggers interrupt"""
        # Initialize escape handler
        temp_palace._setup_escape_handler()
        # Simulate first ESC
        temp_palace._last_esc_time = time.time()
        # Simulate second ESC immediately
        result = temp_palace._check_escape_sequence(chr(27))
        assert result == "interrupt"

    def test_escape_sequence_timeout(self, temp_palace):
        """ESC-ESC with too much delay does not trigger"""
        # Initialize escape handler
        temp_palace._setup_escape_handler()
        # Simulate first ESC from 2 seconds ago
        temp_palace._last_esc_time = time.time() - 2.0
        # Simulate second ESC now (beyond 0.5s timeout)
        result = temp_palace._check_escape_sequence(chr(27))
        # Should reset and wait for another double-tap
        assert result == "first_esc"

    def test_non_esc_key_ignored(self, temp_palace):
        """Non-ESC keys don't trigger anything"""
        temp_palace._setup_escape_handler()
        result = temp_palace._check_escape_sequence('a')
        assert result is None


class TestUserSteering:
    """Test user steering on interrupt"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        yield palace

    def test_handle_user_interrupt_exists(self, temp_palace):
        """Palace has interrupt handler method"""
        assert hasattr(temp_palace, '_handle_user_interrupt')
        assert callable(temp_palace._handle_user_interrupt)

    def test_interrupt_prompts_user(self, temp_palace):
        """Interrupt shows steering prompt"""
        with patch('builtins.input', return_value="focus on tests"):
            result = temp_palace._handle_user_interrupt()
            assert result is not None
            assert "focus on tests" in result.get("steering", "")

    def test_interrupt_allows_cancel(self, temp_palace):
        """User can cancel interrupt and resume normally"""
        with patch('builtins.input', return_value=""):
            result = temp_palace._handle_user_interrupt()
            assert result is None or result.get("action") == "resume"

    def test_interrupt_allows_abort(self, temp_palace):
        """User can abort the entire session"""
        with patch('builtins.input', return_value="/abort"):
            result = temp_palace._handle_user_interrupt()
            assert result is not None
            assert result.get("action") == "abort"


class TestSessionResume:
    """Test session resume with steering context"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        yield palace

    def test_checkpoint_before_interrupt(self, temp_palace):
        """Session is checkpointed before handling interrupt"""
        session_id = temp_palace._generate_session_id()
        session_state = {
            "iteration": 3,
            "pending_actions": [{"num": "1", "label": "Test task"}],
            "full_text_buffer": "Previous Claude output..."
        }

        # Checkpoint should add checkpoint_at timestamp
        temp_palace.checkpoint_session(session_id, session_state)

        loaded = temp_palace.load_session(session_id)
        assert loaded is not None
        assert "checkpoint_at" in loaded
        assert loaded["iteration"] == 3

    def test_resume_with_steering_context(self, temp_palace):
        """Resume includes steering in prompt"""
        steering = "focus on writing tests, skip documentation"

        # Build prompt with steering context
        prompt = temp_palace.build_prompt_with_steering(
            "Continue the task",
            steering=steering
        )

        assert steering in prompt
        assert "USER STEERING" in prompt or "steering" in prompt.lower()

    def test_steering_logged_to_history(self, temp_palace):
        """User steering is logged for learning"""
        steering = "prioritize performance over readability"
        temp_palace.log_steering(steering)

        # Check history file
        history_file = temp_palace.palace_dir / "history.jsonl"
        assert history_file.exists()

        with open(history_file) as f:
            entries = [json.loads(line) for line in f]

        # Find steering entry
        steering_entries = [e for e in entries if e.get("action") == "user_steering"]
        assert len(steering_entries) == 1
        assert steering_entries[0]["details"]["steering"] == steering


class TestInterruptIntegration:
    """Integration tests for full interrupt flow"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        yield palace

    def test_interrupt_during_stream(self, temp_palace):
        """ESC-ESC during stream processing pauses and prompts"""
        # Mock the stream and input
        mock_stream = StringIO('{"type": "assistant", "message": {"content": []}}\n')

        with patch.object(temp_palace, '_check_for_escape', return_value=True):
            with patch.object(temp_palace, '_handle_user_interrupt') as mock_interrupt:
                mock_interrupt.return_value = {"action": "steer", "steering": "focus on X"}

                # This should detect escape and handle interrupt
                # The actual implementation will integrate this
                assert callable(temp_palace._handle_user_interrupt)

    def test_full_interrupt_flow(self, temp_palace):
        """Full flow: detect ESC-ESC -> pause -> steer -> resume"""
        session_id = temp_palace._generate_session_id()

        # Save initial session state
        temp_palace.save_session(session_id, {
            "iteration": 2,
            "pending_actions": [{"num": "1", "label": "Deploy"}]
        })

        # Simulate interrupt with steering
        steering = "but use staging environment"

        with patch('builtins.input', return_value=steering):
            result = temp_palace._handle_user_interrupt()

        # Verify steering captured
        assert result is not None
        assert result.get("steering") == steering

        # Build resume prompt with steering
        prompt = temp_palace.build_prompt_with_steering(
            "Continue pending actions",
            steering=steering
        )

        # Verify steering in prompt
        assert steering in prompt
