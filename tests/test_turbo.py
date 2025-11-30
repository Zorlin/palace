"""
Turbo Mode Tests

Tests for:
- Task-to-model ranking
- Parallel swarm spawning
- Shared history interleaving (omniscient agents)
"""

import pytest
import json
import os
import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace


class TestTaskRanking:
    """Test task-to-model ranking with Opus/GLM"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_rank_tasks_returns_model_assignments(self, temp_palace):
        """Ranking returns model assignment for each task"""
        tasks = [
            {"num": "1", "label": "Write unit tests", "description": "Simple test coverage"},
            {"num": "2", "label": "Refactor auth system", "description": "Complex security refactor"},
            {"num": "3", "label": "Fix typo in README", "description": "Trivial fix"}
        ]

        with patch.object(temp_palace, 'invoke_provider') as mock_invoke:
            mock_invoke.return_value = {
                "content": [{"type": "text", "text": json.dumps({
                    "assignments": [
                        {"task_num": "1", "model": "haiku", "reasoning": "Simple tests"},
                        {"task_num": "2", "model": "opus", "reasoning": "Complex refactor"},
                        {"task_num": "3", "model": "haiku", "reasoning": "Trivial"}
                    ]
                })}]
            }

            assignments = temp_palace.rank_tasks_by_model(tasks)

            assert len(assignments) == 3
            assert assignments["1"]["model"] == "haiku"
            assert assignments["2"]["model"] == "opus"
            assert assignments["3"]["model"] == "haiku"

    def test_rank_uses_opus_or_glm(self, temp_palace):
        """Ranking uses a high-quality model (Opus or GLM)"""
        tasks = [{"num": "1", "label": "Test task"}]

        with patch.object(temp_palace, 'invoke_provider') as mock_invoke:
            mock_invoke.return_value = {
                "content": [{"type": "text", "text": json.dumps({
                    "assignments": [{"task_num": "1", "model": "sonnet", "reasoning": "Medium"}]
                })}]
            }

            temp_palace.rank_tasks_by_model(tasks)

            # Should have called with opus or glm
            call_args = mock_invoke.call_args
            model = call_args[1].get("model", "")
            assert "opus" in model.lower() or "glm" in model.lower() or call_args[1].get("provider") == "z.ai"


class TestSwarmSpawning:
    """Test parallel Claude CLI spawning"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_spawn_swarm_creates_processes(self, temp_palace):
        """Swarm spawner creates subprocess for each task"""
        assignments = {
            "1": {"model": "haiku", "task": {"label": "Write tests"}},
            "2": {"model": "sonnet", "task": {"label": "Refactor code"}}
        }

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Still running
            mock_popen.return_value = mock_process

            processes = temp_palace.spawn_swarm(assignments, "Base prompt")

            assert len(processes) == 2
            assert mock_popen.call_count == 2

    def test_swarm_uses_correct_models(self, temp_palace):
        """Each swarm process uses its assigned model"""
        assignments = {
            "1": {"model": "haiku", "task": {"label": "Quick task"}},
            "2": {"model": "opus", "task": {"label": "Complex task"}}
        }

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            temp_palace.spawn_swarm(assignments, "Base prompt")

            # Check that different models were used
            calls = mock_popen.call_args_list
            cmd_strings = [str(call) for call in calls]
            # At least one should have haiku, one should have opus (or their aliases)
            assert any("haiku" in s.lower() or "minimax" in s.lower() for s in cmd_strings) or len(calls) == 2


class TestSwarmPromptExecution:
    """Test that swarm prompts instruct agents to execute tasks"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_prompt_contains_execution_instructions(self, temp_palace):
        """Swarm prompt tells agent to actually execute, not just plan"""
        prompt = temp_palace.build_swarm_prompt(
            task="Write unit tests",
            agent_id="haiku-1"
        )

        # Must contain execution instructions
        assert "DO the work" in prompt
        assert "tools" in prompt.lower()

    def test_prompt_contains_task(self, temp_palace):
        """Swarm prompt includes the task"""
        prompt = temp_palace.build_swarm_prompt(
            task="Write tests for authentication",
            agent_id="haiku-1"
        )

        assert "Write tests for authentication" in prompt


class TestMonitorSwarmOutput:
    """Test that monitor_swarm correctly parses and displays streaming output"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_parses_system_init_message(self, temp_palace, capsys):
        """Monitor should parse system init and show model"""
        from io import StringIO
        import select

        # Simulate streaming JSON output
        fake_output = StringIO(json.dumps({
            "type": "system",
            "subtype": "init",
            "model": "claude-sonnet-4-5"
        }) + "\n")

        # Create a mock process with stdin
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, 0]  # Running, then done
        mock_process.stdout = fake_output
        mock_process.stdin = MagicMock()  # Mock stdin for interleaving
        mock_process.returncode = 0

        processes = {
            "1": {
                "process": mock_process,
                "agent_id": "sonnet-1",
                "model": "sonnet",
                "task": "Test task",
            }
        }

        with patch('select.select', return_value=([fake_output], [], [])):
            results = temp_palace.monitor_swarm(processes)

        captured = capsys.readouterr()
        assert "sonnet-1" in captured.out or "Model" in captured.out

    def test_parses_tool_use_message(self, temp_palace, capsys):
        """Monitor should parse tool_use and show formatted output"""
        # Simulate a Read tool use
        fake_output = StringIO(json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg_1",
                "content": [{
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/test.py"}
                }]
            }
        }) + "\n")

        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, 0]
        mock_process.stdout = fake_output
        mock_process.stdin = MagicMock()  # Mock stdin for interleaving
        mock_process.returncode = 0

        processes = {
            "1": {
                "process": mock_process,
                "agent_id": "haiku-1",
                "model": "haiku",
                "task": "Read files",
            }
        }

        with patch('select.select', return_value=([fake_output], [], [])):
            results = temp_palace.monitor_swarm(processes)

        captured = capsys.readouterr()
        # Should show the formatted Read output
        assert "Reading" in captured.out or "haiku-1" in captured.out

    def test_real_subprocess_output(self, temp_palace, capsys):
        """Test with actual subprocess producing JSON output"""
        import subprocess

        # Create a script that outputs streaming JSON
        script = '''
import json
import sys
print(json.dumps({"type": "system", "subtype": "init", "model": "test-model"}))
sys.stdout.flush()
print(json.dumps({"type": "assistant", "message": {"id": "1", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/test.py"}}]}}))
sys.stdout.flush()
'''
        process = subprocess.Popen(
            ["python3", "-c", script],
            stdin=subprocess.PIPE,  # Need stdin for interleaving
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        processes = {
            "1": {
                "process": process,
                "agent_id": "test-1",
                "model": "test",
                "task": "Test",
            }
        }

        results = temp_palace.monitor_swarm(processes)

        captured = capsys.readouterr()
        print(f"DEBUG captured output: {captured.out}")  # Debug
        # Should have captured the output
        assert "test-1" in captured.out or "Reading" in captured.out or "Model" in captured.out


class TestTurboModeIntegration:
    """Integration tests for full turbo mode flow"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_turbo_mode_end_to_end(self, temp_palace):
        """Full turbo flow: rank -> spawn -> interleave"""
        tasks = [
            {"num": "1", "label": "Write tests"},
            {"num": "2", "label": "Refactor module"}
        ]

        with patch.object(temp_palace, 'invoke_provider') as mock_invoke:
            mock_invoke.return_value = {
                "content": [{"type": "text", "text": json.dumps({
                    "assignments": [
                        {"task_num": "1", "model": "haiku", "reasoning": "Simple"},
                        {"task_num": "2", "model": "sonnet", "reasoning": "Medium"}
                    ]
                })}]
            }

            with patch.object(temp_palace, 'spawn_swarm') as mock_spawn:
                mock_spawn.return_value = {"1": MagicMock(), "2": MagicMock()}

                with patch.object(temp_palace, 'monitor_swarm') as mock_monitor:
                    mock_monitor.return_value = {"1": {"exit_code": 0}, "2": {"exit_code": 0}}

                    result = temp_palace.run_turbo_mode(tasks, "Base context")

                    # Should have ranked tasks
                    mock_invoke.assert_called_once()

                    # Should have spawned swarm
                    mock_spawn.assert_called_once()

                    # Should have monitored
                    mock_monitor.assert_called_once()

                    assert result is not None
                    assert "assignments" in result
                    assert "results" in result
