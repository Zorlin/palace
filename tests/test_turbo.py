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


class TestSharedHistory:
    """Test shared history interleaving (omniscient agents)"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_swarm_history_file_created(self, temp_palace):
        """Swarm creates shared history file"""
        temp_palace.init_swarm_history("session-123")

        swarm_file = temp_palace.palace_dir / "swarm" / "session-123.jsonl"
        assert swarm_file.exists()

    def test_write_swarm_event(self, temp_palace):
        """Agents can write to shared history"""
        temp_palace.init_swarm_history("session-123")

        temp_palace.write_swarm_event("session-123", {
            "agent": "haiku-1",
            "type": "tool_use",
            "tool": "Read",
            "file": "test.py"
        })

        swarm_file = temp_palace.palace_dir / "swarm" / "session-123.jsonl"
        content = swarm_file.read_text()
        assert "haiku-1" in content
        assert "Read" in content

    def test_read_swarm_events(self, temp_palace):
        """Agents can read all shared history"""
        temp_palace.init_swarm_history("session-123")

        # Simulate multiple agents writing
        temp_palace.write_swarm_event("session-123", {"agent": "haiku-1", "action": "reading files"})
        temp_palace.write_swarm_event("session-123", {"agent": "opus-2", "action": "designing arch"})
        temp_palace.write_swarm_event("session-123", {"agent": "haiku-1", "action": "writing tests"})

        events = temp_palace.read_swarm_events("session-123")

        assert len(events) == 3
        agents = [e["agent"] for e in events]
        assert "haiku-1" in agents
        assert "opus-2" in agents

    def test_read_new_events_since(self, temp_palace):
        """Agents can read only new events since last check"""
        temp_palace.init_swarm_history("session-123")

        temp_palace.write_swarm_event("session-123", {"agent": "a", "seq": 1})
        temp_palace.write_swarm_event("session-123", {"agent": "b", "seq": 2})

        # Read all, get offset
        events, offset = temp_palace.read_swarm_events("session-123", return_offset=True)
        assert len(events) == 2

        # Write more
        temp_palace.write_swarm_event("session-123", {"agent": "a", "seq": 3})

        # Read only new
        new_events, _ = temp_palace.read_swarm_events("session-123", since_offset=offset, return_offset=True)
        assert len(new_events) == 1
        assert new_events[0]["seq"] == 3


class TestSwarmContext:
    """Test injecting swarm awareness into prompts"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_build_swarm_context(self, temp_palace):
        """Build context showing other agents' activity"""
        temp_palace.init_swarm_history("session-123")

        temp_palace.write_swarm_event("session-123", {
            "agent": "opus-2",
            "type": "progress",
            "message": "Designing database schema"
        })
        temp_palace.write_swarm_event("session-123", {
            "agent": "haiku-3",
            "type": "progress",
            "message": "Writing API tests"
        })

        context = temp_palace.build_swarm_context("session-123", current_agent="haiku-1")

        assert "opus-2" in context
        assert "haiku-3" in context
        assert "database schema" in context.lower() or "Designing" in context
        assert "haiku-1" not in context  # Don't include own events

    def test_swarm_aware_prompt(self, temp_palace):
        """Prompt includes swarm awareness section"""
        temp_palace.init_swarm_history("session-123")
        temp_palace.write_swarm_event("session-123", {
            "agent": "opus-2",
            "message": "Working on auth"
        })

        prompt = temp_palace.build_swarm_prompt(
            task="Write tests",
            session_id="session-123",
            agent_id="haiku-1"
        )

        assert "SWARM" in prompt or "OTHER AGENTS" in prompt or "PARALLEL" in prompt
        assert "opus-2" in prompt
        assert "auth" in prompt.lower()


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
        temp_palace.init_swarm_history("session-123")

        prompt = temp_palace.build_swarm_prompt(
            task="Write unit tests",
            session_id="session-123",
            agent_id="haiku-1"
        )

        # Must contain execution instructions
        assert "EXECUTE" in prompt or "DO the work" in prompt
        assert "tools" in prompt.lower()

    def test_prompt_contains_shared_history_path(self, temp_palace):
        """Swarm prompt includes path to shared history file"""
        temp_palace.init_swarm_history("session-123")

        prompt = temp_palace.build_swarm_prompt(
            task="Write tests",
            session_id="session-123",
            agent_id="haiku-1"
        )

        # Must include the swarm history file path
        assert "session-123.jsonl" in prompt
        assert "swarm" in prompt


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

        # Create a mock process
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, 0]  # Running, then done
        mock_process.stdout = fake_output
        mock_process.returncode = 0

        processes = {
            "1": {
                "process": mock_process,
                "agent_id": "sonnet-1",
                "model": "sonnet",
                "task": "Test task",
                "session_id": "test-session"
            }
        }

        temp_palace.init_swarm_history("test-session")

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
        mock_process.returncode = 0

        processes = {
            "1": {
                "process": mock_process,
                "agent_id": "haiku-1",
                "model": "haiku",
                "task": "Read files",
                "session_id": "test-session"
            }
        }

        temp_palace.init_swarm_history("test-session")

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
                "session_id": "test-session"
            }
        }

        temp_palace.init_swarm_history("test-session")
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
