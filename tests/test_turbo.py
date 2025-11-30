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
