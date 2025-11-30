"""
Provider System Tests

Tests for:
- Multi-provider support (Anthropic, Z.ai, OpenRouter)
- Model aliasing
- API format translation (Anthropic <-> OpenAI)
- Benchmarking harness
"""

import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace


class TestProviderConfig:
    """Test provider configuration"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        yield palace

    def test_default_provider_is_anthropic(self, temp_palace):
        """Default provider should be Anthropic"""
        config = temp_palace.get_provider_config()
        assert config["default_provider"] == "anthropic"

    def test_zai_in_default_providers(self, temp_palace):
        """Z.ai should be in default providers (for turbo mode ranking)"""
        config = temp_palace.get_provider_config()
        assert "z.ai" in config["providers"]
        assert config["providers"]["z.ai"]["api_key_env"] == "ZAI_API_KEY"

    def test_openrouter_in_default_providers(self, temp_palace):
        """OpenRouter should be in default providers"""
        config = temp_palace.get_provider_config()
        assert "openrouter" in config["providers"]
        assert config["providers"]["openrouter"]["format"] == "openai"

    def test_glm_alias_in_defaults(self, temp_palace):
        """GLM alias should be in defaults"""
        config = temp_palace.get_provider_config()
        assert "glm" in config["model_aliases"]
        assert config["model_aliases"]["glm"]["provider"] == "z.ai"

    def test_load_global_provider_config(self, temp_palace, tmp_path):
        """Load provider config from ~/.palace/providers.json"""
        # Create a fake global config
        global_palace = tmp_path / "fake_home" / ".palace"
        global_palace.mkdir(parents=True)
        config_path = global_palace / "providers.json"

        providers_config = {
            "model_aliases": {
                "custom-model": {"provider": "z.ai", "model": "custom-glm"}
            }
        }
        with open(config_path, 'w') as f:
            json.dump(providers_config, f)

        # Patch Path.home() to return our fake home
        with patch.object(Path, 'home', return_value=tmp_path / "fake_home"):
            config = temp_palace.get_provider_config()
            assert "custom-model" in config["model_aliases"]
            assert config["model_aliases"]["custom-model"]["model"] == "custom-glm"

    def test_resolve_model_alias(self, temp_palace):
        """Resolve model alias to provider and model (using defaults)"""
        # GLM is in defaults now
        provider, model = temp_palace.resolve_model("glm")
        assert provider == "z.ai"
        assert model == "glm-4.6"

    def test_resolve_unknown_model_uses_anthropic(self, temp_palace):
        """Unknown model names pass through to Anthropic"""
        provider, model = temp_palace.resolve_model("claude-opus-4-5-20250514")
        assert provider == "anthropic"
        assert model == "claude-opus-4-5-20250514"


class TestAnthropicFormat:
    """Test Anthropic API format (native)"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        yield palace

    def test_build_anthropic_request(self, temp_palace):
        """Build request in Anthropic format"""
        request = temp_palace.build_api_request(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hello"}],
            system="You are helpful"
        )

        assert request["model"] == "claude-sonnet-4-5-20250929"
        assert request["messages"] == [{"role": "user", "content": "Hello"}]
        assert request["system"] == "You are helpful"

    def test_zai_uses_anthropic_format(self, temp_palace):
        """Z.ai uses Anthropic format with different base URL (from defaults)"""
        config = temp_palace.get_provider_config()
        assert config["providers"]["z.ai"]["format"] == "anthropic"
        assert "z.ai" in config["providers"]["z.ai"]["base_url"]


class TestOpenAIFormatTranslation:
    """Test Anthropic <-> OpenAI format translation"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        yield palace

    def test_translate_messages_to_openai(self, temp_palace):
        """Translate Anthropic messages to OpenAI format"""
        anthropic_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        openai_messages = temp_palace.translate_to_openai_format(
            messages=anthropic_messages,
            system="You are helpful"
        )

        # OpenAI format has system as first message
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[0]["content"] == "You are helpful"
        assert openai_messages[1]["role"] == "user"
        assert openai_messages[2]["role"] == "assistant"

    def test_translate_tool_use_to_openai(self, temp_palace):
        """Translate Anthropic tool_use to OpenAI function_call"""
        anthropic_content = [
            {"type": "text", "text": "I'll read that file"},
            {
                "type": "tool_use",
                "id": "tool_123",
                "name": "Read",
                "input": {"file_path": "/tmp/test.txt"}
            }
        ]

        openai_message = temp_palace.translate_tool_use_to_openai(anthropic_content)

        assert "tool_calls" in openai_message
        assert openai_message["tool_calls"][0]["function"]["name"] == "Read"

    def test_translate_openai_response_to_anthropic(self, temp_palace):
        """Translate OpenAI response to Anthropic format"""
        openai_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {
                            "name": "Read",
                            "arguments": '{"file_path": "/tmp/test.txt"}'
                        }
                    }]
                }
            }]
        }

        anthropic_response = temp_palace.translate_openai_to_anthropic(openai_response)

        assert anthropic_response["role"] == "assistant"
        assert any(c["type"] == "text" for c in anthropic_response["content"])
        assert any(c["type"] == "tool_use" for c in anthropic_response["content"])


class TestProviderInvocation:
    """Test invoking different providers"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        yield palace

    def test_invoke_anthropic_provider(self, temp_palace):
        """Invoke Anthropic API directly"""
        with patch('anthropic.Anthropic') as mock_client:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Hello!")]
            mock_client.return_value.messages.create.return_value = mock_response

            response = temp_palace.invoke_provider(
                provider="anthropic",
                model="claude-haiku-4-20250514",
                messages=[{"role": "user", "content": "Hi"}]
            )

            mock_client.return_value.messages.create.assert_called_once()

    def test_invoke_openrouter_provider(self, temp_palace):
        """Invoke OpenRouter with format translation (using defaults)"""
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "Hi!"}}]
            }

            # OpenRouter is in defaults now
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                response = temp_palace.invoke_provider(
                    provider="openrouter",
                    model="minimax/minimax-m2",
                    messages=[{"role": "user", "content": "Hi"}]
                )

            # Should have called OpenRouter with OpenAI format
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "openrouter.ai" in call_args[0][0]


class TestBenchmarking:
    """Test model benchmarking system"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        os.chdir(tmp_path)
        palace = Palace()
        yield palace

    def test_benchmark_task_structure(self, temp_palace):
        """Benchmark tasks have required structure"""
        tasks = temp_palace.get_benchmark_tasks()

        assert len(tasks) > 0
        for task in tasks:
            assert "name" in task
            assert "prompt" in task
            assert "expected_capabilities" in task

    def test_run_single_benchmark(self, temp_palace):
        """Run benchmark for single model"""
        with patch.object(temp_palace, 'invoke_provider') as mock_invoke:
            mock_invoke.return_value = {"content": [{"type": "text", "text": "Done!"}]}

            result = temp_palace.run_benchmark(
                model_alias="haiku",
                task={"name": "simple", "prompt": "Say hello", "expected_capabilities": []}
            )

            assert "model" in result
            assert "latency_ms" in result
            assert "response" in result

    def test_judge_with_opus(self, temp_palace):
        """Judge benchmark result with Opus"""
        with patch.object(temp_palace, 'invoke_provider') as mock_invoke:
            mock_invoke.return_value = {
                "content": [{"type": "text", "text": '{"score": 8, "reasoning": "Good response"}'}]
            }

            score = temp_palace.judge_benchmark_result(
                task={"name": "test", "prompt": "Do X", "expected_capabilities": ["accuracy"]},
                response="I did X correctly"
            )

            assert "score" in score
            assert score["score"] >= 0 and score["score"] <= 10
