"""
Multi-provider support and benchmarking system for Palace
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple


class ProviderSystem:
    """Handles multiple AI providers and benchmarking"""

    def get_provider_config(self) -> Dict[str, Any]:
        """
        Load provider configuration.

        Checks (in order, later overrides earlier):
        1. Built-in defaults
        2. Global ~/.palace/providers.json
        3. Project .palace/providers.json

        Returns merged config.
        """
        defaults = {
            "default_provider": "anthropic",
            "providers": {
                "anthropic": {
                    "base_url": "https://api.anthropic.com",
                    "format": "anthropic",
                    "api_key_env": "ANTHROPIC_API_KEY"
                },
                "z.ai": {
                    "base_url": "https://api.z.ai/api/anthropic",
                    "format": "anthropic",
                    "api_key_env": "ZAI_API_KEY"
                },
                "openrouter": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "format": "openai",
                    "api_key_env": "OPENROUTER_API_KEY"
                }
            },
            "model_aliases": {
                "opus": {"provider": "anthropic", "model": "claude-opus-4-5"},
                "sonnet": {"provider": "anthropic", "model": "claude-sonnet-4-5"},
                "haiku": {"provider": "anthropic", "model": "claude-haiku-4-5"},
                "glm": {"provider": "z.ai", "model": "glm-4.6"},
                "glm-fast": {"provider": "z.ai", "model": "glm-4-flash"}
            }
        }

        # Load from ~/.palace/providers.json
        config_path = Path.home() / ".palace" / "providers.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                # Merge user config with defaults
                for key in ["providers", "model_aliases"]:
                    if key in user_config:
                        defaults[key].update(user_config[key])
                if "default_provider" in user_config:
                    defaults["default_provider"] = user_config["default_provider"]
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Failed to load {config_path}: {e}")

        return defaults

    def resolve_model(self, model_or_alias: str) -> Tuple[str, str]:
        """
        Resolve model alias to (provider, model) tuple.

        If alias not found, assumes Anthropic provider with model name as-is.
        """
        config = self.get_provider_config()
        aliases = config.get("model_aliases", {})

        if model_or_alias in aliases:
            alias_config = aliases[model_or_alias]
            return alias_config["provider"], alias_config["model"]

        # Not an alias, assume Anthropic
        return "anthropic", model_or_alias

    def build_api_request(self, provider: str, model: str, messages: List[dict],
                          system: str = None, **kwargs) -> Dict[str, Any]:
        """Build API request in provider's native format"""
        config = self.get_provider_config()
        provider_config = config["providers"].get(provider, {})
        fmt = provider_config.get("format", "anthropic")

        if fmt == "anthropic":
            request = {"model": model, "messages": messages}
            if system:
                request["system"] = system
            request.update(kwargs)
            return request
        else:
            # OpenAI format
            return self._build_openai_request(model, messages, system, **kwargs)

    def _build_openai_request(self, model: str, messages: List[dict],
                               system: str = None, **kwargs) -> Dict[str, Any]:
        """Build request in OpenAI/OpenRouter format"""
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages)

        request = {"model": model, "messages": openai_messages}
        # Translate max_tokens to max_completion_tokens if needed
        if "max_tokens" in kwargs:
            request["max_completion_tokens"] = kwargs.pop("max_tokens")
        request.update(kwargs)
        return request

    def translate_to_openai_format(self, messages: List[dict], system: str = None) -> List[dict]:
        """Translate Anthropic messages to OpenAI format"""
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})

        for msg in messages:
            openai_messages.append({
                "role": msg["role"],
                "content": msg.get("content", "")
            })

        return openai_messages

    def translate_tool_use_to_openai(self, anthropic_content: List[dict]) -> Dict[str, Any]:
        """Translate Anthropic tool_use blocks to OpenAI function_call format"""
        text_parts = []
        tool_calls = []

        for block in anthropic_content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}))
                    }
                })

        result = {"role": "assistant", "content": " ".join(text_parts) if text_parts else None}
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    def translate_openai_to_anthropic(self, openai_response: Dict[str, Any]) -> Dict[str, Any]:
        """Translate OpenAI response to Anthropic format"""
        choice = openai_response.get("choices", [{}])[0]
        message = choice.get("message", {})

        content = []

        # Text content
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})

        # Tool calls
        for tool_call in message.get("tool_calls", []):
            func = tool_call.get("function", {})
            try:
                arguments = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}

            content.append({
                "type": "tool_use",
                "id": tool_call.get("id", ""),
                "name": func.get("name", ""),
                "input": arguments
            })

        return {"role": "assistant", "content": content}

    def invoke_provider(self, provider: str, model: str, messages: List[dict],
                        system: str = None, **kwargs) -> Dict[str, Any]:
        """
        Invoke any configured provider.

        Handles format translation automatically based on provider config.
        """
        import requests
        import anthropic

        config = self.get_provider_config()
        provider_config = config["providers"].get(provider)

        if not provider_config:
            raise ValueError(f"Unknown provider: {provider}")

        fmt = provider_config.get("format", "anthropic")
        base_url = provider_config.get("base_url", "")
        api_key_env = provider_config.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "")

        if fmt == "anthropic":
            # Use Anthropic SDK
            client = anthropic.Anthropic(
                api_key=api_key if api_key else None,
                base_url=base_url if base_url and "anthropic.com" not in base_url else None
            )
            response = client.messages.create(
                model=model,
                messages=messages,
                system=system or "",
                max_tokens=kwargs.get("max_tokens", 4096)
            )
            return {"content": [{"type": "text", "text": c.text} for c in response.content if hasattr(c, 'text')]}

        else:
            # OpenAI-compatible API (OpenRouter, etc.)
            openai_messages = self.translate_to_openai_format(messages, system)
            request_body = {
                "model": model,
                "messages": openai_messages,
                "max_tokens": kwargs.get("max_tokens", 4096)
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # OpenRouter-specific headers
            if "openrouter" in provider.lower():
                headers["HTTP-Referer"] = "https://github.com/anthropics/palace"
                headers["X-Title"] = "Palace RHSI"

            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=request_body
            )
            response.raise_for_status()
            return self.translate_openai_to_anthropic(response.json())

    # ========================================================================
    # Benchmarking System
    # ========================================================================

    def get_benchmark_tasks(self) -> List[Dict[str, Any]]:
        """Get standard benchmark tasks for model evaluation"""
        return [
            {
                "name": "code_generation",
                "prompt": "Write a Python function that calculates the nth Fibonacci number using memoization.",
                "expected_capabilities": ["code_quality", "correctness", "efficiency"]
            },
            {
                "name": "code_analysis",
                "prompt": "Analyze this code and identify potential bugs:\n\ndef divide(a, b):\n    return a / b",
                "expected_capabilities": ["bug_detection", "edge_cases"]
            },
            {
                "name": "refactoring",
                "prompt": "Refactor this code to be more readable:\n\ndef f(x):return[i for i in range(x)if i%2==0]",
                "expected_capabilities": ["readability", "best_practices"]
            },
            {
                "name": "natural_language",
                "prompt": "Explain how a binary search tree works to a beginner programmer.",
                "expected_capabilities": ["clarity", "accuracy", "pedagogy"]
            }
        ]

    def run_benchmark(self, model_alias: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single benchmark task on a model"""
        provider, model = self.resolve_model(model_alias)

        start_time = time.time()
        try:
            response = self.invoke_provider(
                provider=provider,
                model=model,
                messages=[{"role": "user", "content": task["prompt"]}],
                max_tokens=2048
            )
            latency_ms = (time.time() - start_time) * 1000

            response_text = ""
            for block in response.get("content", []):
                if block.get("type") == "text":
                    response_text += block.get("text", "")

            return {
                "model": model_alias,
                "provider": provider,
                "actual_model": model,
                "task": task["name"],
                "latency_ms": latency_ms,
                "response": response_text,
                "success": True
            }

        except Exception as e:
            return {
                "model": model_alias,
                "task": task["name"],
                "latency_ms": (time.time() - start_time) * 1000,
                "response": None,
                "success": False,
                "error": str(e)
            }

    def judge_benchmark_result(self, task: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Judge a benchmark result using Opus as the judge.

        Returns score (0-10) and reasoning.
        """
        judge_prompt = f"""You are evaluating an AI model's response to a coding task.

TASK: {task['prompt']}

EXPECTED CAPABILITIES: {', '.join(task['expected_capabilities'])}

MODEL RESPONSE:
{response}

Score this response from 0-10 based on:
- Correctness and accuracy
- Code quality (if applicable)
- Clarity and helpfulness
- Addressing all aspects of the task

Respond with JSON only:
{{"score": <0-10>, "reasoning": "<brief explanation>"}}"""

        try:
            result = self.invoke_provider(
                provider="anthropic",
                model="claude-opus-4-5",
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=512
            )

            response_text = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    response_text += block.get("text", "")

            # Parse JSON from response
            if "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                return json.loads(response_text[json_start:json_end])

        except Exception as e:
            pass

        return {"score": 0, "reasoning": f"Judge failed: {str(e)}"}

    def run_full_benchmark(self, model_aliases: List[str] = None) -> Dict[str, Any]:
        """
        Run full benchmark suite across multiple models.

        Returns comparative results.
        """
        if model_aliases is None:
            model_aliases = ["opus", "sonnet", "haiku"]

        tasks = self.get_benchmark_tasks()
        results = {"models": {}, "summary": {}}

        for alias in model_aliases:
            results["models"][alias] = {"tasks": [], "total_score": 0, "avg_latency_ms": 0}

            total_latency = 0
            for task in tasks:
                print(f"  Benchmarking {alias} on {task['name']}...")
                result = self.run_benchmark(alias, task)

                if result["success"]:
                    score = self.judge_benchmark_result(task, result["response"])
                    result["score"] = score.get("score", 0)
                    result["reasoning"] = score.get("reasoning", "")
                    results["models"][alias]["total_score"] += result["score"]
                else:
                    result["score"] = 0

                total_latency += result["latency_ms"]
                results["models"][alias]["tasks"].append(result)

            results["models"][alias]["avg_latency_ms"] = total_latency / len(tasks)

        # Summary ranking
        rankings = sorted(
            [(alias, data["total_score"], data["avg_latency_ms"])
             for alias, data in results["models"].items()],
            key=lambda x: (-x[1], x[2])  # Higher score, lower latency
        )
        results["summary"]["rankings"] = [
            {"model": r[0], "total_score": r[1], "avg_latency_ms": r[2]}
            for r in rankings
        ]

        return results
