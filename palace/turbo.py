"""
Turbo mode - parallel swarm execution with model-task ranking
"""

import json
import subprocess
import select
import time
from typing import Dict, Any, List


class TurboMode:
    """Handles parallel Claude CLI execution with shared history"""

    def rank_tasks_by_model(self, tasks: List[dict]) -> Dict[str, Dict[str, Any]]:
        """
        Use a high-quality model (Opus or GLM) to assign tasks to optimal models.

        Returns dict: {task_num: {"model": alias, "reasoning": str, "task": dict}}
        """
        # Build task context for ranking LLM
        task_context = "\n".join([
            f"{t.get('num')}. {t.get('label')} - {t.get('description', '')}"
            for t in tasks
        ])

        ranking_prompt = f"""Analyze these tasks and assign the best model for each:

TASKS:
{task_context}

AVAILABLE MODELS:
- opus: High-quality reasoning, complex logic, architecture decisions
- sonnet: Balanced performance, general coding, refactoring
- haiku: Fast execution, simple tasks, formatting, documentation

For each task, assign the most efficient model (don't over-use opus).

Respond with JSON ONLY:
{{
  "assignments": [
    {{"task_num": "1", "model": "haiku", "reasoning": "Simple task"}},
    ...
  ]
}}"""

        # Use Opus or GLM for ranking (GLM preferred for cost)
        if hasattr(self, 'invoke_provider'):
            try:
                # Try GLM first (cheaper)
                result = self.invoke_provider(
                    provider="z.ai",
                    model="glm-4.6",
                    messages=[{"role": "user", "content": ranking_prompt}],
                    max_tokens=1024
                )
            except:
                # Fallback to Opus
                result = self.invoke_provider(
                    provider="anthropic",
                    model="claude-opus-4-5",
                    messages=[{"role": "user", "content": ranking_prompt}],
                    max_tokens=1024
                )

            # Parse response
            response_text = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    response_text += block.get("text", "")

            if "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                parsed = json.loads(response_text[json_start:json_end])

                # Build assignments dict
                assignments = {}
                for assignment in parsed.get("assignments", []):
                    task_num = assignment.get("task_num")
                    for task in tasks:
                        if task.get("num") == task_num:
                            assignments[task_num] = {
                                "model": assignment.get("model", "sonnet"),
                                "reasoning": assignment.get("reasoning", ""),
                                "task": task
                            }
                            break

                return assignments

        # Fallback: assign sonnet to all
        return {
            t.get("num"): {"model": "sonnet", "reasoning": "Default", "task": t}
            for t in tasks
        }

    def build_swarm_prompt(self, task: str, agent_id: str, base_context: str = None) -> str:
        """
        Build a prompt for a swarm agent.

        IMPORTANT: Instructs agent to EXECUTE, not just plan.
        """
        prompt = f"""# Palace Swarm Agent: {agent_id}

## YOUR TASK
{task}

## INSTRUCTIONS
YOU MUST DO THE WORK - not just plan it. Use your tools to:
1. Read files as needed
2. Edit or write code
3. Run tests if applicable
4. Complete the task fully

After completing, output your result in this format:
RESULT: <summary of what you did>

## COLLABORATION
You're working with other agents in parallel. They may be working on related tasks.
Don't worry about coordination - focus on your assigned task.
"""

        if base_context:
            prompt += f"\n## BASE CONTEXT\n{base_context}\n"

        return prompt

    def spawn_swarm(self, assignments: Dict[str, Dict[str, Any]], base_prompt: str = None) -> Dict[str, Dict]:
        """
        Spawn parallel Claude CLI processes for each task.

        Returns dict of: {task_num: {"process": Popen, "agent_id": str, "model": str, "task": str}}
        """
        processes = {}

        for task_num, assignment in assignments.items():
            model_alias = assignment.get("model", "sonnet")
            task = assignment.get("task", {})
            task_label = task.get("label", f"Task {task_num}")

            agent_id = f"{model_alias}-{task_num}"

            # Build prompt for this agent
            prompt = self.build_swarm_prompt(task_label, agent_id, base_prompt)

            # Resolve model to provider
            if hasattr(self, 'resolve_model'):
                provider, model = self.resolve_model(model_alias)
            else:
                provider, model = "anthropic", f"claude-{model_alias}-4-5"

            # Spawn Claude CLI process with streaming JSON
            # Use claude -p with --output-format stream-json
            cmd = [
                "claude",
                "-p", prompt,
                "--output-format", "stream-json",
                "--permission-prompt-tool", "mcp__palace__handle_permission"
            ]

            # TODO: Add model selection to claude CLI when supported
            # For now, uses default model

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,  # For interleaving
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1  # Line buffered
                )

                processes[task_num] = {
                    "process": proc,
                    "agent_id": agent_id,
                    "model": model_alias,
                    "task": task_label,
                }

            except Exception as e:
                print(f"⚠️  Failed to spawn {agent_id}: {e}")

        return processes

    def monitor_swarm(self, processes: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
        """
        Monitor swarm processes, display their output, and interleave their history.

        Uses select() to multiplex stdout from all processes.
        Forwards each agent's output to all other agents' stdin for omniscience.

        Returns results dict.
        """
        results = {}
        done_agents = set()

        # Set up select for non-blocking reads
        fds_to_task = {}
        for task_num, proc_info in processes.items():
            fds_to_task[proc_info["process"].stdout.fileno()] = task_num

        print("\n🚀 Swarm executing in parallel...\n")

        while len(done_agents) < len(processes):
            # Get all still-running process stdout fds
            readable_fds = [
                processes[task_num]["process"].stdout
                for task_num in processes
                if task_num not in done_agents
            ]

            if not readable_fds:
                break

            # Wait for any process to have output (with timeout)
            try:
                ready, _, _ = select.select(readable_fds, [], [], 0.1)
            except:
                break

            for fd in ready:
                task_num = fds_to_task.get(fd.fileno())
                if not task_num or task_num in done_agents:
                    continue

                proc_info = processes[task_num]
                agent_id = proc_info["agent_id"]

                # Read a line from this agent
                try:
                    line = fd.readline()
                    if not line:
                        # Process finished
                        proc_info["process"].wait()
                        done_agents.add(task_num)
                        results[task_num] = {
                            "exit_code": proc_info["process"].returncode,
                            "agent_id": agent_id
                        }
                        print(f"✅ [{agent_id}] Done (exit={proc_info['process'].returncode})")
                        continue

                    # Parse streaming JSON
                    try:
                        event = json.loads(line)

                        # Display formatted output
                        self._display_swarm_event(agent_id, event)

                        # Forward to other agents (shared history)
                        self._forward_to_other_agents(task_num, line, processes, done_agents)

                        # Check if this agent is done
                        if event.get("type") == "result":
                            done_agents.add(task_num)
                            results[task_num] = {
                                "exit_code": 0,
                                "agent_id": agent_id,
                                "result": event.get("result")
                            }
                            print(f"✅ [{agent_id}] Done")

                    except json.JSONDecodeError:
                        # Non-JSON output, just print
                        print(f"[{agent_id}] {line.rstrip()}")

                except Exception as e:
                    print(f"⚠️  [{agent_id}] Error reading: {e}")
                    done_agents.add(task_num)

            # Check for finished processes
            for task_num, proc_info in processes.items():
                if task_num in done_agents:
                    continue
                if proc_info["process"].poll() is not None:
                    done_agents.add(task_num)
                    results[task_num] = {
                        "exit_code": proc_info["process"].returncode,
                        "agent_id": proc_info["agent_id"]
                    }
                    print(f"✅ [{proc_info['agent_id']}] Exited (code={proc_info['process'].returncode})")

        print("\n✅ All swarm agents complete\n")
        return results

    def _display_swarm_event(self, agent_id: str, event: Dict[str, Any]):
        """Format and display a swarm agent's streaming event"""
        event_type = event.get("type")

        if event_type == "system" and event.get("subtype") == "init":
            model = event.get("model", "?")
            print(f"🤖 [{agent_id}] Model: {model}")

        elif event_type == "assistant":
            message = event.get("message", {})
            for block in message.get("content", []):
                if block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        print(f"💬 [{agent_id}] {text[:80]}{'...' if len(text) > 80 else ''}")
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "?")
                    print(f"🔧 [{agent_id}] Using {tool_name}")

        elif event_type == "tool_result":
            tool_name = event.get("name", "?")
            print(f"✓  [{agent_id}] {tool_name} completed")

        elif event_type == "result":
            result = event.get("result", "")
            print(f"✅ [{agent_id}] Result: {result[:60]}...")

    def _forward_to_other_agents(self, source_task_num: str, line: str,
                                   processes: Dict, done_agents: set):
        """
        Forward event to other agents' stdin for shared history.

        Skips agents that are already done.
        """
        for task_num, proc_info in processes.items():
            if task_num == source_task_num:
                continue  # Don't forward to self
            if task_num in done_agents:
                continue  # Don't forward to finished agents

            try:
                # Write to agent's stdin
                proc_info["process"].stdin.write(line)
                proc_info["process"].stdin.flush()
            except:
                pass  # Agent may have closed stdin

    def run_turbo_mode(self, tasks: List[dict], base_context: str = None) -> Dict[str, Any]:
        """
        Full turbo mode execution:
        1. Rank tasks by optimal model
        2. Spawn swarm in parallel
        3. Monitor and interleave

        Returns:
        - assignments: Task-to-model assignments
        - results: Execution results
        """
        print("\n⚡ TURBO MODE - Parallel swarm with GLM ranking\n")

        # Step 1: Rank tasks
        print("📊 Ranking tasks by model...")
        assignments = self.rank_tasks_by_model(tasks)

        # Display assignments
        print("\n🎯 Task assignments:")
        for task_num, assignment in assignments.items():
            model = assignment.get("model", "?")
            task_label = assignment.get("task", {}).get("label", "?")
            reasoning = assignment.get("reasoning", "")
            print(f"  {task_num}. [{model}] {task_label} - {reasoning}")

        # Step 2: Spawn swarm
        print("\n🚀 Spawning parallel agents...")
        processes = self.spawn_swarm(assignments, base_context)

        # Step 3: Monitor
        results = self.monitor_swarm(processes)

        return {
            "assignments": assignments,
            "results": results
        }

    def _evaluate_continuation_strategy(self, next_tasks: List[str], iteration: int) -> Dict[str, Any]:
        """
        Evaluate whether to auto-continue turbo mode or present options to user.

        Args:
            next_tasks: List of next task descriptions
            iteration: Current RHSI iteration number

        Returns:
            {
                "strategy": "auto_continue" | "present_options",
                "reason": "explanation",
                "confidence": 0.0-1.0
            }
        """
        import os

        if not next_tasks:
            return {
                "strategy": "present_options",
                "reason": "No specific tasks identified - user input needed",
                "confidence": 1.0
            }

        # Get recent history to check for rehashes
        history_context = ""
        history_file = self.palace_dir / "history.jsonl"
        if history_file.exists():
            recent_actions = []
            with open(history_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            if entry.get("action") in ["turbo_complete", "next"]:
                                recent_actions.append(entry)
                        except:
                            pass
            # Last 5 actions
            for action in recent_actions[-5:]:
                history_context += f"- {action.get('action')}: {action.get('details', {})}\n"

        prompt = f"""Evaluate continuation strategy for an RHSI turbo mode loop.

Current iteration: {iteration}

Next tasks identified:
{chr(10).join(f"- {t}" for t in next_tasks)}

Recent history:
{history_context if history_context else "No recent history"}

Decision criteria:
1. **Auto-continue** if:
   - Tasks are obvious completions of previous work
   - Tasks are fixing known issues from last iteration
   - No novel strategic decisions required
   - High confidence the right path is clear

2. **Present options** if:
   - Tasks represent new strategic directions
   - Multiple valid approaches exist
   - User input would materially improve outcome
   - Tasks require clarification or prioritization

Reply with JSON only:
{{"strategy": "auto_continue" or "present_options", "reason": "1-sentence explanation", "confidence": 0.0-1.0}}"""

        try:
            import anthropic

            # Use GLM for quick evaluation
            zai_key = os.environ.get("ZAI_API_KEY", "")
            if zai_key:
                client = anthropic.Anthropic(
                    api_key=zai_key,
                    base_url="https://api.z.ai/api/anthropic"
                )
            else:
                client = anthropic.Anthropic()

            response = client.messages.create(
                model="glm-4.6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Parse JSON
            if "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                result = json.loads(response_text[json_start:json_end])

                # Validate and return
                if result.get("strategy") in ["auto_continue", "present_options"]:
                    return result

            # Fallback
            return {
                "strategy": "present_options",
                "reason": "Unable to evaluate - defaulting to user input",
                "confidence": 0.5
            }

        except Exception as e:
            # On error, default to presenting options (safer)
            return {
                "strategy": "present_options",
                "reason": f"Evaluation error: {str(e)[:30]}",
                "confidence": 0.0
            }
