#!/usr/bin/env python3
"""
Benchmark Palace context overhead

Measures:
- Context gathering time
- Context size in bytes/tokens
- Prompt building time
- Total overhead

Usage:
    python benchmarks/context_overhead.py
"""

import sys
import os
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation using Claude's ~4 chars per token rule.
    Not exact, but good enough for benchmarking.
    """
    return len(text) // 4


def benchmark_context_gathering():
    """Benchmark context gathering performance"""
    palace = Palace()

    print("🔬 Benchmarking Context Gathering")
    print("=" * 60)

    # Warm up
    _ = palace.gather_context()

    # Benchmark
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        context = palace.gather_context()
    end = time.perf_counter()

    avg_time = (end - start) / iterations * 1000  # ms

    print(f"Iterations: {iterations}")
    print(f"Average time: {avg_time:.2f}ms")
    print()

    return context, avg_time


def analyze_context_size(context: dict):
    """Analyze context size and structure"""
    print("📊 Context Size Analysis")
    print("=" * 60)

    # Serialize to JSON
    json_str = json.dumps(context, indent=2)
    json_bytes = len(json_str.encode('utf-8'))
    json_tokens = estimate_tokens(json_str)

    print(f"JSON size: {json_bytes:,} bytes ({json_bytes/1024:.2f} KB)")
    print(f"Estimated tokens: {json_tokens:,}")
    print()

    # Breakdown by component
    print("Component breakdown:")
    for key, value in context.items():
        if key == "recent_history" and value:
            size = len(json.dumps(value))
            print(f"  {key}: {len(value)} entries, {size:,} bytes")
        elif isinstance(value, dict):
            size = len(json.dumps(value))
            print(f"  {key}: {len(value)} items, {size:,} bytes")
        elif isinstance(value, str):
            print(f"  {key}: {len(value):,} chars")
        else:
            print(f"  {key}: {value}")
    print()

    return json_bytes, json_tokens


def benchmark_prompt_building(context: dict):
    """Benchmark prompt building"""
    palace = Palace()

    print("🔨 Benchmarking Prompt Building")
    print("=" * 60)

    task = "Analyze this project and suggest next actions."

    # Warm up
    _ = palace.build_prompt(task, context)

    # Benchmark
    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        prompt = palace.build_prompt(task, context)
    end = time.perf_counter()

    avg_time = (end - start) / iterations * 1000  # ms

    print(f"Iterations: {iterations}")
    print(f"Average time: {avg_time:.2f}ms")
    print()

    # Analyze prompt size
    prompt_bytes = len(prompt.encode('utf-8'))
    prompt_tokens = estimate_tokens(prompt)

    print(f"Prompt size: {prompt_bytes:,} bytes ({prompt_bytes/1024:.2f} KB)")
    print(f"Estimated tokens: {prompt_tokens:,}")
    print()

    return prompt, avg_time, prompt_bytes, prompt_tokens


def benchmark_with_history():
    """Benchmark with varying history sizes"""
    palace = Palace()

    print("📈 Benchmarking with History")
    print("=" * 60)

    # Add varying amounts of history
    history_sizes = [0, 5, 10, 20, 50]
    results = []

    for size in history_sizes:
        # Clear and recreate history
        history_file = palace.palace_dir / "history.jsonl"
        if history_file.exists():
            history_file.unlink()

        # Add history entries
        palace.ensure_palace_dir()
        for i in range(size):
            palace.log_action(f"action_{i}", {
                "detail": f"test_{i}",
                "iteration": i,
                "data": {"key": f"value_{i}"}
            })

        # Gather context
        start = time.perf_counter()
        context = palace.gather_context()
        end = time.perf_counter()

        gather_time = (end - start) * 1000
        context_size = len(json.dumps(context))

        results.append({
            "history_entries": size,
            "gather_time_ms": gather_time,
            "context_bytes": context_size,
            "context_kb": context_size / 1024
        })

        print(f"History: {size:3d} entries | "
              f"Time: {gather_time:6.2f}ms | "
              f"Size: {context_size/1024:6.2f} KB")

    print()
    return results


def summary_report(context_time, context_bytes, context_tokens,
                   prompt_time, prompt_bytes, prompt_tokens):
    """Print summary report"""
    print("📝 Summary Report")
    print("=" * 60)

    total_time = context_time + prompt_time
    total_bytes = context_bytes + prompt_bytes
    total_tokens = context_tokens + prompt_tokens

    print(f"Total overhead per invocation:")
    print(f"  Time: {total_time:.2f}ms")
    print(f"  Size: {total_bytes:,} bytes ({total_bytes/1024:.2f} KB)")
    print(f"  Tokens: {total_tokens:,} (~{total_tokens/200000*100:.3f}% of 200K context)")
    print()

    print("Breakdown:")
    print(f"  Context gathering: {context_time:.2f}ms, {context_bytes:,} bytes, {context_tokens:,} tokens")
    print(f"  Prompt building: {prompt_time:.2f}ms, {prompt_bytes:,} bytes, {prompt_tokens:,} tokens")
    print()

    # Efficiency metrics
    print("Efficiency metrics:")
    print(f"  Bytes per ms: {total_bytes/total_time:,.0f}")
    print(f"  Tokens per ms: {total_tokens/total_time:,.0f}")
    print(f"  Context overhead: {context_bytes/total_bytes*100:.1f}% of total")
    print()


def main():
    """Run all benchmarks"""
    print()
    print("🏛️  Palace Context Overhead Benchmark")
    print("=" * 60)
    print()

    # 1. Context gathering
    context, context_time = benchmark_context_gathering()

    # 2. Context size
    context_bytes, context_tokens = analyze_context_size(context)

    # 3. Prompt building
    prompt, prompt_time, prompt_bytes, prompt_tokens = benchmark_prompt_building(context)

    # 4. History impact
    history_results = benchmark_with_history()

    # 5. Summary
    summary_report(context_time, context_bytes, context_tokens,
                   prompt_time, prompt_bytes, prompt_tokens)

    print("✅ Benchmark complete!")
    print()

    # Save results
    results = {
        "timestamp": time.time(),
        "context_time_ms": context_time,
        "context_bytes": context_bytes,
        "context_tokens": context_tokens,
        "prompt_time_ms": prompt_time,
        "prompt_bytes": prompt_bytes,
        "prompt_tokens": prompt_tokens,
        "total_time_ms": context_time + prompt_time,
        "total_bytes": context_bytes + prompt_bytes,
        "total_tokens": context_tokens + prompt_tokens,
        "history_impact": history_results
    }

    palace = Palace()
    palace.ensure_palace_dir()
    benchmark_file = palace.palace_dir / "benchmark_results.json"
    with open(benchmark_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {benchmark_file}")


if __name__ == "__main__":
    main()
