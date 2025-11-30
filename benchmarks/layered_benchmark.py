#!/usr/bin/env python3
"""
Palace Layered Benchmark

A two-layer benchmark testing Palace's recursive improvement capability:

Layer 1: Palace builds Palace
  - Uses current Palace (iteration 4) to create new Palace (iteration 5)
  - Measures self-improvement capability
  - Validates with tests (50+ tests must pass)

Layer 2: Created-Palace builds Asteroids
  - Uses the newly-created Palace to build Asteroids game
  - Measures real-world project capability
  - Validates game functionality

This architecture tests BOTH:
- Palace's ability to improve itself
- The resulting Palace's ability to build real projects

Metrics (per layer):
- Time to completion
- Number of RHSI iterations
- Test pass rate / functional completeness
- Code quality score
- Lines of code generated

Usage:
    # Full two-layer benchmark with Anthropic
    python benchmarks/layered_benchmark.py --provider anthropic

    # Full two-layer benchmark with Z.ai GLM-4.6
    # Requires: export ZAI_AUTH_TOKEN=<your-key>
    python benchmarks/layered_benchmark.py --provider zai

    # Compare both providers
    python benchmarks/layered_benchmark.py --compare

    # Run only Layer 1 (Palace builds Palace)
    python benchmarks/layered_benchmark.py --layer 1

    # Run only Layer 2 (Palace builds Asteroids)
    python benchmarks/layered_benchmark.py --layer 2

Note: API keys are loaded from environment variables (never hardcoded!)
  - Anthropic: ANTHROPIC_AUTH_TOKEN
  - Z.ai: ZAI_AUTH_TOKEN
"""

import sys
import os
import time
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace, VERSION


@dataclass
class BenchmarkResult:
    """Results from a self-replication benchmark run"""
    provider: str
    model: str
    start_time: float
    end_time: float
    duration_seconds: float
    iterations: int
    total_commits: int
    lines_of_code: int
    tests_passed: int
    tests_total: int
    test_pass_rate: float
    faithfulness_score: float
    token_usage: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


class SelfReplicationBenchmark:
    """Benchmark Palace's self-replication capability"""

    def __init__(self, provider: str = "anthropic", model: Optional[str] = None):
        self.provider = provider
        self.model = model or self._default_model()
        self.work_dir: Optional[Path] = None
        self.original_dir = Path.cwd()

    def _default_model(self) -> str:
        """Get default model for provider"""
        models = {
            "anthropic": "claude-sonnet-4-5",
            "zai": "glm-4.6"
        }
        return models.get(self.provider, "claude-sonnet-4-5")

    def setup_environment(self) -> Path:
        """Create clean environment for replication"""
        print(f"\n🏗️  Setting up clean environment...")

        # Create temporary work directory
        self.work_dir = Path(tempfile.mkdtemp(prefix="palace-bench-"))
        print(f"   Work directory: {self.work_dir}")

        # Copy seed files
        seed_files = [
            "SPEC.md",
            "README.md",
            "requirements.txt",
            ".gitignore",
            "LICENSE"
        ]

        for filename in seed_files:
            src = self.original_dir / filename
            if src.exists():
                shutil.copy2(src, self.work_dir / filename)
                print(f"   ✓ Copied {filename}")

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=self.work_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.name", "Palace Benchmark"],
            cwd=self.work_dir,
            capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.email", "benchmark@palace.dev"],
            cwd=self.work_dir,
            capture_output=True
        )
        subprocess.run(
            ["git", "add", "."],
            cwd=self.work_dir,
            capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial seed files"],
            cwd=self.work_dir,
            capture_output=True
        )
        print(f"   ✓ Initialized git repository")

        # Create minimal Palace stub
        stub_code = '''#!/usr/bin/env python3
"""Palace - Self-improving Claude wrapper"""
VERSION = "0.0.1-benchmark"

# This stub will be replaced by Palace's self-replication
if __name__ == "__main__":
    print("Palace stub - awaiting replication")
'''
        (self.work_dir / "palace.py").write_text(stub_code)
        print(f"   ✓ Created Palace stub")

        return self.work_dir

    def create_asteroids_spec(self) -> str:
        """Generate specification for Asteroids game"""
        spec = """# Asteroids Game Specification

Build a complete Asteroids game in Python using pygame.

## Game Requirements

### Core Mechanics
1. **Player Ship**
   - Rotate left/right with arrow keys
   - Thrust forward with up arrow
   - Shoot projectiles with spacebar
   - Wraps around screen edges
   - Has momentum/inertia physics

2. **Asteroids**
   - Spawn at random positions and velocities
   - Three sizes: large, medium, small
   - Split into smaller asteroids when hit
   - Wrap around screen edges
   - Realistic physics

3. **Collision Detection**
   - Ship collides with asteroids = lose life
   - Bullets collide with asteroids = destroy/split
   - Accurate collision using circular hit boxes

4. **Game State**
   - Score tracking (points for destroying asteroids)
   - Lives system (start with 3 lives)
   - Game over when lives reach 0
   - Level progression (more asteroids per level)
   - Restart functionality

5. **Visual & Audio**
   - Simple vector graphics or sprites
   - Screen boundaries
   - Score and lives display
   - Optional: sound effects

## Technical Requirements

### Code Structure
```
asteroids/
├── main.py          # Entry point, game loop
├── game.py          # Game state management
├── entities.py      # Ship, Asteroid, Bullet classes
├── physics.py       # Physics calculations
├── renderer.py      # Drawing functions
├── tests/
│   ├── test_game.py
│   ├── test_entities.py
│   └── test_physics.py
├── requirements.txt # Dependencies (pygame, pytest)
└── README.md        # Setup and play instructions
```

### Testing
- Unit tests for physics calculations
- Tests for collision detection
- Tests for game state management
- Aim for 70%+ code coverage

### Code Quality
- Clean, documented code
- Type hints where appropriate
- Follows PEP 8
- No unused imports
- DRY principles

## Success Criteria

### Functional (Must Have)
- ✓ Game runs without errors
- ✓ Ship controls work correctly
- ✓ Asteroids spawn and move
- ✓ Shooting mechanic works
- ✓ Collision detection accurate
- ✓ Score increases when asteroids destroyed
- ✓ Lives decrease on collision
- ✓ Game over when lives = 0
- ✓ Can restart game

### Quality (Should Have)
- ✓ All tests pass
- ✓ 70%+ test coverage
- ✓ Code is well-documented
- ✓ README with instructions
- ✓ Clean git history

### Polish (Nice to Have)
- ○ Sound effects
- ○ Particle effects
- ○ High score tracking
- ○ Pause functionality
- ○ Difficulty settings

## Development Approach

Use TDD (Test-Driven Development):
1. Write tests first
2. Implement functionality
3. Refactor for quality
4. Commit frequently

Begin with basic structure, then iterate:
- Iteration 1: Basic window and ship rendering
- Iteration 2: Ship movement and controls
- Iteration 3: Asteroids spawning and movement
- Iteration 4: Collision detection
- Iteration 5: Game state and scoring
- Iteration 6: Polish and testing

Use `/pal-next` to guide development iteration by iteration.
"""
        return spec

    def _setup_api_environment(self):
        """Configure environment variables for API provider"""
        env = os.environ.copy()

        if self.provider == "zai":
            # Z.ai uses Anthropic-compatible API
            if "ZAI_AUTH_TOKEN" in os.environ:
                env["ANTHROPIC_AUTH_TOKEN"] = os.environ["ZAI_AUTH_TOKEN"]
            env["ANTHROPIC_BASE_URL"] = "https://api.z.ai/api/anthropic"
            print(f"   ✓ Configured Z.ai endpoint")
        else:
            # Use default Anthropic configuration
            print(f"   ✓ Using Anthropic API")

        return env

    def run_asteroids_benchmark(self) -> BenchmarkResult:
        """Execute the Asteroids game build benchmark"""
        print(f"\n🚀 Starting Asteroids Build Benchmark")
        print(f"   Provider: {self.provider}")
        print(f"   Model: {self.model}")

        start_time = time.time()

        try:
            # Setup environment
            work_dir = self.setup_environment()
            os.chdir(work_dir)

            # Save the Asteroids specification
            spec_file = work_dir / "SPEC.md"
            spec_file.write_text(self.create_asteroids_spec())
            print(f"   ✓ Created SPEC.md with Asteroids requirements")

            # Create initial README
            readme = """# Asteroids Game

A Python implementation of the classic Asteroids arcade game.

See SPEC.md for full requirements and development approach.

## Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## Testing
```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```
"""
            (work_dir / "README.md").write_text(readme)

            # Setup API environment
            api_env = self._setup_api_environment()

            print(f"\n📝 Asteroids spec ready at: {spec_file}")
            print(f"\n⏱️  Benchmark environment prepared!")
            print(f"\n   To run the benchmark:")
            print(f"   1. cd {work_dir}")
            print(f"   2. Initialize Palace: python3 /path/to/palace.py init")
            print(f"   3. Start RHSI loop: python3 /path/to/palace.py next")
            print(f"   4. Palace will iteratively build the game")
            print(f"   5. Validate with: python benchmarks/self_replication.py --validate {work_dir}")

            result = BenchmarkResult(
                provider=self.provider,
                model=self.model,
                start_time=start_time,
                end_time=time.time(),
                duration_seconds=0,
                iterations=0,
                total_commits=0,
                lines_of_code=0,
                tests_passed=0,
                tests_total=0,
                test_pass_rate=0.0,
                faithfulness_score=0.0,
                success=False,
                error="Manual execution required - benchmark environment ready"
            )

            return result

        except Exception as e:
            end_time = time.time()
            return BenchmarkResult(
                provider=self.provider,
                model=self.model,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=end_time - start_time,
                iterations=0,
                total_commits=0,
                lines_of_code=0,
                tests_passed=0,
                tests_total=0,
                test_pass_rate=0.0,
                faithfulness_score=0.0,
                success=False,
                error=str(e)
            )
        finally:
            os.chdir(self.original_dir)

    def validate_replication(self, work_dir: Path) -> Dict[str, Any]:
        """Validate the replicated Palace"""
        print(f"\n🔍 Validating replication...")

        validation = {
            "files_exist": {},
            "tests_pass": False,
            "test_results": {},
            "functionality": {},
            "faithfulness_score": 0.0
        }

        # Check required files exist
        required_files = [
            "palace.py",
            "CLAUDE.md",
            "SPEC.md",
            "ROADMAP.md",
            "requirements.txt",
            "tests/__init__.py",
            "tests/test_core.py"
        ]

        for filename in required_files:
            exists = (work_dir / filename).exists()
            validation["files_exist"][filename] = exists
            print(f"   {'✓' if exists else '✗'} {filename}")

        # Run tests
        if (work_dir / "tests").exists():
            try:
                result = subprocess.run(
                    ["python3", "-m", "pytest", "tests/", "-v", "--tb=short"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                validation["tests_pass"] = result.returncode == 0
                validation["test_results"]["output"] = result.stdout
                validation["test_results"]["errors"] = result.stderr

                # Parse test results
                if "passed" in result.stdout:
                    import re
                    match = re.search(r'(\d+) passed', result.stdout)
                    if match:
                        validation["test_results"]["passed"] = int(match.group(1))

                print(f"   {'✓' if validation['tests_pass'] else '✗'} Tests: {validation.get('test_results', {}).get('passed', 0)} passed")
            except Exception as e:
                validation["test_results"]["error"] = str(e)
                print(f"   ✗ Test execution failed: {e}")

        # Calculate faithfulness score
        score = 0.0
        total_checks = len(required_files) + 1  # files + tests

        for exists in validation["files_exist"].values():
            if exists:
                score += 1.0

        if validation["tests_pass"]:
            score += 1.0

        validation["faithfulness_score"] = (score / total_checks) * 100.0

        print(f"\n   Faithfulness Score: {validation['faithfulness_score']:.1f}%")

        return validation

    def cleanup(self):
        """Clean up temporary files"""
        if self.work_dir and self.work_dir.exists():
            try:
                shutil.rmtree(self.work_dir)
                print(f"\n🧹 Cleaned up: {self.work_dir}")
            except Exception as e:
                print(f"\n⚠️  Cleanup failed: {e}")


def main():
    """Main benchmark execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Palace Self-Replication Benchmark")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "zai"],
        default="anthropic",
        help="API provider to use"
    )
    parser.add_argument(
        "--model",
        help="Model to use (defaults to provider default)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run benchmarks for all providers and compare"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't cleanup work directory after benchmark"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🏛️  Palace Self-Replication Benchmark")
    print("=" * 70)

    if args.compare:
        print("\n📊 Running comparison benchmark...")
        providers = ["anthropic", "zai"]
        results = []

        for provider in providers:
            bench = SelfReplicationBenchmark(provider=provider)
            result = bench.run_replication()
            results.append(result)

            if not args.no_cleanup:
                bench.cleanup()

        # Display comparison
        print("\n" + "=" * 70)
        print("📈 Comparison Results")
        print("=" * 70)
        for result in results:
            print(f"\n{result.provider} ({result.model}):")
            print(f"   Duration: {result.duration_seconds:.2f}s")
            print(f"   Success: {result.success}")
            if result.error:
                print(f"   Error: {result.error}")
    else:
        bench = SelfReplicationBenchmark(provider=args.provider, model=args.model)
        result = bench.run_replication()

        print("\n" + "=" * 70)
        print("📊 Benchmark Results")
        print("=" * 70)
        print(json.dumps(asdict(result), indent=2))

        if not args.no_cleanup:
            bench.cleanup()

    print("\n✨ Benchmark complete!")


if __name__ == "__main__":
    main()
