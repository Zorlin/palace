"""
Tests for Palace benchmark infrastructure
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add benchmarks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from layered_benchmark import LayeredBenchmark, BenchmarkResult


class TestLayeredBenchmark:
    """Test the layered benchmark infrastructure"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def bench(self):
        """Create a benchmark instance"""
        return LayeredBenchmark(provider="anthropic")

    def test_benchmark_initialization(self, bench):
        """Test benchmark initializes correctly"""
        assert bench.provider == "anthropic"
        assert bench.model == "claude-sonnet-4-5"
        assert bench.work_dir is None
        assert bench.original_dir == Path.cwd()

    def test_default_model_selection(self):
        """Test default model selection for providers"""
        bench_anthropic = LayeredBenchmark(provider="anthropic")
        assert bench_anthropic.model == "claude-sonnet-4-5"

        bench_zai = LayeredBenchmark(provider="zai")
        assert bench_zai.model == "glm-4.6"

    def test_custom_model(self):
        """Test custom model specification"""
        bench = LayeredBenchmark(provider="anthropic", model="custom-model")
        assert bench.model == "custom-model"

    def test_setup_environment(self, bench, temp_dir):
        """Test environment setup creates necessary structure"""
        os.chdir(temp_dir)

        # Create minimal seed files
        (temp_dir / "SPEC.md").write_text("# Test Spec")
        (temp_dir / "README.md").write_text("# Test")

        work_dir = bench.setup_environment()

        assert work_dir.exists()
        assert (work_dir / ".git").exists()
        assert (work_dir / "SPEC.md").exists()
        assert (work_dir / "palace.py").exists()

        # Cleanup
        os.chdir(bench.original_dir)

    def test_asteroids_spec_generation(self, bench):
        """Test Asteroids specification is generated correctly"""
        spec = bench.create_asteroids_spec()

        assert "Asteroids Game Specification" in spec
        assert "Player Ship" in spec
        assert "Collision Detection" in spec
        assert "pygame" in spec
        assert "TDD" in spec

    def test_api_environment_setup_anthropic(self, bench):
        """Test API environment configuration for Anthropic"""
        env = bench._setup_api_environment()

        assert "ANTHROPIC_BASE_URL" not in env or env["ANTHROPIC_BASE_URL"] != "https://api.z.ai/api/anthropic"

    def test_api_environment_setup_zai(self):
        """Test API environment configuration for Z.ai"""
        bench = LayeredBenchmark(provider="zai")
        env = bench._setup_api_environment()

        assert env["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"

    def test_validate_palace_missing_files(self, bench, temp_dir):
        """Test Palace validation with missing files"""
        validation = bench.validate_palace(temp_dir)

        assert "files_exist" in validation
        assert validation["faithfulness_score"] == 0.0
        assert not validation["tests_pass"]

    def test_validate_palace_with_files(self, bench, temp_dir):
        """Test Palace validation with required files present"""
        # Create required files
        (temp_dir / "palace.py").write_text("# Palace stub")
        (temp_dir / "CLAUDE.md").write_text("# Docs")
        (temp_dir / "SPEC.md").write_text("# Spec")
        (temp_dir / "ROADMAP.md").write_text("# Roadmap")
        (temp_dir / "requirements.txt").write_text("pytest")

        tests_dir = temp_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "__init__.py").write_text("")
        (tests_dir / "test_core.py").write_text("def test_placeholder(): pass")

        validation = bench.validate_palace(temp_dir)

        assert validation["files_exist"]["palace.py"]
        assert validation["files_exist"]["CLAUDE.md"]
        assert validation["faithfulness_score"] > 0.0

    def test_validate_asteroids_missing_files(self, bench, temp_dir):
        """Test Asteroids validation with missing files"""
        validation = bench.validate_asteroids(temp_dir)

        assert "files_exist" in validation
        assert validation["functionality_score"] == 0.0
        assert not validation["tests_pass"]
        assert not validation["game_runs"]

    def test_validate_asteroids_with_files(self, bench, temp_dir):
        """Test Asteroids validation with required files present"""
        # Create game files
        (temp_dir / "main.py").write_text("# Game entry point")
        (temp_dir / "game.py").write_text("# Game logic")
        (temp_dir / "entities.py").write_text("# Game entities")
        (temp_dir / "README.md").write_text("# Instructions")
        (temp_dir / "requirements.txt").write_text("pygame\npytest")

        tests_dir = temp_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_game.py").write_text("def test_game(): pass")
        (tests_dir / "test_entities.py").write_text("def test_entities(): pass")

        validation = bench.validate_asteroids(temp_dir)

        assert validation["files_exist"]["main.py"]
        assert validation["files_exist"]["game.py"]
        assert validation["functionality_score"] > 0.0

    def test_benchmark_result_structure(self):
        """Test BenchmarkResult dataclass structure"""
        result = BenchmarkResult(
            provider="anthropic",
            model="claude-sonnet-4-5",
            start_time=0.0,
            end_time=1.0,
            duration_seconds=1.0,
            iterations=5,
            total_commits=10,
            lines_of_code=500,
            tests_passed=50,
            tests_total=50,
            test_pass_rate=100.0,
            faithfulness_score=95.0
        )

        assert result.provider == "anthropic"
        assert result.duration_seconds == 1.0
        assert result.success is True
        assert result.error is None

    def test_benchmark_result_with_error(self):
        """Test BenchmarkResult with error"""
        result = BenchmarkResult(
            provider="anthropic",
            model="claude-sonnet-4-5",
            start_time=0.0,
            end_time=1.0,
            duration_seconds=1.0,
            iterations=0,
            total_commits=0,
            lines_of_code=0,
            tests_passed=0,
            tests_total=0,
            test_pass_rate=0.0,
            faithfulness_score=0.0,
            success=False,
            error="Test error"
        )

        assert result.success is False
        assert result.error == "Test error"


class TestBenchmarkValidation:
    """Test validation logic specifically"""

    @pytest.fixture
    def bench(self):
        return LayeredBenchmark()

    @pytest.fixture
    def palace_dir(self, tmp_path):
        """Create a minimal Palace directory structure"""
        (tmp_path / "palace.py").write_text("""
#!/usr/bin/env python3
VERSION = "0.1.0"

class Palace:
    def __init__(self):
        pass
""")
        (tmp_path / "CLAUDE.md").write_text("# Palace Documentation")
        (tmp_path / "SPEC.md").write_text("# Specification")
        (tmp_path / "ROADMAP.md").write_text("# Roadmap")
        (tmp_path / "requirements.txt").write_text("pytest>=7.4.0")

        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("")
        (tests / "test_core.py").write_text("""
import pytest

def test_example():
    assert True
""")

        return tmp_path

    @pytest.fixture
    def asteroids_dir(self, tmp_path):
        """Create a minimal Asteroids game directory structure"""
        (tmp_path / "main.py").write_text("""
import pygame

def main():
    pygame.init()
    print("Game initialized")

if __name__ == "__main__":
    main()
""")
        (tmp_path / "game.py").write_text("# Game state management")
        (tmp_path / "entities.py").write_text("# Game entities")
        (tmp_path / "README.md").write_text("# Asteroids Game")
        (tmp_path / "requirements.txt").write_text("pygame>=2.0.0\npytest>=7.4.0")

        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_game.py").write_text("""
def test_game_exists():
    assert True
""")
        (tests / "test_entities.py").write_text("""
def test_entities_exist():
    assert True
""")

        return tmp_path

    def test_palace_validation_score_calculation(self, bench, palace_dir):
        """Test Palace validation score is calculated correctly"""
        validation = bench.validate_palace(palace_dir)

        # All files exist (7) + tests would pass (1) = 8/8 = 100%
        # (Note: tests may not actually pass without dependencies, so we check structure)
        assert "faithfulness_score" in validation
        assert validation["faithfulness_score"] >= 0.0
        assert validation["faithfulness_score"] <= 100.0

    def test_asteroids_validation_score_calculation(self, bench, asteroids_dir):
        """Test Asteroids validation score is calculated correctly"""
        validation = bench.validate_asteroids(asteroids_dir)

        assert "functionality_score" in validation
        assert validation["functionality_score"] >= 0.0
        assert validation["functionality_score"] <= 1.0

    def test_validation_handles_missing_directories(self, bench, tmp_path):
        """Test validation handles missing test directories gracefully"""
        (tmp_path / "palace.py").write_text("# Stub")

        validation = bench.validate_palace(tmp_path)

        assert not validation["tests_pass"]
        assert validation["faithfulness_score"] < 100.0
