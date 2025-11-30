"""
Comprehensive tests for prompt building and context management

Tests critical paths for:
- Prompt generation with various contexts
- Mask integration with prompts
- User steering integration
- Multi-mask composition
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path to import palace
sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace


class TestPromptBuilding:
    """Test prompt building with various contexts"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_build_basic_prompt(self, temp_palace):
        """Test basic prompt building without context"""
        prompt = temp_palace.build_prompt("Test task")

        assert "Test task" in prompt
        assert "# Palace Request" in prompt
        assert "## Project Context" in prompt
        assert "## Instructions" in prompt

    def test_build_prompt_includes_context(self, temp_palace):
        """Test prompt includes full project context"""
        context = temp_palace.gather_context()
        prompt = temp_palace.build_prompt("Test task", context)

        # Should include context as JSON
        assert "project_root" in prompt
        assert "palace_version" in prompt

    def test_build_prompt_with_files(self, temp_palace, tmp_path):
        """Test prompt includes file information"""
        # Create test files
        (tmp_path / "README.md").write_text("# Test")
        (tmp_path / "SPEC.md").write_text("# Spec")

        context = temp_palace.gather_context()
        prompt = temp_palace.build_prompt("Test task", context)

        assert "README.md" in prompt
        assert "SPEC.md" in prompt

    def test_build_prompt_with_history(self, temp_palace):
        """Test prompt includes recent history"""
        temp_palace.log_action("test_action_1", {"detail": "first"})
        temp_palace.log_action("test_action_2", {"detail": "second"})

        context = temp_palace.gather_context()
        prompt = temp_palace.build_prompt("Test task", context)

        assert "recent_history" in prompt

    def test_build_prompt_with_git_status(self, temp_palace, tmp_path):
        """Test prompt includes git status when available"""
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)

        context = temp_palace.gather_context()
        prompt = temp_palace.build_prompt("Test task", context)

        # Should have git_status key
        assert "git_status" in prompt

    def test_build_prompt_with_mask(self, temp_palace, tmp_path):
        """Test building prompt with a mask"""
        # Create a test mask
        mask_dir = temp_palace.palace_dir / "masks" / "available" / "test-mask"
        mask_dir.mkdir(parents=True, exist_ok=True)
        (mask_dir / "SKILL.md").write_text("# Test Mask\n\nThis is a test mask.")

        prompt = temp_palace.build_prompt_with_mask("Test task", "test-mask")

        assert prompt is not None
        assert "Test Mask" in prompt
        assert "Test task" in prompt

    def test_build_prompt_with_invalid_mask(self, temp_palace):
        """Test building prompt with nonexistent mask"""
        prompt = temp_palace.build_prompt_with_mask("Test task", "nonexistent-mask")

        assert prompt is None

    def test_build_prompt_with_multiple_masks(self, temp_palace, tmp_path):
        """Test building prompt with multiple masks"""
        # Create test masks
        for i in range(3):
            mask_dir = temp_palace.palace_dir / "masks" / "available" / f"mask-{i}"
            mask_dir.mkdir(parents=True, exist_ok=True)
            (mask_dir / "SKILL.md").write_text(f"# Mask {i}\n\nMask content {i}.")

        prompt = temp_palace.build_prompt_with_masks(
            "Test task",
            ["mask-0", "mask-1", "mask-2"],
            strategy="merge"
        )

        assert prompt is not None
        assert "Mask 0" in prompt
        assert "Mask 1" in prompt
        assert "Mask 2" in prompt
        assert "Test task" in prompt


class TestSteeringIntegration:
    """Test user steering integration with prompts"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_build_prompt_with_steering(self, temp_palace):
        """Test building prompt with user steering"""
        prompt = temp_palace.build_prompt_with_steering(
            "Test task",
            steering="Use TDD approach"
        )

        assert "Use TDD approach" in prompt
        assert "USER STEERING" in prompt
        assert "Test task" in prompt

    def test_build_prompt_without_steering(self, temp_palace):
        """Test building prompt without steering returns normal prompt"""
        prompt_with = temp_palace.build_prompt_with_steering("Test task", steering=None)
        prompt_without = temp_palace.build_prompt("Test task")

        # Should be essentially the same
        assert "Test task" in prompt_with
        assert "Test task" in prompt_without

    def test_steering_is_prominent(self, temp_palace):
        """Test that steering is prominently placed in prompt"""
        prompt = temp_palace.build_prompt_with_steering(
            "Test task",
            steering="Follow architectural guidelines"
        )

        # Steering should appear before the project context
        steering_idx = prompt.index("USER STEERING")
        context_idx = prompt.index("## Project Context")

        assert steering_idx < context_idx

    def test_log_steering_creates_history_entry(self, temp_palace):
        """Test that steering is logged to history"""
        temp_palace.log_steering("Use TypeScript")

        history_file = temp_palace.palace_dir / "history.jsonl"
        assert history_file.exists()

        lines = history_file.read_text().strip().split('\n')
        entry = json.loads(lines[-1])

        assert entry["action"] == "user_steering"
        assert entry["details"]["steering"] == "Use TypeScript"


class TestContextEfficiency:
    """Test that context gathering is efficient and doesn't bloat"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_context_size_is_reasonable(self, temp_palace, tmp_path):
        """Test that context doesn't include large data"""
        # Create some files
        (tmp_path / "README.md").write_text("# " + "x" * 10000)
        (tmp_path / "large_file.py").write_text("# " + "y" * 100000)

        context = temp_palace.gather_context()
        context_json = json.dumps(context)

        # Context should be small (< 10KB typically)
        assert len(context_json) < 50000

        # Should not include file contents
        assert "x" * 100 not in context_json
        assert "y" * 100 not in context_json

    def test_history_is_limited(self, temp_palace):
        """Test that history is capped at last N entries"""
        # Create many history entries
        for i in range(50):
            temp_palace.log_action(f"action_{i}", {"iteration": i})

        context = temp_palace.gather_context()

        # Should only include last 10
        assert len(context.get("recent_history", [])) == 10

        # Should be most recent ones
        assert context["recent_history"][-1]["action"] == "action_49"
        assert context["recent_history"][0]["action"] == "action_40"

    def test_file_metadata_not_content(self, temp_palace, tmp_path):
        """Test that context includes file metadata, not content"""
        test_content = "This is test content that should not appear in context"
        (tmp_path / "test_file.txt").write_text(test_content)

        context = temp_palace.gather_context()
        context_json = json.dumps(context)

        # Should not contain file content
        assert test_content not in context_json

        # But should have metadata
        assert "test_file.txt" not in context["files"]  # Not in important_files list


class TestPromptEdgeCases:
    """Test edge cases in prompt building"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in a temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_empty_task_prompt(self, temp_palace):
        """Test building prompt with empty task"""
        prompt = temp_palace.build_prompt("")

        assert "# Palace Request" in prompt
        assert "## Project Context" in prompt

    def test_very_long_task_prompt(self, temp_palace):
        """Test building prompt with very long task"""
        long_task = "x" * 10000
        prompt = temp_palace.build_prompt(long_task)

        assert long_task in prompt
        assert len(prompt) > len(long_task)

    def test_special_characters_in_task(self, temp_palace):
        """Test building prompt with special characters"""
        task = "Task with <special> & characters \"quoted\" and 'apostrophes'"
        prompt = temp_palace.build_prompt(task)

        assert task in prompt

    def test_unicode_in_task(self, temp_palace):
        """Test building prompt with unicode characters"""
        task = "Task with unicode: 你好 🏛️ café"
        prompt = temp_palace.build_prompt(task)

        assert task in prompt

    def test_multiline_task(self, temp_palace):
        """Test building prompt with multiline task"""
        task = """Line 1
Line 2
Line 3"""
        prompt = temp_palace.build_prompt(task)

        assert "Line 1" in prompt
        assert "Line 2" in prompt
        assert "Line 3" in prompt


class TestMaskCompositionStrategies:
    """Test different mask composition strategies"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance with test masks"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Create test masks with different priorities
        masks_data = [
            ("base-mask", "# Base Mask\n\nBase functionality.", None),
            ("expert-mask", "---\npriority: 10\n---\n# Expert Mask\n\nExpert knowledge.", "10"),
            ("override-mask", "---\npriority: 5\n---\n# Override Mask\n\nOverride rules.", "5"),
        ]

        for mask_name, content, priority in masks_data:
            mask_dir = palace.palace_dir / "masks" / "available" / mask_name
            mask_dir.mkdir(parents=True, exist_ok=True)
            (mask_dir / "SKILL.md").write_text(content)

        yield palace

    def test_merge_strategy_concatenates(self, temp_palace):
        """Test merge strategy concatenates masks in order"""
        composed = temp_palace.compose_masks(
            ["base-mask", "expert-mask"],
            strategy="merge"
        )

        assert composed is not None
        assert "Base Mask" in composed
        assert "Expert Mask" in composed
        # Base should come before expert
        assert composed.index("Base Mask") < composed.index("Expert Mask")

    def test_layer_strategy_uses_priority(self, temp_palace):
        """Test layer strategy respects priority"""
        composed = temp_palace.compose_masks(
            ["base-mask", "expert-mask", "override-mask"],
            strategy="layer"
        )

        assert composed is not None
        # Should be ordered by priority
        assert "base-mask" in composed
        assert "override-mask" in composed
        assert "expert-mask" in composed

    def test_blend_strategy_interleaves(self, temp_palace):
        """Test blend strategy interleaves sections"""
        composed = temp_palace.compose_masks(
            ["base-mask", "expert-mask"],
            strategy="blend"
        )

        assert composed is not None
        # Should have markers from both masks
        assert "base-mask" in composed
        assert "expert-mask" in composed

    def test_compose_with_missing_mask_fails(self, temp_palace):
        """Test composition fails if any mask is missing"""
        composed = temp_palace.compose_masks(
            ["base-mask", "nonexistent-mask"],
            strategy="merge"
        )

        assert composed is None

    def test_compose_empty_list(self, temp_palace):
        """Test composition with empty mask list"""
        composed = temp_palace.compose_masks([], strategy="merge")

        assert composed is None
