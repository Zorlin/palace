"""
Tests for Palace mask system

Tests mask loading, application, and management functionality.
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


class TestMaskLoading:
    """Test basic mask loading functionality"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance with mask directory structure"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Create mask directory structure
        masks_dir = palace.palace_dir / "masks"
        (masks_dir / "available").mkdir(parents=True, exist_ok=True)
        (masks_dir / "custom").mkdir(parents=True, exist_ok=True)

        yield palace

    def test_load_mask_from_available(self, temp_palace):
        """Load a mask from available directory"""
        # Create a test mask
        mask_dir = temp_palace.palace_dir / "masks" / "available" / "test-mask"
        mask_dir.mkdir(parents=True)
        skill_file = mask_dir / "SKILL.md"
        skill_file.write_text("# Test Mask\n\nThis is a test mask.")

        # Load the mask
        content = temp_palace.load_mask("test-mask")
        assert content is not None
        assert "Test Mask" in content

    def test_load_mask_from_custom(self, temp_palace):
        """Load a mask from custom directory"""
        # Create a custom mask
        mask_dir = temp_palace.palace_dir / "masks" / "custom" / "my-mask"
        mask_dir.mkdir(parents=True)
        skill_file = mask_dir / "SKILL.md"
        skill_file.write_text("# My Custom Mask\n\nCustom expertise.")

        # Load the mask
        content = temp_palace.load_mask("my-mask")
        assert content is not None
        assert "My Custom Mask" in content

    def test_load_nonexistent_mask(self, temp_palace):
        """Loading nonexistent mask returns None"""
        content = temp_palace.load_mask("nonexistent")
        assert content is None

    def test_available_mask_takes_precedence(self, temp_palace):
        """Available masks take precedence over custom"""
        # Create both masks with same name
        avail_dir = temp_palace.palace_dir / "masks" / "available" / "dual-mask"
        avail_dir.mkdir(parents=True)
        (avail_dir / "SKILL.md").write_text("# Available Version")

        custom_dir = temp_palace.palace_dir / "masks" / "custom" / "dual-mask"
        custom_dir.mkdir(parents=True)
        (custom_dir / "SKILL.md").write_text("# Custom Version")

        # Load should get available version
        content = temp_palace.load_mask("dual-mask")
        assert "Available Version" in content
        assert "Custom Version" not in content


class TestMaskApplication:
    """Test applying masks to Claude invocations"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance with a test mask"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Create mask directory and a test mask
        mask_dir = palace.palace_dir / "masks" / "available" / "historian"
        mask_dir.mkdir(parents=True, exist_ok=True)
        skill_file = mask_dir / "SKILL.md"
        skill_file.write_text("""# Palace Historian

## Identity
Expert in repository archaeology and cleanup.

## Core Expertise
- Git history analysis
- Space recovery
- Artifact preservation
""")

        yield palace

    def test_build_prompt_with_mask(self, temp_palace):
        """Build prompt with mask content prepended"""
        prompt = "Analyze this repository"
        full_prompt = temp_palace.build_prompt_with_mask(prompt, "historian")

        assert full_prompt is not None
        assert "Palace Historian" in full_prompt
        assert "repository archaeology" in full_prompt
        assert "Analyze this repository" in full_prompt

    def test_build_prompt_without_mask(self, temp_palace):
        """Build prompt without mask when none specified"""
        prompt = "Analyze this repository"
        full_prompt = temp_palace.build_prompt_with_mask(prompt, None)

        assert full_prompt is not None
        assert "Palace Historian" not in full_prompt
        assert "Analyze this repository" in full_prompt

    def test_build_prompt_with_invalid_mask(self, temp_palace):
        """Build prompt with invalid mask returns None"""
        prompt = "Analyze this repository"
        full_prompt = temp_palace.build_prompt_with_mask(prompt, "nonexistent")

        assert full_prompt is None


class TestMaskDiscovery:
    """Test mask discovery and listing"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance with multiple masks"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Create several test masks
        masks_dir = palace.palace_dir / "masks"

        # Available masks
        for name in ["historian", "architect", "tester"]:
            mask_dir = masks_dir / "available" / name
            mask_dir.mkdir(parents=True, exist_ok=True)
            (mask_dir / "SKILL.md").write_text(f"# {name.title()} Mask")

        # Custom masks
        for name in ["my-specialty", "team-expert"]:
            mask_dir = masks_dir / "custom" / name
            mask_dir.mkdir(parents=True, exist_ok=True)
            (mask_dir / "SKILL.md").write_text(f"# {name.title()} Mask")

        yield palace

    def test_list_masks(self, temp_palace):
        """List all available masks"""
        masks = temp_palace.list_masks()

        assert len(masks) >= 5
        mask_names = [m["name"] for m in masks]
        assert "historian" in mask_names
        assert "architect" in mask_names
        assert "my-specialty" in mask_names

    def test_list_masks_includes_type(self, temp_palace):
        """Listed masks include their type (available/custom)"""
        masks = temp_palace.list_masks()

        historian = next(m for m in masks if m["name"] == "historian")
        assert historian["type"] == "available"

        specialty = next(m for m in masks if m["name"] == "my-specialty")
        assert specialty["type"] == "custom"

    def test_list_masks_empty(self, tmp_path):
        """List masks returns empty list when no masks exist"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        masks = palace.list_masks()
        assert masks == []


class TestMaskMetadata:
    """Test mask metadata extraction"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance with mask containing frontmatter"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Create mask with frontmatter
        mask_dir = palace.palace_dir / "masks" / "available" / "meta-mask"
        mask_dir.mkdir(parents=True, exist_ok=True)
        skill_file = mask_dir / "SKILL.md"
        skill_file.write_text("""---
name: meta-mask
description: A mask with metadata
version: 1.0.0
---

# Meta Mask

Content here.
""")

        yield palace

    def test_parse_mask_frontmatter(self, temp_palace):
        """Parse mask frontmatter for metadata"""
        metadata = temp_palace.get_mask_metadata("meta-mask")

        assert metadata is not None
        assert metadata.get("name") == "meta-mask"
        assert metadata.get("description") == "A mask with metadata"
        assert metadata.get("version") == "1.0.0"

    def test_mask_without_frontmatter(self, tmp_path):
        """Masks without frontmatter return minimal metadata"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Create mask without frontmatter
        mask_dir = palace.palace_dir / "masks" / "available" / "simple-mask"
        mask_dir.mkdir(parents=True, exist_ok=True)
        (mask_dir / "SKILL.md").write_text("# Simple Mask\n\nNo metadata.")

        metadata = palace.get_mask_metadata("simple-mask")
        assert metadata is not None
        assert metadata.get("name") == "simple-mask"


class TestMaskComposition:
    """Test mask composition functionality"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance with multiple composable masks"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Create several masks for composition
        masks_dir = palace.palace_dir / "masks" / "available"

        # Mask 1: TDD Expert
        tdd_dir = masks_dir / "tdd-expert"
        tdd_dir.mkdir(parents=True, exist_ok=True)
        (tdd_dir / "SKILL.md").write_text("""---
name: tdd-expert
priority: 1
---

# TDD Expert

## Core Principles
- Write tests first
- Red-Green-Refactor cycle
""")

        # Mask 2: Python Expert
        python_dir = masks_dir / "python-expert"
        python_dir.mkdir(parents=True, exist_ok=True)
        (python_dir / "SKILL.md").write_text("""---
name: python-expert
priority: 2
---

# Python Expert

## Core Expertise
- Type hints
- Pythonic patterns
""")

        # Mask 3: Security Expert
        security_dir = masks_dir / "security-expert"
        security_dir.mkdir(parents=True, exist_ok=True)
        (security_dir / "SKILL.md").write_text("""# Security Expert

## Security Principles
- Input validation
- Secure defaults
""")

        yield palace

    def test_compose_masks_merge_strategy(self, temp_palace):
        """Compose masks with merge strategy"""
        composed = temp_palace.compose_masks(
            ["tdd-expert", "python-expert"],
            strategy="merge"
        )

        assert composed is not None
        assert "TDD Expert" in composed
        assert "Python Expert" in composed
        assert "Mask: tdd-expert" in composed
        assert "Mask: python-expert" in composed

    def test_compose_masks_layer_strategy(self, temp_palace):
        """Compose masks with layer strategy (priority-based)"""
        composed = temp_palace.compose_masks(
            ["tdd-expert", "python-expert"],
            strategy="layer"
        )

        assert composed is not None
        assert "Layer: tdd-expert" in composed
        assert "Layer: python-expert" in composed
        # Lower priority number should come first
        tdd_pos = composed.find("tdd-expert")
        python_pos = composed.find("python-expert")
        assert tdd_pos < python_pos

    def test_compose_masks_blend_strategy(self, temp_palace):
        """Compose masks with blend strategy"""
        composed = temp_palace.compose_masks(
            ["tdd-expert", "security-expert"],
            strategy="blend"
        )

        assert composed is not None
        assert "TDD Expert" in composed
        assert "Security Expert" in composed
        assert "From tdd-expert" in composed or "From security-expert" in composed

    def test_compose_masks_with_missing_mask(self, temp_palace):
        """Composing with missing mask returns None"""
        composed = temp_palace.compose_masks(
            ["tdd-expert", "nonexistent"],
            strategy="merge"
        )

        assert composed is None

    def test_compose_masks_empty_list(self, temp_palace):
        """Composing empty list returns None"""
        composed = temp_palace.compose_masks([], strategy="merge")
        assert composed is None

    def test_build_prompt_with_masks(self, temp_palace):
        """Build prompt with multiple masks composed"""
        prompt = "Write a secure Python web app"
        full_prompt = temp_palace.build_prompt_with_masks(
            prompt,
            ["python-expert", "security-expert"],
            strategy="merge"
        )

        assert full_prompt is not None
        assert "Python Expert" in full_prompt
        assert "Security Expert" in full_prompt
        assert "Write a secure Python web app" in full_prompt

    def test_build_prompt_with_masks_invalid(self, temp_palace):
        """Build prompt with invalid mask in list returns None"""
        prompt = "Write a secure Python web app"
        full_prompt = temp_palace.build_prompt_with_masks(
            prompt,
            ["python-expert", "nonexistent"],
            strategy="merge"
        )

        assert full_prompt is None

    def test_split_mask_into_sections(self, temp_palace):
        """Split mask content into sections by headers"""
        content = """# Section 1
Content 1

## Subsection 1.1
More content

# Section 2
Content 2"""

        sections = temp_palace._split_mask_into_sections(content)
        # Should split on ANY header (# or ##)
        assert len(sections) >= 3
        assert "Section 1" in sections[0]
        # One of the sections should have Section 2
        assert any("Section 2" in s for s in sections)
