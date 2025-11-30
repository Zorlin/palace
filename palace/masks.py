"""
Mask system for Palace - allows loading and composing Claude expertise
"""

from pathlib import Path
from typing import Optional, Dict, Any, List


class MaskSystem:
    """Handles Palace mask loading and composition"""

    def __init__(self, palace_dir: Path):
        self.palace_dir = palace_dir

    def load_mask(self, mask_name: str) -> Optional[str]:
        """
        Load a mask from .palace/masks/

        Searches in order:
        1. .palace/masks/available/{mask_name}/SKILL.md
        2. .palace/masks/custom/{mask_name}/SKILL.md

        Returns mask content or None if not found.
        """
        # Try available masks first
        mask_file = self.palace_dir / "masks" / "available" / mask_name / "SKILL.md"
        if mask_file.exists():
            return mask_file.read_text()

        # Try custom masks
        mask_file = self.palace_dir / "masks" / "custom" / mask_name / "SKILL.md"
        if mask_file.exists():
            return mask_file.read_text()

        return None

    def get_mask_metadata(self, mask_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from mask frontmatter.

        Parses YAML frontmatter if present, otherwise returns minimal metadata.
        """
        content = self.load_mask(mask_name)
        if not content:
            return None

        metadata = {"name": mask_name}

        # Check for frontmatter (--- at start)
        if content.startswith("---"):
            try:
                # Extract frontmatter block
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter_text = parts[1].strip()
                    # Simple YAML-like parsing (key: value)
                    for line in frontmatter_text.split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            metadata[key.strip()] = value.strip()
            except:
                pass

        return metadata

    def list_masks(self) -> List[Dict[str, Any]]:
        """
        List all available masks.

        Returns list of dicts with:
        - name: mask name
        - type: "available" or "custom"
        - path: full path to mask
        """
        masks = []
        masks_dir = self.palace_dir / "masks"

        if not masks_dir.exists():
            return masks

        # List available masks
        available_dir = masks_dir / "available"
        if available_dir.exists():
            for mask_dir in available_dir.iterdir():
                if mask_dir.is_dir() and (mask_dir / "SKILL.md").exists():
                    masks.append({
                        "name": mask_dir.name,
                        "type": "available",
                        "path": str(mask_dir / "SKILL.md")
                    })

        # List custom masks
        custom_dir = masks_dir / "custom"
        if custom_dir.exists():
            for mask_dir in custom_dir.iterdir():
                if mask_dir.is_dir() and (mask_dir / "SKILL.md").exists():
                    masks.append({
                        "name": mask_dir.name,
                        "type": "custom",
                        "path": str(mask_dir / "SKILL.md")
                    })

        return masks

    def compose_masks(self, mask_names: List[str], strategy: str = "merge") -> Optional[str]:
        """
        Compose multiple masks together.

        Strategies:
        - "merge": Concatenate masks in order with separators
        - "layer": Apply masks hierarchically (later masks override earlier)
        - "blend": Interleave sections from each mask

        Returns composed mask content or None if any mask not found.
        """
        if not mask_names:
            return None

        # Load all masks
        mask_contents = []
        for name in mask_names:
            content = self.load_mask(name)
            if not content:
                return None  # Fail if any mask is missing
            mask_contents.append((name, content))

        if strategy == "merge":
            # Simple concatenation with clear separators
            parts = []
            for name, content in mask_contents:
                parts.append(f"# Mask: {name}")
                parts.append(content)
                parts.append("")  # Blank line separator
            return "\n".join(parts)

        elif strategy == "layer":
            # Later masks override earlier ones
            # Use frontmatter priority field if available
            layered_parts = []
            for name, content in mask_contents:
                metadata = self.get_mask_metadata(name)
                priority = int(metadata.get("priority", 0)) if metadata else 0
                layered_parts.append((priority, name, content))

            # Sort by priority (lower = higher precedence)
            layered_parts.sort(key=lambda x: x[0])

            # Build layered content
            result = []
            for _, name, content in layered_parts:
                result.append(f"# Layer: {name}")
                result.append(content)
                result.append("")
            return "\n".join(result)

        elif strategy == "blend":
            # Interleave sections from each mask
            # Split each mask into sections (by headers)
            sections = []
            for name, content in mask_contents:
                mask_sections = self._split_mask_into_sections(content)
                sections.extend([(name, s) for s in mask_sections])

            # Interleave sections
            blended = []
            for name, section in sections:
                blended.append(f"<!-- From {name} -->")
                blended.append(section)
                blended.append("")
            return "\n".join(blended)

        return None

    def _split_mask_into_sections(self, content: str) -> List[str]:
        """
        Split mask content into sections by markdown headers.

        Returns list of section strings.
        """
        sections = []
        current_section = []

        for line in content.split("\n"):
            if line.startswith("#") and current_section:
                # Start new section
                sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        # Add final section
        if current_section:
            sections.append("\n".join(current_section))

        return sections

    def build_prompt_with_mask(self, task_prompt: str, mask_name: Optional[str] = None,
                               context: Dict[str, Any] = None, build_prompt_fn=None) -> Optional[str]:
        """
        Build prompt with optional mask loaded.

        If mask_name is provided, prepends mask content to the prompt.
        Returns None if mask not found.
        """
        # Load mask if specified
        mask_content = None
        if mask_name:
            mask_content = self.load_mask(mask_name)
            if not mask_content:
                return None

        # Build base prompt (use provided function or assume build_prompt method exists)
        if build_prompt_fn:
            base_prompt = build_prompt_fn(task_prompt, context)
        elif hasattr(self, 'build_prompt'):
            base_prompt = self.build_prompt(task_prompt, context)
        else:
            # Fallback: just use task_prompt
            base_prompt = task_prompt

        # Prepend mask content if loaded
        if mask_content:
            return f"{mask_content}\n\n{base_prompt}"

        return base_prompt

    def build_prompt_with_masks(self, task_prompt: str, mask_names: List[str],
                                strategy: str = "merge",
                                context: Dict[str, Any] = None, build_prompt_fn=None) -> Optional[str]:
        """
        Build prompt with multiple masks composed together.

        Args:
            task_prompt: The task description
            mask_names: List of mask names to compose
            strategy: Composition strategy ("merge", "layer", "blend")
            context: Optional context dict
            build_prompt_fn: Optional function to build base prompt

        Returns composed prompt or None if any mask not found.
        """
        if not mask_names:
            if build_prompt_fn:
                return build_prompt_fn(task_prompt, context)
            elif hasattr(self, 'build_prompt'):
                return self.build_prompt(task_prompt, context)
            else:
                return task_prompt

        # Compose masks
        composed_content = self.compose_masks(mask_names, strategy)
        if not composed_content:
            return None

        # Build base prompt
        if build_prompt_fn:
            base_prompt = build_prompt_fn(task_prompt, context)
        elif hasattr(self, 'build_prompt'):
            base_prompt = self.build_prompt(task_prompt, context)
        else:
            base_prompt = task_prompt

        # Prepend composed mask content
        return f"{composed_content}\n\n{base_prompt}"
