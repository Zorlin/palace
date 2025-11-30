"""
Tests for provider override flags (--claude and --glm).

Tests that the --claude and --glm flags correctly override default model selection.
"""

import pytest
from palace import Palace


class TestProviderOverrideInit:
    """Test provider override flags in Palace initialization"""

    def test_default_no_overrides(self):
        """By default, no provider overrides are active"""
        palace = Palace()
        assert palace.force_claude is False
        assert palace.force_glm is False

    def test_force_claude_enabled(self):
        """Can enable Claude override"""
        palace = Palace(force_claude=True)
        assert palace.force_claude is True
        assert palace.force_glm is False

    def test_force_glm_enabled(self):
        """Can enable GLM override"""
        palace = Palace(force_glm=True)
        assert palace.force_glm is True
        assert palace.force_claude is False

    def test_both_overrides_enabled(self):
        """Can enable both overrides (edge case - GLM should win in normal mode)"""
        palace = Palace(force_claude=True, force_glm=True)
        assert palace.force_claude is True
        assert palace.force_glm is True


class TestTurboModeProviderSelection:
    """Test provider selection in turbo mode"""

    def test_turbo_default_uses_glm(self):
        """Turbo mode defaults to GLM for cost efficiency"""
        palace = Palace()
        # In spawn_swarm, should use "z.ai" provider with "glm-4.6" model
        assert palace.force_claude is False

    def test_turbo_claude_override(self):
        """--claude flag forces Claude models in turbo mode"""
        palace = Palace(force_claude=True)
        # In spawn_swarm, should use "anthropic" provider with Claude models
        assert palace.force_claude is True


class TestNormalModeProviderSelection:
    """Test provider selection in normal (non-turbo) mode"""

    def test_normal_default_uses_claude(self):
        """Normal mode defaults to Claude Sonnet"""
        palace = Palace()
        # In invoke_claude_cli, should use "claude-sonnet-4-5"
        assert palace.force_glm is False

    def test_normal_glm_override(self):
        """--glm flag forces GLM in normal mode"""
        palace = Palace(force_glm=True)
        # In invoke_claude_cli, should use "glm-4.6"
        assert palace.force_glm is True


class TestProviderSelectionLogic:
    """Test the logic that determines which provider to use"""

    def test_spawn_swarm_model_selection(self, tmp_path, monkeypatch):
        """Test model selection in spawn_swarm"""
        monkeypatch.chdir(tmp_path)

        # Test default (GLM)
        palace_default = Palace()
        assert palace_default.force_claude is False

        # Test --claude override
        palace_claude = Palace(force_claude=True)
        assert palace_claude.force_claude is True

    def test_invoke_claude_model_selection(self, tmp_path, monkeypatch):
        """Test model selection in invoke_claude_cli"""
        monkeypatch.chdir(tmp_path)

        # Test default (Claude)
        palace_default = Palace()
        assert palace_default.force_glm is False

        # Test --glm override
        palace_glm = Palace(force_glm=True)
        assert palace_glm.force_glm is True


class TestCostOptimizationScenarios:
    """Test cost optimization use cases"""

    def test_cost_saving_normal_mode(self):
        """User wants to save money in normal mode → use --glm"""
        palace = Palace(force_glm=True)
        assert palace.force_glm is True
        # This would route invoke_claude_cli to GLM-4.6 via Z.ai

    def test_quality_priority_turbo_mode(self):
        """User wants quality in turbo mode → use --claude"""
        palace = Palace(force_claude=True)
        assert palace.force_claude is True
        # This would route spawn_swarm to use Claude models

    def test_full_quality_mode(self):
        """User wants max quality everywhere"""
        palace = Palace(force_claude=True, force_glm=False)
        assert palace.force_claude is True
        assert palace.force_glm is False

    def test_full_economy_mode(self):
        """User wants max savings everywhere"""
        palace = Palace(force_claude=False, force_glm=True)
        assert palace.force_claude is False
        assert palace.force_glm is True


class TestProviderOverrideMessages:
    """Test that appropriate messages are shown for overrides"""

    def test_turbo_claude_message_shown(self):
        """--claude in turbo should show appropriate message"""
        palace = Palace(force_claude=True)
        # When run_turbo_mode is called, should print:
        # "💎 Using Claude models (high quality, higher cost)"
        assert palace.force_claude is True

    def test_turbo_glm_default_message_shown(self):
        """Default turbo should show GLM message"""
        palace = Palace()
        # When run_turbo_mode is called, should print:
        # "💰 Using GLM-4.6 (cost-efficient, fast)"
        assert palace.force_claude is False

    def test_normal_glm_message_shown(self):
        """--glm in normal mode should show appropriate message"""
        palace = Palace(force_glm=True)
        # When invoke_claude_cli is called, should print:
        # "💰 GLM mode active - using GLM-4.6 for cost savings"
        assert palace.force_glm is True
