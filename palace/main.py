"""
Main Palace class combining all functionality
"""

from palace.core import Palace as PalaceCore
from palace.actions import ActionParser
from palace.prompts import PromptBuilder
from palace.masks import MaskSystem
from palace.steering import SteeringSystem
from palace.providers import ProviderSystem
from palace.turbo import TurboMode
from palace.recovery import ErrorRecovery
from palace.stream import StreamProcessor
from palace.strict import StrictMode


class Palace(PalaceCore, ActionParser, PromptBuilder, SteeringSystem,
             ProviderSystem, TurboMode, ErrorRecovery, StreamProcessor, StrictMode):
    """
    Complete Palace class with all functionality.

    Inherits from:
    - PalaceCore: Configuration, sessions, logging
    - ActionParser: Action selection parsing
    - PromptBuilder: Prompt construction
    - MaskSystem: Mask loading and composition (via delegation)
    - SteeringSystem: ESC-ESC interrupts and steering
    - ProviderSystem: Multi-provider support and benchmarking
    - TurboMode: Parallel swarm execution
    - ErrorRecovery: Retry logic and error handling
    - StreamProcessor: Stream processing and output formatting
    """

    def __init__(self, strict_mode: bool = True, force_claude: bool = False, force_glm: bool = False):
        # Initialize core with strict mode and provider overrides
        PalaceCore.__init__(self, strict_mode=strict_mode, force_claude=force_claude, force_glm=force_glm)

        # Initialize other components that need setup
        SteeringSystem.__init__(self)

        # Initialize mask system with palace_dir
        self._mask_system = MaskSystem(self.palace_dir)

    # ========================================================================
    # Mask System Delegation
    # ========================================================================
    # Delegate mask methods to the mask system instance

    def load_mask(self, mask_name):
        return self._mask_system.load_mask(mask_name)

    def get_mask_metadata(self, mask_name):
        return self._mask_system.get_mask_metadata(mask_name)

    def list_masks(self):
        return self._mask_system.list_masks()

    def compose_masks(self, mask_names, strategy="merge"):
        return self._mask_system.compose_masks(mask_names, strategy)

    def _split_mask_into_sections(self, content):
        return self._mask_system._split_mask_into_sections(content)

    def build_prompt_with_mask(self, task_prompt, mask_name=None, context=None):
        return self._mask_system.build_prompt_with_mask(
            task_prompt, mask_name, context, build_prompt_fn=self.build_prompt
        )

    def build_prompt_with_masks(self, task_prompt, mask_names, strategy="merge", context=None):
        return self._mask_system.build_prompt_with_masks(
            task_prompt, mask_names, strategy, context, build_prompt_fn=self.build_prompt
        )

    # ========================================================================
    # Additional helper methods from original palace.py
    # ========================================================================

    def _checkpoint_session_impl(self, session_id, state):
        """Implementation for recovery.checkpoint_session delegation"""
        PalaceCore.checkpoint_session(self, session_id, state)
