"""
Palace - A self-improving Claude wrapper
Recursive Hierarchical Self Improvement (RHSI) for Claude

Palace is NOT a replacement for Claude - it's an orchestration layer.
Every command invokes Claude with context, letting Claude use its full power.
"""

VERSION = "0.1.0"

# Import main classes for easy access
from palace.main import Palace
from palace.utils import is_interactive

__all__ = ['Palace', 'VERSION', 'is_interactive']
