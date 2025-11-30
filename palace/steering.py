"""
User steering and interrupt handling for Palace
"""

import time
from typing import Optional, Dict, Any


class SteeringSystem:
    """Handles ESC-ESC interrupts and user steering"""

    def __init__(self):
        self._last_esc_time = None
        self._escape_timeout = 0.5  # 500ms window for double-tap

    def _setup_escape_handler(self):
        """
        Initialize escape sequence detection state.

        Call this before starting stream processing to enable ESC-ESC detection.
        """
        self._last_esc_time = None
        self._escape_timeout = 0.5  # 500ms window for double-tap

    def _check_escape_sequence(self, char: str) -> Optional[str]:
        """
        Check if character is part of ESC-ESC sequence.

        Returns:
        - "first_esc": First ESC detected, waiting for second
        - "interrupt": ESC-ESC sequence completed, trigger interrupt
        - None: Not an ESC key
        """
        if char != chr(27):  # ESC character
            return None

        current_time = time.time()

        if self._last_esc_time is None:
            # First ESC
            self._last_esc_time = current_time
            return "first_esc"

        # Check if within timeout window
        elapsed = current_time - self._last_esc_time
        if elapsed <= self._escape_timeout:
            # Double-tap detected!
            self._last_esc_time = None
            return "interrupt"
        else:
            # Too slow, reset and treat as new first ESC
            self._last_esc_time = current_time
            return "first_esc"

    def _check_for_escape(self) -> bool:
        """
        Non-blocking check for ESC keypress.

        Returns True if ESC-ESC was detected.
        Uses select() to check stdin without blocking.
        """
        import select
        import sys
        import tty
        import termios
        import fcntl
        import os

        # Only works in TTY
        if not sys.stdin.isatty():
            return False

        fd = sys.stdin.fileno()

        try:
            # Save current terminal settings and flags
            old_settings = termios.tcgetattr(fd)
            old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)

            try:
                # Set non-blocking
                fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)

                # Set terminal to cbreak mode (char by char, no echo)
                new_settings = termios.tcgetattr(fd)
                new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO)
                new_settings[6][termios.VMIN] = 0
                new_settings[6][termios.VTIME] = 0
                termios.tcsetattr(fd, termios.TCSANOW, new_settings)

                # Try to read
                try:
                    char = os.read(fd, 1)
                    if char == b'\x1b':  # ESC
                        result = self._check_escape_sequence(chr(27))
                        return result == "interrupt"
                except (BlockingIOError, OSError):
                    pass

            finally:
                # Restore terminal settings
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)

        except Exception:
            pass

        return False

    def _handle_user_interrupt(self) -> Optional[Dict[str, Any]]:
        """
        Handle user interrupt (ESC-ESC).

        Shows steering prompt and returns user input.

        Returns dict with:
        - action: "steer", "resume", or "abort"
        - steering: User's steering text (if action is "steer")

        Returns None if user cancels (empty input).
        """
        print("\n")
        print("⏸️  " + "─" * 50)
        print("   PALACE PAUSED - Enter steering command")
        print("   (Press Enter to resume, /abort to stop)")
        print("   " + "─" * 50)
        print()

        try:
            steering = input("🎯 Steer: ").strip()

            if not steering:
                # Empty input = resume normally
                print("▶️  Resuming...")
                return {"action": "resume"}

            if steering.lower() == "/abort":
                print("🛑 Aborting session...")
                return {"action": "abort"}

            # Log the steering if log_steering method exists
            if hasattr(self, 'log_steering'):
                self.log_steering(steering)

            print(f"✅ Steering applied: {steering[:50]}...")
            return {"action": "steer", "steering": steering}

        except (EOFError, KeyboardInterrupt):
            print("\n▶️  Resuming...")
            return {"action": "resume"}

    def log_steering(self, steering: str):
        """Log user steering to history for learning"""
        if hasattr(self, 'log_action'):
            self.log_action("user_steering", {"steering": steering})
