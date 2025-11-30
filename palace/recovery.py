"""
Error recovery and retry logic for Palace
"""

import time
from typing import Tuple, Any, Optional


class ErrorRecovery:
    """Handles error detection, retry logic, and graceful degradation"""

    def is_transient_error(self, exit_code: int, error_msg: str = "") -> bool:
        """
        Classify error as transient (retry-able) or permanent.

        Transient errors:
        - Network issues (timeout, connection errors)
        - Rate limits (429)
        - Temporary service issues (503)

        Permanent errors:
        - Permission denied (403)
        - User interrupt (130, KeyboardInterrupt)
        - Invalid input (400)
        """
        if exit_code in [130, 2]:  # SIGINT, user abort
            return False

        if exit_code in [403]:  # Permission denied
            return False

        if exit_code == 429:  # Rate limit
            return True

        # Check error message for hints
        error_lower = error_msg.lower()
        transient_keywords = ["timeout", "network", "connection", "temporary", "503", "502"]
        permanent_keywords = ["permission", "denied", "invalid", "forbidden", "400"]

        if any(kw in error_lower for kw in permanent_keywords):
            return False

        if any(kw in error_lower for kw in transient_keywords):
            return True

        # Default: treat as transient for retry
        return True

    def invoke_with_retry(self, prompt: str, max_retries: int = 3) -> Tuple[int, Optional[Any]]:
        """
        Invoke Claude with exponential backoff retry on transient failures.

        Returns: (exit_code, result/actions)
        """
        attempt = 0
        last_error = None

        while attempt < max_retries:
            if attempt > 0:
                # Exponential backoff: 1s, 2s, 4s, ...
                wait_time = 2 ** (attempt - 1)
                print(f"⏳ Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")

                if hasattr(self, 'log_retry_attempt'):
                    self.log_retry_attempt(
                        attempt=attempt,
                        max_retries=max_retries,
                        error=last_error,
                        wait_time=wait_time
                    )

                time.sleep(wait_time)

            # Apply degradation mode based on attempt
            degradation_mode = self.get_degradation_mode(attempt)

            try:
                if hasattr(self, 'invoke_claude_cli'):
                    exit_code, actions = self.invoke_claude_cli(prompt, degradation_mode=degradation_mode)

                    if exit_code == 0:
                        # Success! Log recovery if this was a retry
                        if attempt > 0 and hasattr(self, 'log_error_recovery'):
                            self.log_error_recovery(
                                error_type="transient_failure",
                                recovered=True,
                                attempts=attempt + 1
                            )
                        return exit_code, actions

                    # Check if error is transient
                    if not self.is_transient_error(exit_code):
                        # Permanent error, don't retry
                        if hasattr(self, 'log_error_recovery'):
                            self.log_error_recovery(
                                error_type="permanent_failure",
                                recovered=False,
                                attempts=attempt + 1
                            )
                        return exit_code, actions

                    last_error = f"exit_code={exit_code}"

                else:
                    # No invoke_claude_cli method, fail immediately
                    return 1, None

            except Exception as e:
                last_error = str(e)
                if not self.is_transient_error(1, last_error):
                    return 1, None

            attempt += 1

        # All retries exhausted
        if hasattr(self, 'log_error_recovery'):
            self.log_error_recovery(
                error_type="retry_exhausted",
                recovered=False,
                attempts=max_retries
            )

        return 1, None

    def get_degradation_mode(self, attempt: int) -> str:
        """
        Get degradation mode based on retry attempt.

        Progressively simplifies the invocation to work around issues:
        - attempt 0: "retry" (normal operation)
        - attempt 1: "no-stream" (disable streaming, full buffer mode)
        - attempt 2: "prompt-file" (write prompt to file, minimal interaction)
        - attempt 3+: "fatal" (give up)

        Returns degradation mode string.
        """
        if attempt == 0:
            return "retry"
        elif attempt == 1:
            return "no-stream"
        elif attempt == 2:
            return "prompt-file"
        else:
            return "fatal"

    def checkpoint_session(self, session_id: str, state: dict):
        """
        Checkpoint session before risky operations.

        (Delegates to core.Palace.checkpoint_session)
        """
        if hasattr(self, '_checkpoint_session_impl'):
            self._checkpoint_session_impl(session_id, state)

    def log_retry_attempt(self, attempt: int, max_retries: int, error: str, wait_time: int):
        """Log retry attempt to history"""
        if hasattr(self, 'log_action'):
            self.log_action("retry_attempt", {
                "attempt": attempt,
                "max_retries": max_retries,
                "error": error,
                "wait_time": wait_time
            })

    def log_error_recovery(self, error_type: str, recovered: bool, attempts: int):
        """Log error recovery outcome"""
        if hasattr(self, 'log_action'):
            self.log_action("error_recovery", {
                "error_type": error_type,
                "recovered": recovered,
                "attempts": attempts
            })
