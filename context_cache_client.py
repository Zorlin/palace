#!/usr/bin/env python3
"""
Python client for the Palace Context Cache daemon.

Communicates via Unix socket for near-zero latency context management.

Usage:
    from context_cache_client import ContextCacheClient

    client = ContextCacheClient()
    if client.is_available():
        # Register a new context block
        block_id = client.register("CurrentFile", "Working on palace.py router code")

        # Activate blocks
        client.apply_delta("++1,2,3--4,5,6")

        # Get active context for injection
        context = client.get_context()

        # Get classifier input for 1b model
        classifier_input = client.get_classifier_input()
"""

import socket
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

# Default socket path (matches Rust daemon)
SOCKET_PATH = "/tmp/palace-context-cache.sock"


class ContextCacheClient:
    """Client for the Rust context cache daemon."""

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self._socket: Optional[socket.socket] = None

    def is_available(self) -> bool:
        """Check if the daemon is running and accessible."""
        if not os.path.exists(self.socket_path):
            return False
        try:
            response = self._send("PING")
            return response == "PONG"
        except Exception:
            return False

    def _connect(self) -> socket.socket:
        """Connect to the daemon, reusing connection if possible."""
        if self._socket is None:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(self.socket_path)
            self._socket.settimeout(5.0)  # 5 second timeout
        return self._socket

    def _send(self, command: str) -> str:
        """Send a command and get the response."""
        try:
            sock = self._connect()
            sock.sendall((command + "\n").encode())

            # Read response (single line)
            response = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
                if b"\n" in response:
                    break

            return response.decode().strip()
        except Exception as e:
            # Close socket on error, will reconnect on next call
            self._close()
            raise

    def _close(self):
        """Close the socket connection."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def apply_delta(self, delta: str) -> Dict[str, Any]:
        """
        Apply a context delta.

        Args:
            delta: Delta string like "++1,2,3--4,5,6"

        Returns:
            Dict with active count on success

        Raises:
            Exception on error
        """
        response = self._send(f"DELTA {delta}")
        if response.startswith("OK"):
            # Parse "OK active=N"
            parts = response.split()
            result = {"success": True}
            for part in parts[1:]:
                if "=" in part:
                    key, value = part.split("=", 1)
                    result[key] = int(value) if value.isdigit() else value
            return result
        else:
            raise Exception(response)

    def get_active_ids(self) -> List[int]:
        """Get the list of currently active block IDs."""
        response = self._send("GET_ACTIVE")
        if response.startswith("OK"):
            ids_str = response[3:].strip()
            if not ids_str:
                return []
            return [int(x) for x in ids_str.split(",") if x.strip()]
        else:
            raise Exception(response)

    def get_context(self) -> str:
        """Get the full active context as a string."""
        response = self._send("GET_CONTEXT")
        if response.startswith("OK"):
            # Unescape newlines
            content = response[3:].strip()
            return content.replace("\\n", "\n")
        else:
            raise Exception(response)

    def get_classifier_input(self) -> str:
        """Get input formatted for the 1b classifier."""
        response = self._send("GET_CLASSIFIER_INPUT")
        if response.startswith("OK"):
            content = response[3:].strip()
            return content.replace("\\n", "\n")
        else:
            raise Exception(response)

    def register(self, block_type: str, summary: str) -> int:
        """
        Register a new context block.

        Args:
            block_type: Type like "CurrentFile", "Error", "UserGoal", etc.
            summary: Short description for classifier

        Returns:
            Block ID
        """
        response = self._send(f"REGISTER {block_type} {summary}")
        if response.startswith("OK"):
            return int(response[3:].strip())
        else:
            raise Exception(response)

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        response = self._send("STATS")
        if response.startswith("OK"):
            result = {}
            for part in response[3:].strip().split():
                if "=" in part:
                    key, value = part.split("=", 1)
                    result[key] = int(value) if value.isdigit() else value
            return result
        else:
            raise Exception(response)

    def ping(self) -> bool:
        """Health check."""
        try:
            response = self._send("PING")
            return response == "PONG"
        except Exception:
            return False


class ContextManager:
    """
    Higher-level context management for Palace router.

    Wraps ContextCacheClient with convenient methods for:
    - Registering model responses, tool calls, errors
    - Injecting active context into prompts
    - Running the 1b classifier to manage context window
    """

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.client = ContextCacheClient(socket_path)
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        """Check if context cache is available (cached result)."""
        if self._available is None:
            self._available = self.client.is_available()
        return self._available

    def record_user_message(self, content: str, model_target: Optional[str] = None) -> Optional[int]:
        """Record a user message as a context block."""
        if not self.available:
            return None
        try:
            summary = content[:100] + "..." if len(content) > 100 else content
            if model_target:
                summary = f"@{model_target}: {summary}"
            return self.client.register("UserGoal", summary)
        except Exception:
            return None

    def record_model_response(self, model: str, content: str) -> Optional[int]:
        """Record a model response as a context block."""
        if not self.available:
            return None
        try:
            summary = f"@{model}: {content[:80]}..." if len(content) > 80 else f"@{model}: {content}"
            return self.client.register("ModelResponse", summary)
        except Exception:
            return None

    def record_error(self, error: str) -> Optional[int]:
        """Record an error as a context block."""
        if not self.available:
            return None
        try:
            summary = error[:100] + "..." if len(error) > 100 else error
            return self.client.register("Error", summary)
        except Exception:
            return None

    def record_tool_call(self, tool_name: str, result_summary: str) -> Optional[int]:
        """Record a tool call result as a context block."""
        if not self.available:
            return None
        try:
            summary = f"{tool_name}: {result_summary[:80]}"
            return self.client.register("ToolResult", summary)
        except Exception:
            return None

    def get_active_context(self) -> Optional[str]:
        """Get the current active context for prompt injection."""
        if not self.available:
            return None
        try:
            return self.client.get_context()
        except Exception:
            return None

    def get_classifier_input(self) -> Optional[str]:
        """Get input for the 1b classifier."""
        if not self.available:
            return None
        try:
            return self.client.get_classifier_input()
        except Exception:
            return None

    def apply_classifier_output(self, delta: str) -> bool:
        """Apply the output from the 1b classifier."""
        if not self.available:
            return False
        try:
            self.client.apply_delta(delta)
            return True
        except Exception:
            return False

    def activate_blocks(self, block_ids: List[int]) -> bool:
        """Activate specific blocks by ID."""
        if not self.available or not block_ids:
            return False
        try:
            delta = "++" + ",".join(str(x) for x in block_ids)
            self.client.apply_delta(delta)
            return True
        except Exception:
            return False

    def deactivate_blocks(self, block_ids: List[int]) -> bool:
        """Deactivate specific blocks by ID."""
        if not self.available or not block_ids:
            return False
        try:
            delta = "--" + ",".join(str(x) for x in block_ids)
            self.client.apply_delta(delta)
            return True
        except Exception:
            return False


# Singleton instance for easy access
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get the global context manager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


if __name__ == "__main__":
    # Quick test
    client = ContextCacheClient()

    if client.is_available():
        print("Context cache daemon is running!")

        # Test stats
        stats = client.stats()
        print(f"Stats: {stats}")

        # Test register
        block_id = client.register("CurrentFile", "Testing from Python client")
        print(f"Registered block: {block_id}")

        # Test activate
        result = client.apply_delta(f"++{block_id}")
        print(f"Activated: {result}")

        # Test get active
        active = client.get_active_ids()
        print(f"Active blocks: {active}")

        # Test classifier input
        input_text = client.get_classifier_input()
        print(f"Classifier input:\n{input_text}")
    else:
        print("Context cache daemon is not running.")
        print(f"Start it with: ./palace-context-cache/target/release/palace-cache-daemon")
