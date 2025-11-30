"""
Core Palace class with configuration and session management
"""

import json
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List


class Palace:
    """Palace orchestration layer - coordinates Claude invocations"""

    def __init__(self, strict_mode: bool = True, force_claude: bool = False, force_glm: bool = False) -> None:
        self.project_root = Path.cwd()
        self.palace_dir = self.project_root / ".palace"
        self.config_file = self.palace_dir / "config.json"
        self.strict_mode = strict_mode
        self.modified_files = set()  # Track files modified during execution
        self.force_claude = force_claude  # Use Claude even in turbo mode
        self.force_glm = force_glm  # Use GLM even in normal mode

    def ensure_palace_dir(self) -> None:
        """Ensure .palace directory exists"""
        self.palace_dir.mkdir(exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        """Load Palace configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save Palace configuration"""
        self.ensure_palace_dir()
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def gather_context(self) -> Dict[str, Any]:
        """Gather project context for Claude"""
        import subprocess
        from palace import VERSION

        context = {
            "project_root": str(self.project_root),
            "palace_version": VERSION,
            "files": {},
            "git_status": None,
            "config": self.load_config()
        }

        # Check for important files
        important_files = ["README.md", "SPEC.md", "ROADMAP.md", "package.json",
                          "requirements.txt", "Cargo.toml", "go.mod"]

        for filename in important_files:
            filepath = self.project_root / filename
            if filepath.exists():
                context["files"][filename] = {
                    "exists": True,
                    "size": filepath.stat().st_size
                }

        # Get git status if in a repo
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                context["git_status"] = result.stdout
        except:
            pass

        # Load history
        history_file = self.palace_dir / "history.jsonl"
        if history_file.exists():
            context["recent_history"] = []
            with open(history_file, 'r') as f:
                for line in f:
                    if line.strip():
                        context["recent_history"].append(json.loads(line))
            context["recent_history"] = context["recent_history"][-10:]  # Last 10

        return context

    def log_action(self, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log action to history"""
        self.ensure_palace_dir()
        history_file = self.palace_dir / "history.jsonl"

        entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details or {}
        }

        with open(history_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    # ========================================================================
    # Session Management
    # ========================================================================

    def _generate_session_id(self) -> str:
        """Generate a short, memorable session ID"""
        # Use short UUID prefix + timestamp suffix for uniqueness
        return f"pal-{uuid.uuid4().hex[:6]}"

    def _get_sessions_dir(self) -> Path:
        """Get the sessions directory"""
        sessions_dir = self.palace_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return sessions_dir

    def save_session(self, session_id: str, state: Dict[str, Any]) -> Path:
        """Save session state for later resumption"""
        sessions_dir = self._get_sessions_dir()
        session_file = sessions_dir / f"{session_id}.json"

        state["session_id"] = session_id
        state["updated_at"] = time.time()

        with open(session_file, 'w') as f:
            json.dump(state, f, indent=2)

        return session_file

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a saved session state"""
        sessions_dir = self._get_sessions_dir()
        session_file = sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            return None

        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions"""
        sessions_dir = self._get_sessions_dir()
        sessions = []

        for session_file in sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    state = json.load(f)
                    sessions.append({
                        "session_id": state.get("session_id", session_file.stem),
                        "updated_at": state.get("updated_at"),
                        "iteration": state.get("iteration", 0),
                        "pending_actions": len(state.get("pending_actions", []))
                    })
            except:
                pass

        return sorted(sessions, key=lambda s: s.get("updated_at", 0), reverse=True)

    def export_session(self, session_id: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Export a session to a portable JSON file.

        Includes full session state, history, and metadata for sharing or backup.

        Args:
            session_id: The session ID to export
            output_path: Optional output file path (defaults to current dir)

        Returns path to exported file or None if session not found.
        """
        from palace import VERSION

        session = self.load_session(session_id)
        if not session:
            return None

        # Build export bundle
        export_data = {
            "version": "1.0",
            "exported_at": time.time(),
            "palace_version": VERSION,
            "session": session
        }

        # Add relevant history entries for this session
        history_entries = []
        history_file = self.palace_dir / "history.jsonl"
        if history_file.exists():
            for line in history_file.read_text().strip().split('\n'):
                try:
                    entry = json.loads(line)
                    # Include entries from this session
                    if entry.get("details", {}).get("session_id") == session_id:
                        history_entries.append(entry)
                except:
                    pass

        export_data["history"] = history_entries

        # Determine output path
        if not output_path:
            output_path = f"{session_id}_export.json"

        # Write export file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        return output_path

    def import_session(self, import_path: str, new_session_id: Optional[str] = None) -> Optional[str]:
        """
        Import a session from an exported JSON file.

        Args:
            import_path: Path to the exported session file
            new_session_id: Optional new session ID (generates one if not provided)

        Returns the imported session ID or None if import failed.
        """
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)

            # Validate export format
            if "session" not in import_data:
                return None

            session = import_data["session"]

            # Generate new session ID if not provided
            if not new_session_id:
                new_session_id = self._generate_session_id()

            # Update session ID
            session["session_id"] = new_session_id
            session["imported_at"] = time.time()
            session["imported_from"] = import_path

            # Save imported session
            self.save_session(new_session_id, session)

            # Import history entries if present
            if "history" in import_data and import_data["history"]:
                self.ensure_palace_dir()
                history_file = self.palace_dir / "history.jsonl"
                with open(history_file, 'a') as f:
                    for entry in import_data["history"]:
                        # Update session_id in history entries
                        if "details" in entry and isinstance(entry["details"], dict):
                            entry["details"]["session_id"] = new_session_id
                        f.write(json.dumps(entry) + '\n')

            return new_session_id

        except Exception as e:
            return None

    def checkpoint_session(self, session_id: str, state: Dict[str, Any]):
        """
        Checkpoint session state for recovery.

        Similar to save_session but adds checkpoint_at timestamp.
        """
        state["checkpoint_at"] = time.time()
        self.save_session(session_id, state)
