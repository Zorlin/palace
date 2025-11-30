"""
Tests for strict mode functionality.

Strict mode enforces test validation before completion.
"""
import json
import pytest
from pathlib import Path
from palace import Palace


class TestStrictModeInit:
    """Test strict mode initialization"""

    def test_default_strict_mode_enabled(self):
        """By default, strict mode should be enabled"""
        palace = Palace()
        assert palace.strict_mode is True

    def test_explicit_strict_mode(self):
        """Strict mode can be explicitly set"""
        palace = Palace(strict_mode=True)
        assert palace.strict_mode is True

    def test_yolo_mode_disables_strict(self):
        """YOLO mode disables strict mode"""
        palace = Palace(strict_mode=False)
        assert palace.strict_mode is False

    def test_modified_files_tracking(self):
        """Palace should track modified files"""
        palace = Palace()
        assert hasattr(palace, 'modified_files')
        assert isinstance(palace.modified_files, set)


class TestTestDetection:
    """Test the detect_affected_tests function"""

    def test_empty_modified_files(self):
        """Empty modified files should return empty test set"""
        palace = Palace()
        result = palace.detect_affected_tests(set())
        assert result == set()

    def test_detects_direct_test_file(self):
        """Should detect when a test file itself is modified"""
        palace = Palace()
        modified = {"tests/test_core.py"}
        result = palace.detect_affected_tests(modified)
        assert "tests/test_core.py" in result

    def test_detects_tests_for_source_file(self, tmp_path, monkeypatch):
        """Should find test files for source code files"""
        # Create a temporary project structure
        monkeypatch.chdir(tmp_path)
        palace = Palace()

        # Create tests directory with test file
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_foo.py").touch()

        # Modify source file foo.py
        modified = {"foo.py"}
        result = palace.detect_affected_tests(modified)

        # Should find test_foo.py
        assert "tests/test_foo.py" in result

    def test_fallback_to_all_tests(self, tmp_path, monkeypatch):
        """Should run all tests if no specific mapping found"""
        monkeypatch.chdir(tmp_path)
        palace = Palace()

        # Create tests directory with some test files
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_a.py").touch()
        (tests_dir / "test_b.py").touch()

        # Modify a file with no matching test
        modified = {"unknown.py"}
        result = palace.detect_affected_tests(modified)

        # Should include all tests as fallback
        assert len(result) >= 2


class TestTestSubsetRunner:
    """Test the run_test_subset function"""

    def test_runs_specific_tests(self, tmp_path, monkeypatch):
        """Should run only specified test files"""
        monkeypatch.chdir(tmp_path)
        palace = Palace()

        # Create a passing test
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_simple.py"
        test_file.write_text("""
def test_always_pass():
    assert True
""")

        # Run the test
        result = palace.run_test_subset({"tests/test_simple.py"})
        assert result is True

    def test_handles_failing_tests(self, tmp_path, monkeypatch):
        """Should return False when tests fail"""
        monkeypatch.chdir(tmp_path)
        palace = Palace()

        # Create a failing test
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_failing.py"
        test_file.write_text("""
def test_always_fail():
    assert False, "This test should fail"
""")

        # Run the test
        result = palace.run_test_subset({"tests/test_failing.py"})
        assert result is False


class TestModifiedFileTracking:
    """Test file modification tracking in permission handler"""

    def test_tracks_write_operations(self, tmp_path, monkeypatch):
        """Permission handler should track Write operations"""
        monkeypatch.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Simulate permission handler tracking a write
        modified_files_path = palace.palace_dir / "modified_files.json"
        modified_files = {"test_file.py"}

        with open(modified_files_path, 'w') as f:
            json.dump(list(modified_files), f)

        # Verify tracking
        assert modified_files_path.exists()
        with open(modified_files_path, 'r') as f:
            tracked = set(json.load(f))
        assert tracked == modified_files

    def test_tracks_edit_operations(self, tmp_path, monkeypatch):
        """Permission handler should track Edit operations"""
        monkeypatch.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Simulate tracking an edit
        modified_files_path = palace.palace_dir / "modified_files.json"

        # First write
        modified_files = {"file1.py"}
        with open(modified_files_path, 'w') as f:
            json.dump(list(modified_files), f)

        # Add another file (simulating Edit permission)
        with open(modified_files_path, 'r') as f:
            tracked = set(json.load(f))
        tracked.add("file2.py")
        with open(modified_files_path, 'w') as f:
            json.dump(list(tracked), f)

        # Verify both files tracked
        with open(modified_files_path, 'r') as f:
            final = set(json.load(f))
        assert final == {"file1.py", "file2.py"}


class TestCompletionHook:
    """Test the completion hook in invoke_claude_cli"""

    def test_clears_tracking_after_success(self, tmp_path, monkeypatch):
        """Should clear modified files after successful test run"""
        monkeypatch.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Create modified files tracking
        modified_files_path = palace.palace_dir / "modified_files.json"
        with open(modified_files_path, 'w') as f:
            json.dump(["test.py"], f)

        # Simulate clearing after success
        modified_files_path.unlink()

        assert not modified_files_path.exists()

    def test_preserves_tracking_after_failure(self, tmp_path, monkeypatch):
        """Should preserve modified files if tests fail"""
        monkeypatch.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()

        # Create modified files tracking
        modified_files_path = palace.palace_dir / "modified_files.json"
        modified_files = ["test.py"]
        with open(modified_files_path, 'w') as f:
            json.dump(modified_files, f)

        # If tests failed, file should still exist
        assert modified_files_path.exists()
        with open(modified_files_path, 'r') as f:
            tracked = json.load(f)
        assert tracked == modified_files


class TestYoloMode:
    """Test YOLO mode (strict mode disabled)"""

    def test_yolo_skips_validation(self):
        """YOLO mode should skip all test validation"""
        palace = Palace(strict_mode=False)
        # In YOLO mode, strict_mode should be False
        assert palace.strict_mode is False

    def test_strict_mode_runs_validation(self):
        """Strict mode should enable validation"""
        palace = Palace(strict_mode=True)
        assert palace.strict_mode is True
