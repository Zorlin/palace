"""
Strict mode functionality for Palace.

Strict mode enforces test validation at completion time.
Files can still be modified during execution, but tests must pass before
the session is considered complete.
"""

import subprocess
from pathlib import Path
from typing import Set


class StrictMode:
    """
    Mixin class providing strict mode functionality.

    Requires: self.project_root, self.palace_dir, self.strict_mode
    """

    def detect_affected_tests(self, modified_files: Set[str]) -> Set[str]:
        """
        Detect which test files should run based on modified files.

        Uses naming conventions and file structure to map files to tests.
        Returns set of test file paths.

        Args:
            modified_files: Set of modified file paths

        Returns:
            Set of test file paths to run
        """
        if not modified_files:
            return set()

        test_files = set()
        tests_dir = self.project_root / "tests"

        if not tests_dir.exists():
            return set()

        # For each modified file, try to find corresponding test file
        for modified_file in modified_files:
            try:
                file_path = Path(modified_file)

                # Skip if file is already a test
                if file_path.parts and file_path.parts[0] == "tests":
                    test_files.add(str(file_path))
                    continue

                # Try to find test file by naming convention
                # e.g., palace.py -> test_palace.py or test_core.py
                filename = file_path.stem

                # Look for test files that might test this module
                possible_patterns = [
                    f"test_{filename}.py",
                    f"test_*{filename}*.py",
                    f"test_core.py",  # Core tests likely test main module
                ]

                for pattern in possible_patterns:
                    matching_tests = list(tests_dir.glob(f"**/{pattern}"))
                    for test_file in matching_tests:
                        test_files.add(str(test_file.relative_to(self.project_root)))

            except Exception as e:
                # If detection fails, skip this file
                print(f"⚠️  Could not detect tests for {modified_file}: {e}")
                continue

        # If no specific tests found, run all tests (safer in strict mode)
        if not test_files and modified_files:
            all_tests = list(tests_dir.glob("**/test_*.py"))
            test_files = {str(t.relative_to(self.project_root)) for t in all_tests}

        return test_files

    def run_test_subset(self, test_files: Set[str] = None) -> bool:
        """
        Run a subset of tests.

        Args:
            test_files: Set of test file paths to run. If None, runs all tests.

        Returns:
            True if tests pass, False otherwise
        """
        if not test_files:
            # Run all tests
            cmd = ["python3", "-m", "pytest", "tests/", "-x", "--tb=short"]
        else:
            # Run specific test files
            cmd = ["python3", "-m", "pytest", "-x", "--tb=short"] + list(test_files)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"\n❌ Tests failed:\n{result.stdout}\n{result.stderr}")
                return False

            return True

        except Exception as e:
            print(f"⚠️  Error running tests: {e}")
            return False
