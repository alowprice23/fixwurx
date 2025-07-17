"""
tooling/test_runner.py
──────────────────────
Minimal *deterministic* unit-test harness for Triangulum.

Responsibilities
────────────────
1. Execute **all unit tests** under `tests/unit` (path overridable).
2. Ensure **repeatability** by seeding Python’s RNG and pytest’s hash-seed.
3. Emit a small, machine-readable JSON payload:

       {
         "success": true,
         "duration": 1.84,
         "tests": 123,
         "failures": 0,
         "errors": 0,
         "compressed_log": ""     // only when success==false
       }

4. Keep the compressed failure output ≤ 4096 tokens using
   `tooling.compress.Compressor` (so Verifier can ingest it).

No external dependencies beyond `pytest` itself.
"""

from __future__ import annotations

import json
import os
import platform
import random
import re
import secrets
import signal
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Final, List, Optional, Set, Tuple, Union, Any

# Import Compressor - handle both module structures
try:
    from tooling.compress import Compressor
except ImportError:
    # Direct import for local development
    from compress import Compressor

_RANDOM_SEED: Final[int] = 42
_MAX_TOKENS: Final[int] = 4096
_DEFAULT_TIMEOUT: Final[int] = 300  # 5 minutes timeout for tests
_DEFAULT_WORKERS: Final[int] = 4    # Default number of parallel workers


# ───────────────────────────────────────────────────────────────────────────────
# Test Result Types
# ───────────────────────────────────────────────────────────────────────────────
class TestStatus(Enum):
    """Status of test execution."""
    PASS = "pass"                 # All tests passed
    FAIL = "fail"                 # Some tests failed
    ERROR = "error"               # Test execution error
    TIMEOUT = "timeout"           # Test execution timed out
    SKIPPED = "skipped"           # All tests skipped
    NO_TESTS = "no_tests"         # No tests found
    INVALID = "invalid"           # Invalid test configuration


class TestType(Enum):
    """Type of tests to run."""
    UNIT = "unit"                 # Unit tests
    INTEGRATION = "integration"   # Integration tests
    PLANNER = "planner"           # Planner-specific tests
    FAMILY_TREE = "family_tree"   # Family tree tests
    HANDOFF = "handoff"           # Agent handoff tests
    ALL = "all"                   # All test types


class TestSuite:
    """
    Represents a test suite with specific configuration.
    
    A test suite includes:
    - Path to test files
    - Test type (unit, integration, planner, etc.)
    - Optional filtering criteria
    - Additional pytest arguments
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        test_type: TestType = TestType.UNIT,
        name: Optional[str] = None,
        filters: Optional[Dict[str, str]] = None,
        extra_args: Optional[List[str]] = None
    ):
        """
        Initialize a test suite.
        
        Args:
            path: Path to test files
            test_type: Type of tests in this suite
            name: Optional name for the suite
            filters: Optional filtering criteria (e.g., {"module": "test_planner"})
            extra_args: Additional pytest arguments
        """
        self.path = Path(path)
        self.test_type = test_type
        self.name = name or f"{test_type.value}_tests"
        self.filters = filters or {}
        self.extra_args = extra_args or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": str(self.path),
            "test_type": self.test_type.value,
            "name": self.name,
            "filters": self.filters,
            "extra_args": self.extra_args
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestSuite":
        """Create from dictionary representation."""
        return cls(
            path=data["path"],
            test_type=TestType(data["test_type"]),
            name=data["name"],
            filters=data["filters"],
            extra_args=data["extra_args"]
        )
    
    def build_args(self) -> List[str]:
        """Build pytest arguments from filters."""
        args = list(self.extra_args)
        
        # Add filters
        if "module" in self.filters:
            args.extend(["-k", self.filters["module"]])
        if "marker" in self.filters:
            args.extend(["-m", self.filters["marker"]])
        
        return args


class TestResult:
    """
    Enhanced test result with detailed information.
    
    Provides more detailed information than the original JSON result,
    including coverage data, test status categorization, and error analysis.
    """
    
    def __init__(
        self,
        suite: TestSuite,
        status: TestStatus,
        tests_total: int = 0,
        tests_passed: int = 0,
        tests_failed: int = 0,
        tests_skipped: int = 0,
        tests_error: int = 0,
        duration: float = 0.0,
        coverage: Optional[Dict[str, Any]] = None,
        compressed_log: str = "",
        error_details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a test result.
        
        Args:
            suite: The test suite that was run
            status: Overall status of the test run
            tests_total: Total number of tests executed
            tests_passed: Number of tests that passed
            tests_failed: Number of tests that failed
            tests_skipped: Number of tests that were skipped
            tests_error: Number of tests that had errors
            duration: Duration of the test run in seconds
            coverage: Optional coverage data
            compressed_log: Compressed output for failed tests
            error_details: Details about errors that occurred
        """
        self.suite = suite
        self.status = status
        self.tests_total = tests_total
        self.tests_passed = tests_passed
        self.tests_failed = tests_failed
        self.tests_skipped = tests_skipped
        self.tests_error = tests_error
        self.duration = duration
        self.coverage = coverage or {}
        self.compressed_log = compressed_log
        self.error_details = error_details or {}
        self.timestamp = time.time()
    
    @property
    def success(self) -> bool:
        """Whether the tests were successful."""
        return self.status == TestStatus.PASS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "suite": self.suite.to_dict(),
            "status": self.status.value,
            "success": self.success,
            "tests_total": self.tests_total,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "tests_error": self.tests_error,
            "duration": self.duration,
            "coverage": self.coverage,
            "compressed_log": self.compressed_log,
            "error_details": self.error_details,
            "timestamp": self.timestamp
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestResult":
        """Create from dictionary representation."""
        suite = TestSuite.from_dict(data["suite"])
        return cls(
            suite=suite,
            status=TestStatus(data["status"]),
            tests_total=data["tests_total"],
            tests_passed=data["tests_passed"],
            tests_failed=data["tests_failed"],
            tests_skipped=data["tests_skipped"],
            tests_error=data["tests_error"],
            duration=data["duration"],
            coverage=data["coverage"],
            compressed_log=data["compressed_log"],
            error_details=data["error_details"]
        )
    
    @classmethod
    def from_legacy_json(cls, json_str: str, suite: TestSuite) -> "TestResult":
        """Create from legacy JSON format."""
        data = json.loads(json_str)
        
        # Determine status
        if data["success"]:
            status = TestStatus.PASS
        elif data["errors"] > 0:
            status = TestStatus.ERROR
        else:
            status = TestStatus.FAIL
            
        return cls(
            suite=suite,
            status=status,
            tests_total=data["tests"],
            tests_passed=data["tests"] - data["failures"] - data["errors"],
            tests_failed=data["failures"],
            tests_error=data["errors"],
            duration=data["duration"],
            compressed_log=data.get("compressed_log", "")
        )


# ───────────────────────────────────────────────────────────────────────────────
# Test Discovery and Management
# ───────────────────────────────────────────────────────────────────────────────
def discover_tests(
    base_dir: Union[str, Path] = ".",
    test_type: TestType = TestType.ALL
) -> List[TestSuite]:
    """
    Discover test suites based on directory structure.
    
    Args:
        base_dir: Base directory to search for tests
        test_type: Type of tests to discover
        
    Returns:
        List of discovered test suites
    """
    base_path = Path(base_dir)
    suites = []
    
    # Map of test types to directories and markers
    type_mapping = {
        TestType.UNIT: {
            "paths": ["tests/unit"],
            "marker": "unit",
        },
        TestType.INTEGRATION: {
            "paths": ["tests/integration"],
            "marker": "integration",
        },
        TestType.PLANNER: {
            "paths": ["tests/planner"],
            "marker": "planner",
        },
        TestType.FAMILY_TREE: {
            "paths": ["tests/family_tree"],
            "marker": "family_tree",
        },
        TestType.HANDOFF: {
            "paths": ["tests/handoff"],
            "marker": "handoff",
        },
    }
    
    # Determine which test types to discover
    types_to_discover = [test_type] if test_type != TestType.ALL else list(type_mapping.keys())
    
    for t_type in types_to_discover:
        if t_type in type_mapping:
            mapping = type_mapping[t_type]
            
            # Try each possible path
            for rel_path in mapping["paths"]:
                path = base_path / rel_path
                if path.exists():
                    suite = TestSuite(
                        path=path,
                        test_type=t_type,
                        name=f"{t_type.value}_tests",
                        filters={"marker": mapping["marker"]} if "marker" in mapping else None
                    )
                    suites.append(suite)
    
    return suites


def create_planner_test_suite(
    base_dir: Union[str, Path] = "."
) -> List[TestSuite]:
    """
    Create test suites specifically for planner functionality.
    
    Args:
        base_dir: Base directory for tests
        
    Returns:
        List of planner-specific test suites
    """
    base_path = Path(base_dir)
    suites = []
    
    # Core planner functionality
    planner_suite = TestSuite(
        path=base_path / "tests/planner",
        test_type=TestType.PLANNER,
        name="planner_core",
        filters={"module": "test_planner"}
    )
    suites.append(planner_suite)
    
    # Family tree tests
    family_tree_suite = TestSuite(
        path=base_path / "tests/planner",
        test_type=TestType.FAMILY_TREE,
        name="family_tree",
        filters={"module": "test_family_tree"}
    )
    suites.append(family_tree_suite)
    
    # Agent handoff tests
    handoff_suite = TestSuite(
        path=base_path / "tests/planner",
        test_type=TestType.HANDOFF,
        name="agent_handoff",
        filters={"module": "test_handoff"}
    )
    suites.append(handoff_suite)
    
    return suites


# ───────────────────────────────────────────────────────────────────────────────
# Enhanced Test Runner
# ───────────────────────────────────────────────────────────────────────────────
class TestRunner:
    """
    Enhanced test runner with detailed results and planner-specific features.
    
    Features:
    - Support for different test types
    - Timeout handling
    - Parallel execution
    - Detailed error reporting
    - Filtering and selection
    """
    
    def __init__(
        self,
        base_dir: Union[str, Path] = ".",
        timeout: int = _DEFAULT_TIMEOUT,
        use_coverage: bool = True,
        parallel: bool = True,
        num_workers: int = _DEFAULT_WORKERS,
        verbose: bool = True
    ):
        """
        Initialize a test runner.
        
        Args:
            base_dir: Base directory for tests
            timeout: Timeout in seconds for each test
            use_coverage: Whether to collect coverage data
            parallel: Whether to run tests in parallel
            num_workers: Number of parallel workers
            verbose: Whether to print verbose output
        """
        self.base_dir = Path(base_dir)
        self.timeout = timeout
        self.use_coverage = use_coverage
        self.parallel = parallel
        self.num_workers = num_workers
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.compressor = Compressor(_MAX_TOKENS)
    
    def run_suite(self, suite: TestSuite) -> TestResult:
        """
        Run a test suite and return the result.
        
        Args:
            suite: Test suite to run
            
        Returns:
            Test result
        """
        if self.verbose:
            print(f"Running test suite: {suite.name} ({suite.test_type.value})")
            print(f"  Path: {suite.path}")
            print(f"  Filters: {suite.filters}")
        
        # Build pytest arguments
        extra_args = list(suite.build_args())
        
        # Add coverage if requested
        if self.use_coverage:
            extra_args.extend(["--cov", "--cov-report=term-missing"])
        
        # Add parallel workers if requested
        if self.parallel:
            extra_args.extend(["-n", str(self.num_workers)])
        
        # Add any other needed args
        extra_args.extend(["--no-header", "--no-summary"])
        
        start_time = time.time()
        try:
            # Run the tests with a timeout
            proc = self._run_pytest_with_timeout(suite.path, tuple(extra_args))
            
            # Parse the results
            tests, fails, errors = self._parse_detailed_summary(proc.stdout, proc.stderr)
            
            # Determine the status
            if tests == 0:
                status = TestStatus.NO_TESTS
            elif proc.returncode == 0:
                status = TestStatus.PASS
            elif errors > 0:
                status = TestStatus.ERROR
            else:
                status = TestStatus.FAIL
                
            # Calculate duration
            duration = time.time() - start_time
            
            # Extract coverage data if available
            coverage = self._extract_coverage(proc.stdout) if self.use_coverage else {}
            
            # Generate compressed log for failures
            compressed_log = ""
            if status in (TestStatus.FAIL, TestStatus.ERROR, TestStatus.TIMEOUT):
                log = "\n--- STDOUT ---\n" + proc.stdout + "\n--- STDERR ---\n" + proc.stderr
                compressed_log, _ = self.compressor.compress(log)
            
            # Create the result
            result = TestResult(
                suite=suite,
                status=status,
                tests_total=tests,
                tests_passed=tests - fails - errors,
                tests_failed=fails,
                tests_error=errors,
                duration=duration,
                coverage=coverage,
                compressed_log=compressed_log,
                error_details=self._analyze_failures(proc.stdout, proc.stderr) if fails > 0 or errors > 0 else {}
            )
            
            # Log the result
            if self.verbose:
                passed = tests - fails - errors
                print(f"  Result: {status.value.upper()} - {passed}/{tests} tests passed in {duration:.2f}s")
            
            # Store the result
            self.results.append(result)
            return result
            
        except subprocess.TimeoutExpired:
            # Handle timeout
            duration = time.time() - start_time
            
            if self.verbose:
                print(f"  Result: TIMEOUT - Test execution timed out after {self.timeout}s")
            
            # Create timeout result
            result = TestResult(
                suite=suite,
                status=TestStatus.TIMEOUT,
                duration=duration,
                compressed_log=f"Test execution timed out after {self.timeout} seconds",
                error_details={"type": "timeout", "timeout": self.timeout}
            )
            
            # Store the result
            self.results.append(result)
            return result
    
    def run_suites(self, suites: List[TestSuite]) -> List[TestResult]:
        """
        Run multiple test suites and return the results.
        
        Args:
            suites: List of test suites to run
            
        Returns:
            List of test results
        """
        results = []
        for suite in suites:
            results.append(self.run_suite(suite))
        return results
    
    def run_by_type(self, test_type: TestType = TestType.UNIT) -> List[TestResult]:
        """
        Run all tests of a specific type.
        
        Args:
            test_type: Type of tests to run
            
        Returns:
            List of test results
        """
        suites = discover_tests(self.base_dir, test_type)
        return self.run_suites(suites)
    
    def run_planner_tests(self) -> List[TestResult]:
        """
        Run all planner-specific tests.
        
        Returns:
            List of test results
        """
        suites = create_planner_test_suite(self.base_dir)
        return self.run_suites(suites)
    
    def _run_pytest_with_timeout(
        self, path: Path, extra_args: Tuple[str, ...]
    ) -> subprocess.CompletedProcess:
        """
        Run pytest with a timeout.
        
        Args:
            path: Path to test directory
            extra_args: Additional pytest arguments
            
        Returns:
            Completed process
        
        Raises:
            subprocess.TimeoutExpired: If the test execution times out
        """
        # Set up environment with deterministic seeds
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(_RANDOM_SEED)
        env["COVERAGE_FILE"] = ".coverage"
        
        # Seed random
        random.seed(_RANDOM_SEED)
        
        # Build the command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(path),
            "-v",                # verbose output
            f"--timeout={self.timeout}",  # timeout per test
            "--maxfail=50",      # short-circuit runaway suites
            *extra_args,
        ]
        
        # Run the process with a timeout
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=self.timeout + 10  # Add 10 seconds to account for pytest startup
        )
    
    def _parse_detailed_summary(self, stdout: str, stderr: str) -> Tuple[int, int, int]:
        """
        Parse pytest output to extract detailed test summary.
        
        Args:
            stdout: Standard output from pytest
            stderr: Standard error from pytest
            
        Returns:
            Tuple of (total tests, failures, errors)
        """
        # First try to parse standard pytest summary
        tests, fails, errors = _parse_summary(stdout + stderr)
        
        # If that didn't work, try more advanced parsing
        if tests == 0:
            # Count test cases based on "PASSED", "FAILED", etc. lines
            passed = stdout.count(" PASSED ")
            failed = stdout.count(" FAILED ")
            skipped = stdout.count(" SKIPPED ")
            errors_count = stdout.count(" ERROR ")
            
            tests = passed + failed + skipped + errors_count
            fails = failed
            errors = errors_count
        
        return tests, fails, errors
    
    def _extract_coverage(self, stdout: str) -> Dict[str, Any]:
        """
        Extract coverage data from pytest-cov output.
        
        Args:
            stdout: Standard output from pytest
            
        Returns:
            Dictionary with coverage data
        """
        coverage = {
            "total": 0,
            "covered": 0,
            "percentage": 0.0,
            "missing": [],
            "by_file": {}
        }
        
        # Find coverage section
        for i, line in enumerate(stdout.splitlines()):
            if "---------- coverage:" in line:
                # Parse the coverage data
                for j in range(i + 1, len(stdout.splitlines())):
                    cov_line = stdout.splitlines()[j]
                    
                    # Stop at the end of the coverage section
                    if "---------- " in cov_line and "coverage:" not in cov_line:
                        break
                    
                    # Parse file coverage
                    if " " in cov_line and "%" in cov_line:
                        parts = cov_line.split()
                        if len(parts) >= 2:
                            file_path = parts[0]
                            percentage = parts[-1].rstrip("%")
                            try:
                                coverage["by_file"][file_path] = float(percentage)
                            except ValueError:
                                pass
                            
                # Extract total coverage
                for j in range(i + 1, len(stdout.splitlines())):
                    cov_line = stdout.splitlines()[j]
                    if "TOTAL" in cov_line and "%" in cov_line:
                        parts = cov_line.split()
                        try:
                            coverage["percentage"] = float(parts[-1].rstrip("%"))
                            break
                        except (ValueError, IndexError):
                            pass
                
                break
        
        return coverage
    
    def _analyze_failures(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """
        Analyze test failures to provide more detailed information.
        
        Args:
            stdout: Standard output from pytest
            stderr: Standard error from pytest
            
        Returns:
            Dictionary with failure analysis
        """
        analysis = {
            "files": set(),
            "functions": set(),
            "error_types": set(),
            "error_messages": [],
            "file_line_mapping": {}
        }
        
        # Look for test failures
        error_pattern = re.compile(r"(\w+Error|Exception):\s*(.+?)$")
        file_pattern = re.compile(r"(\w+\.py):(\d+)")
        function_pattern = re.compile(r"def\s+(\w+)\s*\(")
        
        for line in (stdout + stderr).splitlines():
            # Extract error types and messages
            error_match = error_pattern.search(line)
            if error_match:
                error_type = error_match.group(1)
                error_message = error_match.group(2).strip()
                analysis["error_types"].add(error_type)
                analysis["error_messages"].append(error_message)
            
            # Extract files and line numbers
            file_match = file_pattern.search(line)
            if file_match:
                file_name = file_match.group(1)
                line_number = int(file_match.group(2))
                analysis["files"].add(file_name)
                
                if file_name not in analysis["file_line_mapping"]:
                    analysis["file_line_mapping"][file_name] = []
                analysis["file_line_mapping"][file_name].append(line_number)
            
            # Extract function names
            function_match = function_pattern.search(line)
            if function_match:
                function_name = function_match.group(1)
                analysis["functions"].add(function_name)
        
        # Convert sets to lists for JSON serialization
        analysis["files"] = list(analysis["files"])
        analysis["functions"] = list(analysis["functions"])
        analysis["error_types"] = list(analysis["error_types"])
        
        return analysis
    
    def save_results(self, path: Union[str, Path]) -> None:
        """
        Save test results to a JSON file.
        
        Args:
            path: Path to save results to
        """
        results_data = [result.to_dict() for result in self.results]
        
        with open(path, "w") as f:
            json.dump(results_data, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to {path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all test results.
        
        Returns:
            Dictionary with summary data
        """
        summary = {
            "total_suites": len(self.results),
            "passed_suites": sum(1 for r in self.results if r.success),
            "failed_suites": sum(1 for r in self.results if not r.success),
            "total_tests": sum(r.tests_total for r in self.results),
            "passed_tests": sum(r.tests_passed for r in self.results),
            "failed_tests": sum(r.tests_failed for r in self.results),
            "error_tests": sum(r.tests_error for r in self.results),
            "skipped_tests": sum(r.tests_skipped for r in self.results),
            "total_duration": sum(r.duration for r in self.results),
        }
        
        # Calculate overall status
        if summary["total_suites"] == 0:
            summary["status"] = "NO_TESTS"
        elif summary["failed_suites"] == 0:
            summary["status"] = "PASS"
        else:
            summary["status"] = "FAIL"
        
        return summary
        

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def _run_pytest(path: Path, extra_args: Tuple[str, ...]) -> subprocess.CompletedProcess:
    """
    Launch pytest as a subprocess with deterministic env.
    """
    random.seed(_RANDOM_SEED)
    random.seed(_RANDOM_SEED)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(path),
        "-q",                 # quiet
        "--maxfail=50",       # short-circuit runaway suites
        *extra_args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _parse_summary(out: str) -> Tuple[int, int, int]:
    """
    Very small parser for pytest terminal summary lines like:

        === 123 passed, 2 warnings in 1.84s ===
        === 120 passed, 3 failed, 2 errors in 2.22s ===
    """
    tests = fails = errs = 0
    for line in out.splitlines():
        if "passed" in line and "in" in line and "s" in line:
            tokens = line.replace("=", "").replace(",", "").split()
            for i, tok in enumerate(tokens):
                if tok.isdigit():
                    val = int(tok)
                    tag = tokens[i + 1]
                    if tag.startswith("passed"):
                        tests = val
                    elif tag.startswith("failed"):
                        fails = val
                    elif tag.startswith("errors") or tag.startswith("error"):
                        errs = val
            break
    return tests, fails, errs


# ───────────────────────────────────────────────────────────────────────────────
# Public entry
# ───────────────────────────────────────────────────────────────────────────────
def run_unit_tests(
    test_path: str | Path = "tests/unit",
    *pytest_extra: str,
) -> str:
    """
    Run tests and return JSON string (see schema above).
    """
    start = time.perf_counter()
    proc = _run_pytest(Path(test_path), pytest_extra)
    duration = round(time.perf_counter() - start, 3)

    tests, fails, errors = _parse_summary(proc.stdout + proc.stderr)
    success = proc.returncode == 0

    compressed_log = ""
    if not success:
        log = "\n--- STDOUT ---\n" + proc.stdout + "\n--- STDERR ---\n" + proc.stderr
        compressed_log, _ = Compressor(_MAX_TOKENS).compress(log)
    result = {
        "success": success,
        "duration": duration,
        "tests": tests,
        "failures": fails,
        "errors": errors,
        "compressed_log": compressed_log,
    }
    return json.dumps(result, indent=2)


# ───────────────────────────────────────────────────────────────────────────────
# CLI                                                                          
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    import argparse

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Enhanced test runner with planner support")
    parser.add_argument("--path", default="tests/unit", help="test directory")
    parser.add_argument("--type", choices=[t.value for t in TestType], default="unit", 
                      help="test type to run")
    parser.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT, 
                      help="timeout in seconds")
    parser.add_argument("--coverage", action="store_true", default=True, 
                      help="collect coverage data")
    parser.add_argument("--no-coverage", action="store_false", dest="coverage", 
                      help="disable coverage collection")
    parser.add_argument("--parallel", action="store_true", default=True, 
                      help="run tests in parallel")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel", 
                      help="disable parallel execution")
    parser.add_argument("--workers", type=int, default=_DEFAULT_WORKERS, 
                      help="number of parallel workers")
    parser.add_argument("--results", help="path to save detailed results")
    parser.add_argument("--legacy", action="store_true", 
                      help="use legacy runner")
    parser.add_argument("--planner", action="store_true", 
                      help="run planner-specific tests")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="verbose output")
    parser.add_argument("pytest_args", nargs="*", 
                      help="extra args forwarded to pytest")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if using legacy runner
    if args.legacy:
        # Use the legacy runner
        print(
            run_unit_tests(
                args.path,
                *args.pytest_args,
            )
        )
    else:
        # Use the enhanced test runner
        runner = TestRunner(
            base_dir=".",
            timeout=args.timeout,
            use_coverage=args.coverage,
            parallel=args.parallel,
            num_workers=args.workers,
            verbose=args.verbose
        )
        
        # Determine which tests to run
        if args.planner:
            # Run planner-specific tests
            results = runner.run_planner_tests()
        else:
            # Run tests of the specified type
            results = runner.run_by_type(TestType(args.type))
        
        # Save detailed results if requested
        if args.results:
            runner.save_results(args.results)
        
        # Print summary
        summary = runner.get_summary()
        print(json.dumps({
            "success": summary["status"] == "PASS",
            "duration": summary["total_duration"],
            "tests": summary["total_tests"],
            "failures": summary["failed_tests"],
            "errors": summary["error_tests"],
            "compressed_log": results[0].compressed_log if results and not results[0].success else ""
        }, indent=2))
        
        # Exit with appropriate code
        sys.exit(0 if summary["status"] == "PASS" else 1)
