"""
tooling/smoke_runner.py
───────────────────────
Run **environment-heavy** smoke tests *inside the live canary containers*.

The module:

1. Executes `pytest tests/smoke -q` **inside** a service container that
   belongs to an already-running Docker-Compose canary stack.
2. Captures *both* `stdout` and `stderr`.
3. If pytest exits non-zero, uses `tooling.compress.Compressor` to shrink the
   combined output to ≤ 4 096 tokens and returns `(False, compressed_text)`.
4. On success returns `(True, "")`.

CLI usage
─────────
    python -m tooling.smoke_runner \
        --project triangulum_canary --service web --max-tokens 4096

API usage
─────────
    ok, log = run_smoke_tests("triangulum_canary", "web")
    if not ok:
        escalate(log_bits=log)
    
    # Enhanced API with detailed reporting:
    runner = SmokeTestRunner("triangulum_canary", "web")
    result = runner.run()
    if not result.success:
        escalate(result=result)
        send_metrics(result.metrics())

No external dependencies: uses only std-lib `subprocess`, `shlex`,
`tooling.compress.Compressor`.

───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
import time
import re
import hashlib
import argparse
import threading
import signal
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union, Callable, Set

# Import Compressor - handle both module structures
try:
    from tooling.compress import Compressor
except ImportError:
    # Direct import for local development
    from compress import Compressor

# ---------------------------------------------------------------------------—
# Result Types
# ---------------------------------------------------------------------------—
class SmokeResultType(Enum):
    """Enum representing the possible outcomes of smoke tests."""
    PASS = "pass"                 # All tests passed
    FAIL = "fail"                 # Tests failed with assertion/expectation errors
    ERROR = "error"               # Tests had execution errors
    CONTAINER_ERROR = "container_error"  # Container issues (not running, permission, etc.)
    TIMEOUT = "timeout"           # Tests exceeded the timeout limit
    INFRASTRUCTURE = "infrastructure"  # Docker/compose infrastructure issues
    SKIPPED = "skipped"           # Tests were skipped
    PARTIAL_SUCCESS = "partial_success"  # Some tests passed, some failed/skipped
    UNSTABLE = "unstable"         # Tests have inconsistent results across runs
    UNKNOWN = "unknown"           # Unknown outcome


@dataclass
class ContainerStats:
    """Statistics about the container during test execution."""
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    network_in_bytes: int = 0
    network_out_bytes: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    
    @property
    def cpu_efficiency(self) -> float:
        """Calculate CPU efficiency (higher is better)."""
        # Simple heuristic: efficiency drops as CPU usage approaches 100%
        if self.cpu_usage_percent >= 100:
            return 0.5  # Potentially bottlenecked
        return 1.0 - (self.cpu_usage_percent / 200)  # Linear scale from 1.0 to 0.5
    
    @property
    def memory_efficiency(self) -> float:
        """Calculate memory efficiency (higher is better)."""
        # Simple heuristic based on memory usage
        # This would be better with container memory limits for context
        if self.memory_usage_mb < 100:
            return 1.0  # Very efficient
        elif self.memory_usage_mb < 500:
            return 0.9  # Good
        elif self.memory_usage_mb < 1000:
            return 0.8  # Acceptable
        elif self.memory_usage_mb < 2000:
            return 0.7  # Getting high
        else:
            return 0.6  # High memory usage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "network_in_bytes": self.network_in_bytes,
            "network_out_bytes": self.network_out_bytes,
            "disk_read_bytes": self.disk_read_bytes,
            "disk_write_bytes": self.disk_write_bytes,
            "cpu_efficiency": self.cpu_efficiency,
            "memory_efficiency": self.memory_efficiency
        }


@dataclass
class TestCoverage:
    """Test coverage information."""
    components_tested: Set[str] = field(default_factory=set)
    functions_tested: Set[str] = field(default_factory=set)
    lines_covered: int = 0
    lines_total: int = 0
    branches_covered: int = 0
    branches_total: int = 0
    
    @property
    def line_coverage_percent(self) -> float:
        """Calculate line coverage percentage."""
        if self.lines_total == 0:
            return 0.0
        return (self.lines_covered / self.lines_total) * 100
    
    @property
    def branch_coverage_percent(self) -> float:
        """Calculate branch coverage percentage."""
        if self.branches_total == 0:
            return 0.0
        return (self.branches_covered / self.branches_total) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "components_tested": list(self.components_tested),
            "functions_tested": list(self.functions_tested),
            "lines_covered": self.lines_covered,
            "lines_total": self.lines_total,
            "branches_covered": self.branches_covered,
            "branches_total": self.branches_total,
            "line_coverage_percent": self.line_coverage_percent,
            "branch_coverage_percent": self.branch_coverage_percent
        }


@dataclass
class TestFailure:
    """Detailed information about a test failure."""
    test_name: str
    failure_message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    error_type: str = "unknown"
    traceback: Optional[str] = None
    component: Optional[str] = None
    expected_value: Optional[str] = None
    actual_value: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "failure_message": self.failure_message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "error_type": self.error_type,
            "traceback": self.traceback,
            "component": self.component,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value
        }


@dataclass
class SmokeResult:
    """
    Enhanced result data for smoke tests.
    
    Provides detailed information about test execution, including
    container stats, failure details, and metrics.
    """
    # Basic result information
    success: bool
    result_type: SmokeResultType
    log: str = ""
    
    # Test execution details
    duration_seconds: float = 0.0
    exit_code: int = 0
    timestamp: float = field(default_factory=time.time)
    
    # Container information
    compose_project: str = ""
    service: str = ""
    container_id: str = ""
    container_stats: Optional[ContainerStats] = None
    
    # Test details
    test_count: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    tests_error: int = 0
    
    # Test failures
    failures: List[TestFailure] = field(default_factory=list)
    
    # Coverage information
    coverage: Optional[TestCoverage] = None
    
    # Environment information
    environment: Dict[str, str] = field(default_factory=dict)
    
    # Compression details
    original_log_size: int = 0
    compressed_log_size: int = 0
    compression_ratio: float = 0.0
    
    # Failure analysis
    failure_details: Dict[str, Any] = field(default_factory=dict)
    
    # Root cause analysis
    root_cause: Optional[str] = None
    
    # History - for tracking test stability
    run_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    is_retry: bool = False
    retry_count: int = 0
    
    # Planner integration
    planner_bug_id: Optional[str] = None
    planner_execution_id: Optional[str] = None
    
    def metrics(self) -> Dict[str, Any]:
        """Generate metrics for monitoring."""
        metrics = {
            "success": int(self.success),
            "duration_seconds": self.duration_seconds,
            "test_count": self.test_count,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "tests_error": self.tests_error,
            "pass_rate": self.tests_passed / max(1, self.test_count),
            "has_failures": len(self.failures) > 0,
            "exit_code": self.exit_code,
        }
        
        # Add container stats
        if self.container_stats:
            metrics.update({
                "cpu_usage_percent": self.container_stats.cpu_usage_percent,
                "memory_usage_mb": self.container_stats.memory_usage_mb,
                "cpu_efficiency": self.container_stats.cpu_efficiency,
                "memory_efficiency": self.container_stats.memory_efficiency
            })
        
        # Add coverage metrics if available
        if self.coverage:
            metrics.update({
                "line_coverage_percent": self.coverage.line_coverage_percent,
                "branch_coverage_percent": self.coverage.branch_coverage_percent,
                "components_tested_count": len(self.coverage.components_tested),
                "functions_tested_count": len(self.coverage.functions_tested)
            })
        
        return metrics
    
    def get_test_stability_score(self) -> float:
        """
        Calculate test stability score from 0.0 to 1.0.
        
        Higher values indicate more stable tests.
        """
        if self.test_count == 0:
            return 0.0
        
        # Base stability on the pass rate
        base_stability = self.tests_passed / self.test_count
        
        # Adjust for retry history
        if self.retry_count > 0:
            # If we needed retries, the stability decreases
            stability_penalty = min(0.5, self.retry_count * 0.1)
            base_stability = max(0.0, base_stability - stability_penalty)
        
        # Adjust for skipped tests
        if self.tests_skipped > 0:
            skip_ratio = self.tests_skipped / self.test_count
            base_stability = max(0.0, base_stability - (skip_ratio * 0.2))
        
        return base_stability
    
    def get_most_impacted_components(self, limit: int = 3) -> List[str]:
        """Get the most impacted components based on failures."""
        if not self.failures:
            return []
        
        component_counts = {}
        for failure in self.failures:
            if failure.component:
                component_counts[failure.component] = component_counts.get(failure.component, 0) + 1
        
        # Sort by count in descending order
        sorted_components = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N components
        return [component for component, _ in sorted_components[:limit]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary for serialization."""
        result = {
            "success": self.success,
            "result_type": self.result_type.value,
            "log": self.log,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "timestamp": self.timestamp,
            "compose_project": self.compose_project,
            "service": self.service,
            "container_id": self.container_id,
            "test_count": self.test_count,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "tests_error": self.tests_error,
            "failures": [failure.to_dict() for failure in self.failures],
            "environment": self.environment,
            "original_log_size": self.original_log_size,
            "compressed_log_size": self.compressed_log_size,
            "compression_ratio": self.compression_ratio,
            "failure_details": self.failure_details,
            "root_cause": self.root_cause,
            "run_id": self.run_id,
            "is_retry": self.is_retry,
            "retry_count": self.retry_count,
            "planner_bug_id": self.planner_bug_id,
            "planner_execution_id": self.planner_execution_id,
            "stability_score": self.get_test_stability_score(),
            "most_impacted_components": self.get_most_impacted_components()
        }
        
        if self.container_stats:
            result["container_stats"] = self.container_stats.to_dict()
        
        if self.coverage:
            result["coverage"] = self.coverage.to_dict()
        
        return result
    
    def to_json(self) -> str:
        """Convert the result to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_junit_xml(self) -> str:
        """Convert the result to JUnit XML format for CI integration."""
        # Simple JUnit XML generation
        test_cases = []
        
        # Add successful tests as placeholders if we don't have individual results
        for i in range(self.tests_passed):
            test_cases.append(f'''
            <testcase name="passed_test_{i+1}" classname="smoke_tests" time="{self.duration_seconds / max(1, self.test_count)}"/>
            ''')
        
        # Add failed tests
        for i, failure in enumerate(self.failures):
            test_name = failure.test_name or f"failed_test_{i+1}"
            class_name = f"smoke_tests.{failure.component}" if failure.component else "smoke_tests"
            
            test_cases.append(f'''
            <testcase name="{test_name}" classname="{class_name}" time="{self.duration_seconds / max(1, self.test_count)}">
                <failure message="{failure.failure_message}" type="{failure.error_type}">
                    {failure.traceback or ""}
                </failure>
            </testcase>
            ''')
        
        # Add skipped tests as placeholders
        for i in range(self.tests_skipped):
            test_cases.append(f'''
            <testcase name="skipped_test_{i+1}" classname="smoke_tests" time="0">
                <skipped/>
            </testcase>
            ''')
        
        # Add error tests as placeholders
        for i in range(self.tests_error):
            test_cases.append(f'''
            <testcase name="error_test_{i+1}" classname="smoke_tests" time="{self.duration_seconds / max(1, self.test_count)}">
                <error message="Test execution error"/>
            </testcase>
            ''')
        
        # Combine all test cases
        test_case_xml = "".join(test_cases)
        
        # Create the full XML
        return f'''<?xml version="1.0" encoding="UTF-8"?>
        <testsuites>
            <testsuite name="smoke_tests" tests="{self.test_count}" failures="{self.tests_failed}" errors="{self.tests_error}" skipped="{self.tests_skipped}" time="{self.duration_seconds}">
                {test_case_xml}
            </testsuite>
        </testsuites>
        '''
    
    def to_github_actions_annotations(self) -> List[str]:
        """Convert failures to GitHub Actions annotations format."""
        annotations = []
        
        for failure in self.failures:
            file_path = failure.file_path or "unknown"
            line = failure.line_number or 1
            message = failure.failure_message or "Test failed"
            
            annotations.append(f"::error file={file_path},line={line}::{message}")
        
        return annotations
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SmokeResult:
        """Create a SmokeResult from a dictionary."""
        # Create a copy to avoid modifying the input
        data_copy = data.copy()
        
        # Convert string result_type back to enum
        if "result_type" in data_copy:
            data_copy["result_type"] = SmokeResultType(data_copy["result_type"])
        
        # Create ContainerStats instance if present
        if "container_stats" in data_copy:
            stats = data_copy.pop("container_stats")
            data_copy["container_stats"] = ContainerStats(
                cpu_usage_percent=stats.get("cpu_usage_percent", 0.0),
                memory_usage_mb=stats.get("memory_usage_mb", 0.0),
                network_in_bytes=stats.get("network_in_bytes", 0),
                network_out_bytes=stats.get("network_out_bytes", 0),
                disk_read_bytes=stats.get("disk_read_bytes", 0),
                disk_write_bytes=stats.get("disk_write_bytes", 0)
            )
        
        # Create TestCoverage instance if present
        if "coverage" in data_copy:
            coverage_data = data_copy.pop("coverage")
            data_copy["coverage"] = TestCoverage(
                components_tested=set(coverage_data.get("components_tested", [])),
                functions_tested=set(coverage_data.get("functions_tested", [])),
                lines_covered=coverage_data.get("lines_covered", 0),
                lines_total=coverage_data.get("lines_total", 0),
                branches_covered=coverage_data.get("branches_covered", 0),
                branches_total=coverage_data.get("branches_total", 0)
            )
        
        # Create TestFailure instances if present
        if "failures" in data_copy:
            failures = data_copy.pop("failures")
            data_copy["failures"] = [TestFailure(**failure) for failure in failures]
        
        return cls(**data_copy)
    
    @classmethod
    def create_success(cls, compose_project: str, service: str) -> SmokeResult:
        """Create a successful result."""
        return cls(
            success=True,
            result_type=SmokeResultType.PASS,
            compose_project=compose_project,
            service=service,
            environment={"platform": platform.platform()}
        )
    
    @classmethod
    def create_failure(cls, compose_project: str, service: str, 
                      log: str, exit_code: int, 
                      failure_details: Dict[str, Any] = None) -> SmokeResult:
        """Create a failure result."""
        return cls(
            success=False,
            result_type=SmokeResultType.FAIL,
            log=log,
            exit_code=exit_code,
            compose_project=compose_project,
            service=service,
            failure_details=failure_details or {},
            environment={"platform": platform.platform()}
        )
    
    @classmethod
    def create_error(cls, compose_project: str, service: str, 
                    error_message: str, exit_code: int = -1) -> SmokeResult:
        """Create an error result."""
        return cls(
            success=False,
            result_type=SmokeResultType.ERROR,
            log=error_message,
            exit_code=exit_code,
            compose_project=compose_project,
            service=service,
            environment={"platform": platform.platform()}
        )
    
    @classmethod
    def create_container_error(cls, compose_project: str, service: str, 
                             error_message: str) -> SmokeResult:
        """Create a container error result."""
        return cls(
            success=False,
            result_type=SmokeResultType.CONTAINER_ERROR,
            log=error_message,
            compose_project=compose_project,
            service=service,
            environment={"platform": platform.platform()}
        )
    
    @classmethod
    def create_timeout(cls, compose_project: str, service: str, 
                      timeout: float, partial_output: str = "") -> SmokeResult:
        """Create a timeout result."""
        return cls(
            success=False,
            result_type=SmokeResultType.TIMEOUT,
            log=f"{partial_output}\n\nTest execution timed out after {timeout} seconds",
            duration_seconds=timeout,
            compose_project=compose_project,
            service=service,
            environment={"platform": platform.platform()},
            failure_details={"timeout_seconds": timeout}
        )
    
    @classmethod
    def create_partial_success(cls, compose_project: str, service: str,
                             test_count: int, tests_passed: int,
                             tests_failed: int, tests_skipped: int,
                             failures: List[TestFailure] = None) -> SmokeResult:
        """Create a partial success result (some tests passed, some failed)."""
        return cls(
            success=False,  # Overall considered a failure
            result_type=SmokeResultType.PARTIAL_SUCCESS,
            compose_project=compose_project,
            service=service,
            test_count=test_count,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            failures=failures or [],
            environment={"platform": platform.platform()}
        )


# ---------------------------------------------------------------------------—
# Failure Analysis
# ---------------------------------------------------------------------------—
class FailureAnalyzer:
    """
    Enhanced failure analysis for smoke tests.
    
    Features:
    - Detailed error parsing
    - Component identification
    - Root cause analysis
    - Test categorization
    """
    
    def __init__(self, patterns_db: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize a FailureAnalyzer.
        
        Parameters
        ----------
        patterns_db : Optional[Dict[str, Dict[str, Any]]]
            Database of error patterns to match against.
        """
        self.patterns_db = patterns_db or self._get_default_patterns()
    
    def _get_default_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get default error patterns to match against."""
        return {
            "assertion_error": {
                "patterns": [r"AssertionError", r"assert\s+"],
                "component_extraction": r"tests/smoke/test_(\w+)\.py",
                "error_type": "assertion_error"
            },
            "attribute_error": {
                "patterns": [r"AttributeError", r"has no attribute"],
                "component_extraction": r"tests/smoke/test_(\w+)\.py",
                "error_type": "attribute_error"
            },
            "index_error": {
                "patterns": [r"IndexError", r"list index out of range"],
                "component_extraction": r"tests/smoke/test_(\w+)\.py",
                "error_type": "index_error"
            },
            "key_error": {
                "patterns": [r"KeyError", r"dictionary key"],
                "component_extraction": r"tests/smoke/test_(\w+)\.py",
                "error_type": "key_error"
            },
            "import_error": {
                "patterns": [r"ImportError", r"No module named"],
                "component_extraction": r"tests/smoke/test_(\w+)\.py",
                "error_type": "import_error"
            },
            "connection_error": {
                "patterns": [r"ConnectionError", r"Connection refused", r"Failed to establish a connection"],
                "component_extraction": r"tests/smoke/test_(\w+)\.py",
                "error_type": "connection_error"
            },
            "timeout_error": {
                "patterns": [r"TimeoutError", r"timed out"],
                "component_extraction": r"tests/smoke/test_(\w+)\.py",
                "error_type": "timeout_error"
            },
            # Docker/container specific errors
            "container_not_found": {
                "patterns": [r"No such container", r"not found"],
                "error_type": "container_error"
            },
            "container_permission": {
                "patterns": [r"Permission denied", r"Access denied"],
                "error_type": "permission_error"
            }
        }
    
    def parse_pytest_failures(self, stdout: str, stderr: str) -> List[TestFailure]:
        """
        Parse pytest output to extract detailed test failures.
        
        Parameters
        ----------
        stdout : str
            Standard output from pytest.
        stderr : str
            Standard error from pytest.
            
        Returns
        -------
        List[TestFailure]
            List of parsed test failures.
        """
        failures = []
        combined_output = stdout + "\n" + stderr
        
        # Look for individual test failures in pytest output
        failure_sections = re.split(r"_{3,}|={3,}", combined_output)
        for section in failure_sections:
            section = section.strip()
            if not section or "FAILURES" not in section:
                continue
            
            # Extract test name
            test_name_match = re.search(r"(test_\w+)(?:\[.*\])?\s+", section)
            test_name = test_name_match.group(1) if test_name_match else "unknown_test"
            
            # Extract file path and line number
            file_path = None
            line_number = None
            file_match = re.search(r"([\w\/\.]+\.py):(\d+):", section)
            if file_match:
                file_path = file_match.group(1)
                line_number = int(file_match.group(2))
            
            # Extract error type
            error_type = "unknown"
            for pattern_name, pattern_info in self.patterns_db.items():
                for pattern in pattern_info["patterns"]:
                    if re.search(pattern, section, re.IGNORECASE):
                        error_type = pattern_info.get("error_type", pattern_name)
                        break
                if error_type != "unknown":
                    break
            
            # Extract component
            component = None
            if file_path:
                for pattern_info in self.patterns_db.values():
                    if "component_extraction" in pattern_info:
                        comp_match = re.search(pattern_info["component_extraction"], file_path)
                        if comp_match:
                            component = comp_match.group(1)
                            break
            
            # Extract expected/actual values for assertion errors
            expected_value = None
            actual_value = None
            if error_type == "assertion_error":
                expected_match = re.search(r"[Ee]xpected:?\s*['\"]?([^'\"\n]+)['\"]?", section)
                if expected_match:
                    expected_value = expected_match.group(1)
                
                actual_match = re.search(r"[Gg]ot:?\s*['\"]?([^'\"\n]+)['\"]?|[Aa]ctual:?\s*['\"]?([^'\"\n]+)['\"]?", section)
                if actual_match:
                    actual_value = actual_match.group(1) or actual_match.group(2)
            
            # Extract failure message
            failure_message = "Test failed"
            message_lines = section.splitlines()
            for i, line in enumerate(message_lines):
                if "E   " in line:  # pytest error lines start with E
                    failure_message = line.strip().replace("E   ", "")
                    break
            
            # Extract traceback
            traceback = None
            if "Traceback (most recent call last)" in section:
                traceback_start = section.index("Traceback (most recent call last)")
                traceback = section[traceback_start:].strip()
            
            # Create failure object
            failure = TestFailure(
                test_name=test_name,
                failure_message=failure_message,
                file_path=file_path,
                line_number=line_number,
                error_type=error_type,
                traceback=traceback,
                component=component,
                expected_value=expected_value,
                actual_value=actual_value
            )
            
            failures.append(failure)
        
        return failures
    
    def analyze_failures(self, stdout: str, stderr: str, exit_code: int) -> Dict[str, Any]:
        """
        Analyze test failures to provide detailed diagnostic information.
        
        Parameters
        ----------
        stdout : str
            Standard output from pytest.
        stderr : str
            Standard error from pytest.
        exit_code : int
            Exit code from pytest.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with detailed failure analysis.
        """
        # Extract failures from output
        failures = self.parse_pytest_failures(stdout, stderr)
        
        # Create base failure details
        failure_details = {
            "exit_code": exit_code,
            "error_type": "unknown",
            "error_categories": [],
            "affected_components": [],
            "error_count": len(failures)
        }
        
        # Extract error categories and components from failures
        for failure in failures:
            if failure.error_type and failure.error_type != "unknown":
                if failure.error_type not in failure_details["error_categories"]:
                    failure_details["error_categories"].append(failure.error_type)
                
                if failure.component and failure.component not in failure_details["affected_components"]:
                    failure_details["affected_components"].append(failure.component)
        
        # Determine primary error type based on most common category
        if failure_details["error_categories"]:
            # Count occurrences of each error type
            category_counts = {}
            for failure in failures:
                if failure.error_type:
                    category_counts[failure.error_type] = category_counts.get(failure.error_type, 0) + 1
            
            # Set the most common error type as the primary error type
            primary_error_type = max(category_counts.items(), key=lambda x: x[1])[0]
            failure_details["error_type"] = primary_error_type
        
        # Extract test summary from output
        test_summary = {}
        summary_match = re.search(r"(\d+) passed,? (\d+) failed,? (\d+) skipped", stdout)
        if summary_match:
            test_summary["passed"] = int(summary_match.group(1))
            test_summary["failed"] = int(summary_match.group(2))
            test_summary["skipped"] = int(summary_match.group(3))
            
            failure_details["test_summary"] = test_summary
        
        # Perform root cause analysis
        root_cause = self._determine_root_cause(failures, stdout, stderr)
        if root_cause:
            failure_details["root_cause"] = root_cause
        
        return failure_details
    
    def _determine_root_cause(self, failures: List[TestFailure], stdout: str, stderr: str) -> Optional[str]:
        """
        Analyze failures to determine the likely root cause.
        
        Parameters
        ----------
        failures : List[TestFailure]
            List of parsed test failures.
        stdout : str
            Standard output from pytest.
        stderr : str
            Standard error from pytest.
            
        Returns
        -------
        Optional[str]
            Description of the likely root cause, or None if unknown.
        """
        if not failures:
            return None
        
        # Check for common container issues
        if "Error: No such container" in stderr:
            return "Container not found. The specified service container is not running."
        
        if "Permission denied" in stderr:
            return "Permission error accessing the container or executing commands."
        
        if "Connection refused" in stdout or "Connection refused" in stderr:
            return "Connection to service failed. The service might not be ready or listening."
        
        # Check for import issues
        if any(f.error_type == "import_error" for f in failures):
            return "Missing dependencies. Required modules could not be imported."
        
        # Check for assertion failures
        assertion_failures = [f for f in failures if f.error_type == "assertion_error"]
        if assertion_failures:
            # Try to identify patterns in assertion failures
            value_mismatches = []
            for failure in assertion_failures:
                if failure.expected_value and failure.actual_value:
                    value_mismatches.append(f"Expected '{failure.expected_value}' but got '{failure.actual_value}'")
            
            if value_mismatches:
                return f"Value mismatch in assertions: {'; '.join(value_mismatches[:3])}"
            
            return "Test assertions failed, indicating incorrect behavior."
        
        # Check for timeout issues
        if any(f.error_type == "timeout_error" for f in failures):
            return "Operations timed out, indicating performance issues or deadlocks."
        
        # Default to a generic message based on the first failure
        if failures[0].failure_message:
            return f"Error in {failures[0].test_name}: {failures[0].failure_message}"
        
        return None


# ---------------------------------------------------------------------------—
# Container Stats Collection
# ---------------------------------------------------------------------------—
class ContainerStatsCollector:
    """
    Collects resource usage statistics from running containers.
    
    Uses Docker stats API to gather CPU, memory, network, and disk metrics
    during test execution.
    """
    
    def __init__(self, compose_project: str, service: str):
        """
        Initialize a ContainerStatsCollector.
        
        Parameters
        ----------
        compose_project : str
            Docker Compose project name.
        service : str
            Service name within the Docker Compose project.
        """
        self.compose_project = compose_project
        self.service = service
        self.container_id = None
        self.stats = ContainerStats()
        self.running = False
        self.collection_thread = None
    
    def start(self) -> bool:
        """
        Start collecting container statistics in a background thread.
        
        Returns
        -------
        bool
            True if collection started successfully, False otherwise.
        """
        # Find the container ID for the specified service
        try:
            container_id_cmd = [
                "docker", "ps", "--filter", 
                f"name={self.compose_project}_{self.service}", 
                "--format", "{{.ID}}"
            ]
            result = subprocess.run(
                container_id_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                check=True
            )
            container_ids = result.stdout.strip().split('\n')
            if not container_ids or not container_ids[0]:
                return False
            
            self.container_id = container_ids[0]
            self.running = True
            
            # Start collection thread
            self.collection_thread = threading.Thread(
                target=self._collect_stats,
                daemon=True
            )
            self.collection_thread.start()
            
            return True
        
        except (subprocess.SubprocessError, OSError):
            return False
    
    def stop(self) -> None:
        """Stop collecting container statistics."""
        self.running = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
    
    def _collect_stats(self) -> None:
        """
        Background thread method to periodically collect container stats.
        
        Runs until stop() is called or an error occurs.
        """
        if not self.container_id:
            return
        
        try:
            # Set up stats command
            stats_cmd = [
                "docker", "stats", "--no-stream", "--format",
                "{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}",
                self.container_id
            ]
            
            # Collect stats every second
            while self.running:
                try:
                    result = subprocess.run(
                        stats_cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        text=True, 
                        timeout=2.0
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        self._parse_stats(result.stdout.strip())
                except (subprocess.SubprocessError, ValueError, OSError):
                    # On error, sleep and try again
                    pass
                
                # Wait before next collection
                time.sleep(1.0)
        
        except Exception:
            # If anything goes wrong, just stop collecting
            self.running = False
    
    def _parse_stats(self, stats_line: str) -> None:
        """
        Parse Docker stats output and update the stats object.
        
        Parameters
        ----------
        stats_line : str
            Single line of Docker stats output.
        """
        parts = stats_line.split('\t')
        if len(parts) >= 4:
            # Parse CPU usage (e.g., "5.10%")
            cpu_match = re.match(r"([\d.]+)%", parts[0])
            if cpu_match:
                self.stats.cpu_usage_percent = float(cpu_match.group(1))
            
            # Parse memory usage (e.g., "150MiB / 1.944GiB")
            mem_parts = parts[1].split('/')
            if len(mem_parts) >= 1:
                mem_match = re.match(r"([\d.]+)([KMGT]?i?B)", mem_parts[0].strip())
                if mem_match:
                    value = float(mem_match.group(1))
                    unit = mem_match.group(2)
                    
                    # Convert to MB
                    if unit.startswith('K'):
                        value /= 1024
                    elif unit.startswith('G'):
                        value *= 1024
                    elif unit.startswith('T'):
                        value *= 1024 * 1024
                    
                    self.stats.memory_usage_mb = value
            
            # Parse network I/O (e.g., "648B / 648B")
            net_parts = parts[2].split('/')
            if len(net_parts) >= 2:
                # Parse input
                in_match = re.match(r"([\d.]+)([KMGT]?B)", net_parts[0].strip())
                if in_match:
                    value = float(in_match.group(1))
                    unit = in_match.group(2)
                    
                    # Convert to bytes
                    if unit.startswith('K'):
                        value *= 1024
                    elif unit.startswith('M'):
                        value *= 1024 * 1024
                    elif unit.startswith('G'):
                        value *= 1024 * 1024 * 1024
                    elif unit.startswith('T'):
                        value *= 1024 * 1024 * 1024 * 1024
                    
                    self.stats.network_in_bytes = int(value)
                
                # Parse output
                out_match = re.match(r"([\d.]+)([KMGT]?B)", net_parts[1].strip())
                if out_match:
                    value = float(out_match.group(1))
                    unit = out_match.group(2)
                    
                    # Convert to bytes
                    if unit.startswith('K'):
                        value *= 1024
                    elif unit.startswith('M'):
                        value *= 1024 * 1024
                    elif unit.startswith('G'):
                        value *= 1024 * 1024 * 1024
                    elif unit.startswith('T'):
                        value *= 1024 * 1024 * 1024 * 1024
                    
                    self.stats.network_out_bytes = int(value)
            
            # Parse block I/O (e.g., "0B / 0B")
            io_parts = parts[3].split('/')
            if len(io_parts) >= 2:
                # Parse read
                read_match = re.match(r"([\d.]+)([KMGT]?B)", io_parts[0].strip())
                if read_match:
                    value = float(read_match.group(1))
                    unit = read_match.group(2)
                    
                    # Convert to bytes
                    if unit.startswith('K'):
                        value *= 1024
                    elif unit.startswith('M'):
                        value *= 1024 * 1024
                    elif unit.startswith('G'):
                        value *= 1024 * 1024 * 1024
                    elif unit.startswith('T'):
                        value *= 1024 * 1024 * 1024 * 1024
                    
                    self.stats.disk_read_bytes = int(value)
                
                # Parse write
                write_match = re.match(r"([\d.]+)([KMGT]?B)", io_parts[1].strip())
                if write_match:
                    value = float(write_match.group(1))
                    unit = write_match.group(2)
                    
                    # Convert to bytes
                    if unit.startswith('K'):
                        value *= 1024
                    elif unit.startswith('M'):
                        value *= 1024 * 1024
                    elif unit.startswith('G'):
                        value *= 1024 * 1024 * 1024
                    elif unit.startswith('T'):
                        value *= 1024 * 1024 * 1024 * 1024
                    
                    self.stats.disk_write_bytes = int(value)
    
    def get_stats(self) -> ContainerStats:
        """
        Get the current container statistics.
        
        Returns
        -------
        ContainerStats
            The current container statistics.
        """
        return self.stats


# ---------------------------------------------------------------------------—
# Coverage Extractor
# ---------------------------------------------------------------------------—
class CoverageExtractor:
    """
    Extracts test coverage information from pytest output.
    
    Parses coverage reports generated by pytest-cov and extracts
    information about components, functions, lines, and branches.
    """
    
    def extract_coverage(self, stdout: str, stderr: str) -> Optional[TestCoverage]:
        """
        Extract coverage information from pytest output.
        
        Parameters
        ----------
        stdout : str
            Standard output from pytest.
        stderr : str
            Standard error from pytest.
            
        Returns
        -------
        Optional[TestCoverage]
            Extracted coverage information, or None if not available.
        """
        combined_output = stdout + "\n" + stderr
        
        # Check if we have coverage information
        if "TOTAL" not in combined_output or "%" not in combined_output:
            return None
        
        coverage = TestCoverage()
        
        # Extract tested components and functions
        test_imports = re.findall(r"import\s+(\w+(?:\.\w+)*)", combined_output)
        for module in test_imports:
            if "test" not in module:  # Skip test modules
                parts = module.split('.')
                if parts:
                    coverage.components_tested.add(parts[0])
        
        # Extract tested functions
        function_calls = re.findall(r"(\w+)\(", combined_output)
        for func in function_calls:
            if not func.startswith(("test_", "assert", "self")):
                coverage.functions_tested.add(func)
        
        # Extract line coverage
        line_coverage_match = re.search(r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%", combined_output)
        if line_coverage_match:
            statements = int(line_coverage_match.group(1))
            missed = int(line_coverage_match.group(2))
            coverage.lines_total = statements
            coverage.lines_covered = statements - missed
        
        # Extract branch coverage (if available)
        branch_coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+\d+%\s+(\d+)\s+(\d+)\s+(\d+)%", combined_output)
        if branch_coverage_match:
            branches = int(branch_coverage_match.group(1))
            missed = int(branch_coverage_match.group(2))
            coverage.branches_total = branches
            coverage.branches_covered = branches - missed
        
        return coverage


# ---------------------------------------------------------------------------—
# Smoke Test Runner
# ---------------------------------------------------------------------------—
class SmokeTestRunner:
    """
    Enhanced smoke test runner with detailed reporting.
    
    Features:
    - Detailed failure analysis
    - Container resource monitoring
    - Coverage extraction
    - Test categorization
    - Results storage
    """
    
    def __init__(self, compose_project: str, service: str, 
                pytest_args: Optional[List[str]] = None,
                timeout: float = 60.0, 
                max_tokens: int = 4096,
                planner_bug_id: Optional[str] = None,
                planner_execution_id: Optional[str] = None):
        """
        Initialize a SmokeTestRunner.
        
        Parameters
        ----------
        compose_project : str
            Docker Compose project name.
        service : str
            Service name within the Docker Compose project.
        pytest_args : Optional[List[str]]
            Additional arguments to pass to pytest. Defaults to ["tests/smoke", "-q"].
        timeout : float
            Maximum time in seconds to allow for test execution. Defaults to 60 seconds.
        max_tokens : int
            Maximum number of tokens for compressed log output. Defaults to 4096.
        planner_bug_id : Optional[str]
            Optional ID for integration with a planner system.
        planner_execution_id : Optional[str]
            Optional execution ID for integration with a planner system.
        """
        self.compose_project = compose_project
        self.service = service
        self.pytest_args = pytest_args or ["tests/smoke", "-q"]
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.planner_bug_id = planner_bug_id
        self.planner_execution_id = planner_execution_id
        
        self.failure_analyzer = FailureAnalyzer()
        self.stats_collector = ContainerStatsCollector(compose_project, service)
        self.coverage_extractor = CoverageExtractor()
        
        self.result = None
        self._process = None
        self._timer = None
    
    def run(self) -> SmokeResult:
        """
        Run smoke tests and return detailed results.
        
        Returns
        -------
        SmokeResult
            Detailed test result.
        """
        start_time = time.time()
        
        # Start container stats collection
        stats_collecting = self.stats_collector.start()
        
        try:
            # Build the docker-compose exec command
            cmd = ["docker-compose", "-p", self.compose_project, "exec", 
                  "-T", self.service, "python", "-m", "pytest"] + self.pytest_args
            
            # Start the process
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Set up timer for timeout
            self._timer = threading.Timer(self.timeout, self._kill_process)
            self._timer.daemon = True
            self._timer.start()
            
            # Wait for process to complete or timeout
            stdout, stderr = self._process.communicate()
            exit_code = self._process.returncode
            
            # Cancel timer if process completed
            if self._timer.is_alive():
                self._timer.cancel()
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Process results
            if exit_code == 0:
                # Success case
                result = SmokeResult.create_success(self.compose_project, self.service)
                result.duration_seconds = duration
                
                # Extract test counts from output
                summary_match = re.search(r"(\d+) passed(?:,\s*(\d+) skipped)?", stdout)
                if summary_match:
                    passed = int(summary_match.group(1))
                    skipped = int(summary_match.group(2)) if summary_match.group(2) else 0
                    result.test_count = passed + skipped
                    result.tests_passed = passed
                    result.tests_skipped = skipped
            
            elif exit_code == 124 or (self._process.returncode == -9 and duration >= self.timeout):
                # Timeout case
                result = SmokeResult.create_timeout(
                    self.compose_project, self.service, 
                    self.timeout, stdout
                )
                
            elif "No such container" in stderr or "not found" in stderr:
                # Container not found case
                result = SmokeResult.create_container_error(
                    self.compose_project, self.service,
                    f"Container '{self.compose_project}_{self.service}' not found or not running"
                )
                
            else:
                # Failure case - perform detailed analysis
                failure_details = self.failure_analyzer.analyze_failures(stdout, stderr, exit_code)
                
                # Extract test counts
                test_summary = failure_details.get("test_summary", {})
                passed = test_summary.get("passed", 0)
                failed = test_summary.get("failed", 0)
                skipped = test_summary.get("skipped", 0)
                test_count = passed + failed + skipped
                
                # Parse specific failures
                failures = self.failure_analyzer.parse_pytest_failures(stdout, stderr)
                
                if test_count > 0 and passed > 0:
                    # Partial success case
                    result = SmokeResult.create_partial_success(
                        self.compose_project, self.service,
                        test_count, passed, failed, skipped, failures
                    )
                else:
                    # Complete failure case
                    result = SmokeResult.create_failure(
                        self.compose_project, self.service,
                        stdout + "\n" + stderr, exit_code, failure_details
                    )
                
                result.failures = failures
                result.test_count = test_count
                result.tests_passed = passed
                result.tests_failed = failed
                result.tests_skipped = skipped
                
                if "root_cause" in failure_details:
                    result.root_cause = failure_details["root_cause"]
            
            # Common result processing
            result.duration_seconds = duration
            result.exit_code = exit_code
            
            # Get container ID and stats if available
            if stats_collecting:
                result.container_id = self.stats_collector.container_id
                result.container_stats = self.stats_collector.get_stats()
            
            # Extract coverage information if available
            coverage = self.coverage_extractor.extract_coverage(stdout, stderr)
            if coverage:
                result.coverage = coverage
            
            # Set planner integration IDs if provided
            if self.planner_bug_id:
                result.planner_bug_id = self.planner_bug_id
            if self.planner_execution_id:
                result.planner_execution_id = self.planner_execution_id
            
            # Compress logs if needed
            if stdout or stderr:
                combined_log = stdout + "\n" + stderr
                result.original_log_size = len(combined_log)
                
                if len(combined_log) > 0:
                    compressor = Compressor()
                    compressed_log = compressor.compress(combined_log, self.max_tokens)
                    result.log = compressed_log
                    result.compressed_log_size = len(compressed_log)
                    
                    if result.original_log_size > 0:
                        result.compression_ratio = result.compressed_log_size / result.original_log_size
            
            self.result = result
            return result
            
        except Exception as e:
            # Handle any unexpected errors
            error_msg = f"Error running smoke tests: {str(e)}"
            result = SmokeResult.create_error(
                self.compose_project, self.service, error_msg
            )
            result.duration_seconds = time.time() - start_time
            
            self.result = result
            return result
            
        finally:
            # Clean up
            self.stats_collector.stop()
            
            if self._timer and self._timer.is_alive():
                self._timer.cancel()
            
            if self._process and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
    
    def _kill_process(self) -> None:
        """Kill the running process due to timeout."""
        if self._process and self._process.poll() is None:
            self._process.kill()


# ---------------------------------------------------------------------------—
# Legacy API for backward compatibility
# ---------------------------------------------------------------------------—
def run_smoke_tests(compose_project: str, service: str, 
                   max_tokens: int = 4096, 
                   timeout: float = 60.0) -> Tuple[bool, str]:
    """
    Run smoke tests and return simplified results.
    
    This function maintains compatibility with the original API,
    returning a simple (success, log) tuple.
    
    Parameters
    ----------
    compose_project : str
        Docker Compose project name.
    service : str
        Service name within the Docker Compose project.
    max_tokens : int
        Maximum number of tokens for compressed log output. Defaults to 4096.
    timeout : float
        Maximum time in seconds to allow for test execution. Defaults to 60 seconds.
        
    Returns
    -------
    Tuple[bool, str]
        Tuple of (success, log) where success is True if tests passed,
        and log is an empty string on success or compressed error output.
    """
    runner = SmokeTestRunner(
        compose_project=compose_project,
        service=service,
        max_tokens=max_tokens,
        timeout=timeout
    )
    
    result = runner.run()
    return result.success, result.log


# ---------------------------------------------------------------------------—
# Command-line interface
# ---------------------------------------------------------------------------—
def main() -> None:
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run smoke tests inside a Docker Compose service container."
    )
    parser.add_argument(
        "--project", "-p", 
        required=True, 
        help="Docker Compose project name"
    )
    parser.add_argument(
        "--service", "-s", 
        required=True, 
        help="Service name within Docker Compose project"
    )
    parser.add_argument(
        "--max-tokens", "-m", 
        type=int, 
        default=4096, 
        help="Maximum number of tokens in compressed output (default: 4096)"
    )
    parser.add_argument(
        "--timeout", "-t", 
        type=float, 
        default=60.0, 
        help="Maximum execution time in seconds (default: 60.0)"
    )
    parser.add_argument(
        "--json", "-j", 
        action="store_true", 
        help="Output detailed results as JSON"
    )
    parser.add_argument(
        "--junit", 
        action="store_true", 
        help="Output results in JUnit XML format"
    )
    parser.add_argument(
        "--planner-bug-id", 
        help="Optional bug ID for integration with planner systems"
    )
    parser.add_argument(
        "--planner-execution-id", 
        help="Optional execution ID for integration with planner systems"
    )
    parser.add_argument(
        "pytest_args", 
        nargs="*", 
        default=["tests/smoke", "-q"], 
        help="Arguments to pass to pytest (default: tests/smoke -q)"
    )
    
    args = parser.parse_args()
    
    runner = SmokeTestRunner(
        compose_project=args.project,
        service=args.service,
        pytest_args=args.pytest_args,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        planner_bug_id=args.planner_bug_id,
        planner_execution_id=args.planner_execution_id
    )
    
    result = runner.run()
    
    if args.json:
        print(result.to_json())
    elif args.junit:
        print(result.to_junit_xml())
    else:
        if result.success:
            print(f"✅ All tests passed ({result.tests_passed} tests, {result.duration_seconds:.2f}s)")
            sys.exit(0)
        else:
            print(f"❌ Tests failed: {result.result_type.value}")
            print(f"   Passed: {result.tests_passed}, Failed: {result.tests_failed}, Skipped: {result.tests_skipped}")
            
            if result.failures:
                print("\nFailures:")
                for i, failure in enumerate(result.failures[:5], 1):
                    print(f"{i}. {failure.test_name}: {failure.failure_message}")
                
                if len(result.failures) > 5:
                    print(f"... and {len(result.failures) - 5} more failures")
            
            if result.root_cause:
                print(f"\nRoot cause: {result.root_cause}")
            
            print(f"\nLog output:\n{result.log}")
            sys.exit(1)


# ---------------------------------------------------------------------------—
# Module entry point
# ---------------------------------------------------------------------------—
if __name__ == "__main__":
    main()
