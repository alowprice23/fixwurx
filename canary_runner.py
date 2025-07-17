"""
tooling/canary_runner.py
────────────────────────
Runs a single, isolated "canary" test to validate a repair.

This is a lightweight, targeted alternative to a full test suite run.
It's designed to be fast and focused, providing a quick feedback loop
on whether a specific fix has resolved the intended issue without
introducing obvious regressions.

Key Features:
 • Single-test execution
 • Advanced timeout handling with multiple strategies
 • Detailed failure analysis and reporting
 • Planner integration for test selection and result reporting
 • Historical failure pattern detection
 • Configurable resource limits and monitoring
 • Supports multiple test frameworks (pytest, unittest, etc.)

API
───
    runner = CanaryTestRunner(timeout=10.0)
    result = runner.run(
        test_command=["pytest", "tests/test_specific_bug.py"]
    )
    print(result)
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import threading
import time
import traceback
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
import shutil
import logging
import uuid
from typing import List, Optional, Dict, Any, Union, Callable, Tuple, Set


# Configure logging
logger = logging.getLogger("canary_runner")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Result Types
# ──────────────────────────────────────────────────────────────────────────────
class CanaryResultType(Enum):
    """Enum representing the possible outcomes of a canary test run."""
    PASS = "pass"                 # Test passed successfully
    FAIL = "fail"                 # Test failed with assertion/expectation errors
    TIMEOUT = "timeout"           # Test exceeded the timeout limit
    ERROR = "error"               # Test had an execution error (e.g., syntax error)
    RESOURCE_EXCEEDED = "resource_exceeded"  # Test exceeded resource limits
    SKIPPED = "skipped"           # Test was skipped
    INTERRUPTED = "interrupted"   # Test was interrupted
    NOT_FOUND = "not_found"       # Test file or command not found
    UNSTABLE = "unstable"         # Test is flaky (passes sometimes, fails others)
    UNKNOWN = "unknown"           # Unknown outcome


class TimeoutStrategy(Enum):
    """Enum representing different timeout strategies."""
    FIXED = "fixed"               # Fixed timeout value
    ADAPTIVE = "adaptive"         # Adjust timeout based on historical runs
    PERCENTAGE = "percentage"     # Timeout as a percentage of average run time
    PROGRESSIVE = "progressive"   # Gradually increase timeout during reruns
    STEP_BASED = "step_based"     # Different timeouts for different test phases


@dataclass
class ResourceUsage:
    """Tracks resource usage during test execution."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    cpu_readings: List[float] = field(default_factory=list)
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    thread_count: int = 0
    
    @property
    def duration(self) -> float:
        """Return the duration of the test in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert resource usage to a dictionary."""
        return {
            "duration": self.duration,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_cpu_percent": self.avg_cpu_percent,
            "max_cpu_percent": self.max_cpu_percent,
            "io_read_mb": self.io_read_bytes / (1024 * 1024) if self.io_read_bytes else 0,
            "io_write_mb": self.io_write_bytes / (1024 * 1024) if self.io_write_bytes else 0,
            "thread_count": self.thread_count
        }


@dataclass
class TestFixSuggestion:
    """Represents a suggested fix for a failing test."""
    suggestion_type: str  # e.g., "code_change", "dependency", "configuration"
    description: str
    code_snippet: Optional[str] = None
    confidence: float = 0.0
    related_errors: List[str] = field(default_factory=list)
    planner_path_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the suggestion to a dictionary."""
        return {
            "suggestion_type": self.suggestion_type,
            "description": self.description,
            "code_snippet": self.code_snippet,
            "confidence": self.confidence,
            "related_errors": self.related_errors,
            "planner_path_id": self.planner_path_id
        }


@dataclass
class CanaryResult:
    """
    Enhanced result data for a canary test run.
    
    Provides detailed information about test execution, including 
    resource usage, failure details, and categorical result types.
    """
    # Basic result information
    result_type: CanaryResultType
    output: str
    error: Optional[str] = None
    
    # Detailed execution information
    command: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    return_value: Any = None
    
    # Resource tracking
    resources: ResourceUsage = field(default_factory=ResourceUsage)
    
    # Failure analysis
    failure_details: Dict[str, Any] = field(default_factory=dict)
    stacktrace: Optional[str] = None
    root_cause_analysis: Optional[str] = None
    fix_suggestions: List[TestFixSuggestion] = field(default_factory=list)
    
    # Test metadata
    test_id: Optional[str] = None
    test_file: Optional[str] = None
    test_function: Optional[str] = None
    test_framework: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    environment: Dict[str, str] = field(default_factory=lambda: {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "processor": platform.processor()
    })
    
    # Retry information
    retry_count: int = 0
    retry_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Planner integration
    planner_bug_id: Optional[str] = None
    planner_execution_id: Optional[str] = None
    
    @property
    def passed(self) -> bool:
        """Return True if the test passed."""
        return self.result_type == CanaryResultType.PASS
    
    @property
    def timed_out(self) -> bool:
        """Return True if the test timed out."""
        return self.result_type == CanaryResultType.TIMEOUT
    
    @property
    def is_unstable(self) -> bool:
        """Return True if the test is unstable (flaky)."""
        return self.result_type == CanaryResultType.UNSTABLE
    
    def extract_test_info(self) -> None:
        """Extract test file and function information from command and output."""
        # Extract test file from command if possible
        for arg in self.command:
            if arg.endswith('.py') and 'test' in arg.lower():
                self.test_file = arg
                break
        
        # Try to determine test framework
        if 'pytest' in ' '.join(self.command):
            self.test_framework = 'pytest'
        elif 'unittest' in ' '.join(self.command):
            self.test_framework = 'unittest'
        
        # Extract test function from output if possible
        if self.test_framework == 'pytest':
            # Look for patterns like "test_function_name FAILED"
            pattern = r'(\w+)\s+FAILED'
            matches = re.findall(pattern, self.output)
            if matches:
                self.test_function = matches[0]
        
        # If we have a test file but not a function, try to extract from stacktrace
        if self.test_file and not self.test_function and self.stacktrace:
            # Look for patterns like "def test_function_name"
            pattern = r'def\s+(\w+)'
            matches = re.findall(pattern, self.stacktrace)
            if matches:
                self.test_function = matches[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary for serialization."""
        # Extract test info if not already done
        if not self.test_file and not self.test_function:
            self.extract_test_info()
            
        result = {
            "result_type": self.result_type.value,
            "passed": self.passed,
            "timed_out": self.timed_out,
            "is_unstable": self.is_unstable,
            "output": self.output,
            "error": self.error,
            "command": self.command,
            "exit_code": self.exit_code,
            "resources": self.resources.to_dict(),
            "failure_details": self.failure_details,
            "stacktrace": self.stacktrace,
            "root_cause_analysis": self.root_cause_analysis,
            "fix_suggestions": [s.to_dict() for s in self.fix_suggestions],
            "test_id": self.test_id,
            "test_file": self.test_file,
            "test_function": self.test_function,
            "test_framework": self.test_framework,
            "timestamp": self.timestamp,
            "environment": self.environment,
            "retry_count": self.retry_count,
            "retry_results": self.retry_results,
            "planner_bug_id": self.planner_bug_id,
            "planner_execution_id": self.planner_execution_id
        }
        return result
    
    def to_json(self) -> str:
        """Convert the result to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanaryResult:
        """Create a CanaryResult from a dictionary."""
        # Create a copy to avoid modifying the input
        data_copy = data.copy()
        
        # Convert string result_type back to enum
        if "result_type" in data_copy:
            data_copy["result_type"] = CanaryResultType(data_copy["result_type"])
        
        # Create ResourceUsage instance
        if "resources" in data_copy:
            resources = data_copy.pop("resources")
            data_copy["resources"] = ResourceUsage(
                start_time=resources.get("start_time", time.time() - resources.get("duration", 0)),
                end_time=resources.get("start_time", time.time() - resources.get("duration", 0)) + resources.get("duration", 0),
                peak_memory_mb=resources.get("peak_memory_mb", 0.0),
                avg_cpu_percent=resources.get("avg_cpu_percent", 0.0),
                max_cpu_percent=resources.get("max_cpu_percent", 0.0),
                io_read_bytes=int(resources.get("io_read_mb", 0) * 1024 * 1024),
                io_write_bytes=int(resources.get("io_write_mb", 0) * 1024 * 1024),
                thread_count=resources.get("thread_count", 0)
            )
        
        # Convert fix suggestions
        if "fix_suggestions" in data_copy:
            suggestions = data_copy.pop("fix_suggestions")
            data_copy["fix_suggestions"] = [
                TestFixSuggestion(**suggestion) for suggestion in suggestions
            ]
        
        return cls(**data_copy)
    
    @classmethod
    def create_success(cls, output: str, command: List[str], exit_code: int = 0) -> CanaryResult:
        """Create a successful result."""
        resources = ResourceUsage()
        resources.end_time = time.time()
        
        return cls(
            result_type=CanaryResultType.PASS,
            output=output,
            command=command,
            exit_code=exit_code,
            resources=resources
        )
    
    @classmethod
    def create_failure(cls, output: str, error: str, command: List[str], 
                      exit_code: int, failure_details: Dict[str, Any] = None) -> CanaryResult:
        """Create a failure result."""
        resources = ResourceUsage()
        resources.end_time = time.time()
        
        return cls(
            result_type=CanaryResultType.FAIL,
            output=output,
            error=error,
            command=command,
            exit_code=exit_code,
            failure_details=failure_details or {},
            resources=resources
        )
    
    @classmethod
    def create_timeout(cls, command: List[str], timeout: float, 
                     partial_output: str = "", partial_error: str = "") -> CanaryResult:
        """Create a timeout result."""
        resources = ResourceUsage()
        resources.end_time = time.time()
        
        return cls(
            result_type=CanaryResultType.TIMEOUT,
            output=partial_output + f"\n\nTest process timed out after {timeout} seconds.",
            error=partial_error + f"\nTimeout after {timeout} seconds",
            command=command,
            failure_details={"timeout_seconds": timeout},
            resources=resources
        )
    
    @classmethod
    def create_resource_exceeded(cls, command: List[str], 
                              resource_type: str, limit: float,
                              actual: float) -> CanaryResult:
        """Create a resource exceeded result."""
        resources = ResourceUsage()
        resources.end_time = time.time()
        
        return cls(
            result_type=CanaryResultType.RESOURCE_EXCEEDED,
            output=f"Test exceeded {resource_type} limit: {actual} > {limit}",
            error=f"Resource limit exceeded: {resource_type}",
            command=command,
            failure_details={
                "resource_type": resource_type,
                "limit": limit,
                "actual": actual
            },
            resources=resources
        )
    
    @classmethod
    def create_error(cls, error: str, command: List[str], output: str = "") -> CanaryResult:
        """Create an error result."""
        resources = ResourceUsage()
        resources.end_time = time.time()
        
        return cls(
            result_type=CanaryResultType.ERROR,
            output=output,
            error=error,
            command=command,
            stacktrace=traceback.format_exc(),
            resources=resources
        )
    
    @classmethod
    def create_not_found(cls, command: List[str]) -> CanaryResult:
        """Create a not found result."""
        resources = ResourceUsage()
        resources.end_time = time.time()
        
        return cls(
            result_type=CanaryResultType.NOT_FOUND,
            output="",
            error=f"Command not found: {command[0]}",
            command=command,
            resources=resources
        )
    
    @classmethod
    def create_unstable(cls, command: List[str], results: List[CanaryResult]) -> CanaryResult:
        """Create an unstable (flaky) test result."""
        resources = ResourceUsage()
        resources.end_time = time.time()
        
        # Calculate pass rate
        pass_count = sum(1 for r in results if r.passed)
        pass_rate = pass_count / len(results) if results else 0
        
        # Create combined output
        combined_output = f"Test is UNSTABLE (flaky). Pass rate: {pass_rate:.2%}\n\n"
        combined_output += "Run history:\n"
        for i, r in enumerate(results):
            combined_output += f"Run {i+1}: {r.result_type.value.upper()}\n"
        
        # Gather retry results for detailed analysis
        retry_results = [r.to_dict() for r in results]
        
        return cls(
            result_type=CanaryResultType.UNSTABLE,
            output=combined_output,
            error="Test is unstable (passes sometimes, fails others)",
            command=command,
            failure_details={
                "pass_rate": pass_rate,
                "run_count": len(results),
                "pass_count": pass_count,
                "fail_count": len(results) - pass_count
            },
            resources=resources,
            retry_results=retry_results,
            retry_count=len(results)
        )


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Advanced Timeout Handling
# ──────────────────────────────────────────────────────────────────────────────
class TimeoutHandler:
    """
    Advanced timeout handling for test execution.
    
    Features:
    - Multiple timeout strategies
    - Adaptive timeout based on historical execution times
    - Progressive timeout during retries
    - Phase-based timeouts
    """
    
    def __init__(
        self,
        strategy: TimeoutStrategy = TimeoutStrategy.FIXED,
        base_timeout: float = 10.0,
        history_length: int = 5,
        timeout_factor: float = 1.5,
        min_timeout: float = 1.0,
        max_timeout: float = 60.0,
        phase_timeouts: Dict[str, float] = None
    ):
        """
        Initialize a TimeoutHandler.
        
        Parameters
        ----------
        strategy : TimeoutStrategy
            The timeout strategy to use.
        base_timeout : float
            The base timeout value in seconds.
        history_length : int
            Number of historical runs to consider for adaptive timeouts.
        timeout_factor : float
            Multiplier for adaptive and percentage-based strategies.
        min_timeout : float
            Minimum timeout value in seconds.
        max_timeout : float
            Maximum timeout value in seconds.
        phase_timeouts : Dict[str, float]
            Timeout values for different test phases (for step-based strategy).
        """
        self.strategy = strategy
        self.base_timeout = base_timeout
        self.history_length = history_length
        self.timeout_factor = timeout_factor
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.phase_timeouts = phase_timeouts or {
            "startup": base_timeout * 0.2,
            "execution": base_timeout * 0.7,
            "teardown": base_timeout * 0.1
        }
        
        # Historical execution times
        self.history: List[float] = []
    
    def get_timeout(self, retry_count: int = 0, phase: str = "execution") -> float:
        """
        Get the appropriate timeout value based on the strategy.
        
        Parameters
        ----------
        retry_count : int
            The current retry attempt number (0 for first run).
        phase : str
            The current test phase (for step-based strategy).
            
        Returns
        -------
        float
            The timeout value in seconds.
        """
        if self.strategy == TimeoutStrategy.FIXED:
            return self.base_timeout
        
        elif self.strategy == TimeoutStrategy.ADAPTIVE:
            if not self.history:
                return self.base_timeout
            
            # Use average of recent runs, with a safety factor
            avg_time = sum(self.history[-self.history_length:]) / min(len(self.history), self.history_length)
            timeout = avg_time * self.timeout_factor
            
            return max(self.min_timeout, min(self.max_timeout, timeout))
        
        elif self.strategy == TimeoutStrategy.PERCENTAGE:
            if not self.history:
                return self.base_timeout
            
            # Use a percentage of the maximum observed run time
            max_time = max(self.history[-self.history_length:])
            timeout = max_time * self.timeout_factor
            
            return max(self.min_timeout, min(self.max_timeout, timeout))
        
        elif self.strategy == TimeoutStrategy.PROGRESSIVE:
            # Increase timeout for each retry
            factor = 1.0 + (retry_count * 0.5)  # 1.0, 1.5, 2.0, 2.5, ...
            timeout = self.base_timeout * factor
            
            return max(self.min_timeout, min(self.max_timeout, timeout))
        
        elif self.strategy == TimeoutStrategy.STEP_BASED:
            # Use different timeouts for different phases
            if phase in self.phase_timeouts:
                return self.phase_timeouts[phase]
            else:
                return self.phase_timeouts.get("execution", self.base_timeout)
        
        # Default to base timeout
        return self.base_timeout
    
    def record_execution_time(self, duration: float) -> None:
        """
        Record the execution time of a test run.
        
        Parameters
        ----------
        duration : float
            The execution time in seconds.
        """
        self.history.append(duration)
        
        # Trim history to keep only the most recent entries
        if len(self.history) > self.history_length * 2:
            self.history = self.history[-self.history_length:]
    
    def reset_history(self) -> None:
        """Reset the execution time history."""
        self.history.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the timeout handler to a dictionary."""
        return {
            "strategy": self.strategy.value,
            "base_timeout": self.base_timeout,
            "history_length": self.history_length,
            "timeout_factor": self.timeout_factor,
            "min_timeout": self.min_timeout,
            "max_timeout": self.max_timeout,
            "phase_timeouts": self.phase_timeouts,
            "history": self.history.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeoutHandler":
        """Create a TimeoutHandler from a dictionary."""
        strategy = TimeoutStrategy(data.get("strategy", TimeoutStrategy.FIXED.value))
        handler = cls(
            strategy=strategy,
            base_timeout=data.get("base_timeout", 10.0),
            history_length=data.get("history_length", 5),
            timeout_factor=data.get("timeout_factor", 1.5),
            min_timeout=data.get("min_timeout", 1.0),
            max_timeout=data.get("max_timeout", 60.0),
            phase_timeouts=data.get("phase_timeouts", None)
        )
        handler.history = data.get("history", []).copy()
        return handler


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Advanced Failure Analysis
# ──────────────────────────────────────────────────────────────────────────────
class FailureAnalyzer:
    """
    Advanced failure analysis for test results.
    
    Features:
    - Detailed error classification
    - Root cause analysis
    - Fix suggestions
    - Historical failure pattern detection
    """
    
    def __init__(
        self,
        error_patterns: Dict[str, Dict[str, Any]] = None,
        historical_failures: Dict[str, List[Dict[str, Any]]] = None,
        root_cause_threshold: int = 3,
        planner_agent = None
    ):
        """
        Initialize a FailureAnalyzer.
        
        Parameters
        ----------
        error_patterns : Dict[str, Dict[str, Any]]
            Known error patterns and their classifications.
        historical_failures : Dict[str, List[Dict[str, Any]]]
            Historical failure data for pattern detection.
        root_cause_threshold : int
            Minimum number of similar failures to consider a pattern.
        planner_agent : Any
            Optional planner agent for integration.
        """
        self.error_patterns = error_patterns or self._get_default_error_patterns()
        self.historical_failures = historical_failures or {}
        self.root_cause_threshold = root_cause_threshold
        self.planner_agent = planner_agent
    
    def _get_default_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get default error patterns for common test failures."""
        return {
            "assertion_error": {
                "patterns": [r"AssertionError", r"assert\s+"],
                "suggestion_type": "code_change",
                "description": "The test failed because an assertion didn't match expectations."
            },
            "syntax_error": {
                "patterns": [r"SyntaxError", r"IndentationError"],
                "suggestion_type": "code_change",
                "description": "There's a syntax error in the code."
            },
            "import_error": {
                "patterns": [r"ImportError", r"ModuleNotFoundError"],
                "suggestion_type": "dependency",
                "description": "A required module or import is missing."
            },
            "permission_error": {
                "patterns": [r"PermissionError", r"Access denied"],
                "suggestion_type": "configuration",
                "description": "The test lacks permission to access a resource."
            },
            "timeout_error": {
                "patterns": [r"TimeoutError", r"timed out"],
                "suggestion_type": "performance",
                "description": "The test exceeded its time limit."
            },
            "type_error": {
                "patterns": [r"TypeError", r"expected\s+\w+\s+but\s+got\s+\w+"],
                "suggestion_type": "code_change",
                "description": "There's a type mismatch in the code."
            },
            "attribute_error": {
                "patterns": [r"AttributeError", r"has no attribute"],
                "suggestion_type": "code_change",
                "description": "The code is trying to access a non-existent attribute."
            },
            "key_error": {
                "patterns": [r"KeyError", r"key\s+\w+\s+not found"],
                "suggestion_type": "code_change",
                "description": "The code is trying to access a non-existent dictionary key."
            },
            "io_error": {
                "patterns": [r"FileNotFoundError", r"IOError", r"No such file or directory"],
                "suggestion_type": "configuration",
                "description": "A required file is missing or inaccessible."
            },
            "value_error": {
                "patterns": [r"ValueError", r"invalid value"],
                "suggestion_type": "code_change",
                "description": "The code received an inappropriate value."
            }
        }
    
    def analyze_failure(self, stdout: str, stderr: str, exit_code: int) -> Dict[str, Any]:
        """
        Analyze test failure to provide detailed diagnostic information.
        
        Parameters
        ----------
        stdout : str
            Standard output from the process.
        stderr : str
            Standard error output from the process.
        exit_code : int
            Exit code from the process.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with detailed failure analysis.
        """
        combined_output = stdout + "\n" + stderr
        
        failure_details = {
            "exit_code": exit_code,
            "error_type": "unknown",
            "error_message": "",
            "error_location": "",
            "test_status": "",
            "first_error_line": None,
            "first_failure_line": None,
            "error_context": None,
            "test_phase": "execution"
        }
        
        # Look for the test phase
        if "setup" in combined_output.lower():
            failure_details["test_phase"] = "setup"
        elif "teardown" in combined_output.lower():
            failure_details["test_phase"] = "teardown"
        
        # Extract error type from known patterns
        for error_type, error_info in self.error_patterns.items():
            for pattern in error_info["patterns"]:
                if re.search(pattern, combined_output, re.IGNORECASE):
                    failure_details["error_type"] = error_type
                    break
            if failure_details["error_type"] != "unknown":
                break
        
        # Look for pytest-specific patterns
        if "FAILED" in stdout:
            failure_details["test_status"] = "failed"
            
            # Try to extract error message and location
            error_lines = [line for line in stdout.splitlines() if "FAILED" in line]
            if error_lines:
                failure_details["error_message"] = error_lines[0]
                failure_details["first_failure_line"] = error_lines[0]
            
            # Extract assertion error details
            if "AssertionError" in stdout:
                failure_details["error_type"] = "assertion_error"
                # Find the assertion error line and surrounding context
                for i, line in enumerate(stdout.splitlines()):
                    if "AssertionError" in line:
                        failure_details["first_error_line"] = line
                        context_start = max(0, i - 3)
                        context_end = min(len(stdout.splitlines()), i + 5)
                        failure_details["error_context"] = "\n".join(
                            stdout.splitlines()[context_start:context_end]
                        )
                        break
        
        # Look for traceback information
        traceback_start = -1
        traceback_end = -1
        for i, line in enumerate(combined_output.splitlines()):
            if "Traceback (most recent call last)" in line:
                traceback_start = i
            elif traceback_start >= 0 and line.strip() and not line.startswith(" "):
                traceback_end = i
                # Extract the error message
                if failure_details["error_message"] == "":
                    failure_details["error_message"] = line.strip()
                break
        
        # Extract file and line information from traceback
        if traceback_start >= 0:
            for line in combined_output.splitlines()[traceback_start:traceback_end]:
                if ", line " in line and "File " in line:
                    failure_details["error_location"] = line.strip()
                    break
        
        return failure_details
    
    def generate_fix_suggestions(self, failure_details: Dict[str, Any], 
                               output: str, error: str) -> List[TestFixSuggestion]:
        """
        Generate suggested fixes for the failure.
        
        Parameters
        ----------
        failure_details : Dict[str, Any]
            Detailed failure analysis.
        output : str
            Standard output from the process.
        error : str
            Standard error output from the process.
            
        Returns
        -------
        List[TestFixSuggestion]
            List of suggested fixes.
        """
        suggestions = []
        
        # Get basic suggestion from error type
        error_type = failure_details.get("error_type", "unknown")
        if error_type in self.error_patterns:
            pattern_info = self.error_patterns[error_type]
            suggestion = TestFixSuggestion(
                suggestion_type=pattern_info["suggestion_type"],
                description=pattern_info["description"],
                confidence=0.7
            )
            suggestions.append(suggestion)
        
        # Add specific suggestions based on error type
        if error_type == "assertion_error":
            # Try to extract expected vs actual values
            error_context = failure_details.get("error_context", "")
            if "E   assert " in error_context:
                # Extract the assertion line
                for line in error_context.splitlines():
                    if line.strip().startswith("E   assert "):
                        assertion_line = line.strip()[len("E   assert "):]
                        expected_actual = self._parse_assertion_error(assertion_line)
                        if expected_actual:
                            expected, actual = expected_actual
                            suggestion = TestFixSuggestion(
                                suggestion_type="code_change",
                                description=f"Update test to expect {actual} instead of {expected}",
                                confidence=0.6,
                                code_snippet=f"assert value == {actual}  # Was expecting {expected}"
                            )
                            suggestions.append(suggestion)
                        break
        
        elif error_type == "import_error":
            # Suggest installing missing dependency
            missing_module = ""
            if "No module named" in error:
                match = re.search(r"No module named '([^']+)'", error)
                if match:
                    missing_module = match.group(1)
                    suggestion = TestFixSuggestion(
                        suggestion_type="dependency",
                        description=f"Install the missing module: {missing_module}",
                        confidence=0.9,
                        code_snippet=f"pip install {missing_module}"
                    )
                    suggestions.append(suggestion)
        
        elif error_type == "timeout_error":
            # Suggest increasing timeout
            suggestion = TestFixSuggestion(
                suggestion_type="configuration",
                description="Increase the test timeout",
                confidence=0.8,
                code_snippet="@pytest.mark.timeout(30)  # Increase from default"
            )
            suggestions.append(suggestion)
        
        # Use planner for complex suggestions if available
        if self.planner_agent and len(suggestions) < 2:
            try:
                # Attempt to get suggestions from planner
                planner_suggestions = self._get_planner_suggestions(
                    failure_details, output, error
                )
                if planner_suggestions:
                    suggestions.extend(planner_suggestions)
            except Exception as e:
                logger.warning(f"Error getting planner suggestions: {e}")
        
        return suggestions
    
    def _parse_assertion_error(self, assertion_line: str) -> Optional[Tuple[str, str]]:
        """
        Parse an assertion error line to extract expected and actual values.
        
        Parameters
        ----------
        assertion_line : str
            The assertion line from the error output.
            
        Returns
        -------
        Optional[Tuple[str, str]]
            Tuple of (expected, actual) values, or None if parsing failed.
        """
        # Common pytest assertion patterns
        patterns = [
            # assert a == b
            r"assert (.+) == (.+)",
            # assert a is b
            r"assert (.+) is (.+)",
            # assert a in b
            r"assert (.+) in (.+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, assertion_line)
            if match:
                if len(match.groups()) == 2:
                    return match.group(1).strip(), match.group(2).strip()
        
        return None
    
    def _get_planner_suggestions(self, failure_details: Dict[str, Any], 
                              output: str, error: str) -> List[TestFixSuggestion]:
        """
        Get fix suggestions from the planner agent.
        
        Parameters
        ----------
        failure_details : Dict[str, Any]
            Detailed failure analysis.
        output : str
            Standard output from the process.
        error : str
            Standard error output from the process.
            
        Returns
        -------
        List[TestFixSuggestion]
            List of suggested fixes from the planner.
        """
        if not self.planner_agent:
            return []
        
        try:
            # Call planner method to generate suggestions
            # This will depend on the planner agent's API
            planner_data = {
                "failure_details": failure_details,
                "output": output,
                "error": error
            }
            
            # The planner agent should have a method like generate_fix_suggestions
            if hasattr(self.planner_agent, "generate_fix_suggestions"):
                planner_response = self.planner_agent.generate_fix_suggestions(planner_data)
                
                # Convert planner response to TestFixSuggestion objects
                suggestions = []
                for item in planner_response.get("suggestions", []):
                    suggestion = TestFixSuggestion(
                        suggestion_type=item.get("type", "code_change"),
                        description=item.get("description", ""),
                        code_snippet=item.get("code", None),
                        confidence=item.get("confidence", 0.5),
                        planner_path_id=item.get("path_id", None)
                    )
                    suggestions.append(suggestion)
                
                return suggestions
                
        except Exception as e:
            logger.warning(f"Error in planner suggestion generation: {e}")
        
        return []
    
    def analyze_result(self, result: CanaryResult) -> CanaryResult:
        """
        Analyze a test result and enhance it with failure analysis and fix suggestions.
        
        Parameters
        ----------
        result : CanaryResult
            The test result to analyze.
            
        Returns
        -------
        CanaryResult
            The enhanced test result with analysis and suggestions.
        """
        # Skip analysis for successful tests
        if result.passed:
            return result
        
        # Analyze the failure
        if result.result_type == CanaryResultType.FAIL:
            # Detailed failure analysis
            failure_details = self.analyze_failure(
                result.output, result.error or "", result.exit_code or 1
            )
            result.failure_details = failure_details
            
            # Generate fix suggestions
            fix_suggestions = self.generate_fix_suggestions(
                failure_details, result.output, result.error or ""
            )
            result.fix_suggestions = fix_suggestions
            
            # Perform root cause analysis
            test_id = result.test_id or str(uuid.uuid4())
            self._update_historical_failures(test_id, failure_details)
            result.root_cause_analysis = self._perform_root_cause_analysis(test_id)
        
        # Add minimal analysis for timeouts
        elif result.result_type == CanaryResultType.TIMEOUT:
            result.fix_suggestions = [
                TestFixSuggestion(
                    suggestion_type="performance",
                    description="The test exceeded its time limit. Consider optimizing the test or increasing the timeout.",
                    confidence=0.8
                )
            ]
        
        return result
    
    def _update_historical_failures(self, test_id: str, failure_details: Dict[str, Any]) -> None:
        """
        Update historical failure data for pattern detection.
        
        Parameters
        ----------
        test_id : str
            Identifier for the test.
        failure_details : Dict[str, Any]
            Detailed failure analysis.
        """
        if test_id not in self.historical_failures:
            self.historical_failures[test_id] = []
        
        # Add new failure data
        self.historical_failures[test_id].append({
            "timestamp": time.time(),
            **failure_details
        })
        
        # Limit the history size
        if len(self.historical_failures[test_id]) > 20:
            self.historical_failures[test_id] = self.historical_failures[test_id][-20:]
    
    def _perform_root_cause_analysis(self, test_id: str) -> Optional[str]:
        """
        Perform root cause analysis based on historical failures.
        
        Parameters
        ----------
        test_id : str
            Identifier for the test.
            
        Returns
        -------
        Optional[str]
            Root cause analysis text, or None if insufficient data.
        """
        if test_id not in self.historical_failures:
            return None
        
        history = self.historical_failures[test_id]
        if len(history) < self.root_cause_threshold:
            return None
        
        # Count occurrences of each error type
        error_counts = {}
        for failure in history:
            error_type = failure.get("error_type", "unknown")
            if error_type not in error_counts:
                error_counts[error_type] = 0
            error_counts[error_type] += 1
        
        # Find the most common error type
        most_common_error = max(error_counts.items(), key=lambda x: x[1])
        error_type, count = most_common_error
        
        # Generate root cause analysis
        if count >= self.root_cause_threshold:
            percentage = (count / len(history)) * 100
            analysis = f"Root cause analysis: This test has failed {count} times ({percentage:.1f}%) "
            analysis += f"with error type '{error_type}'. "
            
            # Add specific analysis based on error type
            if error_type == "assertion_error":
                analysis += "The test consistently fails due to assertion errors, suggesting a logic issue in the code or test."
            elif error_type == "timeout_error":
                analysis += "The test consistently times out, suggesting performance issues or infinite loops."
            elif error_type == "import_error":
                analysis += "The test consistently fails due to missing imports, suggesting dependency issues."
            else:
                analysis += f"The consistent '{error_type}' errors suggest a systematic issue rather than a random failure."
            
            return analysis
        
        return None
    
    def save_historical_failures(self, path: str) -> None:
        """
        Save historical failure data to a file.
        
        Parameters
        ----------
        path : str
            Path to save the data to.
        """
        with open(path, 'w') as f:
            json.dump(self.historical_failures, f, indent=2)
    
    def load_historical_failures(self, path: str) -> None:
        """
        Load historical failure data from a file.
        
        Parameters
        ----------
        path : str
            Path to load the data from.
        """
        try:
            with open(path, 'r') as f:
                self.historical_failures = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Start with empty history if file doesn't exist or is invalid
            self.historical_failures = {}


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Resource Monitoring
# ──────────────────────────────────────────────────────────────────────────────
class ResourceMonitor:
    """
    Monitor resource usage of a subprocess.
    
    Features:
    - Memory usage tracking
    - CPU usage tracking
    - I/O operations tracking
    - Thread count tracking
    - Resource limit enforcement
    """
    
    def __init__(
        self,
        pid: int,
        memory_limit_mb: Optional[float] = None,
        cpu_limit_percent: Optional[float] = None,
        thread_limit: Optional[int] = None,
        sample_interval: float = 0.5
    ):
        """
        Initialize a ResourceMonitor.
        
        Parameters
        ----------
        pid : int
            Process ID to monitor.
        memory_limit_mb : Optional[float]
            Memory limit in MB, or None for no limit.
        cpu_limit_percent : Optional[float]
            CPU usage limit in percent, or None for no limit.
        thread_limit : Optional[int]
            Thread count limit, or None for no limit.
        sample_interval : float
            Interval in seconds between resource samples.
        """
        self.pid = pid
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_percent = cpu_limit_percent
        self.thread_limit = thread_limit
        self.sample_interval = sample_interval
        
        # Resource usage data
        self.resource_data = ResourceUsage()
        
        # Monitoring thread
        self._stop_monitor = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._resource_lock = threading.Lock()
        
        # Limit exceeded information
        self.limit_exceeded: Optional[Dict[str, Any]] = None
    
    def start(self) -> None:
        """Start monitoring resources."""
        # Initialize resource usage
        self.resource_data = ResourceUsage()
        self.limit_exceeded = None
        
        # Reset stop event
        self._stop_monitor.clear()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop(self) -> ResourceUsage:
        """
        Stop monitoring resources.
        
        Returns
        -------
        ResourceUsage
            Final resource usage data.
        """
        # Signal monitoring thread to stop
        self._stop_monitor.set()
        
        # Wait for thread to terminate
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        
        # Set end time if not already set
        with self._resource_lock:
            if self.resource_data.end_time is None:
                self.resource_data.end_time = time.time()
            
            # Calculate average CPU usage
            if self.resource_data.cpu_readings:
                self.resource_data.avg_cpu_percent = sum(self.resource_data.cpu_readings) / len(self.resource_data.cpu_readings)
            
            return self.resource_data
    
    def _monitor_resources(self) -> None:
        """Monitor resource usage of the process."""
        try:
            # Try to import psutil
            import psutil
            
            try:
                # Get process object
                process = psutil.Process(self.pid)
                
                # Initial I/O counters
                try:
                    initial_io = process.io_counters()
                    initial_read = initial_io.read_bytes
                    initial_write = initial_io.write_bytes
                except (psutil.AccessDenied, AttributeError):
                    initial_read = 0
                    initial_write = 0
                
                # Monitoring loop
                while not self._stop_monitor.is_set():
                    try:
                        # Check if process still exists
                        if not process.is_running():
                            break
                        
                        # Get memory usage
                        memory_info = process.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)
                        
                        # Get CPU usage
                        cpu_percent = process.cpu_percent(interval=0.1)
                        
                        # Get thread count
                        thread_count = len(process.threads())
                        
                        # Get I/O counters
                        try:
                            io = process.io_counters()
                            read_bytes = io.read_bytes - initial_read
                            write_bytes = io.write_bytes - initial_write
                        except (psutil.AccessDenied, AttributeError):
                            read_bytes = 0
                            write_bytes = 0
                        
                        # Update resource data
                        with self._resource_lock:
                            self.resource_data.peak_memory_mb = max(
                                self.resource_data.peak_memory_mb, memory_mb
                            )
                            self.resource_data.cpu_readings.append(cpu_percent)
                            self.resource_data.max_cpu_percent = max(
                                self.resource_data.max_cpu_percent, cpu_percent
                            )
                            self.resource_data.thread_count = max(
                                self.resource_data.thread_count, thread_count
                            )
                            self.resource_data.io_read_bytes = read_bytes
                            self.resource_data.io_write_bytes = write_bytes
                        
                        # Check resource limits
                        if self.memory_limit_mb and memory_mb > self.memory_limit_mb:
                            self.limit_exceeded = {
                                "resource_type": "memory",
                                "limit": self.memory_limit_mb,
                                "actual": memory_mb
                            }
                            # Terminate the process
                            process.terminate()
                            break
                        
                        if self.cpu_limit_percent and cpu_percent > self.cpu_limit_percent:
                            self.limit_exceeded = {
                                "resource_type": "cpu",
                                "limit": self.cpu_limit_percent,
                                "actual": cpu_percent
                            }
                            # For CPU, we might want to give it a chance to calm down
                            # before terminating, so just log for now
                            logger.warning(f"CPU limit exceeded: {cpu_percent}% > {self.cpu_limit_percent}%")
                        
                        if self.thread_limit and thread_count > self.thread_limit:
                            self.limit_exceeded = {
                                "resource_type": "threads",
                                "limit": self.thread_limit,
                                "actual": thread_count
                            }
                            # Terminate the process
                            process.terminate()
                            break
                        
                        # Sleep until next sample
                        time.sleep(self.sample_interval)
                        
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Process has terminated or can't be accessed
                        break
                    except Exception as e:
                        logger.warning(f"Error monitoring resources: {e}")
                        time.sleep(self.sample_interval)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process doesn't exist or can't be accessed
                logger.warning(f"Process {self.pid} not found or access denied")
                
        except ImportError:
            # psutil not available
            logger.warning("psutil not available, resource monitoring disabled")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Enhanced Canary Test Runner
# ──────────────────────────────────────────────────────────────────────────────
class CanaryTestRunner:
    """
    Executes a single test command in a subprocess with enhanced timeout handling,
    detailed result tracking, and automatic retry for flaky tests.
    
    Features:
    - Advanced timeout handling with multiple strategies
    - Resource usage monitoring and limits
    - Detailed failure analysis and reporting
    - Automatic retry for unstable/flaky tests
    - Historical failure pattern detection
    - Planner integration for test selection and result reporting
    """

    def __init__(
        self, 
        # Timeout configuration
        timeout: float = 10.0,
        timeout_strategy: TimeoutStrategy = TimeoutStrategy.FIXED,
        
        # Resource limits
        memory_limit_mb: Optional[float] = None,
        cpu_limit_percent: Optional[float] = None,
        thread_limit: Optional[int] = None,
        
        # Process configuration
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True,
        kill_signal: int = signal.SIGTERM,
        
        # Retry configuration
        retry_on_timeout: bool = True,
        retry_on_failure: bool = False,
        max_retries: int = 3,
        
        # Advanced features
        track_resources: bool = True,
        analyze_failures: bool = True,
        detect_flaky_tests: bool = True,
        
        # Planner integration
        planner_agent = None
    ):
        """
        Initialize a CanaryTestRunner with advanced configuration.
        
        Parameters
        ----------
        timeout : float
            Base timeout value in seconds.
        timeout_strategy : TimeoutStrategy
            Strategy for timeout handling.
        memory_limit_mb : Optional[float]
            Memory limit in MB, or None for no limit.
        cpu_limit_percent : Optional[float]
            CPU usage limit in percent, or None for no limit.
        thread_limit : Optional[int]
            Thread count limit, or None for no limit.
        working_dir : Optional[str]
            Working directory for the subprocess. If None, uses current directory.
        env : Optional[Dict[str, str]]
            Environment variables for the subprocess. If None, inherits current environment.
        capture_output : bool
            Whether to capture stdout and stderr from the subprocess.
        kill_signal : int
            Signal to use for killing the process on timeout.
        retry_on_timeout : bool
            Whether to retry the test if it times out.
        retry_on_failure : bool
            Whether to retry the test if it fails.
        max_retries : int
            Maximum number of retries.
        track_resources : bool
            Whether to track resource usage of the subprocess.
        analyze_failures : bool
            Whether to perform detailed failure analysis.
        detect_flaky_tests : bool
            Whether to detect and report flaky tests.
        planner_agent : Any
            Optional planner agent for integration.
        """
        # Basic configuration
        self.working_dir = working_dir
        self.env = env
        self.capture_output = capture_output
        self.kill_signal = kill_signal
        
        # Resource tracking
        self.track_resources = track_resources
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_percent = cpu_limit_percent
        self.thread_limit = thread_limit
        
        # Retry configuration
        self.retry_on_timeout = retry_on_timeout
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries
        
        # Advanced features
        self.analyze_failures = analyze_failures
        self.detect_flaky_tests = detect_flaky_tests
        
        # Create timeout handler
        self.timeout_handler = TimeoutHandler(
            strategy=timeout_strategy,
            base_timeout=timeout
        )
        
        # Create failure analyzer if needed
        self.failure_analyzer = FailureAnalyzer(planner_agent=planner_agent) if analyze_failures else None
        
        # Store test history
        self.history: List[CanaryResult] = []
        
        # Planner integration
        self.planner_agent = planner_agent
    
    def run(self, test_command: List[str], test_id: Optional[str] = None, 
           bug_id: Optional[str] = None, execution_id: Optional[str] = None) -> CanaryResult:
        """
        Execute the test command with advanced monitoring and return detailed results.

        Parameters
        ----------
        test_command : List[str]
            The command and its arguments to execute (e.g., ["pytest", "test_file.py"]).
        test_id : Optional[str]
            Optional identifier for the test, useful for tracking and history.
        bug_id : Optional[str]
            Optional bug ID for planner integration.
        execution_id : Optional[str]
            Optional execution ID for planner integration.

        Returns
        -------
        CanaryResult
            An object containing the detailed outcome of the test run.
        """
        # Generate test ID if not provided
        if not test_id:
            test_id = str(uuid.uuid4())
        
        # Track retry attempts
        retry_count = 0
        retry_results = []
        
        while True:
            # Get timeout for this attempt
            timeout = self.timeout_handler.get_timeout(retry_count)
            
            # Execute the test
            result = self._execute_test(test_command, timeout, test_id)
            
            # Record execution time if available
            if result.resources and result.resources.duration > 0:
                self.timeout_handler.record_execution_time(result.resources.duration)
            
            # Store retry information
            result.retry_count = retry_count
            
            # Store planner integration info
            result.planner_bug_id = bug_id
            result.planner_execution_id = execution_id
            
            # Add to retry results
            retry_results.append(result)
            
            # Check if we should retry
            should_retry = (
                (result.timed_out and self.retry_on_timeout) or
                (not result.passed and not result.timed_out and self.retry_on_failure)
            ) and retry_count < self.max_retries
            
            if should_retry:
                retry_count += 1
                logger.info(f"Retrying test (attempt {retry_count}/{self.max_retries})...")
                continue
            
            # Check for flaky tests if we have multiple runs
            if len(retry_results) > 1 and self.detect_flaky_tests:
                # Check if test was unstable (sometimes pass, sometimes fail)
                pass_results = [r for r in retry_results if r.passed]
                fail_results = [r for r in retry_results if not r.passed]
                
                if pass_results and fail_results:
                    # Test is flaky - create an unstable result
                    result = CanaryResult.create_unstable(test_command, retry_results)
                    result.test_id = test_id
                    result.planner_bug_id = bug_id
                    result.planner_execution_id = execution_id
            
            # Analyze the final result if requested
            if self.analyze_failures and self.failure_analyzer and not result.passed:
                result = self.failure_analyzer.analyze_result(result)
            
            # Add to history
            self.history.append(result)
            
            # Return the final result
            return result
    
    def _execute_test(self, test_command: List[str], timeout: float, 
                    test_id: Optional[str] = None) -> CanaryResult:
        """
        Execute a single test command with monitoring.
        
        Parameters
        ----------
        test_command : List[str]
            The command and its arguments to execute.
        timeout : float
            Timeout value for this execution.
        test_id : Optional[str]
            Optional test identifier.
            
        Returns
        -------
        CanaryResult
            The test result.
        """
        try:
            # Start resource usage tracking
            resources = ResourceUsage()
            resource_monitor = None
            
            # Start process
            process = subprocess.Popen(
                test_command,
                stdout=subprocess.PIPE if self.capture_output else None,
                stderr=subprocess.PIPE if self.capture_output else None,
                cwd=self.working_dir,
                env=self.env,
                text=True,
                encoding='utf-8'
            )
            
            # Start resource monitoring if enabled
            if self.track_resources:
                try:
                    resource_monitor = ResourceMonitor(
                        pid=process.pid,
                        memory_limit_mb=self.memory_limit_mb,
                        cpu_limit_percent=self.cpu_limit_percent,
                        thread_limit=self.thread_limit
                    )
                    resource_monitor.start()
                except Exception as e:
                    logger.warning(f"Failed to start resource monitoring: {e}")
            
            # Setup more robust timeout handling
            timed_out = False
            stdout = ""
            stderr = ""
            
            try:
                # Use communicate with timeout instead of a separate timer
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Process took too long, terminate it
                timed_out = True
                
                # Try to terminate gracefully first
                try:
                    process.send_signal(self.kill_signal)
                    try:
                        process.wait(timeout=2.0)  # Give it 2 seconds to terminate
                    except subprocess.TimeoutExpired:
                        # Force kill if it didn't respond to the signal
                        process.kill()
                except Exception as e:
                    # Process might have terminated on its own
                    logger.warning(f"Error terminating process: {e}")
                
                # Try to get any output that was generated before timeout
                try:
                    stdout, stderr = process.communicate(timeout=1.0)
                except subprocess.TimeoutExpired:
                    # Couldn't get output, use empty strings
                    stdout = stderr = ""
            finally:
                # Stop resource monitoring
                if resource_monitor:
                    resources = resource_monitor.stop()
                else:
                    resources.end_time = time.time()
            
            # Get exit code (0 if the process completed successfully)
            exit_code = process.returncode if not timed_out else -1
            
            # Check if resource limits were exceeded
            if resource_monitor and resource_monitor.limit_exceeded:
                limit_info = resource_monitor.limit_exceeded
                return CanaryResult.create_resource_exceeded(
                    test_command,
                    limit_info["resource_type"],
                    limit_info["limit"],
                    limit_info["actual"]
                )
            
            # Create result based on outcome
            if timed_out:
                # Handle timeout case
                result = CanaryResult.create_timeout(
                    test_command, timeout, stdout, stderr
                )
                result.resources = resources
                result.test_id = test_id
                return result
            elif exit_code == 0:
                # Test passed
                result = CanaryResult.create_success(
                    stdout, test_command, exit_code
                )
                result.resources = resources
                result.test_id = test_id
                return result
            else:
                # Test failed
                failure_details = {}
                if self.analyze_failures and self.failure_analyzer:
                    failure_details = self.failure_analyzer.analyze_failure(
                        stdout, stderr, exit_code
                    )
                
                result = CanaryResult.create_failure(
                    stdout, stderr, test_command, exit_code, failure_details
                )
                result.resources = resources
                result.test_id = test_id
                return result

        except FileNotFoundError:
            return CanaryResult.create_not_found(test_command)
        except Exception as e:
            logger.error(f"Error executing test: {e}")
            return CanaryResult.create_error(str(e), test_command)
    
    def get_latest_result(self) -> Optional[CanaryResult]:
        """Get the most recent test result."""
        return self.history[-1] if self.history else None
    
    def get_history(self) -> List[CanaryResult]:
        """Get the full history of test results."""
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Clear the test history."""
        self.history.clear()
    
    def save_results(self, path: str) -> None:
        """
        Save test results to a JSON file.
        
        Parameters
        ----------
        path : str
            Path to save the results to.
        """
        results = [result.to_dict() for result in self.history]
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def notify_planner(self, result: CanaryResult) -> None:
        """
        Notify the planner agent about a test result.
        
        Parameters
        ----------
        result : CanaryResult
            The test result to report.
        """
        if not self.planner_agent or not hasattr(self.planner_agent, "report_test_result"):
            return
        
        try:
            # Call planner method to report result
            self.planner_agent.report_test_result(result.to_dict())
        except Exception as e:
            logger.warning(f"Error notifying planner about test result: {e}")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CanaryTestRunner":
        """
        Create a CanaryTestRunner from a configuration dictionary.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.
            
        Returns
        -------
        CanaryTestRunner
            A new CanaryTestRunner instance.
        """
        # Extract planner agent if available
        planner_agent = config.pop("planner_agent", None)
        
        # Extract timeout configuration
        timeout = config.pop("timeout", 10.0)
        timeout_strategy_str = config.pop("timeout_strategy", "fixed")
        timeout_strategy = TimeoutStrategy(timeout_strategy_str)
        
        # Create instance
        return cls(
            timeout=timeout,
            timeout_strategy=timeout_strategy,
            planner_agent=planner_agent,
            **config
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Command Line Interface
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """Command line interface for the canary test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a canary test.")
    
    # Test command
    parser.add_argument(
        "command",
        nargs="+",
        help="Test command to execute."
    )
    
    # Basic options
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds."
    )
    parser.add_argument(
        "--working-dir",
        help="Working directory for the test command."
    )
    parser.add_argument(
        "--output",
        help="Path to save the test results."
    )
    
    # Advanced options
    parser.add_argument(
        "--timeout-strategy",
        choices=[strategy.value for strategy in TimeoutStrategy],
        default=TimeoutStrategy.FIXED.value,
        help="Timeout strategy to use."
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        help="Memory limit in MB."
    )
    parser.add_argument(
        "--cpu-limit",
        type=float,
        help="CPU usage limit in percent."
    )
    parser.add_argument(
        "--retry-on-timeout",
        action="store_true",
        help="Retry on timeout."
    )
    parser.add_argument(
        "--retry-on-failure",
        action="store_true",
        help="Retry on failure."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries."
    )
    parser.add_argument(
        "--no-resource-tracking",
        action="store_true",
        help="Disable resource tracking."
    )
    parser.add_argument(
        "--no-failure-analysis",
        action="store_true",
        help="Disable failure analysis."
    )
    parser.add_argument(
        "--no-flaky-detection",
        action="store_true",
        help="Disable flaky test detection."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create runner
    runner = CanaryTestRunner(
        timeout=args.timeout,
        timeout_strategy=TimeoutStrategy(args.timeout_strategy),
        memory_limit_mb=args.memory_limit,
        cpu_limit_percent=args.cpu_limit,
        working_dir=args.working_dir,
        retry_on_timeout=args.retry_on_timeout,
        retry_on_failure=args.retry_on_failure,
        max_retries=args.max_retries,
        track_resources=not args.no_resource_tracking,
        analyze_failures=not args.no_failure_analysis,
        detect_flaky_tests=not args.no_flaky_detection
    )
    
    # Run test
    result = runner.run(args.command)
    
    # Print result
    print(f"Test result: {result.result_type.value.upper()}")
    print(f"Exit code: {result.exit_code}")
    print(f"Duration: {result.resources.duration:.2f} seconds")
    
    if result.passed:
        print("Test passed successfully!")
    else:
        print("\nTest failed!")
        print(f"Error type: {result.failure_details.get('error_type', 'unknown')}")
        if result.error:
            print(f"Error message: {result.error}")
        
        # Print fix suggestions
        if result.fix_suggestions:
            print("\nFix suggestions:")
            for i, suggestion in enumerate(result.fix_suggestions):
                print(f"{i+1}. {suggestion.description}")
                if suggestion.code_snippet:
                    print(f"   Code: {suggestion.code_snippet}")
    
    # Save results if requested
    if args.output:
        runner.save_results(args.output)
        print(f"Results saved to {args.output}")
    
    # Return exit code
    return 0 if result.passed else 1


if __name__ == "__main__":
    exit(main())
