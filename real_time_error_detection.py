#!/usr/bin/env python3
"""
Real-time Error Detection Module

This module provides real-time error detection capabilities for the auditor agent,
enabling immediate detection, analysis, and reporting of errors as they occur.
"""

import os
import sys
import json
import logging
import time
import threading
import queue
import re
import traceback
import signal
import weakref
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set, Pattern

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("error_detection.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ErrorDetection")

class ErrorPattern:
    """
    Represents a detectable error pattern.
    """
    
    def __init__(self, pattern_id: str, name: str, regex: str, 
                description: str = None, severity: str = "medium",
                tags: List[str] = None, callback: Callable = None):
        """
        Initialize error pattern.
        
        Args:
            pattern_id: Unique pattern ID
            name: Pattern name
            regex: Regular expression pattern
            description: Pattern description
            severity: Error severity (low, medium, high, critical)
            tags: Pattern tags
            callback: Callback function when pattern is matched
        """
        self.pattern_id = pattern_id
        self.name = name
        self.regex = regex
        self.description = description or ""
        self.severity = severity
        self.tags = tags or []
        self.callback = callback
        self.compiled_regex = re.compile(regex, re.MULTILINE)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "regex": self.regex,
            "description": self.description,
            "severity": self.severity,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorPattern':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Error pattern
        """
        return cls(
            pattern_id=data.get("pattern_id"),
            name=data.get("name"),
            regex=data.get("regex"),
            description=data.get("description"),
            severity=data.get("severity", "medium"),
            tags=data.get("tags", [])
        )
    
    def match(self, text: str) -> List[re.Match]:
        """
        Match pattern against text.
        
        Args:
            text: Text to match against
            
        Returns:
            List of matches
        """
        return list(self.compiled_regex.finditer(text))

class ErrorDetectionResult:
    """
    Represents the result of an error detection operation.
    """
    
    def __init__(self, source: str, error_pattern: ErrorPattern, 
                matches: List[re.Match], timestamp: float = None,
                context: Dict[str, Any] = None):
        """
        Initialize error detection result.
        
        Args:
            source: Source of the text (e.g., file name, command output)
            error_pattern: Error pattern that was matched
            matches: List of regex matches
            timestamp: Detection timestamp
            context: Additional context information
        """
        self.source = source
        self.error_pattern = error_pattern
        self.matches = matches
        self.timestamp = timestamp or time.time()
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        # Convert matches to serializable format
        match_data = []
        for match in self.matches:
            match_data.append({
                "start": match.start(),
                "end": match.end(),
                "text": match.group(0),
                "groups": match.groups() if match.groups() else []
            })
        
        return {
            "source": self.source,
            "error_pattern": self.error_pattern.to_dict(),
            "matches": match_data,
            "timestamp": self.timestamp,
            "context": self.context
        }

class ErrorDatabase:
    """
    Database of known error patterns.
    """
    
    def __init__(self):
        """Initialize error database."""
        self.patterns = {}
        self.lock = threading.Lock()
        
        # Register built-in patterns
        self._register_builtin_patterns()
        
        logger.info("Error database initialized")
    
    def _register_builtin_patterns(self) -> None:
        """Register built-in error patterns."""
        # Python exceptions
        self.add_pattern(ErrorPattern(
            pattern_id="python_exception",
            name="Python Exception",
            regex=r"Traceback \(most recent call last\):\s+(?:.*\n)+?.*?(\w+Error|Exception):(.*?)(?:\n|$)",
            description="Python exception traceback",
            severity="high",
            tags=["python", "exception"]
        ))
        
        # Syntax error
        self.add_pattern(ErrorPattern(
            pattern_id="python_syntax_error",
            name="Python Syntax Error",
            regex=r"SyntaxError: (.*?)(?:\n|$)",
            description="Python syntax error",
            severity="high",
            tags=["python", "syntax"]
        ))
        
        # Import error
        self.add_pattern(ErrorPattern(
            pattern_id="python_import_error",
            name="Python Import Error",
            regex=r"ImportError: (.*?)(?:\n|$)",
            description="Python import error",
            severity="high",
            tags=["python", "import"]
        ))
        
        # Assertion error
        self.add_pattern(ErrorPattern(
            pattern_id="python_assertion_error",
            name="Python Assertion Error",
            regex=r"AssertionError: (.*?)(?:\n|$)",
            description="Python assertion error",
            severity="medium",
            tags=["python", "assertion"]
        ))
        
        # File not found
        self.add_pattern(ErrorPattern(
            pattern_id="file_not_found",
            name="File Not Found",
            regex=r"(?:FileNotFoundError|No such file or directory): (.*?)(?:\n|$)",
            description="File not found error",
            severity="medium",
            tags=["file", "io"]
        ))
        
        # Permission denied
        self.add_pattern(ErrorPattern(
            pattern_id="permission_denied",
            name="Permission Denied",
            regex=r"(?:PermissionError|Permission denied): (.*?)(?:\n|$)",
            description="Permission denied error",
            severity="medium",
            tags=["permission", "security"]
        ))
        
        # Timeout error
        self.add_pattern(ErrorPattern(
            pattern_id="timeout_error",
            name="Timeout Error",
            regex=r"(?:TimeoutError|Timed out): (.*?)(?:\n|$)",
            description="Timeout error",
            severity="medium",
            tags=["timeout", "performance"]
        ))
        
        # Memory error
        self.add_pattern(ErrorPattern(
            pattern_id="memory_error",
            name="Memory Error",
            regex=r"(?:MemoryError|Out of memory): (.*?)(?:\n|$)",
            description="Memory error",
            severity="high",
            tags=["memory", "resource"]
        ))
        
        # Network error
        self.add_pattern(ErrorPattern(
            pattern_id="network_error",
            name="Network Error",
            regex=r"(?:ConnectionError|ConnectionRefusedError|ConnectionResetError|ConnectionAbortedError|socket error): (.*?)(?:\n|$)",
            description="Network error",
            severity="medium",
            tags=["network", "connectivity"]
        ))
        
        # General warning
        self.add_pattern(ErrorPattern(
            pattern_id="warning",
            name="Warning",
            regex=r"(?:WARNING|Warning): (.*?)(?:\n|$)",
            description="General warning",
            severity="low",
            tags=["warning"]
        ))
        
        # Security warning
        self.add_pattern(ErrorPattern(
            pattern_id="security_warning",
            name="Security Warning",
            regex=r"(?:SECURITY|Security) (?:WARNING|Warning): (.*?)(?:\n|$)",
            description="Security warning",
            severity="high",
            tags=["security", "warning"]
        ))
        
        # JSON parsing error
        self.add_pattern(ErrorPattern(
            pattern_id="json_error",
            name="JSON Error",
            regex=r"(?:JSONDecodeError|Invalid JSON): (.*?)(?:\n|$)",
            description="JSON parsing error",
            severity="medium",
            tags=["json", "parsing"]
        ))
        
        # SQL error
        self.add_pattern(ErrorPattern(
            pattern_id="sql_error",
            name="SQL Error",
            regex=r"(?:SQLError|SQL syntax error): (.*?)(?:\n|$)",
            description="SQL error",
            severity="medium",
            tags=["sql", "database"]
        ))
    
    def add_pattern(self, pattern: ErrorPattern) -> bool:
        """
        Add an error pattern to the database.
        
        Args:
            pattern: Error pattern
            
        Returns:
            Whether the pattern was added
        """
        with self.lock:
            if pattern.pattern_id in self.patterns:
                logger.warning(f"Pattern with ID {pattern.pattern_id} already exists")
                return False
            
            self.patterns[pattern.pattern_id] = pattern
            logger.debug(f"Added pattern: {pattern.name} ({pattern.pattern_id})")
            return True
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """
        Remove an error pattern from the database.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Whether the pattern was removed
        """
        with self.lock:
            if pattern_id not in self.patterns:
                logger.warning(f"Pattern with ID {pattern_id} not found")
                return False
            
            del self.patterns[pattern_id]
            logger.debug(f"Removed pattern: {pattern_id}")
            return True
    
    def get_pattern(self, pattern_id: str) -> Optional[ErrorPattern]:
        """
        Get an error pattern by ID.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Error pattern, or None if not found
        """
        with self.lock:
            return self.patterns.get(pattern_id)
    
    def get_patterns(self, tags: List[str] = None, severity: str = None) -> List[ErrorPattern]:
        """
        Get error patterns matching criteria.
        
        Args:
            tags: Filter by tags
            severity: Filter by severity
            
        Returns:
            List of matching error patterns
        """
        with self.lock:
            patterns = list(self.patterns.values())
            
            # Filter by tags
            if tags:
                patterns = [p for p in patterns if any(tag in p.tags for tag in tags)]
            
            # Filter by severity
            if severity:
                patterns = [p for p in patterns if p.severity == severity]
            
            return patterns
    
    def load_patterns_from_file(self, file_path: str) -> int:
        """
        Load error patterns from a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Number of patterns loaded
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            count = 0
            
            for pattern_data in data:
                pattern = ErrorPattern.from_dict(pattern_data)
                if self.add_pattern(pattern):
                    count += 1
            
            logger.info(f"Loaded {count} patterns from {file_path}")
            return count
        except Exception as e:
            logger.error(f"Error loading patterns from {file_path}: {e}")
            return 0
    
    def save_patterns_to_file(self, file_path: str) -> int:
        """
        Save error patterns to a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Number of patterns saved
        """
        try:
            with self.lock:
                patterns_data = [p.to_dict() for p in self.patterns.values()]
            
            with open(file_path, "w") as f:
                json.dump(patterns_data, f, indent=2)
            
            logger.info(f"Saved {len(patterns_data)} patterns to {file_path}")
            return len(patterns_data)
        except Exception as e:
            logger.error(f"Error saving patterns to {file_path}: {e}")
            return 0

class ErrorDetector:
    """
    Detects errors in text data.
    """
    
    def __init__(self, error_database: ErrorDatabase = None):
        """
        Initialize error detector.
        
        Args:
            error_database: Error database
        """
        self.error_database = error_database or ErrorDatabase()
        self.listeners = []
        
        logger.info("Error detector initialized")
    
    def add_listener(self, listener: Callable[[ErrorDetectionResult], None]) -> None:
        """
        Add a listener for error detection results.
        
        Args:
            listener: Listener function
        """
        if listener not in self.listeners:
            self.listeners.append(listener)
            logger.debug(f"Added error listener: {listener.__name__}")
    
    def remove_listener(self, listener: Callable[[ErrorDetectionResult], None]) -> bool:
        """
        Remove a listener.
        
        Args:
            listener: Listener function
            
        Returns:
            Whether the listener was removed
        """
        if listener in self.listeners:
            self.listeners.remove(listener)
            logger.debug(f"Removed error listener: {listener.__name__}")
            return True
        
        return False
    
    def detect_errors(self, text: str, source: str = "unknown", 
                     context: Dict[str, Any] = None) -> List[ErrorDetectionResult]:
        """
        Detect errors in text.
        
        Args:
            text: Text to analyze
            source: Source of the text
            context: Additional context information
            
        Returns:
            List of error detection results
        """
        results = []
        
        # Get all patterns
        patterns = self.error_database.get_patterns()
        
        # Match each pattern
        for pattern in patterns:
            matches = pattern.match(text)
            
            if matches:
                result = ErrorDetectionResult(source, pattern, matches, context=context)
                results.append(result)
                
                # Notify listeners
                for listener in self.listeners:
                    try:
                        listener(result)
                    except Exception as e:
                        logger.error(f"Error in listener: {e}")
                
                # Call pattern callback if specified
                if pattern.callback:
                    try:
                        pattern.callback(result)
                    except Exception as e:
                        logger.error(f"Error in pattern callback: {e}")
        
        return results

class ErrorTracker:
    """
    Tracks errors over time.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize error tracker.
        
        Args:
            max_history: Maximum number of errors to track
        """
        self.errors = {}
        self.error_history = []
        self.max_history = max_history
        self.lock = threading.Lock()
        
        logger.info("Error tracker initialized")
    
    def add_error(self, result: ErrorDetectionResult) -> None:
        """
        Add an error detection result.
        
        Args:
            result: Error detection result
        """
        with self.lock:
            # Add to history
            self.error_history.append(result)
            
            # Trim history if needed
            if len(self.error_history) > self.max_history:
                self.error_history = self.error_history[-self.max_history:]
            
            # Track by pattern ID
            pattern_id = result.error_pattern.pattern_id
            
            if pattern_id not in self.errors:
                self.errors[pattern_id] = {
                    "pattern": result.error_pattern,
                    "count": 0,
                    "first_seen": result.timestamp,
                    "last_seen": result.timestamp,
                    "sources": set()
                }
            
            self.errors[pattern_id]["count"] += 1
            self.errors[pattern_id]["last_seen"] = result.timestamp
            self.errors[pattern_id]["sources"].add(result.source)
    
    def get_error_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get error statistics.
        
        Returns:
            Dictionary of error statistics
        """
        with self.lock:
            # Convert to serializable format
            stats = {}
            
            for pattern_id, error_data in self.errors.items():
                stats[pattern_id] = {
                    "pattern": error_data["pattern"].to_dict(),
                    "count": error_data["count"],
                    "first_seen": error_data["first_seen"],
                    "last_seen": error_data["last_seen"],
                    "sources": list(error_data["sources"])
                }
            
            return stats
    
    def get_error_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get error history.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of error detection results
        """
        with self.lock:
            # Convert to serializable format
            history = [result.to_dict() for result in self.error_history]
            
            # Apply limit
            if limit is not None:
                history = history[-limit:]
            
            return history
    
    def get_error_count(self, pattern_id: str = None, 
                       since: float = None) -> int:
        """
        Get error count.
        
        Args:
            pattern_id: Filter by pattern ID
            since: Filter by timestamp
            
        Returns:
            Error count
        """
        with self.lock:
            if pattern_id:
                # Count for specific pattern
                if pattern_id not in self.errors:
                    return 0
                
                if since is None:
                    return self.errors[pattern_id]["count"]
                
                # Count errors since timestamp
                count = 0
                for result in self.error_history:
                    if result.error_pattern.pattern_id == pattern_id and result.timestamp >= since:
                        count += 1
                
                return count
            else:
                # Count all errors
                if since is None:
                    return sum(error_data["count"] for error_data in self.errors.values())
                
                # Count errors since timestamp
                count = 0
                for result in self.error_history:
                    if result.timestamp >= since:
                        count += 1
                
                return count
    
    def clear_errors(self) -> int:
        """
        Clear all tracked errors.
        
        Returns:
            Number of errors cleared
        """
        with self.lock:
            count = len(self.error_history)
            self.errors = {}
            self.error_history = []
            return count

class StreamErrorDetector:
    """
    Detects errors in output streams (stdout, stderr).
    """
    
    def __init__(self, error_detector: ErrorDetector = None):
        """
        Initialize stream error detector.
        
        Args:
            error_detector: Error detector
        """
        self.error_detector = error_detector or ErrorDetector()
        self.stdout_buffer = ""
        self.stderr_buffer = ""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.patched = False
        
        logger.info("Stream error detector initialized")
    
    def start(self) -> None:
        """Start stream error detection."""
        if self.patched:
            logger.warning("Stream error detection already started")
            return
        
        # Patch stdout and stderr
        sys.stdout = self._create_stream_wrapper(sys.stdout, self._stdout_write)
        sys.stderr = self._create_stream_wrapper(sys.stderr, self._stderr_write)
        
        self.patched = True
        logger.info("Stream error detection started")
    
    def stop(self) -> None:
        """Stop stream error detection."""
        if not self.patched:
            logger.warning("Stream error detection not started")
            return
        
        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        self.patched = False
        logger.info("Stream error detection stopped")
    
    def _create_stream_wrapper(self, stream, write_func):
        """
        Create a wrapper for a stream.
        
        Args:
            stream: Stream to wrap
            write_func: Write function
            
        Returns:
            Stream wrapper
        """
        class StreamWrapper:
            def __init__(self, stream, write_func):
                self.stream = stream
                self.write_func = write_func
            
            def write(self, text):
                self.stream.write(text)
                self.write_func(text)
                return len(text)
            
            def flush(self):
                self.stream.flush()
            
            def isatty(self):
                return hasattr(self.stream, 'isatty') and self.stream.isatty()
            
            def fileno(self):
                return self.stream.fileno()
            
            def __getattr__(self, attr):
                return getattr(self.stream, attr)
        
        return StreamWrapper(stream, write_func)
    
    def _stdout_write(self, text: str) -> None:
        """
        Process stdout text.
        
        Args:
            text: Text written to stdout
        """
        self.stdout_buffer += text
        
        # Process complete lines
        if '\n' in self.stdout_buffer:
            lines = self.stdout_buffer.split('\n')
            self.stdout_buffer = lines.pop()
            
            # Detect errors in complete lines
            for line in lines:
                self.error_detector.detect_errors(line, source="stdout")
    
    def _stderr_write(self, text: str) -> None:
        """
        Process stderr text.
        
        Args:
            text: Text written to stderr
        """
        self.stderr_buffer += text
        
        # Process complete lines
        if '\n' in self.stderr_buffer:
            lines = self.stderr_buffer.split('\n')
            self.stderr_buffer = lines.pop()
            
            # Detect errors in complete lines
            for line in lines:
                self.error_detector.detect_errors(line, source="stderr")
    
    def __del__(self):
        """Clean up when object is deleted."""
        self.stop()

class ExceptionErrorDetector:
    """
    Detects uncaught exceptions.
    """
    
    def __init__(self, error_detector: ErrorDetector = None):
        """
        Initialize exception error detector.
        
        Args:
            error_detector: Error detector
        """
        self.error_detector = error_detector or ErrorDetector()
        self.original_excepthook = sys.excepthook
        self.patched = False
        
        logger.info("Exception error detector initialized")
    
    def start(self) -> None:
        """Start exception error detection."""
        if self.patched:
            logger.warning("Exception error detection already started")
            return
        
        # Patch sys.excepthook
        sys.excepthook = self._excepthook
        
        self.patched = True
        logger.info("Exception error detection started")
    
    def stop(self) -> None:
        """Stop exception error detection."""
        if not self.patched:
            logger.warning("Exception error detection not started")
            return
        
        # Restore original excepthook
        sys.excepthook = self.original_excepthook
        
        self.patched = False
        logger.info("Exception error detection stopped")
    
    def _excepthook(self, exc_type, exc_value, exc_traceback):
        """
        Custom exception hook.
        
        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        # Get traceback as text
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # Detect errors
        self.error_detector.detect_errors(tb_text, source="exception", context={
            "exc_type": exc_type.__name__,
            "exc_value": str(exc_value)
        })
        
        # Call original excepthook
        self.original_excepthook(exc_type, exc_value, exc_traceback)
    
    def __del__(self):
        """Clean up when object is deleted."""
        self.stop()

class LoggingErrorDetector:
    """
    Detects errors in log messages.
    """
    
    def __init__(self, error_detector: ErrorDetector = None):
        """
        Initialize logging error detector.
        
        Args:
            error_detector: Error detector
        """
        self.error_detector = error_detector or ErrorDetector()
        self.original_emit = None
        self.patched = False
        
        logger.info("Logging error detector initialized")
    
    def start(self) -> None:
        """Start logging error detection."""
        if self.patched:
            logger.warning("Logging error detection already started")
            return
        
        # Patch logging.Handler.emit
        self.original_emit = logging.Handler.emit
        logging.Handler.emit = self._create_emit_wrapper(self.original_emit)
        
        self.patched = True
        logger.info("Logging error detection started")
    
    def stop(self) -> None:
        """Stop logging error detection."""
        if not self.patched:
            logger.warning("Logging error detection not started")
            return
        
        # Restore original emit
        logging.Handler.emit = self.original_emit
        
        self.patched = False
        logger.info("Logging error detection stopped")
    
    def _create_emit_wrapper(self, original_emit):
        """
        Create a wrapper for the emit method.
        
        Args:
            original_emit: Original emit method
            
        Returns:
            Emit wrapper
        """
        error_detector = self.error_detector
        
        def emit_wrapper(self, record):
            # Call original emit
            original_emit(self, record)
            
            # Skip records from error detector to avoid infinite loop
            if record.name == "ErrorDetection":
                return
            
            # Detect errors in log message
            if record.levelno >= logging.ERROR:
                # Detect errors in formatted message
                try:
                    message = self.format(record)
                    error_detector.detect_errors(message, source=f"log.{record.name}", context={
                        "level": record.levelname,
                        "logger": record.name,
                        "line": record.lineno,
                        "file": record.pathname
                    })
                except Exception:
                    # If formatting fails, use the raw message
                    error_detector.detect_errors(record.getMessage(), source=f"log.{record.name}", context={
                        "level": record.levelname,
                        "logger": record.name,
                        "line": record.lineno,
                        "file": record.pathname
                    })
        
        return emit_wrapper
    
    def __del__(self):
        """Clean up when object is deleted."""
        self.stop()

class SignalErrorDetector:
    """
    Detects errors from signals.
    """
    
    def __init__(self, error_detector: ErrorDetector = None):
        """
        Initialize signal error detector.
        
        Args:
            error_detector: Error detector
        """
        self.error_detector = error_detector or ErrorDetector()
        self.original_handlers = {}
        self.patched_signals = set()
        
        logger.info("Signal error detector initialized")
    
    def start(self, signals: List[int] = None) -> None:
        """
        Start signal error detection.
        
        Args:
            signals: Signals to detect
        """
        if not signals:
            # Default signals to detect
            signals = [
                signal.SIGINT,  # Keyboard interrupt
                signal.SIGTERM,  # Termination signal
                signal.SIGABRT,  # Abort signal
                signal.SIGSEGV,  # Segmentation fault
                signal.SIGFPE,   # Floating point exception
                signal.SIGILL    # Illegal instruction
            ]
        
        # Patch signal handlers
        for sig in signals:
            if sig in self.patched_signals:
                continue
            
            try:
                self.original_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._create_signal_handler(sig))
                self.patched_signals.add(sig)
            except (ValueError, OSError, RuntimeError) as e:
                logger.warning(f"Could not set handler for signal {sig}: {e}")
        
        logger.info(f"Signal error detection started for {len(self.patched_signals)} signals")
    
    def stop(self) -> None:
        """Stop signal error detection."""
        if not self.patched_signals:
            logger.warning("Signal error detection not started")
            return
        
        # Restore original handlers
        for sig in list(self.patched_signals):
            try:
                signal.signal(sig, self.original_handlers[sig])
                self.patched_signals.remove(sig)
            except (ValueError, OSError, RuntimeError) as e:
                logger.warning(f"Could not restore handler for signal {sig}: {e}")
        
        self.original_handlers = {}
        logger.info("Signal error detection stopped")
    
    def _create_signal_handler(self, sig: int):
        """
        Create a signal handler.
        
        Args:
            sig: Signal number
            
        Returns:
            Signal handler
        """
        error_detector = self.error_detector
        original_handler = self.original_handlers[sig]
        
        def signal_handler(signum, frame):
            # Detect error
            error_detector.detect_errors(
                f"Received signal {signum} ({signal.Signals(signum).name})",
                source="signal",
                context={
                    "signal": signum,
                    "signal_name": signal.Signals(signum).name,
