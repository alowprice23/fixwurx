#!/usr/bin/env python3
"""
advanced_error_classification.py
───────────────────────────────
Advanced error classification system for the FixWurx platform.

This module provides sophisticated error classification, categorization,
pattern recognition, and analysis to help with troubleshooting and
automating error resolution.
"""

import os
import sys
import re
import json
import logging
import traceback
import time
import hashlib
import difflib
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
import inspect
from collections import defaultdict, Counter

# Internal imports
from shell_environment import register_event_handler, emit_event, EventType
from sensor_network import SensorType, DataType, Sensor, SensorMetadata, CustomSensor, add_sensor

# Configure logging
logger = logging.getLogger("ErrorClassification")

# Constants
MAX_ERROR_HISTORY = 1000  # Maximum number of errors to keep in memory
MAX_STACK_DEPTH = 50  # Maximum stack depth to analyze
DEFAULT_SIMILARITY_THRESHOLD = 0.85  # Default similarity threshold for grouping errors
ERROR_DB_PATH = "~/.fixwurx/error_db.json"

class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = auto()      # Debug-level issue, not affecting functionality
    INFO = auto()       # Informational issue, minimal impact
    WARNING = auto()    # Warning, potential issue but not breaking
    ERROR = auto()      # Error, functionality affected
    CRITICAL = auto()   # Critical error, system stability affected
    FATAL = auto()      # Fatal error, system cannot continue

class ErrorCategory(Enum):
    """Error categories."""
    SYNTAX = auto()         # Syntax errors
    RUNTIME = auto()        # Runtime errors
    LOGICAL = auto()        # Logical errors
    TYPE = auto()           # Type errors
    VALUE = auto()          # Value errors
    IO = auto()             # I/O errors
    NETWORK = auto()        # Network errors
    RESOURCE = auto()       # Resource errors
    PERMISSION = auto()     # Permission errors
    CONFIGURATION = auto()  # Configuration errors
    DEPENDENCY = auto()     # Dependency errors
    INTEGRATION = auto()    # Integration errors
    TIMEOUT = auto()        # Timeout errors
    MEMORY = auto()         # Memory errors
    CONCURRENCY = auto()    # Concurrency errors
    STATE = auto()          # State errors
    UNKNOWN = auto()        # Unknown errors

class ErrorImpact(Enum):
    """Error impact levels."""
    NONE = auto()       # No impact
    LOW = auto()        # Low impact, minimal disruption
    MEDIUM = auto()     # Medium impact, some functionality affected
    HIGH = auto()       # High impact, major functionality affected
    CRITICAL = auto()   # Critical impact, system unstable

class ErrorTrend(Enum):
    """Error trend types."""
    INCREASING = auto()     # Increasing frequency
    DECREASING = auto()     # Decreasing frequency
    STABLE = auto()         # Stable frequency
    SPORADIC = auto()       # Sporadic occurrence
    NEW = auto()            # New error
    RESOLVED = auto()       # Resolved error
    RECURRING = auto()      # Recurring error after resolution

@dataclass
class ErrorContext:
    """Context information for an error."""
    user: Optional[str] = None
    session_id: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    working_directory: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    process_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    system_state: Dict[str, Any] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user": self.user,
            "session_id": self.session_id,
            "command": self.command,
            "args": self.args,
            "working_directory": self.working_directory,
            "environment_vars": self.environment_vars,
            "process_id": self.process_id,
            "timestamp": self.timestamp,
            "system_state": self.system_state,
            "custom_data": self.custom_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorContext':
        """Create from dictionary."""
        return cls(
            user=data.get("user"),
            session_id=data.get("session_id"),
            command=data.get("command"),
            args=data.get("args"),
            working_directory=data.get("working_directory"),
            environment_vars=data.get("environment_vars", {}),
            process_id=data.get("process_id"),
            timestamp=data.get("timestamp", time.time()),
            system_state=data.get("system_state", {}),
            custom_data=data.get("custom_data", {})
        )

@dataclass
class StackFrame:
    """Stack frame information."""
    filename: str
    lineno: int
    function: str
    code_context: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "lineno": self.lineno,
            "function": self.function,
            "code_context": self.code_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StackFrame':
        """Create from dictionary."""
        return cls(
            filename=data["filename"],
            lineno=data["lineno"],
            function=data["function"],
            code_context=data.get("code_context")
        )

@dataclass
class ErrorDetails:
    """Detailed error information."""
    error_type: str
    error_message: str
    traceback: Optional[List[StackFrame]] = None
    context: Optional[ErrorContext] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": [frame.to_dict() for frame in self.traceback] if self.traceback else None,
            "context": self.context.to_dict() if self.context else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorDetails':
        """Create from dictionary."""
        return cls(
            error_type=data["error_type"],
            error_message=data["error_message"],
            traceback=[StackFrame.from_dict(frame) for frame in data["traceback"]] if data.get("traceback") else None,
            context=ErrorContext.from_dict(data["context"]) if data.get("context") else None
        )

@dataclass
class ErrorPattern:
    """Error pattern information."""
    pattern_id: str
    error_type: str
    pattern: str  # Regex pattern or similarity pattern
    examples: List[str] = field(default_factory=list)
    description: Optional[str] = None
    fix_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "error_type": self.error_type,
            "pattern": self.pattern,
            "examples": self.examples,
            "description": self.description,
            "fix_suggestions": self.fix_suggestions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorPattern':
        """Create from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            error_type=data["error_type"],
            pattern=data["pattern"],
            examples=data.get("examples", []),
            description=data.get("description"),
            fix_suggestions=data.get("fix_suggestions", [])
        )

@dataclass
class ClassifiedError:
    """Classified error information."""
    error_id: str
    timestamp: float
    details: ErrorDetails
    category: ErrorCategory
    severity: ErrorSeverity
    impact: ErrorImpact
    pattern_id: Optional[str] = None
    is_resolved: bool = False
    resolution_time: Optional[float] = None
    resolution_steps: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "details": self.details.to_dict(),
            "category": self.category.name,
            "severity": self.severity.name,
            "impact": self.impact.name,
            "pattern_id": self.pattern_id,
            "is_resolved": self.is_resolved,
            "resolution_time": self.resolution_time,
            "resolution_steps": self.resolution_steps
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassifiedError':
        """Create from dictionary."""
        return cls(
            error_id=data["error_id"],
            timestamp=data["timestamp"],
            details=ErrorDetails.from_dict(data["details"]),
            category=ErrorCategory[data["category"]],
            severity=ErrorSeverity[data["severity"]],
            impact=ErrorImpact[data["impact"]],
            pattern_id=data.get("pattern_id"),
            is_resolved=data.get("is_resolved", False),
            resolution_time=data.get("resolution_time"),
            resolution_steps=data.get("resolution_steps")
        )

class ErrorClassifier:
    """
    Error classifier for the FixWurx platform.
    
    This class provides sophisticated error classification, categorization,
    pattern recognition, and analysis.
    """
    
    def __init__(self):
        """Initialize the error classifier."""
        self._error_history: List[ClassifiedError] = []
        self._error_patterns: Dict[str, ErrorPattern] = {}
        self._error_groups: Dict[str, List[str]] = {}  # pattern_id -> [error_id, ...]
        self._error_counts: Dict[str, int] = {}  # error_type -> count
        self._category_rules: Dict[str, ErrorCategory] = self._initialize_category_rules()
        self._severity_rules: Dict[str, ErrorSeverity] = self._initialize_severity_rules()
        self._impact_rules: Dict[str, ErrorImpact] = self._initialize_impact_rules()
        self._similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD
        
        # Load patterns from database
        self._load_patterns()
        
        # Register error sensor
        self._register_error_sensors()
        
        logger.info("Error classifier initialized")
    
    def _initialize_category_rules(self) -> Dict[str, ErrorCategory]:
        """Initialize category rules."""
        rules = {
            # Syntax errors
            "SyntaxError": ErrorCategory.SYNTAX,
            "IndentationError": ErrorCategory.SYNTAX,
            "TabError": ErrorCategory.SYNTAX,
            
            # Runtime errors
            "RuntimeError": ErrorCategory.RUNTIME,
            "RecursionError": ErrorCategory.RUNTIME,
            "NotImplementedError": ErrorCategory.RUNTIME,
            "SystemError": ErrorCategory.RUNTIME,
            
            # Logical errors
            "AssertionError": ErrorCategory.LOGICAL,
            
            # Type errors
            "TypeError": ErrorCategory.TYPE,
            "AttributeError": ErrorCategory.TYPE,
            
            # Value errors
            "ValueError": ErrorCategory.VALUE,
            "UnicodeError": ErrorCategory.VALUE,
            "UnicodeDecodeError": ErrorCategory.VALUE,
            "UnicodeEncodeError": ErrorCategory.VALUE,
            "UnicodeTranslateError": ErrorCategory.VALUE,
            
            # I/O errors
            "IOError": ErrorCategory.IO,
            "FileNotFoundError": ErrorCategory.IO,
            "FileExistsError": ErrorCategory.IO,
            "IsADirectoryError": ErrorCategory.IO,
            "NotADirectoryError": ErrorCategory.IO,
            
            # Network errors
            "ConnectionError": ErrorCategory.NETWORK,
            "ConnectionRefusedError": ErrorCategory.NETWORK,
            "ConnectionResetError": ErrorCategory.NETWORK,
            "ConnectionAbortedError": ErrorCategory.NETWORK,
            "TimeoutError": ErrorCategory.NETWORK,
            
            # Resource errors
            "MemoryError": ErrorCategory.RESOURCE,
            "BufferError": ErrorCategory.RESOURCE,
            "OSError": ErrorCategory.RESOURCE,
            "ProcessLookupError": ErrorCategory.RESOURCE,
            
            # Permission errors
            "PermissionError": ErrorCategory.PERMISSION,
            
            # Other errors
            "ImportError": ErrorCategory.DEPENDENCY,
            "ModuleNotFoundError": ErrorCategory.DEPENDENCY,
            "KeyError": ErrorCategory.STATE,
            "IndexError": ErrorCategory.STATE,
            "StopIteration": ErrorCategory.STATE,
            "OverflowError": ErrorCategory.VALUE,
            "ZeroDivisionError": ErrorCategory.VALUE,
            "FloatingPointError": ErrorCategory.VALUE,
            "SystemExit": ErrorCategory.RUNTIME,
            "KeyboardInterrupt": ErrorCategory.RUNTIME,
            "GeneratorExit": ErrorCategory.RUNTIME,
            "Exception": ErrorCategory.UNKNOWN
        }
        
        return rules
    
    def _initialize_severity_rules(self) -> Dict[str, ErrorSeverity]:
        """Initialize severity rules."""
        rules = {
            # Fatal errors
            "SystemExit": ErrorSeverity.FATAL,
            "KeyboardInterrupt": ErrorSeverity.FATAL,
            
            # Critical errors
            "MemoryError": ErrorSeverity.CRITICAL,
            "SystemError": ErrorSeverity.CRITICAL,
            "RecursionError": ErrorSeverity.CRITICAL,
            
            # Errors
            "RuntimeError": ErrorSeverity.ERROR,
            "TypeError": ErrorSeverity.ERROR,
            "ValueError": ErrorSeverity.ERROR,
            "AttributeError": ErrorSeverity.ERROR,
            "IOError": ErrorSeverity.ERROR,
            "FileNotFoundError": ErrorSeverity.ERROR,
            "SyntaxError": ErrorSeverity.ERROR,
            "ImportError": ErrorSeverity.ERROR,
            "ModuleNotFoundError": ErrorSeverity.ERROR,
            "ConnectionError": ErrorSeverity.ERROR,
            "OSError": ErrorSeverity.ERROR,
            "PermissionError": ErrorSeverity.ERROR,
            
            # Warnings
            "DeprecationWarning": ErrorSeverity.WARNING,
            "ResourceWarning": ErrorSeverity.WARNING,
            "UserWarning": ErrorSeverity.WARNING,
            "PendingDeprecationWarning": ErrorSeverity.WARNING,
            "SyntaxWarning": ErrorSeverity.WARNING,
            "RuntimeWarning": ErrorSeverity.WARNING,
            "FutureWarning": ErrorSeverity.WARNING,
            "ImportWarning": ErrorSeverity.WARNING,
            "UnicodeWarning": ErrorSeverity.WARNING,
            "BytesWarning": ErrorSeverity.WARNING,
            
            # Default
            "Exception": ErrorSeverity.ERROR
        }
        
        return rules
    
    def _initialize_impact_rules(self) -> Dict[str, ErrorImpact]:
        """Initialize impact rules."""
        rules = {
            # Critical impact
            "SystemExit": ErrorImpact.CRITICAL,
            "KeyboardInterrupt": ErrorImpact.CRITICAL,
            "MemoryError": ErrorImpact.CRITICAL,
            "SystemError": ErrorImpact.CRITICAL,
            
            # High impact
            "RuntimeError": ErrorImpact.HIGH,
            "RecursionError": ErrorImpact.HIGH,
            "ImportError": ErrorImpact.HIGH,
            "ModuleNotFoundError": ErrorImpact.HIGH,
            "SyntaxError": ErrorImpact.HIGH,
            "IOError": ErrorImpact.HIGH,
            "ConnectionError": ErrorImpact.HIGH,
            "OSError": ErrorImpact.HIGH,
            "PermissionError": ErrorImpact.HIGH,
            
            # Medium impact
            "TypeError": ErrorImpact.MEDIUM,
            "ValueError": ErrorImpact.MEDIUM,
            "AttributeError": ErrorImpact.MEDIUM,
            "KeyError": ErrorImpact.MEDIUM,
            "IndexError": ErrorImpact.MEDIUM,
            "FileNotFoundError": ErrorImpact.MEDIUM,
            
            # Low impact
            "DeprecationWarning": ErrorImpact.LOW,
            "ResourceWarning": ErrorImpact.LOW,
            "UserWarning": ErrorImpact.LOW,
            "PendingDeprecationWarning": ErrorImpact.LOW,
            "SyntaxWarning": ErrorImpact.LOW,
            "RuntimeWarning": ErrorImpact.LOW,
            "FutureWarning": ErrorImpact.LOW,
            "ImportWarning": ErrorImpact.LOW,
            "UnicodeWarning": ErrorImpact.LOW,
            "BytesWarning": ErrorImpact.LOW,
            
            # Default
            "Exception": ErrorImpact.MEDIUM
        }
        
        return rules
    
    def _load_patterns(self) -> None:
        """Load error patterns from database."""
        db_path = os.path.expanduser(ERROR_DB_PATH)
        if not os.path.exists(db_path):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Create empty database
            self._save_patterns()
            return
        
        try:
            with open(db_path, 'r') as f:
                data = json.load(f)
                
                # Load patterns
                for pattern_data in data.get("patterns", []):
                    pattern = ErrorPattern.from_dict(pattern_data)
                    self._error_patterns[pattern.pattern_id] = pattern
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
    
    def _save_patterns(self) -> None:
        """Save error patterns to database."""
        db_path = os.path.expanduser(ERROR_DB_PATH)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Save patterns
            data = {
                "patterns": [pattern.to_dict() for pattern in self._error_patterns.values()]
            }
            
            with open(db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def _register_error_sensors(self) -> None:
        """Register error sensors."""
        # Register error count sensor
        error_count_metadata = SensorMetadata(
            id="error.count",
            name="Error Count",
            description="Number of errors by type",
            type=SensorType.APPLICATION,
            data_type=DataType.DICT,
            unit="count"
        )
        
        error_count_sensor = CustomSensor(
            metadata=error_count_metadata,
            sample_func=self.get_error_counts
        )
        
        add_sensor(error_count_sensor)
        
        # Register error severity sensor
        error_severity_metadata = SensorMetadata(
            id="error.severity",
            name="Error Severity",
            description="Number of errors by severity",
            type=SensorType.APPLICATION,
            data_type=DataType.DICT,
            unit="count"
        )
        
        error_severity_sensor = CustomSensor(
            metadata=error_severity_metadata,
            sample_func=self.get_severity_counts
        )
        
        add_sensor(error_severity_sensor)
        
        # Register error category sensor
        error_category_metadata = SensorMetadata(
            id="error.category",
            name="Error Category",
            description="Number of errors by category",
            type=SensorType.APPLICATION,
            data_type=DataType.DICT,
            unit="count"
        )
        
        error_category_sensor = CustomSensor(
            metadata=error_category_metadata,
            sample_func=self.get_category_counts
        )
        
        add_sensor(error_category_sensor)
    
    def capture_exception(self, exc_info: Optional[Tuple] = None,
                         context: Optional[ErrorContext] = None) -> ClassifiedError:
        """
        Capture and classify an exception.
        
        Args:
            exc_info: Exception info tuple (type, value, traceback).
            context: Error context.
            
        Returns:
            Classified error.
        """
        if exc_info is None:
            exc_info = sys.exc_info()
        
        exc_type, exc_value, exc_traceback = exc_info
        
        # Extract error type and message
        error_type = exc_type.__name__ if exc_type else "Unknown"
        error_message = str(exc_value) if exc_value else ""
        
        # Extract traceback
        tb_frames = []
        if exc_traceback:
            for frame_info in traceback.extract_tb(exc_traceback, limit=MAX_STACK_DEPTH):
                frame = StackFrame(
                    filename=frame_info.filename,
                    lineno=frame_info.lineno,
                    function=frame_info.name,
                    code_context=[frame_info.line] if frame_info.line else None
                )
                tb_frames.append(frame)
        
        # Create error details
        details = ErrorDetails(
            error_type=error_type,
            error_message=error_message,
            traceback=tb_frames,
            context=context
        )
        
        # Classify error
        return self.classify_error(details)
    
    def classify_error(self, details: ErrorDetails) -> ClassifiedError:
        """
        Classify an error.
        
        Args:
            details: Error details.
            
        Returns:
            Classified error.
        """
        # Generate error ID
        error_id = self._generate_error_id(details)
        
        # Determine category
        category = self._determine_category(details.error_type)
        
        # Determine severity
        severity = self._determine_severity(details.error_type)
        
        # Determine impact
        impact = self._determine_impact(details.error_type)
        
        # Match error pattern
        pattern_id = self._match_error_pattern(details)
        
        # Create classified error
        error = ClassifiedError(
            error_id=error_id,
            timestamp=time.time(),
            details=details,
            category=category,
            severity=severity,
            impact=impact,
            pattern_id=pattern_id
        )
        
        # Add to history
        self._add_to_history(error)
        
        # Update error counts
        self._update_error_counts(details.error_type)
        
        # Emit error event
        self._emit_error_event(error)
        
        return error
    
    def _generate_error_id(self, details: ErrorDetails) -> str:
        """
        Generate an error ID.
        
        Args:
            details: Error details.
            
        Returns:
            Error ID.
        """
        # Create a hash of error type, message, and first few frames
        hash_input = f"{details.error_type}:{details.error_message}"
        
        if details.traceback:
            for i, frame in enumerate(details.traceback[:3]):  # Use first 3 frames
                hash_input += f":{frame.filename}:{frame.lineno}:{frame.function}"
        
        # Create hash
        error_hash = hashlib.md5(hash_input.encode()).hexdigest()
        
        return f"error_{int(time.time())}_{error_hash[:8]}"
    
    def _determine_category(self, error_type: str) -> ErrorCategory:
        """
        Determine the error category.
        
        Args:
            error_type: Error type.
            
        Returns:
            Error category.
        """
        # Check if we have a rule for this error type
        if error_type in self._category_rules:
            return self._category_rules[error_type]
        
        # Check if we have a rule for a parent error type
        for parent_type, category in self._category_rules.items():
            try:
                parent_class = eval(parent_type)
                error_class = eval(error_type)
                if issubclass(error_class, parent_class):
                    return category
            except (NameError, TypeError):
                continue
        
        # Default to UNKNOWN
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error_type: str) -> ErrorSeverity:
        """
        Determine the error severity.
        
        Args:
            error_type: Error type.
            
        Returns:
            Error severity.
        """
        # Check if we have a rule for this error type
        if error_type in self._severity_rules:
            return self._severity_rules[error_type]
        
        # Check if we have a rule for a parent error type
        for parent_type, severity in self._severity_rules.items():
            try:
                parent_class = eval(parent_type)
                error_class = eval(error_type)
                if issubclass(error_class, parent_class):
                    return severity
            except (NameError, TypeError):
                continue
        
        # Default to ERROR
        return ErrorSeverity.ERROR
    
    def _determine_impact(self, error_type: str) -> ErrorImpact:
        """
        Determine the error impact.
        
        Args:
            error_type: Error type.
            
        Returns:
            Error impact.
        """
        # Check if we have a rule for this error type
        if error_type in self._impact_rules:
            return self._impact_rules[error_type]
        
        # Check if we have a rule for a parent error type
        for parent_type, impact in self._impact_rules.items():
            try:
                parent_class = eval(parent_type)
                error_class = eval(error_type)
                if issubclass(error_class, parent_class):
                    return impact
            except (NameError, TypeError):
                continue
        
        # Default to MEDIUM
        return ErrorImpact.MEDIUM
    
    def _match_error_pattern(self, details: ErrorDetails) -> Optional[str]:
        """
        Match error to a known pattern.
        
        Args:
            details: Error details.
            
        Returns:
            Pattern ID or None if no match.
        """
        # First, try exact match by error type and message
        error_key = f"{details.error_type}:{details.error_message}"
        
        # Check for exact matches
        for pattern_id, pattern in self._error_patterns.items():
            if pattern.error_type == details.error_type:
                # Try regex match
                try:
                    if re.match(pattern.pattern, details.error_message):
                        return pattern_id
                except re.error:
                    # Not a valid regex, try string similarity
                    pass
        
        # Try string similarity
        for pattern_id, pattern in self._error_patterns.items():
            if pattern.error_type == details.error_type:
                # Compare with examples
                for example in pattern.examples:
                    similarity = self._calculate_similarity(details.error_message, example)
                    if similarity >= self._similarity_threshold:
                        return pattern_id
        
        # No match found, create a new pattern if it's a unique error
        return self._create_pattern_if_unique(details)
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings.
        
        Args:
            str1: First string.
            str2: Second string.
            
        Returns:
            Similarity score (0-1).
        """
        return difflib.SequenceMatcher(None, str1, str2).ratio()
    
    def _create_pattern_if_unique(self, details: ErrorDetails) -> Optional[str]:
        """
        Create a new pattern if the error is unique.
        
        Args:
            details: Error details.
            
        Returns:
            Pattern ID or None if not unique.
        """
        # Check if this is a unique error
        for pattern_id, pattern in self._error_patterns.items():
            if pattern.error_type == details.error_type:
                for example in pattern.examples:
                    similarity = self._calculate_similarity(details.error_message, example)
                    if similarity >= self._similarity_threshold:
                        return None
        
        # Create a new pattern
        pattern_id = f"pattern_{int(time.time())}_{len(self._error_patterns)}"
        
        # Try to create a regex pattern
        pattern_str = self._create_regex_pattern(details.error_message)
        
        # Create pattern
        pattern = ErrorPattern(
            pattern_id=pattern_id,
            error_type=details.error_type,
            pattern=pattern_str,
            examples=[details.error_message]
        )
        
        # Add to patterns
        self._error_patterns[pattern_id] = pattern
        
        # Save patterns
        self._save_patterns()
        
        return pattern_id
    
    def _create_regex_pattern(self, error_message: str) -> str:
        """
        Create a regex pattern from an error message.
        
        Args:
            error_message: Error message.
            
        Returns:
            Regex pattern.
        """
        # Escape special characters
        pattern = re.escape(error_message)
        
        # Replace common variable parts
        pattern = re.sub(r'\\\'[^\\\']+\\\'', r'\\\'[^\\\']+\\\'', pattern)  # Replace quoted strings
        pattern = re.sub(r'\\"[^\\"]+\\"', r'\\"[^\\"]+\\"', pattern)  # Replace double-quoted strings
        pattern = re.sub(r'\\d+', r'\\d+', pattern)  # Replace numbers
        pattern = re.sub(r'0x[0-9a-f]+', r'0x[0-9a-f]+', pattern)  # Replace hex numbers
        
        return pattern
    
    def _add_to_history(self, error: ClassifiedError) -> None:
        """
        Add error to history.
        
        Args:
            error: Classified error.
        """
        # Add to history
        self._error_history.append(error)
        
        # Trim history if needed
        if len(self._error_history) > MAX_ERROR_HISTORY:
            self._error_history = self._error_history[-MAX_ERROR_HISTORY:]
        
        # Add to error groups
        if error.pattern_id:
            if error.pattern_id not in self._error_groups:
                self._error_groups[error.pattern_id] = []
            self._error_groups[error.pattern_id].append(error.error_id)
    
    def _update_error_counts(self, error_type: str) -> None:
        """
        Update error counts.
        
        Args:
            error_type: Error type.
        """
        if error_type not in self._error_counts:
            self._error_counts[error_type] = 0
        
        self._error_counts[error_type] += 1
    
    def _emit_error_event(self, error: ClassifiedError) -> None:
        """
        Emit error event.
        
        Args:
            error: Classified error.
        """
        # Create event data
        event_data = {
            "error_id": error.error_id,
            "error_type": error.details.error_type,
            "error_message": error.details.error_message,
            "category": error.category.name,
            "severity": error.severity.name,
            "impact": error.impact.name,
            "timestamp": error.timestamp
        }
        
        # Emit event
        try:
            emit_event(EventType.ERROR, event_data)
        except Exception as e:
            logger.error(f"Error emitting error event: {e}")
    
    def resolve_error(self, error_id: str, resolution_steps: Optional[List[str]] = None) -> bool:
        """
        Mark an error as resolved.
        
        Args:
            error_id: Error ID.
            resolution_steps: Optional resolution steps.
            
        Returns:
            True if error was resolved, False otherwise.
        """
        # Find error
        for error in self._error_history:
            if error.error_id == error_id:
                error.is_resolved = True
                error.resolution_time = time.time()
                error.resolution_steps = resolution_steps
                
                # Emit resolution event
                self._emit_resolution_event(error)
                
                return True
        
        return False
    
    def _emit_resolution_event(self, error: ClassifiedError) -> None:
        """
        Emit resolution event.
        
        Args:
            error: Resolved error.
        """
        # Create event data
        event_data = {
            "error_id": error.error_id,
            "error_type": error.details.error_type,
            "resolution_time": error.resolution_time,
            "resolution_steps": error.resolution_steps
        }
        
        # Emit event
        try:
            emit_event(EventType.ERROR_RESOLVED, event_data)
        except Exception as e:
            logger.error(f"Error emitting resolution event: {e}")
    
    def get_error(self, error_id: str) -> Optional[ClassifiedError]:
        """
        Get an error by ID.
        
        Args:
            error_id: Error ID.
            
        Returns:
            Error or None if not found.
        """
        for error in self._error_history:
            if error.error_id == error_id:
                return error
        
        return None
    
    def get_errors(self, error_type: Optional[str] = None,
                   category: Optional[ErrorCategory] = None,
                   severity: Optional[ErrorSeverity] = None,
                   pattern_id: Optional[str] = None,
                   is_resolved: Optional[bool] = None,
                   limit: int = 100) -> List[ClassifiedError]:
        """
        Get errors matching criteria.
        
        Args:
            error_type: Error type or None for all types.
            category: Error category or None for all categories.
            severity: Error severity or None for all severities.
            pattern_id: Pattern ID or None for all patterns.
            is_resolved: Resolution status or None for all statuses.
            limit: Maximum number of errors to return.
            
        Returns:
            List of errors matching criteria.
        """
        # Filter errors
        filtered_errors = []
        for error in self._error_history:
            if error_type is not None and error.details.error_type != error_type:
                continue
            
            if category is not None and error.category != category:
                continue
            
            if severity is not None and error.severity != severity:
                continue
            
            if pattern_id is not None and error.pattern_id != pattern_id:
                continue
            
            if is_resolved is not None and error.is_resolved != is_resolved:
                continue
            
            filtered_errors.append(error)
        
        # Sort by timestamp (newest first)
        filtered_errors.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        return filtered_errors[:limit]
    
    def get_error_pattern(self, pattern_id: str) -> Optional[ErrorPattern]:
        """
        Get an error pattern.
        
        Args:
            pattern_id: Pattern ID.
            
        Returns:
            Pattern or None if not found.
        """
        return self._error_patterns.get(pattern_id)
    
    def get_error_patterns(self, error_type: Optional[str] = None) -> List[ErrorPattern]:
        """
        Get error patterns.
        
        Args:
            error_type: Error type or None for all types.
            
        Returns:
            List of patterns.
        """
        if error_type is None:
            return list(self._error_patterns.values())
        
        return [p for p in self._error_patterns.values() if p.error_type == error_type]
    
    def add_fix_suggestion(self, pattern_id: str, suggestion: str) -> bool:
        """
        Add a fix suggestion to an error pattern.
        
        Args:
            pattern_id: Pattern ID.
            suggestion: Fix suggestion.
            
        Returns:
            True if suggestion was added, False otherwise.
        """
        pattern = self.get_error_pattern(pattern_id)
        if pattern is None:
            return False
        
        if suggestion not in pattern.fix_suggestions:
            pattern.fix_suggestions.append(suggestion)
            self._save_patterns()
        
        return True
    
    def update_pattern_description(self, pattern_id: str, description: str) -> bool:
        """
        Update error pattern description.
        
        Args:
            pattern_id: Pattern ID.
            description: Pattern description.
            
        Returns:
            True if description was updated, False otherwise.
        """
        pattern = self.get_error_pattern(pattern_id)
        if pattern is None:
            return False
        
        pattern.description = description
        self._save_patterns()
        
        return True
    
    def get_error_trends(self, error_type: Optional[str] = None,
                         period: int = 86400) -> Dict[str, ErrorTrend]:
        """
        Get error trends.
        
        Args:
            error_type: Error type or None for all types.
            period: Period in seconds to analyze.
            
        Returns:
            Dictionary of error types to trends.
        """
        now = time.time()
        start_time = now - period
        
        # Get errors in period
        errors = self.get_errors(error_type=error_type, limit=MAX_ERROR_HISTORY)
        errors_in_period = [e for e in errors if e.timestamp >= start_time]
        
        # Group by error type
        error_types = {}
        for error in errors_in_period:
            if error.details.error_type not in error_types:
                error_types[error.details.error_type] = []
            error_types[error.details.error_type].append(error)
        
        # Analyze trends
        trends = {}
        for err_type, err_list in error_types.items():
            # Count errors in each half of the period
            half_period = period / 2
            first_half = len([e for e in err_list if e.timestamp < start_time + half_period])
            second_half = len([e for e in err_list if e.timestamp >= start_time + half_period])
            
            # Determine trend
            if len(err_list) == 1:
                trends[err_type] = ErrorTrend.NEW
            elif all(e.is_resolved for e in err_list):
                trends[err_type] = ErrorTrend.RESOLVED
            elif first_half == 0 and second_half > 0:
                trends[err_type] = ErrorTrend.NEW
            elif second_half > first_half * 1.5:
                trends[err_type] = ErrorTrend.INCREASING
            elif first_half > second_half * 1.5:
                trends[err_type] = ErrorTrend.DECREASING
            elif abs(second_half - first_half) <= 1:
                trends[err_type] = ErrorTrend.STABLE
            else:
                trends[err_type] = ErrorTrend.SPORADIC
        
        return trends
    
    def get_error_counts(self) -> Dict[str, int]:
        """
        Get error counts by type.
        
        Returns:
            Dictionary of error types to counts.
        """
        return self._error_counts.copy()
    
    def get_severity_counts(self) -> Dict[str, int]:
        """
        Get error counts by severity.
        
        Returns:
            Dictionary of severity levels to counts.
        """
        counts = {}
        for error in self._error_history:
            severity = error.severity.name
            if severity not in counts:
                counts[severity] = 0
            counts[severity] += 1
        
        return counts
    
    def get_category_counts(self) -> Dict[str, int]:
        """
        Get error counts by category.
        
        Returns:
            Dictionary of categories to counts.
        """
        counts = {}
        for error in self._error_history:
            category = error.category.name
            if category not in counts:
                counts[category] = 0
            counts[category] += 1
        
        return counts
    
    def get_similar_errors(self, error_id: str, threshold: Optional[float] = None) -> List[ClassifiedError]:
        """
        Get errors similar to the specified error.
        
        Args:
            error_id: Error ID.
            threshold: Similarity threshold or None for default.
            
        Returns:
            List of similar errors.
        """
        if threshold is None:
            threshold = self._similarity_threshold
        
        # Get error
        error = self.get_error(error_id)
        if error is None:
            return []
        
        # Get errors of the same type
        same_type_errors = self.get_errors(error_type=error.details.error_type)
        
        # Find similar errors
        similar_errors = []
        for other in same_type_errors:
            if other.error_id == error_id:
                continue
            
            similarity = self._calculate_similarity(
                error.details.error_message, other.details.error_message
            )
            
            if similarity >= threshold:
                similar_errors.append(other)
        
        return similar_errors
    
    def analyze_error(self, error_id: str) -> Dict[str, Any]:
        """
        Analyze an error in detail.
        
        Args:
            error_id: Error ID.
            
        Returns:
            Analysis results.
        """
        # Get error
        error = self.get_error(error_id)
        if error is None:
            return {"error": "Error not found"}
        
        # Get pattern
        pattern = self.get_error_pattern(error.pattern_id) if error.pattern_id else None
        
        # Get similar errors
        similar_errors = self.get_similar_errors(error_id)
        
        # Get trend
        trends = self.get_error_trends(error_type=error.details.error_type)
        trend = trends.get(error.details.error_type)
        
        # Analyze traceback
        traceback_analysis = self._analyze_traceback(error.details.traceback) if error.details.traceback else None
        
        # Assemble results
        results = {
            "error_id": error.error_id,
            "error_type": error.details.error_type,
            "error_message": error.details.error_message,
            "category": error.category.name,
            "severity": error.severity.name,
            "impact": error.impact.name,
            "timestamp": error.timestamp,
            "is_resolved": error.is_resolved,
            "resolution_time": error.resolution_time,
            "resolution_steps": error.resolution_steps,
            "pattern": pattern.to_dict() if pattern else None,
            "similar_errors": [e.error_id for e in similar_errors],
            "trend": trend.name if trend else None,
            "traceback_analysis": traceback_analysis,
            "fix_suggestions": pattern.fix_suggestions if pattern else []
        }
        
        return results
    
    def _analyze_traceback(self, traceback: List[StackFrame]) -> Dict[str, Any]:
        """
        Analyze a traceback.
        
        Args:
            traceback: List of stack frames.
            
        Returns:
            Analysis results.
        """
        # Count frames by file
        files = {}
        for frame in traceback:
            if frame.filename not in files:
                files[frame.filename] = 0
            files[frame.filename] += 1
        
        # Count frames by function
        functions = {}
        for frame in traceback:
            if frame.function not in functions:
                functions[frame.function] = 0
            functions[frame.function] += 1
        
        # Find most common file and function
        most_common_file = max(files.items(), key=lambda x: x[1])[0] if files else None
        most_common_function = max(functions.items(), key=lambda x: x[1])[0] if functions else None
        
        # Assemble results
        results = {
            "frames": len(traceback),
            "files": files,
            "functions": functions,
            "most_common_file": most_common_file,
            "most_common_function": most_common_function,
            "root_frame": traceback[-1].to_dict() if traceback else None,
            "error_frame": traceback[0].to_dict() if traceback else None
        }
        
        return results
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """
        Set similarity threshold.
        
        Args:
            threshold: Similarity threshold (0-1).
        """
        self._similarity_threshold = max(0.0, min(1.0, threshold))
    
    def get_similarity_threshold(self) -> float:
        """
        Get similarity threshold.
        
        Returns:
            Similarity threshold.
        """
        return self._similarity_threshold
    
    def clear_history(self) -> None:
        """Clear error history."""
        self._error_history.clear()
        self._error_groups.clear()
        self._error_counts.clear()
    
    def export_data(self) -> Dict[str, Any]:
        """
        Export error data.
        
        Returns:
            Exported data.
        """
        return {
            "patterns": [p.to_dict() for p in self._error_patterns.values()],
            "errors": [e.to_dict() for e in self._error_history],
            "error_counts": self._error_counts,
            "error_groups": self._error_groups
        }
    
    def import_data(self, data: Dict[str, Any]) -> None:
        """
        Import error data.
        
        Args:
            data: Exported data.
        """
        # Import patterns
        for pattern_data in data.get("patterns", []):
            pattern = ErrorPattern.from_dict(pattern_data)
            self._error_patterns[pattern.pattern_id] = pattern
        
        # Import errors
        for error_data in data.get("errors", []):
            error = ClassifiedError.from_dict(error_data)
            self._error_history.append(error)
        
        # Import error counts
        self._error_counts.update(data.get("error_counts", {}))
        
        # Import error groups
        self._error_groups.update(data.get("error_groups", {}))
        
        # Save patterns
        self._save_patterns()


# Create singleton error classifier
_error_classifier = None

def get_error_classifier() -> ErrorClassifier:
    """
    Get the singleton error classifier instance.
    
    Returns:
        Error classifier instance.
    """
    global _error_classifier
    
    if _error_classifier is None:
        _error_classifier = ErrorClassifier()
    
    return _error_classifier

def capture_exception(exc_info: Optional[Tuple] = None,
                     context: Optional[ErrorContext] = None) -> ClassifiedError:
    """
    Capture and classify an exception.
    
    Args:
        exc_info: Exception info tuple (type, value, traceback).
        context: Error context.
        
    Returns:
        Classified error.
    """
    return get_error_classifier().capture_exception(exc_info, context)

def classify_error(details: ErrorDetails) -> ClassifiedError:
    """
    Classify an error.
    
    Args:
        details: Error details.
        
    Returns:
        Classified error.
    """
    return get_error_classifier().classify_error(details)

def resolve_error(error_id: str, resolution_steps: Optional[List[str]] = None) -> bool:
    """
    Mark an error as resolved.
    
    Args:
        error_id: Error ID.
        resolution_steps: Optional resolution steps.
        
    Returns:
        True if error was resolved, False otherwise.
    """
    return get_error_classifier().resolve_error(error_id, resolution_steps)

def get_error(error_id: str) -> Optional[ClassifiedError]:
    """
    Get an error by ID.
    
    Args:
        error_id: Error ID.
        
    Returns:
        Error or None if not found.
    """
    return get_error_classifier().get_error(error_id)

def get_errors(error_type: Optional[str] = None,
              category: Optional[ErrorCategory] = None,
              severity: Optional[ErrorSeverity] = None,
              pattern_id: Optional[str] = None,
              is_resolved: Optional[bool] = None,
              limit: int = 100) -> List[ClassifiedError]:
    """
    Get errors matching criteria.
    
    Args:
        error_type: Error type or None for all types.
        category: Error category or None for all categories.
        severity: Error severity or None for all severities.
        pattern_id: Pattern ID or None for all patterns.
        is_resolved: Resolution status or None for all statuses.
        limit: Maximum number of errors to return.
        
    Returns:
        List of errors matching criteria.
    """
    return get_error_classifier().get_errors(
        error_type, category, severity, pattern_id, is_resolved, limit
    )

def analyze_error(error_id: str) -> Dict[str, Any]:
    """
    Analyze an error in detail.
    
    Args:
        error_id: Error ID.
        
    Returns:
        Analysis results.
    """
    return get_error_classifier().analyze_error(error_id)

def get_error_trends(error_type: Optional[str] = None,
                    period: int = 86400) -> Dict[str, ErrorTrend]:
    """
    Get error trends.
    
    Args:
        error_type: Error type or None for all types.
        period: Period in seconds to analyze.
        
    Returns:
        Dictionary of error types to trends.
    """
    return get_error_classifier().get_error_trends(error_type, period)

def get_error_counts() -> Dict[str, int]:
    """
    Get error counts by type.
    
    Returns:
        Dictionary of error types to counts.
    """
    return get_error_classifier().get_error_counts()

def get_severity_counts() -> Dict[str, int]:
    """
    Get error counts by severity.
    
    Returns:
        Dictionary of severity levels to counts.
    """
    return get_error_classifier().get_severity_counts()

def get_category_counts() -> Dict[str, int]:
    """
    Get error counts by category.
    
    Returns:
        Dictionary of categories to counts.
    """
    return get_error_classifier().get_category_counts()

def add_fix_suggestion(pattern_id: str, suggestion: str) -> bool:
    """
    Add a fix suggestion to an error pattern.
    
    Args:
        pattern_id: Pattern ID.
        suggestion: Fix suggestion.
        
    Returns:
        True if suggestion was added, False otherwise.
    """
    return get_error_classifier().add_fix_suggestion(pattern_id, suggestion)

def update_pattern_description(pattern_id: str, description: str) -> bool:
    """
    Update error pattern description.
    
    Args:
        pattern_id: Pattern ID.
        description: Pattern description.
        
    Returns:
        True if description was updated, False otherwise.
    """
    return get_error_classifier().update_pattern_description(pattern_id, description)

def clear_error_history() -> None:
    """Clear error history."""
    get_error_classifier().clear_history()

def export_error_data() -> Dict[str, Any]:
    """
    Export error data.
    
    Returns:
        Exported data.
    """
    return get_error_classifier().export_data()

def import_error_data(data: Dict[str, Any]) -> None:
    """
    Import error data.
    
    Args:
        data: Exported data.
    """
    get_error_classifier().import_data(data)

def exception_handler(func):
    """
    Decorator to automatically capture and classify exceptions.
    
    Args:
        func: Function to decorate.
        
    Returns:
        Decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            # Capture exception
            context = ErrorContext(
                function=func.__name__,
                args=list(map(str, args)),
                custom_data={"kwargs": {k: str(v) for k, v in kwargs.items()}}
            )
            
            error = capture_exception(context=context)
            
            # Re-raise exception
            raise
    
    return wrapper

# Initialize error classifier
if not any(arg.endswith('test.py') for arg in sys.argv):
    get_error_classifier()
