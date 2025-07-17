#!/usr/bin/env python3
"""
error_reporting.py
─────────────────
Error reporting infrastructure for the FixWurx platform.

This module provides a comprehensive error reporting system that collects,
formats, and distributes error reports to various output destinations,
including console, log files, dashboards, and external services.
"""

import os
import sys
import json
import logging
import time
import traceback
import datetime
import re
import smtplib
import threading
import uuid
import socket
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import urllib.request
import urllib.error
import urllib.parse
import queue
import hashlib

# Internal imports
from shell_environment import register_event_handler, emit_event, EventType
from advanced_error_classification import (
    ErrorSeverity, ErrorCategory, ErrorImpact, ErrorTrend,
    ClassifiedError, ErrorPattern, ErrorDetails, StackFrame, ErrorContext,
    get_error, get_errors, analyze_error
)

# Configure logging
logger = logging.getLogger("ErrorReporting")

# Constants
DEFAULT_EMAIL_SENDER = "fixwurx-noreply@example.com"
DEFAULT_SLACK_CHANNEL = "#fixwurx-alerts"
DEFAULT_REPORT_FORMAT = "text"  # "text", "html", "json", "markdown"
DEFAULT_MAX_QUEUE_SIZE = 1000
DEFAULT_DISPATCH_INTERVAL = 1.0  # seconds
DEFAULT_COOLDOWN_PERIOD = 300  # seconds
DEFAULT_LOG_PATH = "~/.fixwurx/error_reports"
DEFAULT_EMAIL_SUBJECT_PREFIX = "[FixWurx Alert]"
DEFAULT_WEBHOOK_TIMEOUT = 10  # seconds
DEFAULT_ERROR_HISTORY_COUNT = 5  # number of similar errors to include

class ReportFormat(Enum):
    """Report format types."""
    TEXT = auto()      # Plain text format
    HTML = auto()      # HTML format
    JSON = auto()      # JSON format
    MARKDOWN = auto()  # Markdown format

class ReportDestination(Enum):
    """Report destination types."""
    CONSOLE = auto()   # Console output
    LOG_FILE = auto()  # Log file
    EMAIL = auto()     # Email
    WEBHOOK = auto()   # Webhook (generic)
    SLACK = auto()     # Slack webhook
    TEAMS = auto()     # Microsoft Teams webhook
    CUSTOM = auto()    # Custom destination

class ReportingRule:
    """Rule for determining when and where to send error reports."""
    
    def __init__(self, name: str, 
                destinations: List[Dict[str, Any]],
                severity: Optional[List[ErrorSeverity]] = None,
                category: Optional[List[ErrorCategory]] = None,
                impact: Optional[List[ErrorImpact]] = None,
                error_types: Optional[List[str]] = None,
                cooldown_period: int = DEFAULT_COOLDOWN_PERIOD,
                format: ReportFormat = ReportFormat.TEXT,
                include_similar_errors: bool = True,
                include_suggestions: bool = True,
                include_stacktrace: bool = True,
                include_system_info: bool = True,
                custom_formatter: Optional[Callable[[ClassifiedError, Dict[str, Any]], str]] = None):
        """
        Initialize a reporting rule.
        
        Args:
            name: Rule name.
            destinations: List of destination configurations.
            severity: List of error severities to match or None for all.
            category: List of error categories to match or None for all.
            impact: List of error impacts to match or None for all.
            error_types: List of error types to match or None for all.
            cooldown_period: Cooldown period in seconds.
            format: Report format.
            include_similar_errors: Whether to include similar errors.
            include_suggestions: Whether to include fix suggestions.
            include_stacktrace: Whether to include stack trace.
            include_system_info: Whether to include system info.
            custom_formatter: Custom formatter function.
        """
        self.name = name
        self.destinations = destinations
        self.severity = severity
        self.category = category
        self.impact = impact
        self.error_types = error_types
        self.cooldown_period = cooldown_period
        self.format = format
        self.include_similar_errors = include_similar_errors
        self.include_suggestions = include_suggestions
        self.include_stacktrace = include_stacktrace
        self.include_system_info = include_system_info
        self.custom_formatter = custom_formatter
        self.last_reported: Dict[str, float] = {}  # error_type -> timestamp
    
    def matches(self, error: ClassifiedError) -> bool:
        """
        Check if error matches this rule.
        
        Args:
            error: Error to check.
            
        Returns:
            True if error matches, False otherwise.
        """
        # Check severity
        if self.severity is not None and error.severity not in self.severity:
            return False
        
        # Check category
        if self.category is not None and error.category not in self.category:
            return False
        
        # Check impact
        if self.impact is not None and error.impact not in self.impact:
            return False
        
        # Check error type
        if self.error_types is not None and error.details.error_type not in self.error_types:
            return False
        
        # Check cooldown
        error_type = error.details.error_type
        if error_type in self.last_reported:
            elapsed = time.time() - self.last_reported[error_type]
            if elapsed < self.cooldown_period:
                return False
        
        return True
    
    def update_last_reported(self, error: ClassifiedError) -> None:
        """
        Update last reported timestamp.
        
        Args:
            error: Reported error.
        """
        self.last_reported[error.details.error_type] = time.time()

@dataclass
class ErrorReport:
    """Error report data."""
    error: ClassifiedError
    rule: ReportingRule
    report_id: str = field(default_factory=lambda: f"report_{uuid.uuid4().hex}")
    timestamp: float = field(default_factory=time.time)
    formatted_report: Optional[str] = None
    destinations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "error_id": self.error.error_id,
            "rule_name": self.rule.name,
            "timestamp": self.timestamp,
            "destinations": self.destinations,
            "metadata": self.metadata,
            "sent": self.sent
        }

class ErrorReporter:
    """
    Error reporter for the FixWurx platform.
    
    This class manages error reporting rules, formats error reports,
    and dispatches them to configured destinations.
    """
    
    def __init__(self):
        """Initialize the error reporter."""
        self._rules: List[ReportingRule] = []
        self._report_queue = queue.Queue(maxsize=DEFAULT_MAX_QUEUE_SIZE)
        self._stop_dispatch = threading.Event()
        self._dispatch_thread = None
        self._formatters: Dict[ReportFormat, Callable[[ClassifiedError, Dict[str, Any]], str]] = {
            ReportFormat.TEXT: self._format_text,
            ReportFormat.HTML: self._format_html,
            ReportFormat.JSON: self._format_json,
            ReportFormat.MARKDOWN: self._format_markdown
        }
        self._dispatchers: Dict[ReportDestination, Callable[[ErrorReport], bool]] = {
            ReportDestination.CONSOLE: self._dispatch_console,
            ReportDestination.LOG_FILE: self._dispatch_log_file,
            ReportDestination.EMAIL: self._dispatch_email,
            ReportDestination.WEBHOOK: self._dispatch_webhook,
            ReportDestination.SLACK: self._dispatch_slack,
            ReportDestination.TEAMS: self._dispatch_teams,
            ReportDestination.CUSTOM: self._dispatch_custom
        }
        
        # Register event handler for errors
        try:
            register_event_handler(EventType.ERROR, self._handle_error_event)
        except Exception as e:
            logger.error(f"Failed to register error event handler: {e}")
        
        # Start dispatch thread
        self.start_dispatch_thread()
        
        logger.info("Error reporter initialized")
    
    def add_rule(self, rule: ReportingRule) -> None:
        """
        Add a reporting rule.
        
        Args:
            rule: Reporting rule.
        """
        self._rules.append(rule)
        logger.info(f"Added reporting rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a reporting rule.
        
        Args:
            rule_name: Rule name.
            
        Returns:
            True if rule was removed, False otherwise.
        """
        for i, rule in enumerate(self._rules):
            if rule.name == rule_name:
                del self._rules[i]
                logger.info(f"Removed reporting rule: {rule_name}")
                return True
        
        return False
    
    def get_rules(self) -> List[ReportingRule]:
        """
        Get all reporting rules.
        
        Returns:
            List of reporting rules.
        """
        return self._rules.copy()
    
    def report_error(self, error_id: str) -> bool:
        """
        Report an error.
        
        Args:
            error_id: Error ID.
            
        Returns:
            True if error was reported, False otherwise.
        """
        # Get error
        error = get_error(error_id)
        if error is None:
            logger.error(f"Error not found: {error_id}")
            return False
        
        # Check rules
        reported = False
        for rule in self._rules:
            if rule.matches(error):
                # Create report
                report = ErrorReport(
                    error=error,
                    rule=rule,
                    destinations=rule.destinations.copy()
                )
                
                # Format report
                report.formatted_report = self._format_report(report)
                
                # Queue report
                try:
                    self._report_queue.put(report, block=False)
                    reported = True
                    
                    # Update last reported
                    rule.update_last_reported(error)
                    
                    logger.info(f"Queued error report: {report.report_id} for error: {error_id}")
                except queue.Full:
                    logger.error("Report queue is full, dropping report")
        
        return reported
    
    def _format_report(self, report: ErrorReport) -> str:
        """
        Format an error report.
        
        Args:
            report: Error report.
            
        Returns:
            Formatted report.
        """
        rule = report.rule
        error = report.error
        
        # Get formatter
        formatter = None
        if rule.custom_formatter is not None:
            formatter = rule.custom_formatter
        elif rule.format in self._formatters:
            formatter = self._formatters[rule.format]
        else:
            formatter = self._formatters[ReportFormat.TEXT]
        
        # Prepare options
        options = {
            "include_similar_errors": rule.include_similar_errors,
            "include_suggestions": rule.include_suggestions,
            "include_stacktrace": rule.include_stacktrace,
            "include_system_info": rule.include_system_info
        }
        
        # Format report
        return formatter(error, options)
    
    def _format_text(self, error: ClassifiedError, options: Dict[str, Any]) -> str:
        """
        Format error as text.
        
        Args:
            error: Error to format.
            options: Formatting options.
            
        Returns:
            Formatted text.
        """
        lines = []
        
        # Basic info
        lines.append("========== ERROR REPORT ==========")
        lines.append(f"Error ID: {error.error_id}")
        lines.append(f"Timestamp: {datetime.datetime.fromtimestamp(error.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Type: {error.details.error_type}")
        lines.append(f"Category: {error.category.name}")
        lines.append(f"Severity: {error.severity.name}")
        lines.append(f"Impact: {error.impact.name}")
        lines.append(f"Message: {error.details.error_message}")
        
        # Stack trace
        if options.get("include_stacktrace", True) and error.details.traceback:
            lines.append("\n----- STACK TRACE -----")
            for i, frame in enumerate(error.details.traceback):
                lines.append(f"  {i+1}. {frame.filename}:{frame.lineno} in {frame.function}")
                if frame.code_context:
                    for line in frame.code_context:
                        lines.append(f"     {line.strip()}")
        
        # Similar errors
        if options.get("include_similar_errors", True):
            # Analyze error to get similar errors
            analysis = analyze_error(error.error_id)
            similar_errors = analysis.get("similar_errors", [])
            
            if similar_errors:
                lines.append("\n----- SIMILAR ERRORS -----")
                for i, error_id in enumerate(similar_errors[:DEFAULT_ERROR_HISTORY_COUNT]):
                    similar = get_error(error_id)
                    if similar:
                        lines.append(f"  {i+1}. {similar.details.error_type}: {similar.details.error_message}")
                        lines.append(f"     ID: {similar.error_id}")
                        lines.append(f"     Time: {datetime.datetime.fromtimestamp(similar.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                        lines.append(f"     Resolved: {similar.is_resolved}")
        
        # Fix suggestions
        if options.get("include_suggestions", True):
            # Analyze error to get fix suggestions
            analysis = analyze_error(error.error_id)
            suggestions = analysis.get("fix_suggestions", [])
            
            if suggestions:
                lines.append("\n----- FIX SUGGESTIONS -----")
                for i, suggestion in enumerate(suggestions):
                    lines.append(f"  {i+1}. {suggestion}")
        
        # System info
        if options.get("include_system_info", True):
            lines.append("\n----- SYSTEM INFO -----")
            lines.append(f"  Hostname: {socket.gethostname()}")
            lines.append(f"  Platform: {sys.platform}")
            lines.append(f"  Python: {sys.version}")
            
            # Add context info if available
            if error.details.context:
                if error.details.context.user:
                    lines.append(f"  User: {error.details.context.user}")
                if error.details.context.session_id:
                    lines.append(f"  Session: {error.details.context.session_id}")
                if error.details.context.command:
                    lines.append(f"  Command: {error.details.context.command}")
                if error.details.context.working_directory:
                    lines.append(f"  Working Dir: {error.details.context.working_directory}")
        
        lines.append("=================================")
        
        return "\n".join(lines)
    
    def _format_html(self, error: ClassifiedError, options: Dict[str, Any]) -> str:
        """
        Format error as HTML.
        
        Args:
            error: Error to format.
            options: Formatting options.
            
        Returns:
            Formatted HTML.
        """
        html = ['<!DOCTYPE html><html><head><style>',
                'body { font-family: Arial, sans-serif; margin: 20px; }',
                '.error-report { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }',
                '.error-header { background-color: #f8f8f8; padding: 10px; margin-bottom: 15px; }',
                '.error-section { margin-top: 20px; }',
                '.error-section h3 { color: #333; border-bottom: 1px solid #ddd; padding-bottom: 5px; }',
                '.stack-frame { margin-bottom: 10px; }',
                '.code-context { font-family: monospace; background-color: #f5f5f5; padding: 5px; margin-left: 20px; }',
                '.severity-ERROR { color: #e74c3c; }',
                '.severity-WARNING { color: #f39c12; }',
                '.severity-CRITICAL { color: #c0392b; font-weight: bold; }',
                '.severity-FATAL { color: #c0392b; font-weight: bold; text-decoration: underline; }',
                '</style></head><body>']
        
        # Basic info
        html.append('<div class="error-report">')
        html.append('<div class="error-header">')
        html.append(f'<h2>Error Report: <span class="severity-{error.severity.name}">{error.details.error_type}</span></h2>')
        html.append(f'<p><strong>ID:</strong> {error.error_id}</p>')
        html.append(f'<p><strong>Timestamp:</strong> {datetime.datetime.fromtimestamp(error.timestamp).strftime("%Y-%m-%d %H:%M:%S")}</p>')
        html.append(f'<p><strong>Category:</strong> {error.category.name}</p>')
        html.append(f'<p><strong>Severity:</strong> <span class="severity-{error.severity.name}">{error.severity.name}</span></p>')
        html.append(f'<p><strong>Impact:</strong> {error.impact.name}</p>')
        html.append(f'<p><strong>Message:</strong> {error.details.error_message}</p>')
        html.append('</div>')
        
        # Stack trace
        if options.get("include_stacktrace", True) and error.details.traceback:
            html.append('<div class="error-section">')
            html.append('<h3>Stack Trace</h3>')
            for i, frame in enumerate(error.details.traceback):
                html.append(f'<div class="stack-frame">')
                html.append(f'<p><strong>{i+1}.</strong> {frame.filename}:{frame.lineno} in <strong>{frame.function}</strong></p>')
                if frame.code_context:
                    html.append('<div class="code-context">')
                    for line in frame.code_context:
                        html.append(f'<pre>{line.strip()}</pre>')
                    html.append('</div>')
                html.append('</div>')
            html.append('</div>')
        
        # Similar errors
        if options.get("include_similar_errors", True):
            # Analyze error to get similar errors
            analysis = analyze_error(error.error_id)
            similar_errors = analysis.get("similar_errors", [])
            
            if similar_errors:
                html.append('<div class="error-section">')
                html.append('<h3>Similar Errors</h3>')
                html.append('<ul>')
                for i, error_id in enumerate(similar_errors[:DEFAULT_ERROR_HISTORY_COUNT]):
                    similar = get_error(error_id)
                    if similar:
                        html.append('<li>')
                        html.append(f'<p><strong>{similar.details.error_type}:</strong> {similar.details.error_message}</p>')
                        html.append(f'<p>ID: {similar.error_id}</p>')
                        html.append(f'<p>Time: {datetime.datetime.fromtimestamp(similar.timestamp).strftime("%Y-%m-%d %H:%M:%S")}</p>')
                        html.append(f'<p>Resolved: {similar.is_resolved}</p>')
                        html.append('</li>')
                html.append('</ul>')
                html.append('</div>')
        
        # Fix suggestions
        if options.get("include_suggestions", True):
            # Analyze error to get fix suggestions
            analysis = analyze_error(error.error_id)
            suggestions = analysis.get("fix_suggestions", [])
            
            if suggestions:
                html.append('<div class="error-section">')
                html.append('<h3>Fix Suggestions</h3>')
                html.append('<ol>')
                for suggestion in suggestions:
                    html.append(f'<li>{suggestion}</li>')
                html.append('</ol>')
                html.append('</div>')
        
        # System info
        if options.get("include_system_info", True):
            html.append('<div class="error-section">')
            html.append('<h3>System Info</h3>')
            html.append('<ul>')
            html.append(f'<li><strong>Hostname:</strong> {socket.gethostname()}</li>')
            html.append(f'<li><strong>Platform:</strong> {sys.platform}</li>')
            html.append(f'<li><strong>Python:</strong> {sys.version}</li>')
            
            # Add context info if available
            if error.details.context:
                if error.details.context.user:
                    html.append(f'<li><strong>User:</strong> {error.details.context.user}</li>')
                if error.details.context.session_id:
                    html.append(f'<li><strong>Session:</strong> {error.details.context.session_id}</li>')
                if error.details.context.command:
                    html.append(f'<li><strong>Command:</strong> {error.details.context.command}</li>')
                if error.details.context.working_directory:
                    html.append(f'<li><strong>Working Dir:</strong> {error.details.context.working_directory}</li>')
            
            html.append('</ul>')
            html.append('</div>')
        
        html.append('</div>')
        html.append('</body></html>')
        
        return ''.join(html)
    
    def _format_json(self, error: ClassifiedError, options: Dict[str, Any]) -> str:
        """
        Format error as JSON.
        
        Args:
            error: Error to format.
            options: Formatting options.
            
        Returns:
            Formatted JSON.
        """
        # Create report data
        report_data = {
            "error_id": error.error_id,
            "timestamp": error.timestamp,
            "error_type": error.details.error_type,
            "error_message": error.details.error_message,
            "category": error.category.name,
            "severity": error.severity.name,
            "impact": error.impact.name
        }
        
        # Add stack trace
        if options.get("include_stacktrace", True) and error.details.traceback:
            report_data["stack_trace"] = [
                {
                    "filename": frame.filename,
                    "lineno": frame.lineno,
                    "function": frame.function,
                    "code_context": frame.code_context
                }
                for frame in error.details.traceback
            ]
        
        # Add similar errors
        if options.get("include_similar_errors", True):
            # Analyze error to get similar errors
            analysis = analyze_error(error.error_id)
            similar_errors = analysis.get("similar_errors", [])
            
            if similar_errors:
                report_data["similar_errors"] = []
                for error_id in similar_errors[:DEFAULT_ERROR_HISTORY_COUNT]:
                    similar = get_error(error_id)
                    if similar:
                        report_data["similar_errors"].append({
                            "error_id": similar.error_id,
                            "timestamp": similar.timestamp,
                            "error_type": similar.details.error_type,
                            "error_message": similar.details.error_message,
                            "is_resolved": similar.is_resolved
                        })
        
        # Add fix suggestions
        if options.get("include_suggestions", True):
            # Analyze error to get fix suggestions
            analysis = analyze_error(error.error_id)
            suggestions = analysis.get("fix_suggestions", [])
            
            if suggestions:
                report_data["fix_suggestions"] = suggestions
        
        # Add system info
        if options.get("include_system_info", True):
            report_data["system_info"] = {
                "hostname": socket.gethostname(),
                "platform": sys.platform,
                "python": sys.version
            }
            
            # Add context info if available
            if error.details.context:
                context_data = {}
                if error.details.context.user:
                    context_data["user"] = error.details.context.user
                if error.details.context.session_id:
                    context_data["session_id"] = error.details.context.session_id
                if error.details.context.command:
                    context_data["command"] = error.details.context.command
                if error.details.context.working_directory:
                    context_data["working_directory"] = error.details.context.working_directory
                
                if context_data:
                    report_data["context"] = context_data
        
        # Convert to JSON
        return json.dumps(report_data, indent=2)
    
    def _format_markdown(self, error: ClassifiedError, options: Dict[str, Any]) -> str:
        """
        Format error as Markdown.
        
        Args:
            error: Error to format.
            options: Formatting options.
            
        Returns:
            Formatted Markdown.
        """
        lines = []
        
        # Basic info
        lines.append("# Error Report")
        lines.append(f"**Error ID:** {error.error_id}  ")
        lines.append(f"**Timestamp:** {datetime.datetime.fromtimestamp(error.timestamp).strftime('%Y-%m-%d %H:%M:%S')}  ")
        lines.append(f"**Type:** {error.details.error_type}  ")
        lines.append(f"**Category:** {error.category.name}  ")
        lines.append(f"**Severity:** {error.severity.name}  ")
        lines.append(f"**Impact:** {error.impact.name}  ")
        lines.append(f"**Message:** {error.details.error_message}  ")
        
        # Stack trace
        if options.get("include_stacktrace", True) and error.details.traceback:
            lines.append("\n## Stack Trace\n")
            for i, frame in enumerate(error.details.traceback):
                lines.append(f"{i+1}. **{frame.filename}:{frame.lineno}** in `{frame.function}`")
                if frame.code_context:
                    lines.append("   ```python")
                    for line in frame.code_context:
                        lines.append(f"   {line.strip()}")
                    lines.append("   ```")
        
        # Similar errors
        if options.get("include_similar_errors", True):
            # Analyze error to get similar errors
            analysis = analyze_error(error.error_id)
            similar_errors = analysis.get("similar_errors", [])
            
            if similar_errors:
                lines.append("\n## Similar Errors\n")
                for i, error_id in enumerate(similar_errors[:DEFAULT_ERROR_HISTORY_COUNT]):
                    similar = get_error(error_id)
                    if similar:
                        lines.append(f"{i+1}. **{similar.details.error_type}:** {similar.details.error_message}  ")
                        lines.append(f"   - ID: `{similar.error_id}`  ")
                        lines.append(f"   - Time: {datetime.datetime.fromtimestamp(similar.timestamp).strftime('%Y-%m-%d %H:%M:%S')}  ")
                        lines.append(f"   - Resolved: {similar.is_resolved}  ")
        
        # Fix suggestions
        if options.get("include_suggestions", True):
            # Analyze error to get fix suggestions
            analysis = analyze_error(error.error_id)
            suggestions = analysis.get("fix_suggestions", [])
            
            if suggestions:
                lines.append("\n## Fix Suggestions\n")
                for i, suggestion in enumerate(suggestions):
                    lines.append(f"{i+1}. {suggestion}")
        
        # System info
        if options.get("include_system_info", True):
            lines.append("\n## System Info\n")
            lines.append(f"- **Hostname:** {socket.gethostname()}  ")
            lines.append(f"- **Platform:** {sys.platform}  ")
            lines.append(f"- **Python:** {sys.version}  ")
            
            # Add context info if available
            if error.details.context:
                if error.details.context.user:
                    lines.append(f"- **User:** {error.details.context.user}  ")
                if error.details.context.session_id:
                    lines.append(f"- **Session:** {error.details.context.session_id}  ")
                if error.details.context.command:
                    lines.append(f"- **Command:** {error.details.context.command}  ")
                if error.details.context.working_directory:
                    lines.append(f"- **Working Dir:** {error.details.context.working_directory}  ")
        
        return "\n".join(lines)
    
    def start_dispatch_thread(self) -> None:
        """Start the dispatch thread."""
        if self._dispatch_thread is None or not self._dispatch_thread.is_alive():
            self._stop_dispatch.clear()
            self._dispatch_thread = threading.Thread(
                target=self._dispatch_loop,
                daemon=True,
                name="ErrorReportDispatchThread"
            )
            self._dispatch_thread.start()
            logger.info("Started error report dispatch thread")
    
    def stop_dispatch_thread(self) -> None:
        """Stop the dispatch thread."""
        if self._dispatch_thread and self._dispatch_thread.is_alive():
            self._stop_dispatch.set()
            self._dispatch_thread.join(timeout=5.0)
            logger.info("Stopped error report dispatch thread")
    
    def _dispatch_loop(self) -> None:
        """Dispatch loop for error reports."""
        while not self._stop_dispatch.is_set():
            try:
                # Get report from queue with timeout
                try:
                    report = self._report_queue.get(timeout=DEFAULT_DISPATCH_INTERVAL)
                    self._dispatch_report(report)
                    self._report_queue.task_done()
                except queue.Empty:
                    pass
            except Exception as e:
                logger.error(f"Error in dispatch loop: {e}")
            
            # Sleep for a short time
            time.sleep(0.1)
    
    def _dispatch_report(self, report: ErrorReport) -> None:
        """
        Dispatch error report to destinations.
        
        Args:
            report: Error report.
        """
        success = False
        
        # Dispatch to each destination
        for dest_config in report.destinations:
            try:
                # Get destination type
                dest_type = ReportDestination[dest_config["type"]]
                
                # Get dispatcher
                if dest_type in self._dispatchers:
                    dispatcher = self._dispatchers[dest_type]
                    
                    # Dispatch report
                    result = dispatcher(report, dest_config)
                    success = success or result
                else:
                    logger.error(f"Unknown destination type: {dest_type}")
            except Exception as e:
                logger.error(f"Error dispatching report to {dest_config.get('type', 'unknown')}: {e}")
        
        # Update report status
        report.sent = success
        
        # Log result
        if success:
            logger.info(f"Successfully dispatched report: {report.report_id}")
        else:
            logger.error(f"Failed to dispatch report: {report.report_id}")
    
    def _dispatch_console(self, report: ErrorReport, config: Dict[str, Any]) -> bool:
        """
        Dispatch report to console.
        
        Args:
            report: Error report.
            config: Destination configuration.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get colored output flag
            colored = config.get("colored", True)
            
            # Get output stream
            stream_name = config.get("stream", "stderr")
            stream = sys.stderr if stream_name == "stderr" else sys.stdout
            
            # Print report
            if colored:
                # Add color codes based on severity
                severity = report.error.severity.name
                color_code = {
                    "DEBUG": "\033[37m",  # White
                    "INFO": "\033[34m",   # Blue
                    "WARNING": "\033[33m", # Yellow
                    "ERROR": "\033[31m",   # Red
                    "CRITICAL": "\033[1;31m", # Bold red
                    "FATAL": "\033[1;37;41m"  # White on red
                }.get(severity, "\033[0m")
                
                reset_code = "\033[0m"
                
                # Add color to report
                report_text = report.formatted_report
                report_text = re.sub(
                    r"(Error ID:.*)",
                    f"{color_code}\\1{reset_code}",
                    report_text
                )
                report_text = re.sub(
                    r"(Severity:.*)",
                    f"{color_code}\\1{reset_code}",
                    report_text
                )
                
                print(report_text, file=stream)
            else:
                print(report.formatted_report, file=stream)
            
            return True
        except Exception as e:
            logger.error(f"Error dispatching to console: {e}")
            return False
    
    def _dispatch_log_file(self, report: ErrorReport, config: Dict[str, Any]) -> bool:
        """
        Dispatch report to log file.
        
        Args:
            report: Error report.
            config: Destination configuration.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get log file path
            log_path = config.get("path", DEFAULT_LOG_PATH)
            log_path = os.path.expanduser(log_path)
            
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Generate filename
            timestamp = datetime.datetime.fromtimestamp(report.timestamp).strftime("%Y%m%d_%H%M%S")
            error_type = report.error.details.error_type
            filename = f"{timestamp}_{error_type}_{report.report_id}.log"
            
            # Append to log file
            full_path = os.path.join(log_path, filename)
            with open(full_path, "w") as f:
                f.write(report.formatted_report)
            
            logger.info(f"Wrote error report to: {full_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error dispatching to log file: {e}")
            return False
    
    def _dispatch_email(self, report: ErrorReport, config: Dict[str, Any]) -> bool:
        """
        Dispatch report to email.
        
        Args:
            report: Error report.
            config: Destination configuration.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get email settings
            recipients = config.get("recipients", [])
            if not recipients:
                logger.error("No email recipients specified")
                return False
            
            sender = config.get("sender", DEFAULT_EMAIL_SENDER)
            subject_prefix = config.get("subject_prefix", DEFAULT_EMAIL_SUBJECT_PREFIX)
            smtp_host = config.get("smtp_host", "localhost")
            smtp_port = config.get("smtp_port", 25)
            smtp_user = config.get("smtp_user")
            smtp_password = config.get("smtp_password")
            use_tls = config.get("use_tls", False)
            use_html = config.get("use_html", False)
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = ", ".join(recipients)
            
            # Create subject
            error_type = report.error.details.error_type
            error_message = report.error.details.error_message
            subject = f"{subject_prefix} {error_type}: {error_message[:50]}"
            if len(error_message) > 50:
                subject += "..."
            
            msg["Subject"] = subject
            
            # Attach report
            if use_html and report.rule.format == ReportFormat.HTML:
                # Use HTML report as is
                msg.attach(MIMEText(report.formatted_report, "html"))
            elif use_html:
                # Convert to HTML
                formatter = self._formatters[ReportFormat.HTML]
                options = {
                    "include_similar_errors": report.rule.include_similar_errors,
                    "include_suggestions": report.rule.include_suggestions,
                    "include_stacktrace": report.rule.include_stacktrace,
                    "include_system_info": report.rule.include_system_info
                }
                html_report = formatter(report.error, options)
                msg.attach(MIMEText(html_report, "html"))
            else:
                # Use text report
                msg.attach(MIMEText(report.formatted_report, "plain"))
            
            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if use_tls:
                    server.starttls()
                
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                
                server.send_message(msg)
            
            logger.info(f"Sent email report to: {', '.join(recipients)}")
            
            return True
        except Exception as e:
            logger.error(f"Error dispatching to email: {e}")
            return False
    
    def _dispatch_webhook(self, report: ErrorReport, config: Dict[str, Any]) -> bool:
        """
        Dispatch report to webhook.
        
        Args:
            report: Error report.
            config: Destination configuration.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get webhook settings
            url = config.get("url")
            if not url:
                logger.error("No webhook URL specified")
                return False
            
            method = config.get("method", "POST")
            headers = config.get("headers", {})
            timeout = config.get("timeout", DEFAULT_WEBHOOK_TIMEOUT)
            
            # Set default content type
            if "Content-Type" not in headers:
                headers["Content-Type"] = "application/json"
            
            # Prepare payload
            if report.rule.format == ReportFormat.JSON:
                # Use JSON report as is
                payload = report.formatted_report
            else:
                # Convert error to JSON
                formatter = self._formatters[ReportFormat.JSON]
                options = {
                    "include_similar_errors": report.rule.include_similar_errors,
                    "include_suggestions": report.rule.include_suggestions,
                    "include_stacktrace": report.rule.include_stacktrace,
                    "include_system_info": report.rule.include_system_info
                }
                payload = formatter(report.error, options)
            
            # Send request
            req = urllib.request.Request(
                url,
                data=payload.encode("utf-8"),
                headers=headers,
                method=method
            )
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_code = response.getcode()
                logger.info(f"Webhook response: {response_code}")
                
                return 200 <= response_code < 300
        except Exception as e:
            logger.error(f"Error dispatching to webhook: {e}")
            return False
    
    def _dispatch_slack(self, report: ErrorReport, config: Dict[str, Any]) -> bool:
        """
        Dispatch report to Slack.
        
        Args:
            report: Error report.
            config: Destination configuration.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get Slack settings
            webhook_url = config.get("webhook_url")
            if not webhook_url:
                logger.error("No Slack webhook URL specified")
                return False
            
            channel = config.get("channel", DEFAULT_SLACK_CHANNEL)
            username = config.get("username", "FixWurx Error Reporter")
            icon_emoji = config.get("icon_emoji", ":warning:")
            timeout = config.get("timeout", DEFAULT_WEBHOOK_TIMEOUT)
            
            # Create Slack message
            error = report.error
            
            # Build message blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Error Report: {error.details.error_type}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*ID:* {error.error_id}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:* {datetime.datetime.fromtimestamp(error.timestamp).strftime('%Y-%m-%d %H:%M:%S')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Category:* {error.category.name}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Severity:* {error.severity.name}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Impact:* {error.impact.name}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Message:* {error.details.error_message}"
                    }
                }
            ]
            
            # Add stack trace
            if report.rule.include_stacktrace and error.details.traceback:
                trace_text = "*Stack Trace:*\n```"
                for i, frame in enumerate(error.details.traceback[:5]):  # First 5 frames
                    trace_text += f"{frame.filename}:{frame.lineno} in {frame.function}\n"
                trace_text += "```"
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": trace_text
                    }
                })
            
            # Add system info
            if report.rule.include_system_info:
                system_text = "*System Info:*\n"
                system_text += f"• Hostname: {socket.gethostname()}\n"
                system_text += f"• Platform: {sys.platform}\n"
                
                # Add context info if available
                if error.details.context:
                    if error.details.context.user:
                        system_text += f"• User: {error.details.context.user}\n"
                    if error.details.context.command:
                        system_text += f"• Command: {error.details.context.command}\n"
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": system_text
                    }
                })
            
            # Create payload
            payload = {
                "channel": channel,
                "username": username,
                "icon_emoji": icon_emoji,
                "blocks": blocks
            }
            
            # Send request
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_code = response.getcode()
                logger.info(f"Slack webhook response: {response_code}")
                
                return 200 <= response_code < 300
        except Exception as e:
            logger.error(f"Error dispatching to Slack: {e}")
            return False
    
    def _dispatch_teams(self, report: ErrorReport, config: Dict[str, Any]) -> bool:
        """
        Dispatch report to Microsoft Teams.
        
        Args:
            report: Error report.
            config: Destination configuration.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get Teams settings
            webhook_url = config.get("webhook_url")
            if not webhook_url:
                logger.error("No Teams webhook URL specified")
                return False
            
            timeout = config.get("timeout", DEFAULT_WEBHOOK_TIMEOUT)
            
            # Create Teams message
            error = report.error
            
            # Map severity to color
            color = {
                ErrorSeverity.DEBUG: "007bff",    # Blue
                ErrorSeverity.INFO: "17a2b8",     # Info blue
                ErrorSeverity.WARNING: "ffc107",  # Yellow
                ErrorSeverity.ERROR: "dc3545",    # Red
                ErrorSeverity.CRITICAL: "6610f2", # Purple
                ErrorSeverity.FATAL: "343a40"     # Dark gray
            }.get(error.severity, "6c757d")
            
            # Create facts
            facts = [
                {
                    "name": "Error ID",
                    "value": error.error_id
                },
                {
                    "name": "Timestamp",
                    "value": datetime.datetime.fromtimestamp(error.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                },
                {
                    "name": "Category",
                    "value": error.category.name
                },
                {
                    "name": "Severity",
                    "value": error.severity.name
                },
                {
                    "name": "Impact",
                    "value": error.impact.name
                }
            ]
            
            # Add system info
            if report.rule.include_system_info:
                facts.append({
                    "name": "Hostname",
                    "value": socket.gethostname()
                })
                facts.append({
                    "name": "Platform",
                    "value": sys.platform
                })
                
                # Add context info if available
                if error.details.context:
                    if error.details.context.user:
                        facts.append({
                            "name": "User",
                            "value": error.details.context.user
                        })
                    if error.details.context.command:
                        facts.append({
                            "name": "Command",
                            "value": error.details.context.command
                        })
            
            # Create sections
            sections = [
                {
                    "activityTitle": f"Error Report: {error.details.error_type}",
                    "activitySubtitle": error.details.error_message,
                    "facts": facts,
                    "markdown": True
                }
            ]
            
            # Add stack trace
            if report.rule.include_stacktrace and error.details.traceback:
                trace_text = "**Stack Trace:**\n\n"
                for i, frame in enumerate(error.details.traceback[:5]):  # First 5 frames
                    trace_text += f"{i+1}. {frame.filename}:{frame.lineno} in {frame.function}\n"
                
                sections.append({
                    "title": "Stack Trace",
                    "text": trace_text
                })
            
            # Create payload
            payload = {
                "type": "MessageCard",
                "context": "http://schema.org/extensions",
                "themeColor": color,
                "title": f"Error: {error.details.error_type}",
                "text": error.details.error_message,
                "sections": sections
            }
            
            # Send request
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_code = response.getcode()
                logger.info(f"Teams webhook response: {response_code}")
                
                return 200 <= response_code < 300
        except Exception as e:
            logger.error(f"Error dispatching to Teams: {e}")
            return False
    
    def _dispatch_custom(self, report: ErrorReport, config: Dict[str, Any]) -> bool:
        """
        Dispatch report to custom destination.
        
        Args:
            report: Error report.
            config: Destination configuration.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get custom handler
            handler = config.get("handler")
            if not handler:
                logger.error("No custom handler specified")
                return False
            
            # Check if handler is a callable
            if not callable(handler):
                logger.error("Custom handler is not callable")
                return False
            
            # Call handler
            result = handler(report.error, report.formatted_report, config)
            
            return bool(result)
        except Exception as e:
            logger.error(f"Error dispatching to custom destination: {e}")
            return False
    
    def _handle_error_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle error event.
        
        Args:
            event_data: Event data.
        """
        # Get error ID
        error_id = event_data.get("error_id")
        if not error_id:
            logger.error("Error event missing error_id")
            return
        
        # Report error
        self.report_error(error_id)


# Create singleton error reporter
_error_reporter = None

def get_error_reporter() -> ErrorReporter:
    """
    Get the singleton error reporter instance.
    
    Returns:
        Error reporter instance.
    """
    global _error_reporter
    
    if _error_reporter is None:
        _error_reporter = ErrorReporter()
    
    return _error_reporter

def add_reporting_rule(rule: ReportingRule) -> None:
    """
    Add a reporting rule.
    
    Args:
        rule: Reporting rule.
    """
    get_error_reporter().add_rule(rule)

def remove_reporting_rule(rule_name: str) -> bool:
    """
    Remove a reporting rule.
    
    Args:
        rule_name: Rule name.
        
    Returns:
        True if rule was removed, False otherwise.
    """
    return get_error_reporter().remove_rule(rule_name)

def get_reporting_rules() -> List[ReportingRule]:
    """
    Get all reporting rules.
    
    Returns:
        List of reporting rules.
    """
    return get_error_reporter().get_rules()

def report_error(error_id: str) -> bool:
    """
    Report an error.
    
    Args:
        error_id: Error ID.
        
    Returns:
        True if error was reported, False otherwise.
    """
    return get_error_reporter().report_error(error_id)

def create_console_rule(name: str,
                      severity: Optional[List[ErrorSeverity]] = None,
                      colored: bool = True,
                      stream: str = "stderr",
                      **kwargs) -> ReportingRule:
    """
    Create a console reporting rule.
    
    Args:
        name: Rule name.
        severity: List of error severities to match or None for all.
        colored: Whether to use color in output.
        stream: Output stream ("stderr" or "stdout").
        **kwargs: Additional rule options.
        
    Returns:
        Reporting rule.
    """
    destinations = [{
        "type": "CONSOLE",
        "colored": colored,
        "stream": stream
    }]
    
    return ReportingRule(name=name, destinations=destinations, severity=severity, **kwargs)

def create_log_file_rule(name: str,
                       path: Optional[str] = None,
                       severity: Optional[List[ErrorSeverity]] = None,
                       **kwargs) -> ReportingRule:
    """
    Create a log file reporting rule.
    
    Args:
        name: Rule name.
        path: Log file path or None for default.
        severity: List of error severities to match or None for all.
        **kwargs: Additional rule options.
        
    Returns:
        Reporting rule.
    """
    destinations = [{
        "type": "LOG_FILE",
        "path": path or DEFAULT_LOG_PATH
    }]
    
    return ReportingRule(name=name, destinations=destinations, severity=severity, **kwargs)

def create_email_rule(name: str,
                    recipients: List[str],
                    sender: Optional[str] = None,
                    subject_prefix: Optional[str] = None,
                    smtp_host: str = "localhost",
                    smtp_port: int = 25,
                    smtp_user: Optional[str] = None,
                    smtp_password: Optional[str] = None,
                    use_tls: bool = False,
                    use_html: bool = False,
                    severity: Optional[List[ErrorSeverity]] = None,
                    **kwargs) -> ReportingRule:
    """
    Create an email reporting rule.
    
    Args:
        name: Rule name.
        recipients: List of email recipients.
        sender: Email sender or None for default.
        subject_prefix: Subject prefix or None for default.
        smtp_host: SMTP host.
        smtp_port: SMTP port.
        smtp_user: SMTP username or None.
        smtp_password: SMTP password or None.
        use_tls: Whether to use TLS.
        use_html: Whether to use HTML format.
        severity: List of error severities to match or None for all.
        **kwargs: Additional rule options.
        
    Returns:
        Reporting rule.
    """
    destinations = [{
        "type": "EMAIL",
        "recipients": recipients,
        "sender": sender or DEFAULT_EMAIL_SENDER,
        "subject_prefix": subject_prefix or DEFAULT_EMAIL_SUBJECT_PREFIX,
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "smtp_user": smtp_user,
        "smtp_password": smtp_password,
        "use_tls": use_tls,
        "use_html": use_html
    }]
    
    # Use HTML format if requested
    format = ReportFormat.HTML if use_html else ReportFormat.TEXT
    
    return ReportingRule(name=name, destinations=destinations, severity=severity, format=format, **kwargs)

def create_slack_rule(name: str,
                    webhook_url: str,
                    channel: Optional[str] = None,
                    username: Optional[str] = None,
                    icon_emoji: Optional[str] = None,
                    severity: Optional[List[ErrorSeverity]] = None,
                    **kwargs) -> ReportingRule:
    """
    Create a Slack reporting rule.
    
    Args:
        name: Rule name.
        webhook_url: Slack webhook URL.
        channel: Slack channel or None for default.
        username: Slack username or None for default.
        icon_emoji: Slack icon emoji or None for default.
        severity: List of error severities to match or None for all.
        **kwargs: Additional rule options.
        
    Returns:
        Reporting rule.
    """
    destinations = [{
        "type": "SLACK",
        "webhook_url": webhook_url,
        "channel": channel or DEFAULT_SLACK_CHANNEL,
        "username": username or "FixWurx Error Reporter",
        "icon_emoji": icon_emoji or ":warning:"
    }]
    
    return ReportingRule(name=name, destinations=destinations, severity=severity, **kwargs)

def create_teams_rule(name: str,
                    webhook_url: str,
                    severity: Optional[List[ErrorSeverity]] = None,
                    **kwargs) -> ReportingRule:
    """
    Create a Microsoft Teams reporting rule.
    
    Args:
        name: Rule name.
        webhook_url: Teams webhook URL.
        severity: List of error severities to match or None for all.
        **kwargs: Additional rule options.
        
    Returns:
        Reporting rule.
    """
    destinations = [{
        "type": "TEAMS",
        "webhook_url": webhook_url
    }]
    
    return ReportingRule(name=name, destinations=destinations, severity=severity, **kwargs)

def create_webhook_rule(name: str,
                      url: str,
                      method: str = "POST",
                      headers: Optional[Dict[str, str]] = None,
                      timeout: int = DEFAULT_WEBHOOK_TIMEOUT,
                      severity: Optional[List[ErrorSeverity]] = None,
                      **kwargs) -> ReportingRule:
    """
    Create a webhook reporting rule.
    
    Args:
        name: Rule name.
        url: Webhook URL.
        method: HTTP method.
        headers: HTTP headers or None for default.
        timeout: Timeout in seconds.
        severity: List of error severities to match or None for all.
        **kwargs: Additional rule options.
        
    Returns:
        Reporting rule.
    """
    destinations = [{
        "type": "WEBHOOK",
        "url": url,
        "method": method,
        "headers": headers or {},
        "timeout": timeout
    }]
    
    # Use JSON format for webhooks
    format = ReportFormat.JSON
    
    return ReportingRule(name=name, destinations=destinations, severity=severity, format=format, **kwargs)

def create_multi_destination_rule(name: str,
                               destinations: List[Dict[str, Any]],
                               severity: Optional[List[ErrorSeverity]] = None,
                               **kwargs) -> ReportingRule:
    """
    Create a multi-destination reporting rule.
    
    Args:
        name: Rule name.
        destinations: List of destination configurations.
        severity: List of error severities to match or None for all.
        **kwargs: Additional rule options.
        
    Returns:
        Reporting rule.
    """
    return ReportingRule(name=name, destinations=destinations, severity=severity, **kwargs)

# Initialize error reporter with default rules
if not any(arg.endswith('test.py') for arg in sys.argv):
    reporter = get_error_reporter()
    
    # Add default console rule for ERROR and above
    console_rule = create_console_rule(
        name="default_console",
        severity=[ErrorSeverity.ERROR, ErrorSeverity.CRITICAL, ErrorSeverity.FATAL],
        format=ReportFormat.TEXT,
        include_similar_errors=True,
        include_suggestions=True,
        include_stacktrace=True,
        include_system_info=True
    )
    reporter.add_rule(console_rule)
    
    # Add default log file rule for all errors
    log_rule = create_log_file_rule(
        name="default_log_file",
        format=ReportFormat.TEXT,
        include_similar_errors=True,
        include_suggestions=True,
        include_stacktrace=True,
        include_system_info=True
    )
    reporter.add_rule(log_rule)
