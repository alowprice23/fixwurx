"""
log_retention.py
───────────────
Log retention policy system for FixWurx.

Provides:
- Automatic log rotation
- Policy-based retention (time, size, severity)
- Log archiving and compression
- Log cleanup scheduling

This system ensures that logs are properly managed throughout their lifecycle,
from creation to archival or deletion, based on configurable policies.
"""

import os
import re
import gzip
import time
import json
import shutil
import logging
import datetime
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable

# Import modules if available
try:
    from access_control import log_action
    AUDIT_LOGGING_AVAILABLE = True
except ImportError:
    AUDIT_LOGGING_AVAILABLE = False
    # Simple logging fallback
    def log_action(username, action, target=None, details=None):
        """Fallback logging function if access_control module is not available."""
        logging.info(f"ACTION: {username} - {action}" + 
                    (f" - Target: {target}" if target else "") +
                    (f" - Details: {details}" if details else ""))

try:
    from config_manager import ConfigManager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(".triangulum/log_retention.log"),
        logging.StreamHandler()
    ]
)
retention_logger = logging.getLogger("log_retention")

# Default paths
TRIANGULUM_DIR = Path(".triangulum")
LOG_DIR = TRIANGULUM_DIR / "logs"
ARCHIVE_DIR = LOG_DIR / "archive"

# Default retention policy
DEFAULT_RETENTION_POLICY = {
    "max_size_mb": 100,            # Maximum size of log directory in MB
    "max_age_days": 30,            # Maximum age of log files in days
    "rotation_size_mb": 10,        # Size at which to rotate a log file in MB
    "archive_older_than_days": 7,  # Archive logs older than this many days
    "delete_archives_older_than_days": 90,  # Delete archives older than this many days
    "min_free_space_percent": 10,  # Minimum free disk space to maintain
    "severity_retention": {        # Retention periods by severity level
        "CRITICAL": 365,           # Keep critical logs for 1 year
        "ERROR": 180,              # Keep error logs for 6 months
        "WARNING": 90,             # Keep warning logs for 3 months
        "INFO": 30,                # Keep info logs for 1 month
        "DEBUG": 7                 # Keep debug logs for 1 week
    },
    "excluded_logs": [             # Logs that should not be deleted
        "audit.log",
        "security.log"
    ],
    "protected_logs": [            # Logs that should not be rotated
        "critical_errors.log"
    ],
    "compression_enabled": True,   # Whether to compress archived logs
    "compression_level": 6         # Compression level (1-9, 9 being highest)
}

# Log severity mappings
LOG_SEVERITY_LEVELS = {
    "CRITICAL": 50,
    "ERROR": 40,
    "WARNING": 30,
    "INFO": 20,
    "DEBUG": 10,
    "NOTSET": 0
}

# File patterns
LOG_FILE_PATTERN = r".*\.log$"
ROTATED_LOG_PATTERN = r".*\.log\.\d{8}$"
ARCHIVED_LOG_PATTERN = r".*\.log\.\d{8}\.gz$"


class LogRetentionError(Exception):
    """Exception raised for log retention errors."""
    pass


class LogRetentionPolicy:
    """
    Represents a log retention policy.
    
    Attributes:
        policy: Dictionary containing the policy settings
    """
    
    def __init__(self, policy: Dict[str, Any] = None):
        """
        Initialize a log retention policy.
        
        Args:
            policy: Dictionary containing the policy settings, or None to use defaults
        """
        self.policy = policy or DEFAULT_RETENTION_POLICY.copy()
    
    def should_rotate(self, log_path: Path) -> bool:
        """
        Check if a log file should be rotated.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            True if the log file should be rotated, False otherwise
        """
        # Check if the log is protected
        if log_path.name in self.policy.get("protected_logs", []):
            return False
        
        # Check if the log exists
        if not log_path.exists():
            return False
        
        # Check if the log has reached the rotation size
        rotation_size_bytes = self.policy.get("rotation_size_mb", 10) * 1024 * 1024
        return log_path.stat().st_size >= rotation_size_bytes
    
    def should_archive(self, log_path: Path) -> bool:
        """
        Check if a log file should be archived.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            True if the log file should be archived, False otherwise
        """
        # Check if the log exists
        if not log_path.exists():
            return False
        
        # Check if the log is a rotated log file
        if not re.match(ROTATED_LOG_PATTERN, log_path.name):
            return False
        
        # Get the log's modification time
        mtime = log_path.stat().st_mtime
        age_days = (time.time() - mtime) / (24 * 3600)
        
        # Check if the log is old enough to archive
        return age_days >= self.policy.get("archive_older_than_days", 7)
    
    def should_delete(self, log_path: Path) -> bool:
        """
        Check if a log file should be deleted.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            True if the log file should be deleted, False otherwise
        """
        # Check if the log exists
        if not log_path.exists():
            return False
        
        # Check if the log is excluded
        if log_path.name in self.policy.get("excluded_logs", []):
            return False
        
        # Get the log's modification time
        mtime = log_path.stat().st_mtime
        age_days = (time.time() - mtime) / (24 * 3600)
        
        # Check if it's an archived log
        if re.match(ARCHIVED_LOG_PATTERN, log_path.name):
            return age_days >= self.policy.get("delete_archives_older_than_days", 90)
        
        # If it's a regular log, check by severity
        severity = self._get_log_severity(log_path)
        severity_max_age = self.policy.get("severity_retention", {}).get(severity, 30)
        
        return age_days >= severity_max_age
    
    def _get_log_severity(self, log_path: Path) -> str:
        """
        Get the severity level of a log file based on its name and contents.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            Severity level (CRITICAL, ERROR, WARNING, INFO, DEBUG, or NOTSET)
        """
        # Try to determine severity from the filename
        name = log_path.name.lower()
        if "critical" in name:
            return "CRITICAL"
        elif "error" in name:
            return "ERROR"
        elif "warning" in name or "warn" in name:
            return "WARNING"
        elif "info" in name:
            return "INFO"
        elif "debug" in name:
            return "DEBUG"
        
        # If we can't determine from the name, sample the file
        try:
            # Sample the beginning of the file
            sample_size = 10 * 1024  # 10 KB sample
            with open(log_path, 'r', errors='ignore') as f:
                sample = f.read(sample_size)
            
            # Count occurrences of each severity level
            counts = {
                "CRITICAL": sample.count("[CRITICAL]"),
                "ERROR": sample.count("[ERROR]"),
                "WARNING": sample.count("[WARNING]"),
                "INFO": sample.count("[INFO]"),
                "DEBUG": sample.count("[DEBUG]")
            }
            
            # Find the highest severity that appears at least once
            for severity in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]:
                if counts[severity] > 0:
                    return severity
            
            # If no severity markers found, default to INFO
            return "INFO"
        except Exception:
            # If we encounter any issues, default to INFO
            return "INFO"


class LogRetentionManager:
    """
    Manages log retention according to policies.
    
    Features:
    - Log rotation
    - Log archiving
    - Log deletion
    - Policy enforcement
    """
    
    def __init__(
        self,
        log_dir: Path = LOG_DIR,
        archive_dir: Path = ARCHIVE_DIR,
        policy: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the log retention manager.
        
        Args:
            log_dir: Directory containing log files
            archive_dir: Directory for archived log files
            policy: Retention policy dictionary, or None to use default
        """
        self.log_dir = Path(log_dir)
        self.archive_dir = Path(archive_dir)
        
        # Create directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize policy
        if CONFIG_MANAGER_AVAILABLE and policy is None:
            # Try to load policy from config
            try:
                config_manager = ConfigManager()
                log_config = config_manager.get_section("logging")
                if log_config:
                    retention_config = {
                        "max_size_mb": log_config.get("max_file_size_mb", 10) * log_config.get("max_files", 5),
                        "max_age_days": log_config.get("retention_days", 30),
                        "rotation_size_mb": log_config.get("max_file_size_mb", 10),
                    }
                    policy = {**DEFAULT_RETENTION_POLICY, **retention_config}
            except Exception as e:
                retention_logger.warning(f"Failed to load policy from config_manager: {e}")
        
        self.policy = LogRetentionPolicy(policy)
        
        # Scheduled execution
        self._timer = None
        self._stop_event = threading.Event()
    
    def start_scheduler(self, interval_hours: float = 24.0) -> None:
        """
        Start the scheduler to periodically enforce policies.
        
        Args:
            interval_hours: Interval between policy enforcements in hours
        """
        if self._timer is not None:
            return
        
        # Schedule the first run
        self._schedule_next_run(interval_hours * 3600)
        
        # Log startup
        retention_logger.info("Log retention scheduler started")
        
        # Audit log
        if AUDIT_LOGGING_AVAILABLE:
            log_action(
                username="system",
                action="LOG_RETENTION_SCHEDULER_START",
                details=f"Started log retention scheduler with interval {interval_hours:.1f} hours"
            )
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler."""
        if self._timer is None:
            return
        
        self._stop_event.set()
        self._timer.cancel()
        self._timer = None
        
        # Log shutdown
        retention_logger.info("Log retention scheduler stopped")
        
        # Audit log
        if AUDIT_LOGGING_AVAILABLE:
            log_action(
                username="system",
                action="LOG_RETENTION_SCHEDULER_STOP"
            )
    
    def _schedule_next_run(self, delay: float) -> None:
        """
        Schedule the next policy enforcement.
        
        Args:
            delay: Delay in seconds
        """
        if self._stop_event.is_set():
            return
        
        # Schedule the timer
        self._timer = threading.Timer(
            delay,
            self._scheduler_callback
        )
        self._timer.daemon = True
        self._timer.start()
    
    def _scheduler_callback(self) -> None:
        """Callback for scheduled policy enforcement."""
        if self._stop_event.is_set():
            return
        
        try:
            # Enforce policies
            retention_logger.info("Running scheduled log retention policy enforcement")
            self.enforce_policies()
            
            # Schedule the next run
            self._schedule_next_run(24 * 3600)  # 24 hours
        except Exception as e:
            retention_logger.error(f"Error in scheduler callback: {e}")
            
            # Schedule the next run (shortened interval after error)
            self._schedule_next_run(4 * 3600)  # 4 hours
    
    def enforce_policies(self) -> Dict[str, Any]:
        """
        Enforce all retention policies.
        
        Returns:
            Dictionary with policy enforcement results
        """
        start_time = time.time()
        results = {
            "rotated": 0,
            "archived": 0,
            "deleted": 0,
            "errors": 0,
            "space_freed_bytes": 0,
            "timestamp": start_time,
            "duration_seconds": 0
        }
        
        try:
            # Rotate logs
            rotated_results = self.rotate_logs()
            results["rotated"] = rotated_results["rotated"]
            results["errors"] += rotated_results["errors"]
            
            # Archive logs
            archive_results = self.archive_logs()
            results["archived"] = archive_results["archived"]
            results["errors"] += archive_results["errors"]
            results["space_freed_bytes"] += archive_results["space_freed_bytes"]
            
            # Delete logs
            delete_results = self.delete_logs()
            results["deleted"] = delete_results["deleted"]
            results["errors"] += delete_results["errors"]
            results["space_freed_bytes"] += delete_results["space_freed_bytes"]
            
            # Enforce size limits
            size_results = self.enforce_size_limits()
            results["deleted"] += size_results["deleted"]
            results["errors"] += size_results["errors"]
            results["space_freed_bytes"] += size_results["space_freed_bytes"]
            
            # Calculate duration
            results["duration_seconds"] = time.time() - start_time
            
            # Log summary
            retention_logger.info(
                f"Policy enforcement completed: "
                f"rotated {results['rotated']}, "
                f"archived {results['archived']}, "
                f"deleted {results['deleted']}, "
                f"freed {results['space_freed_bytes'] / (1024 * 1024):.2f} MB, "
                f"took {results['duration_seconds']:.2f} seconds"
            )
            
            # Audit log
            if AUDIT_LOGGING_AVAILABLE:
                log_action(
                    username="system",
                    action="LOG_RETENTION_POLICY_ENFORCEMENT",
                    details=f"Rotated {results['rotated']}, archived {results['archived']}, "
                            f"deleted {results['deleted']}, "
                            f"freed {results['space_freed_bytes'] / (1024 * 1024):.2f} MB"
                )
        except Exception as e:
            # Log error
            retention_logger.error(f"Error enforcing policies: {e}")
            results["errors"] += 1
        
        return results
    
    def rotate_logs(self) -> Dict[str, Any]:
        """
        Rotate log files that exceed the rotation size.
        
        Returns:
            Dictionary with rotation results
        """
        results = {
            "rotated": 0,
            "errors": 0
        }
        
        # Get log files
        log_files = list(self.log_dir.glob("*.log"))
        
        for log_path in log_files:
            try:
                # Check if the log should be rotated
                if self.policy.should_rotate(log_path):
                    # Rotate the log
                    rotated_path = self._rotate_log(log_path)
                    if rotated_path:
                        results["rotated"] += 1
                        retention_logger.info(f"Rotated log file: {log_path} -> {rotated_path}")
            except Exception as e:
                # Log error and continue
                retention_logger.error(f"Error rotating log file {log_path}: {e}")
                results["errors"] += 1
        
        return results
    
    def archive_logs(self) -> Dict[str, Any]:
        """
        Archive rotated log files.
        
        Returns:
            Dictionary with archival results
        """
        results = {
            "archived": 0,
            "errors": 0,
            "space_freed_bytes": 0
        }
        
        # Get rotated log files
        rotated_logs = []
        for file in self.log_dir.glob("*"):
            if re.match(ROTATED_LOG_PATTERN, file.name):
                rotated_logs.append(file)
        
        for log_path in rotated_logs:
            try:
                # Check if the log should be archived
                if self.policy.should_archive(log_path):
                    # Archive the log
                    archived_path, space_saved = self._archive_log(log_path)
                    if archived_path:
                        results["archived"] += 1
                        results["space_freed_bytes"] += space_saved
                        retention_logger.info(
                            f"Archived log file: {log_path} -> {archived_path} "
                            f"(saved {space_saved / 1024:.2f} KB)"
                        )
            except Exception as e:
                # Log error and continue
                retention_logger.error(f"Error archiving log file {log_path}: {e}")
                results["errors"] += 1
        
        return results
    
    def delete_logs(self) -> Dict[str, Any]:
        """
        Delete log files according to the retention policy.
        
        Returns:
            Dictionary with deletion results
        """
        results = {
            "deleted": 0,
            "errors": 0,
            "space_freed_bytes": 0
        }
        
        # Get all log files (normal, rotated, and archived)
        all_logs = list(self.log_dir.glob("*.log*"))
        all_logs.extend(list(self.archive_dir.glob("*.gz")))
        
        for log_path in all_logs:
            try:
                # Check if the log should be deleted
                if self.policy.should_delete(log_path):
                    # Get file size before deletion
                    file_size = log_path.stat().st_size
                    
                    # Delete the log
                    log_path.unlink()
                    
                    # Update results
                    results["deleted"] += 1
                    results["space_freed_bytes"] += file_size
                    
                    retention_logger.info(
                        f"Deleted log file: {log_path} "
                        f"(freed {file_size / 1024:.2f} KB)"
                    )
            except Exception as e:
                # Log error and continue
                retention_logger.error(f"Error deleting log file {log_path}: {e}")
                results["errors"] += 1
        
        return results
    
    def enforce_size_limits(self) -> Dict[str, Any]:
        """
        Enforce maximum size limits for log directories.
        
        Returns:
            Dictionary with enforcement results
        """
        results = {
            "deleted": 0,
            "errors": 0,
            "space_freed_bytes": 0
        }
        
        try:
            # Check total size of log directory
            log_dir_size = self._get_directory_size(self.log_dir)
            archive_dir_size = self._get_directory_size(self.archive_dir)
            total_size = log_dir_size + archive_dir_size
            
            # Get maximum size
            max_size_bytes = self.policy.policy.get("max_size_mb", 100) * 1024 * 1024
            
            # Check if we need to enforce size limits
            if total_size <= max_size_bytes:
                return results
            
            # Calculate how much space we need to free
            space_to_free = total_size - max_size_bytes
            retention_logger.info(
                f"Log storage exceeds maximum size: "
                f"{total_size / (1024 * 1024):.2f} MB > "
                f"{max_size_bytes / (1024 * 1024):.2f} MB, "
                f"need to free {space_to_free / (1024 * 1024):.2f} MB"
            )
            
            # Delete archived logs first, oldest first
            if space_to_free > 0:
                archive_deletion = self._delete_oldest_logs(
                    self.archive_dir,
                    space_to_free,
                    exclude_patterns=self.policy.policy.get("excluded_logs", [])
                )
                results["deleted"] += archive_deletion["deleted"]
                results["errors"] += archive_deletion["errors"]
                results["space_freed_bytes"] += archive_deletion["space_freed_bytes"]
                space_to_free -= archive_deletion["space_freed_bytes"]
            
            # Then delete rotated logs, oldest first
            if space_to_free > 0:
                rotated_logs = [
                    f for f in self.log_dir.glob("*")
                    if re.match(ROTATED_LOG_PATTERN, f.name)
                ]
                rotated_deletion = self._delete_oldest_logs(
                    self.log_dir,
                    space_to_free,
                    file_list=rotated_logs,
                    exclude_patterns=self.policy.policy.get("excluded_logs", [])
                )
                results["deleted"] += rotated_deletion["deleted"]
                results["errors"] += rotated_deletion["errors"]
                results["space_freed_bytes"] += rotated_deletion["space_freed_bytes"]
                space_to_free -= rotated_deletion["space_freed_bytes"]
            
            # Finally, delete regular logs, starting with DEBUG and INFO
            if space_to_free > 0:
                # Delete by severity, lowest severity first
                for severity in ["DEBUG", "INFO", "WARNING", "ERROR"]:
                    if space_to_free <= 0:
                        break
                    
                    # Find logs of this severity
                    severity_logs = []
                    for log_path in self.log_dir.glob("*.log"):
                        if log_path.name in self.policy.policy.get("excluded_logs", []):
                            continue
                        
                        log_severity = self.policy._get_log_severity(log_path)
                        if log_severity == severity:
                            severity_logs.append(log_path)
                    
                    # Delete oldest logs of this severity
                    severity_deletion = self._delete_oldest_logs(
                        self.log_dir,
                        space_to_free,
                        file_list=severity_logs
                    )
                    results["deleted"] += severity_deletion["deleted"]
                    results["errors"] += severity_deletion["errors"]
                    results["space_freed_bytes"] += severity_deletion["space_freed_bytes"]
                    space_to_free -= severity_deletion["space_freed_bytes"]
        except Exception as e:
            # Log error
            retention_logger.error(f"Error enforcing size limits: {e}")
            results["errors"] += 1
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of log retention.
        
        Returns:
            Dictionary with status information
        """
        # Calculate sizes
        log_dir_size = self._get_directory_size(self.log_dir)
        archive_dir_size = self._get_directory_size(self.archive_dir)
        total_size = log_dir_size + archive_dir_size
        
        # Count files
        log_file_count = len(list(self.log_dir.glob("*.log")))
        rotated_file_count = len([
            f for f in self.log_dir.glob("*")
            if re.match(ROTATED_LOG_PATTERN, f.name)
        ])
        archived_file_count = len(list(self.archive_dir.glob("*.gz")))
        
        # Get disk usage
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.log_dir)
            free_percent = (free / total) * 100
        except Exception:
            total = used = free = free_percent = 0
        
        # Get policy information
        policy_info = self.policy.policy.copy()
        
        # Build status
        status = {
            "log_directory": str(self.log_dir),
            "archive_directory": str(self.archive_dir),
            "log_directory_size_mb": log_dir_size / (1024 * 1024),
            "archive_directory_size_mb": archive_dir_size / (1024 * 1024),
            "total_size_mb": total_size / (1024 * 1024),
            "log_file_count": log_file_count,
            "rotated_file_count": rotated_file_count,
            "archived_file_count": archived_file_count,
            "total_file_count": log_file_count + rotated_file_count + archived_file_count,
            "disk_usage": {
                "total_gb": total / (1024 * 1024 * 1024),
                "used_gb": used / (1024 * 1024 * 1024),
                "free_gb": free / (1024 * 1024 * 1024),
                "free_percent": free_percent
            },
            "policy": policy_info
        }
        
        return status
    
    def _rotate_log(self, log_path: Path) -> Optional[Path]:
        """
        Rotate a log file.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            Path to the rotated log file, or None if rotation failed
        """
        # Check if the file exists and is not protected
        if not log_path.exists() or log_path.name in self.policy.policy.get("protected_logs", []):
            return None
        
        # Generate rotated filename with date
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        rotated_path = log_path.with_name(f"{log_path.name}.{date_str}")
        
        # Handle case where rotated file already exists
        counter = 1
        while rotated_path.exists():
            rotated_path = log_path.with_name(f"{log_path.name}.{date_str}.{counter}")
            counter += 1
        
        try:
            # Rename the log file
            shutil.copy2(log_path, rotated_path)
            
            # Truncate the original log file
            with open(log_path, 'w') as f:
                f.write("")
            
            return rotated_path
        except Exception as e:
            retention_logger.error(f"Failed to rotate log {log_path}: {e}")
            return None
    
    def _archive_log(self, log_path: Path) -> Tuple[Optional[Path], int]:
        """
        Archive a log file.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            Tuple of (archived path or None, space saved in bytes)
        """
        # Check if the file exists
        if not log_path.exists():
            return None, 0
        
        # Check if compression is enabled
        if not self.policy.policy.get("compression_enabled", True):
            # Simply move the file
            archived_path = self.archive_dir / log_path.name
            
            # Handle case where archived file already exists
            counter = 1
            while archived_path.exists():
                archived_path = self.archive_dir / f"{log_path.stem}.{counter}{log_path.suffix}"
                counter += 1
            
            try:
                # Move the file
                shutil.move(log_path, archived_path)
                return archived_path, 0
            except Exception as e:
                retention_logger.error(f"Failed to archive log {log_path}: {e}")
                return None, 0
        
        # Generate archived filename
        archived_path = self.archive_dir / f"{log_path.name}.gz"
        
        # Handle case where archived file already exists
        counter = 1
        while archived_path.exists():
            archived_path = self.archive_dir / f"{log_path.name}.{counter}.gz"
            counter += 1
        
        try:
            # Get original file size
            original_size = log_path.stat().st_size
            
            # Compress the file
            with open(log_path, 'rb') as f_in:
                with gzip.open(
                    archived_path,
                    'wb',
                    compresslevel=self.policy.policy.get("compression_level", 6)
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Get compressed file size
            compressed_size = archived_path.stat().st_size
            
            # Delete the original file
            log_path.unlink()
            
            # Calculate space saved
            space_saved = original_size - compressed_size
            
            return archived_path, space_saved
        except Exception as e:
            retention_logger.error(f"Failed to archive log {log_path}: {e}")
            return None, 0
    
    def _get_directory_size(self, directory: Path) -> int:
        """
        Get the total size of a directory in bytes.
        
        Args:
            directory: Path to the directory
            
        Returns:
            Total size in bytes
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = Path(dirpath) / filename
                if file_path.exists():
                    total_size += file_path.stat().st_size
        
        return total_size
    
    def _delete_oldest_logs(
        self,
        directory: Path,
        space_to_free: int,
        file_list: Optional[List[Path]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Delete oldest log files to free up space.
        
        Args:
            directory: Directory containing log files
            space_to_free: Space to free in bytes
            file_list: Optional list of files to consider
            exclude_patterns: List of filename patterns to exclude
            
        Returns:
            Dictionary with deletion results
        """
        results = {
            "deleted": 0,
            "errors": 0,
            "space_freed_bytes": 0
        }
        
        # Use provided file list or get all files in directory
        if file_list is None:
            files = list(directory.glob("*"))
        else:
            files = file_list.copy()
        
        # Filter out excluded files
        if exclude_patterns:
            for pattern in exclude_patterns:
                files = [f for f in files if pattern not in f.name]
        
        # Sort files by modification time (oldest first)
        files.sort(key=lambda f: f.stat().st_mtime if f.exists() else float('inf'))
        
        # Delete files until we've freed enough space
        freed_space = 0
        for file_path in files:
            # Check if we've freed enough space
            if freed_space >= space_to_free:
                break
            
            try:
                # Get file size
                file_size = file_path.stat().st_size
                
                # Delete file
                file_path.unlink()
                
                # Update results
                results["deleted"] += 1
                freed_space += file_size
                results["space_freed_bytes"] += file_size
                
                retention_logger.info(
                    f"Deleted log file {file_path} to free space "
                    f"(freed {file_size / 1024:.2f} KB)"
                )
            except Exception as e:
                # Log error and continue
                retention_logger.error(f"Error deleting log file {file_path}: {e}")
                results["errors"] += 1
        
        return results


# CLI interface
def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(description="Log Retention Policy Tool")
    
    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show log retention status")
    
    # Enforce command
    enforce_parser = subparsers.add_parser("enforce", help="Enforce retention policies")
    enforce_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    # Rotate command
    rotate_parser = subparsers.add_parser("rotate", help="Rotate log files")
    rotate_parser.add_argument(
        "--log",
        type=str,
        help="Specific log file to rotate"
    )
    
    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Archive rotated log files")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete old log files")
    delete_parser.add_argument(
        "--older-than",
        type=int,
        help="Delete logs older than this many days"
    )
    
    # Configure command
    config_parser = subparsers.add_parser("config", help="Configure retention policy")
    config_parser.add_argument(
        "--max-size",
        type=int,
        help="Maximum size of log directory in MB"
    )
    config_parser.add_argument(
        "--max-age",
        type=int,
        help="Maximum age of log files in days"
    )
    config_parser.add_argument(
        "--rotation-size",
        type=int,
        help="Size at which to rotate a log file in MB"
    )
    
    # Schedule command
    schedule_parser = subparsers.add_parser("schedule", help="Run the scheduler")
    schedule_parser.add_argument(
        "--interval",
        type=float,
        default=24.0,
        help="Interval between policy enforcements in hours"
    )
    
    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Create log retention manager
    try:
        manager = LogRetentionManager()
        
        # Handle command
        if args.command == "status":
            # Get status
            status = manager.get_status()
            
            # Print status
            print("=== Log Retention Status ===")
            print(f"Log directory: {status['log_directory']}")
            print(f"Archive directory: {status['archive_directory']}")
            print(f"Total size: {status['total_size_mb']:.2f} MB")
            print(f"  Log directory: {status['log_directory_size_mb']:.2f} MB")
            print(f"  Archive directory: {status['archive_directory_size_mb']:.2f} MB")
            print(f"File count: {status['total_file_count']}")
            print(f"  Log files: {status['log_file_count']}")
            print(f"  Rotated files: {status['rotated_file_count']}")
            print(f"  Archived files: {status['archived_file_count']}")
            
            # Print disk usage
            disk = status["disk_usage"]
            print(f"Disk usage:")
            print(f"  Total: {disk['total_gb']:.2f} GB")
            print(f"  Used: {disk['used_gb']:.2f} GB")
            print(f"  Free: {disk['free_gb']:.2f} GB ({disk['free_percent']:.1f}%)")
            
            # Print policy
            policy = status["policy"]
            print(f"Policy:")
            print(f"  Max size: {policy['max_size_mb']} MB")
            print(f"  Max age: {policy['max_age_days']} days")
            print(f"  Rotation size: {policy['rotation_size_mb']} MB")
            print(f"  Archive older than: {policy['archive_older_than_days']} days")
            print(f"  Delete archives older than: {policy['delete_archives_older_than_days']} days")
            
        elif args.command == "enforce":
            if args.dry_run:
                # TODO: Implement dry run mode
                print("Dry run mode not implemented yet")
            else:
                # Enforce policies
                print("Enforcing retention policies...")
                results = manager.enforce_policies()
                
                # Print results
                print(f"Rotated: {results['rotated']}")
                print(f"Archived: {results['archived']}")
                print(f"Deleted: {results['deleted']}")
                print(f"Errors: {results['errors']}")
                print(f"Space freed: {results['space_freed_bytes'] / (1024 * 1024):.2f} MB")
                print(f"Duration: {results['duration_seconds']:.2f} seconds")
                
        elif args.command == "rotate":
            if args.log:
                # Rotate specific log
                log_path = Path(args.log)
                if not log_path.exists():
                    print(f"Log file not found: {log_path}")
                    return
                
                # Rotate log
                print(f"Rotating log file: {log_path}")
                rotated_path = manager._rotate_log(log_path)
                if rotated_path:
                    print(f"Rotated to: {rotated_path}")
                else:
                    print("Failed to rotate log file")
            else:
                # Rotate all logs
                print("Rotating log files...")
                results = manager.rotate_logs()
                
                # Print results
                print(f"Rotated: {results['rotated']}")
                print(f"Errors: {results['errors']}")
                
        elif args.command == "archive":
            # Archive logs
            print("Archiving log files...")
            results = manager.archive_logs()
            
            # Print results
            print(f"Archived: {results['archived']}")
            print(f"Errors: {results['errors']}")
            print(f"Space saved: {results['space_freed_bytes'] / (1024 * 1024):.2f} MB")
            
        elif args.command == "delete":
            older_than = args.older_than
            if older_than:
                # Override policy temporarily
                manager.policy.policy["max_age_days"] = older_than
            
            # Delete logs
            print("Deleting log files...")
            results = manager.delete_logs()
            
            # Print results
            print(f"Deleted: {results['deleted']}")
            print(f"Errors: {results['errors']}")
            print(f"Space freed: {results['space_freed_bytes'] / (1024 * 1024):.2f} MB")
            
        elif args.command == "config":
            # Not implemented yet - would update config_manager or direct policy
            print("Config command not implemented yet")
            
        elif args.command == "schedule":
            # Start scheduler
            print(f"Starting log retention scheduler (interval: {args.interval} hours)...")
            manager.start_scheduler(args.interval)
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Stopping scheduler...")
                manager.stop_scheduler()
            
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
