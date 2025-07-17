#!/usr/bin/env python3
"""
process_monitoring.py
────────────────────
Real-time process monitoring for the FixWurx platform.

This module provides comprehensive monitoring of processes spawned by the shell,
including resource usage tracking, state management, and event notifications.
"""

import os
import sys
import time
import signal
import logging
import threading
import subprocess
import psutil
import json
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
import queue
import re

# Internal imports
from shell_environment import register_event_handler
from shell_environment import EventType
from shell_environment import emit_event

# Configure logging
logger = logging.getLogger("ProcessMonitoring")

# Constants
DEFAULT_UPDATE_INTERVAL = 1.0  # seconds
DEFAULT_HISTORY_SIZE = 100  # data points
DEFAULT_CPU_THRESHOLD = 90.0  # percent
DEFAULT_MEMORY_THRESHOLD = 90.0  # percent
DEFAULT_IO_THRESHOLD = 80.0  # percent
DEFAULT_PROCESS_TIMEOUT = 300.0  # seconds
DEFAULT_ZOMBIE_CHECK_INTERVAL = 60.0  # seconds
DEFAULT_MAX_PROCESS_AGE = 24 * 60 * 60  # 1 day in seconds

class ProcessState(Enum):
    """Process states."""
    STARTING = auto()
    RUNNING = auto()
    SLEEPING = auto()
    DISK_SLEEP = auto()
    STOPPED = auto()
    TRACING_STOP = auto()
    ZOMBIE = auto()
    DEAD = auto()
    WAKE_KILL = auto()
    WAKING = auto()
    IDLE = auto()
    LOCKED = auto()
    WAITING = auto()
    SUSPENDED = auto()
    PARKED = auto()
    UNKNOWN = auto()

    @classmethod
    def from_psutil(cls, status: str) -> 'ProcessState':
        """Convert psutil status to ProcessState."""
        mapping = {
            'running': cls.RUNNING,
            'sleeping': cls.SLEEPING,
            'disk-sleep': cls.DISK_SLEEP,
            'stopped': cls.STOPPED,
            'tracing-stop': cls.TRACING_STOP,
            'zombie': cls.ZOMBIE,
            'dead': cls.DEAD,
            'wake-kill': cls.WAKE_KILL,
            'waking': cls.WAKING,
            'idle': cls.IDLE,
            'locked': cls.LOCKED,
            'waiting': cls.WAITING,
            'suspended': cls.SUSPENDED,
            'parked': cls.PARKED
        }
        return mapping.get(status, cls.UNKNOWN)

class MonitoringLevel(Enum):
    """Process monitoring levels."""
    MINIMAL = auto()  # Basic info: PID, name, status
    STANDARD = auto()  # + CPU, memory usage
    DETAILED = auto()  # + I/O, network, open files
    COMPREHENSIVE = auto()  # + environment, command line, threads
    DEBUG = auto()  # Everything available

@dataclass
class ProcessMetrics:
    """Process performance metrics."""
    pid: int
    name: str
    create_time: float
    status: ProcessState
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_rss: int = 0  # Resident Set Size in bytes
    memory_vms: int = 0  # Virtual Memory Size in bytes
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    io_read_count: int = 0
    io_write_count: int = 0
    network_sent_bytes: int = 0
    network_recv_bytes: int = 0
    open_files_count: int = 0
    thread_count: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "pid": self.pid,
            "name": self.name,
            "create_time": self.create_time,
            "status": self.status.name,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_rss": self.memory_rss,
            "memory_vms": self.memory_vms,
            "io_read_bytes": self.io_read_bytes,
            "io_write_bytes": self.io_write_bytes,
            "io_read_count": self.io_read_count,
            "io_write_count": self.io_write_count,
            "network_sent_bytes": self.network_sent_bytes,
            "network_recv_bytes": self.network_recv_bytes,
            "open_files_count": self.open_files_count,
            "thread_count": self.thread_count,
            "timestamp": self.timestamp
        }

@dataclass
class ProcessInfo:
    """Process information with performance history."""
    pid: int
    name: str
    create_time: float
    command_line: List[str]
    cwd: str
    owner: str
    shell_command: Optional[str] = None
    monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD
    history: List[ProcessMetrics] = field(default_factory=list)
    history_size: int = DEFAULT_HISTORY_SIZE
    active: bool = True
    parent_pid: Optional[int] = None
    children_pids: List[int] = field(default_factory=list)
    cpu_threshold: float = DEFAULT_CPU_THRESHOLD
    memory_threshold: float = DEFAULT_MEMORY_THRESHOLD
    io_threshold: float = DEFAULT_IO_THRESHOLD
    timeout: Optional[float] = None
    exit_code: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    
    def add_metrics(self, metrics: ProcessMetrics) -> None:
        """Add metrics to history."""
        self.history.append(metrics)
        if len(self.history) > self.history_size:
            self.history.pop(0)
    
    def get_latest_metrics(self) -> Optional[ProcessMetrics]:
        """Get latest metrics."""
        if self.history:
            return self.history[-1]
        return None
    
    def get_average_cpu(self, timeframe: int = 5) -> float:
        """Get average CPU usage over the last n metrics."""
        if not self.history:
            return 0.0
        n = min(timeframe, len(self.history))
        if n == 0:
            return 0.0
        return sum(m.cpu_percent for m in self.history[-n:]) / n
    
    def get_average_memory(self, timeframe: int = 5) -> float:
        """Get average memory usage over the last n metrics."""
        if not self.history:
            return 0.0
        n = min(timeframe, len(self.history))
        if n == 0:
            return 0.0
        return sum(m.memory_percent for m in self.history[-n:]) / n
    
    def get_io_rate(self, timeframe: int = 5) -> Tuple[float, float]:
        """Get I/O rate (bytes/sec) over the last n metrics."""
        if len(self.history) < 2:
            return 0.0, 0.0
        
        n = min(timeframe + 1, len(self.history))
        if n < 2:
            return 0.0, 0.0
        
        metrics = self.history[-(n):]
        oldest = metrics[0]
        newest = metrics[-1]
        
        time_diff = newest.timestamp - oldest.timestamp
        if time_diff <= 0:
            return 0.0, 0.0
        
        read_diff = newest.io_read_bytes - oldest.io_read_bytes
        write_diff = newest.io_write_bytes - oldest.io_write_bytes
        
        read_rate = read_diff / time_diff
        write_rate = write_diff / time_diff
        
        return read_rate, write_rate
    
    def get_runtime(self) -> float:
        """Get process runtime in seconds."""
        if self.ended_at:
            return self.ended_at - self.started_at
        return time.time() - self.started_at
    
    def to_dict(self, include_history: bool = False) -> Dict[str, Any]:
        """Convert process info to dictionary."""
        result = {
            "pid": self.pid,
            "name": self.name,
            "create_time": self.create_time,
            "command_line": self.command_line,
            "cwd": self.cwd,
            "owner": self.owner,
            "shell_command": self.shell_command,
            "monitoring_level": self.monitoring_level.name,
            "active": self.active,
            "parent_pid": self.parent_pid,
            "children_pids": self.children_pids,
            "cpu_threshold": self.cpu_threshold,
            "memory_threshold": self.memory_threshold,
            "io_threshold": self.io_threshold,
            "timeout": self.timeout,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "runtime": self.get_runtime()
        }
        
        latest = self.get_latest_metrics()
        if latest:
            result["latest_metrics"] = latest.to_dict()
            result["average_cpu"] = self.get_average_cpu()
            result["average_memory"] = self.get_average_memory()
            result["io_rates"] = {
                "read_rate": self.get_io_rate()[0],
                "write_rate": self.get_io_rate()[1]
            }
        
        if include_history:
            result["history"] = [m.to_dict() for m in self.history]
        
        return result

class ProcessMonitoringError(Exception):
    """Exception raised for process monitoring errors."""
    pass

class ProcessTimeoutError(ProcessMonitoringError):
    """Exception raised when a process exceeds its timeout."""
    pass

class ProcessMonitor:
    """
    Real-time process monitoring for the FixWurx platform.
    
    This class provides monitoring of processes spawned by the shell,
    including resource usage tracking, state management, and notifications.
    """
    
    def __init__(self):
        """Initialize the process monitor."""
        self._processes: Dict[int, ProcessInfo] = {}
        self._monitored_commands: Dict[str, List[int]] = {}
        self._processes_lock = threading.RLock()
        self._update_interval = DEFAULT_UPDATE_INTERVAL
        self._default_level = MonitoringLevel.STANDARD
        
        # Thread for updating process metrics
        self._metrics_thread = None
        self._stop_metrics = threading.Event()
        
        # Thread for checking for zombie processes
        self._zombie_thread = None
        self._stop_zombie = threading.Event()
        
        # Thread for checking process timeouts
        self._timeout_thread = None
        self._stop_timeout = threading.Event()
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {
            "process_started": [],
            "process_ended": [],
            "process_threshold_exceeded": [],
            "process_timeout": [],
            "process_zombie": [],
            "process_error": []
        }
        
        # Register with shell environment
        self._register_shell_handlers()
        
        logger.info("Process monitor initialized")
    
    def _register_shell_handlers(self) -> None:
        """Register event handlers with shell environment."""
        try:
            register_event_handler(EventType.PROCESS_STARTED, self._handle_process_started)
            register_event_handler(EventType.PROCESS_ENDED, self._handle_process_ended)
            register_event_handler(EventType.COMMAND_STARTED, self._handle_command_started)
            register_event_handler(EventType.COMMAND_ENDED, self._handle_command_ended)
            logger.debug("Registered shell event handlers")
        except Exception as e:
            logger.error(f"Error registering shell event handlers: {e}")
    
    def _handle_process_started(self, event_data: Dict[str, Any]) -> None:
        """Handle process started event from shell."""
        try:
            pid = event_data.get("pid")
            if not pid:
                logger.error("Process started event missing PID")
                return
            
            command = event_data.get("command")
            self.track_process(pid, shell_command=command)
            
        except Exception as e:
            logger.error(f"Error handling process started event: {e}")
    
    def _handle_process_ended(self, event_data: Dict[str, Any]) -> None:
        """Handle process ended event from shell."""
        try:
            pid = event_data.get("pid")
            if not pid:
                logger.error("Process ended event missing PID")
                return
            
            exit_code = event_data.get("exit_code")
            self.mark_process_ended(pid, exit_code)
            
        except Exception as e:
            logger.error(f"Error handling process ended event: {e}")
    
    def _handle_command_started(self, event_data: Dict[str, Any]) -> None:
        """Handle command started event from shell."""
        try:
            command = event_data.get("command")
            if not command:
                logger.error("Command started event missing command")
                return
            
            # Initialize tracking for this command
            self._monitored_commands[command] = []
            
        except Exception as e:
            logger.error(f"Error handling command started event: {e}")
    
    def _handle_command_ended(self, event_data: Dict[str, Any]) -> None:
        """Handle command ended event from shell."""
        try:
            command = event_data.get("command")
            if not command:
                logger.error("Command ended event missing command")
                return
            
            # Remove tracking for this command
            if command in self._monitored_commands:
                del self._monitored_commands[command]
            
        except Exception as e:
            logger.error(f"Error handling command ended event: {e}")
    
    def start(self) -> None:
        """Start the process monitor."""
        # Start metrics update thread
        if self._metrics_thread is None or not self._metrics_thread.is_alive():
            self._stop_metrics.clear()
            self._metrics_thread = threading.Thread(
                target=self._update_metrics_loop,
                daemon=True,
                name="ProcessMetricsThread"
            )
            self._metrics_thread.start()
        
        # Start zombie check thread
        if self._zombie_thread is None or not self._zombie_thread.is_alive():
            self._stop_zombie.clear()
            self._zombie_thread = threading.Thread(
                target=self._check_zombies_loop,
                daemon=True,
                name="ZombieCheckThread"
            )
            self._zombie_thread.start()
        
        # Start timeout check thread
        if self._timeout_thread is None or not self._timeout_thread.is_alive():
            self._stop_timeout.clear()
            self._timeout_thread = threading.Thread(
                target=self._check_timeouts_loop,
                daemon=True,
                name="TimeoutCheckThread"
            )
            self._timeout_thread.start()
        
        logger.info("Process monitor started")
    
    def stop(self) -> None:
        """Stop the process monitor."""
        # Stop metrics update thread
        if self._metrics_thread and self._metrics_thread.is_alive():
            self._stop_metrics.set()
            self._metrics_thread.join(timeout=2.0)
        
        # Stop zombie check thread
        if self._zombie_thread and self._zombie_thread.is_alive():
            self._stop_zombie.set()
            self._zombie_thread.join(timeout=2.0)
        
        # Stop timeout check thread
        if self._timeout_thread and self._timeout_thread.is_alive():
            self._stop_timeout.set()
            self._timeout_thread.join(timeout=2.0)
        
        logger.info("Process monitor stopped")
    
    def _update_metrics_loop(self) -> None:
        """Update metrics for all monitored processes."""
        while not self._stop_metrics.is_set():
            try:
                self._update_all_metrics()
            except Exception as e:
                logger.error(f"Error updating process metrics: {e}")
            
            # Sleep before next update
            time.sleep(self._update_interval)
    
    def _update_all_metrics(self) -> None:
        """Update metrics for all monitored processes."""
        with self._processes_lock:
            # Get copy of process IDs to avoid modification during iteration
            pids = list(self._processes.keys())
        
        for pid in pids:
            try:
                self._update_process_metrics(pid)
            except Exception as e:
                logger.error(f"Error updating metrics for process {pid}: {e}")
    
    def _update_process_metrics(self, pid: int) -> None:
        """
        Update metrics for a specific process.
        
        Args:
            pid: Process ID.
        """
        with self._processes_lock:
            if pid not in self._processes:
                return
            
            process_info = self._processes[pid]
            if not process_info.active:
                return
        
        try:
            # Get process from psutil
            process = psutil.Process(pid)
            
            # Check if process is still running
            if not process.is_running():
                self.mark_process_ended(pid)
                return
            
            # Get metrics based on monitoring level
            metrics = self._collect_process_metrics(process, process_info.monitoring_level)
            
            # Add metrics to process info
            with self._processes_lock:
                if pid in self._processes:
                    process_info = self._processes[pid]
                    process_info.add_metrics(metrics)
                    
                    # Check thresholds
                    self._check_process_thresholds(process_info)
        
        except psutil.NoSuchProcess:
            # Process no longer exists
            self.mark_process_ended(pid)
        except psutil.AccessDenied:
            logger.warning(f"Access denied for process {pid}")
        except Exception as e:
            logger.error(f"Error updating metrics for process {pid}: {e}")
            self._emit_event("process_error", {
                "pid": pid,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    def _collect_process_metrics(self, process: psutil.Process, 
                               level: MonitoringLevel) -> ProcessMetrics:
        """
        Collect metrics for a process based on monitoring level.
        
        Args:
            process: psutil Process object.
            level: Monitoring level.
            
        Returns:
            ProcessMetrics object.
        """
        # Basic info (always collected)
        pid = process.pid
        name = process.name()
        create_time = process.create_time()
        status = ProcessState.from_psutil(process.status())
        
        # Create metrics object
        metrics = ProcessMetrics(
            pid=pid,
            name=name,
            create_time=create_time,
            status=status,
            timestamp=time.time()
        )
        
        # Standard level (CPU, memory)
        if level.value >= MonitoringLevel.STANDARD.value:
            try:
                metrics.cpu_percent = process.cpu_percent(interval=None)
                memory_info = process.memory_info()
                metrics.memory_percent = process.memory_percent()
                metrics.memory_rss = memory_info.rss
                metrics.memory_vms = memory_info.vms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Detailed level (I/O, network, open files)
        if level.value >= MonitoringLevel.DETAILED.value:
            try:
                io_counters = process.io_counters()
                metrics.io_read_bytes = io_counters.read_bytes
                metrics.io_write_bytes = io_counters.write_bytes
                metrics.io_read_count = io_counters.read_count
                metrics.io_write_count = io_counters.write_count
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                pass
            
            try:
                # Get network stats (sum of all connections)
                connections = process.connections()
                net_io = process.net_io_counters() if hasattr(process, 'net_io_counters') else None
                if net_io:
                    metrics.network_sent_bytes = net_io.bytes_sent
                    metrics.network_recv_bytes = net_io.bytes_recv
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                pass
            
            try:
                metrics.open_files_count = len(process.open_files())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Comprehensive level (threads)
        if level.value >= MonitoringLevel.COMPREHENSIVE.value:
            try:
                metrics.thread_count = len(process.threads())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return metrics
    
    def _check_process_thresholds(self, process_info: ProcessInfo) -> None:
        """
        Check if process metrics exceed thresholds.
        
        Args:
            process_info: Process information.
        """
        latest = process_info.get_latest_metrics()
        if not latest:
            return
        
        # Check CPU threshold
        if latest.cpu_percent > process_info.cpu_threshold:
            self._emit_event("process_threshold_exceeded", {
                "pid": process_info.pid,
                "name": process_info.name,
                "metric": "cpu",
                "value": latest.cpu_percent,
                "threshold": process_info.cpu_threshold
            })
        
        # Check memory threshold
        if latest.memory_percent > process_info.memory_threshold:
            self._emit_event("process_threshold_exceeded", {
                "pid": process_info.pid,
                "name": process_info.name,
                "metric": "memory",
                "value": latest.memory_percent,
                "threshold": process_info.memory_threshold
            })
        
        # Check I/O threshold
        io_read_rate, io_write_rate = process_info.get_io_rate()
        total_io_rate = io_read_rate + io_write_rate
        
        # We need a baseline to compare against
        # For simplicity, we'll use a fixed value based on disk speed
        # In a real system, this would be based on actual disk performance
        estimated_disk_speed = 100 * 1024 * 1024  # 100 MB/s
        io_percent = (total_io_rate / estimated_disk_speed) * 100
        
        if io_percent > process_info.io_threshold:
            self._emit_event("process_threshold_exceeded", {
                "pid": process_info.pid,
                "name": process_info.name,
                "metric": "io",
                "value": io_percent,
                "threshold": process_info.io_threshold,
                "read_rate": io_read_rate,
                "write_rate": io_write_rate
            })
    
    def _check_zombies_loop(self) -> None:
        """Check for zombie processes periodically."""
        while not self._stop_zombie.is_set():
            try:
                self._check_zombie_processes()
            except Exception as e:
                logger.error(f"Error checking zombie processes: {e}")
            
            # Sleep before next check
            time.sleep(DEFAULT_ZOMBIE_CHECK_INTERVAL)
    
    def _check_zombie_processes(self) -> None:
        """Check for zombie processes."""
        with self._processes_lock:
            # Get copy of process IDs to avoid modification during iteration
            pids = list(self._processes.keys())
        
        for pid in pids:
            try:
                process = psutil.Process(pid)
                
                # Check if process is a zombie
                if process.status() == "zombie":
                    with self._processes_lock:
                        if pid in self._processes:
                            process_info = self._processes[pid]
                            
                            # Emit event
                            self._emit_event("process_zombie", {
                                "pid": pid,
                                "name": process_info.name,
                                "runtime": process_info.get_runtime()
                            })
                            
                            # Try to terminate zombie
                            try:
                                process.terminate()
                                logger.info(f"Terminated zombie process {pid}")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                            
                            # Mark as ended
                            self.mark_process_ended(pid)
            
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.mark_process_ended(pid)
            except Exception as e:
                logger.error(f"Error checking zombie process {pid}: {e}")
    
    def _check_timeouts_loop(self) -> None:
        """Check for process timeouts periodically."""
        while not self._stop_timeout.is_set():
            try:
                self._check_process_timeouts()
            except Exception as e:
                logger.error(f"Error checking process timeouts: {e}")
            
            # Sleep before next check
            time.sleep(1.0)
    
    def _check_process_timeouts(self) -> None:
        """Check for process timeouts."""
        current_time = time.time()
        
        with self._processes_lock:
            # Get copy of process IDs to avoid modification during iteration
            pids = list(self._processes.keys())
        
        for pid in pids:
            with self._processes_lock:
                if pid not in self._processes:
                    continue
                
                process_info = self._processes[pid]
                
                # Skip inactive processes
                if not process_info.active:
                    continue
                
                # Check if process has a timeout
                if process_info.timeout is not None:
                    runtime = process_info.get_runtime()
                    
                    # Check if timeout exceeded
                    if runtime > process_info.timeout:
                        # Emit event
                        self._emit_event("process_timeout", {
                            "pid": pid,
                            "name": process_info.name,
                            "runtime": runtime,
                            "timeout": process_info.timeout
                        })
                        
                        # Try to terminate process
                        try:
                            process = psutil.Process(pid)
                            process.terminate()
                            logger.info(f"Terminated process {pid} due to timeout")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        
                        # Mark as ended
                        self.mark_process_ended(pid)
                
                # Check if process is too old (regardless of timeout)
                if process_info.get_runtime() > DEFAULT_MAX_PROCESS_AGE:
                    logger.warning(f"Process {pid} ({process_info.name}) is too old: {process_info.get_runtime()} seconds")
    
    def track_process(self, pid: int, shell_command: Optional[str] = None,
                    monitoring_level: Optional[MonitoringLevel] = None,
                    timeout: Optional[float] = None) -> None:
        """
        Start tracking a process.
        
        Args:
            pid: Process ID.
            shell_command: Shell command that started the process.
            monitoring_level: Monitoring level.
            timeout: Process timeout in seconds.
        """
        try:
            # Get process from psutil
            process = psutil.Process(pid)
            
            # Get process info
            name = process.name()
            create_time = process.create_time()
            
            try:
                command_line = process.cmdline()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                command_line = []
            
            try:
                cwd = process.cwd()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                cwd = ""
            
            try:
                user = process.username()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                user = ""
            
            try:
                parent = process.parent()
                parent_pid = parent.pid if parent else None
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                parent_pid = None
            
            try:
                children = process.children()
                children_pids = [child.pid for child in children]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                children_pids = []
            
            # Create process info
            process_info = ProcessInfo(
                pid=pid,
                name=name,
                create_time=create_time,
                command_line=command_line,
                cwd=cwd,
                owner=user,
                shell_command=shell_command,
                monitoring_level=monitoring_level or self._default_level,
                parent_pid=parent_pid,
                children_pids=children_pids,
                timeout=timeout
            )
            
            # Add initial metrics
            metrics = self._collect_process_metrics(process, process_info.monitoring_level)
            process_info.add_metrics(metrics)
            
            # Add to tracked processes
            with self._processes_lock:
                self._processes[pid] = process_info
            
            # Add to command tracking if shell command provided
            if shell_command and shell_command in self._monitored_commands:
                self._monitored_commands[shell_command].append(pid)
            
            # Emit event
            self._emit_event("process_started", {
                "pid": pid,
                "name": name,
                "command": shell_command
            })
            
            logger.info(f"Started tracking process {pid} ({name})")
        
        except psutil.NoSuchProcess:
            logger.warning(f"Process {pid} no longer exists")
        except Exception as e:
            logger.error(f"Error tracking process {pid}: {e}")
            import traceback
            self._emit_event("process_error", {
                "pid": pid,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    def mark_process_ended(self, pid: int, exit_code: Optional[int] = None) -> None:
        """
        Mark a process as ended.
        
        Args:
            pid: Process ID.
            exit_code: Process exit code.
        """
        with self._processes_lock:
            if pid not in self._processes:
                return
            
            process_info = self._processes[pid]
            
            # Skip if already marked as ended
            if not process_info.active:
                return
            
            # Mark as inactive
            process_info.active = False
            process_info.ended_at = time.time()
            process_info.exit_code = exit_code
            
            # Get command for event
            command = process_info.shell_command
            
            # Remove from command tracking
            if command and command in self._monitored_commands:
                if pid in self._monitored_commands[command]:
                    self._monitored_commands[command].remove(pid)
        
        # Emit event
        self._emit_event("process_ended", {
            "pid": pid,
            "name": process_info.name,
            "exit_code": exit_code,
            "runtime": process_info.get_runtime()
        })
        
        logger.info(f"Process {pid} ({process_info.name}) ended with exit code {exit_code}")
    
    def get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a tracked process.
        
        Args:
            pid: Process ID.
            
        Returns:
            Process information dictionary or None if not found.
        """
        with self._processes_lock:
            if pid not in self._processes:
                return None
            
            return self._processes[pid].to_dict()
    
    def get_all_processes(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get information about all tracked processes.
        
        Args:
            active_only: Whether to include only active processes.
            
        Returns:
            List of process information dictionaries.
        """
        with self._processes_lock:
            if active_only:
                return [p.to_dict() for p in self._processes.values() if p.active]
            else:
                return [p.to_dict() for p in self._processes.values()]
    
    def get_command_processes(self, command: str) -> List[Dict[str, Any]]:
        """
        Get information about processes started by a command.
        
        Args:
            command: Shell command.
            
        Returns:
            List of process information dictionaries.
        """
        with self._processes_lock:
            if command not in self._monitored_commands:
                return []
            
            pids = self._monitored_commands[command]
            return [self._processes[pid].to_dict() for pid in pids if pid in self._processes]
    
    def set_process_timeout(self, pid: int, timeout: float) -> bool:
        """
        Set a timeout for a process.
        
        Args:
            pid: Process ID.
            timeout: Timeout in seconds.
            
        Returns:
            True if timeout was set, False otherwise.
        """
        with self._processes_lock:
            if pid not in self._processes:
                return False
            
            self._processes[pid].timeout = timeout
            return True
    
    def set_process_monitoring_level(self, pid: int, level: MonitoringLevel) -> bool:
        """
        Set the monitoring level for a process.
        
        Args:
            pid: Process ID.
            level: Monitoring level.
            
        Returns:
            True if level was set, False otherwise.
        """
        with self._processes_lock:
            if pid not in self._processes:
                return False
            
            self._processes[pid].monitoring_level = level
            return True
    
    def register_event_handler(self, event_type: str, handler: Callable) -> bool:
        """
        Register an event handler.
        
        Args:
            event_type: Event type.
            handler: Event handler.
            
        Returns:
            True if handler was registered, False otherwise.
        """
        if event_type not in self._event_handlers:
            return False
        
        self._event_handlers[event_type].append(handler)
        return True
    
    def unregister_event_handler(self, event_type: str, handler: Callable) -> bool:
        """
        Unregister an event handler.
        
        Args:
            event_type: Event type.
            handler: Event handler.
            
        Returns:
            True if handler was unregistered, False otherwise.
        """
        if event_type not in self._event_handlers:
            return False
        
        if handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)
            return True
        
        return False
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Emit an event.
        
        Args:
            event_type: Event type.
            event_data: Event data.
        """
        if event_type not in self._event_handlers:
            return
        
        # Add timestamp to event data
        event_data["timestamp"] = time.time()
        
        # Call handlers
        for handler in self._event_handlers[event_type]:
            try:
                handler(event_data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
        
        # Emit to shell environment
        try:
            # Map event type to shell event type
            shell_event_mapping = {
                "process_started": EventType.PROCESS_STARTED,
                "process_ended": EventType.PROCESS_ENDED,
                "process_threshold_exceeded": EventType.PROCESS_THRESHOLD_EXCEEDED,
                "process_timeout": EventType.PROCESS_TIMEOUT,
                "process_zombie": EventType.PROCESS_ZOMBIE,
                "process_error": EventType.PROCESS_ERROR
            }
            
            if event_type in shell_event_mapping:
                emit_event(shell_event_mapping[event_type], event_data)
        except Exception as e:
            logger.error(f"Error emitting event to shell: {e}")
    
    def terminate_process(self, pid: int, force: bool = False) -> bool:
        """
        Terminate a process.
        
        Args:
            pid: Process ID.
            force: Whether to force termination.
            
        Returns:
            True if process was terminated, False otherwise.
        """
        try:
            process = psutil.Process(pid)
            
            if force:
                process.kill()
            else:
                process.terminate()
            
            # Mark as ended
            self.mark_process_ended(pid, -9 if force else -15)
            
            return True
        
        except psutil.NoSuchProcess:
            # Process already gone, just mark as ended
            self.mark_process_ended(pid, -9 if force else -15)
            return True
        
        except Exception as e:
            logger.error(f"Error terminating process {pid}: {e}")
            return False
    
    def terminate_command_processes(self, command: str, force: bool = False) -> int:
        """
        Terminate all processes started by a command.
        
        Args:
            command: Shell command.
            force: Whether to force termination.
            
        Returns:
            Number of processes terminated.
        """
        with self._processes_lock:
            if command not in self._monitored_commands:
                return 0
            
            pids = self._monitored_commands[command].copy()
        
        count = 0
        for pid in pids:
            if self.terminate_process(pid, force):
                count += 1
        
        return count
    
    def get_system_resource_usage(self) -> Dict[str, Any]:
        """
        Get system-wide resource usage.
        
        Returns:
            Dictionary with system resource usage.
        """
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory
        memory = psutil.virtual_memory()
        
        # Disk
        disk = psutil.disk_usage('/')
        
        # Network
        network = psutil.net_io_counters()
        
        # Process count
        process_count = len(psutil.pids())
        
        return {
            "timestamp": time.time(),
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
                "freq": {
                    "current": cpu_freq.current if cpu_freq else None,
                    "min": cpu_freq.min if cpu_freq else None,
                    "max": cpu_freq.max if cpu_freq else None
                }
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            "processes": {
                "count": process_count,
                "monitored": len(self._processes),
                "active_monitored": sum(1 for p in self._processes.values() if p.active)
            }
        }
    
    def cleanup_old_processes(self, max_age: float = DEFAULT_MAX_PROCESS_AGE) -> int:
        """
        Remove old inactive processes from tracking.
        
        Args:
            max_age: Maximum age in seconds.
            
        Returns:
            Number of processes removed.
        """
        current_time = time.time()
        to_remove = []
        
        with self._processes_lock:
            for pid, process_info in self._processes.items():
                if not process_info.active and process_info.ended_at:
                    age = current_time - process_info.ended_at
                    if age > max_age:
                        to_remove.append(pid)
            
            # Remove processes
            for pid in to_remove:
                del self._processes[pid]
        
        return len(to_remove)

# Create global instance
process_monitor = ProcessMonitor()

def start_monitoring():
    """Start process monitoring."""
    process_monitor.start()

def stop_monitoring():
    """Stop process monitoring."""
    process_monitor.stop()

def track_process(pid: int, shell_command: Optional[str] = None,
                monitoring_level: Optional[MonitoringLevel] = None,
                timeout: Optional[float] = None) -> None:
    """Track a process."""
    process_monitor.track_process(pid, shell_command, monitoring_level, timeout)

def get_process_info(pid: int) -> Optional[Dict[str, Any]]:
    """Get process information."""
    return process_monitor.get_process_info(pid)

def get_all_processes(active_only: bool = True) -> List[Dict[str, Any]]:
    """Get all tracked processes."""
    return process_monitor.get_all_processes(active_only)

def get_system_resource_usage() -> Dict[str, Any]:
    """Get system resource usage."""
    return process_monitor.get_system_resource_usage()

def terminate_process(pid: int, force: bool = False) -> bool:
    """Terminate a process."""
    return process_monitor.terminate_process(pid, force)

def register_event_handler(event_type: str, handler: Callable) -> bool:
    """Register an event handler."""
    return process_monitor.register_event_handler(event_type, handler)
