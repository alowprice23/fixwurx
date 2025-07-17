"""
FixWurx Auditor Threading Safety Sensor

This module implements a sensor for monitoring threading and concurrency issues,
detecting deadlocks, race conditions, and thread safety violations.
"""

import logging
import time
import threading
import os
import sys
import traceback
import gc
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from collections import defaultdict

from sensor_base import ErrorSensor
from error_report import ErrorReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [ThreadingSafety] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('threading_safety_sensor')


class ThreadingSafetySensor(ErrorSensor):
    """
    Monitors threading and concurrency issues to detect deadlocks and race conditions.
    
    This sensor tracks thread creation, termination, lock acquisition patterns,
    deadlock potential, thread starvation, and other threading safety issues.
    """
    
    def __init__(self, 
                component_name: str = "ThreadingSafety",
                config: Optional[Dict[str, Any]] = None):
        """Initialize the ThreadingSafetySensor."""
        super().__init__(
            sensor_id="threading_safety_sensor",
            component_name=component_name,
            config=config or {}
        )
        
        # Extract configuration values with defaults
        self.check_intervals = {
            "thread_status": self.config.get("thread_status_interval", 30),  # 30 seconds
            "deadlock": self.config.get("deadlock_check_interval", 60),  # 1 minute
            "lock_usage": self.config.get("lock_usage_interval", 120),  # 2 minutes
        }
        
        self.thresholds = {
            "max_threads": self.config.get("max_threads", 100),  # Maximum number of threads
            "thread_lifetime_warning": self.config.get("thread_lifetime_warning", 3600),  # 1 hour
            "lock_hold_time_warning": self.config.get("lock_hold_time_warning", 10),  # 10 seconds
            "contention_ratio_warning": self.config.get("contention_ratio_warning", 0.3),  # 30% contention
            "max_waiting_threads": self.config.get("max_waiting_threads", 5)  # Maximum threads waiting for a lock
        }
        
        # Initialize thread metrics
        self.last_check_times = {check_type: 0 for check_type in self.check_intervals}
        self.metrics = {
            "thread_counts": [],  # [(timestamp, count), ...]
            "thread_history": {},  # {thread_id: {start_time, name, etc.}, ...}
            "active_threads": {},  # {thread_id: {name, status, etc.}, ...}
            "lock_history": defaultdict(list),  # {lock_id: [(acquire_time, release_time, thread_id), ...], ...}
            "current_locks": {},  # {lock_id: {thread_id, acquire_time, etc.}, ...}
            "lock_contentions": defaultdict(int),  # {lock_id: contention_count, ...}
            "deadlock_checks": [],  # [(timestamp, result), ...]
            "thread_exceptions": []  # [(timestamp, thread_id, exception), ...]
        }
        
        # Setup thread monitoring
        self._setup_thread_monitoring()
        
        # Start time
        self.start_time = time.time()
        
        logger.info(f"Initialized ThreadingSafetySensor for {component_name}")
    
    def _setup_thread_monitoring(self) -> None:
        """Set up thread monitoring hooks."""
        # Try to patch threading module to monitor thread creation and termination
        try:
            # Store original Thread.__init__ and __stop methods
            self._original_thread_init = threading.Thread.__init__
            self._original_thread_stop = getattr(threading.Thread, "_Thread__stop", None)
            
            # Create patched methods
            def patched_init(thread_self, *args, **kwargs):
                # Call original __init__
                self._original_thread_init(thread_self, *args, **kwargs)
                # Record thread creation
                self._record_thread_creation(thread_self)
            
            def patched_stop(thread_self, *args, **kwargs):
                # Record thread termination
                self._record_thread_termination(thread_self)
                # Call original __stop
                if self._original_thread_stop:
                    return self._original_thread_stop(thread_self, *args, **kwargs)
            
            # Apply patches if not in lightweight mode
            if not self.config.get("lightweight_mode", False):
                threading.Thread.__init__ = patched_init
                if self._original_thread_stop:
                    setattr(threading.Thread, "_Thread__stop", patched_stop)
                
                logger.info("Thread monitoring hooks installed")
            
            # Also patch lock acquisition if not in lightweight mode
            if not self.config.get("lightweight_mode", False):
                # Store original acquire and release methods
                self._original_lock_acquire = threading.Lock.acquire
                self._original_lock_release = threading.Lock.release
                
                # Create patched methods
                def patched_acquire(lock_self, blocking=True, timeout=-1):
                    # Record lock acquisition attempt
                    start_time = time.time()
                    thread_id = threading.get_ident()
                    
                    # Call original acquire
                    result = self._original_lock_acquire(lock_self, blocking, timeout)
                    
                    # Record successful acquisition
                    if result:
                        self._record_lock_acquisition(lock_self, thread_id, start_time)
                    elif blocking:
                        # Record contention
                        self._record_lock_contention(lock_self, thread_id)
                    
                    return result
                
                def patched_release(lock_self):
                    # Record lock release
                    thread_id = threading.get_ident()
                    self._record_lock_release(lock_self, thread_id)
                    
                    # Call original release
                    return self._original_lock_release(lock_self)
                
                # Apply patches
                threading.Lock.acquire = patched_acquire
                threading.Lock.release = patched_release
                
                logger.info("Lock monitoring hooks installed")
        
        except Exception as e:
            logger.error(f"Failed to set up thread monitoring: {str(e)}")
    
    def _record_thread_creation(self, thread: threading.Thread) -> None:
        """
        Record thread creation event.
        
        Args:
            thread: The thread being created
        """
        try:
            thread_id = thread.ident or id(thread)  # Use thread.ident if available, otherwise use object id
            thread_info = {
                "name": thread.name,
                "daemon": thread.daemon,
                "start_time": time.time(),
                "stack_trace": traceback.format_stack(),
                "creator_thread": threading.current_thread().name
            }
            
            # Store thread info
            self.metrics["thread_history"][thread_id] = thread_info
            self.metrics["active_threads"][thread_id] = thread_info.copy()
            self.metrics["active_threads"][thread_id]["status"] = "created"
            
            logger.debug(f"Thread created: {thread.name} (id: {thread_id})")
            
        except Exception as e:
            logger.error(f"Error recording thread creation: {str(e)}")
    
    def _record_thread_termination(self, thread: threading.Thread) -> None:
        """
        Record thread termination event.
        
        Args:
            thread: The thread being terminated
        """
        try:
            thread_id = thread.ident or id(thread)
            
            # Update thread history
            if thread_id in self.metrics["thread_history"]:
                self.metrics["thread_history"][thread_id]["end_time"] = time.time()
                self.metrics["thread_history"][thread_id]["lifetime"] = (
                    self.metrics["thread_history"][thread_id]["end_time"] - 
                    self.metrics["thread_history"][thread_id]["start_time"]
                )
            
            # Remove from active threads
            if thread_id in self.metrics["active_threads"]:
                del self.metrics["active_threads"][thread_id]
            
            logger.debug(f"Thread terminated: {thread.name} (id: {thread_id})")
            
        except Exception as e:
            logger.error(f"Error recording thread termination: {str(e)}")
    
    def _record_lock_acquisition(self, lock: threading.Lock, thread_id: int, start_time: float) -> None:
        """
        Record lock acquisition event.
        
        Args:
            lock: The lock being acquired
            thread_id: ID of the thread acquiring the lock
            start_time: Time when acquisition was attempted
        """
        try:
            lock_id = id(lock)
            acquire_time = time.time()
            wait_time = acquire_time - start_time
            
            # Record current lock holder
            self.metrics["current_locks"][lock_id] = {
                "thread_id": thread_id,
                "acquire_time": acquire_time,
                "wait_time": wait_time,
                "thread_name": threading.current_thread().name
            }
            
            logger.debug(f"Lock acquired: {lock_id} by thread {thread_id} (wait: {wait_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"Error recording lock acquisition: {str(e)}")
    
    def _record_lock_release(self, lock: threading.Lock, thread_id: int) -> None:
        """
        Record lock release event.
        
        Args:
            lock: The lock being released
            thread_id: ID of the thread releasing the lock
        """
        try:
            lock_id = id(lock)
            release_time = time.time()
            
            # Get acquisition info
            if lock_id in self.metrics["current_locks"]:
                acquire_info = self.metrics["current_locks"][lock_id]
                hold_time = release_time - acquire_info["acquire_time"]
                
                # Record in history
                self.metrics["lock_history"][lock_id].append((
                    acquire_info["acquire_time"],
                    release_time,
                    thread_id,
                    hold_time
                ))
                
                # Keep history limited
                if len(self.metrics["lock_history"][lock_id]) > 100:
                    self.metrics["lock_history"][lock_id] = self.metrics["lock_history"][lock_id][-100:]
                
                # Remove from current locks
                del self.metrics["current_locks"][lock_id]
                
                logger.debug(f"Lock released: {lock_id} by thread {thread_id} (held: {hold_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"Error recording lock release: {str(e)}")
    
    def _record_lock_contention(self, lock: threading.Lock, thread_id: int) -> None:
        """
        Record lock contention event.
        
        Args:
            lock: The lock being contended
            thread_id: ID of the thread attempting to acquire the lock
        """
        try:
            lock_id = id(lock)
            
            # Increment contention count
            self.metrics["lock_contentions"][lock_id] += 1
            
            # Keep contention counts limited
            if len(self.metrics["lock_contentions"]) > 100:
                # Keep only the top 100 contentious locks
                sorted_contentions = sorted(
                    self.metrics["lock_contentions"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:100]
                self.metrics["lock_contentions"] = defaultdict(int, sorted_contentions)
            
            logger.debug(f"Lock contention: {lock_id} by thread {thread_id}")
            
        except Exception as e:
            logger.error(f"Error recording lock contention: {str(e)}")
    
    def monitor(self, data: Any = None) -> List[ErrorReport]:
        """
        Monitor threading and concurrency issues.
        
        Args:
            data: Optional data, unused in this sensor
            
        Returns:
            List of error reports for detected issues
        """
        self.last_check_time = time.time()
        reports = []
        
        # Perform thread status check if needed
        if self.last_check_time - self.last_check_times["thread_status"] >= self.check_intervals["thread_status"]:
            status_reports = self._check_thread_status()
            if status_reports:
                reports.extend(status_reports)
            self.last_check_times["thread_status"] = self.last_check_time
        
        # Perform deadlock check if needed
        if self.last_check_time - self.last_check_times["deadlock"] >= self.check_intervals["deadlock"]:
            deadlock_reports = self._check_for_deadlocks()
            if deadlock_reports:
                reports.extend(deadlock_reports)
            self.last_check_times["deadlock"] = self.last_check_time
        
        # Perform lock usage check if needed
        if self.last_check_time - self.last_check_times["lock_usage"] >= self.check_intervals["lock_usage"]:
            lock_reports = self._check_lock_usage()
            if lock_reports:
                reports.extend(lock_reports)
            self.last_check_times["lock_usage"] = self.last_check_time
        
        return reports
    
    def _check_thread_status(self) -> List[ErrorReport]:
        """
        Check thread status and detect issues.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Get current thread count
            current_threads = threading.enumerate()
            thread_count = len(current_threads)
            
            # Record thread count
            self.metrics["thread_counts"].append((time.time(), thread_count))
            if len(self.metrics["thread_counts"]) > 100:
                self.metrics["thread_counts"] = self.metrics["thread_counts"][-100:]
            
            # Update active threads if not using threading hooks
            if self.config.get("lightweight_mode", False):
                self.metrics["active_threads"] = {}
                for thread in current_threads:
                    thread_id = thread.ident or id(thread)
                    self.metrics["active_threads"][thread_id] = {
                        "name": thread.name,
                        "daemon": thread.daemon,
                        "status": "active" if thread.is_alive() else "inactive",
                        "observed_time": time.time()
                    }
            
            # Check for excessive thread count
            if thread_count > self.thresholds["max_threads"]:
                reports.append(self.report_error(
                    error_type="EXCESSIVE_THREAD_COUNT",
                    severity="HIGH" if thread_count > 2 * self.thresholds["max_threads"] else "MEDIUM",
                    details={
                        "message": f"Excessive thread count: {thread_count} > {self.thresholds['max_threads']}",
                        "thread_count": thread_count,
                        "threshold": self.thresholds["max_threads"]
                    },
                    context={
                        "thread_names": [t.name for t in current_threads[:20]],  # First 20 threads
                        "suggested_action": "Use thread pools or reduce concurrent operations"
                    }
                ))
            
            # Check for long-running threads
            current_time = time.time()
            long_running_threads = []
            
            for thread_id, info in self.metrics["active_threads"].items():
                if "start_time" in info and info.get("status") != "terminated":
                    runtime = current_time - info["start_time"]
                    if runtime > self.thresholds["thread_lifetime_warning"]:
                        long_running_threads.append({
                            "thread_id": thread_id,
                            "name": info["name"],
                            "runtime_hours": runtime / 3600,
                            "start_time": info["start_time"]
                        })
            
            # Report long-running threads
            if long_running_threads:
                reports.append(self.report_error(
                    error_type="LONG_RUNNING_THREADS",
                    severity="MEDIUM",
                    details={
                        "message": f"Detected {len(long_running_threads)} long-running threads",
                        "threshold_hours": self.thresholds["thread_lifetime_warning"] / 3600
                    },
                    context={
                        "threads": long_running_threads,
                        "suggested_action": "Check for threads that should have terminated or implement timeouts"
                    }
                ))
            
            # Check for thread growth rate
            if len(self.metrics["thread_counts"]) >= 5:
                # Get the 5 most recent counts
                recent_counts = [c[1] for c in self.metrics["thread_counts"][-5:]]
                oldest_count = recent_counts[0]
                newest_count = recent_counts[-1]
                
                # Calculate growth rate
                if oldest_count > 0:
                    growth_pct = (newest_count - oldest_count) / oldest_count * 100
                    
                    # Significant growth in short time
                    if growth_pct > 30:  # More than 30% growth
                        reports.append(self.report_error(
                            error_type="RAPID_THREAD_GROWTH",
                            severity="MEDIUM",
                            details={
                                "message": f"Thread count growing rapidly: {growth_pct:.1f}% increase",
                                "growth_percentage": growth_pct,
                                "initial_count": oldest_count,
                                "current_count": newest_count
                            },
                            context={
                                "recent_counts": recent_counts,
                                "suggested_action": "Check for thread leaks or use a thread pool"
                            }
                        ))
            
        except Exception as e:
            logger.error(f"Error in thread status check: {str(e)}")
        
        return reports
    
    def _check_for_deadlocks(self) -> List[ErrorReport]:
        """
        Check for potential deadlocks.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Build lock dependency graph
            lock_graph = {}
            waiting_threads = {}
            
            # In a real implementation, we would analyze actual lock wait patterns
            # For this demo version, we'll use a simple check based on current locks
            
            # Check for threads waiting too long for locks
            current_time = time.time()
            for lock_id, lock_info in self.metrics["current_locks"].items():
                hold_time = current_time - lock_info["acquire_time"]
                
                # Long-held locks might indicate issues
                if hold_time > self.thresholds["lock_hold_time_warning"]:
                    reports.append(self.report_error(
                        error_type="LONG_LOCK_HOLD_TIME",
                        severity="MEDIUM",
                        details={
                            "message": f"Lock held for {hold_time:.1f}s > {self.thresholds['lock_hold_time_warning']}s",
                            "lock_id": lock_id,
                            "hold_time_seconds": hold_time,
                            "threshold": self.thresholds["lock_hold_time_warning"],
                            "thread_id": lock_info["thread_id"],
                            "thread_name": lock_info["thread_name"]
                        },
                        context={
                            "suggested_action": "Check for operations performed while holding locks or use fine-grained locking"
                        }
                    ))
            
            # Record deadlock check result
            self.metrics["deadlock_checks"].append((current_time, len(reports) > 0))
            if len(self.metrics["deadlock_checks"]) > 50:
                self.metrics["deadlock_checks"] = self.metrics["deadlock_checks"][-50:]
            
        except Exception as e:
            logger.error(f"Error in deadlock check: {str(e)}")
        
        return reports
    
    def _check_lock_usage(self) -> List[ErrorReport]:
        """
        Check lock usage patterns and detect issues.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Check for highly contended locks
            highly_contended = []
            
            for lock_id, contention_count in self.metrics["lock_contentions"].items():
                # Calculate contention ratio if we have history
                contention_ratio = 0
                if lock_id in self.metrics["lock_history"] and self.metrics["lock_history"][lock_id]:
                    total_acquisitions = len(self.metrics["lock_history"][lock_id]) + contention_count
                    if total_acquisitions > 0:
                        contention_ratio = contention_count / total_acquisitions
                
                if contention_ratio > self.thresholds["contention_ratio_warning"]:
                    highly_contended.append({
                        "lock_id": lock_id,
                        "contention_count": contention_count,
                        "contention_ratio": contention_ratio
                    })
            
            # Report highly contended locks
            if highly_contended:
                reports.append(self.report_error(
                    error_type="HIGH_LOCK_CONTENTION",
                    severity="MEDIUM",
                    details={
                        "message": f"Detected {len(highly_contended)} highly contended locks",
                        "threshold": self.thresholds["contention_ratio_warning"]
                    },
                    context={
                        "contended_locks": highly_contended,
                        "suggested_action": "Use finer-grained locking, reduce lock scope, or consider lock-free alternatives"
                    }
                ))
            
            # Check for inefficient lock usage patterns
            # In a real implementation, we would do more sophisticated analysis
            
        except Exception as e:
            logger.error(f"Error in lock usage check: {str(e)}")
        
        return reports
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the sensor and monitored component."""
        try:
            # Get latest thread count
            if self.metrics["thread_counts"]:
                thread_count = self.metrics["thread_counts"][-1][1]
            else:
                thread_count = len(threading.enumerate())
            
            # Count active locks
            active_lock_count = len(self.metrics["current_locks"])
            
            # Count recent deadlock warnings
            recent_deadlock_warnings = sum(1 for check in self.metrics["deadlock_checks"][-10:] if check[1]) if self.metrics["deadlock_checks"] else 0
            
            # Calculate lock contention
            total_contentions = sum(self.metrics["lock_contentions"].values())
            
            # Calculate health score (0-100)
            # Components:
            # - Thread count health (30 points)
            # - Lock contention health (30 points)
            # - Deadlock warnings (40 points)
            
            thread_health = 30 * (1 - min(1, thread_count / self.thresholds["max_threads"]))
            
            # Lock contention health
            contention_health = 30
            if self.metrics["lock_history"]:
                total_acquisitions = sum(len(history) for history in self.metrics["lock_history"].values())
                if total_acquisitions > 0 and total_contentions > 0:
                    contention_ratio = total_contentions / (total_acquisitions + total_contentions)
                    contention_health = 30 * (1 - min(1, contention_ratio / self.thresholds["contention_ratio_warning"]))
            
            # Deadlock warning health
            deadlock_health = 40
            if self.metrics["deadlock_checks"]:
                recent_checks = min(10, len(self.metrics["deadlock_checks"]))
                if recent_checks > 0:
                    warning_ratio = recent_deadlock_warnings / recent_checks
                    deadlock_health = 40 * (1 - warning_ratio)
            
            # Overall health score
            health_score = thread_health + contention_health + deadlock_health
            
            return {
                "sensor_id": self.sensor_id,
                "component_name": self.component_name,
                "last_check_time": self.last_check_time,
                "health_score": health_score,
                "thread_count": thread_count,
                "active_locks": active_lock_count,
                "lock_contentions": total_contentions,
                "deadlock_warnings": recent_deadlock_warnings,
                "monitored_since": self.start_time
            }
            
        except Exception as e:
            logger.error(f"Error in get_status: {str(e)}")
            return {
                "sensor_id": self.sensor_id,
                "component_name": self.component_name,
                "last_check_time": self.last_check_time,
                "health_score": 0,  # Assume worst health if we can't calculate
                "error": str(e)
            }
    
    def __del__(self):
        """Clean up resources when the sensor is destroyed."""
        # Restore original threading methods if we patched them
        try:
            if not self.config.get("lightweight_mode", False):
                if hasattr(self, '_original_thread_init'):
                    threading.Thread.__init__ = self._original_thread_init
                
                if hasattr(self, '_original_thread_stop') and self._original_thread_stop:
                    setattr(threading.Thread, "_Thread__stop", self._original_thread_stop)
                
                if hasattr(self, '_original_lock_acquire'):
                    threading.Lock.acquire = self._original_lock_acquire
                
                if hasattr(self, '_original_lock_release'):
                    threading.Lock.release = self._original_lock_release
                
                logger.info("Thread monitoring hooks removed")
        except Exception as e:
            logger.error(f"Error restoring original threading methods: {str(e)}")


# Factory function to create a sensor instance
def create_threading_safety_sensor(config: Optional[Dict[str, Any]] = None) -> ThreadingSafetySensor:
    """
    Create and initialize a threading safety sensor.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized ThreadingSafetySensor
    """
    return ThreadingSafetySensor(config=config)
