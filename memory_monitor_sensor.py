"""
FixWurx Auditor Memory Monitor Sensor

This module implements a sensor for monitoring memory usage and detecting potential
memory leaks, allocation issues, and other memory-related problems.
"""

import logging
import time
import math
import os
import psutil
import gc
import sys
import tracemalloc
from typing import Dict, List, Set, Any, Optional, Union, Tuple

from sensor_base import ErrorSensor
from error_report import ErrorReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [MemoryMonitor] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('memory_monitor_sensor')


class MemoryMonitorSensor(ErrorSensor):
    """
    Monitors system and process memory usage to detect leaks and allocation issues.
    
    This sensor tracks memory usage trends, identifies potential leaks, monitors
    allocation patterns, and suggests memory optimization strategies.
    """
    
    def __init__(self, 
                component_name: str = "SystemMemory",
                config: Optional[Dict[str, Any]] = None):
        """Initialize the MemoryMonitorSensor."""
        super().__init__(
            sensor_id="memory_monitor_sensor",
            component_name=component_name,
            config=config or {}
        )
        
        # Extract configuration values with defaults
        self.check_intervals = {
            "usage": self.config.get("usage_check_interval", 60),  # 1 minute
            "leak": self.config.get("leak_check_interval", 300),  # 5 minutes
            "gc": self.config.get("gc_check_interval", 600),  # 10 minutes
        }
        
        self.thresholds = {
            "max_process_memory_pct": self.config.get("max_process_memory_pct", 30),  # Max 30% of system memory
            "max_system_memory_pct": self.config.get("max_system_memory_pct", 85),  # Max 85% system memory usage
            "leak_growth_mb_per_hour": self.config.get("leak_growth_mb_per_hour", 10),  # 10 MB/hour growth may indicate leak
            "max_fragmentation_pct": self.config.get("max_fragmentation_pct", 15),  # 15% fragmentation tolerance
            "max_objects_growth_pct": self.config.get("max_objects_growth_pct", 20)  # 20% growth in object count
        }
        
        # Tracemalloc settings
        self.use_tracemalloc = self.config.get("use_tracemalloc", False)
        self.tracemalloc_snapshot_count = self.config.get("tracemalloc_snapshot_count", 5)
        
        # Initialize memory metrics
        self.last_check_times = {check_type: 0 for check_type in self.check_intervals}
        self.metrics = {
            "process_memory": [],  # [(timestamp, memory_info), ...]
            "system_memory": [],  # [(timestamp, memory_info), ...]
            "object_counts": [],  # [(timestamp, type_counts), ...]
            "gc_stats": [],  # [(timestamp, gc_info), ...]
            "tracemalloc_snapshots": []  # [(timestamp, snapshot_summary), ...]
        }
        
        # Start tracemalloc if enabled
        if self.use_tracemalloc:
            try:
                tracemalloc.start()
                logger.info("Tracemalloc started")
            except Exception as e:
                logger.error(f"Failed to start tracemalloc: {str(e)}")
                self.use_tracemalloc = False
        
        # Process info
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        
        logger.info(f"Initialized MemoryMonitorSensor for {component_name}")
    
    def monitor(self, data: Any = None) -> List[ErrorReport]:
        """
        Monitor memory usage and detect potential issues.
        
        Args:
            data: Optional data, unused in this sensor
            
        Returns:
            List of error reports for detected issues
        """
        self.last_check_time = time.time()
        reports = []
        
        # Perform usage check if needed
        if self.last_check_time - self.last_check_times["usage"] >= self.check_intervals["usage"]:
            usage_reports = self._check_memory_usage()
            if usage_reports:
                reports.extend(usage_reports)
            self.last_check_times["usage"] = self.last_check_time
        
        # Perform leak detection if needed
        if self.last_check_time - self.last_check_times["leak"] >= self.check_intervals["leak"]:
            leak_reports = self._check_for_leaks()
            if leak_reports:
                reports.extend(leak_reports)
            self.last_check_times["leak"] = self.last_check_time
        
        # Perform GC check if needed
        if self.last_check_time - self.last_check_times["gc"] >= self.check_intervals["gc"]:
            gc_reports = self._check_gc_health()
            if gc_reports:
                reports.extend(gc_reports)
            self.last_check_times["gc"] = self.last_check_time
        
        return reports
    
    def _get_process_memory(self) -> Dict[str, Any]:
        """
        Get current process memory usage.
        
        Returns:
            Dictionary with memory information
        """
        try:
            mem_info = self.process.memory_info()
            
            # Calculate memory usage in MB
            rss_mb = mem_info.rss / (1024 * 1024)  # Resident Set Size
            vms_mb = mem_info.vms / (1024 * 1024)  # Virtual Memory Size
            
            # Get system memory info for context
            sys_mem = psutil.virtual_memory()
            total_mb = sys_mem.total / (1024 * 1024)
            
            # Calculate percentage of system memory used by this process
            process_pct = (rss_mb / total_mb) * 100
            
            return {
                "rss_mb": rss_mb,
                "vms_mb": vms_mb,
                "process_pct": process_pct,
                "pid": self.process.pid,
                "num_threads": self.process.num_threads(),
                "cpu_percent": self.process.cpu_percent()
            }
            
        except Exception as e:
            logger.error(f"Error getting process memory: {str(e)}")
            return {
                "rss_mb": 0,
                "vms_mb": 0,
                "process_pct": 0,
                "error": str(e)
            }
    
    def _get_system_memory(self) -> Dict[str, Any]:
        """
        Get current system memory usage.
        
        Returns:
            Dictionary with memory information
        """
        try:
            sys_mem = psutil.virtual_memory()
            
            # Calculate memory values in MB
            total_mb = sys_mem.total / (1024 * 1024)
            available_mb = sys_mem.available / (1024 * 1024)
            used_mb = sys_mem.used / (1024 * 1024)
            
            # Get swap info
            swap = psutil.swap_memory()
            swap_total_mb = swap.total / (1024 * 1024)
            swap_used_mb = swap.used / (1024 * 1024)
            
            return {
                "total_mb": total_mb,
                "available_mb": available_mb,
                "used_mb": used_mb,
                "percent": sys_mem.percent,
                "swap_total_mb": swap_total_mb,
                "swap_used_mb": swap_used_mb,
                "swap_percent": swap.percent
            }
            
        except Exception as e:
            logger.error(f"Error getting system memory: {str(e)}")
            return {
                "total_mb": 0,
                "available_mb": 0,
                "used_mb": 0,
                "percent": 0,
                "error": str(e)
            }
    
    def _get_object_counts(self) -> Dict[str, Any]:
        """
        Get counts of Python objects by type.
        
        Returns:
            Dictionary with object count information
        """
        try:
            type_counts = {}
            total_objects = 0
            
            # Count objects by type
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                total_objects += 1
            
            # Get the top 10 types by count
            top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "total_objects": total_objects,
                "type_counts": dict(top_types),
                "gc_objects": len(gc.get_objects())
            }
            
        except Exception as e:
            logger.error(f"Error getting object counts: {str(e)}")
            return {
                "total_objects": 0,
                "type_counts": {},
                "error": str(e)
            }
    
    def _get_gc_stats(self) -> Dict[str, Any]:
        """
        Get garbage collection statistics.
        
        Returns:
            Dictionary with GC statistics
        """
        try:
            # Get GC counts (generations)
            counts = gc.get_count()
            
            # Get GC threshold settings
            thresholds = gc.get_threshold()
            
            # Force collection to get stats
            gc.collect()
            
            # Get statistics
            stats = {
                "counts": counts,
                "thresholds": thresholds,
                "enabled": gc.isenabled(),
                "objects": len(gc.get_objects()),
                "garbage": len(gc.garbage)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting GC stats: {str(e)}")
            return {
                "error": str(e)
            }
    
    def _get_tracemalloc_snapshot(self) -> Dict[str, Any]:
        """
        Get tracemalloc snapshot information if enabled.
        
        Returns:
            Dictionary with tracemalloc statistics or empty dict if not enabled
        """
        if not self.use_tracemalloc:
            return {}
        
        try:
            # Take a snapshot
            snapshot = tracemalloc.take_snapshot()
            
            # Get top statistics
            top_stats = snapshot.statistics('lineno')
            
            # Format the top 10 allocations
            top_10 = []
            for stat in top_stats[:10]:
                frame = stat.traceback[0]
                top_10.append({
                    "filename": frame.filename,
                    "lineno": frame.lineno,
                    "size_kb": stat.size / 1024,
                    "count": stat.count
                })
            
            return {
                "top_allocations": top_10,
                "total_traced_memory": tracemalloc.get_traced_memory()[0] / (1024 * 1024)  # MB
            }
            
        except Exception as e:
            logger.error(f"Error getting tracemalloc snapshot: {str(e)}")
            return {
                "error": str(e)
            }
    
    def _check_memory_usage(self) -> List[ErrorReport]:
        """
        Check current memory usage and detect issues.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Get current memory metrics
            process_mem = self._get_process_memory()
            system_mem = self._get_system_memory()
            object_counts = self._get_object_counts()
            
            # Record metrics
            current_time = time.time()
            self.metrics["process_memory"].append((current_time, process_mem))
            self.metrics["system_memory"].append((current_time, system_mem))
            self.metrics["object_counts"].append((current_time, object_counts))
            
            # Keep limited history
            if len(self.metrics["process_memory"]) > 100:
                self.metrics["process_memory"] = self.metrics["process_memory"][-100:]
            if len(self.metrics["system_memory"]) > 100:
                self.metrics["system_memory"] = self.metrics["system_memory"][-100:]
            if len(self.metrics["object_counts"]) > 50:
                self.metrics["object_counts"] = self.metrics["object_counts"][-50:]
            
            # Check if process is using too much memory
            if process_mem["process_pct"] > self.thresholds["max_process_memory_pct"]:
                reports.append(self.report_error(
                    error_type="HIGH_PROCESS_MEMORY_USAGE",
                    severity="MEDIUM",
                    details={
                        "message": f"Process memory usage is above threshold: {process_mem['process_pct']:.1f}% > {self.thresholds['max_process_memory_pct']}% of system memory",
                        "rss_mb": process_mem["rss_mb"],
                        "vms_mb": process_mem["vms_mb"],
                        "process_pct": process_mem["process_pct"],
                        "threshold": self.thresholds["max_process_memory_pct"]
                    },
                    context={
                        "pid": process_mem["pid"],
                        "num_threads": process_mem.get("num_threads", "N/A"),
                        "cpu_percent": process_mem.get("cpu_percent", "N/A"),
                        "uptime_hrs": (current_time - self.start_time) / 3600,
                        "suggested_action": "Check for memory leaks, optimize memory usage, or increase available memory"
                    }
                ))
            
            # Check if system memory is too high
            if system_mem["percent"] > self.thresholds["max_system_memory_pct"]:
                reports.append(self.report_error(
                    error_type="HIGH_SYSTEM_MEMORY_USAGE",
                    severity="HIGH" if system_mem["percent"] > 95 else "MEDIUM",
                    details={
                        "message": f"System memory usage is above threshold: {system_mem['percent']:.1f}% > {self.thresholds['max_system_memory_pct']}%",
                        "used_mb": system_mem["used_mb"],
                        "available_mb": system_mem["available_mb"],
                        "percent": system_mem["percent"],
                        "threshold": self.thresholds["max_system_memory_pct"]
                    },
                    context={
                        "swap_used_mb": system_mem.get("swap_used_mb", "N/A"),
                        "swap_percent": system_mem.get("swap_percent", "N/A"),
                        "suggested_action": "Free system memory, close unnecessary applications, or increase available memory"
                    }
                ))
            
            # Check if swap usage is high
            if system_mem.get("swap_percent", 0) > 50:
                reports.append(self.report_error(
                    error_type="HIGH_SWAP_USAGE",
                    severity="MEDIUM",
                    details={
                        "message": f"High swap memory usage: {system_mem['swap_percent']:.1f}% of swap space",
                        "swap_used_mb": system_mem["swap_used_mb"],
                        "swap_total_mb": system_mem["swap_total_mb"],
                        "swap_percent": system_mem["swap_percent"]
                    },
                    context={
                        "system_memory_pct": system_mem["percent"],
                        "suggested_action": "Increase physical memory or optimize memory usage to reduce swap utilization"
                    }
                ))
            
            # Optional: add tracemalloc snapshot if enabled
            if self.use_tracemalloc:
                tracemalloc_data = self._get_tracemalloc_snapshot()
                if tracemalloc_data and not "error" in tracemalloc_data:
                    self.metrics["tracemalloc_snapshots"].append((current_time, tracemalloc_data))
                    if len(self.metrics["tracemalloc_snapshots"]) > self.tracemalloc_snapshot_count:
                        self.metrics["tracemalloc_snapshots"] = self.metrics["tracemalloc_snapshots"][-self.tracemalloc_snapshot_count:]
            
        except Exception as e:
            logger.error(f"Error in memory usage check: {str(e)}")
        
        return reports
    
    def _check_for_leaks(self) -> List[ErrorReport]:
        """
        Check for potential memory leaks.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Need at least two measurements to check for leaks
            if len(self.metrics["process_memory"]) < 2:
                return reports
            
            # Get oldest and newest measurements
            oldest = self.metrics["process_memory"][0]
            newest = self.metrics["process_memory"][-1]
            
            # Calculate time difference in hours
            time_diff_hours = (newest[0] - oldest[0]) / 3600
            
            # Need at least 10 minutes of data
            if time_diff_hours < (10 / 60):
                return reports
            
            # Calculate RSS growth rate
            rss_diff = newest[1]["rss_mb"] - oldest[1]["rss_mb"]
            rss_growth_per_hour = rss_diff / time_diff_hours
            
            # Check for significant memory growth
            if rss_growth_per_hour > self.thresholds["leak_growth_mb_per_hour"]:
                # Get additional object count data if available
                object_growth_desc = ""
                if len(self.metrics["object_counts"]) >= 2:
                    oldest_count = self.metrics["object_counts"][0][1]["total_objects"]
                    newest_count = self.metrics["object_counts"][-1][1]["total_objects"]
                    object_diff = newest_count - oldest_count
                    object_growth_pct = (object_diff / oldest_count * 100) if oldest_count > 0 else 0
                    
                    object_growth_desc = f" Object count increased by {object_growth_pct:.1f}% ({object_diff} objects)."
                
                reports.append(self.report_error(
                    error_type="POTENTIAL_MEMORY_LEAK",
                    severity="HIGH" if rss_growth_per_hour > 3 * self.thresholds["leak_growth_mb_per_hour"] else "MEDIUM",
                    details={
                        "message": f"Memory usage growing at {rss_growth_per_hour:.1f} MB/hour, which exceeds the threshold of {self.thresholds['leak_growth_mb_per_hour']} MB/hour.{object_growth_desc}",
                        "growth_rate_mb_per_hour": rss_growth_per_hour,
                        "total_growth_mb": rss_diff,
                        "time_period_hours": time_diff_hours,
                        "threshold": self.thresholds["leak_growth_mb_per_hour"]
                    },
                    context={
                        "current_rss_mb": newest[1]["rss_mb"],
                        "object_count": self.metrics["object_counts"][-1][1]["total_objects"] if self.metrics["object_counts"] else "N/A",
                        "uptime_hrs": (time.time() - self.start_time) / 3600,
                        "suggested_action": "Check for unclosed resources, circular references, or objects that aren't being garbage collected"
                    }
                ))
            
            # Check for object count growth
            if len(self.metrics["object_counts"]) >= 2:
                oldest_obj = self.metrics["object_counts"][0][1]["total_objects"]
                newest_obj = self.metrics["object_counts"][-1][1]["total_objects"]
                
                if oldest_obj > 0:
                    growth_pct = (newest_obj - oldest_obj) / oldest_obj * 100
                    
                    if growth_pct > self.thresholds["max_objects_growth_pct"]:
                        # Get the types that grew the most
                        growing_types = []
                        
                        if "type_counts" in self.metrics["object_counts"][0][1] and "type_counts" in self.metrics["object_counts"][-1][1]:
                            old_counts = self.metrics["object_counts"][0][1]["type_counts"]
                            new_counts = self.metrics["object_counts"][-1][1]["type_counts"]
                            
                            # Find types that grew significantly
                            for type_name, count in new_counts.items():
                                old_count = old_counts.get(type_name, 0)
                                if old_count > 0:
                                    type_growth_pct = (count - old_count) / old_count * 100
                                    if type_growth_pct > 50 and (count - old_count) > 100:  # Only include significant growth
                                        growing_types.append({
                                            "type": type_name,
                                            "growth_pct": type_growth_pct,
                                            "old_count": old_count,
                                            "new_count": count
                                        })
                        
                        # Sort by growth percentage
                        growing_types.sort(key=lambda x: x["growth_pct"], reverse=True)
                        
                        reports.append(self.report_error(
                            error_type="OBJECT_COUNT_GROWTH",
                            severity="MEDIUM",
                            details={
                                "message": f"Object count growing at {growth_pct:.1f}%, which exceeds the threshold of {self.thresholds['max_objects_growth_pct']}%",
                                "growth_pct": growth_pct,
                                "total_growth": newest_obj - oldest_obj,
                                "current_objects": newest_obj,
                                "threshold": self.thresholds["max_objects_growth_pct"]
                            },
                            context={
                                "growing_types": growing_types[:5],  # Top 5 growing types
                                "uptime_hrs": (time.time() - self.start_time) / 3600,
                                "suggested_action": "Check for accumulating objects or collections that aren't being cleared"
                            }
                        ))
            
            # Check tracemalloc data if available
            if len(self.metrics["tracemalloc_snapshots"]) >= 2:
                oldest_trace = self.metrics["tracemalloc_snapshots"][0][1]
                newest_trace = self.metrics["tracemalloc_snapshots"][-1][1]
                
                if "total_traced_memory" in oldest_trace and "total_traced_memory" in newest_trace:
                    trace_diff_mb = newest_trace["total_traced_memory"] - oldest_trace["total_traced_memory"]
                    trace_growth_per_hour = trace_diff_mb / time_diff_hours
                    
                    if trace_growth_per_hour > self.thresholds["leak_growth_mb_per_hour"]:
                        reports.append(self.report_error(
                            error_type="TRACEMALLOC_MEMORY_GROWTH",
                            severity="MEDIUM",
                            details={
                                "message": f"Traced memory growing at {trace_growth_per_hour:.1f} MB/hour",
                                "growth_rate_mb_per_hour": trace_growth_per_hour,
                                "total_growth_mb": trace_diff_mb,
                                "threshold": self.thresholds["leak_growth_mb_per_hour"]
                            },
                            context={
                                "top_allocations": newest_trace.get("top_allocations", []),
                                "suggested_action": "Examine top memory allocations and check for accumulating objects"
                            }
                        ))
            
        except Exception as e:
            logger.error(f"Error in leak detection: {str(e)}")
        
        return reports
    
    def _check_gc_health(self) -> List[ErrorReport]:
        """
        Check garbage collection health.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Get current GC stats
            gc_stats = self._get_gc_stats()
            
            # Record metrics
            current_time = time.time()
            self.metrics["gc_stats"].append((current_time, gc_stats))
            
            # Keep limited history
            if len(self.metrics["gc_stats"]) > 20:
                self.metrics["gc_stats"] = self.metrics["gc_stats"][-20:]
            
            # Check if GC is enabled
            if not gc_stats.get("enabled", True):
                reports.append(self.report_error(
                    error_type="GC_DISABLED",
                    severity="HIGH",
                    details={
                        "message": "Garbage collection is disabled",
                    },
                    context={
                        "suggested_action": "Enable garbage collection with gc.enable()"
                    }
                ))
            
            # Check for high garbage count
            if gc_stats.get("garbage", 0) > 100:
                reports.append(self.report_error(
                    error_type="HIGH_GC_GARBAGE_COUNT",
                    severity="MEDIUM",
                    details={
                        "message": f"High number of uncollectable objects: {gc_stats['garbage']}",
                        "garbage_count": gc_stats["garbage"]
                    },
                    context={
                        "suggested_action": "Check for circular references that are preventing garbage collection"
                    }
                ))
            
            # Check for frequent collections in higher generations
            if len(self.metrics["gc_stats"]) >= 2:
                prev_stats = self.metrics["gc_stats"][-2][1]
                curr_stats = gc_stats
                
                if "counts" in prev_stats and "counts" in curr_stats:
                    gen2_diff = curr_stats["counts"][2] - prev_stats["counts"][2]
                    
                    # If gen2 collections have happened frequently, that's a sign of memory pressure
                    time_diff = current_time - self.metrics["gc_stats"][-2][0]
                    if gen2_diff > 0 and time_diff < 300:  # At least one gen2 collection in last 5 minutes
                        reports.append(self.report_error(
                            error_type="FREQUENT_FULL_GC",
                            severity="LOW",
                            details={
                                "message": f"Frequent full garbage collections ({gen2_diff} in the last {time_diff:.0f} seconds)",
                                "gen2_collections": gen2_diff,
                                "time_period_sec": time_diff
                            },
                            context={
                                "collection_counts": curr_stats["counts"],
                                "suggested_action": "Optimize memory usage to reduce garbage collection overhead"
                            }
                        ))
            
        except Exception as e:
            logger.error(f"Error in GC health check: {str(e)}")
        
        return reports
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the sensor and monitored component."""
        try:
            # Get latest memory metrics
            if not self.metrics["process_memory"]:
                process_mem = self._get_process_memory()
            else:
                process_mem = self.metrics["process_memory"][-1][1]
                
            if not self.metrics["system_memory"]:
                system_mem = self._get_system_memory()
            else:
                system_mem = self.metrics["system_memory"][-1][1]
            
            # Calculate memory growth rate if enough data
            growth_rate_mb_per_hour = 0
            if len(self.metrics["process_memory"]) >= 2:
                oldest = self.metrics["process_memory"][0]
                newest = self.metrics["process_memory"][-1]
                time_diff_hours = max(0.001, (newest[0] - oldest[0]) / 3600)  # Avoid division by zero
                rss_diff = newest[1]["rss_mb"] - oldest[1]["rss_mb"]
                growth_rate_mb_per_hour = rss_diff / time_diff_hours
            
            # Calculate health score (0-100)
            # Components: 
            # - Process memory usage (40 points)
            # - System memory usage (20 points)
            # - Memory growth rate (40 points)
            
            process_score = 40 * (1 - min(1, process_mem["process_pct"] / self.thresholds["max_process_memory_pct"]))
            system_score = 20 * (1 - min(1, system_mem["percent"] / self.thresholds["max_system_memory_pct"]))
            growth_score = 40 * (1 - min(1, abs(growth_rate_mb_per_hour) / self.thresholds["leak_growth_mb_per_hour"]))
            
            health_score = process_score + system_score + growth_score
            
            return {
                "sensor_id": self.sensor_id,
                "component_name": self.component_name,
                "last_check_time": self.last_check_time,
                "health_score": health_score,
                "process_memory_mb": process_mem["rss_mb"],
                "process_memory_pct": process_mem["process_pct"],
                "system_memory_pct": system_mem["percent"],
                "growth_rate_mb_per_hour": growth_rate_mb_per_hour,
                "uptime_hrs": (time.time() - self.start_time) / 3600
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
        if self.use_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()


# Factory function to create a sensor instance
def create_memory_monitor_sensor(config: Optional[Dict[str, Any]] = None) -> MemoryMonitorSensor:
    """
    Create and initialize a memory monitor sensor.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized MemoryMonitorSensor
    """
    return MemoryMonitorSensor(config=config)
