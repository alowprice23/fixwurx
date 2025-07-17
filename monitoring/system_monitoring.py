#!/usr/bin/env python3
"""
System Monitoring Module

This module provides real-time system monitoring capabilities, including metrics collection,
alert thresholds, performance trending, error rate tracking, and service health monitoring.
"""

import os
import sys
import json
import logging
import time
import threading
import queue
import socket
import platform
import psutil
import datetime
import requests
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("system_monitoring.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SystemMonitoring")

class MetricCollector:
    """
    Collects system metrics.
    """
    
    def __init__(self, collect_interval: int = 5):
        """
        Initialize metric collector.
        
        Args:
            collect_interval: Interval in seconds to collect metrics
        """
        self.collect_interval = collect_interval
        self.metrics = {}
        self.running = False
        self.collection_thread = None
        self.stop_event = threading.Event()
        self.callbacks = []
        
        logger.info("Metric collector initialized")
    
    def start(self) -> None:
        """Start collecting metrics."""
        if self.running:
            logger.warning("Metric collector already running")
            return
        
        self.running = True
        self.stop_event.clear()
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("Metric collector started")
    
    def stop(self) -> None:
        """Stop collecting metrics."""
        if not self.running:
            logger.warning("Metric collector not running")
            return
        
        self.running = False
        self.stop_event.set()
        if self.collection_thread:
            self.collection_thread.join(timeout=10)
            if self.collection_thread.is_alive():
                logger.warning("Metric collection thread did not terminate gracefully")
        
        logger.info("Metric collector stopped")
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a callback for metric updates.
        
        Args:
            callback: Callback function that receives the metrics
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            logger.debug(f"Added metric callback: {callback.__name__}")
    
    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Remove a callback.
        
        Args:
            callback: Callback function
            
        Returns:
            Whether the callback was removed
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.debug(f"Removed metric callback: {callback.__name__}")
            return True
        
        return False
    
    def _collection_loop(self) -> None:
        """Collection loop for metrics."""
        while not self.stop_event.is_set():
            try:
                # Collect all metrics
                current_metrics = {
                    "timestamp": time.time(),
                    "system": self._collect_system_metrics(),
                    "cpu": self._collect_cpu_metrics(),
                    "memory": self._collect_memory_metrics(),
                    "disk": self._collect_disk_metrics(),
                    "network": self._collect_network_metrics(),
                    "process": self._collect_process_metrics()
                }
                
                # Update metrics
                self.metrics = current_metrics
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(current_metrics)
                    except Exception as e:
                        logger.error(f"Error in metric callback: {e}")
                
                # Wait for next collection interval
                self.stop_event.wait(self.collect_interval)
            except Exception as e:
                logger.error(f"Error in metric collection loop: {e}")
                self.stop_event.wait(1)  # Wait a bit before retrying
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect system metrics.
        
        Returns:
            System metrics
        """
        return {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "uptime": time.time() - psutil.boot_time()
        }
    
    def _collect_cpu_metrics(self) -> Dict[str, Any]:
        """
        Collect CPU metrics.
        
        Returns:
            CPU metrics
        """
        return {
            "usage_percent": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(),
            "physical_count": psutil.cpu_count(logical=False),
            "load_avg": os.getloadavg() if hasattr(os, "getloadavg") else None
        }
    
    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """
        Collect memory metrics.
        
        Returns:
            Memory metrics
        """
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
            "swap_total": psutil.swap_memory().total,
            "swap_used": psutil.swap_memory().used,
            "swap_percent": psutil.swap_memory().percent
        }
    
    def _collect_disk_metrics(self) -> Dict[str, Any]:
        """
        Collect disk metrics.
        
        Returns:
            Disk metrics
        """
        # Get disk usage for all partitions
        partitions = {}
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                partitions[partition.mountpoint] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent,
                    "fstype": partition.fstype
                }
            except PermissionError:
                # Some partitions may not be accessible
                pass
        
        # Get disk IO statistics
        try:
            disk_io = psutil.disk_io_counters()
            io_metrics = {
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count,
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "read_time": disk_io.read_time,
                "write_time": disk_io.write_time
            }
        except Exception:
            io_metrics = {}
        
        return {
            "partitions": partitions,
            "io": io_metrics
        }
    
    def _collect_network_metrics(self) -> Dict[str, Any]:
        """
        Collect network metrics.
        
        Returns:
            Network metrics
        """
        try:
            network_io = psutil.net_io_counters()
            return {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv,
                "errin": network_io.errin,
                "errout": network_io.errout,
                "dropin": network_io.dropin,
                "dropout": network_io.dropout
            }
        except Exception:
            return {}
    
    def _collect_process_metrics(self) -> Dict[str, Any]:
        """
        Collect process metrics.
        
        Returns:
            Process metrics
        """
        # Get current process
        process = psutil.Process()
        
        # Get process metrics
        return {
            "pid": process.pid,
            "cpu_percent": process.cpu_percent(interval=0.1),
            "memory_percent": process.memory_percent(),
            "memory_rss": process.memory_info().rss,
            "memory_vms": process.memory_info().vms,
            "threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the latest metrics.
        
        Returns:
            Latest metrics
        """
        return self.metrics
    
    def get_historical_metrics(self, metric_path: str, duration: int = 3600) -> List[Tuple[float, Any]]:
        """
        Get historical metrics from the metrics database.
        
        Args:
            metric_path: Path to metric (e.g., "cpu.usage_percent")
            duration: Duration in seconds to look back
            
        Returns:
            List of (timestamp, value) tuples
        """
        # This is a placeholder implementation
        # A real implementation would query a time-series database
        return []

class AlertManager:
    """
    Manages alert thresholds and triggers.
    """
    
    def __init__(self, metric_collector: MetricCollector = None):
        """
        Initialize alert manager.
        
        Args:
            metric_collector: Metric collector
        """
        self.metric_collector = metric_collector
        self.alert_rules = []
        self.alert_history = []
        self.max_history = 1000
        self.alert_callbacks = []
        
        # If metric collector is provided, register callback
        if self.metric_collector:
            self.metric_collector.add_callback(self.check_alerts)
        
        logger.info("Alert manager initialized")
    
    def add_alert_rule(self, rule: Dict[str, Any]) -> None:
        """
        Add an alert rule.
        
        Args:
            rule: Alert rule definition
                {
                    "name": "High CPU Usage",
                    "metric_path": "cpu.usage_percent",
                    "condition": ">",
                    "threshold": 80,
                    "duration": 300,  # in seconds
                    "severity": "warning",
                    "description": "CPU usage is over 80%"
                }
        """
        # Add additional fields to rule
        rule["active"] = False
        rule["triggered_at"] = None
        rule["recovered_at"] = None
        rule["last_checked"] = time.time()
        rule["history"] = []
        
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule['name']}")
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            Whether the rule was removed
        """
        for i, rule in enumerate(self.alert_rules):
            if rule["name"] == rule_name:
                del self.alert_rules[i]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
        
        return False
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a callback for alert events.
        
        Args:
            callback: Callback function that receives the alert event
        """
        if callback not in self.alert_callbacks:
            self.alert_callbacks.append(callback)
            logger.debug(f"Added alert callback: {callback.__name__}")
    
    def remove_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Remove a callback.
        
        Args:
            callback: Callback function
            
        Returns:
            Whether the callback was removed
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            logger.debug(f"Removed alert callback: {callback.__name__}")
            return True
        
        return False
    
    def check_alerts(self, metrics: Dict[str, Any]) -> None:
        """
        Check all alert rules against the current metrics.
        
        Args:
            metrics: Current metrics
        """
        timestamp = metrics.get("timestamp", time.time())
        
        for rule in self.alert_rules:
            try:
                # Get metric value
                metric_path = rule["metric_path"]
                metric_value = self._get_metric_value(metrics, metric_path)
                
                if metric_value is None:
                    continue
                
                # Check condition
                condition = rule["condition"]
                threshold = rule["threshold"]
                
                rule_triggered = self._check_condition(metric_value, condition, threshold)
                
                # Update rule history
                rule["history"].append({
                    "timestamp": timestamp,
                    "value": metric_value,
                    "triggered": rule_triggered
                })
                
                # Trim history
                if len(rule["history"]) > 100:
                    rule["history"] = rule["history"][-100:]
                
                # Check if rule should be activated or deactivated
                duration = rule.get("duration", 0)
                
                if rule_triggered:
                    # Check if rule is already active
                    if not rule["active"]:
                        # Check if condition has been true for the required duration
                        triggered_duration = 0
                        
                        for entry in reversed(rule["history"]):
                            if entry["triggered"]:
                                triggered_duration += self.metric_collector.collect_interval
                            else:
                                break
                        
                        if triggered_duration >= duration:
                            # Activate the rule
                            rule["active"] = True
                            rule["triggered_at"] = timestamp
                            rule["recovered_at"] = None
                            
                            # Create alert event
                            alert_event = {
                                "type": "alert",
                                "rule": rule["name"],
                                "metric_path": metric_path,
                                "metric_value": metric_value,
                                "threshold": threshold,
                                "condition": condition,
                                "severity": rule.get("severity", "warning"),
                                "description": rule.get("description", ""),
                                "timestamp": timestamp
                            }
                            
                            # Add to alert history
                            self.alert_history.append(alert_event)
                            
                            # Trim history
                            if len(self.alert_history) > self.max_history:
                                self.alert_history = self.alert_history[-self.max_history:]
                            
                            # Notify callbacks
                            for callback in self.alert_callbacks:
                                try:
                                    callback(alert_event)
                                except Exception as e:
                                    logger.error(f"Error in alert callback: {e}")
                            
                            logger.warning(f"Alert triggered: {rule['name']} - {metric_value} {condition} {threshold}")
                else:
                    # Check if rule is active
                    if rule["active"]:
                        # Check if condition has been false for the required recovery duration
                        recovery_duration = rule.get("recovery_duration", duration)
                        recovered_duration = 0
                        
                        for entry in reversed(rule["history"]):
                            if not entry["triggered"]:
                                recovered_duration += self.metric_collector.collect_interval
                            else:
                                break
                        
                        if recovered_duration >= recovery_duration:
                            # Deactivate the rule
                            rule["active"] = False
                            rule["recovered_at"] = timestamp
                            
                            # Create recovery event
                            recovery_event = {
                                "type": "recovery",
                                "rule": rule["name"],
                                "metric_path": metric_path,
                                "metric_value": metric_value,
                                "threshold": threshold,
                                "condition": condition,
                                "severity": rule.get("severity", "warning"),
                                "description": rule.get("description", ""),
                                "timestamp": timestamp,
                                "alert_duration": timestamp - rule["triggered_at"] if rule["triggered_at"] else 0
                            }
                            
                            # Add to alert history
                            self.alert_history.append(recovery_event)
                            
                            # Trim history
                            if len(self.alert_history) > self.max_history:
                                self.alert_history = self.alert_history[-self.max_history:]
                            
                            # Notify callbacks
                            for callback in self.alert_callbacks:
                                try:
                                    callback(recovery_event)
                                except Exception as e:
                                    logger.error(f"Error in alert callback: {e}")
                            
                            logger.info(f"Alert recovered: {rule['name']} - {metric_value} {condition} {threshold}")
                
                # Update last checked timestamp
                rule["last_checked"] = timestamp
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    def _get_metric_value(self, metrics: Dict[str, Any], metric_path: str) -> Any:
        """
        Get a metric value from the metrics dictionary using a dot-notation path.
        
        Args:
            metrics: Metrics dictionary
            metric_path: Path to metric (e.g., "cpu.usage_percent")
            
        Returns:
            Metric value, or None if not found
        """
        parts = metric_path.split(".")
        value = metrics
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    def _check_condition(self, value: Any, condition: str, threshold: Any) -> bool:
        """
        Check if a value meets a condition.
        
        Args:
            value: Value to check
            condition: Condition operator (">", ">=", "<", "<=", "==", "!=")
            threshold: Threshold value
            
        Returns:
            Whether the condition is met
        """
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        else:
            logger.error(f"Invalid condition: {condition}")
            return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get active alerts.
        
        Returns:
            List of active alert rules
        """
        return [rule for rule in self.alert_rules if rule["active"]]
    
    def get_alert_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get alert history.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of alert events
        """
        if limit is not None:
            return self.alert_history[-limit:]
        else:
            return self.alert_history

class PerformanceTracker:
    """
    Tracks and analyzes performance trends.
    """
    
    def __init__(self, metric_collector: MetricCollector = None):
        """
        Initialize performance tracker.
        
        Args:
            metric_collector: Metric collector
        """
        self.metric_collector = metric_collector
        self.tracked_metrics = {}
        self.retention_days = 30
        self.db_path = "performance_trends.db"
        
        # If metric collector is provided, register callback
        if self.metric_collector:
            self.metric_collector.add_callback(self.record_metrics)
        
        logger.info("Performance tracker initialized")
    
    def track_metric(self, metric_path: str, description: str = None) -> None:
        """
        Track a metric for performance trending.
        
        Args:
            metric_path: Path to metric (e.g., "cpu.usage_percent")
            description: Metric description
        """
        self.tracked_metrics[metric_path] = {
            "description": description or f"Trend for {metric_path}",
            "last_value": None,
            "last_timestamp": None,
            "data_points": []
        }
        logger.info(f"Tracking metric: {metric_path}")
    
    def untrack_metric(self, metric_path: str) -> bool:
        """
        Stop tracking a metric.
        
        Args:
            metric_path: Path to metric
            
        Returns:
            Whether the metric was untracked
        """
        if metric_path in self.tracked_metrics:
            del self.tracked_metrics[metric_path]
            logger.info(f"Untracked metric: {metric_path}")
            return True
        
        return False
    
    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Record metrics for tracked metrics.
        
        Args:
            metrics: Current metrics
        """
        timestamp = metrics.get("timestamp", time.time())
        
        for metric_path in self.tracked_metrics:
            try:
                # Get metric value
                metric_value = self._get_metric_value(metrics, metric_path)
                
                if metric_value is not None:
                    # Update tracked metric
                    self.tracked_metrics[metric_path]["last_value"] = metric_value
                    self.tracked_metrics[metric_path]["last_timestamp"] = timestamp
                    
                    # Add data point
                    self.tracked_metrics[metric_path]["data_points"].append({
                        "timestamp": timestamp,
                        "value": metric_value
                    })
                    
                    # Trim data points to retain only recent data
                    cutoff = time.time() - (self.retention_days * 86400)
                    data_points = self.tracked_metrics[metric_path]["data_points"]
                    self.tracked_metrics[metric_path]["data_points"] = [
                        point for point in data_points if point["timestamp"] >= cutoff
                    ]
                    
                    # Store in database (placeholder implementation)
                    # A real implementation would store data in a time-series database
            except Exception as e:
                logger.error(f"Error recording metric {metric_path}: {e}")
    
    def _get_metric_value(self, metrics: Dict[str, Any], metric_path: str) -> Any:
        """
        Get a metric value from the metrics dictionary using a dot-notation path.
        
        Args:
            metrics: Metrics dictionary
            metric_path: Path to metric (e.g., "cpu.usage_percent")
            
        Returns:
            Metric value, or None if not found
        """
        parts = metric_path.split(".")
        value = metrics
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    def get_trend(self, metric_path: str, duration: int = 86400, resolution: int = 60) -> Dict[str, Any]:
        """
        Get trend data for a metric.
        
        Args:
            metric_path: Path to metric
            duration: Duration in seconds to look back
            resolution: Resolution in seconds for data points
            
        Returns:
            Trend data
        """
        if metric_path not in self.tracked_metrics:
            return {
                "metric_path": metric_path,
                "data_points": [],
                "trend": None,
                "error": "Metric not tracked"
            }
        
        # Get data points within duration
        cutoff = time.time() - duration
        data_points = [
            point for point in self.tracked_metrics[metric_path]["data_points"]
            if point["timestamp"] >= cutoff
        ]
        
        # Resample data points to requested resolution
        if resolution > 0 and len(data_points) > 0:
            resampled_points = []
            start_time = data_points[0]["timestamp"]
            end_time = data_points[-1]["timestamp"]
            
            for t in np.arange(start_time, end_time + resolution, resolution):
                # Find data points within this time bucket
                bucket_points = [
                    point for point in data_points
                    if t <= point["timestamp"] < t + resolution
                ]
                
                if bucket_points:
                    # Calculate average value
                    avg_value = sum(point["value"] for point in bucket_points) / len(bucket_points)
                    
                    resampled_points.append({
                        "timestamp": t,
                        "value": avg_value
                    })
            
            data_points = resampled_points
        
        # Calculate trend
        trend = None
        
        if len(data_points) > 1:
            # Simple linear regression
            x = np.array([point["timestamp"] for point in data_points])
            y = np.array([point["value"] for point in data_points])
            
            # Normalize x for numerical stability
            x_mean = np.mean(x)
            x = x - x_mean
            
            # Calculate slope and intercept
            slope = np.sum(x * y) / np.sum(x * x)
            intercept = np.mean(y)
            
            # Determine trend direction
            if abs(slope) < 1e-6:
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
        
        return {
            "metric_path": metric_path,
            "data_points": data_points,
            "trend": trend
        }
    
    def plot_trend(self, metric_path: str, duration: int = 86400, output_file: str = None) -> str:
        """
        Plot trend for a metric and save to file.
        
        Args:
            metric_path: Path to metric
            duration: Duration in seconds to look back
            output_file: Output file path
            
        Returns:
            Output file path
        """
        # Get trend data
        trend_data = self.get_trend(metric_path, duration)
        
        if not trend_data["data_points"]:
            logger.warning(f"No data points for metric {metric_path}")
            return None
        
        # Create output file path if not provided
        if output_file is None:
            os.makedirs("trend_plots", exist_ok=True)
            output_file = f"trend_plots/{metric_path.replace('.', '_')}_{int(time.time())}.png"
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        x = [datetime.datetime.fromtimestamp(point["timestamp"]) for point in trend_data["data_points"]]
        y = [point["value"] for point in trend_data["data_points"]]
        
        plt.plot(x, y, "b-")
        plt.scatter(x, y, color="b", alpha=0.5)
        
        # Add trend line
        if len(x) > 1 and trend_data["trend"] is not None:
            z = np.polyfit(
                [point["timestamp"] for point in trend_data["data_points"]],
                [point["value"] for point in trend_data["data_points"]],
                1
            )
            p = np.poly1d(z)
            
            plt.plot(
                x,
                p([point["timestamp"] for point in trend_data["data_points"]]),
                "r--",
                label=f"Trend: {trend_data['trend']}"
            )
        
        # Add labels and title
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Trend for {metric_path}")
        plt.legend()
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file)
        plt.close()
        
        return output_file

class ErrorRateTracker:
    """
    Tracks error rates.
    """
    
    def __init__(self):
        """Initialize error rate tracker."""
        self.error_counts = {}
        self.request_counts = {}
        self.error_history = []
        self.max_history = 1000
        
        logger.info("Error rate tracker initialized")
    
    def record_request(self, service: str, endpoint: str = None, status_code: int = None) -> None:
        """
        Record a request.
        
        Args:
            service: Service name
            endpoint: Endpoint name
            status_code: HTTP status code
        """
        timestamp = time.time()
        
        # Initialize service if needed
        if service not in self.request_counts:
            self.request_counts[service] = {
                "total": 0,
                "endpoints": {}
            }
            self.error_counts[service] = {
                "total": 0,
                "endpoints": {}
            }
        
        # Increment request count
        self.request_counts[service]["total"] += 1
        
        # Initialize endpoint if needed and specified
        if endpoint is not None:
            if endpoint not in self.request_counts[service]["endpoints"]:
                self.request_counts[service]["endpoints"][endpoint] = 0
                self.error_counts[service]["endpoints"][endpoint] = 0
            
            # Increment endpoint request count
            self.request_counts[service]["endpoints"][endpoint] += 1
        
        # Record error if status code is an error
        if status_code is not None and status_code >= 400:
            # Increment error count
            self.error_counts[service]["total"] += 1
            
            if endpoint is not None:
                self.error_counts[service]["endpoints"][endpoint] += 1
            
            # Add to error history
            error_entry = {
                "timestamp": timestamp,
                "service": service,
                "endpoint": endpoint,
                "status_code": status_code
            }
            
            self.error_history.append(error_entry)
            
            # Trim error history
            if len(self.error_history) > self.max_history:
                self.error_history = self.error_history[-self.max_history:]
    
    def get_error_rate(
