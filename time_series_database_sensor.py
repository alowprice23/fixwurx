"""
FixWurx Auditor Time Series Database Sensor

This module implements a sensor for monitoring the time series database component,
ensuring data integrity, appropriate retention policies, and optimal performance.
"""

import logging
import time
import datetime
import math
import random
import os
import json
import statistics
from typing import Dict, List, Set, Any, Optional, Union, Tuple, Callable

from sensor_base import ErrorSensor
from error_report import ErrorReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [TimeSeriesDBSensor] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('time_series_database_sensor')


class TimeSeriesDatabaseSensor(ErrorSensor):
    """
    Monitors time series database health, performance, and data integrity.
    
    This sensor tracks query performance, data insertion rates, time-based indexing,
    data retention policies, and detects gaps or anomalies in time series data.
    """
    
    def __init__(self, 
                component_name: str = "TimeSeriesDatabase",
                config: Optional[Dict[str, Any]] = None):
        """Initialize the TimeSeriesDatabaseSensor."""
        super().__init__(
            sensor_id="time_series_database_sensor",
            component_name=component_name,
            config=config or {}
        )
        
        # Extract configuration values with defaults
        self.check_intervals = {
            "performance": self.config.get("performance_check_interval", 120),  # 2 minutes
            "integrity": self.config.get("integrity_check_interval", 300),  # 5 minutes
            "retention": self.config.get("retention_check_interval", 3600),  # 1 hour
        }
        
        self.thresholds = {
            "max_query_time_ms": self.config.get("max_query_time_ms", 200),
            "max_insertion_time_ms": self.config.get("max_insertion_time_ms", 50),
            "max_data_gap_percentage": self.config.get("max_data_gap_percentage", 5.0),
            "min_data_points_per_hour": self.config.get("min_data_points_per_hour", 10),
            "max_storage_growth_per_day": self.config.get("max_storage_growth_per_day", 500)  # MB
        }
        
        # Database connection settings
        self.db_interface = self.config.get("db_interface", None)
        self.mock_mode = self.config.get("mock_mode", True)
        
        # Database metrics and tracking
        self.last_check_times = {check_type: 0 for check_type in self.check_intervals}
        self.metrics = {
            "query_times": [],  # [(timestamp, query_time_ms), ...]
            "insertion_times": [],  # [(timestamp, insertion_time_ms), ...]
            "data_point_counts": [],  # [(timestamp, count), ...]
            "storage_sizes": [],  # [(timestamp, size_mb), ...]
            "retention_metrics": {
                "retention_policies": {},
                "archived_data": {},
                "purged_data": {}
            },
            "data_gaps": [],  # [(series_name, start_time, end_time, gap_percentage), ...]
            "index_health": 1.0,  # 0.0-1.0 score
        }
        
        # Series to monitor (could be dynamically configured)
        self.monitored_series = self.config.get("monitored_series", [
            "system_metrics",
            "error_rates",
            "performance_benchmarks",
            "resource_usage",
            "query_statistics"
        ])
        
        # Initialize mock data if in mock mode
        if self.mock_mode:
            self._initialize_mock_data()
        
        logger.info(f"Initialized TimeSeriesDatabaseSensor for {component_name}")
    
    def monitor(self, data: Any = None) -> List[ErrorReport]:
        """
        Monitor time series database metrics.
        
        Args:
            data: Optional data for monitoring, such as query statistics
            
        Returns:
            List of error reports for detected issues
        """
        self.last_check_time = time.time()
        reports = []
        
        # Update metrics if data was provided
        if data and isinstance(data, dict):
            self._update_metrics_from_data(data)
        
        # Perform performance check if needed
        if self.last_check_time - self.last_check_times["performance"] >= self.check_intervals["performance"]:
            performance_reports = self._check_performance()
            if performance_reports:
                reports.extend(performance_reports)
            self.last_check_times["performance"] = self.last_check_time
        
        # Perform integrity check if needed
        if self.last_check_time - self.last_check_times["integrity"] >= self.check_intervals["integrity"]:
            integrity_reports = self._check_data_integrity()
            if integrity_reports:
                reports.extend(integrity_reports)
            self.last_check_times["integrity"] = self.last_check_time
        
        # Perform retention check if needed
        if self.last_check_time - self.last_check_times["retention"] >= self.check_intervals["retention"]:
            retention_reports = self._check_retention_policies()
            if retention_reports:
                reports.extend(retention_reports)
            self.last_check_times["retention"] = self.last_check_time
        
        return reports
    
    def _initialize_mock_data(self) -> None:
        """Initialize mock data for testing in mock mode."""
        # Generate mock query times
        current_time = time.time()
        for i in range(100):
            timestamp = current_time - (100 - i) * 60  # Every minute
            query_time = random.uniform(10, 150)  # 10-150ms
            self.metrics["query_times"].append((timestamp, query_time))
        
        # Generate mock insertion times
        for i in range(100):
            timestamp = current_time - (100 - i) * 60  # Every minute
            insertion_time = random.uniform(5, 30)  # 5-30ms
            self.metrics["insertion_times"].append((timestamp, insertion_time))
        
        # Generate mock data point counts
        for i in range(24):
            timestamp = current_time - (24 - i) * 3600  # Every hour
            count = random.randint(800, 1200)  # 800-1200 points
            self.metrics["data_point_counts"].append((timestamp, count))
        
        # Generate mock storage sizes
        for i in range(30):
            timestamp = current_time - (30 - i) * 86400  # Every day
            size_mb = 1000 + i * random.uniform(100, 400)  # Growing by 100-400 MB per day
            self.metrics["storage_sizes"].append((timestamp, size_mb))
        
        # Generate mock retention policies
        self.metrics["retention_metrics"]["retention_policies"] = {
            "raw_data": {
                "duration_days": 30,
                "downsampling": None
            },
            "hourly_aggregates": {
                "duration_days": 90,
                "downsampling": "1h"
            },
            "daily_aggregates": {
                "duration_days": 365,
                "downsampling": "1d"
            }
        }
        
        # Generate mock archived data
        self.metrics["retention_metrics"]["archived_data"] = {
            "last_archive_time": current_time - 86400,  # Yesterday
            "archive_count": 15,
            "oldest_archive": current_time - 15 * 86400,
            "newest_archive": current_time - 86400
        }
        
        # Generate mock purged data
        self.metrics["retention_metrics"]["purged_data"] = {
            "last_purge_time": current_time - 86400 * 2,  # 2 days ago
            "purged_points": 250000,
            "purged_series": ["old_metrics", "deprecated_stats"]
        }
        
        # Generate mock data gaps
        if random.random() < 0.3:  # 30% chance to have data gaps
            series_name = random.choice(self.monitored_series)
            start_time = current_time - random.randint(1, 10) * 3600
            end_time = start_time + random.randint(1, 5) * 60
            gap_percentage = random.uniform(1, 10)
            
            self.metrics["data_gaps"].append((
                series_name, start_time, end_time, gap_percentage
            ))
        
        logger.info("Initialized mock data for time series database sensor")
    
    def _update_metrics_from_data(self, data: Dict[str, Any]) -> None:
        """
        Update metrics from provided data.
        
        Args:
            data: Dictionary containing time series database metrics
        """
        current_time = time.time()
        
        # Update query times if provided
        if "query_time_ms" in data:
            self.metrics["query_times"].append((current_time, data["query_time_ms"]))
            # Keep only the most recent 1000 measurements
            if len(self.metrics["query_times"]) > 1000:
                self.metrics["query_times"] = self.metrics["query_times"][-1000:]
        
        # Update insertion times if provided
        if "insertion_time_ms" in data:
            self.metrics["insertion_times"].append((current_time, data["insertion_time_ms"]))
            # Keep only the most recent 1000 measurements
            if len(self.metrics["insertion_times"]) > 1000:
                self.metrics["insertion_times"] = self.metrics["insertion_times"][-1000:]
        
        # Update data point count if provided
        if "data_point_count" in data:
            self.metrics["data_point_counts"].append((current_time, data["data_point_count"]))
            # Keep only the most recent 168 measurements (1 week hourly)
            if len(self.metrics["data_point_counts"]) > 168:
                self.metrics["data_point_counts"] = self.metrics["data_point_counts"][-168:]
        
        # Update storage size if provided
        if "storage_size_mb" in data:
            self.metrics["storage_sizes"].append((current_time, data["storage_size_mb"]))
            # Keep only the most recent 90 measurements (90 days)
            if len(self.metrics["storage_sizes"]) > 90:
                self.metrics["storage_sizes"] = self.metrics["storage_sizes"][-90:]
        
        # Update data gaps if provided
        if "data_gaps" in data and isinstance(data["data_gaps"], list):
            for gap in data["data_gaps"]:
                if isinstance(gap, tuple) and len(gap) == 4:
                    self.metrics["data_gaps"].append(gap)
            
            # Keep only the most recent 100 gaps
            if len(self.metrics["data_gaps"]) > 100:
                self.metrics["data_gaps"] = self.metrics["data_gaps"][-100:]
        
        # Update index health if provided
        if "index_health" in data:
            self.metrics["index_health"] = data["index_health"]
        
        # Update retention metrics if provided
        if "retention_policies" in data:
            self.metrics["retention_metrics"]["retention_policies"] = data["retention_policies"]
        if "archived_data" in data:
            self.metrics["retention_metrics"]["archived_data"] = data["archived_data"]
        if "purged_data" in data:
            self.metrics["retention_metrics"]["purged_data"] = data["purged_data"]
    
    def _check_performance(self) -> List[ErrorReport]:
        """
        Check time series database performance metrics.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # In a real implementation, this would execute some benchmark queries
            # and measure their performance. In mock mode, we'll use the mock data.
            
            # If in mock mode, simulate some performance data
            if self.mock_mode:
                current_time = time.time()
                query_time = random.uniform(10, 250)  # 10-250ms
                self.metrics["query_times"].append((current_time, query_time))
                
                insertion_time = random.uniform(5, 60)  # 5-60ms
                self.metrics["insertion_times"].append((current_time, insertion_time))
            
            # Check recent query times
            recent_query_times = [t[1] for t in self.metrics["query_times"][-20:]]
            if recent_query_times:
                avg_query_time = sum(recent_query_times) / len(recent_query_times)
                max_query_time = max(recent_query_times)
                
                # Check if average query time is above threshold
                if avg_query_time > self.thresholds["max_query_time_ms"]:
                    reports.append(self.report_error(
                        error_type="HIGH_QUERY_LATENCY",
                        severity="MEDIUM",
                        details={
                            "message": f"Average query time is above threshold: {avg_query_time:.1f}ms > {self.thresholds['max_query_time_ms']}ms",
                            "avg_query_time_ms": avg_query_time,
                            "max_query_time_ms": max_query_time,
                            "threshold": self.thresholds["max_query_time_ms"]
                        },
                        context={
                            "recent_query_times": self.metrics["query_times"][-10:],
                            "suggested_action": "Check for complex queries, optimize indexes, or increase database resources"
                        }
                    ))
                
                # Check for spikes in query time
                if max_query_time > self.thresholds["max_query_time_ms"] * 2:
                    reports.append(self.report_error(
                        error_type="QUERY_LATENCY_SPIKE",
                        severity="MEDIUM",
                        details={
                            "message": f"Detected spike in query time: {max_query_time:.1f}ms > {self.thresholds['max_query_time_ms'] * 2}ms",
                            "max_query_time_ms": max_query_time,
                            "avg_query_time_ms": avg_query_time,
                            "threshold": self.thresholds["max_query_time_ms"] * 2
                        },
                        context={
                            "recent_query_times": self.metrics["query_times"][-10:],
                            "suggested_action": "Investigate possible concurrent resource-intensive operations"
                        }
                    ))
            
            # Check recent insertion times
            recent_insertion_times = [t[1] for t in self.metrics["insertion_times"][-20:]]
            if recent_insertion_times:
                avg_insertion_time = sum(recent_insertion_times) / len(recent_insertion_times)
                max_insertion_time = max(recent_insertion_times)
                
                # Check if average insertion time is above threshold
                if avg_insertion_time > self.thresholds["max_insertion_time_ms"]:
                    reports.append(self.report_error(
                        error_type="HIGH_INSERTION_LATENCY",
                        severity="MEDIUM",
                        details={
                            "message": f"Average insertion time is above threshold: {avg_insertion_time:.1f}ms > {self.thresholds['max_insertion_time_ms']}ms",
                            "avg_insertion_time_ms": avg_insertion_time,
                            "max_insertion_time_ms": max_insertion_time,
                            "threshold": self.thresholds["max_insertion_time_ms"]
                        },
                        context={
                            "recent_insertion_times": self.metrics["insertion_times"][-10:],
                            "suggested_action": "Check for index rebuild operations, high write load, or storage issues"
                        }
                    ))
            
            # Calculate growth rate of storage
            if len(self.metrics["storage_sizes"]) >= 2:
                newest = self.metrics["storage_sizes"][-1]
                oldest = self.metrics["storage_sizes"][0]
                
                days_diff = (newest[0] - oldest[0]) / 86400  # Convert seconds to days
                if days_diff >= 1:  # Only calculate if we have at least 1 day of data
                    size_diff = newest[1] - oldest[1]  # MB difference
                    daily_growth = size_diff / days_diff
                    
                    # Check if daily growth exceeds threshold
                    if daily_growth > self.thresholds["max_storage_growth_per_day"]:
                        reports.append(self.report_error(
                            error_type="HIGH_STORAGE_GROWTH_RATE",
                            severity="MEDIUM",
                            details={
                                "message": f"Time series database storage growth rate is high: {daily_growth:.1f}MB/day > {self.thresholds['max_storage_growth_per_day']}MB/day",
                                "growth_rate_mb_per_day": daily_growth,
                                "threshold": self.thresholds["max_storage_growth_per_day"],
                                "current_size_mb": newest[1]
                            },
                            context={
                                "days_measured": days_diff,
                                "suggested_action": "Review data retention policies, implement downsampling, or increase storage capacity"
                            }
                        ))
            
            # Check for query performance trends (detecting gradual slowdown)
            if len(self.metrics["query_times"]) >= 50:
                trend = self._calculate_trend([t[1] for t in self.metrics["query_times"][-50:]])
                if trend > 0.3:  # Significant upward trend
                    reports.append(self.report_error(
                        error_type="QUERY_PERFORMANCE_DEGRADATION",
                        severity="LOW",
                        details={
                            "message": "Query performance is gradually degrading over time",
                            "trend_coefficient": trend,
                            "recent_avg_query_time": sum(t[1] for t in self.metrics["query_times"][-10:]) / 10
                        },
                        context={
                            "suggested_action": "Perform database maintenance, optimize indexes, or add caching"
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Error in performance check: {str(e)}")
        
        return reports
    
    def _check_data_integrity(self) -> List[ErrorReport]:
        """
        Check time series data integrity.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # In a real implementation, this would query the database to check for gaps,
            # inconsistencies, and other data integrity issues.
            # In mock mode, we'll simulate some issues.
            
            # If in mock mode, simulate some data integrity issues
            if self.mock_mode:
                current_time = time.time()
                
                # Simulate data point count (how many points were recorded in the last hour)
                points_last_hour = random.randint(20, 120)
                self.metrics["data_point_counts"].append((current_time, points_last_hour))
                
                # Occasionally simulate a data gap
                if random.random() < 0.1:  # 10% chance
                    series_name = random.choice(self.monitored_series)
                    start_time = current_time - random.randint(1, 5) * 3600
                    end_time = start_time + random.randint(10, 60) * 60
                    gap_percentage = random.uniform(1, 15)
                    
                    self.metrics["data_gaps"].append((
                        series_name, start_time, end_time, gap_percentage
                    ))
            
            # Check for data gaps
            current_gaps = []
            for gap in self.metrics["data_gaps"]:
                series_name, start_time, end_time, gap_percentage = gap
                
                # Only consider recent gaps (within the last day)
                if time.time() - end_time < 86400:
                    current_gaps.append(gap)
                    
                    # Report gaps that exceed threshold
                    if gap_percentage > self.thresholds["max_data_gap_percentage"]:
                        gap_duration_min = (end_time - start_time) / 60
                        reports.append(self.report_error(
                            error_type="TIME_SERIES_DATA_GAP",
                            severity="HIGH" if gap_percentage > 20 else "MEDIUM",
                            details={
                                "message": f"Detected {gap_percentage:.1f}% data gap in '{series_name}' series",
                                "series_name": series_name,
                                "gap_percentage": gap_percentage,
                                "gap_duration_minutes": gap_duration_min,
                                "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
                                "end_time": datetime.datetime.fromtimestamp(end_time).isoformat(),
                                "threshold": self.thresholds["max_data_gap_percentage"]
                            },
                            context={
                                "suggested_action": "Check data collection processes and database connectivity"
                            }
                        ))
            
            # Update current gaps list
            self.metrics["data_gaps"] = current_gaps
            
            # Check for insufficient data points
            recent_point_counts = [c[1] for c in self.metrics["data_point_counts"][-24:] if time.time() - c[0] < 86400]
            if recent_point_counts:
                hourly_avg = sum(recent_point_counts) / len(recent_point_counts)
                min_count = min(recent_point_counts)
                
                # Check if minimum hourly count is below threshold
                if min_count < self.thresholds["min_data_points_per_hour"]:
                    reports.append(self.report_error(
                        error_type="INSUFFICIENT_DATA_POINTS",
                        severity="MEDIUM",
                        details={
                            "message": f"Insufficient data points recorded: {min_count} < {self.thresholds['min_data_points_per_hour']} points per hour",
                            "min_points_per_hour": min_count,
                            "avg_points_per_hour": hourly_avg,
                            "threshold": self.thresholds["min_data_points_per_hour"]
                        },
                        context={
                            "recent_counts": self.metrics["data_point_counts"][-10:],
                            "suggested_action": "Check data collection frequency and ensure all metrics are being recorded"
                        }
                    ))
            
            # Check for index health
            if self.metrics["index_health"] < 0.9:
                reports.append(self.report_error(
                    error_type="TIME_INDEX_DEGRADATION",
                    severity="HIGH" if self.metrics["index_health"] < 0.7 else "MEDIUM",
                    details={
                        "message": f"Time series index health is degraded: {self.metrics['index_health']:.2f} score",
                        "index_health_score": self.metrics["index_health"],
                        "threshold": 0.9
                    },
                    context={
                        "suggested_action": "Run index optimization, check for fragmentation, or rebuild indexes"
                    }
                ))
            
            # Check for data consistency across time series
            # In a real implementation, this would check for correlations between related time series
            # For this demo, we'll simulate a consistency check
            if self.mock_mode and random.random() < 0.05:  # 5% chance of consistency issue
                reports.append(self.report_error(
                    error_type="TIME_SERIES_CONSISTENCY_ERROR",
                    severity="MEDIUM",
                    details={
                        "message": "Detected inconsistency between related time series data",
                        "series": ["system_metrics", "resource_usage"],
                        "time_range": f"{(time.time() - 3600):.0f} - {time.time():.0f}"
                    },
                    context={
                        "description": "Resource usage shows spikes that don't correlate with system metrics",
                        "suggested_action": "Verify data collection synchronization across metrics"
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error in data integrity check: {str(e)}")
        
        return reports
    
    def _check_retention_policies(self) -> List[ErrorReport]:
        """
        Check time series database retention policies.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # In a real implementation, this would verify that retention policies
            # are properly configured and enforced, checking for data that should
            # be archived or purged.
            # In mock mode, we'll simulate some retention issues.
            
            # If in mock mode, simulate retention policy state
            if self.mock_mode:
                current_time = time.time()
                
                # Simulate archive data
                self.metrics["retention_metrics"]["archived_data"] = {
                    "last_archive_time": current_time - random.randint(1, 10) * 86400,
                    "archive_count": random.randint(10, 30),
                    "oldest_archive": current_time - random.randint(30, 90) * 86400,
                    "newest_archive": current_time - random.randint(1, 10) * 86400
                }
                
                # Simulate purge data
                self.metrics["retention_metrics"]["purged_data"] = {
                    "last_purge_time": current_time - random.randint(1, 30) * 86400,
                    "purged_points": random.randint(10000, 1000000),
                    "purged_series": random.sample(self.monitored_series, random.randint(0, 2))
                }
            
            # Check if archiving is happening recently enough
            archive_data = self.metrics["retention_metrics"]["archived_data"]
            if "last_archive_time" in archive_data:
                days_since_archive = (time.time() - archive_data["last_archive_time"]) / 86400
                
                if days_since_archive > 7:  # More than 7 days since last archive
                    reports.append(self.report_error(
                        error_type="ARCHIVE_POLICY_DELAYED",
                        severity="MEDIUM" if days_since_archive > 14 else "LOW",
                        details={
                            "message": f"Time series data hasn't been archived in {days_since_archive:.1f} days",
                            "days_since_archive": days_since_archive,
                            "last_archive_time": datetime.datetime.fromtimestamp(archive_data["last_archive_time"]).isoformat(),
                            "threshold": 7
                        },
                        context={
                            "archive_count": archive_data.get("archive_count", "Unknown"),
                            "suggested_action": "Check archiving process and schedule"
                        }
                    ))
            
            # Check if purging is happening
            purge_data = self.metrics["retention_metrics"]["purged_data"]
            if "last_purge_time" in purge_data:
                days_since_purge = (time.time() - purge_data["last_purge_time"]) / 86400
                
                if days_since_purge > 30:  # More than 30 days since last purge
                    reports.append(self.report_error(
                        error_type="PURGE_POLICY_DELAYED",
                        severity="MEDIUM" if days_since_purge > 60 else "LOW",
                        details={
                            "message": f"Time series data hasn't been purged in {days_since_purge:.1f} days",
                            "days_since_purge": days_since_purge,
                            "last_purge_time": datetime.datetime.fromtimestamp(purge_data["last_purge_time"]).isoformat(),
                            "threshold": 30
                        },
                        context={
                            "purged_points": purge_data.get("purged_points", "Unknown"),
                            "suggested_action": "Verify purging process and retention policies"
                        }
                    ))
            
            # Check storage growth rate for indication of retention policy issues
            if len(self.metrics["storage_sizes"]) >= 7:  # At least a week of data
                # Calculate 7-day growth rate
                week_ago_size = None
                for ts, size in self.metrics["storage_sizes"]:
                    if time.time() - ts >= 7 * 86400:
                        week_ago_size = size
                        break
                
                if week_ago_size is not None:
                    current_size = self.metrics["storage_sizes"][-1][1]
                    weekly_growth_mb = current_size - week_ago_size
                    
                    # If growing very fast, might indicate retention policy issues
                    if weekly_growth_mb > self.thresholds["max_storage_growth_per_day"] * 7 * 1.5:
                        reports.append(self.report_error(
                            error_type="EXCESSIVE_STORAGE_GROWTH",
                            severity="HIGH",
                            details={
                                "message": f"Time series database growing too rapidly: {weekly_growth_mb:.1f}MB in 7 days",
                                "weekly_growth_mb": weekly_growth_mb,
                                "current_size_mb": current_size,
                                "threshold": self.thresholds["max_storage_growth_per_day"] * 7 * 1.5
                            },
                            context={
                                "retention_policies": self.metrics["retention_metrics"]["retention_policies"],
                                "suggested_action": "Review retention policies, add downsampling, or implement more aggressive purging"
                            }
                        ))
            
            # Check for missing retention policies for critical series
            retention_policies = self.metrics["retention_metrics"]["retention_policies"]
            missing_policies = []
            
            for series in self.monitored_series:
                if series not in retention_policies and "default" not in retention_policies:
                    missing_policies.append(series)
            
            if missing_policies:
                reports.append(self.report_error(
                    error_type="MISSING_RETENTION_POLICIES",
                    severity="MEDIUM",
                    details={
                        "message": f"{len(missing_policies)} time series lack retention policies",
                        "affected_series": missing_policies
                    },
                    context={
                        "suggested_action": "Define retention policies for all monitored series or create a default policy"
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error in retention policy check: {str(e)}")
        
        return reports
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend using linear regression.
        
        Args:
            values: List of values to analyze
            
        Returns:
            Trend coefficient (positive = increasing, negative = decreasing)
        """
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        y = values
        
        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Calculate slope
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # Normalize to a -1 to 1 range based on the range of y values
        y_range = max(y) - min(y) if max(y) != min(y) else 1.0
        normalized_slope = slope * (n / y_range)
        
        # Clamp to -1 to 1 range
        return max(-1.0, min(1.0, normalized_slope))
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the sensor and monitored component."""
        # Calculate recent averages
        recent_query_times = [t[1] for t in self.metrics["query_times"][-20:]]
        avg_query_time = sum(recent_query_times) / len(recent_query_times) if recent_query_times else 0
        
        recent_insertion_times = [t[1] for t in self.metrics["insertion_times"][-20:]]
        avg_insertion_time = sum(recent_insertion_times) / len(recent_insertion_times) if recent_insertion_times else 0
        
        # Count recent data gaps
        recent_gaps = len([g for g in self.metrics["data_gaps"] 
                         if time.time() - g[2] < 86400])  # gaps in the last 24 hours
        
        # Calculate a health score
        query_score = 100 - min(100, (avg_query_time / self.thresholds["max_query_time_ms"]) * 50)
        insertion_score = 100 - min(100, (avg_insertion_time / self.thresholds["max_insertion_time_ms"]) * 50)
        gap_penalty = min(50, recent_gaps * 10)
        index_score = self.metrics["index_health"] * 100
        
        # Combined score
        health_score = (query_score * 0.3 + insertion_score * 0.2 + index_score * 0.4) - gap_penalty
        health_score = max(0, min(100, health_score))
        
        return {
            "sensor_id": self.sensor_id,
            "component_name": self.component_name,
            "last_check_time": self.last_check_time,
            "health_score": health_score,
            "avg_query_time_ms": avg_query_time,
            "avg_insertion_time_ms": avg_insertion_time,
            "data_gaps_24h": recent_gaps,
            "index_health": self.metrics["index_health"],
            "storage_mb": self.metrics["storage_sizes"][-1][1] if self.metrics["storage_sizes"] else "Unknown"
        }


# Factory function to create a sensor instance
def create_time_series_database_sensor(config: Optional[Dict[str, Any]] = None) -> TimeSeriesDatabaseSensor:
    """
    Create and initialize a time series database sensor.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized TimeSeriesDatabaseSensor
    """
    return TimeSeriesDatabaseSensor(config=config)
