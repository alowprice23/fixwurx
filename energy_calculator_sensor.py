"""
FixWurx Auditor Energy Calculator Sensor

This module implements a sensor for monitoring and analyzing the energy efficiency
of the auditor system, tracking computational resource usage, and estimating
the energy impact of debugging operations.
"""

import logging
import time
import math
import random
import os
import psutil
from typing import Dict, List, Set, Any, Optional, Union, Tuple

from sensor_base import ErrorSensor
from error_report import ErrorReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [EnergyCalculator] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('energy_calculator_sensor')


class EnergyCalculatorSensor(ErrorSensor):
    """
    Monitors energy usage and efficiency metrics for the auditor system.
    
    This sensor tracks CPU, memory, and I/O operations to estimate energy consumption,
    provides efficiency metrics, and suggests optimizations to reduce computational
    overhead of debugging operations.
    """
    
    def __init__(self, 
                component_name: str = "SystemEnergy",
                config: Optional[Dict[str, Any]] = None):
        """Initialize the EnergyCalculatorSensor."""
        super().__init__(
            sensor_id="energy_calculator_sensor",
            component_name=component_name,
            config=config or {}
        )
        
        # Extract configuration values with defaults
        self.check_intervals = {
            "usage": self.config.get("usage_check_interval", 30),  # 30 seconds
            "trend": self.config.get("trend_check_interval", 300),  # 5 minutes
            "efficiency": self.config.get("efficiency_check_interval", 600),  # 10 minutes
        }
        
        self.thresholds = {
            "max_cpu_percent": self.config.get("max_cpu_percent", 80),
            "max_memory_percent": self.config.get("max_memory_percent", 75),
            "max_io_rate_mb": self.config.get("max_io_rate_mb", 50),
            "min_energy_efficiency": self.config.get("min_energy_efficiency", 0.6),
            "max_watt_hours": self.config.get("max_watt_hours", 5.0)
        }
        
        # Energy coefficients
        self.energy_coefficients = {
            "cpu_watt_per_percent": self.config.get("cpu_watt_per_percent", 0.015),
            "memory_watt_per_gb": self.config.get("memory_watt_per_gb", 0.05),
            "io_watt_per_mb": self.config.get("io_watt_per_mb", 0.001),
            "base_watt": self.config.get("base_watt", 0.5)
        }
        
        # Initialize metrics
        self.last_check_times = {check_type: 0 for check_type in self.check_intervals}
        self.metrics = {
            "cpu_usage": [],  # [(timestamp, percent), ...]
            "memory_usage": [],  # [(timestamp, percent), ...]
            "io_rates": [],  # [(timestamp, mb_per_sec), ...]
            "energy_consumption": [],  # [(timestamp, watt_hours), ...]
            "energy_efficiency": [],  # [(timestamp, efficiency_score), ...]
            "operations_metrics": {
                "bug_fixes": 0,
                "energy_per_fix": 0.0,
                "optimizations_applied": 0
            }
        }
        
        self.cumulative_energy_wh = 0.0
        self.start_time = time.time()
        self.last_io_stats = self._get_io_stats()
        
        logger.info(f"Initialized EnergyCalculatorSensor for {component_name}")
    
    def monitor(self, data: Any = None) -> List[ErrorReport]:
        """
        Monitor energy usage and efficiency metrics.
        
        Args:
            data: Optional data for monitoring, such as bug fix counts
            
        Returns:
            List of error reports for detected issues
        """
        self.last_check_time = time.time()
        reports = []
        
        # Update metrics if provided
        if data and isinstance(data, dict):
            if "bug_fixes" in data:
                self.metrics["operations_metrics"]["bug_fixes"] = data["bug_fixes"]
            if "optimizations_applied" in data:
                self.metrics["operations_metrics"]["optimizations_applied"] = data["optimizations_applied"]
        
        # Perform usage check if needed
        if self.last_check_time - self.last_check_times["usage"] >= self.check_intervals["usage"]:
            usage_reports = self._check_usage()
            if usage_reports:
                reports.extend(usage_reports)
            self.last_check_times["usage"] = self.last_check_time
        
        # Perform trend check if needed
        if self.last_check_time - self.last_check_times["trend"] >= self.check_intervals["trend"]:
            trend_reports = self._check_trends()
            if trend_reports:
                reports.extend(trend_reports)
            self.last_check_times["trend"] = self.last_check_time
        
        # Perform efficiency check if needed
        if self.last_check_time - self.last_check_times["efficiency"] >= self.check_intervals["efficiency"]:
            efficiency_reports = self._check_efficiency()
            if efficiency_reports:
                reports.extend(efficiency_reports)
            self.last_check_times["efficiency"] = self.last_check_time
        
        return reports
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage as a percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception as e:
            logger.error(f"Error getting CPU usage: {str(e)}")
            # Fallback to simulate CPU usage
            return random.uniform(10, 70)
    
    def _get_memory_usage(self) -> Tuple[float, float]:
        """
        Get memory usage as a percentage and in GB.
        
        Returns:
            Tuple of (percent, gb)
        """
        try:
            memory = psutil.virtual_memory()
            percent = memory.percent
            gb = memory.used / (1024 * 1024 * 1024)  # Convert bytes to GB
            return percent, gb
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            # Fallback to simulate memory usage
            percent = random.uniform(20, 60)
            gb = random.uniform(1, 4)
            return percent, gb
    
    def _get_io_stats(self) -> Dict[str, int]:
        """
        Get disk I/O statistics.
        
        Returns:
            Dictionary with read_bytes and write_bytes
        """
        try:
            io_counters = psutil.disk_io_counters()
            if io_counters:
                return {
                    "read_bytes": io_counters.read_bytes,
                    "write_bytes": io_counters.write_bytes
                }
        except Exception as e:
            logger.error(f"Error getting I/O stats: {str(e)}")
        
        # Fallback to simulate I/O stats
        return {
            "read_bytes": random.randint(1000000, 5000000),
            "write_bytes": random.randint(1000000, 5000000)
        }
    
    def _calculate_io_rate(self) -> float:
        """
        Calculate I/O rate in MB/s based on previous reading.
        
        Returns:
            I/O rate in MB/s
        """
        current_stats = self._get_io_stats()
        
        if not self.last_io_stats:
            self.last_io_stats = current_stats
            return 0.0
        
        # Calculate bytes transferred since last check
        read_diff = current_stats["read_bytes"] - self.last_io_stats["read_bytes"]
        write_diff = current_stats["write_bytes"] - self.last_io_stats["write_bytes"]
        total_bytes = read_diff + write_diff
        
        # Calculate time difference
        time_diff = time.time() - self.last_check_times["usage"]
        if time_diff <= 0:
            return 0.0
        
        # Calculate rate in MB/s
        mb_per_sec = total_bytes / (1024 * 1024) / time_diff
        
        # Update last stats
        self.last_io_stats = current_stats
        
        return mb_per_sec
    
    def _calculate_energy_consumption(self, 
                                    cpu_percent: float, 
                                    memory_gb: float, 
                                    io_mb_per_sec: float, 
                                    duration_sec: float) -> float:
        """
        Calculate energy consumption in watt-hours.
        
        Args:
            cpu_percent: CPU usage percentage
            memory_gb: Memory usage in GB
            io_mb_per_sec: I/O rate in MB/s
            duration_sec: Duration in seconds
            
        Returns:
            Energy consumption in watt-hours
        """
        # Calculate power usage in watts
        cpu_watts = cpu_percent * self.energy_coefficients["cpu_watt_per_percent"]
        memory_watts = memory_gb * self.energy_coefficients["memory_watt_per_gb"]
        io_watts = io_mb_per_sec * self.energy_coefficients["io_watt_per_mb"]
        base_watts = self.energy_coefficients["base_watt"]
        
        total_watts = cpu_watts + memory_watts + io_watts + base_watts
        
        # Convert to watt-hours
        watt_hours = total_watts * (duration_sec / 3600)
        
        return watt_hours
    
    def _calculate_energy_efficiency(self) -> float:
        """
        Calculate energy efficiency score (0-1).
        
        Returns:
            Efficiency score
        """
        # Get latest metrics
        if not self.metrics["cpu_usage"] or not self.metrics["memory_usage"]:
            return 1.0  # Default to perfect efficiency if no data
        
        # Get average resource usage
        avg_cpu = sum(u[1] for u in self.metrics["cpu_usage"][-10:]) / min(10, len(self.metrics["cpu_usage"]))
        avg_memory = sum(u[1] for u in self.metrics["memory_usage"][-10:]) / min(10, len(self.metrics["memory_usage"]))
        
        # Calculate efficiency based on resource usage
        # Higher resource usage = lower efficiency
        cpu_efficiency = 1.0 - (avg_cpu / 100)
        memory_efficiency = 1.0 - (avg_memory / 100)
        
        # Calculate overall efficiency (weighted average)
        efficiency = 0.6 * cpu_efficiency + 0.4 * memory_efficiency
        
        # Adjust based on operations metrics if available
        bug_fixes = self.metrics["operations_metrics"]["bug_fixes"]
        if bug_fixes > 0 and self.cumulative_energy_wh > 0:
            energy_per_fix = self.cumulative_energy_wh / bug_fixes
            self.metrics["operations_metrics"]["energy_per_fix"] = energy_per_fix
            
            # Adjust efficiency based on energy per fix
            # Lower energy per fix = higher efficiency
            energy_factor = max(0, 1.0 - (energy_per_fix / self.thresholds["max_watt_hours"]))
            efficiency = 0.7 * efficiency + 0.3 * energy_factor
        
        return max(0, min(1, efficiency))
    
    def _check_usage(self) -> List[ErrorReport]:
        """
        Check current resource usage and energy consumption.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Get resource usage
            cpu_percent = self._get_cpu_usage()
            memory_percent, memory_gb = self._get_memory_usage()
            io_mb_per_sec = self._calculate_io_rate()
            
            # Record metrics
            current_time = time.time()
            self.metrics["cpu_usage"].append((current_time, cpu_percent))
            self.metrics["memory_usage"].append((current_time, memory_percent))
            self.metrics["io_rates"].append((current_time, io_mb_per_sec))
            
            # Keep only the last 100 measurements
            if len(self.metrics["cpu_usage"]) > 100:
                self.metrics["cpu_usage"] = self.metrics["cpu_usage"][-100:]
            if len(self.metrics["memory_usage"]) > 100:
                self.metrics["memory_usage"] = self.metrics["memory_usage"][-100:]
            if len(self.metrics["io_rates"]) > 100:
                self.metrics["io_rates"] = self.metrics["io_rates"][-100:]
            
            # Calculate energy consumption since last check
            duration_sec = current_time - self.last_check_times["usage"] if self.last_check_times["usage"] > 0 else 30
            energy_wh = self._calculate_energy_consumption(
                cpu_percent, memory_gb, io_mb_per_sec, duration_sec
            )
            
            # Add to cumulative energy
            self.cumulative_energy_wh += energy_wh
            
            # Record energy consumption
            self.metrics["energy_consumption"].append((current_time, energy_wh))
            if len(self.metrics["energy_consumption"]) > 100:
                self.metrics["energy_consumption"] = self.metrics["energy_consumption"][-100:]
            
            # Check for high CPU usage
            if cpu_percent > self.thresholds["max_cpu_percent"]:
                reports.append(self.report_error(
                    error_type="HIGH_CPU_USAGE",
                    severity="MEDIUM",
                    details={
                        "message": f"CPU usage is above threshold: {cpu_percent:.1f}% > {self.thresholds['max_cpu_percent']}%",
                        "cpu_percent": cpu_percent,
                        "threshold": self.thresholds["max_cpu_percent"]
                    },
                    context={
                        "recent_cpu_usage": self.metrics["cpu_usage"][-10:],
                        "duration_sec": duration_sec
                    }
                ))
            
            # Check for high memory usage
            if memory_percent > self.thresholds["max_memory_percent"]:
                reports.append(self.report_error(
                    error_type="HIGH_MEMORY_USAGE",
                    severity="MEDIUM",
                    details={
                        "message": f"Memory usage is above threshold: {memory_percent:.1f}% > {self.thresholds['max_memory_percent']}%",
                        "memory_percent": memory_percent,
                        "memory_gb": memory_gb,
                        "threshold": self.thresholds["max_memory_percent"]
                    },
                    context={
                        "recent_memory_usage": self.metrics["memory_usage"][-10:],
                        "duration_sec": duration_sec
                    }
                ))
            
            # Check for high I/O rate
            if io_mb_per_sec > self.thresholds["max_io_rate_mb"]:
                reports.append(self.report_error(
                    error_type="HIGH_IO_RATE",
                    severity="LOW",
                    details={
                        "message": f"I/O rate is above threshold: {io_mb_per_sec:.1f} MB/s > {self.thresholds['max_io_rate_mb']} MB/s",
                        "io_mb_per_sec": io_mb_per_sec,
                        "threshold": self.thresholds["max_io_rate_mb"]
                    },
                    context={
                        "recent_io_rates": self.metrics["io_rates"][-10:],
                        "duration_sec": duration_sec
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error in usage check: {str(e)}")
        
        return reports
    
    def _check_trends(self) -> List[ErrorReport]:
        """
        Check resource usage and energy consumption trends.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Need at least 5 measurements to analyze trends
            if len(self.metrics["cpu_usage"]) < 5 or len(self.metrics["memory_usage"]) < 5:
                return reports
            
            # Calculate trends using linear regression
            cpu_trend = self._calculate_trend(self.metrics["cpu_usage"][-20:])
            memory_trend = self._calculate_trend(self.metrics["memory_usage"][-20:])
            energy_trend = self._calculate_trend(self.metrics["energy_consumption"][-20:])
            
            # Check for continuously increasing CPU usage
            if cpu_trend > 0.5:  # Significant upward trend
                reports.append(self.report_error(
                    error_type="INCREASING_CPU_USAGE",
                    severity="LOW",
                    details={
                        "message": "CPU usage is showing a significant upward trend",
                        "cpu_trend": cpu_trend,
                        "current_cpu": self.metrics["cpu_usage"][-1][1]
                    },
                    context={
                        "recent_cpu_usage": self.metrics["cpu_usage"][-10:],
                        "suggested_action": "Consider optimizing CPU-intensive operations or adding rate limiting"
                    }
                ))
            
            # Check for continuously increasing memory usage (potential leak)
            if memory_trend > 0.3:  # Significant upward trend
                reports.append(self.report_error(
                    error_type="INCREASING_MEMORY_USAGE",
                    severity="MEDIUM",
                    details={
                        "message": "Memory usage is showing a consistent upward trend (potential memory leak)",
                        "memory_trend": memory_trend,
                        "current_memory": self.metrics["memory_usage"][-1][1]
                    },
                    context={
                        "recent_memory_usage": self.metrics["memory_usage"][-10:],
                        "suggested_action": "Check for memory leaks in long-running processes"
                    }
                ))
            
            # Check for rapidly increasing energy consumption
            if energy_trend > 0.2:  # Significant upward trend
                reports.append(self.report_error(
                    error_type="INCREASING_ENERGY_CONSUMPTION",
                    severity="LOW",
                    details={
                        "message": "Energy consumption is increasing rapidly",
                        "energy_trend": energy_trend,
                        "current_energy_rate": self.metrics["energy_consumption"][-1][1]
                    },
                    context={
                        "recent_energy_consumption": self.metrics["energy_consumption"][-10:],
                        "cumulative_energy_wh": self.cumulative_energy_wh,
                        "suggested_action": "Optimize resource usage to reduce energy consumption"
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error in trend check: {str(e)}")
        
        return reports
    
    def _check_efficiency(self) -> List[ErrorReport]:
        """
        Check energy efficiency metrics.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Calculate energy efficiency
            efficiency = self._calculate_energy_efficiency()
            
            # Record efficiency metric
            current_time = time.time()
            self.metrics["energy_efficiency"].append((current_time, efficiency))
            if len(self.metrics["energy_efficiency"]) > 50:
                self.metrics["energy_efficiency"] = self.metrics["energy_efficiency"][-50:]
            
            # Check if efficiency is below threshold
            if efficiency < self.thresholds["min_energy_efficiency"]:
                # Calculate how long the system has been inefficient
                low_efficiency_duration = 0
                for ts, eff in reversed(self.metrics["energy_efficiency"]):
                    if eff < self.thresholds["min_energy_efficiency"]:
                        low_efficiency_duration = current_time - ts
                    else:
                        break
                
                reports.append(self.report_error(
                    error_type="LOW_ENERGY_EFFICIENCY",
                    severity="MEDIUM" if efficiency < 0.4 else "LOW",
                    details={
                        "message": f"Energy efficiency is below threshold: {efficiency:.2f} < {self.thresholds['min_energy_efficiency']}",
                        "efficiency": efficiency,
                        "threshold": self.thresholds["min_energy_efficiency"],
                        "duration_sec": low_efficiency_duration
                    },
                    context={
                        "recent_efficiency": self.metrics["energy_efficiency"][-10:],
                        "cumulative_energy_wh": self.cumulative_energy_wh,
                        "energy_per_fix": self.metrics["operations_metrics"]["energy_per_fix"],
                        "optimization_suggestions": self._generate_optimization_suggestions()
                    }
                ))
            
            # Check energy per bug fix if available
            bug_fixes = self.metrics["operations_metrics"]["bug_fixes"]
            if bug_fixes > 0:
                energy_per_fix = self.cumulative_energy_wh / bug_fixes
                
                if energy_per_fix > self.thresholds["max_watt_hours"]:
                    reports.append(self.report_error(
                        error_type="HIGH_ENERGY_PER_FIX",
                        severity="LOW",
                        details={
                            "message": f"Energy consumption per bug fix is above threshold: {energy_per_fix:.2f} Wh > {self.thresholds['max_watt_hours']} Wh",
                            "energy_per_fix": energy_per_fix,
                            "threshold": self.thresholds["max_watt_hours"],
                            "bug_fixes": bug_fixes
                        },
                        context={
                            "cumulative_energy_wh": self.cumulative_energy_wh,
                            "optimization_suggestions": self._generate_optimization_suggestions()
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Error in efficiency check: {str(e)}")
        
        return reports
    
    def _calculate_trend(self, data_points: List[Tuple[float, float]]) -> float:
        """
        Calculate trend using linear regression.
        
        Args:
            data_points: List of (timestamp, value) tuples
            
        Returns:
            Trend coefficient (positive = increasing, negative = decreasing)
        """
        if len(data_points) < 2:
            return 0.0
        
        n = len(data_points)
        x = [i for i in range(n)]
        y = [point[1] for point in data_points]
        
        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Calculate slope (normalized to -1 to 1 range)
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
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """
        Generate suggestions to optimize energy usage.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Analyze recent metrics
        if self.metrics["cpu_usage"] and self.metrics["memory_usage"]:
            avg_cpu = sum(u[1] for u in self.metrics["cpu_usage"][-10:]) / min(10, len(self.metrics["cpu_usage"]))
            avg_memory = sum(u[1] for u in self.metrics["memory_usage"][-10:]) / min(10, len(self.metrics["memory_usage"]))
            
            # CPU optimizations
            if avg_cpu > 70:
                suggestions.append("Reduce CPU load by implementing rate limiting or batching for CPU-intensive operations")
                suggestions.append("Consider optimizing algorithms with high computational complexity")
            
            # Memory optimizations
            if avg_memory > 60:
                suggestions.append("Reduce memory usage by implementing more efficient data structures")
                suggestions.append("Check for memory leaks in long-running processes")
            
            # I/O optimizations
            if self.metrics["io_rates"] and sum(r[1] for r in self.metrics["io_rates"][-10:]) / min(10, len(self.metrics["io_rates"])) > 20:
                suggestions.append("Reduce disk I/O by implementing caching or buffering")
                suggestions.append("Consider compressing data before storage to reduce I/O")
            
            # General optimizations
            suggestions.append("Implement power-aware scheduling of non-time-critical tasks")
            
            # If very inefficient, add more aggressive suggestions
            if len(self.metrics["energy_efficiency"]) > 0 and self.metrics["energy_efficiency"][-1][1] < 0.4:
                suggestions.append("Consider suspending or removing non-essential background processes")
                suggestions.append("Implement more aggressive power management policies")
        
        return suggestions
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the sensor and monitored component."""
        # Calculate averages
        avg_cpu = 0.0
        if self.metrics["cpu_usage"]:
            avg_cpu = sum(u[1] for u in self.metrics["cpu_usage"][-10:]) / min(10, len(self.metrics["cpu_usage"]))
        
        avg_memory = 0.0
        if self.metrics["memory_usage"]:
            avg_memory = sum(u[1] for u in self.metrics["memory_usage"][-10:]) / min(10, len(self.metrics["memory_usage"]))
        
        avg_io = 0.0
        if self.metrics["io_rates"]:
            avg_io = sum(r[1] for r in self.metrics["io_rates"][-10:]) / min(10, len(self.metrics["io_rates"]))
        
        # Calculate efficiency
        efficiency = 1.0
        if self.metrics["energy_efficiency"]:
            efficiency = self.metrics["energy_efficiency"][-1][1]
        
        # Calculate uptime
        uptime_sec = time.time() - self.start_time
        
        # Calculate health score (0-100)
        health_score = (
            (1.0 - (avg_cpu / 100)) * 30 +  # CPU score (0-30)
            (1.0 - (avg_memory / 100)) * 30 +  # Memory score (0-30)
            efficiency * 40  # Efficiency score (0-40)
        )
        
        return {
            "sensor_id": self.sensor_id,
            "component_name": self.component_name,
            "last_check_time": self.last_check_time,
            "health_score": health_score,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "avg_io_rate_mb": avg_io,
            "energy_efficiency": efficiency,
            "cumulative_energy_wh": self.cumulative_energy_wh,
            "uptime_sec": uptime_sec,
            "energy_per_fix": self.metrics["operations_metrics"]["energy_per_fix"] 
                             if self.metrics["operations_metrics"]["bug_fixes"] > 0 else 0.0,
            "bug_fixes": self.metrics["operations_metrics"]["bug_fixes"],
            "optimizations_applied": self.metrics["operations_metrics"]["optimizations_applied"]
        }


# Factory function to create a sensor instance
def create_energy_calculator_sensor(config: Optional[Dict[str, Any]] = None) -> EnergyCalculatorSensor:
    """
    Create and initialize an energy calculator sensor.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized EnergyCalculatorSensor
    """
    return EnergyCalculatorSensor(config=config)
