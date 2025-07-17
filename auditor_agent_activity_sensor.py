"""
FixWurx Auditor Agent Activity Sensor

This module implements a sensor for monitoring auditor agent activities,
tracking operations, interactions, and behavior patterns.
"""

import logging
import time
import os
import json
import datetime
import math
import random
from typing import Dict, List, Set, Any, Optional, Union, Tuple, Callable
from collections import defaultdict, deque

from sensor_base import ErrorSensor
from error_report import ErrorReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [AgentActivity] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('auditor_agent_activity_sensor')


class AuditorAgentActivitySensor(ErrorSensor):
    """
    Monitors auditor agent activities to detect issues with agent behavior.
    
    This sensor tracks operation patterns, agent interactions, response quality,
    and detects issues such as inefficient behavior, halted progress, or 
    deviations from expected behavior.
    """
    
    def __init__(self, 
                component_name: str = "AuditorAgent",
                config: Optional[Dict[str, Any]] = None):
        """Initialize the AuditorAgentActivitySensor."""
        super().__init__(
            sensor_id="auditor_agent_activity_sensor",
            component_name=component_name,
            config=config or {}
        )
        
        # Extract configuration values with defaults
        self.check_intervals = {
            "activity": self.config.get("activity_check_interval", 60),  # 1 minute
            "quality": self.config.get("quality_check_interval", 300),  # 5 minutes
            "pattern": self.config.get("pattern_check_interval", 600),  # 10 minutes
        }
        
        self.thresholds = {
            "max_idle_time": self.config.get("max_idle_time", 300),  # 5 minutes
            "max_loop_count": self.config.get("max_loop_count", 5),  # Maximum repeated operations
            "min_quality_score": self.config.get("min_quality_score", 0.7),  # 70% quality
            "max_sequential_errors": self.config.get("max_sequential_errors", 3),  # Max consecutive errors
            "max_daily_operations": self.config.get("max_daily_operations", 1000)  # Maximum operations per day
        }
        
        # Activity tracking
        self.activities_dir = self.config.get("activities_dir", "auditor_data/agent_activities")
        os.makedirs(self.activities_dir, exist_ok=True)
        
        # Hooks for activity tracking
        self.agent_hooks_installed = False
        self.activity_callback = None
        
        # Initialize activity metrics
        self.last_check_times = {check_type: 0 for check_type in self.check_intervals}
        self.metrics = {
            "activities": [],  # [(timestamp, activity_type, details), ...]
            "operations": defaultdict(int),  # {operation_type: count, ...}
            "response_quality": [],  # [(timestamp, quality_score), ...]
            "errors": [],  # [(timestamp, error_type, details), ...]
            "interaction_patterns": [],  # [(operation_sequence, count), ...]
            "daily_stats": {},  # {date_str: {operations, quality, etc.}, ...}
            "recent_activities": deque(maxlen=100)  # Recent activities for quick access
        }
        
        # Operation sequences for pattern detection
        self.current_sequence = []
        self.sequence_window = self.config.get("sequence_window", 5)  # Operations to consider for patterns
        
        # Time tracking
        self.start_time = time.time()
        self.last_activity_time = self.start_time
        
        # Mock mode for testing
        self.mock_mode = self.config.get("mock_mode", False)
        if self.mock_mode:
            self._initialize_mock_data()
        
        logger.info(f"Initialized AuditorAgentActivitySensor for {component_name}")
    
    def register_activity_callback(self, callback: Callable) -> None:
        """
        Register a callback to receive activity notifications.
        
        Args:
            callback: Function to call with activity details
        """
        self.activity_callback = callback
        logger.info("Activity callback registered")
    
    def record_activity(self, activity_type: str, details: Dict[str, Any]) -> None:
        """
        Record an agent activity.
        
        Args:
            activity_type: Type of activity (e.g., "query", "fix", "analyze")
            details: Details of the activity
        """
        try:
            current_time = time.time()
            
            # Create activity record
            activity = {
                "timestamp": current_time,
                "activity_type": activity_type,
                "details": details
            }
            
            # Update last activity time
            self.last_activity_time = current_time
            
            # Add to metrics
            self.metrics["activities"].append((current_time, activity_type, details))
            self.metrics["operations"][activity_type] += 1
            self.metrics["recent_activities"].append(activity)
            
            # Update current sequence for pattern detection
            self.current_sequence.append(activity_type)
            if len(self.current_sequence) > self.sequence_window:
                self.current_sequence.pop(0)
            
            # Update daily stats
            date_str = datetime.datetime.fromtimestamp(current_time).strftime("%Y-%m-%d")
            if date_str not in self.metrics["daily_stats"]:
                self.metrics["daily_stats"][date_str] = {
                    "operations": defaultdict(int),
                    "total_operations": 0,
                    "errors": 0,
                    "quality_scores": []
                }
            
            self.metrics["daily_stats"][date_str]["operations"][activity_type] += 1
            self.metrics["daily_stats"][date_str]["total_operations"] += 1
            
            # Save activity to file
            self._save_activity(activity)
            
            # Notify callback if registered
            if self.activity_callback:
                self.activity_callback(activity)
            
            logger.debug(f"Recorded activity: {activity_type}")
            
        except Exception as e:
            logger.error(f"Error recording activity: {str(e)}")
    
    def record_error(self, error_type: str, details: Dict[str, Any]) -> None:
        """
        Record an agent error.
        
        Args:
            error_type: Type of error
            details: Error details
        """
        try:
            current_time = time.time()
            
            # Create error record
            error = {
                "timestamp": current_time,
                "error_type": error_type,
                "details": details
            }
            
            # Add to metrics
            self.metrics["errors"].append((current_time, error_type, details))
            
            # Update daily stats
            date_str = datetime.datetime.fromtimestamp(current_time).strftime("%Y-%m-%d")
            if date_str in self.metrics["daily_stats"]:
                self.metrics["daily_stats"][date_str]["errors"] += 1
            
            logger.debug(f"Recorded error: {error_type}")
            
        except Exception as e:
            logger.error(f"Error recording agent error: {str(e)}")
    
    def record_quality_score(self, score: float, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a quality score for agent responses.
        
        Args:
            score: Quality score (0.0-1.0)
            details: Optional details about the quality assessment
        """
        try:
            current_time = time.time()
            
            # Create quality record
            quality = {
                "timestamp": current_time,
                "score": score,
                "details": details or {}
            }
            
            # Add to metrics
            self.metrics["response_quality"].append((current_time, score))
            
            # Update daily stats
            date_str = datetime.datetime.fromtimestamp(current_time).strftime("%Y-%m-%d")
            if date_str in self.metrics["daily_stats"]:
                self.metrics["daily_stats"][date_str]["quality_scores"].append(score)
            
            logger.debug(f"Recorded quality score: {score:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording quality score: {str(e)}")
    
    def _save_activity(self, activity: Dict[str, Any]) -> None:
        """
        Save activity to file.
        
        Args:
            activity: Activity data to save
        """
        try:
            # Create filename based on timestamp
            timestamp = activity["timestamp"]
            date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            
            # Create daily directory if it doesn't exist
            daily_dir = os.path.join(self.activities_dir, date_str)
            os.makedirs(daily_dir, exist_ok=True)
            
            # Create filename
            filename = f"{int(timestamp)}.json"
            filepath = os.path.join(daily_dir, filename)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(activity, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving activity to file: {str(e)}")
    
    def _initialize_mock_data(self) -> None:
        """Initialize mock data for testing."""
        current_time = time.time()
        
        # Generate some mock activities
        activity_types = ["analyze", "fix", "verify", "query", "report"]
        
        # Generate activities over the past day
        for i in range(100):
            # Random time in the past day
            timestamp = current_time - random.uniform(0, 86400)
            activity_type = random.choice(activity_types)
            
            # Create mock details
            details = {
                "duration_sec": random.uniform(0.5, 10),
                "target": f"component_{random.randint(1, 5)}",
                "success": random.random() > 0.2  # 80% success rate
            }
            
            # Add to metrics
            self.metrics["activities"].append((timestamp, activity_type, details))
            self.metrics["operations"][activity_type] += 1
            
            # Update daily stats
            date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            if date_str not in self.metrics["daily_stats"]:
                self.metrics["daily_stats"][date_str] = {
                    "operations": defaultdict(int),
                    "total_operations": 0,
                    "errors": 0,
                    "quality_scores": []
                }
            
            self.metrics["daily_stats"][date_str]["operations"][activity_type] += 1
            self.metrics["daily_stats"][date_str]["total_operations"] += 1
            
            # Add to recent activities
            activity = {
                "timestamp": timestamp,
                "activity_type": activity_type,
                "details": details
            }
            self.metrics["recent_activities"].append(activity)
        
        # Generate mock quality scores
        for i in range(20):
            timestamp = current_time - random.uniform(0, 86400)
            score = random.uniform(0.5, 1.0)
            
            self.metrics["response_quality"].append((timestamp, score))
            
            date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            if date_str in self.metrics["daily_stats"]:
                self.metrics["daily_stats"][date_str]["quality_scores"].append(score)
        
        # Generate mock errors
        error_types = ["timeout", "api_failure", "parsing_error", "logic_error"]
        for i in range(10):
            timestamp = current_time - random.uniform(0, 86400)
            error_type = random.choice(error_types)
            
            details = {
                "component": f"component_{random.randint(1, 5)}",
                "message": f"Mock error message for {error_type}"
            }
            
            self.metrics["errors"].append((timestamp, error_type, details))
            
            date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            if date_str in self.metrics["daily_stats"]:
                self.metrics["daily_stats"][date_str]["errors"] += 1
        
        # Generate mock interaction patterns
        sequences = []
        for i in range(20):
            seq = [random.choice(activity_types) for _ in range(self.sequence_window)]
            sequences.append(tuple(seq))
        
        # Count occurrences of each sequence
        sequence_counts = defaultdict(int)
        for seq in sequences:
            sequence_counts[seq] += 1
        
        # Store the most common sequences
        self.metrics["interaction_patterns"] = sorted(
            sequence_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10 patterns
        
        logger.info("Initialized mock data for agent activity sensor")
    
    def monitor(self, data: Any = None) -> List[ErrorReport]:
        """
        Monitor agent activities and detect issues.
        
        Args:
            data: Optional data for monitoring, such as new activities
            
        Returns:
            List of error reports for detected issues
        """
        self.last_check_time = time.time()
        reports = []
        
        # Process incoming activity data if provided
        if data:
            if isinstance(data, dict):
                # Single activity
                if "activity_type" in data and "details" in data:
                    self.record_activity(data["activity_type"], data["details"])
                # Quality score
                elif "quality_score" in data:
                    self.record_quality_score(data["quality_score"], data.get("details"))
                # Error
                elif "error_type" in data and "details" in data:
                    self.record_error(data["error_type"], data["details"])
            elif isinstance(data, list):
                # List of activities
                for item in data:
                    if isinstance(item, dict):
                        if "activity_type" in item and "details" in item:
                            self.record_activity(item["activity_type"], item["details"])
        
        # Perform activity check if needed
        if self.last_check_time - self.last_check_times["activity"] >= self.check_intervals["activity"]:
            activity_reports = self._check_activity_patterns()
            if activity_reports:
                reports.extend(activity_reports)
            self.last_check_times["activity"] = self.last_check_time
        
        # Perform quality check if needed
        if self.last_check_time - self.last_check_times["quality"] >= self.check_intervals["quality"]:
            quality_reports = self._check_response_quality()
            if quality_reports:
                reports.extend(quality_reports)
            self.last_check_times["quality"] = self.last_check_time
        
        # Perform pattern check if needed
        if self.last_check_time - self.last_check_times["pattern"] >= self.check_intervals["pattern"]:
            pattern_reports = self._check_behavior_patterns()
            if pattern_reports:
                reports.extend(pattern_reports)
            self.last_check_times["pattern"] = self.last_check_time
        
        return reports
    
    def _check_activity_patterns(self) -> List[ErrorReport]:
        """
        Check for issues in agent activity patterns.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Check for agent inactivity
            idle_time = time.time() - self.last_activity_time
            if idle_time > self.thresholds["max_idle_time"]:
                reports.append(self.report_error(
                    error_type="AGENT_INACTIVITY",
                    severity="MEDIUM" if idle_time > 2 * self.thresholds["max_idle_time"] else "LOW",
                    details={
                        "message": f"Agent has been inactive for {idle_time:.1f} seconds",
                        "idle_time_seconds": idle_time,
                        "threshold": self.thresholds["max_idle_time"],
                        "last_activity_time": self.last_activity_time
                    },
                    context={
                        "last_activity": list(self.metrics["recent_activities"])[-1] if self.metrics["recent_activities"] else None,
                        "suggested_action": "Check if the agent is stuck or waiting for resources"
                    }
                ))
            
            # Check for excessive operations
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            if today_str in self.metrics["daily_stats"]:
                daily_ops = self.metrics["daily_stats"][today_str]["total_operations"]
                
                if daily_ops > self.thresholds["max_daily_operations"]:
                    reports.append(self.report_error(
                        error_type="EXCESSIVE_OPERATIONS",
                        severity="MEDIUM",
                        details={
                            "message": f"Agent performed {daily_ops} operations today, exceeding the threshold of {self.thresholds['max_daily_operations']}",
                            "operation_count": daily_ops,
                            "threshold": self.thresholds["max_daily_operations"]
                        },
                        context={
                            "operation_breakdown": dict(self.metrics["daily_stats"][today_str]["operations"]),
                            "suggested_action": "Check for inefficient workflows or infinite loops"
                        }
                    ))
            
            # Check for repetitive operations (potential loops)
            recent_activities = list(self.metrics["recent_activities"])
            if len(recent_activities) >= self.thresholds["max_loop_count"]:
                # Get the most recent activities
                recent_ops = [a["activity_type"] for a in recent_activities[-self.thresholds["max_loop_count"]:]]
                
                # Check if they're all the same
                if len(set(recent_ops)) == 1:
                    reports.append(self.report_error(
                        error_type="REPETITIVE_OPERATIONS",
                        severity="MEDIUM",
                        details={
                            "message": f"Agent performed the same operation ({recent_ops[0]}) {len(recent_ops)} times in succession",
                            "operation_type": recent_ops[0],
                            "repetition_count": len(recent_ops),
                            "threshold": self.thresholds["max_loop_count"]
                        },
                        context={
                            "recent_activities": recent_activities[-self.thresholds["max_loop_count"]:],
                            "suggested_action": "Check for loop conditions or retry logic"
                        }
                    ))
            
            # Check for sequential errors
            recent_errors = self.metrics["errors"][-self.thresholds["max_sequential_errors"]:]
            if len(recent_errors) >= self.thresholds["max_sequential_errors"]:
                # Calculate time span between first and last error
                if recent_errors:
                    time_span = recent_errors[-1][0] - recent_errors[0][0]
                    
                    # If errors occurred within a short time
                    if time_span < 300:  # 5 minutes
                        reports.append(self.report_error(
                            error_type="SEQUENTIAL_ERRORS",
                            severity="HIGH",
                            details={
                                "message": f"Agent encountered {len(recent_errors)} sequential errors within {time_span:.1f} seconds",
                                "error_count": len(recent_errors),
                                "time_span_seconds": time_span,
                                "threshold": self.thresholds["max_sequential_errors"]
                            },
                            context={
                                "recent_errors": recent_errors,
                                "suggested_action": "Check for systemic issues or resource problems"
                            }
                        ))
            
        except Exception as e:
            logger.error(f"Error in activity pattern check: {str(e)}")
        
        return reports
    
    def _check_response_quality(self) -> List[ErrorReport]:
        """
        Check for issues in agent response quality.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Check recent quality scores
            recent_scores = [s[1] for s in self.metrics["response_quality"][-10:]]
            if recent_scores:
                avg_score = sum(recent_scores) / len(recent_scores)
                
                if avg_score < self.thresholds["min_quality_score"]:
                    reports.append(self.report_error(
                        error_type="LOW_RESPONSE_QUALITY",
                        severity="HIGH" if avg_score < 0.5 else "MEDIUM",
                        details={
                            "message": f"Agent response quality is below threshold: {avg_score:.2f} < {self.thresholds['min_quality_score']}",
                            "average_score": avg_score,
                            "threshold": self.thresholds["min_quality_score"]
                        },
                        context={
                            "recent_scores": recent_scores,
                            "suggested_action": "Check agent configuration or knowledge sources"
                        }
                    ))
            
            # Check for declining quality trend
            if len(recent_scores) >= 5:
                # Simple trend analysis
                first_half = recent_scores[:len(recent_scores)//2]
                second_half = recent_scores[len(recent_scores)//2:]
                
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                
                # If second half is significantly worse than first half
                if first_avg - second_avg > 0.2:  # 20% decline
                    reports.append(self.report_error(
                        error_type="DECLINING_QUALITY_TREND",
                        severity="MEDIUM",
                        details={
                            "message": f"Agent response quality is declining: {first_avg:.2f} → {second_avg:.2f}",
                            "initial_average": first_avg,
                            "current_average": second_avg,
                            "decline_percentage": ((first_avg - second_avg) / first_avg) * 100
                        },
                        context={
                            "quality_trend": list(zip(range(len(recent_scores)), recent_scores)),
                            "suggested_action": "Check for knowledge drift or resource constraints"
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Error in response quality check: {str(e)}")
        
        return reports
    
    def _check_behavior_patterns(self) -> List[ErrorReport]:
        """
        Check for issues in agent behavior patterns.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Analyze operation distribution
            total_ops = sum(self.metrics["operations"].values())
            if total_ops > 0:
                op_distribution = {op: count / total_ops for op, count in self.metrics["operations"].items()}
                
                # Check for imbalanced operations
                # For example, too many analyze operations compared to fix operations
                if "analyze" in op_distribution and "fix" in op_distribution:
                    if op_distribution["analyze"] > 5 * op_distribution["fix"]:
                        reports.append(self.report_error(
                            error_type="OPERATION_IMBALANCE",
                            severity="LOW",
                            details={
                                "message": f"Agent performs excessive analysis compared to fixes: {op_distribution['analyze']:.1%} vs {op_distribution['fix']:.1%}",
                                "analyze_ratio": op_distribution["analyze"],
                                "fix_ratio": op_distribution["fix"],
                                "imbalance_factor": op_distribution["analyze"] / op_distribution["fix"] if op_distribution["fix"] > 0 else float('inf')
                            },
                            context={
                                "operation_distribution": op_distribution,
                                "suggested_action": "Check decision thresholds or confidence levels"
                            }
                        ))
            
            # Look for inefficient patterns
            # In a real implementation, we would use more sophisticated analysis
            # For this demo, we'll look for simple patterns like oscillation between states
            
            # Check if the agent is oscillating between two states
            if len(self.metrics["interaction_patterns"]) > 0:
                for pattern, count in self.metrics["interaction_patterns"]:
                    # Check for oscillation pattern (e.g., analyze-fix-analyze-fix)
                    if len(pattern) >= 4 and len(set(pattern)) <= 2:
                        # Check if the pattern alternates
                        alternating = True
                        for i in range(2, len(pattern)):
                            if pattern[i] != pattern[i-2]:
                                alternating = False
                                break
                        
                        if alternating:
                            reports.append(self.report_error(
                                error_type="OSCILLATING_BEHAVIOR",
                                severity="MEDIUM",
                                details={
                                    "message": f"Agent is oscillating between states: {'-'.join(pattern)}",
                                    "pattern": pattern,
                                    "occurrence_count": count
                                },
                                context={
                                    "suggested_action": "Check for conflicting goals or decision boundaries"
                                }
                            ))
            
        except Exception as e:
            logger.error(f"Error in behavior pattern check: {str(e)}")
        
        return reports
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the sensor and monitored component."""
        try:
            # Get recent activity count
            recent_activity_count = len(self.metrics["recent_activities"])
            
            # Calculate idle time
            idle_time = time.time() - self.last_activity_time
            
            # Calculate recent error rate
            total_errors = len(self.metrics["errors"])
            recent_errors = sum(1 for e in self.metrics["errors"] if time.time() - e[0] < 3600)  # Last hour
            
            # Calculate average quality score
            recent_scores = [s[1] for s in self.metrics["response_quality"][-10:]]
            avg_quality = sum(recent_scores) / len(recent_scores) if recent_scores else 1.0
            
            # Calculate operation distribution
            total_ops = sum(self.metrics["operations"].values())
            op_distribution = {op: count / total_ops for op, count in self.metrics["operations"].items()} if total_ops > 0 else {}
            
            # Calculate health score (0-100)
            # Components:
            # - Activity health (30 points) - based on idle time
            # - Error health (30 points) - based on recent error rate
            # - Quality health (40 points) - based on average quality score
            
            activity_health = 30 * (1 - min(1, idle_time / self.thresholds["max_idle_time"]))
            
            error_health = 30
            if total_ops > 0:
                error_rate = recent_errors / max(1, recent_activity_count)
                error_health = 30 * (1 - min(1, error_rate * 10))  # Scale error rate
            
            quality_health = 40 * min(1, avg_quality / self.thresholds["min_quality_score"])
            
            # Overall health score
            health_score = activity_health + error_health + quality_health
            
            return {
                "sensor_id": self.sensor_id,
                "component_name": self.component_name,
                "last_check_time": self.last_check_time,
                "health_score": health_score,
                "total_activities": sum(self.metrics["operations"].values()),
                "recent_activities": recent_activity_count,
                "idle_time_seconds": idle_time,
                "error_count": total_errors,
                "recent_errors": recent_errors,
                "avg_quality_score": avg_quality,
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


# Factory function to create a sensor instance
def create_auditor_agent_activity_sensor(config: Optional[Dict[str, Any]] = None) -> AuditorAgentActivitySensor:
    """
    Create and initialize an auditor agent activity sensor.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized AuditorAgentActivitySensor
    """
    return AuditorAgentActivitySensor(config=config)
