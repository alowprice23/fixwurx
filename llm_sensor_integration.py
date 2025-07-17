"""
FixWurx Auditor LLM Sensor Integration

This module connects the error sensor framework with the LLM capabilities
of the auditor system, enabling intelligent error analysis, pattern recognition,
explanation generation, and self-diagnostic capabilities.
"""

import logging
import datetime
import json
import re
from typing import Dict, List, Set, Any, Optional, Union, Tuple, Type

# Import sensor components
from sensor_registry import SensorRegistry, ErrorSensor, ErrorReport, SensorManager
from component_sensors import (
    ObligationLedgerSensor, EnergyCalculatorSensor, ProofMetricsSensor,
    MetaAwarenessSensor, GraphDatabaseSensor, TimeSeriesDatabaseSensor,
    DocumentStoreSensor, BenchmarkingSensor
)

# Import LLM components
from llm_integrations import LLMManager, LLMResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [LLMSensorIntegration] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('llm_sensor_integration')


class SensorDataProvider:
    """
    Provides sensor data to the LLM components for analysis.
    Acts as an interface between the sensor framework and LLM system.
    """
    
    def __init__(self, registry: SensorRegistry, sensor_manager: SensorManager):
        """
        Initialize the sensor data provider.
        
        Args:
            registry: The sensor registry
            sensor_manager: The sensor manager
        """
        self.registry = registry
        self.sensor_manager = sensor_manager
        
    def get_active_sensors(self) -> List[Dict[str, Any]]:
        """
        Get information about all active sensors.
        
        Returns:
            List of sensor information dictionaries
        """
        all_sensors = []
        
        for sensor_id, sensor in self.registry.sensors.items():
            if sensor.enabled:
                all_sensors.append({
                    "sensor_id": sensor_id,
                    "component_name": sensor.component_name,
                    "sensitivity": sensor.sensitivity,
                    "last_check_time": sensor.last_check_time
                })
        
        return all_sensors
    
    def get_recent_errors(self, limit: int = 10, 
                          component_name: Optional[str] = None,
                          error_type: Optional[str] = None,
                          severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent error reports matching the specified criteria.
        
        Args:
            limit: Maximum number of reports to return
            component_name: Filter by component name (optional)
            error_type: Filter by error type (optional)
            severity: Filter by severity (optional)
            
        Returns:
            List of error report dictionaries
        """
        # Query errors from registry
        error_reports = self.registry.query_errors(
            component_name=component_name,
            error_type=error_type,
            severity=severity,
            status=None
        )
        
        # Sort by timestamp (most recent first)
        error_reports.sort(key=lambda r: r.timestamp, reverse=True)
        
        # Limit and convert to dictionaries
        return [report.to_dict() for report in error_reports[:limit]]
    
    def get_error_trends(self) -> Dict[str, Any]:
        """
        Get error trends analysis.
        
        Returns:
            Dictionary with trend analysis data
        """
        return self.registry.get_error_trends()
    
    def get_sensor_status(self, sensor_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of a specific sensor or all sensors.
        
        Args:
            sensor_id: ID of the sensor to get status for, or None for all sensors
            
        Returns:
            Sensor status dictionary
        """
        return self.registry.get_sensor_status(sensor_id)
    
    def get_component_health(self, component_name: str) -> Dict[str, Any]:
        """
        Get health assessment for a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component health assessment
        """
        # Get errors for this component
        errors = self.registry.query_errors(component_name=component_name)
        
        # Calculate health metrics
        total_errors = len(errors)
        critical_errors = sum(1 for e in errors if e.severity == "CRITICAL")
        high_errors = sum(1 for e in errors if e.severity == "HIGH")
        medium_errors = sum(1 for e in errors if e.severity == "MEDIUM")
        low_errors = sum(1 for e in errors if e.severity == "LOW")
        
        # Calculate health score (100 = perfect health, 0 = critical failure)
        health_score = 100
        if total_errors > 0:
            # Deduct points based on error severity
            health_score -= critical_errors * 25
            health_score -= high_errors * 10
            health_score -= medium_errors * 5
            health_score -= low_errors * 1
            
            # Ensure score is between 0 and 100
            health_score = max(0, min(100, health_score))
        
        # Determine health status
        if health_score >= 90:
            status = "HEALTHY"
        elif health_score >= 70:
            status = "MINOR_ISSUES"
        elif health_score >= 40:
            status = "DEGRADED"
        else:
            status = "CRITICAL"
        
        return {
            "component_name": component_name,
            "health_score": health_score,
            "status": status,
            "error_counts": {
                "total": total_errors,
                "critical": critical_errors,
                "high": high_errors,
                "medium": medium_errors,
                "low": low_errors
            },
            "recent_errors": [e.to_dict() for e in errors[:5]]  # Up to 5 recent errors
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health assessment.
        
        Returns:
            System health assessment
        """
        # Get all components
        components = set(sensor.component_name for sensor in self.registry.sensors.values())
        
        # Get health for each component
        component_health = {}
        for component in components:
            component_health[component] = self.get_component_health(component)
        
        # Calculate overall health score (average of component scores)
        if component_health:
            overall_score = sum(c["health_score"] for c in component_health.values()) / len(component_health)
        else:
            overall_score = 100  # Default if no components
        
        # Determine overall status
        if overall_score >= 90:
            overall_status = "HEALTHY"
        elif overall_score >= 70:
            overall_status = "MINOR_ISSUES"
        elif overall_score >= 40:
            overall_status = "DEGRADED"
        else:
            overall_status = "CRITICAL"
        
        return {
            "overall_health_score": overall_score,
            "overall_status": overall_status,
            "component_health": component_health,
            "assessment_time": datetime.datetime.now().isoformat()
        }
    
    def collect_sensor_data_for_llm(self) -> Dict[str, Any]:
        """
        Collect comprehensive sensor data for LLM analysis.
        
        Returns:
            Dictionary with all relevant sensor data
        """
        return {
            "active_sensors": self.get_active_sensors(),
            "recent_errors": self.get_recent_errors(limit=20),
            "error_trends": self.get_error_trends(),
            "system_health": self.get_system_health(),
            "sensor_manager_status": self.sensor_manager.get_status(),
            "collection_time": datetime.datetime.now().isoformat()
        }


class ErrorContextualizer:
    """
    Enhances error reports with LLM-generated context, explanations,
    and recommendations.
    """
    
    def __init__(self, llm_manager: LLMManager, data_provider: SensorDataProvider):
        """
        Initialize the error contextualizer.
        
        Args:
            llm_manager: The LLM manager for natural language processing
            data_provider: The sensor data provider
        """
        self.llm_manager = llm_manager
        self.data_provider = data_provider
        
    def contextualize(self, error_report: ErrorReport) -> Dict[str, Any]:
        """
        Contextualize an error report with additional information
        and natural language explanations, and update the report with
        standardized extended fields.
        
        Args:
            error_report: The error report to contextualize
            
        Returns:
            Enhanced error report with context
        """
        # Generate explanations and recommendations
        explanation = self._generate_explanation(error_report)
        recommendations = self._generate_recommendations(error_report)
        historical_context = self._add_historical_context(error_report)
        related_components = self._identify_related_components(error_report)
        
        # Update the error report with extended fields
        
        # Set recommendations field
        error_report.recommendations = recommendations
        
        # Create root_cause structure
        error_report.root_cause = {
            "cause_type": "analysis",
            "confidence": 0.8,
            "details": {
                "description": explanation,
                "source_file": error_report.context.get("source_file", "unknown"),
                "source_function": error_report.context.get("source_function", "unknown"),
                "line_number": error_report.context.get("line_number", 0)
            },
            "potential_causes": []
        }
        
        # Create impact structure
        error_report.impact = {
            "severity": error_report.severity,
            "scope": "multi_component" if related_components else "single_component",
            "affected_components": [error_report.component_name] + related_components,
            "affected_functionality": error_report.context.get("affected_functionality", []),
            "user_impact": error_report.context.get("user_impact", "Unknown impact on users"),
            "system_impact": error_report.context.get("system_impact", "Unknown impact on system")
        }
        
        # Set related_errors field from historical context
        if historical_context and "similar_past_errors" in historical_context:
            error_report.related_errors = [
                e.get("error_id") for e in historical_context["similar_past_errors"]
                if e.get("error_id") != error_report.error_id
            ]
        
        # Get the updated report as dictionary for return
        report_dict = error_report.to_dict()
        
        # Add non-standard fields for backward compatibility
        report_dict["explanation"] = explanation
        report_dict["historical_context"] = historical_context
        report_dict["related_components"] = related_components
        
        return report_dict
    
    def _generate_explanation(self, error_report: ErrorReport) -> str:
        """
        Generate a natural language explanation of the error.
        
        Args:
            error_report: The error report
            
        Returns:
            Natural language explanation
        """
        # Prepare prompt for LLM
        component_name = error_report.component_name
        error_type = error_report.error_type
        severity = error_report.severity
        details = json.dumps(error_report.details)
        
        prompt = (
            f"You are the Auditor system analyzing an error. Explain the following error in clear, "
            f"technical language. Focus on what this means for the system and why it matters.\n\n"
            f"Component: {component_name}\n"
            f"Error Type: {error_type}\n"
            f"Severity: {severity}\n"
            f"Details: {details}\n\n"
            f"Explanation:"
        )
        
        try:
            # Get explanation from LLM
            response = self.llm_manager.chat(
                role="system",
                content=prompt,
                task_type="explain",
                complexity="medium"
            )
            
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return f"Error explanation unavailable: {str(e)}"
    
    def _generate_recommendations(self, error_report: ErrorReport) -> List[str]:
        """
        Generate recommendations for addressing the error.
        
        Args:
            error_report: The error report
            
        Returns:
            List of recommendations
        """
        # Prepare prompt for LLM
        component_name = error_report.component_name
        error_type = error_report.error_type
        severity = error_report.severity
        details = json.dumps(error_report.details)
        
        prompt = (
            f"You are the Auditor system analyzing an error. Provide 3-5 specific, actionable "
            f"recommendations to fix or mitigate this error. Focus on technical solutions.\n\n"
            f"Component: {component_name}\n"
            f"Error Type: {error_type}\n"
            f"Severity: {severity}\n"
            f"Details: {details}\n\n"
            f"Recommendations (numbered list):"
        )
        
        try:
            # Get recommendations from LLM
            response = self.llm_manager.chat(
                role="system",
                content=prompt,
                task_type="explain",
                complexity="medium"
            )
            
            # Parse numbered list from response
            recommendations = []
            for line in response.text.strip().split('\n'):
                # Extract numbered items (1. Item, 2. Item, etc.)
                match = re.match(r'^\s*(\d+)[\.\)]\s+(.+)$', line)
                if match:
                    recommendations.append(match.group(2).strip())
            
            # If we couldn't parse numbered items, just return the whole text
            if not recommendations:
                recommendations = [response.text.strip()]
            
            return recommendations
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return [f"Recommendations unavailable: {str(e)}"]
    
    def _add_historical_context(self, error_report: ErrorReport) -> Dict[str, Any]:
        """
        Add historical context to the error report.
        
        Args:
            error_report: The error report
            
        Returns:
            Historical context dictionary
        """
        # Query similar errors in the past
        similar_errors = self.data_provider.get_recent_errors(
            limit=5,
            component_name=error_report.component_name,
            error_type=error_report.error_type
        )
        
        # Get error trends
        trends = self.data_provider.get_error_trends()
        
        # Check if this is a recurring issue
        is_recurring = len(similar_errors) > 1
        
        # Generate frequency description
        if not similar_errors:
            frequency = "First occurrence"
        elif len(similar_errors) == 1:
            frequency = "Second occurrence"
        elif len(similar_errors) < 5:
            frequency = f"{len(similar_errors) + 1}rd occurrence"
        else:
            frequency = f"Recurring issue ({len(similar_errors) + 1} occurrences)"
        
        return {
            "similar_past_errors": similar_errors,
            "is_recurring": is_recurring,
            "frequency": frequency,
            "component_error_count": trends.get("by_component", {}).get(error_report.component_name, 0)
        }
    
    def _identify_related_components(self, error_report: ErrorReport) -> List[str]:
        """
        Identify components that might be related to this error.
        
        Args:
            error_report: The error report
            
        Returns:
            List of related component names
        """
        # This would be more sophisticated in a real implementation,
        # potentially using a graph of component relationships
        # For now, return a simple list based on common relationships
        component_relationships = {
            "ObligationLedger": ["RepoModules", "EnergyCalculator"],
            "EnergyCalculator": ["ObligationLedger", "ProofMetrics"],
            "ProofMetrics": ["EnergyCalculator", "MetaAwareness"],
            "MetaAwareness": ["ProofMetrics", "GraphDatabase"],
            "GraphDatabase": ["MetaAwareness", "DocumentStore"],
            "TimeSeriesDatabase": ["BenchmarkingSystem", "GraphDatabase"],
            "DocumentStore": ["GraphDatabase", "TimeSeriesDatabase"],
            "BenchmarkingSystem": ["TimeSeriesDatabase"]
        }
        
        return component_relationships.get(error_report.component_name, [])


class ErrorPatternRecognizer:
    """
    Identifies patterns in error reports to detect recurring issues,
    potential root causes, and system weaknesses.
    """
    
    def __init__(self, data_provider: SensorDataProvider, llm_manager: LLMManager):
        """
        Initialize the error pattern recognizer.
        
        Args:
            data_provider: The sensor data provider
            llm_manager: The LLM manager
        """
        self.data_provider = data_provider
        self.llm_manager = llm_manager
        
    def analyze_patterns(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze error patterns over a specified timeframe.
        
        Args:
            timeframe_hours: Timeframe for analysis in hours
            
        Returns:
            Pattern analysis results
        """
        # Get recent errors within timeframe
        recent_errors = self.data_provider.get_recent_errors(limit=100)
        
        # Filter by timeframe
        cutoff_time = (datetime.datetime.now() - 
                      datetime.timedelta(hours=timeframe_hours)).isoformat()
        
        timeframe_errors = [
            e for e in recent_errors 
            if e["timestamp"] >= cutoff_time
        ]
        
        # Run statistical pattern detection
        statistical_patterns = self._detect_statistical_patterns(timeframe_errors)
        
        # Run LLM-based pattern analysis
        llm_patterns = self._detect_llm_patterns(timeframe_errors)
        
        return {
            "timeframe_hours": timeframe_hours,
            "error_count": len(timeframe_errors),
            "statistical_patterns": statistical_patterns,
            "llm_patterns": llm_patterns,
            "analysis_time": datetime.datetime.now().isoformat()
        }
    
    def _detect_statistical_patterns(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect statistical patterns in errors.
        
        Args:
            errors: List of error reports
            
        Returns:
            Statistical pattern analysis
        """
        # Count errors by component
        component_counts = {}
        for error in errors:
            component = error["component_name"]
            if component not in component_counts:
                component_counts[component] = 0
            component_counts[component] += 1
        
        # Count errors by type
        type_counts = {}
        for error in errors:
            error_type = error["error_type"]
            if error_type not in type_counts:
                type_counts[error_type] = 0
            type_counts[error_type] += 1
        
        # Count errors by severity
        severity_counts = {}
        for error in errors:
            severity = error["severity"]
            if severity not in severity_counts:
                severity_counts[severity] = 0
            severity_counts[severity] += 1
        
        # Find co-occurring errors (errors that happen close together)
        co_occurrences = self._find_co_occurrences(errors)
        
        # Find temporal patterns (time-based patterns)
        temporal_patterns = self._find_temporal_patterns(errors)
        
        return {
            "component_distribution": component_counts,
            "error_type_distribution": type_counts,
            "severity_distribution": severity_counts,
            "co_occurrences": co_occurrences,
            "temporal_patterns": temporal_patterns
        }
    
    def _find_co_occurrences(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find co-occurring errors (errors that happen close together).
        
        Args:
            errors: List of error reports
            
        Returns:
            List of co-occurrence patterns
        """
        # Sort errors by timestamp
        sorted_errors = sorted(errors, key=lambda e: e["timestamp"])
        
        # Define "close together" as within 5 minutes
        time_threshold = 5 * 60  # 5 minutes in seconds
        
        co_occurrences = []
        
        # Find errors that occur close together
        for i in range(len(sorted_errors) - 1):
            for j in range(i + 1, len(sorted_errors)):
                time_i = datetime.datetime.fromisoformat(sorted_errors[i]["timestamp"])
                time_j = datetime.datetime.fromisoformat(sorted_errors[j]["timestamp"])
                
                # Check if errors are close in time
                time_diff = (time_j - time_i).total_seconds()
                if time_diff <= time_threshold:
                    # These errors co-occur
                    co_occurrences.append({
                        "error1": {
                            "error_id": sorted_errors[i]["error_id"],
                            "component_name": sorted_errors[i]["component_name"],
                            "error_type": sorted_errors[i]["error_type"]
                        },
                        "error2": {
                            "error_id": sorted_errors[j]["error_id"],
                            "component_name": sorted_errors[j]["component_name"],
                            "error_type": sorted_errors[j]["error_type"]
                        },
                        "time_difference_seconds": time_diff
                    })
                else:
                    # We've moved too far in time, break inner loop
                    break
        
        return co_occurrences
    
    def _find_temporal_patterns(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find temporal patterns in errors.
        
        Args:
            errors: List of error reports
            
        Returns:
            Temporal pattern analysis
        """
        # Extract hour of day for each error
        hours = []
        for error in errors:
            timestamp = datetime.datetime.fromisoformat(error["timestamp"])
            hours.append(timestamp.hour)
        
        # Count errors by hour
        hour_counts = {}
        for hour in range(24):
            hour_counts[hour] = hours.count(hour)
        
        # Determine peak hours (hours with most errors)
        if hours:
            peak_hours = [
                hour for hour, count in hour_counts.items()
                if count == max(hour_counts.values())
            ]
        else:
            peak_hours = []
        
        return {
            "hour_distribution": hour_counts,
            "peak_hours": peak_hours
        }
    
    def _detect_llm_patterns(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to detect patterns in errors.
        
        Args:
            errors: List of error reports
            
        Returns:
            LLM pattern analysis
        """
        if not errors:
            return {
                "identified_patterns": [],
                "root_cause_hypotheses": []
            }
        
        # Prepare error summary for LLM
        error_summary = []
        for i, error in enumerate(errors[:20]):  # Limit to 20 errors to avoid context limit
            error_summary.append(
                f"Error {i+1}:\n"
                f"  Component: {error['component_name']}\n"
                f"  Type: {error['error_type']}\n"
                f"  Severity: {error['severity']}\n"
                f"  Timestamp: {error['timestamp']}\n"
                f"  Details: {json.dumps(error['details'])}\n"
            )
        
        error_text = "\n".join(error_summary)
        
        # Prepare prompt for pattern recognition
        prompt = (
            f"You are the Auditor system analyzing error patterns. Review these {len(errors)} errors "
            f"and identify meaningful patterns, potential root causes, and system weaknesses.\n\n"
            f"{error_text}\n\n"
            f"First, identify 3-5 distinct patterns in these errors (e.g., related components, "
            f"cascading failures, timing correlations).\n\n"
            f"Second, propose 2-3 root cause hypotheses that could explain these patterns.\n\n"
            f"Format your response with these exact headings:\n"
            f"IDENTIFIED PATTERNS:\n"
            f"1. (First pattern)\n"
            f"2. (Second pattern)\n"
            f"...\n\n"
            f"ROOT CAUSE HYPOTHESES:\n"
            f"1. (First hypothesis)\n"
            f"2. (Second hypothesis)\n"
            f"..."
        )
        
        try:
            # Get pattern analysis from LLM
            response = self.llm_manager.chat(
                role="system",
                content=prompt,
                task_type="analyze",
                complexity="high"
            )
            
            # Parse response
            text = response.text.strip()
            
            # Extract patterns
            patterns_match = re.search(r'IDENTIFIED PATTERNS:(.*?)ROOT CAUSE HYPOTHESES:', 
                                     text, re.DOTALL)
            
            if patterns_match:
                patterns_text = patterns_match.group(1).strip()
                patterns = []
                
                # Extract numbered items
                for line in patterns_text.split('\n'):
                    match = re.match(r'^\s*(\d+)[\.)\s]\s*(.+)$', line)
                    if match:
                        patterns.append(match.group(2).strip())
            else:
                patterns = []
            
            # Extract hypotheses
            hypotheses_match = re.search(r'ROOT CAUSE HYPOTHESES:(.*?)$', text, re.DOTALL)
            
            if hypotheses_match:
                hypotheses_text = hypotheses_match.group(1).strip()
                hypotheses = []
                
                # Extract numbered items
                for line in hypotheses_text.split('\n'):
                    match = re.match(r'^\s*(\d+)[\.)\s]\s*(.+)$', line)
                    if match:
                        hypotheses.append(match.group(2).strip())
            else:
                hypotheses = []
            
            return {
                "identified_patterns": patterns,
                "root_cause_hypotheses": hypotheses
            }
        except Exception as e:
            logger.error(f"Failed to perform LLM pattern analysis: {e}")
            return {
                "identified_patterns": [],
                "root_cause_hypotheses": [],
                "error": str(e)
            }


class SelfDiagnosisProvider:
    """
    Provides self-diagnostic capabilities to the auditor agent via LLM integration.
    Allows the auditor to explain its internal state and diagnose issues.
    """
    
    def __init__(self, data_provider: SensorDataProvider, llm_manager: LLMManager):
        """
        Initialize the self-diagnosis provider.
        
        Args:
            data_provider: The sensor data provider
            llm_manager: The LLM manager
        """
        self.data_provider = data_provider
        self.llm_manager = llm_manager
    
    def diagnose_issue(self, issue_description: str) -> str:
        """
        Use LLM to diagnose an internal issue based on sensor data.
        
        Args:
            issue_description: Description of the issue to diagnose
            
        Returns:
            Diagnosis text
        """
        # Collect relevant sensor data
        sensor_data = self.data_provider.collect_sensor_data_for_llm()
        
        # Extract key data points for diagnosis
        system_health = sensor_data["system_health"]
        recent_errors = sensor_data["recent_errors"]
        
        # Prepare data summary for LLM
        system_health_summary = (
            f"Overall System Health: Score {system_health['overall_health_score']:.1f}/100, "
            f"Status: {system_health['overall_status']}\n\n"
        )
        
        component_summaries = []
        for name, health in system_health["component_health"].items():
            component_summaries.append(
                f"Component {name}: Score {health['health_score']:.1f}/100, "
                f"Status: {health['status']}, "
                f"Errors: {health['error_counts']['total']} "
                f"({health['error_counts']['critical']} critical, "
                f"{health['error_counts']['high']} high)"
            )
        
        component_health_summary = "Component Health:\n" + "\n".join(component_summaries)
        
        recent_error_summary = "Recent Errors:\n"
        for i, error in enumerate(recent_errors[:5]):  # Show top 5 errors
            recent_error_summary += (
                f"{i+1}. Component: {error['component_name']}, "
                f"Type: {error['error_type']}, "
                f"Severity: {error['severity']}\n"
            )
        
        # Prepare prompt for diagnosis
        prompt = (
            f"You are the Auditor system's self-diagnostic module. A user has reported "
            f"the following issue:\n\n"
            f"{issue_description}\n\n"
            f"Based on your internal sensor data and error reports, provide a thorough "
            f"diagnostic analysis of what might be causing this issue. Here is your "
            f"current system state:\n\n"
            f"{system_health_summary}\n"
            f"{component_health_summary}\n\n"
            f"{recent_error_summary}\n\n"
            f"Diagnostic analysis:"
        )
        
        try:
            # Get diagnosis from LLM
            response = self.llm_manager.chat(
                role="system",
                content=prompt,
                task_type="diagnose",
                complexity="high"
            )
            
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate diagnosis: {e}")
            return f"Diagnosis unavailable: {str(e)}"
    
    def explain_internal_state(self) -> Dict[str, Any]:
        """
        Generate an explanation of the current internal state.
        
        Returns:
            Dictionary with explanations of internal state
        """
        # Collect sensor data
        sensor_data = self.data_provider.collect_sensor_data_for_llm()
        
        # Get system health
        system_health = sensor_data["system_health"]
        
        # Prepare prompt for general state explanation
        general_prompt = (
            f"You are the Auditor system explaining your current internal state. "
            f"Provide a concise, high-level summary of your current status, health, "
            f"and any noteworthy conditions. Here is your current system data:\n\n"
            f"Overall Health Score: {system_health['overall_health_score']:.1f}/100\n"
            f"Overall Status: {system_health['overall_status']}\n"
            f"Component Statuses: {', '.join([f'{name}: {health['status']}' for name, health in system_health['component_health'].items()])}\n"
            f"Recent Error Count: {len(sensor_data['recent_errors'])}\n\n"
            f"Summary of current state:"
        )
        
        # Prepare prompt for component-specific explanations
        component_prompts = {}
        for component_name, health in system_health["component_health"].items():
            # Get component-specific errors
            component_errors = [e for e in sensor_data["recent_errors"] 
                              if e["component_name"] == component_name]
            
            component_prompts[component_name] = (
                f"You are the Auditor system explaining the state of your {component_name} component. "
                f"Provide a concise explanation of this component's current status, any issues it's "
                f"experiencing, and what that means for system functionality. Here is the component data:\n\n"
                f"Health Score: {health['health_score']:.1f}/100\n"
                f"Status: {health['status']}\n"
                f"Error Counts: {health['error_counts']['total']} total "
                f"({health['error_counts']['critical']} critical, "
                f"{health['error_counts']['high']} high, "
                f"{health['error_counts']['medium']} medium, "
                f"{health['error_counts']['low']} low)\n\n"
                f"Recent Errors: {len(component_errors)}\n\n"
                f"Component state explanation:"
            )
        
        try:
            # Get general state explanation from LLM
            general_response = self.llm_manager.chat(
                role="system",
                content=general_prompt,
                task_type="explain",
                complexity="medium"
            )
            
            general_explanation = general_response.text.strip()
            
            # Get component-specific explanations
            component_explanations = {}
            for component_name, prompt in component_prompts.items():
                try:
                    response = self.llm_manager.chat(
                        role="system",
                        content=prompt,
                        task_type="explain",
                        complexity="medium"
                    )
                    
                    component_explanations[component_name] = response.text.strip()
                except Exception as e:
                    logger.error(f"Failed to generate explanation for {component_name}: {e}")
                    component_explanations[component_name] = f"Explanation unavailable: {str(e)}"
            
            return {
                "general_explanation": general_explanation,
                "component_explanations": component_explanations,
                "system_health": system_health,
                "generated_at": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to generate state explanations: {e}")
            return {
                "general_explanation": f"State explanation unavailable: {str(e)}",
                "component_explanations": {},
                "system_health": system_health,
                "generated_at": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
    def suggest_corrections(self, error_report: ErrorReport) -> List[Dict[str, Any]]:
        """
        Suggest potential corrections for reported errors.
        
        Args:
            error_report: The error report
            
        Returns:
            List of suggested corrections
        """
        # Prepare prompt for corrections
        component_name = error_report.component_name
        error_type = error_report.error_type
        severity = error_report.severity
        details = json.dumps(error_report.details)
        
        prompt = (
            f"You are the Auditor system suggesting corrections for an error. Provide 3-5 specific, "
            f"actionable corrections to fix this error. Each correction should include specific "
            f"implementation steps. Focus on technical solutions.\n\n"
            f"Component: {component_name}\n"
            f"Error Type: {error_type}\n"
            f"Severity: {severity}\n"
            f"Details: {details}\n\n"
            f"For each correction, provide:\n"
            f"1. A title/summary\n"
            f"2. Complexity (LOW, MEDIUM, HIGH)\n"
            f"3. Implementation steps (numbered list)\n"
            f"4. Expected impact\n\n"
            f"Format each correction as:\n"
            f"CORRECTION X: [Title]\n"
            f"COMPLEXITY: [Level]\n"
            f"STEPS:\n"
            f"1. [First step]\n"
            f"2. [Second step]\n"
            f"...\n"
            f"EXPECTED IMPACT: [Impact description]"
        )
        
        try:
            # Get correction suggestions from LLM
            response = self.llm_manager.chat(
                role="system",
                content=prompt,
                task_type="fix",
                complexity="high"
            )
            
            # Parse corrections from response
            text = response.text.strip()
            correction_blocks = re.split(r'CORRECTION \d+:', text)[1:]  # Skip the part before the first correction
            
            corrections = []
            for block in correction_blocks:
                # Extract title
                title = block.strip().split('\n')[0].strip()
                
                # Extract complexity
                complexity_match = re.search(r'COMPLEXITY:\s*(\w+)', block)
                complexity = complexity_match.group(1) if complexity_match else "MEDIUM"
                
                # Extract steps
                steps = []
                steps_match = re.search(r'STEPS:(.*?)EXPECTED IMPACT:', block, re.DOTALL)
                if steps_match:
                    steps_text = steps_match.group(1).strip()
                    for line in steps_text.split('\n'):
                        step_match = re.match(r'^\s*\d+\.\s+(.+)$', line)
                        if step_match:
                            steps.append(step_match.group(1).strip())
                
                # Extract impact
                impact_match = re.search(r'EXPECTED IMPACT:\s*(.*?)$', block, re.DOTALL)
                impact = impact_match.group(1).strip() if impact_match else ""
                
                corrections.append({
                    "title": title,
                    "complexity": complexity,
                    "steps": steps,
                    "expected_impact": impact
                })
            
            return corrections
        except Exception as e:
            logger.error(f"Failed to generate correction suggestions: {e}")
            return [{
                "title": "Error suggestion unavailable",
                "complexity": "UNKNOWN",
                "steps": [f"Error generating suggestions: {str(e)}"],
                "expected_impact": "Unknown"
            }]


# Factory function to create LLM integration components
def create_llm_integration(registry: SensorRegistry, sensor_manager: SensorManager, 
                         llm_manager: LLMManager) -> Dict[str, Any]:
    """
    Create and configure LLM integration components.
    
    Args:
        registry: The sensor registry
        sensor_manager: The sensor manager
        llm_manager: The LLM manager
        
    Returns:
        Dictionary of LLM integration components
    """
    # Create data provider
    data_provider = SensorDataProvider(registry, sensor_manager)
    
    # Create error contextualizer
    error_contextualizer = ErrorContextualizer(llm_manager, data_provider)
    
    # Create pattern recognizer
    pattern_recognizer = ErrorPatternRecognizer(data_provider, llm_manager)
    
    # Create self-diagnosis provider
    self_diagnosis = SelfDiagnosisProvider(data_provider, llm_manager)
    
    return {
        "data_provider": data_provider,
        "error_contextualizer": error_contextualizer,
        "pattern_recognizer": pattern_recognizer,
        "self_diagnosis": self_diagnosis
    }
