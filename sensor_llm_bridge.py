"""
FixWurx Auditor Sensor-LLM Bridge

This module creates a bridge between the sensor framework and the auditor's
LLM capabilities, enabling self-awareness and introspection. It allows the
auditor to analyze its own sensor data and generate natural language
explanations about its internal state.
"""

import os
import logging
import datetime
import json
import inspect
import importlib
import textwrap
from typing import Dict, List, Set, Any, Optional, Union, Callable

# Import sensor components
from error_report import ErrorReport
from sensor_base import ErrorSensor
from sensor_registry import SensorRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [SensorLLMBridge] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('sensor_llm_bridge')


class SensorLLMBridge:
    """
    Bridge between the sensor framework and the auditor's LLM capabilities.
    Provides methods for the auditor to introspect its own state through sensors.
    """
    
    def __init__(self, registry: SensorRegistry, llm_interface=None):
        """
        Initialize the Sensor-LLM Bridge.
        
        Args:
            registry: The sensor registry
            llm_interface: Interface to the LLM system (optional)
        """
        self.registry = registry
        self.llm_interface = llm_interface
        self.cached_error_analyses = {}
        self.cached_sensor_states = {}
        self.cached_introspection = {}
        self.last_analysis_time = None
        
        logger.info("Initialized Sensor-LLM Bridge")
    
    def analyze_errors(self, max_age_minutes: int = 60) -> Dict[str, Any]:
        """
        Analyze current error reports for the LLM.
        
        Args:
            max_age_minutes: Maximum age of error reports to include (in minutes)
            
        Returns:
            Analysis results
        """
        self.last_analysis_time = datetime.datetime.now()
        
        # Get all open error reports
        reports = self.registry.query_errors(status="OPEN")
        
        # Filter by age
        if max_age_minutes > 0:
            cutoff = datetime.datetime.now() - datetime.timedelta(minutes=max_age_minutes)
            reports = [r for r in reports if 
                      datetime.datetime.fromisoformat(r.timestamp) >= cutoff]
        
        # Group by component and severity
        by_component = {}
        by_severity = {}
        
        for report in reports:
            # Group by component
            if report.component_name not in by_component:
                by_component[report.component_name] = []
            by_component[report.component_name].append(report)
            
            # Group by severity
            if report.severity not in by_severity:
                by_severity[report.severity] = []
            by_severity[report.severity].append(report)
        
        # Generate analysis
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_errors": len(reports),
            "by_component": {
                component: len(reports) for component, reports in by_component.items()
            },
            "by_severity": {
                severity: len(reports) for severity, reports in by_severity.items()
            },
            "critical_errors": [self._format_error_for_llm(r) for r in by_severity.get("CRITICAL", [])],
            "high_severity_errors": [self._format_error_for_llm(r) for r in by_severity.get("HIGH", [])],
            "components_with_most_errors": sorted(
                by_component.keys(), 
                key=lambda c: len(by_component[c]), 
                reverse=True
            )[:3],
            "natural_language_summary": self._generate_error_summary(reports, by_component, by_severity)
        }
        
        # Cache analysis
        self.cached_error_analyses[self.last_analysis_time.isoformat()] = analysis
        
        return analysis
    
    def get_component_health(self, component_name: str) -> Dict[str, Any]:
        """
        Get health information about a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component health information
        """
        # Get sensors for this component
        sensors = self.registry.get_sensors_for_component(component_name)
        
        # Get recent errors for this component
        recent_errors = self.registry.query_errors(component_name=component_name)
        
        # Calculate health metrics
        total_errors = len(recent_errors)
        open_errors = len([r for r in recent_errors if r.status == "OPEN"])
        critical_errors = len([r for r in recent_errors if r.severity == "CRITICAL"])
        high_errors = len([r for r in recent_errors if r.severity == "HIGH"])
        
        # Calculate health score (0-100)
        if total_errors == 0:
            health_score = 100
        else:
            # Weight by severity
            weighted_sum = (
                critical_errors * 5.0 +  # Critical errors have 5x weight
                high_errors * 3.0 +      # High errors have 3x weight
                open_errors * 1.0        # Open errors have 1x weight
            )
            health_score = max(0, 100 - weighted_sum * 10)
        
        health = {
            "component_name": component_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "sensors": [s.sensor_id for s in sensors],
            "health_score": health_score,
            "error_counts": {
                "total": total_errors,
                "open": open_errors,
                "critical": critical_errors,
                "high": high_errors,
            },
            "status": self._determine_health_status(health_score),
            "natural_language_summary": self._generate_component_health_summary(
                component_name, health_score, recent_errors
            )
        }
        
        # Cache health state
        component_key = f"{component_name}_{datetime.datetime.now().isoformat()}"
        self.cached_sensor_states[component_key] = health
        
        return health
    
    def get_system_introspection(self) -> Dict[str, Any]:
        """
        Generate introspection information about the system.
        
        Returns:
            System introspection information
        """
        # Get all components with sensors
        component_names = self.registry.get_all_component_names()
        
        # Get health for each component
        component_health = {}
        for component_name in component_names:
            component_health[component_name] = self.get_component_health(component_name)
        
        # Calculate overall system health
        if component_health:
            overall_health = sum(c["health_score"] for c in component_health.values()) / len(component_health)
        else:
            overall_health = 100
        
        # Get code structure information
        code_structure = self._analyze_code_structure()
        
        introspection = {
            "timestamp": datetime.datetime.now().isoformat(),
            "components": list(component_names),
            "overall_health": overall_health,
            "overall_status": self._determine_health_status(overall_health),
            "component_health": component_health,
            "code_structure": code_structure,
            "natural_language_summary": self._generate_system_summary(
                overall_health, component_health, code_structure
            )
        }
        
        # Cache introspection
        self.cached_introspection[introspection["timestamp"]] = introspection
        
        return introspection
    
    def explain_error(self, error_id: str) -> Dict[str, Any]:
        """
        Generate a detailed explanation of an error for the LLM.
        
        Args:
            error_id: ID of the error to explain
            
        Returns:
            Error explanation
        """
        # Get the error report
        report = self.registry.get_error_report(error_id)
        if not report:
            return {
                "error": f"Error {error_id} not found",
                "natural_language_explanation": f"I couldn't find an error with ID {error_id} in my records."
            }
        
        # Get the sensor that generated this error
        sensor = self.registry.get_sensor(report.sensor_id)
        sensor_info = sensor.get_status() if sensor else {"status": "unknown"}
        
        # Get any related errors
        related_errors = []
        for related_id in report.related_errors:
            related = self.registry.get_error_report(related_id)
            if related:
                related_errors.append(self._format_error_for_llm(related))
        
        # Format the error explanation
        explanation = {
            "error_id": error_id,
            "timestamp": report.timestamp,
            "component": report.component_name,
            "error_type": report.error_type,
            "severity": report.severity,
            "status": report.status,
            "details": report.details,
            "context": report.context,
            "sensor": sensor_info,
            "related_errors": related_errors,
            "resolution": report.resolution,
            "root_cause": report.root_cause,
            "impact": report.impact,
            "recommendations": report.recommendations,
            "natural_language_explanation": self._generate_error_explanation(report, sensor_info)
        }
        
        return explanation
    
    def get_code_for_component(self, component_name: str) -> Dict[str, Any]:
        """
        Get code information for a component to enable the LLM to understand
        the implementation.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Code information
        """
        # Try to find the module for this component
        try:
            # This is a simplified approach - in a real system, we would need a mapping
            module_name = component_name.lower()
            module = importlib.import_module(module_name)
            
            # Get source code
            source = inspect.getsource(module)
            
            # Get classes and functions
            classes = {}
            functions = {}
            
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module.__name__:
                    classes[name] = inspect.getsource(obj)
                elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                    functions[name] = inspect.getsource(obj)
            
            return {
                "component_name": component_name,
                "module_name": module.__name__,
                "source_summary": self._summarize_source(source),
                "classes": list(classes.keys()),
                "functions": list(functions.keys()),
                "class_details": classes,
                "function_details": functions
            }
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not get code for component {component_name}: {e}")
            return {
                "component_name": component_name,
                "error": f"Could not get code: {str(e)}"
            }
    
    def _format_error_for_llm(self, report: ErrorReport) -> Dict[str, Any]:
        """
        Format an error report for the LLM.
        
        Args:
            report: Error report
            
        Returns:
            Formatted error information
        """
        return {
            "error_id": report.error_id,
            "component": report.component_name,
            "type": report.error_type,
            "severity": report.severity,
            "status": report.status,
            "timestamp": report.timestamp,
            "summary": report.details.get("message", "No summary available")
        }
    
    def _generate_error_summary(self, reports: List[ErrorReport], 
                              by_component: Dict[str, List[ErrorReport]],
                              by_severity: Dict[str, List[ErrorReport]]) -> str:
        """
        Generate a natural language summary of errors.
        
        Args:
            reports: List of error reports
            by_component: Reports grouped by component
            by_severity: Reports grouped by severity
            
        Returns:
            Natural language summary
        """
        if not reports:
            return "I have not detected any errors in my system recently."
        
        # Count by severity
        critical_count = len(by_severity.get("CRITICAL", []))
        high_count = len(by_severity.get("HIGH", []))
        medium_count = len(by_severity.get("MEDIUM", []))
        low_count = len(by_severity.get("LOW", []))
        
        # Get top 3 components with errors
        top_components = sorted(
            by_component.keys(), 
            key=lambda c: len(by_component[c]), 
            reverse=True
        )[:3]
        
        # Build summary
        summary = f"I have detected {len(reports)} errors in my system. "
        
        # Add severity breakdown
        severity_parts = []
        if critical_count:
            severity_parts.append(f"{critical_count} critical")
        if high_count:
            severity_parts.append(f"{high_count} high severity")
        if medium_count:
            severity_parts.append(f"{medium_count} medium severity")
        if low_count:
            severity_parts.append(f"{low_count} low severity")
        
        if severity_parts:
            summary += "These include " + ", ".join(severity_parts) + ". "
        
        # Add component breakdown
        if top_components:
            component_parts = []
            for component in top_components:
                count = len(by_component[component])
                component_parts.append(f"{count} in {component}")
            
            summary += "The most affected components are: " + ", ".join(component_parts) + ". "
        
        # Add critical error details if any
        if critical_count:
            summary += "\n\nCritical errors include: "
            for i, report in enumerate(by_severity["CRITICAL"][:3]):  # Top 3 critical errors
                summary += f"\n- {report.error_type}: {report.details.get('message', 'No details')}"
            
            if critical_count > 3:
                summary += f"\n- ...and {critical_count - 3} more critical errors"
        
        return summary
    
    def _determine_health_status(self, health_score: float) -> str:
        """
        Determine health status based on health score.
        
        Args:
            health_score: Health score (0-100)
            
        Returns:
            Health status string
        """
        if health_score >= 90:
            return "HEALTHY"
        elif health_score >= 70:
            return "STABLE"
        elif health_score >= 50:
            return "DEGRADED"
        elif health_score >= 30:
            return "IMPAIRED"
        else:
            return "CRITICAL"
    
    def _generate_component_health_summary(self, component_name: str, 
                                         health_score: float,
                                         errors: List[ErrorReport]) -> str:
        """
        Generate a natural language summary of component health.
        
        Args:
            component_name: Name of the component
            health_score: Health score (0-100)
            errors: List of errors for this component
            
        Returns:
            Natural language summary
        """
        status = self._determine_health_status(health_score)
        
        summary = f"The {component_name} component is currently {status.lower()} "
        summary += f"with a health score of {health_score:.1f}/100. "
        
        if not errors:
            summary += "No errors have been detected in this component."
            return summary
        
        # Group errors by severity
        by_severity = {}
        for error in errors:
            if error.severity not in by_severity:
                by_severity[error.severity] = []
            by_severity[error.severity].append(error)
        
        # Add error counts
        error_parts = []
        if "CRITICAL" in by_severity:
            error_parts.append(f"{len(by_severity['CRITICAL'])} critical")
        if "HIGH" in by_severity:
            error_parts.append(f"{len(by_severity['HIGH'])} high severity")
        if "MEDIUM" in by_severity:
            error_parts.append(f"{len(by_severity['MEDIUM'])} medium severity")
        if "LOW" in by_severity:
            error_parts.append(f"{len(by_severity['LOW'])} low severity")
        
        if error_parts:
            summary += "There are " + ", ".join(error_parts) + " errors. "
        
        # Add most common error types
        error_types = {}
        for error in errors:
            if error.error_type not in error_types:
                error_types[error.error_type] = 0
            error_types[error.error_type] += 1
        
        top_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:2]
        if top_errors:
            summary += "The most common issues are: " + ", ".join(f"{name} ({count})" for name, count in top_errors) + "."
        
        return summary
    
    def _generate_system_summary(self, overall_health: float,
                               component_health: Dict[str, Dict[str, Any]],
                               code_structure: Dict[str, Any]) -> str:
        """
        Generate a natural language summary of the overall system.
        
        Args:
            overall_health: Overall health score (0-100)
            component_health: Health information for each component
            code_structure: Code structure information
            
        Returns:
            Natural language summary
        """
        status = self._determine_health_status(overall_health)
        
        summary = f"My system is currently {status.lower()} with an overall health score of {overall_health:.1f}/100. "
        
        # Count components by status
        status_counts = {}
        for health in component_health.values():
            component_status = health["status"]
            if component_status not in status_counts:
                status_counts[component_status] = 0
            status_counts[component_status] += 1
        
        # Add component status breakdown
        if status_counts:
            status_parts = []
            for status, count in sorted(status_counts.items()):
                status_parts.append(f"{count} {status.lower()}")
            
            summary += "I have " + ", ".join(status_parts) + " components. "
        
        # Add information about most critical components
        critical_components = []
        for name, health in component_health.items():
            if health["status"] in ["CRITICAL", "IMPAIRED"]:
                critical_components.append((name, health["health_score"]))
        
        if critical_components:
            components_str = ", ".join(f"{name} ({score:.1f}/100)" for name, score in critical_components)
            summary += f"My most critical components are: {components_str}. "
        
        # Add code structure information
        if code_structure:
            summary += f"\n\nI am composed of approximately {code_structure.get('file_count', 0)} files "
            summary += f"with {code_structure.get('class_count', 0)} classes "
            summary += f"and {code_structure.get('function_count', 0)} functions. "
            
            if "top_modules" in code_structure:
                summary += "My main modules include: " + ", ".join(code_structure["top_modules"]) + "."
        
        return summary
    
    def _generate_error_explanation(self, report: ErrorReport, 
                                  sensor_info: Dict[str, Any]) -> str:
        """
        Generate a natural language explanation of an error.
        
        Args:
            report: Error report
            sensor_info: Information about the sensor
            
        Returns:
            Natural language explanation
        """
        explanation = f"I detected a {report.severity.lower()} severity error "
        explanation += f"of type '{report.error_type}' in the {report.component_name} component. "
        
        # Add error details
        if "message" in report.details:
            explanation += f"The specific issue is: {report.details['message']}. "
        
        # Add error context
        if report.context:
            context_str = ", ".join(f"{k}: {v}" for k, v in report.context.items() 
                                   if not isinstance(v, (dict, list)))
            if context_str:
                explanation += f"Additional context: {context_str}. "
        
        # Add resolution status
        if report.status == "RESOLVED":
            explanation += f"This issue has been resolved: {report.resolution}. "
        elif report.status == "ACKNOWLEDGED":
            explanation += "This issue has been acknowledged but not yet resolved. "
        else:
            explanation += "This issue is currently open and unresolved. "
        
        # Add root cause if available
        if report.root_cause:
            explanation += f"Root cause: {report.root_cause}. "
        
        # Add recommendations if available
        if report.recommendations:
            explanation += "Recommendations: "
            for i, rec in enumerate(report.recommendations[:3]):  # Top 3 recommendations
                if isinstance(rec, dict) and "message" in rec:
                    explanation += f"\n- {rec['message']}"
                elif isinstance(rec, str):
                    explanation += f"\n- {rec}"
        
        return explanation
    
    def _analyze_code_structure(self) -> Dict[str, Any]:
        """
        Analyze code structure of the system.
        
        Returns:
            Code structure information
        """
        # This would be more sophisticated in a real implementation
        return {
            "file_count": 50,
            "class_count": 120,
            "function_count": 350,
            "top_modules": ["auditor", "sensor_registry", "error_report", "sensor_base"]
        }
    
    def _summarize_source(self, source: str) -> str:
        """
        Summarize source code.
        
        Args:
            source: Source code
            
        Returns:
            Summary of source code
        """
        # Extract docstring if present
        if '"""' in source:
            docstring_start = source.find('"""')
            docstring_end = source.find('"""', docstring_start + 3)
            if docstring_end > docstring_start:
                docstring = source[docstring_start + 3:docstring_end].strip()
                return docstring
        
        # Count lines, classes, functions
        lines = source.split('\n')
        class_count = source.count('class ')
        function_count = source.count('def ')
        
        return f"{len(lines)} lines of code with {class_count} classes and {function_count} functions"
