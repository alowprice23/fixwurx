#!/usr/bin/env python3
"""
Auditor Report Query

This module provides an interface for the Meta Agent and other components
to query the Auditor Agent for reports and insights. It enables the
integration of auditor insights into the conversational interface.
"""

import logging
import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AuditorReportQuery")

class AuditorReportQuery:
    """
    Interface for querying the Auditor Agent for reports and insights.
    
    This class provides methods for other components, particularly the
    Meta Agent, to request and receive reports and insights from the
    Auditor Agent, allowing integration of audit data into the
    conversational interface.
    """
    
    def __init__(self, registry: Any = None):
        """
        Initialize the Auditor Report Query interface.
        
        Args:
            registry: Component registry for accessing other components
        """
        self.registry = registry
        self.auditor_agent = self._get_auditor_agent()
        self.alert_system = self._get_alert_system()
        logger.info("Auditor Report Query interface initialized")
    
    def _get_auditor_agent(self) -> Any:
        """
        Get the Auditor Agent instance from the registry.
        
        Returns:
            Auditor Agent instance or None if not available
        """
        if self.registry:
            auditor_agent = self.registry.get_component("auditor_agent")
            if auditor_agent:
                return auditor_agent
        
        logger.warning("Auditor Agent not available")
        return None
    
    def _get_alert_system(self) -> Any:
        """
        Get the Alert System instance from the registry.
        
        Returns:
            Alert System instance or None if not available
        """
        if self.registry:
            alert_system = self.registry.get_component("alert_system")
            if alert_system:
                return alert_system
        
        logger.warning("Alert System not available")
        return None
    
    def get_latest_report(self, report_type: str = "full") -> Dict[str, Any]:
        """
        Get the latest audit report from the Auditor Agent.
        
        Args:
            report_type: Type of report to retrieve (full, compliance, performance)
            
        Returns:
            Latest audit report or empty dict if not available
        """
        if not self.auditor_agent:
            logger.warning("Cannot get latest report: Auditor Agent not available")
            return {"error": "Auditor Agent not available", "timestamp": time.time()}
        
        try:
            # Check if auditor_agent has generate_report method
            if hasattr(self.auditor_agent, "generate_report"):
                return self.auditor_agent.generate_report(report_type)
            else:
                logger.warning("Auditor Agent does not have generate_report method")
                return {"error": "Auditor Agent does not support report generation", "timestamp": time.time()}
        except Exception as e:
            logger.error(f"Error getting latest report: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def get_compliance_status(self, standard: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the current compliance status from the Auditor Agent.
        
        Args:
            standard: Specific compliance standard to check (optional)
            
        Returns:
            Compliance status data or empty dict if not available
        """
        if not self.auditor_agent:
            logger.warning("Cannot get compliance status: Auditor Agent not available")
            return {"error": "Auditor Agent not available", "timestamp": time.time()}
        
        try:
            # Check if auditor_agent has get_compliance_status method
            if hasattr(self.auditor_agent, "get_compliance_status"):
                return self.auditor_agent.get_compliance_status(standard)
            else:
                logger.warning("Auditor Agent does not have get_compliance_status method")
                return {"error": "Auditor Agent does not support compliance status", "timestamp": time.time()}
        except Exception as e:
            logger.error(f"Error getting compliance status: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get the current performance metrics from the Auditor Agent.
        
        Returns:
            Performance metrics data or empty dict if not available
        """
        if not self.auditor_agent:
            logger.warning("Cannot get performance metrics: Auditor Agent not available")
            return {"error": "Auditor Agent not available", "timestamp": time.time()}
        
        try:
            # Check if auditor_agent has get_performance_metrics method
            if hasattr(self.auditor_agent, "get_performance_metrics"):
                return self.auditor_agent.get_performance_metrics()
            else:
                logger.warning("Auditor Agent does not have get_performance_metrics method")
                return {"error": "Auditor Agent does not support performance metrics", "timestamp": time.time()}
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def get_agent_activities(self, agent_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent activities of a specific agent or all agents.
        
        Args:
            agent_type: Type of agent to get activities for (optional)
            limit: Maximum number of activities to return
            
        Returns:
            List of agent activities or empty list if not available
        """
        if not self.auditor_agent:
            logger.warning("Cannot get agent activities: Auditor Agent not available")
            return []
        
        try:
            # Check if auditor_agent has get_agent_activities method
            if hasattr(self.auditor_agent, "get_agent_activities"):
                if agent_type:
                    return self.auditor_agent.get_agent_activities(agent_type, limit)
                else:
                    # Get activities for all agent types
                    all_activities = []
                    agent_types = ["planner", "observer", "analyst", "verifier", "meta"]
                    for agent in agent_types:
                        activities = self.auditor_agent.get_agent_activities(agent, limit)
                        all_activities.extend(activities)
                    
                    # Sort by timestamp (newest first)
                    all_activities.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                    
                    # Apply limit
                    return all_activities[:limit]
            else:
                logger.warning("Auditor Agent does not have get_agent_activities method")
                return []
        except Exception as e:
            logger.error(f"Error getting agent activities: {e}")
            return []
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the current system health status from the Auditor Agent.
        
        Returns:
            System health data including compliance, performance, and alerts
        """
        # Start with basic health info
        health = {
            "timestamp": time.time(),
            "status": "unknown"
        }
        
        # Add compliance data if available
        try:
            compliance = self.get_compliance_status()
            if isinstance(compliance, dict) and "error" not in compliance:
                health["compliance"] = compliance
                
                # Calculate overall compliance status
                compliant_standards = sum(1 for standard in compliance.values() 
                                         if isinstance(standard, dict) and standard.get("compliant", False))
                total_standards = len(compliance)
                
                if total_standards > 0:
                    health["compliance_percentage"] = (compliant_standards / total_standards) * 100
        except Exception as e:
            logger.error(f"Error getting compliance for system health: {e}")
        
        # Add performance data if available
        try:
            performance = self.get_performance_metrics()
            if isinstance(performance, dict) and "error" not in performance:
                health["performance"] = performance
                
                # Determine performance status
                if "cpu_usage" in performance and "memory_usage" in performance:
                    cpu = performance.get("cpu_usage", 0)
                    memory = performance.get("memory_usage", 0)
                    
                    if cpu > 90 or memory > 90:
                        health["performance_status"] = "critical"
                    elif cpu > 75 or memory > 75:
                        health["performance_status"] = "warning"
                    else:
                        health["performance_status"] = "good"
        except Exception as e:
            logger.error(f"Error getting performance for system health: {e}")
        
        # Add alert data if available
        try:
            if self.alert_system:
                # Get alert statistics
                if hasattr(self.alert_system, "get_statistics"):
                    alert_stats = self.alert_system.get_statistics()
                    health["alerts"] = alert_stats
                    
                    # Determine alert status
                    if alert_stats.get("by_severity", {}).get("critical", 0) > 0:
                        health["alert_status"] = "critical"
                    elif alert_stats.get("by_severity", {}).get("high", 0) > 0:
                        health["alert_status"] = "high"
                    elif alert_stats.get("by_severity", {}).get("medium", 0) > 0:
                        health["alert_status"] = "medium"
                    else:
                        health["alert_status"] = "normal"
        except Exception as e:
            logger.error(f"Error getting alerts for system health: {e}")
        
        # Determine overall system status
        try:
            if "alert_status" in health and "performance_status" in health:
                if health["alert_status"] == "critical" or health["performance_status"] == "critical":
                    health["status"] = "critical"
                elif health["alert_status"] == "high" or health["performance_status"] == "warning":
                    health["status"] = "warning"
                else:
                    health["status"] = "normal"
        except Exception as e:
            logger.error(f"Error determining overall system health status: {e}")
        
        return health
    
    def get_latest_alerts(self, severity: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the latest alerts from the Alert System.
        
        Args:
            severity: Filter alerts by severity (optional)
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert objects or empty list if not available
        """
        if not self.alert_system:
            logger.warning("Cannot get latest alerts: Alert System not available")
            return []
        
        try:
            # Check if alert_system has list_alerts method
            if hasattr(self.alert_system, "list_alerts"):
                filters = {}
                if severity:
                    filters["severity"] = severity
                filters["limit"] = limit
                
                return self.alert_system.list_alerts(filters)
            else:
                logger.warning("Alert System does not have list_alerts method")
                return []
        except Exception as e:
            logger.error(f"Error getting latest alerts: {e}")
            return []
    
    def get_insights(self, analysis_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get LLM-powered insights from the Auditor Agent.
        
        Args:
            analysis_type: Type of analysis to perform
            data: Data to analyze
            
        Returns:
            Insights generated by the Auditor Agent's LLM
        """
        if not self.auditor_agent:
            logger.warning("Cannot get insights: Auditor Agent not available")
            return {"error": "Auditor Agent not available", "timestamp": time.time()}
        
        try:
            # Check if auditor_agent has analyze_with_llm method
            if hasattr(self.auditor_agent, "analyze_with_llm"):
                return self.auditor_agent.analyze_with_llm(analysis_type, data)
            else:
                logger.warning("Auditor Agent does not have analyze_with_llm method")
                return {"error": "Auditor Agent does not support LLM analysis", "timestamp": time.time()}
        except Exception as e:
            logger.error(f"Error getting insights: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def generate_ci_report(self) -> Dict[str, Any]:
        """
        Generate a report specifically for the Conversational Interface.
        
        This method combines data from various sources to create a comprehensive
        report that can be presented to the user through the Meta Agent.
        
        Returns:
            Report formatted for presentation in the Conversational Interface
        """
        report = {
            "timestamp": time.time(),
            "report_id": f"ci-report-{int(time.time())}",
            "sections": []
        }
        
        # Add system health section
        try:
            health = self.get_system_health()
            health_section = {
                "title": "System Health",
                "data": health,
                "summary": f"System Status: {health.get('status', 'Unknown').upper()}"
            }
            
            if health.get("status") == "critical":
                health_section["priority"] = "high"
                health_section["color"] = "red"
            elif health.get("status") == "warning":
                health_section["priority"] = "medium"
                health_section["color"] = "yellow"
            else:
                health_section["priority"] = "low"
                health_section["color"] = "green"
            
            report["sections"].append(health_section)
        except Exception as e:
            logger.error(f"Error generating health section for CI report: {e}")
        
        # Add alerts section
        try:
            alerts = self.get_latest_alerts(limit=3)
            if alerts:
                alerts_section = {
                    "title": "Recent Alerts",
                    "data": alerts,
                    "summary": f"{len(alerts)} recent alerts detected"
                }
                
                # Determine priority based on highest severity alert
                highest_severity = max((alert.get("severity", "low") for alert in alerts), 
                                      key=lambda s: ["info", "low", "medium", "high", "critical"].index(s))
                
                if highest_severity in ["critical", "high"]:
                    alerts_section["priority"] = "high"
                    alerts_section["color"] = "red"
                elif highest_severity == "medium":
                    alerts_section["priority"] = "medium"
                    alerts_section["color"] = "yellow"
                else:
                    alerts_section["priority"] = "low"
                    alerts_section["color"] = "green"
                
                report["sections"].append(alerts_section)
        except Exception as e:
            logger.error(f"Error generating alerts section for CI report: {e}")
        
        # Add compliance section
        try:
            compliance = self.get_compliance_status()
            if isinstance(compliance, dict) and "error" not in compliance:
                # Calculate compliance percentage
                compliant_standards = sum(1 for standard in compliance.values() 
                                         if isinstance(standard, dict) and standard.get("compliant", False))
                total_standards = len(compliance) or 1  # Avoid division by zero
                compliance_percentage = (compliant_standards / total_standards) * 100
                
                compliance_section = {
                    "title": "Compliance Status",
                    "data": compliance,
                    "summary": f"Compliance: {compliance_percentage:.1f}% of standards met"
                }
                
                if compliance_percentage < 70:
                    compliance_section["priority"] = "high"
                    compliance_section["color"] = "red"
                elif compliance_percentage < 90:
                    compliance_section["priority"] = "medium"
                    compliance_section["color"] = "yellow"
                else:
                    compliance_section["priority"] = "low"
                    compliance_section["color"] = "green"
                
                report["sections"].append(compliance_section)
        except Exception as e:
            logger.error(f"Error generating compliance section for CI report: {e}")
        
        # Add performance section
        try:
            performance = self.get_performance_metrics()
            if isinstance(performance, dict) and "error" not in performance:
                performance_section = {
                    "title": "Performance Metrics",
                    "data": performance,
                    "summary": "System performance metrics"
                }
                
                # Determine priority based on CPU and memory usage
                cpu = performance.get("cpu_usage", 0)
                memory = performance.get("memory_usage", 0)
                
                if cpu > 90 or memory > 90:
                    performance_section["priority"] = "high"
                    performance_section["color"] = "red"
                    performance_section["summary"] = f"Performance CRITICAL: CPU {cpu:.1f}%, Memory {memory:.1f}%"
                elif cpu > 75 or memory > 75:
                    performance_section["priority"] = "medium"
                    performance_section["color"] = "yellow"
                    performance_section["summary"] = f"Performance WARNING: CPU {cpu:.1f}%, Memory {memory:.1f}%"
                else:
                    performance_section["priority"] = "low"
                    performance_section["color"] = "green"
                    performance_section["summary"] = f"Performance GOOD: CPU {cpu:.1f}%, Memory {memory:.1f}%"
                
                report["sections"].append(performance_section)
        except Exception as e:
            logger.error(f"Error generating performance section for CI report: {e}")
        
        # Sort sections by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        report["sections"].sort(key=lambda s: priority_order.get(s.get("priority"), 999))
        
        # Add overall summary
        try:
            high_priority_sections = sum(1 for s in report["sections"] if s.get("priority") == "high")
            medium_priority_sections = sum(1 for s in report["sections"] if s.get("priority") == "medium")
            
            if high_priority_sections > 0:
                report["summary"] = f"ATTENTION REQUIRED: {high_priority_sections} critical issues detected"
                report["status"] = "critical"
            elif medium_priority_sections > 0:
                report["summary"] = f"CAUTION: {medium_priority_sections} warnings detected"
                report["status"] = "warning"
            else:
                report["summary"] = "All systems operating normally"
                report["status"] = "normal"
        except Exception as e:
            logger.error(f"Error generating summary for CI report: {e}")
            report["summary"] = "System status report generated"
            report["status"] = "unknown"
        
        return report

# Singleton instance
_instance = None

def get_instance(registry: Any = None) -> AuditorReportQuery:
    """
    Get or create the singleton instance of the Auditor Report Query interface.
    
    Args:
        registry: Component registry
        
    Returns:
        AuditorReportQuery instance
    """
    global _instance
    if _instance is None:
        _instance = AuditorReportQuery(registry)
    return _instance

def register(registry: Any) -> None:
    """
    Register the Auditor Report Query interface with the component registry.
    
    Args:
        registry: Component registry
    """
    if registry:
        instance = get_instance(registry)
        registry.register_component("auditor_report_query", instance)
        logger.info("Registered Auditor Report Query interface with component registry")
