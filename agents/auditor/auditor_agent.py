#!/usr/bin/env python3
"""
Auditor Agent

This module provides the Auditor Agent, which is responsible for monitoring, 
logging, and auditing the activities of all other agents in the FixWurx system.
It ensures compliance with security and performance standards, and provides
insights into system operation.
"""

import logging
import os
import sys
import time
import json
import openai
import threading
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AuditorAgent")

# OpenAI configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-4o"

class AuditorAgent:
    """
    Auditor Agent class that monitors and audits the FixWurx system.
    
    This agent is responsible for:
    1. Monitoring agent activities
    2. Logging system events
    3. Ensuring compliance with security standards
    4. Auditing performance metrics
    5. Generating audit reports
    6. Detecting anomalies in system behavior
    7. Providing insights through LLM analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Auditor Agent.
        
        Args:
            config: Configuration dictionary for the Auditor Agent
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.audit_interval = self.config.get("audit_interval", 300)  # 5 minutes
        self.log_retention_days = self.config.get("log_retention_days", 30)
        self.compliance_standards = self.config.get("compliance_standards", ["security", "performance", "reliability"])
        
        # Initialize storage
        self.audit_storage_path = Path(self.config.get("audit_storage_path", ".triangulum/auditor"))
        self.audit_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "audit_cycles": 0,
            "compliance_checks": 0,
            "anomalies_detected": 0,
            "reports_generated": 0,
            "events_logged": 0,
            "agents_monitored": 0
        }
        
        # Initialize LLM configuration
        self.llm_config = {
            "model": self.config.get("llm_model", DEFAULT_MODEL),
            "temperature": self.config.get("llm_temperature", 0.2),
            "max_tokens": self.config.get("llm_max_tokens", 1000),
            "top_p": self.config.get("llm_top_p", 1.0),
            "presence_penalty": self.config.get("llm_presence_penalty", 0.0),
            "frequency_penalty": self.config.get("llm_frequency_penalty", 0.0)
        }
        
        # Initialize OpenAI client if key is available
        self.openai_client = None
        if OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Initialize audit data
        self.agent_activities = {}
        self.system_events = []
        self.compliance_results = {}
        self.performance_metrics = {}
        
        # Initialize audit thread
        self._shutdown = threading.Event()
        self._audit_thread = None
        
        # Initialize sensors
        self.sensors = self._initialize_sensors()
        
        logger.info("Auditor Agent initialized")
    
    def start_auditing(self) -> bool:
        """
        Start the auditing thread.
        
        Returns:
            True if started successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Auditor Agent is disabled")
            return False
        
        if self._audit_thread is not None and self._audit_thread.is_alive():
            logger.warning("Audit thread already running")
            return False
        
        try:
            self._shutdown.clear()
            self._audit_thread = threading.Thread(target=self._audit_loop)
            self._audit_thread.daemon = True
            self._audit_thread.start()
            logger.info("Audit thread started")
            return True
        except Exception as e:
            logger.error(f"Error starting audit thread: {e}")
            return False
    
    def stop_auditing(self) -> bool:
        """
        Stop the auditing thread.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if self._audit_thread is None or not self._audit_thread.is_alive():
            logger.warning("Audit thread not running")
            return False
        
        try:
            self._shutdown.set()
            self._audit_thread.join(timeout=2.0)
            logger.info("Audit thread stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping audit thread: {e}")
            return False
    
    def _audit_loop(self) -> None:
        """
        Main audit loop.
        
        This loop runs in a separate thread and periodically
        audits the system, checking for compliance, monitoring
        performance, and generating reports.
        """
        while not self._shutdown.is_set():
            try:
                # Perform audit activities
                self._collect_agent_activities()
                self._check_compliance()
                self._monitor_performance()
                self._detect_anomalies()
                
                # Generate reports periodically
                if self.metrics["audit_cycles"] % 5 == 0:  # Every 5 cycles
                    self._generate_audit_report()
                
                # Update metrics
                self.metrics["audit_cycles"] += 1
                
                # Wait for next cycle
                self._shutdown.wait(self.audit_interval)
            except Exception as e:
                logger.error(f"Error in audit loop: {e}")
                # Brief pause before continuing
                time.sleep(1.0)
    
    def _initialize_sensors(self) -> Dict[str, Any]:
        """
        Initialize sensors for monitoring different aspects of the system.
        
        Returns:
            Dictionary of sensors
        """
        sensors = {}
        
        # Initialize sensors based on configuration
        sensor_configs = self.config.get("sensors", {})
        for sensor_type, sensor_config in sensor_configs.items():
            if sensor_config.get("enabled", True):
                try:
                    # In a real implementation, this would create actual sensor instances
                    sensors[sensor_type] = {
                        "type": sensor_type,
                        "config": sensor_config,
                        "enabled": True,
                        "last_reading": None
                    }
                    logger.info(f"Initialized {sensor_type} sensor")
                except Exception as e:
                    logger.error(f"Error initializing {sensor_type} sensor: {e}")
        
        return sensors
    
    def _collect_agent_activities(self) -> None:
        """
        Collect activities from all agents in the system.
        """
        # In a real implementation, this would collect actual agent activities
        # For now, we'll just simulate activity collection
        
        # Simulated agent types
        agent_types = ["planner", "observer", "analyst", "verifier", "meta"]
        
        # Collect activities for each agent type
        for agent_type in agent_types:
            if agent_type not in self.agent_activities:
                self.agent_activities[agent_type] = []
            
            # Add a simulated activity
            self.agent_activities[agent_type].append({
                "timestamp": time.time(),
                "type": "heartbeat",
                "details": f"{agent_type.capitalize()} agent active"
            })
            
            # Limit activity history
            max_activities = self.config.get("max_activity_history", 100)
            if len(self.agent_activities[agent_type]) > max_activities:
                self.agent_activities[agent_type] = self.agent_activities[agent_type][-max_activities:]
        
        # Update metrics
        self.metrics["agents_monitored"] = len(self.agent_activities)
        self.metrics["events_logged"] += len(agent_types)
    
    def _check_compliance(self) -> None:
        """
        Check compliance with security and performance standards.
        """
        # In a real implementation, this would check actual compliance
        # For now, we'll just simulate compliance checks
        
        for standard in self.compliance_standards:
            # Simulate compliance check
            compliant = True  # Assume compliant
            findings = []
            
            # Simulate random findings
            if standard == "security" and time.time() % 60 < 2:  # Occasional security finding
                compliant = False
                findings.append({
                    "severity": "medium",
                    "description": "Simulated security finding",
                    "recommendation": "Apply security patches"
                })
            
            # Store compliance result
            self.compliance_results[standard] = {
                "timestamp": time.time(),
                "compliant": compliant,
                "findings": findings
            }
        
        # Update metrics
        self.metrics["compliance_checks"] += len(self.compliance_standards)
    
    def _monitor_performance(self) -> None:
        """
        Monitor system performance metrics.
        """
        # In a real implementation, this would monitor actual performance
        # For now, we'll just simulate performance monitoring
        
        # Simulated performance metrics
        self.performance_metrics = {
            "cpu_usage": 30 + (time.time() % 20),  # 30-50%
            "memory_usage": 40 + (time.time() % 30),  # 40-70%
            "api_latency": 50 + (time.time() % 100),  # 50-150ms
            "throughput": 100 + (time.time() % 50),  # 100-150 req/s
            "timestamp": time.time()
        }
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect anomalies in system behavior using LLM analysis.
        
        Returns:
            List of detected anomalies
        """
        # Skip if LLM is not available
        if not self.openai_client:
            return []
        
        # Collect system data for anomaly detection
        system_data = {
            "agent_activities": {
                agent: len(activities) for agent, activities in self.agent_activities.items()
            },
            "performance": self.performance_metrics,
            "compliance": {
                standard: result["compliant"] for standard, result in self.compliance_results.items() if standard != "security"
            }
        }
        
        # Create prompt for anomaly detection
        prompt = f"""
        As an Auditor Agent, analyze the following system data for anomalies:
        
        System Data:
        {json.dumps(system_data, indent=2)}
        
        Please detect any anomalies in:
        1. Agent activity patterns
        2. Performance metrics
        3. Compliance status
        
        Format your response as a JSON array of anomalies, where each anomaly is an object with:
        - type: The type of anomaly (activity, performance, compliance)
        - severity: The severity of the anomaly (low, medium, high)
        - description: A description of the anomaly
        - recommendation: A recommendation for addressing the anomaly
        """
        
        # Call LLM for anomaly detection
        anomalies = []
        try:
            response = self._call_llm(prompt)
            
            # Parse response
            try:
                # Look for JSON array in the response
                json_start = response.find("[")
                json_end = response.rfind("]") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = response[json_start:json_end]
                    anomalies = json.loads(json_content)
                else:
                    # Try to parse the entire response as JSON
                    anomalies = json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM anomaly detection response as JSON")
                anomalies = []
            
            # Update metrics
            if anomalies:
                self.metrics["anomalies_detected"] += len(anomalies)
                
                # Log anomalies
                for anomaly in anomalies:
                    logger.warning(f"Anomaly detected: {anomaly.get('description')}")
                
                # Store anomalies
                anomaly_path = self.audit_storage_path / f"anomalies_{int(time.time())}.json"
                with open(anomaly_path, 'w') as f:
                    json.dump(anomalies, f, indent=2)
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    def _generate_audit_report(self) -> Dict[str, Any]:
        """
        Generate an audit report using LLM analysis.
        
        Returns:
            Audit report dictionary
        """
        # Skip if LLM is not available
        if not self.openai_client:
            # Generate a basic report without LLM
            report = {
                "timestamp": time.time(),
                "report_id": f"report-{int(time.time())}",
                "audit_cycles": self.metrics["audit_cycles"],
                "agents_monitored": self.metrics["agents_monitored"],
                "compliance_status": {
                    standard: result["compliant"] for standard, result in self.compliance_results.items()
                },
                "performance_summary": self.performance_metrics,
                "anomalies_detected": self.metrics["anomalies_detected"]
            }
            
            # Store report
            report_path = self.audit_storage_path / f"report_{report['report_id']}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Update metrics
            self.metrics["reports_generated"] += 1
            
            return report
        
        # Collect system data for report generation
        system_data = {
            "agent_activities": {
                agent: len(activities) for agent, activities in self.agent_activities.items()
            },
            "performance": self.performance_metrics,
            "compliance": {
                standard: result["compliant"] for standard, result in self.compliance_results.items()
            },
            "metrics": self.metrics
        }
        
        # Create prompt for report generation
        prompt = f"""
        As an Auditor Agent, generate a comprehensive audit report based on the following system data:
        
        System Data:
        {json.dumps(system_data, indent=2)}
        
        Please include:
        1. Executive summary
        2. Compliance status for each standard
        3. Performance analysis
        4. Agent activity summary
        5. Key findings
        6. Recommendations
        
        Format your response as a JSON object with the following keys:
        - report_id: A unique identifier for the report
        - timestamp: The current timestamp
        - executive_summary: A brief summary of the audit
        - compliance_status: Object with compliance status for each standard
        - performance_analysis: Analysis of performance metrics
        - agent_activity: Summary of agent activities
        - key_findings: Array of key findings
        - recommendations: Array of recommendations
        """
        
        # Call LLM for report generation
        try:
            response = self._call_llm(prompt)
            
            # Parse response
            try:
                # Look for JSON object in the response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = response[json_start:json_end]
                    report = json.loads(json_content)
                else:
                    # Try to parse the entire response as JSON
                    report = json.loads(response)
                
                # Add metadata
                report["generated_by"] = "Auditor Agent"
                report["generated_at"] = time.time()
                
                # Store report
                report_path = self.audit_storage_path / f"report_{report.get('report_id', int(time.time()))}.json"
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                # Update metrics
                self.metrics["reports_generated"] += 1
                
                return report
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM report generation response as JSON")
        except Exception as e:
            logger.error(f"Error generating audit report: {e}")
        
        # Fallback to basic report
        return self._generate_basic_report()
    
    def _generate_basic_report(self) -> Dict[str, Any]:
        """
        Generate a basic audit report without LLM.
        
        Returns:
            Basic audit report dictionary
        """
        report = {
            "timestamp": time.time(),
            "report_id": f"report-{int(time.time())}",
            "generated_by": "Auditor Agent",
            "executive_summary": "Basic audit report generated without LLM",
            "audit_cycles": self.metrics["audit_cycles"],
            "agents_monitored": self.metrics["agents_monitored"],
            "compliance_status": {
                standard: result["compliant"] for standard, result in self.compliance_results.items()
            },
            "performance_summary": self.performance_metrics,
            "anomalies_detected": self.metrics["anomalies_detected"]
        }
        
        # Store report
        report_path = self.audit_storage_path / f"report_{report['report_id']}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Update metrics
        self.metrics["reports_generated"] += 1
        
        return report
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with a prompt and get a response.
        
        Args:
            prompt: Prompt string for the LLM
            
        Returns:
            Response string from the LLM
        """
        if not self.openai_client:
            return ""
        
        try:
            # Make the API call
            response = self.openai_client.chat.completions.create(
                model=self.llm_config["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant for an Auditor Agent that monitors and audits a system of agents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"],
                top_p=self.llm_config["top_p"],
                presence_penalty=self.llm_config["presence_penalty"],
                frequency_penalty=self.llm_config["frequency_penalty"]
            )
            
            # Extract response text
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return ""
    
    def log_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        Log an event in the audit log.
        
        Args:
            event_type: Type of event
            event_data: Event data
            
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            # Create event record
            event = {
                "timestamp": time.time(),
                "type": event_type,
                "data": event_data
            }
            
            # Add to system events
            self.system_events.append(event)
            
            # Limit event history
            max_events = self.config.get("max_event_history", 1000)
            if len(self.system_events) > max_events:
                self.system_events = self.system_events[-max_events:]
            
            # Log to file
            event_path = self.audit_storage_path / f"events_{time.strftime('%Y%m%d')}.jsonl"
            with open(event_path, 'a') as f:
                f.write(json.dumps(event) + "\n")
            
            # Update metrics
            self.metrics["events_logged"] += 1
            
            return True
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            return False
    
    def get_agent_activities(self, agent_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the activities of a specific agent.
        
        Args:
            agent_type: Type of agent
            limit: Maximum number of activities to return
            
        Returns:
            List of agent activities
        """
        if agent_type not in self.agent_activities:
            return []
        
        activities = self.agent_activities[agent_type]
        return activities[-limit:] if limit > 0 else activities
    
    def get_compliance_status(self, standard: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Get the compliance status for a specific standard or all standards.
        
        Args:
            standard: Specific compliance standard (optional)
            
        Returns:
            Compliance status dictionary
        """
        if standard:
            return self.compliance_results.get(standard, {})
        else:
            return self.compliance_results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get the current performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        return self.performance_metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the Auditor Agent metrics.
        
        Returns:
            Metrics dictionary
        """
        return self.metrics
    
    def generate_report(self, report_type: str = "full") -> Dict[str, Any]:
        """
        Generate an audit report.
        
        Args:
            report_type: Type of report to generate
            
        Returns:
            Audit report dictionary
        """
        if report_type == "full":
            return self._generate_audit_report()
        elif report_type == "compliance":
            # Generate compliance-focused report
            return {
                "timestamp": time.time(),
                "report_id": f"compliance-{int(time.time())}",
                "report_type": "compliance",
                "compliance_status": self.compliance_results
            }
        elif report_type == "performance":
            # Generate performance-focused report
            return {
                "timestamp": time.time(),
                "report_id": f"performance-{int(time.time())}",
                "report_type": "performance",
                "performance_metrics": self.performance_metrics
            }
        else:
            logger.warning(f"Unknown report type: {report_type}")
            return self._generate_audit_report()
    
    def analyze_with_llm(self, analysis_type: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data using LLM.
        
        Args:
            analysis_type: Type of analysis to perform
            analysis_data: Data to analyze
            
        Returns:
            Analysis results
        """
        if not self.openai_client:
            return {
                "error": "LLM not available",
                "analysis_type": analysis_type,
                "timestamp": time.time()
            }
        
        try:
            # Create prompt based on analysis type
            if analysis_type == "anomaly":
                prompt = f"""
                As an Auditor Agent, analyze the following data for anomalies:
                
                {json.dumps(analysis_data, indent=2)}
                
                Please identify any anomalies in the data and provide a detailed analysis.
                Format your response as a JSON object with the following keys:
                - anomalies_found: boolean indicating whether anomalies were found
                - anomaly_count: number of anomalies found
                - anomalies: array of anomaly objects, each with type, severity, description, and recommendation
                - summary: brief summary of the findings
                """
            elif analysis_type == "compliance":
                prompt = f"""
                As an Auditor Agent, analyze the following data for compliance issues:
                
                {json.dumps(analysis_data, indent=2)}
                
                Please identify any compliance issues in the data and provide a detailed analysis.
                Format your response as a JSON object with the following keys:
                - compliant: boolean indicating whether the data is compliant
                - issue_count: number of compliance issues found
                - issues: array of compliance issue objects, each with standard, severity, description, and recommendation
                - summary: brief summary of the findings
                """
            elif analysis_type == "performance":
                prompt = f"""
                As an Auditor Agent, analyze the following performance data:
                
                {json.dumps(analysis_data, indent=2)}
                
                Please identify any performance issues in the data and provide a detailed analysis.
                Format your response as a JSON object with the following keys:
                - performance_rating: rating from 1-10 of overall performance
                - issue_count: number of performance issues found
                - issues: array of performance issue objects, each with metric, severity, description, and recommendation
                - optimization_opportunities: array of optimization opportunity objects
                - summary: brief summary of the findings
                """
            else:
                prompt = f"""
                As an Auditor Agent, analyze the following data:
                
                {json.dumps(analysis_data, indent=2)}
                
                Please provide a detailed analysis of the data.
                Format your response as a JSON object with relevant analysis findings.
                """
            
            # Call LLM
            response = self._call_llm(prompt)
            
            # Parse response
            try:
                # Look for JSON object in the response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = response[json_start:json_end]
                    analysis = json.loads(json_content)
                else:
                    # Try to parse the entire response as JSON
                    analysis = json.loads(response)
                
                # Add metadata
                analysis["analysis_type"] = analysis_type
                analysis["analyzed_at"] = time.time()
                
                return analysis
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM analysis response as JSON")
                return {
                    "error": "Failed to parse LLM response",
                    "analysis_type": analysis_type,
                    "raw_response": response,
                    "timestamp": time.time()
                }
        except Exception as e:
            logger.error(f"Error analyzing with LLM: {e}")
            return {
                "error": str(e),
                "analysis_type": analysis_type,
                "timestamp": time.time()
            }

# Create a singleton instance
_instance = None

def get_instance(config: Optional[Dict[str, Any]] = None) -> AuditorAgent:
    """
    Get the singleton instance of the Auditor Agent.
    
    Args:
        config: Configuration dictionary for the Auditor Agent
        
    Returns:
        AuditorAgent instance
    """
    global _instance
    if _instance is None:
        _instance = AuditorAgent(config)
    return _instance
