#!/usr/bin/env python3
"""
FixWurx Advanced Error Analysis

This module implements advanced error analysis capabilities for the FixWurx Auditor Agent,
including root cause analysis, impact assessment, error correlation, and resolution tracking.
"""

import os
import sys
import logging
import json
import yaml
import time
import datetime
import re
import networkx as nx
from typing import Dict, List, Set, Any, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [ErrorAnalysis] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('error_analysis')


class ErrorRepository:
    """
    Centralized repository for error storage and retrieval.
    """
    
    def __init__(self, document_store=None, graph_db=None):
        """
        Initialize the Error Repository.
        
        Args:
            document_store: Document store instance for error storage
            graph_db: Graph database instance for error relationships
        """
        self.document_store = document_store
        self.graph_db = graph_db
        self.local_cache = {}
        
        # Create necessary collections if document store is available
        if self.document_store:
            try:
                if not self.document_store.collection_exists("errors"):
                    self.document_store.create_collection("errors")
                if not self.document_store.collection_exists("error_analyses"):
                    self.document_store.create_collection("error_analyses")
            except Exception as e:
                logger.error(f"Failed to create error collections: {e}")
        
        logger.info("Error Repository initialized")
    
    def store_error(self, error_data: Dict[str, Any]) -> str:
        """
        Store an error in the repository.
        
        Args:
            error_data: Error data dictionary
        
        Returns:
            Error ID
        """
        # Generate error ID if not provided
        if "error_id" not in error_data:
            error_data["error_id"] = self._generate_error_id(error_data)
        
        # Add timestamp if not provided
        if "timestamp" not in error_data:
            error_data["timestamp"] = datetime.datetime.now().isoformat()
        
        # Store in document store if available
        if self.document_store:
            try:
                self.document_store.create_document(
                    collection_name="errors",
                    doc_id=error_data["error_id"],
                    fields=error_data
                )
                logger.info(f"Stored error: {error_data['error_id']}")
            except Exception as e:
                logger.error(f"Failed to store error: {e}")
                # Fall back to local cache
                self.local_cache[error_data["error_id"]] = error_data
        else:
            # Store in local cache
            self.local_cache[error_data["error_id"]] = error_data
        
        # Create graph relationships if graph DB is available
        if self.graph_db:
            try:
                # Add error node
                self.graph_db.add_node(
                    node_id=error_data["error_id"],
                    node_type="error",
                    properties={"severity": error_data.get("severity", "unknown"),
                               "status": error_data.get("status", "open")}
                )
                
                # Connect to component
                if "component" in error_data:
                    self.graph_db.add_edge(
                        from_id=error_data["component"],
                        to_id=error_data["error_id"],
                        edge_type="has_error",
                        properties={}
                    )
                
                # Connect to related errors
                if "related_errors" in error_data:
                    for related_id in error_data["related_errors"]:
                        self.graph_db.add_edge(
                            from_id=error_data["error_id"],
                            to_id=related_id,
                            edge_type="related_to",
                            properties={}
                        )
                
                logger.info(f"Created graph relationships for error: {error_data['error_id']}")
            except Exception as e:
                logger.error(f"Failed to create graph relationships: {e}")
        
        return error_data["error_id"]
    
    def get_error(self, error_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an error from the repository.
        
        Args:
            error_id: Error ID
        
        Returns:
            Error data dictionary or None if not found
        """
        # Try document store first
        if self.document_store:
            try:
                error = self.document_store.get_document(
                    collection_name="errors",
                    doc_id=error_id
                )
                if error:
                    return error
            except Exception as e:
                logger.error(f"Failed to get error from document store: {e}")
        
        # Fall back to local cache
        return self.local_cache.get(error_id)
    
    def update_error(self, error_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an error in the repository.
        
        Args:
            error_id: Error ID
            updates: Updates to apply
        
        Returns:
            True if update was successful, False otherwise
        """
        # Get current error data
        error = self.get_error(error_id)
        if not error:
            logger.warning(f"Error not found: {error_id}")
            return False
        
        # Apply updates
        error.update(updates)
        
        # Update in document store if available
        if self.document_store:
            try:
                self.document_store.update_document(
                    collection_name="errors",
                    doc_id=error_id,
                    updates=updates
                )
                logger.info(f"Updated error: {error_id}")
                success = True
            except Exception as e:
                logger.error(f"Failed to update error in document store: {e}")
                success = False
        else:
            # Update in local cache
            self.local_cache[error_id] = error
            success = True
        
        # Update graph relationships if graph DB is available
        if self.graph_db and success:
            try:
                # Update error node properties
                self.graph_db.update_node(
                    node_id=error_id,
                    properties={"severity": error.get("severity", "unknown"),
                               "status": error.get("status", "open")}
                )
                
                # Update related errors
                if "related_errors" in updates:
                    # Get current related errors
                    current_related = set(self.graph_db.get_edges(
                        from_id=error_id,
                        edge_type="related_to"
                    ))
                    
                    # Add new relationships
                    for related_id in updates["related_errors"]:
                        if related_id not in current_related:
                            self.graph_db.add_edge(
                                from_id=error_id,
                                to_id=related_id,
                                edge_type="related_to",
                                properties={}
                            )
                
                logger.info(f"Updated graph relationships for error: {error_id}")
            except Exception as e:
                logger.error(f"Failed to update graph relationships: {e}")
        
        return success
    
    def query_errors(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query errors in the repository.
        
        Args:
            query: Query dictionary
        
        Returns:
            List of matching error data dictionaries
        """
        # Try document store first
        if self.document_store:
            try:
                errors = self.document_store.query_documents(
                    collection_name="errors",
                    query=query
                )
                if errors:
                    return errors
            except Exception as e:
                logger.error(f"Failed to query errors from document store: {e}")
        
        # Fall back to local cache
        results = []
        for error in self.local_cache.values():
            matches = True
            for key, value in query.items():
                if key not in error or error[key] != value:
                    matches = False
                    break
            if matches:
                results.append(error)
        
        return results
    
    def get_related_errors(self, error_id: str) -> List[Dict[str, Any]]:
        """
        Get errors related to the given error.
        
        Args:
            error_id: Error ID
        
        Returns:
            List of related error data dictionaries
        """
        related_errors = []
        
        # Try graph DB first
        if self.graph_db:
            try:
                related_ids = self.graph_db.get_edges(
                    from_id=error_id,
                    edge_type="related_to"
                )
                
                for related_id in related_ids:
                    related_error = self.get_error(related_id)
                    if related_error:
                        related_errors.append(related_error)
            except Exception as e:
                logger.error(f"Failed to get related errors from graph DB: {e}")
        
        # Fall back to data in error record
        if not related_errors:
            error = self.get_error(error_id)
            if error and "related_errors" in error:
                for related_id in error["related_errors"]:
                    related_error = self.get_error(related_id)
                    if related_error:
                        related_errors.append(related_error)
        
        return related_errors
    
    def _generate_error_id(self, error_data: Dict[str, Any]) -> str:
        """
        Generate a unique error ID.
        
        Args:
            error_data: Error data dictionary
        
        Returns:
            Generated error ID
        """
        # Format: ERR-YYYYMMDD-HASH
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        
        # Create a hash from component and message
        component = error_data.get("component", "unknown")
        message = error_data.get("message", "unknown")
        import hashlib
        hash_input = f"{component}:{message}".encode('utf-8')
        hash_str = hashlib.md5(hash_input).hexdigest()[:8]
        
        return f"ERR-{date_str}-{hash_str}"


class ErrorAnalyzer:
    """
    Advanced error analyzer for detecting patterns, conducting root cause analysis,
    and assessing impact.
    """
    
    def __init__(self, error_repository: ErrorRepository, document_store=None, graph_db=None, time_series_db=None):
        """
        Initialize the Error Analyzer.
        
        Args:
            error_repository: Error repository instance
            document_store: Document store instance
            graph_db: Graph database instance
            time_series_db: Time series database instance
        """
        self.error_repository = error_repository
        self.document_store = document_store
        self.graph_db = graph_db
        self.time_series_db = time_series_db
        
        # Load analysis rules
        self.rules = self._load_analysis_rules()

    def _load_analysis_rules(self):
        # Simple implementation for testing
        rules = {
            "root_cause": {
                "patterns": [
                    {"pattern": "null|none", "cause_type": "null_reference", "confidence": 0.8},
                    {"pattern": "timeout", "cause_type": "timeout", "confidence": 0.8}
                ]
            },
            "impact": {
                "severity_weights": {
                    "critical": 1.0,
                    "high": 0.8,
                    "medium": 0.5,
                    "low": 0.2
                }
            }
        }
        return rules

    def _find_related_errors(self, error):
        # Simple implementation for testing
        related_errors = []
        if "component" in error:
            component_errors = self.error_repository.query_errors({"component": error["component"]})
            for comp_error in component_errors:
                if comp_error["error_id"] != error["error_id"]:
                    related_errors.append(comp_error)
        return related_errors[:10]

    def _generate_recommendations(self, error, root_cause, impact):
        # Simple implementation for testing
        recommendations = ["Add comprehensive error handling"]
        return recommendations

    def _generate_report_recommendations(self, trends, patterns, errors):
        # Simple implementation for testing
        recommendations = ["Implement comprehensive error handling across all components"]
        return recommendations

    def _load_analysis_rules(self):
        # Simple implementation for testing
        rules = {
            "root_cause": {
                "patterns": [
                    {"pattern": "null|none", "cause_type": "null_reference", "confidence": 0.8},
                    {"pattern": "timeout", "cause_type": "timeout", "confidence": 0.8}
                ]
            },
            "impact": {
                "severity_weights": {
                    "critical": 1.0,
                    "high": 0.8,
                    "medium": 0.5,
                    "low": 0.2
                }
            }
        }
        return rules

    def _find_related_errors(self, error):
        # Simple implementation for testing
        related_errors = []
        if "component" in error:
            component_errors = self.error_repository.query_errors({"component": error["component"]})
            for comp_error in component_errors:
                if comp_error["error_id"] != error["error_id"]:
                    related_errors.append(comp_error)
        return related_errors[:10]

    def _generate_recommendations(self, error, root_cause, impact):
        # Simple implementation for testing
        recommendations = ["Add comprehensive error handling"]
        return recommendations

    def _generate_report_recommendations(self, trends, patterns, errors):
        # Simple implementation for testing
        recommendations = ["Implement comprehensive error handling across all components"]
        return recommendations
        
        # Initialize analysis graph
        self.analysis_graph = nx.DiGraph()
        
        logger.info("Error Analyzer initialized")
    
    def analyze_error(self, error_id: str) -> Dict[str, Any]:
        """
        Analyze an error to determine root cause, impact, and relationships.
        
        Args:
            error_id: Error ID
        
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing error: {error_id}")
        
        # Get error data
        error = self.error_repository.get_error(error_id)
        if not error:
            logger.warning(f"Error not found: {error_id}")
            return {"error": "Error not found"}
        
        # Perform root cause analysis
        root_cause = self._perform_root_cause_analysis(error)
        
        # Assess impact
        impact = self._assess_impact(error)
        
        # Find related errors
        related_errors = self._find_related_errors(error)
        
        # Compile analysis results
        analysis = {
            "error_id": error_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "root_cause": root_cause,
            "impact": impact,
            "related_errors": related_errors,
            "recommendations": self._generate_recommendations(error, root_cause, impact)
        }
        
        # Store analysis results
        if self.document_store:
            try:
                self.document_store.create_document(
                    collection_name="error_analyses",
                    doc_id=f"ANALYSIS-{error_id}",
                    fields=analysis
                )
                logger.info(f"Stored analysis for error: {error_id}")
            except Exception as e:
                logger.error(f"Failed to store analysis: {e}")
        
        # Update error with related errors
        related_ids = [r["error_id"] for r in related_errors]
        self.error_repository.update_error(error_id, {"related_errors": related_ids})
        
        return analysis
    
    def batch_analyze_errors(self, query: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a batch of errors.
        
        Args:
            query: Query to filter errors (optional)
        
        Returns:
            Dictionary mapping error IDs to analysis results
        """
        logger.info("Batch analyzing errors")
        
        # Get errors to analyze
        if query:
            errors = self.error_repository.query_errors(query)
        else:
            # Get all errors with status "open"
            errors = self.error_repository.query_errors({"status": "open"})
        
        # Analyze each error
        results = {}
        for error in errors:
            error_id = error["error_id"]
            results[error_id] = self.analyze_error(error_id)
        
        logger.info(f"Batch analyzed {len(results)} errors")
        return results
    
    def identify_error_trends(self) -> Dict[str, Any]:
        """
        Identify trends in errors over time.
        
        Returns:
            Trend analysis results
        """
        logger.info("Identifying error trends")
        
        # Get all errors
        errors = self.error_repository.query_errors({})
        
        # Group errors by component
        component_errors = {}
        for error in errors:
            component = error.get("component", "unknown")
            if component not in component_errors:
                component_errors[component] = []
            component_errors[component].append(error)
        
        # Group errors by date
        date_errors = {}
        for error in errors:
            timestamp = error.get("timestamp", datetime.datetime.now().isoformat())
            date = timestamp.split("T")[0]
            if date not in date_errors:
                date_errors[date] = []
            date_errors[date].append(error)
        
        # Calculate trends
        component_trends = {}
        for component, component_errors_list in component_errors.items():
            component_trends[component] = {
                "total": len(component_errors_list),
                "open": sum(1 for e in component_errors_list if e.get("status") == "open"),
                "resolved": sum(1 for e in component_errors_list if e.get("status") == "resolved"),
                "by_severity": {
                    "critical": sum(1 for e in component_errors_list if e.get("severity") == "critical"),
                    "high": sum(1 for e in component_errors_list if e.get("severity") == "high"),
                    "medium": sum(1 for e in component_errors_list if e.get("severity") == "medium"),
                    "low": sum(1 for e in component_errors_list if e.get("severity") == "low")
                }
            }
        
        date_trends = {}
        for date, date_errors_list in sorted(date_errors.items()):
            date_trends[date] = {
                "total": len(date_errors_list),
                "by_severity": {
                    "critical": sum(1 for e in date_errors_list if e.get("severity") == "critical"),
                    "high": sum(1 for e in date_errors_list if e.get("severity") == "high"),
                    "medium": sum(1 for e in date_errors_list if e.get("severity") == "medium"),
                    "low": sum(1 for e in date_errors_list if e.get("severity") == "low")
                }
            }
        
        # Store trend data in time series database if available
        if self.time_series_db:
            try:
                # Store daily error counts
                for date, trends in date_trends.items():
                    timestamp = datetime.datetime.fromisoformat(f"{date}T00:00:00")
                    self.time_series_db.add_point(
                        series_name="error_counts",
                        timestamp=timestamp,
                        values={
                            "total": trends["total"],
                            "critical": trends["by_severity"]["critical"],
                            "high": trends["by_severity"]["high"],
                            "medium": trends["by_severity"]["medium"],
                            "low": trends["by_severity"]["low"]
                        }
                    )
                logger.info("Stored error trends in time series database")
            except Exception as e:
                logger.error(f"Failed to store error trends: {e}")
        
        return {
            "by_component": component_trends,
            "by_date": date_trends,
            "summary": {
                "total_errors": len(errors),
                "open_errors": sum(1 for e in errors if e.get("status") == "open"),
                "resolved_errors": sum(1 for e in errors if e.get("status") == "resolved"),
                "by_severity": {
                    "critical": sum(1 for e in errors if e.get("severity") == "critical"),
                    "high": sum(1 for e in errors if e.get("severity") == "high"),
                    "medium": sum(1 for e in errors if e.get("severity") == "medium"),
                    "low": sum(1 for e in errors if e.get("severity") == "low")
                }
            }
        }
    
    def identify_error_patterns(self) -> List[Dict[str, Any]]:
        """
        Identify recurring patterns in errors.
        
        Returns:
            List of identified patterns
        """
        logger.info("Identifying error patterns")
        
        # Get all errors
        errors = self.error_repository.query_errors({})
        
        # Group errors by message pattern
        message_patterns = {}
        for error in errors:
            message = error.get("message", "")
            # Remove specific values (numbers, IDs) to identify patterns
            pattern = re.sub(r'\b\d+\b', 'NUM', message)
            pattern = re.sub(r'\b[0-9a-f]{8,}\b', 'ID', pattern)
            
            if pattern not in message_patterns:
                message_patterns[pattern] = []
            message_patterns[pattern].append(error)
        
        # Group errors by stack trace pattern
        stack_patterns = {}
        for error in errors:
            stack_trace = error.get("stack_trace", "")
            if not stack_trace:
                continue
            
            # Extract function calls from stack trace
            matches = re.findall(r'File "([^"]+)", line \d+, in (\w+)', stack_trace)
            if not matches:
                continue
            
            # Create pattern from function calls
            pattern = " -> ".join(f"{m[1]}" for m in matches)
            
            if pattern not in stack_patterns:
                stack_patterns[pattern] = []
            stack_patterns[pattern].append(error)
        
        # Find significant patterns (more than one occurrence)
        significant_patterns = []
        
        for pattern, pattern_errors in message_patterns.items():
            if len(pattern_errors) > 1:
                significant_patterns.append({
                    "type": "message_pattern",
                    "pattern": pattern,
                    "occurrences": len(pattern_errors),
                    "errors": [e["error_id"] for e in pattern_errors],
                    "components": list(set(e.get("component", "unknown") for e in pattern_errors))
                })
        
        for pattern, pattern_errors in stack_patterns.items():
            if len(pattern_errors) > 1:
                significant_patterns.append({
                    "type": "stack_pattern",
                    "pattern": pattern,
                    "occurrences": len(pattern_errors),
                    "errors": [e["error_id"] for e in pattern_errors],
                    "components": list(set(e.get("component", "unknown") for e in pattern_errors))
                })
        
        # Sort patterns by number of occurrences
        significant_patterns.sort(key=lambda p: p["occurrences"], reverse=True)
        
        # Store pattern data if document store is available
        if self.document_store:
            try:
                self.document_store.create_document(
                    collection_name="error_analyses",
                    doc_id=f"PATTERNS-{datetime.datetime.now().strftime('%Y%m%d')}",
                    fields={"patterns": significant_patterns}
                )
                logger.info("Stored error patterns")
            except Exception as e:
                logger.error(f"Failed to store error patterns: {e}")
        
        return significant_patterns
    
    def generate_error_report(self, query: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive error report.
        
        Args:
            query: Query to filter errors (optional)
        
        Returns:
            Error report
        """
        logger.info("Generating error report")
        
        # Identify trends
        trends = self.identify_error_trends()
        
        # Identify patterns
        patterns = self.identify_error_patterns()
        
        # Get errors to include in report
        if query:
            errors = self.error_repository.query_errors(query)
        else:
            # Get recent errors (last 30 days)
            thirty_days_ago = (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat()
            errors = self.error_repository.query_errors({"timestamp": {"$gte": thirty_days_ago}})
        
        # Sort errors by severity and timestamp
        errors.sort(key=lambda e: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(e.get("severity", "low"), 4),
            e.get("timestamp", "")
        ))
        
        # Compile report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": trends["summary"],
            "trends": {
                "by_component": trends["by_component"],
                "by_date": trends["by_date"]
            },
            "patterns": patterns[:10],  # Top 10 patterns
            "errors": errors[:100],  # Top 100 errors
            "recommendations": self._generate_report_recommendations(trends, patterns, errors)
        }
        
        # Store report if document store is available
        if self.document_store:
            try:
                self.document_store.create_document(
                    collection_name="error_reports",
                    doc_id=f"REPORT-{datetime.datetime.now().strftime('%Y%m%d')}",
                    fields=report
                )
                logger.info("Stored error report")
            except Exception as e:
                logger.error(f"Failed to store error report: {e}")
        
        return report
    
    def _perform_root_cause_analysis(self, error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform root cause analysis on an error.
        
        Args:
            error: Error data dictionary
        
        Returns:
            Root cause analysis results
        """
        # In a real implementation, this would use more sophisticated techniques
        # For now, we'll implement a basic version
        
        root_cause = {
            "cause_type": "unknown",
            "confidence": 0.0,
            "details": {},
            "potential_causes": []
        }
        
        # Extract information from error
        component = error.get("component", "unknown")
        message = error.get("message", "")
        stack_trace = error.get("stack_trace", "")
        
        # Check for known patterns in message
        if "null" in message.lower() or "none" in message.lower():
            root_cause["cause_type"] = "null_reference"
            root_cause["confidence"] = 0.8
            root_cause["details"]["description"] = "Null reference or missing value"
        elif "timeout" in message.lower():
            root_cause["cause_type"] = "timeout"
            root_cause["confidence"] = 0.8
            root_cause["details"]["description"] = "Operation timed out"
        elif "permission" in message.lower() or "access" in message.lower():
            root_cause["cause_type"] = "permission"
            root_cause["confidence"] = 0.8
            root_cause["details"]["description"] = "Permission or access issue"
        elif "syntax" in message.lower():
            root_cause["cause_type"] = "syntax_error"
            root_cause["confidence"] = 0.9
            root_cause["details"]["description"] = "Syntax error in code or configuration"
        elif "connect" in message.lower():
            root_cause["cause_type"] = "connection"
            root_cause["confidence"] = 0.8
            root_cause["details"]["description"] = "Connection issue"
        
        # Check stack trace for additional information
        if stack_trace:
            # Extract function calls from stack trace
            matches = re.findall(r'File "([^"]+)", line \d+, in (\w+)', stack_trace)
            if matches:
                # Last function call is likely the source of the error
                root_cause["details"]["source_file"] = matches[-1][0]
                root_cause["details"]["source_function"] = matches[-1][1]
                
                # Check for common error locations
                for file_path, func_name in matches:
                    if "test" in file_path:
                        root_cause["potential_causes"].append({
                            "cause_type": "test_issue",
                            "confidence": 0.6,
                            "details": {
                                "description": "Error occurred in test code",
                                "source_file": file_path,
                                "source_function": func_name
                            }
                        })
                    elif "database" in file_path or "db" in file_path:
                        root_cause["potential_causes"].append({
                            "cause_type": "database_issue",
                            "confidence": 0.7,
                            "details": {
                                "description": "Error occurred in database-related code",
                                "source_file": file_path,
                                "source_function": func_name
                            }
                        })
        
        # Check for related errors that might provide insight
        related_errors = self.error_repository.get_related_errors(error["error_id"])
        if related_errors:
            # Look for patterns in related errors
            related_components = [e.get("component", "unknown") for e in related_errors]
            if len(set(related_components)) > 1:
                root_cause["potential_causes"].append({
                    "cause_type": "cross_component",
                    "confidence": 0.7,
                    "details": {
                        "description": "Error may be caused by interaction between components",
                        "components": list(set(related_components))
                    }
                })
        
        # If we couldn't determine a specific cause type, try to infer from component
        if root_cause["cause_type"] == "unknown":
            if "database" in component.lower() or "db" in component.lower():
                root_cause["cause_type"] = "database_issue"
                root_cause["confidence"] = 0.6
                root_cause["details"]["description"] = "Issue related to database operations"
            elif "network" in component.lower() or "connect" in component.lower():
                root_cause["cause_type"] = "network_issue"
                root_cause["confidence"] = 0.6
                root_cause["details"]["description"] = "Issue related to network operations"
            elif "config" in component.lower():
                root_cause["cause_type"] = "configuration_issue"
                root_cause["confidence"] = 0.6
                root_cause["details"]["description"] = "Issue related to configuration"
        
        return root_cause
    
    def _assess_impact(self, error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the impact of an error.
        
        Args:
            error: Error data dictionary
        
        Returns:
            Impact assessment results
        """
        impact = {
            "severity": error.get("severity", "low"),
            "scope": "unknown",
            "affected_components": [],
            "affected_functionality": [],
            "user_impact": "unknown",
            "system_impact": "unknown"
        }
        
        # Determine affected components
        component = error.get("component", "unknown")
        impact["affected_components"].append(component)
        
        # Check for related errors to identify broader impact
        related_errors = self.error_repository.get_related_errors(error["error_id"])
        for related_error in related_errors:
            related_component = related_error.get("component", "unknown")
            if related_component not in impact["affected_components"]:
                impact["affected_components"].append(related_component)
        
        # Determine scope based on affected components
        if len(impact["affected_components"]) > 1:
            impact["scope"] = "multi_component"
        else:
            impact["scope"] = "single_component"
        
        # Infer affected functionality
        if "database" in component.lower() or "db" in component.lower():
            impact["affected_functionality"].append("data_storage")
        elif "api" in component.lower():
            impact["affected_functionality"].append("api_endpoints")
        elif "ui" in component.lower() or "interface" in component.lower():
            impact["affected_functionality"].append("user_interface")
        elif "auth" in component.lower():
            impact["affected_functionality"].append("authentication")
        
        # Determine user impact based on severity and affected functionality
        severity = error.get

# Added methods for ErrorAnalyzer class
def _load_analysis_rules(self):
    # Simple implementation for testing
    rules = {
        "root_cause": {
            "patterns": [
                {"pattern": "null|none", "cause_type": "null_reference", "confidence": 0.8},
                {"pattern": "timeout", "cause_type": "timeout", "confidence": 0.8}
            ]
        },
        "impact": {
            "severity_weights": {
                "critical": 1.0,
                "high": 0.8,
                "medium": 0.5,
                "low": 0.2
            }
        }
    }
    logger.info("Loaded analysis rules")
    return rules

def _find_related_errors(self, error):
    # Simple implementation for testing
    related_errors = []
    if "component" in error:
        component_errors = self.error_repository.query_errors({"component": error["component"]})
        for comp_error in component_errors:
            if comp_error["error_id"] != error["error_id"]:
                related_errors.append(comp_error)
    return related_errors[:10]

def _generate_recommendations(self, error, root_cause, impact):
    # Simple implementation for testing
    recommendations = ["Add comprehensive error handling"]
    return recommendations

def _generate_report_recommendations(self, trends, patterns, errors):
    # Simple implementation for testing
    recommendations = ["Implement comprehensive error handling across all components"]
    return recommendations
