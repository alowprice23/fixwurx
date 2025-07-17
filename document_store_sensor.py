"""
FixWurx Auditor Document Store Sensor

This module implements a sensor for monitoring the document store component,
ensuring data integrity, indexing performance, and query efficiency.
"""

import logging
import time
import datetime
import math
import random
import os
import json
import hashlib
from typing import Dict, List, Set, Any, Optional, Union, Tuple, Callable

from sensor_base import ErrorSensor
from error_report import ErrorReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [DocumentStoreSensor] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('document_store_sensor')


class DocumentStoreSensor(ErrorSensor):
    """
    Monitors document store health, performance, and data integrity.
    
    This sensor tracks document retrieval and storage performance,
    index optimization, document integrity, and search functionality.
    """
    
    def __init__(self, 
                component_name: str = "DocumentStore",
                config: Optional[Dict[str, Any]] = None):
        """Initialize the DocumentStoreSensor."""
        super().__init__(
            sensor_id="document_store_sensor",
            component_name=component_name,
            config=config or {}
        )
        
        # Extract configuration values with defaults
        self.check_intervals = {
            "performance": self.config.get("performance_check_interval", 120),  # 2 minutes
            "integrity": self.config.get("integrity_check_interval", 300),  # 5 minutes
            "index": self.config.get("index_check_interval", 600),  # 10 minutes
        }
        
        self.thresholds = {
            "max_query_time_ms": self.config.get("max_query_time_ms", 150),
            "max_storage_time_ms": self.config.get("max_storage_time_ms", 200),
            "max_index_latency_ms": self.config.get("max_index_latency_ms", 250),
            "max_error_rate": self.config.get("max_error_rate", 0.05),  # 5%
            "min_search_recall": self.config.get("min_search_recall", 0.9),  # 90%
            "max_doc_store_size_gb": self.config.get("max_doc_store_size_gb", 10.0)  # 10 GB
        }
        
        # Document store connection settings
        self.doc_store_interface = self.config.get("doc_store_interface", None)
        self.mock_mode = self.config.get("mock_mode", True)
        
        # Initialize metrics and tracking
        self.last_check_times = {check_type: 0 for check_type in self.check_intervals}
        self.metrics = {
            "query_times": [],  # [(timestamp, query_time_ms), ...]
            "storage_times": [],  # [(timestamp, storage_time_ms), ...]
            "index_metrics": {
                "latency_ms": [],  # [(timestamp, latency_ms), ...]
                "size_mb": [],  # [(timestamp, size_mb), ...]
                "fragmentation": []  # [(timestamp, fragmentation_pct), ...]
            },
            "error_rates": {
                "query_errors": [],  # [(timestamp, error_rate), ...]
                "storage_errors": []  # [(timestamp, error_rate), ...]
            },
            "search_metrics": {
                "recall": [],  # [(timestamp, recall_score), ...]
                "precision": []  # [(timestamp, precision_score), ...]
            },
            "storage_metrics": {
                "total_docs": [],  # [(timestamp, count), ...]
                "total_size_mb": [],  # [(timestamp, size_mb), ...]
                "avg_doc_size_kb": []  # [(timestamp, size_kb), ...]
            },
            "integrity_issues": []  # List of document integrity issues
        }
        
        # Document collections to monitor
        self.monitored_collections = self.config.get("monitored_collections", [
            "system_documentation",
            "error_reports",
            "audit_logs",
            "configurations",
            "knowledge_base"
        ])
        
        # Initialize mock data if in mock mode
        if self.mock_mode:
            self._initialize_mock_data()
        
        logger.info(f"Initialized DocumentStoreSensor for {component_name}")
    
    def monitor(self, data: Any = None) -> List[ErrorReport]:
        """
        Monitor document store metrics.
        
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
        
        # Perform index check if needed
        if self.last_check_time - self.last_check_times["index"] >= self.check_intervals["index"]:
            index_reports = self._check_index_health()
            if index_reports:
                reports.extend(index_reports)
            self.last_check_times["index"] = self.last_check_time
        
        return reports
    
    def _initialize_mock_data(self) -> None:
        """Initialize mock data for testing in mock mode."""
        # Generate mock query times
        current_time = time.time()
        for i in range(100):
            timestamp = current_time - (100 - i) * 60  # Every minute
            query_time = random.uniform(20, 130)  # 20-130ms
            self.metrics["query_times"].append((timestamp, query_time))
        
        # Generate mock storage times
        for i in range(100):
            timestamp = current_time - (100 - i) * 60  # Every minute
            storage_time = random.uniform(30, 180)  # 30-180ms
            self.metrics["storage_times"].append((timestamp, storage_time))
        
        # Generate mock index metrics
        for i in range(50):
            timestamp = current_time - (50 - i) * 1800  # Every 30 minutes
            latency = random.uniform(50, 200)  # 50-200ms
            size = 500 + i * random.uniform(1, 5)  # Growing index size
            fragmentation = min(80, max(0, 5 + i * 0.5 + random.uniform(-2, 5)))  # Increasing fragmentation
            
            self.metrics["index_metrics"]["latency_ms"].append((timestamp, latency))
            self.metrics["index_metrics"]["size_mb"].append((timestamp, size))
            self.metrics["index_metrics"]["fragmentation"].append((timestamp, fragmentation))
        
        # Generate mock error rates
        for i in range(24):
            timestamp = current_time - (24 - i) * 3600  # Every hour
            query_error_rate = random.uniform(0, 0.08)  # 0-8%
            storage_error_rate = random.uniform(0, 0.06)  # 0-6%
            
            self.metrics["error_rates"]["query_errors"].append((timestamp, query_error_rate))
            self.metrics["error_rates"]["storage_errors"].append((timestamp, storage_error_rate))
        
        # Generate mock search metrics
        for i in range(24):
            timestamp = current_time - (24 - i) * 3600  # Every hour
            recall = random.uniform(0.85, 0.99)  # 85-99%
            precision = random.uniform(0.80, 0.98)  # 80-98%
            
            self.metrics["search_metrics"]["recall"].append((timestamp, recall))
            self.metrics["search_metrics"]["precision"].append((timestamp, precision))
        
        # Generate mock storage metrics
        for i in range(30):
            timestamp = current_time - (30 - i) * 86400  # Every day
            total_docs = 10000 + i * random.randint(100, 500)
            total_size = 1000 + i * random.uniform(30, 80)  # MB
            avg_doc_size = (total_size * 1024) / total_docs if total_docs > 0 else 0
            
            self.metrics["storage_metrics"]["total_docs"].append((timestamp, total_docs))
            self.metrics["storage_metrics"]["total_size_mb"].append((timestamp, total_size))
            self.metrics["storage_metrics"]["avg_doc_size_kb"].append((timestamp, avg_doc_size))
        
        # Occasionally generate mock integrity issues
        if random.random() < 0.3:  # 30% chance
            collection = random.choice(self.monitored_collections)
            doc_id = f"doc_{random.randint(1000, 9999)}"
            issue_type = random.choice(["corruption", "missing_field", "schema_violation", "duplicate"])
            
            self.metrics["integrity_issues"].append({
                "timestamp": current_time - random.randint(0, 86400),
                "collection": collection,
                "doc_id": doc_id,
                "issue_type": issue_type,
                "details": f"Sample {issue_type} issue in document {doc_id}"
            })
        
        logger.info("Initialized mock data for document store sensor")
    
    def _update_metrics_from_data(self, data: Dict[str, Any]) -> None:
        """
        Update metrics from provided data.
        
        Args:
            data: Dictionary containing document store metrics
        """
        current_time = time.time()
        
        # Update query times if provided
        if "query_time_ms" in data:
            self.metrics["query_times"].append((current_time, data["query_time_ms"]))
            # Keep only the most recent 1000 measurements
            if len(self.metrics["query_times"]) > 1000:
                self.metrics["query_times"] = self.metrics["query_times"][-1000:]
        
        # Update storage times if provided
        if "storage_time_ms" in data:
            self.metrics["storage_times"].append((current_time, data["storage_time_ms"]))
            # Keep only the most recent 1000 measurements
            if len(self.metrics["storage_times"]) > 1000:
                self.metrics["storage_times"] = self.metrics["storage_times"][-1000:]
        
        # Update index metrics if provided
        if "index_latency_ms" in data:
            self.metrics["index_metrics"]["latency_ms"].append((current_time, data["index_latency_ms"]))
            if len(self.metrics["index_metrics"]["latency_ms"]) > 200:
                self.metrics["index_metrics"]["latency_ms"] = self.metrics["index_metrics"]["latency_ms"][-200:]
        
        if "index_size_mb" in data:
            self.metrics["index_metrics"]["size_mb"].append((current_time, data["index_size_mb"]))
            if len(self.metrics["index_metrics"]["size_mb"]) > 100:
                self.metrics["index_metrics"]["size_mb"] = self.metrics["index_metrics"]["size_mb"][-100:]
        
        if "index_fragmentation" in data:
            self.metrics["index_metrics"]["fragmentation"].append((current_time, data["index_fragmentation"]))
            if len(self.metrics["index_metrics"]["fragmentation"]) > 100:
                self.metrics["index_metrics"]["fragmentation"] = self.metrics["index_metrics"]["fragmentation"][-100:]
        
        # Update error rates if provided
        if "query_error_rate" in data:
            self.metrics["error_rates"]["query_errors"].append((current_time, data["query_error_rate"]))
            if len(self.metrics["error_rates"]["query_errors"]) > 100:
                self.metrics["error_rates"]["query_errors"] = self.metrics["error_rates"]["query_errors"][-100:]
        
        if "storage_error_rate" in data:
            self.metrics["error_rates"]["storage_errors"].append((current_time, data["storage_error_rate"]))
            if len(self.metrics["error_rates"]["storage_errors"]) > 100:
                self.metrics["error_rates"]["storage_errors"] = self.metrics["error_rates"]["storage_errors"][-100:]
        
        # Update search metrics if provided
        if "search_recall" in data:
            self.metrics["search_metrics"]["recall"].append((current_time, data["search_recall"]))
            if len(self.metrics["search_metrics"]["recall"]) > 100:
                self.metrics["search_metrics"]["recall"] = self.metrics["search_metrics"]["recall"][-100:]
        
        if "search_precision" in data:
            self.metrics["search_metrics"]["precision"].append((current_time, data["search_precision"]))
            if len(self.metrics["search_metrics"]["precision"]) > 100:
                self.metrics["search_metrics"]["precision"] = self.metrics["search_metrics"]["precision"][-100:]
        
        # Update storage metrics if provided
        if "total_docs" in data:
            self.metrics["storage_metrics"]["total_docs"].append((current_time, data["total_docs"]))
            if len(self.metrics["storage_metrics"]["total_docs"]) > 100:
                self.metrics["storage_metrics"]["total_docs"] = self.metrics["storage_metrics"]["total_docs"][-100:]
        
        if "total_size_mb" in data:
            self.metrics["storage_metrics"]["total_size_mb"].append((current_time, data["total_size_mb"]))
            if len(self.metrics["storage_metrics"]["total_size_mb"]) > 100:
                self.metrics["storage_metrics"]["total_size_mb"] = self.metrics["storage_metrics"]["total_size_mb"][-100:]
        
        if "avg_doc_size_kb" in data:
            self.metrics["storage_metrics"]["avg_doc_size_kb"].append((current_time, data["avg_doc_size_kb"]))
            if len(self.metrics["storage_metrics"]["avg_doc_size_kb"]) > 100:
                self.metrics["storage_metrics"]["avg_doc_size_kb"] = self.metrics["storage_metrics"]["avg_doc_size_kb"][-100:]
        
        # Update integrity issues if provided
        if "integrity_issues" in data and isinstance(data["integrity_issues"], list):
            for issue in data["integrity_issues"]:
                if isinstance(issue, dict) and "collection" in issue and "doc_id" in issue:
                    self.metrics["integrity_issues"].append(issue)
            
            # Keep only the most recent 100 issues
            if len(self.metrics["integrity_issues"]) > 100:
                self.metrics["integrity_issues"] = self.metrics["integrity_issues"][-100:]
    
    def _check_performance(self) -> List[ErrorReport]:
        """
        Check document store performance metrics.
        
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
                
                # Generate mock query time
                query_time = random.uniform(20, 180)  # 20-180ms
                self.metrics["query_times"].append((current_time, query_time))
                
                # Generate mock storage time
                storage_time = random.uniform(30, 220)  # 30-220ms
                self.metrics["storage_times"].append((current_time, storage_time))
                
                # Generate mock error rates (occasionally simulate errors)
                if random.random() < 0.1:  # 10% chance
                    query_error_rate = random.uniform(0.02, 0.1)  # 2-10%
                    self.metrics["error_rates"]["query_errors"].append((current_time, query_error_rate))
                else:
                    self.metrics["error_rates"]["query_errors"].append((current_time, 0.0))
                    
                if random.random() < 0.05:  # 5% chance
                    storage_error_rate = random.uniform(0.01, 0.08)  # 1-8%
                    self.metrics["error_rates"]["storage_errors"].append((current_time, storage_error_rate))
                else:
                    self.metrics["error_rates"]["storage_errors"].append((current_time, 0.0))
            
            # Check recent query times
            recent_query_times = [t[1] for t in self.metrics["query_times"][-20:]]
            if recent_query_times:
                avg_query_time = sum(recent_query_times) / len(recent_query_times)
                max_query_time = max(recent_query_times)
                
                # Check if average query time is above threshold
                if avg_query_time > self.thresholds["max_query_time_ms"]:
                    reports.append(self.report_error(
                        error_type="HIGH_DOC_QUERY_LATENCY",
                        severity="MEDIUM",
                        details={
                            "message": f"Average document query time is above threshold: {avg_query_time:.1f}ms > {self.thresholds['max_query_time_ms']}ms",
                            "avg_query_time_ms": avg_query_time,
                            "max_query_time_ms": max_query_time,
                            "threshold": self.thresholds["max_query_time_ms"]
                        },
                        context={
                            "recent_query_times": self.metrics["query_times"][-10:],
                            "suggested_action": "Optimize document queries or review indexing strategy"
                        }
                    ))
            
            # Check recent storage times
            recent_storage_times = [t[1] for t in self.metrics["storage_times"][-20:]]
            if recent_storage_times:
                avg_storage_time = sum(recent_storage_times) / len(recent_storage_times)
                max_storage_time = max(recent_storage_times)
                
                # Check if average storage time is above threshold
                if avg_storage_time > self.thresholds["max_storage_time_ms"]:
                    reports.append(self.report_error(
                        error_type="HIGH_DOC_STORAGE_LATENCY",
                        severity="MEDIUM",
                        details={
                            "message": f"Average document storage time is above threshold: {avg_storage_time:.1f}ms > {self.thresholds['max_storage_time_ms']}ms",
                            "avg_storage_time_ms": avg_storage_time,
                            "max_storage_time_ms": max_storage_time,
                            "threshold": self.thresholds["max_storage_time_ms"]
                        },
                        context={
                            "recent_storage_times": self.metrics["storage_times"][-10:],
                            "suggested_action": "Check for document store disk I/O issues or review indexing performance"
                        }
                    ))
            
            # Check error rates
            recent_query_errors = [e[1] for e in self.metrics["error_rates"]["query_errors"][-10:]]
            recent_storage_errors = [e[1] for e in self.metrics["error_rates"]["storage_errors"][-10:]]
            
            if recent_query_errors:
                avg_query_error_rate = sum(recent_query_errors) / len(recent_query_errors)
                
                if avg_query_error_rate > self.thresholds["max_error_rate"]:
                    reports.append(self.report_error(
                        error_type="HIGH_DOC_QUERY_ERROR_RATE",
                        severity="HIGH",
                        details={
                            "message": f"Document query error rate is above threshold: {avg_query_error_rate:.1%} > {self.thresholds['max_error_rate']:.1%}",
                            "error_rate": avg_query_error_rate,
                            "threshold": self.thresholds["max_error_rate"]
                        },
                        context={
                            "recent_error_rates": recent_query_errors,
                            "suggested_action": "Investigate query errors, check document schemas, or review query validation"
                        }
                    ))
            
            if recent_storage_errors:
                avg_storage_error_rate = sum(recent_storage_errors) / len(recent_storage_errors)
                
                if avg_storage_error_rate > self.thresholds["max_error_rate"]:
                    reports.append(self.report_error(
                        error_type="HIGH_DOC_STORAGE_ERROR_RATE",
                        severity="HIGH",
                        details={
                            "message": f"Document storage error rate is above threshold: {avg_storage_error_rate:.1%} > {self.thresholds['max_error_rate']:.1%}",
                            "error_rate": avg_storage_error_rate,
                            "threshold": self.thresholds["max_error_rate"]
                        },
                        context={
                            "recent_error_rates": recent_storage_errors,
                            "suggested_action": "Check storage subsystem, validate documents before storage, or review disk space"
                        }
                    ))
            
            # Check search recall
            recent_recall = [r[1] for r in self.metrics["search_metrics"]["recall"][-10:]]
            if recent_recall:
                avg_recall = sum(recent_recall) / len(recent_recall)
                
                if avg_recall < self.thresholds["min_search_recall"]:
                    reports.append(self.report_error(
                        error_type="LOW_SEARCH_RECALL",
                        severity="MEDIUM",
                        details={
                            "message": f"Document search recall is below threshold: {avg_recall:.1%} < {self.thresholds['min_search_recall']:.1%}",
                            "recall_score": avg_recall,
                            "threshold": self.thresholds["min_search_recall"]
                        },
                        context={
                            "recent_recall_scores": recent_recall,
                            "suggested_action": "Review search algorithms, update document indexing, or improve tokenization"
                        }
                    ))
            
            # Check document store size
            if self.metrics["storage_metrics"]["total_size_mb"]:
                latest_size_mb = self.metrics["storage_metrics"]["total_size_mb"][-1][1]
                size_gb = latest_size_mb / 1024
                
                if size_gb > self.thresholds["max_doc_store_size_gb"]:
                    reports.append(self.report_error(
                        error_type="DOCUMENT_STORE_SIZE_LIMIT",
                        severity="MEDIUM",
                        details={
                            "message": f"Document store size exceeds threshold: {size_gb:.2f}GB > {self.thresholds['max_doc_store_size_gb']}GB",
                            "size_gb": size_gb,
                            "threshold": self.thresholds["max_doc_store_size_gb"]
                        },
                        context={
                            "current_doc_count": self.metrics["storage_metrics"]["total_docs"][-1][1] if self.metrics["storage_metrics"]["total_docs"] else "Unknown",
                            "suggested_action": "Implement document archiving, optimize storage, or increase capacity"
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Error in performance check: {str(e)}")
        
        return reports
    
    def _check_data_integrity(self) -> List[ErrorReport]:
        """
        Check document store data integrity.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # In a real implementation, this would analyze documents for integrity issues
            # In mock mode, we'll simulate some issues
            
            # If in mock mode, occasionally simulate a new integrity issue
            if self.mock_mode and random.random() < 0.1:  # 10% chance
                current_time = time.time()
                collection = random.choice(self.monitored_collections)
                doc_id = f"doc_{random.randint(1000, 9999)}"
                issue_type = random.choice(["corruption", "missing_field", "schema_violation", "duplicate"])
                
                self.metrics["integrity_issues"].append({
                    "timestamp": current_time,
                    "collection": collection,
                    "doc_id": doc_id,
                    "issue_type": issue_type,
                    "details": f"Sample {issue_type} issue in document {doc_id}"
                })
            
            # Check for recent integrity issues (past 24 hours)
            recent_issues = [
                issue for issue in self.metrics["integrity_issues"]
                if time.time() - issue.get("timestamp", 0) < 86400
            ]
            
            # Group issues by type
            issues_by_type = {}
            for issue in recent_issues:
                issue_type = issue.get("issue_type", "unknown")
                if issue_type not in issues_by_type:
                    issues_by_type[issue_type] = []
                issues_by_type[issue_type].append(issue)
            
            # Report if there are multiple issues of the same type
            for issue_type, issues in issues_by_type.items():
                if len(issues) >= 2:
                    reports.append(self.report_error(
                        error_type=f"DOCUMENT_{issue_type.upper()}_ISSUES",
                        severity="HIGH" if issue_type in ["corruption", "duplicate"] else "MEDIUM",
                        details={
                            "message": f"Multiple document {issue_type} issues detected",
                            "issue_count": len(issues),
                            "issue_type": issue_type
                        },
                        context={
                            "affected_collections": list(set(issue.get("collection") for issue in issues)),
                            "examples": issues[:3],
                            "suggested_action": self._get_integrity_action(issue_type)
                        }
                    ))
            
            # Report if there are issues across multiple collections
            affected_collections = set(issue.get("collection") for issue in recent_issues)
            if len(affected_collections) >= 2:
                reports.append(self.report_error(
                    error_type="WIDESPREAD_DOCUMENT_INTEGRITY_ISSUES",
                    severity="HIGH",
                    details={
                        "message": f"Document integrity issues detected across {len(affected_collections)} collections",
                        "issue_count": len(recent_issues),
                        "affected_collections": list(affected_collections)
                    },
                    context={
                        "issue_types": list(issues_by_type.keys()),
                        "suggested_action": "Run comprehensive document store integrity check and repair operations"
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error in data integrity check: {str(e)}")
        
        return reports
    
    def _check_index_health(self) -> List[ErrorReport]:
        """
        Check document store index health.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # In a real implementation, this would analyze index health metrics
            # In mock mode, we'll simulate some issues
            
            # If in mock mode, update index metrics
            if self.mock_mode:
                current_time = time.time()
                
                # Index latency
                latency = random.uniform(50, 300)  # 50-300ms
                self.metrics["index_metrics"]["latency_ms"].append((current_time, latency))
                
                # Index size
                if self.metrics["index_metrics"]["size_mb"]:
                    last_size = self.metrics["index_metrics"]["size_mb"][-1][1]
                    new_size = last_size + random.uniform(0, 5)  # Grow by 0-5 MB
                else:
                    new_size = 500  # Start at 500 MB
                self.metrics["index_metrics"]["size_mb"].append((current_time, new_size))
                
                # Index fragmentation
                if self.metrics["index_metrics"]["fragmentation"]:
                    last_frag = self.metrics["index_metrics"]["fragmentation"][-1][1]
                    # Fragmentation tends to increase but can occasionally decrease (after optimization)
                    if random.random() < 0.1:  # 10% chance of optimization
                        new_frag = max(0, last_frag - random.uniform(10, 20))
                    else:
                        new_frag = min(100, last_frag + random.uniform(-1, 3))
                else:
                    new_frag = random.uniform(5, 15)  # Start at 5-15%
                self.metrics["index_metrics"]["fragmentation"].append((current_time, new_frag))
            
            # Check index latency
            recent_latencies = [l[1] for l in self.metrics["index_metrics"]["latency_ms"][-10:]]
            if recent_latencies:
                avg_latency = sum(recent_latencies) / len(recent_latencies)
                
                if avg_latency > self.thresholds["max_index_latency_ms"]:
                    reports.append(self.report_error(
                        error_type="HIGH_INDEX_LATENCY",
                        severity="MEDIUM",
                        details={
                            "message": f"Document index latency is above threshold: {avg_latency:.1f}ms > {self.thresholds['max_index_latency_ms']}ms",
                            "avg_latency_ms": avg_latency,
                            "threshold": self.thresholds["max_index_latency_ms"]
                        },
                        context={
                            "recent_latencies": recent_latencies,
                            "suggested_action": "Optimize index operations, check for resource contention, or update indexing strategy"
                        }
                    ))
            
            # Check index fragmentation
            recent_fragmentation = [f[1] for f in self.metrics["index_metrics"]["fragmentation"][-5:]]
            if recent_fragmentation:
                avg_fragmentation = sum(recent_fragmentation) / len(recent_fragmentation)
                
                # High fragmentation affects performance
                if avg_fragmentation > 30:  # More than 30% fragmentation
                    reports.append(self.report_error(
                        error_type="HIGH_INDEX_FRAGMENTATION",
                        severity="HIGH" if avg_fragmentation > 50 else "MEDIUM",
                        details={
                            "message": f"Document index fragmentation is high: {avg_fragmentation:.1f}%",
                            "fragmentation": avg_fragmentation,
                            "threshold": 30
                        },
                        context={
                            "recent_fragmentation": recent_fragmentation,
                            "suggested_action": "Run index defragmentation or rebuilding operations"
                        }
                    ))
            
            # Check for index size trends
            if len(self.metrics["index_metrics"]["size_mb"]) >= 10:
                # Calculate growth rate
                first = self.metrics["index_metrics"]["size_mb"][-10]
                last = self.metrics["index_metrics"]["size_mb"][-1]
                
                days_diff = (last[0] - first[0]) / 86400  # Convert seconds to days
                if days_diff > 0:
                    size_diff = last[1] - first[1]  # MB difference
                    daily_growth = size_diff / days_diff
                    
                    # Fast-growing index might indicate inefficient indexing
                    if daily_growth > 50:  # More than 50 MB/day
                        reports.append(self.report_error(
                            error_type="RAPID_INDEX_GROWTH",
                            severity="MEDIUM",
                            details={
                                "message": f"Document index is growing rapidly: {daily_growth:.1f} MB/day",
                                "growth_rate_mb_per_day": daily_growth,
                                "current_size_mb": last[1]
                            },
                            context={
                                "days_measured": days_diff,
                                "suggested_action": "Review indexing strategy, check for over-indexing, or implement index pruning"
                            }
                        ))
            
        except Exception as e:
            logger.error(f"Error in index health check: {str(e)}")
        
        return reports
    
    def _get_integrity_action(self, issue_type: str) -> str:
        """
        Get suggested action for document integrity issue.
        
        Args:
            issue_type: Type of integrity issue
            
        Returns:
            Suggested action
        """
        actions = {
            "corruption": "Run document repair utilities and restore from backup if necessary",
            "missing_field": "Update document schema validation and enforce required fields",
            "schema_violation": "Fix document schema or update validation rules",
            "duplicate": "Remove duplicate documents and implement uniqueness constraints",
            "unknown": "Investigate document integrity issues and implement appropriate fixes"
        }
        
        return actions.get(issue_type, actions["unknown"])
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the sensor and monitored component."""
        # Calculate recent performance metrics
        recent_query_times = [t[1] for t in self.metrics["query_times"][-20:]]
        avg_query_time = sum(recent_query_times) / len(recent_query_times) if recent_query_times else 0
        
        recent_storage_times = [t[1] for t in self.metrics["storage_times"][-20:]]
        avg_storage_time = sum(recent_storage_times) / len(recent_storage_times) if recent_storage_times else 0
        
        # Calculate recent error rates
        recent_query_errors = [e[1] for e in self.metrics["error_rates"]["query_errors"][-10:]]
        avg_query_error_rate = sum(recent_query_errors) / len(recent_query_errors) if recent_query_errors else 0
        
        recent_storage_errors = [e[1] for e in self.metrics["error_rates"]["storage_errors"][-10:]]
        avg_storage_error_rate = sum(recent_storage_errors) / len(recent_storage_errors) if recent_storage_errors else 0
        
        # Get recent index metrics
        recent_index_latency = [l[1] for l in self.metrics["index_metrics"]["latency_ms"][-10:]]
        avg_index_latency = sum(recent_index_latency) / len(recent_index_latency) if recent_index_latency else 0
        
        recent_fragmentation = [f[1] for f in self.metrics["index_metrics"]["fragmentation"][-5:]]
        avg_fragmentation = sum(recent_fragmentation) / len(recent_fragmentation) if recent_fragmentation else 0
        
        # Count recent integrity issues
        recent_integrity_issues = len([
            issue for issue in self.metrics["integrity_issues"]
            if time.time() - issue.get("timestamp", 0) < 86400
        ])
        
        # Calculate a health score (0-100)
        query_score = 100 - min(100, (avg_query_time / self.thresholds["max_query_time_ms"]) * 50)
        storage_score = 100 - min(100, (avg_storage_time / self.thresholds["max_storage_time_ms"]) * 50)
        error_score = 100 - min(100, (((avg_query_error_rate + avg_storage_error_rate) / 2) / self.thresholds["max_error_rate"]) * 100)
        index_score = 100 - min(100, ((avg_index_latency / self.thresholds["max_index_latency_ms"]) * 30 + (avg_fragmentation / 100) * 70))
        integrity_penalty = min(50, recent_integrity_issues * 10)
        
        # Weighted health score
        health_score = (query_score * 0.25 + storage_score * 0.25 + error_score * 0.2 + index_score * 0.3) - integrity_penalty
        health_score = max(0, min(100, health_score))
        
        return {
            "sensor_id": self.sensor_id,
            "component_name": self.component_name,
            "last_check_time": self.last_check_time,
            "health_score": health_score,
            "avg_query_time_ms": avg_query_time,
            "avg_storage_time_ms": avg_storage_time,
            "avg_query_error_rate": avg_query_error_rate,
            "avg_storage_error_rate": avg_storage_error_rate,
            "avg_index_latency_ms": avg_index_latency,
            "avg_index_fragmentation": avg_fragmentation,
            "integrity_issues_24h": recent_integrity_issues,
            "total_docs": self.metrics["storage_metrics"]["total_docs"][-1][1] if self.metrics["storage_metrics"]["total_docs"] else 0,
            "total_size_mb": self.metrics["storage_metrics"]["total_size_mb"][-1][1] if self.metrics["storage_metrics"]["total_size_mb"] else 0
        }


# Factory function to create a sensor instance
def create_document_store_sensor(config: Optional[Dict[str, Any]] = None) -> DocumentStoreSensor:
    """
    Create and initialize a document store sensor.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized DocumentStoreSensor
    """
    return DocumentStoreSensor(config=config)
