"""
FixWurx Auditor - Error Aggregation System

This module implements a sophisticated system for aggregating, correlating,
and prioritizing error reports from multiple sensors.
"""

import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import heapq
from datetime import datetime

from error_report import ErrorReport

logger = logging.getLogger('error_aggregation_system')

class ErrorAggregationSystem:
    """
    System for aggregating and prioritizing error reports from multiple sensors.
    
    This system implements several key strategies:
    1. Temporal correlation - grouping errors that occur close in time
    2. Root cause analysis - identifying common underlying causes
    3. Impact assessment - prioritizing based on system impact
    4. Dynamic thresholding - adjusting sensitivity based on patterns
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ErrorAggregationSystem.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.reports_dir = self.config.get("reports_dir", "auditor_data/reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Time window for temporal correlation (seconds)
        self.time_window = self.config.get("time_window", 300)
        
        # Active error groups
        self.error_groups: Dict[str, Dict[str, Any]] = {}
        
        # Error type priorities (higher is more important)
        self.error_priorities = {
            "CRITICAL": 100,
            "HIGH": 80,
            "MEDIUM": 50,
            "LOW": 20,
            "INFO": 10
        }
        
        # Component priorities (higher is more important)
        self.component_priorities = self.config.get("component_priorities", {})
        
        # Default component priority
        self.default_component_priority = self.config.get("default_component_priority", 50)
        
        # Correlation strategies
        self.correlation_strategies = [
            self._correlate_by_component,
            self._correlate_by_error_type,
            self._correlate_by_time,
            self._correlate_by_context
        ]
        
        # Track already processed reports
        self.processed_reports: Set[str] = set()
        
        # Load existing groups
        self._load_existing_groups()
        
        logger.info(f"Initialized ErrorAggregationSystem with storage at {self.reports_dir}")
    
    def process_report(self, report: ErrorReport) -> Optional[str]:
        """
        Process a new error report.
        
        Args:
            report: The error report to process
            
        Returns:
            ID of the error group if the report was aggregated, None if filtered
        """
        # Check if we've already seen this exact report
        report_hash = self._compute_report_hash(report)
        if report_hash in self.processed_reports:
            logger.debug(f"Skipping duplicate report: {report_hash}")
            return None
        
        self.processed_reports.add(report_hash)
        
        # Try to find a matching group
        group_id = self._find_matching_group(report)
        
        if group_id:
            # Add to existing group
            self._add_to_group(group_id, report)
            logger.info(f"Added report to existing group {group_id}")
            return group_id
        else:
            # Create new group
            new_group_id = self._create_new_group(report)
            logger.info(f"Created new error group {new_group_id}")
            return new_group_id
    
    def get_prioritized_groups(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top priority error groups.
        
        Args:
            limit: Maximum number of groups to return
            
        Returns:
            List of error groups sorted by priority
        """
        # Create a priority queue
        priority_queue = []
        
        for group_id, group in self.error_groups.items():
            # Only include active groups
            if group["status"] == "active":
                # Calculate priority score
                priority = self._calculate_group_priority(group)
                
                # Add to queue (negative priority for max heap)
                heapq.heappush(priority_queue, (-priority, group_id))
        
        # Extract top groups
        result = []
        for _ in range(min(limit, len(priority_queue))):
            if not priority_queue:
                break
                
            _, group_id = heapq.heappop(priority_queue)
            result.append(self.error_groups[group_id])
        
        return result
    
    def _compute_report_hash(self, report: ErrorReport) -> str:
        """
        Compute a hash for the report to detect duplicates.
        
        Args:
            report: The error report
            
        Returns:
            Hash string for the report
        """
        # In a real implementation, we would use a more sophisticated
        # hashing strategy, potentially using a combination of error type,
        # component, timestamp, and specific details
        details_str = json.dumps(report.details, sort_keys=True) if report.details else ""
        components = [
            report.component_name,
            report.error_type,
            details_str
        ]
        return ":".join(components)
    
    def _find_matching_group(self, report: ErrorReport) -> Optional[str]:
        """
        Find a matching error group for the report.
        
        Args:
            report: The error report
            
        Returns:
            ID of the matching group, or None if no match
        """
        # Apply each correlation strategy
        for strategy in self.correlation_strategies:
            group_id = strategy(report)
            if group_id:
                return group_id
        
        return None
    
    def _correlate_by_component(self, report: ErrorReport) -> Optional[str]:
        """Correlate by component and error type."""
        for group_id, group in self.error_groups.items():
            # Skip inactive groups
            if group["status"] != "active":
                continue
                
            if (group["component"] == report.component_name and
                group["error_type"] == report.error_type and
                time.time() - group["last_updated"] < self.time_window):
                return group_id
        return None
    
    def _correlate_by_error_type(self, report: ErrorReport) -> Optional[str]:
        """Correlate by error type across components."""
        for group_id, group in self.error_groups.items():
            # Skip inactive groups
            if group["status"] != "active":
                continue
                
            if (group["error_type"] == report.error_type and
                time.time() - group["last_updated"] < self.time_window / 2):
                return group_id
        return None
    
    def _correlate_by_time(self, report: ErrorReport) -> Optional[str]:
        """Correlate by temporal proximity."""
        report_time = time.time()
        
        for group_id, group in self.error_groups.items():
            # Skip inactive groups
            if group["status"] != "active":
                continue
                
            if report_time - group["last_updated"] < 30:  # 30 seconds
                # Check if there's some context overlap
                if self._has_context_overlap(report, group):
                    return group_id
        
        return None
    
    def _correlate_by_context(self, report: ErrorReport) -> Optional[str]:
        """Correlate by context similarity."""
        for group_id, group in self.error_groups.items():
            # Skip inactive groups
            if group["status"] != "active":
                continue
                
            if self._context_similarity(report.context, group["context"]) > 0.7:
                return group_id
        return None
    
    def _has_context_overlap(self, report: ErrorReport, group: Dict[str, Any]) -> bool:
        """Check if there's meaningful overlap in context."""
        # This is a simplified implementation
        # A real implementation would use more sophisticated similarity metrics
        if not report.context or not group["context"]:
            return False
            
        for key, value in report.context.items():
            if key in group["context"] and group["context"][key] == value:
                return True
                
        return False
    
    def _context_similarity(self, context1: Dict[str, Any], 
                           context2: Dict[str, Any]) -> float:
        """Calculate context similarity (0-1)."""
        # This is a simplified implementation
        # A real implementation would use more sophisticated similarity metrics
        if not context1 or not context2:
            return 0.0
            
        common_keys = set(context1.keys()) & set(context2.keys())
        all_keys = set(context1.keys()) | set(context2.keys())
        
        if not all_keys:
            return 0.0
            
        matching = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                matching += 1
                
        return matching / len(all_keys)
    
    def _create_new_group(self, report: ErrorReport) -> str:
        """
        Create a new error group.
        
        Args:
            report: The first error report in the group
            
        Returns:
            ID of the new group
        """
        # Generate group ID
        timestamp = int(time.time())
        group_id = f"group_{timestamp}_{report.component_name}_{report.error_type}"
        
        # Create group
        self.error_groups[group_id] = {
            "id": group_id,
            "component": report.component_name,
            "error_type": report.error_type,
            "severity": report.severity,
            "first_seen": time.time(),
            "last_updated": time.time(),
            "count": 1,
            "reports": [self._report_to_dict(report)],
            "context": report.context.copy() if report.context else {},
            "status": "active"
        }
        
        # Save to file
        self._save_group(group_id)
        
        return group_id
    
    def _add_to_group(self, group_id: str, report: ErrorReport) -> None:
        """
        Add a report to an existing group.
        
        Args:
            group_id: ID of the group
            report: The error report to add
        """
        group = self.error_groups[group_id]
        
        # Update group
        group["last_updated"] = time.time()
        group["count"] += 1
        
        # Update severity to the highest
        if self.error_priorities.get(report.severity, 0) > self.error_priorities.get(group["severity"], 0):
            group["severity"] = report.severity
        
        # Add report
        group["reports"].append(self._report_to_dict(report))
        
        # Keep only the most recent 100 reports
        if len(group["reports"]) > 100:
            group["reports"] = group["reports"][-100:]
        
        # Update context
        if report.context:
            for key, value in report.context.items():
                if key not in group["context"]:
                    group["context"][key] = value
        
        # Save to file
        self._save_group(group_id)
    
    def _report_to_dict(self, report: ErrorReport) -> Dict[str, Any]:
        """Convert an ErrorReport to a dictionary."""
        return {
            "timestamp": time.time(),
            "component": report.component_name,
            "sensor_id": report.sensor_id,
            "error_type": report.error_type,
            "severity": report.severity,
            "details": report.details,
            "context": report.context
        }
    
    def _save_group(self, group_id: str) -> None:
        """
        Save an error group to file.
        
        Args:
            group_id: ID of the group to save
        """
        group = self.error_groups[group_id]
        
        # Create filename
        filename = f"{group_id}.json"
        filepath = os.path.join(self.reports_dir, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(group, f, indent=2)
    
    def _load_existing_groups(self) -> None:
        """Load existing error groups from disk."""
        try:
            if not os.path.exists(self.reports_dir):
                return
                
            loaded_count = 0
            for filename in os.listdir(self.reports_dir):
                if filename.endswith(".json") and filename.startswith("group_"):
                    try:
                        filepath = os.path.join(self.reports_dir, filename)
                        with open(filepath, 'r') as f:
                            group = json.load(f)
                            
                        group_id = group.get("id")
                        if group_id:
                            self.error_groups[group_id] = group
                            loaded_count += 1
                            
                            # Add report hashes to processed set
                            for report in group.get("reports", []):
                                if "component" in report and "error_type" in report:
                                    details_str = json.dumps(report.get("details", {}), sort_keys=True)
                                    components = [
                                        report["component"],
                                        report["error_type"],
                                        details_str
                                    ]
                                    hash_str = ":".join(components)
                                    self.processed_reports.add(hash_str)
                    except Exception as e:
                        logger.error(f"Error loading group from {filename}: {str(e)}")
                        
            logger.info(f"Loaded {loaded_count} existing error groups")
        except Exception as e:
            logger.error(f"Error loading existing groups: {str(e)}")
    
    def _calculate_group_priority(self, group: Dict[str, Any]) -> float:
        """
        Calculate priority score for an error group.
        
        Args:
            group: The error group
            
        Returns:
            Priority score (higher is more important)
        """
        # Base priority from severity
        priority = self.error_priorities.get(group["severity"], 0)
        
        # Adjust for component importance
        component_factor = self.component_priorities.get(
            group["component"], self.default_component_priority
        ) / 50.0
        priority *= component_factor
        
        # Adjust for frequency/volume
        # More frequent errors get higher priority
        frequency_factor = min(2.0, 0.5 + (group["count"] / 10.0))
        priority *= frequency_factor
        
        # Adjust for recency
        # More recent errors get higher priority
        age_seconds = time.time() - group["last_updated"]
        recency_factor = max(0.5, 2.0 - (age_seconds / self.time_window))
        priority *= recency_factor
        
        # Adjust for persistence
        # Long-running error groups get higher priority
        duration = time.time() - group["first_seen"]
        if duration > 3600:  # More than an hour
            priority *= 1.5
        
        return priority
    
    def acknowledge_group(self, group_id: str, 
                         acknowledgement: Dict[str, Any]) -> bool:
        """
        Acknowledge an error group.
        
        Args:
            group_id: ID of the group to acknowledge
            acknowledgement: Acknowledgement details
            
        Returns:
            True if successful, False otherwise
        """
        if group_id not in self.error_groups:
            return False
        
        group = self.error_groups[group_id]
        
        # Update status
        group["status"] = "acknowledged"
        group["acknowledged_at"] = time.time()
        group["acknowledgement"] = acknowledgement
        
        # Save to file
        self._save_group(group_id)
        
        logger.info(f"Acknowledged error group {group_id}")
        return True
    
    def resolve_group(self, group_id: str, 
                     resolution: Dict[str, Any]) -> bool:
        """
        Mark an error group as resolved.
        
        Args:
            group_id: ID of the group to resolve
            resolution: Resolution details
            
        Returns:
            True if successful, False otherwise
        """
        if group_id not in self.error_groups:
            return False
        
        group = self.error_groups[group_id]
        
        # Update status
        group["status"] = "resolved"
        group["resolved_at"] = time.time()
        group["resolution"] = resolution
        
        # Save to file
        self._save_group(group_id)
        
        logger.info(f"Resolved error group {group_id}")
        return True
    
    def reopen_group(self, group_id: str) -> bool:
        """
        Reopen a previously acknowledged or resolved group.
        
        Args:
            group_id: ID of the group to reopen
            
        Returns:
            True if successful, False otherwise
        """
        if group_id not in self.error_groups:
            return False
        
        group = self.error_groups[group_id]
        
        # Update status
        group["status"] = "active"
        group["reopened_at"] = time.time()
        
        # Save to file
        self._save_group(group_id)
        
        logger.info(f"Reopened error group {group_id}")
        return True
    
    def get_group(self, group_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific error group.
        
        Args:
            group_id: ID of the group to get
            
        Returns:
            Error group details or None if not found
        """
        return self.error_groups.get(group_id)
    
    def get_groups_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get all error groups with a specific status.
        
        Args:
            status: Status to filter by (active, acknowledged, resolved)
            
        Returns:
            List of error groups
        """
        return [group for group in self.error_groups.values() 
                if group["status"] == status]
    
    def get_groups_by_component(self, component: str) -> List[Dict[str, Any]]:
        """
        Get all error groups for a specific component.
        
        Args:
            component: Component name to filter by
            
        Returns:
            List of error groups
        """
        return [group for group in self.error_groups.values() 
                if group["component"] == component]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about error groups.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_groups": len(self.error_groups),
            "active_groups": 0,
            "acknowledged_groups": 0,
            "resolved_groups": 0,
            "groups_by_severity": defaultdict(int),
            "groups_by_component": defaultdict(int),
            "total_errors": 0
        }
        
        for group in self.error_groups.values():
            stats[f"{group['status']}_groups"] += 1
            stats["groups_by_severity"][group["severity"]] += 1
            stats["groups_by_component"][group["component"]] += 1
            stats["total_errors"] += group["count"]
        
        return stats

# Factory function
def create_error_aggregation_system(config: Optional[Dict[str, Any]] = None) -> ErrorAggregationSystem:
    """
    Create and initialize an ErrorAggregationSystem.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized ErrorAggregationSystem
    """
    return ErrorAggregationSystem(config)
