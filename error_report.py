"""
FixWurx Auditor Error Report

This module implements the standard error report format for all sensors.
It encapsulates error details, metadata, and context, implementing the
standardized error format as specified in error_format_standardization.md.
"""

import datetime
import uuid
import logging
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [ErrorReport] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('error_report')


class ErrorReport:
    """
    Standard error report format for all sensors.
    Encapsulates error details, metadata, and context.
    Implements the standardized error format as specified in error_format_standardization.md.
    """
    
    def __init__(self, 
                 sensor_id: str,
                 component_name: str,
                 error_type: str,
                 severity: str,
                 details: Dict[str, Any],
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize a new error report.
        
        Args:
            sensor_id: ID of the sensor that generated this report
            component_name: Name of the component being monitored
            error_type: Type of error detected
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
            details: Specific details about the error
            context: Additional context information
        """
        # Core fields (required)
        self.error_id = f"ERR-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"
        self.timestamp = datetime.datetime.now().isoformat()
        self.sensor_id = sensor_id
        self.component_name = component_name
        self.error_type = error_type
        self.severity = severity
        self.details = details
        self.context = context or {}
        self.status = "OPEN"  # OPEN, ACKNOWLEDGED, RESOLVED
        
        # Resolution fields (optional)
        self.resolution = None
        self.resolution_timestamp = None
        
        # Extended fields (optional) as specified in error_format_standardization.md
        self.root_cause = None
        self.impact = None
        self.related_errors = []
        self.recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error report to a dictionary."""
        result = {
            # Core fields
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "sensor_id": self.sensor_id,
            "component_name": self.component_name,
            "error_type": self.error_type,
            "severity": self.severity,
            "details": self.details,
            "context": self.context,
            "status": self.status,
            
            # Resolution fields
            "resolution": self.resolution,
            "resolution_timestamp": self.resolution_timestamp,
            
            # Extended fields
            "root_cause": self.root_cause,
            "impact": self.impact,
            "related_errors": self.related_errors,
            "recommendations": self.recommendations
        }
        
        # Filter out None values for cleaner serialization
        return {k: v for k, v in result.items() if v is not None}
    
    def resolve(self, resolution: str) -> None:
        """
        Mark the error as resolved.
        
        Args:
            resolution: Description of how the error was resolved
        """
        self.status = "RESOLVED"
        self.resolution = resolution
        self.resolution_timestamp = datetime.datetime.now().isoformat()
        logger.info(f"Error {self.error_id} resolved: {resolution}")
    
    def acknowledge(self) -> None:
        """Mark the error as acknowledged."""
        self.status = "ACKNOWLEDGED"
        logger.info(f"Error {self.error_id} acknowledged")
    
    def add_root_cause(self, root_cause: str) -> None:
        """
        Add root cause analysis to the error report.
        
        Args:
            root_cause: Description of the root cause
        """
        self.root_cause = root_cause
        logger.info(f"Root cause added to error {self.error_id}")
    
    def add_impact(self, impact: Dict[str, Any]) -> None:
        """
        Add impact assessment to the error report.
        
        Args:
            impact: Description of the impact
        """
        self.impact = impact
        logger.info(f"Impact assessment added to error {self.error_id}")
    
    def add_related_error(self, error_id: str) -> None:
        """
        Add a related error to this report.
        
        Args:
            error_id: ID of the related error
        """
        if error_id not in self.related_errors:
            self.related_errors.append(error_id)
            logger.info(f"Related error {error_id} added to error {self.error_id}")
    
    def add_recommendation(self, recommendation: Dict[str, Any]) -> None:
        """
        Add a recommendation for resolving this error.
        
        Args:
            recommendation: Recommendation details
        """
        self.recommendations.append(recommendation)
        logger.info(f"Recommendation added to error {self.error_id}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorReport':
        """
        Create an ErrorReport from a dictionary.
        
        Args:
            data: Dictionary representation of an error report
            
        Returns:
            ErrorReport instance
        """
        report = cls(
            sensor_id=data.get("sensor_id", "unknown"),
            component_name=data.get("component_name", "unknown"),
            error_type=data.get("error_type", "unknown"),
            severity=data.get("severity", "LOW"),
            details=data.get("details", {}),
            context=data.get("context", {})
        )
        # Core fields
        report.error_id = data.get("error_id", report.error_id)
        report.timestamp = data.get("timestamp", report.timestamp)
        report.status = data.get("status", report.status)
        
        # Resolution fields
        report.resolution = data.get("resolution", report.resolution)
        report.resolution_timestamp = data.get("resolution_timestamp", report.resolution_timestamp)
        
        # Extended fields
        report.root_cause = data.get("root_cause", report.root_cause)
        report.impact = data.get("impact", report.impact)
        report.related_errors = data.get("related_errors", report.related_errors)
        report.recommendations = data.get("recommendations", report.recommendations)
        
        return report
