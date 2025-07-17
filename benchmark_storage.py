"""
FixWurx Auditor Benchmark Storage System

This module provides persistent storage for benchmark metrics, organized by
session ID and project. It integrates with the auditor's sensor framework
to store and retrieve benchmark data for analysis and reporting.
"""

import os
import json
import datetime
import logging
import shutil
from typing import Dict, List, Set, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [BenchmarkStorage] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('benchmark_storage')


class BenchmarkStorage:
    """
    Stores benchmark data organized by session ID and project.
    Provides persistence and retrieval capabilities for benchmark metrics.
    """
    
    def __init__(self, base_path: str = "auditor_data/benchmarks"):
        """
        Initialize the benchmark storage system.
        
        Args:
            base_path: Base directory for storing benchmark data
        """
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        self.session_metadata = {}
        
        # Load existing sessions
        self._load_existing_sessions()
        
        logger.info(f"Initialized BenchmarkStorage at {self.base_path}")
    
    def _load_existing_sessions(self) -> None:
        """Load metadata for existing benchmark sessions."""
        index_path = os.path.join(self.base_path, "sessions_index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    self.session_metadata = json.load(f)
                logger.info(f"Loaded {len(self.session_metadata)} existing benchmark sessions")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading sessions index: {e}")
                self.session_metadata = {}
    
    def _save_sessions_index(self) -> None:
        """Save the sessions index to disk."""
        index_path = os.path.join(self.base_path, "sessions_index.json")
        try:
            with open(index_path, 'w') as f:
                json.dump(self.session_metadata, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving sessions index: {e}")
    
    def create_session(self, 
                      project_name: str, 
                      session_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new benchmark session.
        
        Args:
            project_name: Name of the project
            session_id: Optional session ID (generated if not provided)
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        # Generate session ID if not provided
        if not session_id:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            session_id = f"benchmark-{timestamp}-{project_name.replace(' ', '-')}"
        
        # Create project directory if it doesn't exist
        project_dir = os.path.join(self.base_path, project_name)
        os.makedirs(project_dir, exist_ok=True)
        
        # Create session directory
        session_dir = os.path.join(project_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Create metrics directory
        metrics_dir = os.path.join(session_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Create reports directory
        reports_dir = os.path.join(session_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Save session metadata
        metadata = metadata or {}
        metadata.update({
            "session_id": session_id,
            "project_name": project_name,
            "created_at": datetime.datetime.now().isoformat(),
            "metrics_stored": [],
            "reports_stored": 0
        })
        
        self.session_metadata[session_id] = metadata
        
        # Save session metadata to session directory
        session_metadata_path = os.path.join(session_dir, "metadata.json")
        with open(session_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update sessions index
        self._save_sessions_index()
        
        logger.info(f"Created benchmark session: {session_id} for project: {project_name}")
        
        return session_id
    
    def store_metrics(self, 
                     session_id: str, 
                     metrics: Dict[str, Any],
                     timestamp: Optional[str] = None) -> bool:
        """
        Store metrics for a session.
        
        Args:
            session_id: Session ID
            metrics: Metrics to store
            timestamp: Optional timestamp (generated if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.session_metadata:
            logger.error(f"Session {session_id} not found")
            return False
        
        # Get project name from session metadata
        project_name = self.session_metadata[session_id]["project_name"]
        
        # Generate timestamp if not provided
        if not timestamp:
            timestamp = datetime.datetime.now().isoformat()
        
        # Generate filename
        filename = f"metrics_{timestamp.replace(':', '-').replace('.', '-')}.json"
        
        # Get metrics directory
        metrics_dir = os.path.join(self.base_path, project_name, session_id, "metrics")
        metrics_path = os.path.join(metrics_dir, filename)
        
        # Store metrics
        try:
            with open(metrics_path, 'w') as f:
                data = {
                    "timestamp": timestamp,
                    "session_id": session_id,
                    "metrics": metrics
                }
                json.dump(data, f, indent=2)
            
            # Update session metadata
            if "metrics_stored" not in self.session_metadata[session_id]:
                self.session_metadata[session_id]["metrics_stored"] = []
            
            self.session_metadata[session_id]["metrics_stored"].append(filename)
            self.session_metadata[session_id]["last_updated"] = timestamp
            
            # Update sessions index
            self._save_sessions_index()
            
            logger.info(f"Stored metrics for session {session_id} at {timestamp}")
            return True
            
        except IOError as e:
            logger.error(f"Error storing metrics for session {session_id}: {e}")
            return False
    
    def store_error_report(self, 
                          session_id: str, 
                          report: Dict[str, Any],
                          timestamp: Optional[str] = None) -> bool:
        """
        Store an error report for a session.
        
        Args:
            session_id: Session ID
            report: Error report to store
            timestamp: Optional timestamp (generated if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.session_metadata:
            logger.error(f"Session {session_id} not found")
            return False
        
        # Get project name from session metadata
        project_name = self.session_metadata[session_id]["project_name"]
        
        # Generate timestamp if not provided
        if not timestamp:
            timestamp = datetime.datetime.now().isoformat()
        
        # Generate filename
        report_count = self.session_metadata[session_id].get("reports_stored", 0) + 1
        filename = f"report_{report_count:04d}_{timestamp.replace(':', '-').replace('.', '-')}.json"
        
        # Get reports directory
        reports_dir = os.path.join(self.base_path, project_name, session_id, "reports")
        report_path = os.path.join(reports_dir, filename)
        
        # Store report
        try:
            with open(report_path, 'w') as f:
                data = {
                    "timestamp": timestamp,
                    "session_id": session_id,
                    "report": report
                }
                json.dump(data, f, indent=2)
            
            # Update session metadata
            self.session_metadata[session_id]["reports_stored"] = report_count
            self.session_metadata[session_id]["last_updated"] = timestamp
            
            # Update sessions index
            self._save_sessions_index()
            
            logger.info(f"Stored error report {report_count} for session {session_id}")
            return True
            
        except IOError as e:
            logger.error(f"Error storing report for session {session_id}: {e}")
            return False
    
    def get_session_metrics(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all metrics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of metrics dictionaries
        """
        if session_id not in self.session_metadata:
            logger.error(f"Session {session_id} not found")
            return []
        
        # Get project name from session metadata
        project_name = self.session_metadata[session_id]["project_name"]
        
        # Get metrics directory
        metrics_dir = os.path.join(self.base_path, project_name, session_id, "metrics")
        
        metrics_files = self.session_metadata[session_id].get("metrics_stored", [])
        metrics_list = []
        
        for filename in metrics_files:
            metrics_path = os.path.join(metrics_dir, filename)
            try:
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics_data = json.load(f)
                        metrics_list.append(metrics_data)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading metrics file {filename}: {e}")
        
        return metrics_list
    
    def get_session_reports(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all error reports for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of report dictionaries
        """
        if session_id not in self.session_metadata:
            logger.error(f"Session {session_id} not found")
            return []
        
        # Get project name from session metadata
        project_name = self.session_metadata[session_id]["project_name"]
        
        # Get reports directory
        reports_dir = os.path.join(self.base_path, project_name, session_id, "reports")
        
        reports_list = []
        
        try:
            if os.path.exists(reports_dir):
                for filename in os.listdir(reports_dir):
                    if filename.endswith('.json'):
                        report_path = os.path.join(reports_dir, filename)
                        try:
                            with open(report_path, 'r') as f:
                                report_data = json.load(f)
                                reports_list.append(report_data)
                        except (json.JSONDecodeError, IOError) as e:
                            logger.error(f"Error loading report file {filename}: {e}")
        except IOError as e:
            logger.error(f"Error accessing reports directory for session {session_id}: {e}")
        
        return reports_list
    
    def get_project_sessions(self, project_name: str) -> List[str]:
        """
        Get all session IDs for a project.
        
        Args:
            project_name: Project name
            
        Returns:
            List of session IDs
        """
        sessions = []
        for session_id, metadata in self.session_metadata.items():
            if metadata.get("project_name") == project_name:
                sessions.append(session_id)
        
        return sessions
    
    def get_all_projects(self) -> List[str]:
        """
        Get all project names.
        
        Returns:
            List of project names
        """
        projects = set()
        for metadata in self.session_metadata.values():
            projects.add(metadata.get("project_name", ""))
        
        return list(sorted(filter(None, projects)))
    
    def export_session_data(self, 
                           session_id: str, 
                           export_path: str,
                           include_metrics: bool = True,
                           include_reports: bool = True) -> bool:
        """
        Export session data to a directory.
        
        Args:
            session_id: Session ID
            export_path: Export directory path
            include_metrics: Whether to include metrics
            include_reports: Whether to include reports
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.session_metadata:
            logger.error(f"Session {session_id} not found")
            return False
        
        # Get project name from session metadata
        project_name = self.session_metadata[session_id]["project_name"]
        
        # Create export directory
        os.makedirs(export_path, exist_ok=True)
        
        # Export session metadata
        session_metadata_path = os.path.join(export_path, "session_metadata.json")
        try:
            with open(session_metadata_path, 'w') as f:
                json.dump(self.session_metadata[session_id], f, indent=2)
        except IOError as e:
            logger.error(f"Error exporting session metadata: {e}")
            return False
        
        # Export metrics if requested
        if include_metrics:
            metrics_export_path = os.path.join(export_path, "metrics")
            os.makedirs(metrics_export_path, exist_ok=True)
            
            source_metrics_dir = os.path.join(self.base_path, project_name, session_id, "metrics")
            if os.path.exists(source_metrics_dir):
                for filename in os.listdir(source_metrics_dir):
                    if filename.endswith('.json'):
                        source_path = os.path.join(source_metrics_dir, filename)
                        dest_path = os.path.join(metrics_export_path, filename)
                        try:
                            shutil.copy2(source_path, dest_path)
                        except IOError as e:
                            logger.error(f"Error copying metrics file {filename}: {e}")
        
        # Export reports if requested
        if include_reports:
            reports_export_path = os.path.join(export_path, "reports")
            os.makedirs(reports_export_path, exist_ok=True)
            
            source_reports_dir = os.path.join(self.base_path, project_name, session_id, "reports")
            if os.path.exists(source_reports_dir):
                for filename in os.listdir(source_reports_dir):
                    if filename.endswith('.json'):
                        source_path = os.path.join(source_reports_dir, filename)
                        dest_path = os.path.join(reports_export_path, filename)
                        try:
                            shutil.copy2(source_path, dest_path)
                        except IOError as e:
                            logger.error(f"Error copying report file {filename}: {e}")
        
        logger.info(f"Exported session {session_id} data to {export_path}")
        return True
    
    def generate_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Generate a summary of a benchmark session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Summary dictionary
        """
        if session_id not in self.session_metadata:
            logger.error(f"Session {session_id} not found")
            return {}
        
        # Get session metrics
        metrics_list = self.get_session_metrics(session_id)
        
        # Get session reports
        reports_list = self.get_session_reports(session_id)
        
        # Calculate summary statistics
        summary = {
            "session_id": session_id,
            "project_name": self.session_metadata[session_id]["project_name"],
            "created_at": self.session_metadata[session_id]["created_at"],
            "last_updated": self.session_metadata[session_id].get("last_updated"),
            "metrics_count": len(metrics_list),
            "reports_count": len(reports_list),
            "first_metrics": metrics_list[0]["metrics"] if metrics_list else None,
            "last_metrics": metrics_list[-1]["metrics"] if metrics_list else None,
            "key_metrics": {}
        }
        
        # Extract key metrics from first and last measurement
        if metrics_list and len(metrics_list) >= 2:
            first_metrics = metrics_list[0]["metrics"]
            last_metrics = metrics_list[-1]["metrics"]
            
            # Key metrics to track
            key_metrics = [
                "bug_detection_recall",
                "bug_fix_yield",
                "mttd",
                "mttr",
                "energy_reduction_pct",
                "proof_coverage_delta",
                "test_pass_ratio",
                "regression_introduction_rate",
                "hallucination_rate",
                "lyapunov_descent_consistency",
                "aggregate_confidence_score"
            ]
            
            for metric in key_metrics:
                if metric in first_metrics and metric in last_metrics:
                    summary["key_metrics"][metric] = {
                        "initial": first_metrics.get(metric),
                        "final": last_metrics.get(metric),
                        "delta": last_metrics.get(metric) - first_metrics.get(metric)
                    }
        
        return summary


# Integration with PerformanceBenchmarkSensor
def integrate_with_benchmark_sensor(storage: BenchmarkStorage, 
                                  sensor,
                                  project_name: str,
                                  session_id: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Integrate a BenchmarkStorage with a PerformanceBenchmarkSensor.
    
    Args:
        storage: BenchmarkStorage instance
        sensor: PerformanceBenchmarkSensor instance
        project_name: Project name
        session_id: Optional session ID (generated if not provided)
        metadata: Optional session metadata
        
    Returns:
        Session ID
    """
    # Create session if not provided
    if not session_id:
        # Use the sensor's session ID if available
        if hasattr(sensor, 'session_metrics') and 'session_id' in sensor.session_metrics:
            session_id = sensor.session_metrics['session_id']
        else:
            session_id = None
    
    # Create metadata if not provided
    if not metadata:
        metadata = {
            "component_name": sensor.component_name,
            "sensor_id": sensor.sensor_id
        }
    
    # Create session
    session_id = storage.create_session(project_name, session_id, metadata)
    
    # Store initial metrics
    if hasattr(sensor, 'session_metrics'):
        storage.store_metrics(session_id, sensor.session_metrics)
    
    # Set up callback for metrics updates
    def metrics_update_callback(metrics):
        storage.store_metrics(session_id, metrics)
    
    # Set up callback for error reports
    def error_report_callback(report):
        storage.store_error_report(session_id, report.to_dict() if hasattr(report, 'to_dict') else report)
    
    # Attach callbacks to sensor if it supports them
    if hasattr(sensor, 'add_metrics_callback'):
        sensor.add_metrics_callback(metrics_update_callback)
    
    if hasattr(sensor, 'add_report_callback'):
        sensor.add_report_callback(error_report_callback)
    
    return session_id
