"""
FixWurx Auditor Graph Database Sensor

This module implements a sensor for monitoring the graph database component,
detecting anomalies like broken connections, orphaned nodes, circular references,
and performance degradation patterns.
"""

import logging
import time
import math
from typing import Dict, List, Set, Any, Optional, Union, Tuple

from sensor_base import ErrorSensor
from error_report import ErrorReport
import graph_database  # Import the graph database module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [GraphDBSensor] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('graph_database_sensor')


class GraphDatabaseSensor(ErrorSensor):
    """
    Monitors the graph database for structural issues, performance problems,
    and data integrity violations.
    """
    
    def __init__(self, 
                component_name: str = "GraphDatabase",
                config: Optional[Dict[str, Any]] = None):
        """Initialize the GraphDatabaseSensor."""
        super().__init__(
            sensor_id="graph_database_sensor",
            component_name=component_name,
            config=config or {}
        )
        
        # Extract configuration values with defaults
        self.check_intervals = {
            "structure": self.config.get("structure_check_interval", 300),  # 5 minutes
            "performance": self.config.get("performance_check_interval", 60),  # 1 minute
            "integrity": self.config.get("integrity_check_interval", 600),  # 10 minutes
        }
        
        self.thresholds = {
            "max_orphaned_nodes_ratio": self.config.get("max_orphaned_nodes_ratio", 0.05),
            "max_query_time_ms": self.config.get("max_query_time_ms", 500),
            "min_node_degree": self.config.get("min_node_degree", 1),
            "max_circular_refs": self.config.get("max_circular_refs", 10)
        }
        
        # Initialize counters and performance metrics
        self.last_check_times = {check_type: 0 for check_type in self.check_intervals}
        self.performance_metrics = {
            "query_times": [],
            "node_count": 0,
            "edge_count": 0,
            "circular_refs": 0,
            "orphaned_nodes": 0
        }
        
        # Initialize cache to avoid repeated complex queries
        self.cache = {}
        self.cache_ttl = self.config.get("cache_ttl", 60)  # 1 minute cache TTL
        self.cache_last_update = 0
        
        logger.info(f"Initialized GraphDatabaseSensor for {component_name}")
    
    def monitor(self, data: Any = None) -> List[ErrorReport]:
        """
        Monitor the graph database for issues.
        
        Args:
            data: Optional data to use for monitoring, otherwise will query the DB
            
        Returns:
            List of error reports for detected issues
        """
        self.last_check_time = time.time()
        reports = []
        
        # Check if we need to update the cache
        if self.last_check_time - self.cache_last_update > self.cache_ttl:
            self._update_cache()
        
        # Perform structure check if needed
        if self.last_check_time - self.last_check_times["structure"] >= self.check_intervals["structure"]:
            reports.extend(self._check_structure())
            self.last_check_times["structure"] = self.last_check_time
        
        # Perform performance check if needed
        if self.last_check_time - self.last_check_times["performance"] >= self.check_intervals["performance"]:
            reports.extend(self._check_performance())
            self.last_check_times["performance"] = self.last_check_time
        
        # Perform integrity check if needed
        if self.last_check_time - self.last_check_times["integrity"] >= self.check_intervals["integrity"]:
            reports.extend(self._check_integrity())
            self.last_check_times["integrity"] = self.last_check_time
        
        return reports
    
    def _update_cache(self) -> None:
        """Update the internal cache with fresh data from the graph database."""
        try:
            # Get DB instance - safely handle if it's a class or singleton
            db = self._get_db_instance()
            
            # Time the query operation
            start_time = time.time()
            
            # Gather basic metrics
            self.cache["node_count"] = db.get_node_count() if hasattr(db, 'get_node_count') else 0
            self.cache["edge_count"] = db.get_edge_count() if hasattr(db, 'get_edge_count') else 0
            
            # Get orphaned nodes (nodes with no connections)
            if hasattr(db, 'get_orphaned_nodes'):
                self.cache["orphaned_nodes"] = db.get_orphaned_nodes()
            else:
                # Fallback using basic operations if specialized method doesn't exist
                self.cache["orphaned_nodes"] = self._detect_orphaned_nodes(db)
            
            # Get circular references
            if hasattr(db, 'detect_circular_references'):
                self.cache["circular_refs"] = db.detect_circular_references()
            else:
                # Fallback implementation
                self.cache["circular_refs"] = self._detect_circular_references(db)
            
            # Record query time
            query_time = (time.time() - start_time) * 1000  # Convert to ms
            self.performance_metrics["query_times"].append(query_time)
            
            # Keep only the last 10 query times
            if len(self.performance_metrics["query_times"]) > 10:
                self.performance_metrics["query_times"] = self.performance_metrics["query_times"][-10:]
            
            self.cache_last_update = self.last_check_time
            
        except Exception as e:
            logger.error(f"Error updating cache: {str(e)}")
            # Return gracefully, let the sensor continue with potentially stale data
    
    def _get_db_instance(self):
        """Get the graph database instance safely handling different patterns."""
        if hasattr(graph_database, 'instance') and callable(getattr(graph_database, 'instance', None)):
            return graph_database.instance()
        elif hasattr(graph_database, 'GraphDatabase'):
            return graph_database.GraphDatabase()
        elif hasattr(graph_database, 'graph_db'):
            return graph_database.graph_db
        else:
            # Last resort - assume the module itself is the instance
            return graph_database
    
    def _detect_orphaned_nodes(self, db) -> List[Any]:
        """Detect orphaned nodes (nodes with no connections)."""
        try:
            # Try to use the basic operations to find orphaned nodes
            orphaned = []
            
            # This is a simplified implementation - actual implementation would
            # depend on the specific graph database interface
            if hasattr(db, 'get_all_nodes') and hasattr(db, 'get_node_edges'):
                nodes = db.get_all_nodes()
                for node in nodes:
                    edges = db.get_node_edges(node)
                    if not edges:
                        orphaned.append(node)
            
            return orphaned
        except Exception as e:
            logger.error(f"Error detecting orphaned nodes: {str(e)}")
            return []
    
    def _detect_circular_references(self, db) -> List[List[Any]]:
        """Detect circular references in the graph."""
        try:
            # Try to use the basic operations to find circular references
            # This is a simplified placeholder - actual implementation would
            # use depth-first search or similar algorithm
            circular_refs = []
            
            # Placeholder for the actual implementation
            # In a real scenario, this would implement a graph traversal algorithm
            
            return circular_refs
        except Exception as e:
            logger.error(f"Error detecting circular references: {str(e)}")
            return []
    
    def _check_structure(self) -> List[ErrorReport]:
        """Check the graph database structure for issues."""
        reports = []
        
        try:
            # Check orphaned nodes ratio
            node_count = self.cache.get("node_count", 0)
            if node_count > 0:
                orphaned_nodes = self.cache.get("orphaned_nodes", [])
                orphaned_ratio = len(orphaned_nodes) / node_count
                
                if orphaned_ratio > self.thresholds["max_orphaned_nodes_ratio"]:
                    reports.append(self.report_error(
                        error_type="EXCESSIVE_ORPHANED_NODES",
                        severity="MEDIUM",
                        details={
                            "message": "Excessive orphaned nodes detected in graph database",
                            "orphaned_ratio": orphaned_ratio,
                            "threshold": self.thresholds["max_orphaned_nodes_ratio"],
                            "orphaned_count": len(orphaned_nodes),
                            "node_count": node_count
                        },
                        context={
                            "sample_nodes": orphaned_nodes[:5] if len(orphaned_nodes) > 5 else orphaned_nodes
                        }
                    ))
            
            # Check circular references
            circular_refs = self.cache.get("circular_refs", [])
            if len(circular_refs) > self.thresholds["max_circular_refs"]:
                reports.append(self.report_error(
                    error_type="EXCESSIVE_CIRCULAR_REFERENCES",
                    severity="HIGH",
                    details={
                        "message": "Excessive circular references detected in graph database",
                        "circular_refs_count": len(circular_refs),
                        "threshold": self.thresholds["max_circular_refs"]
                    },
                    context={
                        "sample_refs": circular_refs[:3] if len(circular_refs) > 3 else circular_refs
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error checking structure: {str(e)}")
            # Continue with other checks even if one fails
        
        return reports
    
    def _check_performance(self) -> List[ErrorReport]:
        """Check the graph database performance."""
        reports = []
        
        try:
            # Check query time
            query_times = self.performance_metrics.get("query_times", [])
            if query_times:
                avg_query_time = sum(query_times) / len(query_times)
                
                if avg_query_time > self.thresholds["max_query_time_ms"]:
                    reports.append(self.report_error(
                        error_type="SLOW_GRAPH_QUERIES",
                        severity="MEDIUM",
                        details={
                            "message": "Graph database queries are slower than threshold",
                            "avg_query_time_ms": avg_query_time,
                            "threshold_ms": self.thresholds["max_query_time_ms"]
                        },
                        context={
                            "recent_query_times_ms": query_times,
                            "node_count": self.cache.get("node_count", 0),
                            "edge_count": self.cache.get("edge_count", 0)
                        }
                    ))
        except Exception as e:
            logger.error(f"Error checking performance: {str(e)}")
            # Continue with other checks even if one fails
        
        return reports
    
    def _check_integrity(self) -> List[ErrorReport]:
        """Check the graph database integrity."""
        reports = []
        
        try:
            # Placeholder for more complex integrity checks
            # These would typically involve checking for data consistency,
            # validating that edges connect to valid nodes, etc.
            
            # For now, we just check if the database is accessible
            db = self._get_db_instance()
            if not db:
                reports.append(self.report_error(
                    error_type="GRAPH_DB_UNAVAILABLE",
                    severity="CRITICAL",
                    details={
                        "message": "Graph database is unavailable"
                    }
                ))
        except Exception as e:
            logger.error(f"Error checking integrity: {str(e)}")
            reports.append(self.report_error(
                error_type="GRAPH_DB_ERROR",
                severity="CRITICAL",
                details={
                    "message": f"Error accessing graph database: {str(e)}"
                }
            ))
        
        return reports
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the sensor and monitored component."""
        return {
            "sensor_id": self.sensor_id,
            "component_name": self.component_name,
            "last_check_time": self.last_check_time,
            "node_count": self.cache.get("node_count", 0),
            "edge_count": self.cache.get("edge_count", 0),
            "orphaned_nodes_count": len(self.cache.get("orphaned_nodes", [])),
            "circular_refs_count": len(self.cache.get("circular_refs", [])),
            "avg_query_time_ms": (
                sum(self.performance_metrics.get("query_times", [0])) / 
                max(1, len(self.performance_metrics.get("query_times", [0])))
            ),
        }


# Factory function to create a sensor instance
def create_graph_database_sensor(config: Optional[Dict[str, Any]] = None) -> GraphDatabaseSensor:
    """
    Create and initialize a graph database sensor.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized GraphDatabaseSensor
    """
    return GraphDatabaseSensor(config=config)
