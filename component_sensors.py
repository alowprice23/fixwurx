"""
FixWurx Auditor Component Sensors

This module implements specific sensors for each component of the auditor system.
These sensors monitor for errors, anomalies, and other issues in their respective
components and report them through the sensor registry.
"""

import logging
import datetime
import math
from typing import Dict, List, Set, Any, Optional, Union, Tuple, Type

# Import base sensor framework
from sensor_registry import ErrorSensor, ErrorReport

# Import auditor components for monitoring
from auditor import (
    ObligationLedger, RepoModules, EnergyCalculator,
    ProofMetrics, MetaAwareness
)
from graph_database import GraphDatabase, Node, Edge
from time_series_database import TimeSeriesDatabase
from document_store import DocumentStore
from benchmarking_system import BenchmarkingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [ComponentSensors] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('component_sensors')


class ObligationLedgerSensor(ErrorSensor):
    """
    Monitors the ObligationLedger component for errors related to
    Δ-Closure computation, rule application, and missing obligations.
    """
    
    def __init__(self, component_name: str = "ObligationLedger", config: Dict[str, Any] = None):
        """Initialize the ObligationLedger sensor."""
        super().__init__(
            sensor_id="obligation_ledger_sensor",
            component_name=component_name,
            config=config
        )
        # Specific thresholds for this sensor
        self.rule_application_threshold = self.config.get("rule_application_threshold", 0.95)
        self.max_missing_obligations = self.config.get("max_missing_obligations", 0)
        self.max_circular_dependencies = self.config.get("max_circular_dependencies", 0)
        
    def monitor(self, data: Any) -> List[ErrorReport]:
        """
        Monitor ObligationLedger data for errors.
        
        Args:
            data: ObligationLedger instance or data dictionary
            
        Returns:
            List of error reports
        """
        self.last_check_time = datetime.datetime.now()
        reports = []
        
        # Check data type
        if isinstance(data, ObligationLedger):
            ledger = data
            # Extract relevant data
            obligations = ledger.get_all() if hasattr(ledger, 'get_all') else set()
            delta_rules = ledger.delta_rules if hasattr(ledger, 'delta_rules') else []
        elif isinstance(data, dict):
            # Data is already a dictionary
            ledger = None
            obligations = data.get('obligations', set())
            delta_rules = data.get('delta_rules', [])
        else:
            # Unknown data type
            logger.warning(f"Unknown data type for ObligationLedger monitoring: {type(data)}")
            return reports
        
        # Check for empty obligations
        if not obligations:
            reports.append(self.report_error(
                error_type="EMPTY_OBLIGATIONS",
                severity="HIGH",
                details={"message": "No obligations found in ledger"},
                context={"obligations_count": 0}
            ))
        
        # Check for empty delta rules
        if not delta_rules:
            reports.append(self.report_error(
                error_type="EMPTY_DELTA_RULES",
                severity="HIGH",
                details={"message": "No Δ-rules found in ledger"},
                context={"rules_count": 0}
            ))
        
        # If we have both obligations and rules, check for rule application issues
        if obligations and delta_rules:
            reports.extend(self._check_rule_application(obligations, delta_rules))
            reports.extend(self._check_circular_dependencies(delta_rules))
        
        return reports
    
    def _check_rule_application(self, obligations: Set[str], delta_rules: List[Dict]) -> List[ErrorReport]:
        """
        Check for rule application issues.
        
        Args:
            obligations: Set of obligations
            delta_rules: List of delta rules
            
        Returns:
            List of error reports
        """
        reports = []
        
        # Count how many obligations have matching rules
        matched_obligations = 0
        
        for obligation in obligations:
            for rule in delta_rules:
                pattern = rule.get("pattern", "")
                if pattern in obligation:
                    matched_obligations += 1
                    break
        
        # Calculate match ratio
        if obligations:
            match_ratio = matched_obligations / len(obligations)
            
            # Check if below threshold
            if match_ratio < self.rule_application_threshold:
                reports.append(self.report_error(
                    error_type="RULE_APPLICATION_INSUFFICIENT",
                    severity="MEDIUM",
                    details={
                        "message": f"Rule application ratio below threshold",
                        "match_ratio": match_ratio,
                        "threshold": self.rule_application_threshold
                    },
                    context={
                        "total_obligations": len(obligations),
                        "matched_obligations": matched_obligations
                    }
                ))
        
        return reports
    
    def _check_circular_dependencies(self, delta_rules: List[Dict]) -> List[ErrorReport]:
        """
        Check for circular dependencies in delta rules.
        
        Args:
            delta_rules: List of delta rules
            
        Returns:
            List of error reports
        """
        reports = []
        
        # Build dependency graph
        dependencies = {}
        
        for rule in delta_rules:
            pattern = rule.get("pattern", "")
            transforms_to = rule.get("transforms_to", [])
            
            if pattern and transforms_to:
                if pattern not in dependencies:
                    dependencies[pattern] = set()
                
                for transform in transforms_to:
                    dependencies[pattern].add(transform)
        
        # Check for circular dependencies
        circular = self._find_circular_dependencies(dependencies)
        
        if circular and len(circular) > self.max_circular_dependencies:
            reports.append(self.report_error(
                error_type="CIRCULAR_DEPENDENCIES",
                severity="HIGH",
                details={
                    "message": "Circular dependencies detected in Δ-rules",
                    "circular_count": len(circular)
                },
                context={"circular_paths": list(circular)}
            ))
        
        return reports
    
    def _find_circular_dependencies(self, dependencies: Dict[str, Set[str]]) -> Set[Tuple[str, ...]]:
        """
        Find circular dependencies in the dependency graph.
        
        Args:
            dependencies: Dependency graph
            
        Returns:
            Set of circular dependency paths
        """
        circular = set()
        visited = set()
        path = []
        
        def dfs(node):
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                circular.add(tuple(path[cycle_start:]))
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for neighbor in dependencies.get(node, set()):
                dfs(neighbor)
            
            path.pop()
        
        # Run DFS from each node
        for node in dependencies:
            dfs(node)
        
        return circular


class EnergyCalculatorSensor(ErrorSensor):
    """
    Monitors the EnergyCalculator component for issues related to
    energy convergence, gradient calculation, and other mathematical properties.
    """
    
    def __init__(self, component_name: str = "EnergyCalculator", config: Dict[str, Any] = None):
        """Initialize the EnergyCalculator sensor."""
        super().__init__(
            sensor_id="energy_calculator_sensor",
            component_name=component_name,
            config=config
        )
        # Specific thresholds for this sensor
        self.energy_delta_threshold = self.config.get("energy_delta_threshold", 1e-7)
        self.lambda_threshold = self.config.get("lambda_threshold", 0.9)
        self.gradient_norm_min = self.config.get("gradient_norm_min", 0.0)
        
    def monitor(self, data: Any) -> List[ErrorReport]:
        """
        Monitor EnergyCalculator data for errors.
        
        Args:
            data: EnergyCalculator instance or data dictionary
            
        Returns:
            List of error reports
        """
        self.last_check_time = datetime.datetime.now()
        reports = []
        
        # Extract energy metrics
        if isinstance(data, EnergyCalculator):
            calculator = data
            # Get energy metrics
            try:
                E, delta_E, lamb = calculator.get_metrics()
                gradient_norm = calculator.calculate_gradient()
            except Exception as e:
                reports.append(self.report_error(
                    error_type="ENERGY_CALCULATION_ERROR",
                    severity="CRITICAL",
                    details={"message": f"Error calculating energy metrics: {str(e)}"},
                    context={"error": str(e)}
                ))
                return reports
        elif isinstance(data, dict):
            # Data is already a dictionary
            E = data.get('E', 0.0)
            delta_E = data.get('delta_E', 0.0)
            lamb = data.get('lambda', 0.0)
            gradient_norm = data.get('gradient_norm', 0.0)
        else:
            # Unknown data type
            logger.warning(f"Unknown data type for EnergyCalculator monitoring: {type(data)}")
            return reports
        
        # Check energy delta (convergence)
        if delta_E > self.energy_delta_threshold:
            reports.append(self.report_error(
                error_type="ENERGY_NOT_CONVERGED",
                severity="HIGH",
                details={
                    "message": "Energy not converged to minimum",
                    "delta_E": delta_E,
                    "threshold": self.energy_delta_threshold
                },
                context={"E": E, "lambda": lamb}
            ))
        
        # Check lambda (contraction constant)
        if lamb >= self.lambda_threshold:
            reports.append(self.report_error(
                error_type="LAMBDA_EXCEEDS_THRESHOLD",
                severity="MEDIUM",
                details={
                    "message": "Lambda exceeds threshold",
                    "lambda": lamb,
                    "threshold": self.lambda_threshold
                },
                context={"E": E, "delta_E": delta_E}
            ))
        
        # Check gradient norm
        if gradient_norm < self.gradient_norm_min:
            reports.append(self.report_error(
                error_type="GRADIENT_NORM_TOO_SMALL",
                severity="MEDIUM",
                details={
                    "message": "Gradient norm too small",
                    "gradient_norm": gradient_norm,
                    "min_threshold": self.gradient_norm_min
                },
                context={"E": E, "delta_E": delta_E, "lambda": lamb}
            ))
        
        return reports


class ProofMetricsSensor(ErrorSensor):
    """
    Monitors the ProofMetrics component for issues related to
    proof coverage, residual risk, and verification counts.
    """
    
    def __init__(self, component_name: str = "ProofMetrics", config: Dict[str, Any] = None):
        """Initialize the ProofMetrics sensor."""
        super().__init__(
            sensor_id="proof_metrics_sensor",
            component_name=component_name,
            config=config
        )
        # Specific thresholds for this sensor
        self.min_coverage = self.config.get("min_coverage", 0.9)
        self.max_bug_probability = self.config.get("max_bug_probability", 1.1e-4)
        self.min_verification_count = self.config.get("min_verification_count", 10)
        
    def monitor(self, data: Any) -> List[ErrorReport]:
        """
        Monitor ProofMetrics data for errors.
        
        Args:
            data: ProofMetrics instance or data dictionary
            
        Returns:
            List of error reports
        """
        self.last_check_time = datetime.datetime.now()
        reports = []
        
        # Extract proof metrics
        if isinstance(data, ProofMetrics):
            metrics = data
            # Get proof metrics
            try:
                f, n, rho, eps = metrics.get_metrics()
                # Calculate bug probability
                p_bug = (1 - f) * rho + f * 2 * math.exp(-2 * n * eps**2)
                verified_count = metrics.verified_obligations
                total_count = metrics.total_obligations
            except Exception as e:
                reports.append(self.report_error(
                    error_type="PROOF_METRICS_CALCULATION_ERROR",
                    severity="CRITICAL",
                    details={"message": f"Error calculating proof metrics: {str(e)}"},
                    context={"error": str(e)}
                ))
                return reports
        elif isinstance(data, dict):
            # Data is already a dictionary
            f = data.get('f', 0.0)
            n = data.get('n', 0)
            rho = data.get('rho', 0.0)
            eps = data.get('eps', 0.0)
            p_bug = data.get('p_bug', 0.0)
            verified_count = data.get('verified_count', 0)
            total_count = data.get('total_count', 0)
        else:
            # Unknown data type
            logger.warning(f"Unknown data type for ProofMetrics monitoring: {type(data)}")
            return reports
        
        # Check coverage
        if f < self.min_coverage:
            reports.append(self.report_error(
                error_type="COVERAGE_TOO_LOW",
                severity="HIGH",
                details={
                    "message": "Proof coverage below threshold",
                    "coverage": f,
                    "threshold": self.min_coverage
                },
                context={
                    "verified_count": verified_count,
                    "total_count": total_count
                }
            ))
        
        # Check bug probability
        if p_bug > self.max_bug_probability:
            reports.append(self.report_error(
                error_type="BUG_PROBABILITY_TOO_HIGH",
                severity="HIGH",
                details={
                    "message": "Residual bug probability exceeds threshold",
                    "p_bug": p_bug,
                    "threshold": self.max_bug_probability
                },
                context={"f": f, "n": n, "rho": rho, "eps": eps}
            ))
        
        # Check verification count
        if verified_count < self.min_verification_count:
            reports.append(self.report_error(
                error_type="VERIFICATION_COUNT_TOO_LOW",
                severity="MEDIUM",
                details={
                    "message": "Verification count below threshold",
                    "verified_count": verified_count,
                    "threshold": self.min_verification_count
                },
                context={"total_count": total_count, "coverage": f}
            ))
        
        return reports


class MetaAwarenessSensor(ErrorSensor):
    """
    Monitors the MetaAwareness component for issues related to
    semantic drift, reflection stability, and Lyapunov trends.
    """
    
    def __init__(self, component_name: str = "MetaAwareness", config: Dict[str, Any] = None):
        """Initialize the MetaAwareness sensor."""
        super().__init__(
            sensor_id="meta_awareness_sensor",
            component_name=component_name,
            config=config
        )
        # Specific thresholds for this sensor
        self.max_drift = self.config.get("max_drift", 0.02)
        self.max_perturbation = self.config.get("max_perturbation", 0.005)
        self.min_phi_decrease = self.config.get("min_phi_decrease", 0.0)
        
    def monitor(self, data: Any) -> List[ErrorReport]:
        """
        Monitor MetaAwareness data for errors.
        
        Args:
            data: MetaAwareness instance or data dictionary
            
        Returns:
            List of error reports
        """
        self.last_check_time = datetime.datetime.now()
        reports = []
        
        # Extract meta-awareness metrics
        if isinstance(data, MetaAwareness):
            meta = data
            # Get metrics
            try:
                drift = meta.semantic_drift()
                perturb = meta.lindblad_perturb()
                monotone = meta.lyapunov_monotone()
                phi_values = meta.phi_values
            except Exception as e:
                reports.append(self.report_error(
                    error_type="META_AWARENESS_CALCULATION_ERROR",
                    severity="CRITICAL",
                    details={"message": f"Error calculating meta-awareness metrics: {str(e)}"},
                    context={"error": str(e)}
                ))
                return reports
        elif isinstance(data, dict):
            # Data is already a dictionary
            drift = data.get('drift', 0.0)
            perturb = data.get('perturbation', 0.0)
            monotone = data.get('monotone', True)
            phi_values = data.get('phi_values', [])
        else:
            # Unknown data type
            logger.warning(f"Unknown data type for MetaAwareness monitoring: {type(data)}")
            return reports
        
        # Check semantic drift
        if drift > self.max_drift:
            reports.append(self.report_error(
                error_type="SEMANTIC_DRIFT_TOO_HIGH",
                severity="HIGH",
                details={
                    "message": "Semantic drift exceeds threshold",
                    "drift": drift,
                    "threshold": self.max_drift
                },
                context={"perturbation": perturb, "monotone": monotone}
            ))
        
        # Check perturbation
        if perturb > self.max_perturbation:
            reports.append(self.report_error(
                error_type="PERTURBATION_TOO_HIGH",
                severity="MEDIUM",
                details={
                    "message": "Reflection perturbation exceeds threshold",
                    "perturbation": perturb,
                    "threshold": self.max_perturbation
                },
                context={"drift": drift, "monotone": monotone}
            ))
        
        # Check Lyapunov monotonicity
        if not monotone:
            reports.append(self.report_error(
                error_type="LYAPUNOV_NOT_MONOTONE",
                severity="HIGH",
                details={
                    "message": "Lyapunov function not monotonically decreasing"
                },
                context={
                    "drift": drift,
                    "perturbation": perturb,
                    "phi_values": phi_values[-10:] if phi_values else []
                }
            ))
        
        # Check Phi values if available
        if phi_values and len(phi_values) >= 2:
            # Check if Phi is decreasing
            if phi_values[-1] >= phi_values[-2] + self.min_phi_decrease:
                reports.append(self.report_error(
                    error_type="PHI_NOT_DECREASING",
                    severity="MEDIUM",
                    details={
                        "message": "Phi value not decreasing",
                        "current_phi": phi_values[-1],
                        "previous_phi": phi_values[-2],
                        "min_decrease": self.min_phi_decrease
                    },
                    context={"phi_values": phi_values[-10:]}
                ))
        
        return reports


class GraphDatabaseSensor(ErrorSensor):
    """
    Monitors the GraphDatabase component for issues related to
    node/edge consistency, relationship integrity, and graph structure.
    """
    
    def __init__(self, component_name: str = "GraphDatabase", config: Dict[str, Any] = None):
        """Initialize the GraphDatabase sensor."""
        super().__init__(
            sensor_id="graph_database_sensor",
            component_name=component_name,
            config=config
        )
        # Specific thresholds for this sensor
        self.max_orphaned_nodes = self.config.get("max_orphaned_nodes", 0)
        self.max_dangling_edges = self.config.get("max_dangling_edges", 0)
        self.min_node_count = self.config.get("min_node_count", 1)
        
    def monitor(self, data: Any) -> List[ErrorReport]:
        """
        Monitor GraphDatabase data for errors.
        
        Args:
            data: GraphDatabase instance or data dictionary
            
        Returns:
            List of error reports
        """
        self.last_check_time = datetime.datetime.now()
        reports = []
        
        # Extract graph database metrics
        if isinstance(data, GraphDatabase):
            graph_db = data
            # Get nodes and edges
            try:
                nodes = graph_db.get_all_nodes()
                edges = graph_db.get_all_edges()
            except Exception as e:
                reports.append(self.report_error(
                    error_type="GRAPH_DATABASE_ACCESS_ERROR",
                    severity="CRITICAL",
                    details={"message": f"Error accessing graph database: {str(e)}"},
                    context={"error": str(e)}
                ))
                return reports
        elif isinstance(data, dict):
            # Data is already a dictionary
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
        else:
            # Unknown data type
            logger.warning(f"Unknown data type for GraphDatabase monitoring: {type(data)}")
            return reports
        
        # Check node count
        if len(nodes) < self.min_node_count:
            reports.append(self.report_error(
                error_type="INSUFFICIENT_NODES",
                severity="MEDIUM",
                details={
                    "message": "Insufficient nodes in graph database",
                    "node_count": len(nodes),
                    "threshold": self.min_node_count
                },
                context={"edge_count": len(edges)}
            ))
        
        # Check for dangling edges (edges referring to non-existent nodes)
        node_ids = {node.id for node in nodes} if hasattr(nodes[0], 'id') else set(nodes)
        dangling_edges = []
        
        for edge in edges:
            if hasattr(edge, 'source') and hasattr(edge, 'target'):
                source = edge.source
                target = edge.target
            else:
                source = edge.get('source')
                target = edge.get('target')
            
            if source not in node_ids or target not in node_ids:
                dangling_edges.append(edge)
        
        if len(dangling_edges) > self.max_dangling_edges:
            reports.append(self.report_error(
                error_type="DANGLING_EDGES",
                severity="HIGH",
                details={
                    "message": "Dangling edges detected in graph database",
                    "dangling_count": len(dangling_edges),
                    "threshold": self.max_dangling_edges
                },
                context={
                    "dangling_edges": [
                        {
                            "source": e.source if hasattr(e, 'source') else e.get('source'),
                            "target": e.target if hasattr(e, 'target') else e.get('target')
                        }
                        for e in dangling_edges[:10]  # First 10 for context
                    ]
                }
            ))
        
        # Check for orphaned nodes (nodes with no edges)
        connected_nodes = set()
        for edge in edges:
            if hasattr(edge, 'source') and hasattr(edge, 'target'):
                connected_nodes.add(edge.source)
                connected_nodes.add(edge.target)
            else:
                connected_nodes.add(edge.get('source'))
                connected_nodes.add(edge.get('target'))
        
        orphaned_nodes = node_ids - connected_nodes
        
        if len(orphaned_nodes) > self.max_orphaned_nodes:
            reports.append(self.report_error(
                error_type="ORPHANED_NODES",
                severity="MEDIUM",
                details={
                    "message": "Orphaned nodes detected in graph database",
                    "orphaned_count": len(orphaned_nodes),
                    "threshold": self.max_orphaned_nodes
                },
                context={"orphaned_nodes": list(orphaned_nodes)[:10]}  # First 10 for context
            ))
        
        return reports


class TimeSeriesDatabaseSensor(ErrorSensor):
    """
    Monitors the TimeSeriesDatabase component for issues related to
    data gaps, anomalies, and metric trends.
    """
    
    def __init__(self, component_name: str = "TimeSeriesDatabase", config: Dict[str, Any] = None):
        """Initialize the TimeSeriesDatabase sensor."""
        super().__init__(
            sensor_id="time_series_database_sensor",
            component_name=component_name,
            config=config
        )
        # Specific thresholds for this sensor
        self.max_gap_seconds = self.config.get("max_gap_seconds", 300)  # 5 minutes
        self.anomaly_z_score = self.config.get("anomaly_z_score", 3.0)
        self.min_points_for_analysis = self.config.get("min_points_for_analysis", 10)
        
    def monitor(self, data: Any) -> List[ErrorReport]:
        """
        Monitor TimeSeriesDatabase data for errors.
        
        Args:
            data: TimeSeriesDatabase instance or data dictionary
            
        Returns:
            List of error reports
        """
        self.last_check_time = datetime.datetime.now()
        reports = []
        
        # Extract time series database metrics
        if isinstance(data, TimeSeriesDatabase):
            ts_db = data
            # Get series
            try:
                all_series = ts_db.get_all_time_series()
            except Exception as e:
                reports.append(self.report_error(
                    error_type="TIME_SERIES_DATABASE_ACCESS_ERROR",
                    severity="CRITICAL",
                    details={"message": f"Error accessing time series database: {str(e)}"},
                    context={"error": str(e)}
                ))
                return reports
        elif isinstance(data, dict):
            # Data is already a dictionary
            all_series = data.get('time_series', [])
        else:
            # Unknown data type
            logger.warning(f"Unknown data type for TimeSeriesDatabase monitoring: {type(data)}")
            return reports
        
        # Check each time series
        for series in all_series:
            series_name = series.name if hasattr(series, 'name') else series.get('name')
            points = series.points if hasattr(series, 'points') else series.get('points', [])
            
            # Check point count
            if len(points) < self.min_points_for_analysis:
                reports.append(self.report_error(
                    error_type="INSUFFICIENT_DATA_POINTS",
                    severity="LOW",
                    details={
                        "message": f"Insufficient data points in time series '{series_name}'",
                        "point_count": len(points),
                        "threshold": self.min_points_for_analysis
                    },
                    context={"series_name": series_name}
                ))
                continue  # Skip further analysis for this series
            
            # Check for data gaps
            reports.extend(self._check_data_gaps(series_name, points))
            
            # Check for anomalies
            reports.extend(self._check_anomalies(series_name, points))
        
        return reports
    
    def _check_data_gaps(self, series_name: str, points: List[Any]) -> List[ErrorReport]:
        """
        Check for gaps in time series data.
        
        Args:
            series_name: Name of the time series
            points: List of data points
            
        Returns:
            List of error reports
        """
        reports = []
        
        # Sort points by timestamp
        sorted_points = sorted(
            points,
            key=lambda p: p.timestamp if hasattr(p, 'timestamp') else p.get('timestamp')
        )
        
        # Check for gaps
        gaps = []
        for i in range(1, len(sorted_points)):
            prev = sorted_points[i-1]
            curr = sorted_points[i]
            
            prev_time = prev.timestamp if hasattr(prev, 'timestamp') else prev.get('timestamp')
            curr_time = curr.timestamp if hasattr(curr, 'timestamp') else curr.get('timestamp')
            
            if isinstance(prev_time, str):
                prev_time = datetime.datetime.fromisoformat(prev_time)
            if isinstance(curr_time, str):
                curr_time = datetime.datetime.fromisoformat(curr_time)
            
            gap_seconds = (curr_time - prev_time).total_seconds()
            
            if gap_seconds > self.max_gap_seconds:
                gaps.append({
                    "start_time": prev_time.isoformat(),
                    "end_time": curr_time.isoformat(),
                    "gap_seconds": gap_seconds
                })
        
        if gaps:
            reports.append(self.report_error(
                error_type="TIME_SERIES_DATA_GAPS",
                severity="MEDIUM",
                details={
                    "message": f"Data gaps detected in time series '{series_name}'",
                    "gap_count": len(gaps),
                    "max_gap_seconds": self.max_gap_seconds
                },
                context={"gaps": gaps[:5]}  # First 5 gaps for context
            ))
        
        return reports
    
    def _check_anomalies(self, series_name: str, points: List[Any]) -> List[ErrorReport]:
        """
        Check for anomalies in time series data.
        
        Args:
            series_name: Name of the time series
            points: List of data points
            
        Returns:
            List of error reports
        """
        reports = []
        
        # Extract values for each metric
        metrics = {}
        
        for point in points:
            values = point.values if hasattr(point, 'values') else point.get('values', {})
            timestamp = point.timestamp if hasattr(point, 'timestamp') else point.get('timestamp')
            
            # Add each metric value to its corresponding list
            for metric_name, value in values.items():
                if isinstance(value, (int, float)):
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append((timestamp, value))
        
        # Check each metric for anomalies
        for metric_name, values in metrics.items():
            # Need enough values for statistical analysis
            if len(values) < self.min_points_for_analysis:
                continue
                
            # Extract just the values (not timestamps) for statistical analysis
            metric_values = [v[1] for v in values]
            
            # Calculate mean and standard deviation
            mean = sum(metric_values) / len(metric_values)
            std_dev = (sum((x - mean) ** 2 for x in metric_values) / len(metric_values)) ** 0.5
            
            if std_dev == 0:
                # No variation, skip anomaly detection
                continue
                
            # Find anomalies (values more than z standard deviations from mean)
            anomalies = []
            for timestamp, value in values:
                z_score = abs(value - mean) / std_dev
                if z_score > self.anomaly_z_score:
                    anomalies.append({
                        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime.datetime) else timestamp,
                        "value": value,
                        "z_score": z_score,
                        "mean": mean,
                        "std_dev": std_dev
                    })
            
            # Report if anomalies found
            if anomalies:
                reports.append(self.report_error(
                    error_type="TIME_SERIES_ANOMALIES",
                    severity="HIGH",
                    details={
                        "message": f"Anomalies detected in metric '{metric_name}' of series '{series_name}'",
                        "anomaly_count": len(anomalies),
                        "z_score_threshold": self.anomaly_z_score
                    },
                    context={
                        "anomalies": anomalies[:5],  # First 5 anomalies for context
                        "mean": mean,
                        "std_dev": std_dev
                    }
                ))
        
        return reports


class DocumentStoreSensor(ErrorSensor):
    """
    Monitors the DocumentStore component for issues related to
    document integrity, schema validation, and reference consistency.
    """
    
    def __init__(self, component_name: str = "DocumentStore", config: Dict[str, Any] = None):
        """Initialize the DocumentStore sensor."""
        super().__init__(
            sensor_id="document_store_sensor",
            component_name=component_name,
            config=config
        )
        # Specific thresholds for this sensor
        self.max_invalid_documents = self.config.get("max_invalid_documents", 0)
        self.max_missing_references = self.config.get("max_missing_references", 0)
        self.min_collection_size = self.config.get("min_collection_size", 1)
        
    def monitor(self, data: Any) -> List[ErrorReport]:
        """
        Monitor DocumentStore data for errors.
        
        Args:
            data: DocumentStore instance or data dictionary
            
        Returns:
            List of error reports
        """
        self.last_check_time = datetime.datetime.now()
        reports = []
        
        # Extract document store data
        if isinstance(data, DocumentStore):
            doc_store = data
            # Get collections and documents
            try:
                collections = doc_store.list_collections()
                collection_data = {}
                
                for collection_name in collections:
                    collection = doc_store.get_collection(collection_name)
                    if collection:
                        documents = collection.get_all_documents()
                        collection_data[collection_name] = {
                            "documents": documents,
                            "document_count": len(documents)
                        }
            except Exception as e:
                reports.append(self.report_error(
                    error_type="DOCUMENT_STORE_ACCESS_ERROR",
                    severity="CRITICAL",
                    details={"message": f"Error accessing document store: {str(e)}"},
                    context={"error": str(e)}
                ))
                return reports
        elif isinstance(data, dict):
            # Data is already a dictionary
            collections = data.get('collections', [])
            collection_data = data.get('collection_data', {})
        else:
            # Unknown data type
            logger.warning(f"Unknown data type for DocumentStore monitoring: {type(data)}")
            return reports
        
        # Check for empty collections
        if not collections:
            reports.append(self.report_error(
                error_type="NO_COLLECTIONS",
                severity="MEDIUM",
                details={"message": "No collections found in document store"},
                context={"collections_count": 0}
            ))
        
        # Check each collection
        for collection_name, data in collection_data.items():
            documents = data.get("documents", [])
            document_count = data.get("document_count", len(documents))
            
            # Check collection size
            if document_count < self.min_collection_size:
                reports.append(self.report_error(
                    error_type="COLLECTION_TOO_SMALL",
                    severity="LOW",
                    details={
                        "message": f"Collection '{collection_name}' has too few documents",
                        "document_count": document_count,
                        "threshold": self.min_collection_size
                    },
                    context={"collection_name": collection_name}
                ))
            
            # Check document integrity
            reports.extend(self._check_document_integrity(collection_name, documents))
            
            # Check for missing references
            reports.extend(self._check_missing_references(collection_name, documents, collection_data))
        
        return reports
    
    def _check_document_integrity(self, collection_name: str, documents: List[Any]) -> List[ErrorReport]:
        """
        Check for document integrity issues.
        
        Args:
            collection_name: Name of the collection
            documents: List of documents
            
        Returns:
            List of error reports
        """
        reports = []
        invalid_documents = []
        
        for document in documents:
            # Check for required fields
            doc_id = document.id if hasattr(document, 'id') else document.get('id')
            doc_type = document.type if hasattr(document, 'type') else document.get('type')
            fields = document.fields if hasattr(document, 'fields') else document.get('fields', {})
            
            if not doc_id or not doc_type:
                invalid_documents.append({
                    "id": doc_id or "unknown",
                    "reason": "Missing required fields (id or type)"
                })
                continue
            
            # Check fields based on document type
            # This would be more sophisticated in a real implementation,
            # potentially with schema validation per document type
            if not fields:
                invalid_documents.append({
                    "id": doc_id,
                    "reason": "Empty fields"
                })
        
        if len(invalid_documents) > self.max_invalid_documents:
            reports.append(self.report_error(
                error_type="INVALID_DOCUMENTS",
                severity="HIGH",
                details={
                    "message": f"Invalid documents detected in collection '{collection_name}'",
                    "invalid_count": len(invalid_documents),
                    "threshold": self.max_invalid_documents
                },
                context={
                    "invalid_documents": invalid_documents[:10]  # First 10 for context
                }
            ))
        
        return reports
    
    def _check_missing_references(self, collection_name: str, documents: List[Any],
                                collection_data: Dict[str, Any]) -> List[ErrorReport]:
        """
        Check for missing references between documents.
        
        Args:
            collection_name: Name of the collection
            documents: List of documents
            collection_data: Data for all collections
            
        Returns:
            List of error reports
        """
        reports = []
        missing_references = []
        
        # Build document ID sets for each collection
        document_ids = {}
        for coll_name, data in collection_data.items():
            coll_documents = data.get("documents", [])
            document_ids[coll_name] = {
                doc.id if hasattr(doc, 'id') else doc.get('id')
                for doc in coll_documents
            }
        
        # Check for references in fields
        for document in documents:
            doc_id = document.id if hasattr(document, 'id') else document.get('id')
            fields = document.fields if hasattr(document, 'fields') else document.get('fields', {})
            
            # Look for fields that appear to be references
            for field_name, field_value in fields.items():
                if field_name.endswith('_id') or field_name.endswith('_ref') or field_name == 'parent_id':
                    # Determine which collection this should reference
                    target_collection = field_name.replace('_id', '').replace('_ref', '')
                    
                    # If field points to same collection by default
                    if target_collection == 'parent' or target_collection == '':
                        target_collection = collection_name
                    
                    # Check if reference exists
                    if target_collection in document_ids and field_value not in document_ids[target_collection]:
                        missing_references.append({
                            "source_id": doc_id,
                            "source_collection": collection_name,
                            "reference_field": field_name,
                            "reference_value": field_value,
                            "target_collection": target_collection
                        })
        
        if len(missing_references) > self.max_missing_references:
            reports.append(self.report_error(
                error_type="MISSING_REFERENCES",
                severity="HIGH",
                details={
                    "message": f"Missing references detected in collection '{collection_name}'",
                    "missing_count": len(missing_references),
                    "threshold": self.max_missing_references
                },
                context={
                    "missing_references": missing_references[:10]  # First 10 for context
                }
            ))
        
        return reports


class BenchmarkingSensor(ErrorSensor):
    """
    Monitors the BenchmarkingSystem component for issues related to
    performance regression, test failures, and benchmark consistency.
    """
    
    def __init__(self, component_name: str = "BenchmarkingSystem", config: Dict[str, Any] = None):
        """Initialize the BenchmarkingSystem sensor."""
        super().__init__(
            sensor_id="benchmarking_sensor",
            component_name=component_name,
            config=config
        )
        # Specific thresholds for this sensor
        self.regression_threshold_pct = self.config.get("regression_threshold_pct", 10)  # 10% regression
        self.max_std_deviation_pct = self.config.get("max_std_deviation_pct", 5)  # 5% std deviation
        self.min_iterations = self.config.get("min_iterations", 3)
        
    def monitor(self, data: Any) -> List[ErrorReport]:
        """
        Monitor BenchmarkingSystem data for errors.
        
        Args:
            data: BenchmarkingSystem instance or data dictionary
            
        Returns:
            List of error reports
        """
        self.last_check_time = datetime.datetime.now()
        reports = []
        
        # Extract benchmarking data
        if isinstance(data, BenchmarkingSystem):
            benchmark_system = data
            # Get benchmarks
            try:
                benchmarks = benchmark_system.get_all_benchmarks()
            except Exception as e:
                reports.append(self.report_error(
                    error_type="BENCHMARKING_SYSTEM_ACCESS_ERROR",
                    severity="CRITICAL",
                    details={"message": f"Error accessing benchmarking system: {str(e)}"},
                    context={"error": str(e)}
                ))
                return reports
        elif isinstance(data, dict):
            # Data is already a dictionary
            benchmarks = data.get('benchmarks', [])
        else:
            # Unknown data type
            logger.warning(f"Unknown data type for BenchmarkingSystem monitoring: {type(data)}")
            return reports
        
        # Check for empty benchmarks
        if not benchmarks:
            reports.append(self.report_error(
                error_type="NO_BENCHMARKS",
                severity="MEDIUM",
                details={"message": "No benchmarks found in benchmarking system"},
                context={"benchmarks_count": 0}
            ))
            return reports
        
        # Group benchmarks by name
        benchmark_groups = {}
        for benchmark in benchmarks:
            name = benchmark.config.name if hasattr(benchmark, 'config') else benchmark.get('name')
            if name not in benchmark_groups:
                benchmark_groups[name] = []
            benchmark_groups[name].append(benchmark)
        
        # Check each benchmark group
        for name, group in benchmark_groups.items():
            # Sort by timestamp
            sorted_group = sorted(
                group,
                key=lambda b: b.timestamp if hasattr(b, 'timestamp') else b.get('timestamp')
            )
            
            # Check for performance regressions
            reports.extend(self._check_performance_regression(name, sorted_group))
            
            # Check for consistency issues
            reports.extend(self._check_benchmark_consistency(name, sorted_group))
        
        return reports
    
    def _check_performance_regression(self, benchmark_name: str, benchmarks: List[Any]) -> List[ErrorReport]:
        """
        Check for performance regressions in benchmarks.
        
        Args:
            benchmark_name: Name of the benchmark
            benchmarks: List of benchmark results, sorted by timestamp
            
        Returns:
            List of error reports
        """
        reports = []
        
        # Need at least 2 benchmarks to compare
        if len(benchmarks) < 2:
            return reports
        
        # Get latest and previous benchmark
        latest = benchmarks[-1]
        previous = benchmarks[-2]
        
        # Extract metrics
        latest_metrics = latest.statistics if hasattr(latest, 'statistics') else latest.get('statistics', {})
        previous_metrics = previous.statistics if hasattr(previous, 'statistics') else previous.get('statistics', {})
        
        # Check each metric for regression
        regressions = []
        
        for metric_name in set(latest_metrics.keys()) | set(previous_metrics.keys()):
            if metric_name not in latest_metrics or metric_name not in previous_metrics:
                continue
                
            latest_value = latest_metrics[metric_name].get('mean', 0) if isinstance(latest_metrics[metric_name], dict) else latest_metrics[metric_name]
            previous_value = previous_metrics[metric_name].get('mean', 0) if isinstance(previous_metrics[metric_name], dict) else previous_metrics[metric_name]
            
            # Skip if values are zero or not numeric
            if not isinstance(latest_value, (int, float)) or not isinstance(previous_value, (int, float)):
                continue
            if previous_value == 0:
                continue
                
            # Calculate percent change
            percent_change = ((latest_value - previous_value) / previous_value) * 100
            
            # Determine if this is a regression
            # For metrics where higher is worse (like execution_time), positive change is a regression
            # For metrics where higher is better (like throughput), negative change is a regression
            is_regression = False
            
            if metric_name in ['execution_time', 'latency', 'memory_usage', 'cpu_usage']:
                is_regression = percent_change > self.regression_threshold_pct
            else:
                is_regression = percent_change < -self.regression_threshold_pct
            
            if is_regression:
                regressions.append({
                    "metric": metric_name,
                    "previous_value": previous_value,
                    "latest_value": latest_value,
                    "percent_change": percent_change
                })
        
        if regressions:
            reports.append(self.report_error(
                error_type="PERFORMANCE_REGRESSION",
                severity="HIGH",
                details={
                    "message": f"Performance regression detected in benchmark '{benchmark_name}'",
                    "regression_count": len(regressions),
                    "threshold_pct": self.regression_threshold_pct
                },
                context={
                    "regressions": regressions,
                    "latest_timestamp": latest.timestamp if hasattr(latest, 'timestamp') else latest.get('timestamp'),
                    "previous_timestamp": previous.timestamp if hasattr(previous, 'timestamp') else previous.get('timestamp')
                }
            ))
        
        return reports
    
    def _check_benchmark_consistency(self, benchmark_name: str, benchmarks: List[Any]) -> List[ErrorReport]:
        """
        Check for consistency issues in benchmarks.
        
        Args:
            benchmark_name: Name of the benchmark
            benchmarks: List of benchmark results
            
        Returns:
            List of error reports
        """
        reports = []
        
        # Get latest benchmark
        if not benchmarks:
            return reports
        
        latest = benchmarks[-1]
        
        # Check iteration count
        iterations = latest.config.iterations if hasattr(latest, 'config') else latest.get('iterations', 0)
        
        if iterations < self.min_iterations:
            reports.append(self.report_error(
                error_type="INSUFFICIENT_ITERATIONS",
                severity="MEDIUM",
                details={
                    "message": f"Insufficient iterations in benchmark '{benchmark_name}'",
                    "iterations": iterations,
                    "threshold": self.min_iterations
                },
                context={"benchmark_name": benchmark_name}
            ))
        
        # Check standard deviation
        statistics = latest.statistics if hasattr(latest, 'statistics') else latest.get('statistics', {})
        
        for metric_name, metric_data in statistics.items():
            if not isinstance(metric_data, dict):
                continue
                
            mean = metric_data.get('mean', 0)
            std_dev = metric_data.get('std_dev', 0)
            
            if mean == 0 or not isinstance(mean, (int, float)) or not isinstance(std_dev, (int, float)):
                continue
                
            # Calculate percent std deviation
            std_dev_pct = (std_dev / mean) * 100
            
            if std_dev_pct > self.max_std_deviation_pct:
                reports.append(self.report_error(
                    error_type="HIGH_VARIANCE",
                    severity="MEDIUM",
                    details={
                        "message": f"High variance in metric '{metric_name}' of benchmark '{benchmark_name}'",
                        "std_dev_pct": std_dev_pct,
                        "threshold_pct": self.max_std_deviation_pct
                    },
                    context={
                        "metric": metric_name,
                        "mean": mean,
                        "std_dev": std_dev
                    }
                ))
        
        return reports
