"""
FixWurx Auditor Performance Benchmark Sensor

This module implements a comprehensive sensor for monitoring the auditor's 
performance metrics, including debugging effectiveness, repair efficiency,
and self-awareness indicators. It tracks 20 key performance indicators that
provide a complete view of the auditor's capabilities.
"""

import logging
import datetime
import math
import statistics
from typing import Dict, List, Set, Any, Optional, Union, Tuple, Callable

# Import sensor base class
from sensor_base import ErrorSensor
from error_report import ErrorReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [PerformanceBenchmarkSensor] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('performance_benchmark_sensor')


class PerformanceBenchmarkSensor(ErrorSensor):
    """
    Monitors the auditor's performance across 20 key metrics, tracking debugging
    effectiveness, repair efficiency, and self-awareness capabilities.
    """
    
    def __init__(self, component_name: str = "AuditorPerformance", config: Dict[str, Any] = None):
        """Initialize the Performance Benchmark sensor."""
        super().__init__(
            sensor_id="performance_benchmark_sensor",
            component_name=component_name,
            config=config
        )
        
        # Initialize benchmark thresholds from config
        self._init_thresholds()
        
        # Initialize session data
        self.session_start_time = datetime.datetime.now()
        self.session_metrics = {}
        self.history = {metric: [] for metric in self._get_all_metric_names()}
        self.reset_session()
        
        logger.info("Performance Benchmark Sensor initialized with 20 KPIs")
    
    def _init_thresholds(self):
        """Initialize threshold values for benchmark metrics."""
        self.thresholds = {
            # Detection and fix metrics
            "bug_detection_recall": self._get_threshold("bug_detection_recall", 0.8),
            "bug_fix_yield": self._get_threshold("bug_fix_yield", 0.7),
            "mttd": self._get_threshold("mttd", 10),  # iterations
            "mttr": self._get_threshold("mttr", 20),  # iterations
            
            # Energy and convergence metrics
            "convergence_iterations": self._get_threshold("convergence_iterations", 50),
            "energy_reduction_pct": self._get_threshold("energy_reduction_pct", 0.7),
            "proof_coverage_delta": self._get_threshold("proof_coverage_delta", 0.1),
            "residual_risk_improvement": self._get_threshold("residual_risk_improvement", 0.01),
            
            # Test and quality metrics
            "test_pass_ratio": self._get_threshold("test_pass_ratio", 0.95),
            "regression_introduction_rate": self._get_threshold("regression_introduction_rate", 0.2),
            "duplicate_module_ratio": self._get_threshold("duplicate_module_ratio", 0.1),
            
            # Agent and resource metrics
            "agent_coordination_overhead": self._get_threshold("agent_coordination_overhead", 10),
            "token_per_fix_efficiency": self._get_threshold("token_per_fix_efficiency", 5000),
            "hallucination_rate": self._get_threshold("hallucination_rate", 0.05),
            
            # Time and stability metrics
            "certainty_gap_closure_time": self._get_threshold("certainty_gap_closure_time", 300),  # seconds
            "lyapunov_descent_consistency": self._get_threshold("lyapunov_descent_consistency", 0.95),
            "meta_guard_breach_count": self._get_threshold("meta_guard_breach_count", 1),
            
            # Resource and documentation metrics
            "resource_footprint_change": self._get_threshold("resource_footprint_change", 0.2),
            "documentation_completeness_delta": self._get_threshold("documentation_completeness_delta", 0.0),
            "aggregate_confidence_score": self._get_threshold("aggregate_confidence_score", 0.8)
        }
    
    def _get_all_metric_names(self) -> List[str]:
        """Get names of all metrics tracked by this sensor."""
        return list(self.thresholds.keys())
    
    def reset_session(self):
        """Reset the session metrics for a new debugging session."""
        self.session_start_time = datetime.datetime.now()
        self.session_metrics = {
            # Session metadata
            "session_id": f"session-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "start_time": self.session_start_time.isoformat(),
            "end_time": None,
            "duration_seconds": 0,
            
            # Bug detection and fix metrics
            "detected_bugs": 0,
            "total_known_bugs": 0,
            "resolved_bugs": 0,
            "diagnose_iterations": [],
            "repair_iterations": [],
            
            # Energy and convergence metrics
            "energy_start": 0.0,
            "energy_current": 0.0,
            "lambda_current": 0.0,
            "convergence_iter_count": 0,
            "proof_coverage_start": 0.0,
            "proof_coverage_current": 0.0,
            "residual_risk_start": 0.0,
            "residual_risk_current": 0.0,
            
            # Test metrics
            "tests_passed": 0,
            "tests_total": 0,
            "new_failures": 0,
            "patches_applied": 0,
            
            # Code and agent metrics
            "redundant_ast_nodes": 0,
            "total_ast_nodes": 1,  # Avoid division by zero
            "agent_messages": 0,
            "tokens_used_start": 0,
            "tokens_used_current": 0,
            "lines_without_obligation": 0,
            "total_generated_lines": 1,  # Avoid division by zero
            
            # Time and stability metrics
            "auditor_fail_time": None,
            "auditor_pass_time": None,
            "lyapunov_decreasing_iterations": 0,
            "total_iterations": 0,
            "meta_guard_breaches": 0,
            
            # Resource and documentation metrics
            "memory_usage_start_mb": 0.0,
            "memory_usage_current_mb": 0.0,
            "cpu_usage_start_pct": 0.0,
            "cpu_usage_current_pct": 0.0,
            "doc_lines_start": 0,
            "code_lines_start": 1,  # Avoid division by zero
            "doc_lines_current": 0,
            "code_lines_current": 1,  # Avoid division by zero
        }
        
        logger.info(f"Started new benchmarking session: {self.session_metrics['session_id']}")
    
    def monitor(self, data: Any) -> List[ErrorReport]:
        """
        Monitor performance benchmark data.
        
        Args:
            data: Benchmark data or update information
            
        Returns:
            List of error reports for metrics that exceed thresholds
        """
        self.last_check_time = datetime.datetime.now()
        reports = []
        
        # Update metrics with the provided data
        if isinstance(data, dict):
            self._update_metrics(data)
        else:
            logger.warning(f"Unknown data type for performance benchmarking: {type(data)}")
            return reports
        
        # Calculate derived metrics
        self._calculate_derived_metrics()
        
        # Check metrics against thresholds
        reports.extend(self._check_detection_fix_metrics())
        reports.extend(self._check_energy_convergence_metrics())
        reports.extend(self._check_test_quality_metrics())
        reports.extend(self._check_agent_resource_metrics())
        reports.extend(self._check_time_stability_metrics())
        reports.extend(self._check_resource_documentation_metrics())
        
        # Update the metric history
        self._update_history()
        
        return reports
    
    def _update_metrics(self, data: Dict[str, Any]) -> None:
        """
        Update session metrics with new data.
        
        Args:
            data: Dictionary of metrics to update
        """
        # Update metrics that are directly provided
        for key, value in data.items():
            if key in self.session_metrics:
                self.session_metrics[key] = value
        
        # Handle special update commands
        if "bug_detected" in data and data["bug_detected"]:
            self.session_metrics["detected_bugs"] += 1
        
        if "bug_resolved" in data and data["bug_resolved"]:
            self.session_metrics["resolved_bugs"] += 1
        
        if "diagnose_iteration" in data and data["diagnose_iteration"]:
            self.session_metrics["diagnose_iterations"].append(data.get("iteration_count", 1))
        
        if "repair_iteration" in data and data["repair_iteration"]:
            self.session_metrics["repair_iterations"].append(data.get("iteration_count", 1))
        
        if "auditor_fail" in data and data["auditor_fail"]:
            self.session_metrics["auditor_fail_time"] = datetime.datetime.now()
        
        if "auditor_pass" in data and data["auditor_pass"]:
            self.session_metrics["auditor_pass_time"] = datetime.datetime.now()
        
        if "patch_applied" in data and data["patch_applied"]:
            self.session_metrics["patches_applied"] += 1
            if "new_failures" in data:
                self.session_metrics["new_failures"] += data["new_failures"]
        
        if "lyapunov_update" in data and data["lyapunov_update"]:
            self.session_metrics["total_iterations"] += 1
            if data.get("decreasing", False):
                self.session_metrics["lyapunov_decreasing_iterations"] += 1
        
        if "meta_guard_breach" in data and data["meta_guard_breach"]:
            self.session_metrics["meta_guard_breaches"] += 1
        
        # Update session duration
        self.session_metrics["end_time"] = datetime.datetime.now().isoformat()
        duration = (datetime.datetime.now() - self.session_start_time).total_seconds()
        self.session_metrics["duration_seconds"] = duration
    
    def _calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from raw session data."""
        metrics = self.session_metrics
        
        # Store derived metrics in a separate structure to avoid modifying while iterating
        derived = {}
        
        # Bug detection and fix metrics
        if metrics["total_known_bugs"] > 0:
            derived["bug_detection_recall"] = metrics["detected_bugs"] / metrics["total_known_bugs"]
        else:
            derived["bug_detection_recall"] = 1.0  # Perfect recall if no known bugs
        
        if metrics["detected_bugs"] > 0:
            derived["bug_fix_yield"] = metrics["resolved_bugs"] / metrics["detected_bugs"]
        else:
            derived["bug_fix_yield"] = 1.0  # Perfect yield if no detected bugs
        
        # Mean time metrics
        if metrics["diagnose_iterations"]:
            derived["mttd"] = statistics.mean(metrics["diagnose_iterations"])
        else:
            derived["mttd"] = 0
        
        if metrics["repair_iterations"]:
            derived["mttr"] = statistics.mean(metrics["repair_iterations"])
        else:
            derived["mttr"] = 0
        
        # Energy and convergence metrics
        if metrics["energy_start"] > 0:
            derived["energy_reduction_pct"] = (metrics["energy_start"] - metrics["energy_current"]) / metrics["energy_start"]
        else:
            derived["energy_reduction_pct"] = 0
        
        derived["proof_coverage_delta"] = metrics["proof_coverage_current"] - metrics["proof_coverage_start"]
        derived["residual_risk_improvement"] = metrics["residual_risk_start"] - metrics["residual_risk_current"]
        
        # Test metrics
        if metrics["tests_total"] > 0:
            derived["test_pass_ratio"] = metrics["tests_passed"] / metrics["tests_total"]
        else:
            derived["test_pass_ratio"] = 1.0  # Perfect ratio if no tests
        
        if metrics["patches_applied"] > 0:
            derived["regression_introduction_rate"] = metrics["new_failures"] / metrics["patches_applied"]
        else:
            derived["regression_introduction_rate"] = 0  # No regressions if no patches
        
        # Code and agent metrics
        derived["duplicate_module_ratio"] = metrics["redundant_ast_nodes"] / metrics["total_ast_nodes"]
        
        if metrics["resolved_bugs"] > 0:
            derived["agent_coordination_overhead"] = metrics["agent_messages"] / metrics["resolved_bugs"]
            derived["token_per_fix_efficiency"] = (metrics["tokens_used_current"] - metrics["tokens_used_start"]) / metrics["resolved_bugs"]
        else:
            derived["agent_coordination_overhead"] = metrics["agent_messages"]
            derived["token_per_fix_efficiency"] = metrics["tokens_used_current"] - metrics["tokens_used_start"]
        
        derived["hallucination_rate"] = metrics["lines_without_obligation"] / metrics["total_generated_lines"]
        
        # Time and stability metrics
        if metrics["auditor_fail_time"] and metrics["auditor_pass_time"]:
            derived["certainty_gap_closure_time"] = (metrics["auditor_pass_time"] - metrics["auditor_fail_time"]).total_seconds()
        else:
            derived["certainty_gap_closure_time"] = metrics["duration_seconds"]  # Use session duration as fallback
        
        if metrics["total_iterations"] > 0:
            derived["lyapunov_descent_consistency"] = metrics["lyapunov_decreasing_iterations"] / metrics["total_iterations"]
        else:
            derived["lyapunov_descent_consistency"] = 1.0  # Perfect consistency if no iterations
        
        # Resource and documentation metrics
        derived["resource_footprint_change"] = (
            (metrics["memory_usage_current_mb"] / max(metrics["memory_usage_start_mb"], 1)) - 1 +
            (metrics["cpu_usage_current_pct"] / max(metrics["cpu_usage_start_pct"], 1)) - 1
        ) / 2  # Average of memory and CPU change
        
        doc_ratio_start = metrics["doc_lines_start"] / metrics["code_lines_start"]
        doc_ratio_current = metrics["doc_lines_current"] / metrics["code_lines_current"]
        derived["documentation_completeness_delta"] = doc_ratio_current - doc_ratio_start
        
        # Calculate aggregate confidence score (weighted average of key metrics)
        weights = {
            "bug_detection_recall": 0.1,
            "bug_fix_yield": 0.15,
            "energy_reduction_pct": 0.1,
            "test_pass_ratio": 0.15,
            "regression_introduction_rate": 0.1,
            "lyapunov_descent_consistency": 0.1,
            "hallucination_rate": 0.1,
            "documentation_completeness_delta": 0.05,
            "resource_footprint_change": 0.05,
            "meta_guard_breach_count": 0.1
        }
        
        # For regression_introduction_rate, resource_footprint_change, hallucination_rate, 
        # and meta_guard_breach_count, lower values are better, so we use (1 - value)
        score_components = {
            "bug_detection_recall": derived["bug_detection_recall"],
            "bug_fix_yield": derived["bug_fix_yield"],
            "energy_reduction_pct": derived["energy_reduction_pct"],
            "test_pass_ratio": derived["test_pass_ratio"],
            "regression_introduction_rate": 1 - min(derived["regression_introduction_rate"], 1),
            "lyapunov_descent_consistency": derived["lyapunov_descent_consistency"],
            "hallucination_rate": 1 - derived["hallucination_rate"],
            "documentation_completeness_delta": max(0, derived["documentation_completeness_delta"]),
            "resource_footprint_change": max(0, 1 - derived["resource_footprint_change"]),
            "meta_guard_breach_count": 1 - min(metrics["meta_guard_breaches"] / 10, 1)  # Scale to 0-1
        }
        
        weighted_sum = sum(weights[k] * score_components[k] for k in weights)
        total_weight = sum(weights.values())
        derived["aggregate_confidence_score"] = weighted_sum / total_weight
        
        # Add derived metrics to the session metrics
        for key, value in derived.items():
            self.session_metrics[key] = value
    
    def _check_detection_fix_metrics(self) -> List[ErrorReport]:
        """Check detection and fix metrics against thresholds."""
        reports = []
        metrics = self.session_metrics
        
        # Bug detection recall
        if "bug_detection_recall" in metrics and metrics["bug_detection_recall"] < self.thresholds["bug_detection_recall"]:
            reports.append(self.report_error(
                error_type="LOW_BUG_DETECTION_RECALL",
                severity="HIGH",
                details={
                    "message": "Bug detection recall below threshold",
                    "recall": metrics["bug_detection_recall"],
                    "threshold": self.thresholds["bug_detection_recall"]
                },
                context={
                    "detected_bugs": metrics["detected_bugs"],
                    "total_known_bugs": metrics["total_known_bugs"]
                }
            ))
        
        # Bug fix yield
        if "bug_fix_yield" in metrics and metrics["bug_fix_yield"] < self.thresholds["bug_fix_yield"]:
            reports.append(self.report_error(
                error_type="LOW_BUG_FIX_YIELD",
                severity="HIGH",
                details={
                    "message": "Bug fix yield below threshold",
                    "yield": metrics["bug_fix_yield"],
                    "threshold": self.thresholds["bug_fix_yield"]
                },
                context={
                    "resolved_bugs": metrics["resolved_bugs"],
                    "detected_bugs": metrics["detected_bugs"]
                }
            ))
        
        # Mean time to diagnose
        if "mttd" in metrics and metrics["mttd"] > self.thresholds["mttd"]:
            reports.append(self.report_error(
                error_type="HIGH_MTTD",
                severity="MEDIUM",
                details={
                    "message": "Mean time to diagnose above threshold",
                    "mttd": metrics["mttd"],
                    "threshold": self.thresholds["mttd"]
                },
                context={
                    "diagnose_iterations": metrics["diagnose_iterations"]
                }
            ))
        
        # Mean time to repair
        if "mttr" in metrics and metrics["mttr"] > self.thresholds["mttr"]:
            reports.append(self.report_error(
                error_type="HIGH_MTTR",
                severity="MEDIUM",
                details={
                    "message": "Mean time to repair above threshold",
                    "mttr": metrics["mttr"],
                    "threshold": self.thresholds["mttr"]
                },
                context={
                    "repair_iterations": metrics["repair_iterations"]
                }
            ))
        
        return reports
    
    def _check_energy_convergence_metrics(self) -> List[ErrorReport]:
        """Check energy and convergence metrics against thresholds."""
        reports = []
        metrics = self.session_metrics
        
        # Convergence iterations
        if "convergence_iter_count" in metrics and metrics["convergence_iter_count"] > self.thresholds["convergence_iterations"]:
            reports.append(self.report_error(
                error_type="SLOW_CONVERGENCE",
                severity="MEDIUM",
                details={
                    "message": "Convergence taking too many iterations",
                    "iterations": metrics["convergence_iter_count"],
                    "threshold": self.thresholds["convergence_iterations"]
                },
                context={
                    "energy_current": metrics["energy_current"],
                    "lambda_current": metrics["lambda_current"]
                }
            ))
        
        # Energy reduction
        if "energy_reduction_pct" in metrics and metrics["energy_reduction_pct"] < self.thresholds["energy_reduction_pct"]:
            reports.append(self.report_error(
                error_type="LOW_ENERGY_REDUCTION",
                severity="HIGH",
                details={
                    "message": "Energy reduction percentage below threshold",
                    "reduction": metrics["energy_reduction_pct"],
                    "threshold": self.thresholds["energy_reduction_pct"]
                },
                context={
                    "energy_start": metrics["energy_start"],
                    "energy_current": metrics["energy_current"]
                }
            ))
        
        # Proof coverage delta
        if "proof_coverage_delta" in metrics and metrics["proof_coverage_delta"] < self.thresholds["proof_coverage_delta"]:
            reports.append(self.report_error(
                error_type="LOW_PROOF_COVERAGE_IMPROVEMENT",
                severity="MEDIUM",
                details={
                    "message": "Proof coverage improvement below threshold",
                    "delta": metrics["proof_coverage_delta"],
                    "threshold": self.thresholds["proof_coverage_delta"]
                },
                context={
                    "coverage_start": metrics["proof_coverage_start"],
                    "coverage_current": metrics["proof_coverage_current"]
                }
            ))
        
        # Residual risk improvement
        if "residual_risk_improvement" in metrics and metrics["residual_risk_improvement"] < self.thresholds["residual_risk_improvement"]:
            reports.append(self.report_error(
                error_type="LOW_RISK_IMPROVEMENT",
                severity="HIGH",
                details={
                    "message": "Residual risk improvement below threshold",
                    "improvement": metrics["residual_risk_improvement"],
                    "threshold": self.thresholds["residual_risk_improvement"]
                },
                context={
                    "risk_start": metrics["residual_risk_start"],
                    "risk_current": metrics["residual_risk_current"]
                }
            ))
        
        return reports
    
    def _check_test_quality_metrics(self) -> List[ErrorReport]:
        """Check test and quality metrics against thresholds."""
        reports = []
        metrics = self.session_metrics
        
        # Test pass ratio
        if "test_pass_ratio" in metrics and metrics["test_pass_ratio"] < self.thresholds["test_pass_ratio"]:
            reports.append(self.report_error(
                error_type="LOW_TEST_PASS_RATIO",
                severity="CRITICAL",
                details={
                    "message": "Test pass ratio below threshold",
                    "ratio": metrics["test_pass_ratio"],
                    "threshold": self.thresholds["test_pass_ratio"]
                },
                context={
                    "tests_passed": metrics["tests_passed"],
                    "tests_total": metrics["tests_total"]
                }
            ))
        
        # Regression introduction rate
        if "regression_introduction_rate" in metrics and metrics["regression_introduction_rate"] > self.thresholds["regression_introduction_rate"]:
            reports.append(self.report_error(
                error_type="HIGH_REGRESSION_RATE",
                severity="HIGH",
                details={
                    "message": "Regression introduction rate above threshold",
                    "rate": metrics["regression_introduction_rate"],
                    "threshold": self.thresholds["regression_introduction_rate"]
                },
                context={
                    "new_failures": metrics["new_failures"],
                    "patches_applied": metrics["patches_applied"]
                }
            ))
        
        # Duplicate module ratio
        if "duplicate_module_ratio" in metrics and metrics["duplicate_module_ratio"] > self.thresholds["duplicate_module_ratio"]:
            reports.append(self.report_error(
                error_type="HIGH_CODE_DUPLICATION",
                severity="MEDIUM",
                details={
                    "message": "Duplicate module ratio above threshold",
                    "ratio": metrics["duplicate_module_ratio"],
                    "threshold": self.thresholds["duplicate_module_ratio"]
                },
                context={
                    "redundant_ast_nodes": metrics["redundant_ast_nodes"],
                    "total_ast_nodes": metrics["total_ast_nodes"]
                }
            ))
        
        return reports
    
    def _check_agent_resource_metrics(self) -> List[ErrorReport]:
        """Check agent and resource metrics against thresholds."""
        reports = []
        metrics = self.session_metrics
        
        # Agent coordination overhead
        if "agent_coordination_overhead" in metrics and metrics["agent_coordination_overhead"] > self.thresholds["agent_coordination_overhead"]:
            reports.append(self.report_error(
                error_type="HIGH_AGENT_COORDINATION_OVERHEAD",
                severity="LOW",
                details={
                    "message": "Agent coordination overhead above threshold",
                    "overhead": metrics["agent_coordination_overhead"],
                    "threshold": self.thresholds["agent_coordination_overhead"]
                },
                context={
                    "agent_messages": metrics["agent_messages"],
                    "resolved_bugs": metrics["resolved_bugs"]
                }
            ))
        
        # Token per fix efficiency
        if "token_per_fix_efficiency" in metrics and metrics["token_per_fix_efficiency"] > self.thresholds["token_per_fix_efficiency"]:
            reports.append(self.report_error(
                error_type="LOW_TOKEN_EFFICIENCY",
                severity="MEDIUM",
                details={
                    "message": "Token per fix efficiency below threshold",
                    "tokens_per_fix": metrics["token_per_fix_efficiency"],
                    "threshold": self.thresholds["token_per_fix_efficiency"]
                },
                context={
                    "tokens_used_start": metrics["tokens_used_start"],
                    "tokens_used_current": metrics["tokens_used_current"],
                    "resolved_bugs": metrics["resolved_bugs"]
                }
            ))
        
        # Hallucination rate
        if "hallucination_rate" in metrics and metrics["hallucination_rate"] > self.thresholds["hallucination_rate"]:
            reports.append(self.report_error(
                error_type="HIGH_HALLUCINATION_RATE",
                severity="HIGH",
                details={
                    "message": "Hallucination rate above threshold",
                    "rate": metrics["hallucination_rate"],
                    "threshold": self.thresholds["hallucination_rate"]
                },
                context={
                    "lines_without_obligation": metrics["lines_without_obligation"],
                    "total_generated_lines": metrics["total_generated_lines"]
                }
            ))
        
        return reports
    
    def _check_time_stability_metrics(self) -> List[ErrorReport]:
        """Check time and stability metrics against thresholds."""
        reports = []
        metrics = self.session_metrics
        
        # Certainty gap closure time
        if "certainty_gap_closure_time" in metrics and metrics["certainty_gap_closure_time"] > self.thresholds["certainty_gap_closure_time"]:
            reports.append(self.report_error(
                error_type="SLOW_CERTAINTY_CONVERGENCE",
                severity="MEDIUM",
                details={
                    "message": "Certainty gap closure taking too long",
                    "time_seconds": metrics["certainty_gap_closure_time"],
                    "threshold": self.thresholds["certainty_gap_closure_time"]
                },
                context={
                    "auditor_fail_time": metrics["auditor_fail_time"],
                    "auditor_pass_time": metrics["auditor_pass_time"]
                }
            ))
        
        # Lyapunov descent consistency
        if "lyapunov_descent_consistency" in metrics and metrics["lyapunov_descent_consistency"] < self.thresholds["lyapunov_descent_consistency"]:
            reports.append(self.report_error(
                error_type="LOW_LYAPUNOV_CONSISTENCY",
                severity="HIGH",
                details={
                    "message": "Lyapunov descent consistency below threshold",
                    "consistency": metrics["lyapunov_descent_consistency"],
                    "threshold": self.thresholds["lyapunov_descent_consistency"]
                },
                context={
                    "decreasing_iterations": metrics["lyapunov_decreasing_iterations"],
                    "total_iterations": metrics["total_iterations"]
                }
            ))
        
        # Meta guard breach count
        if metrics["meta_guard_breaches"] > self.thresholds["meta_guard_breach_count"]:
            reports.append(self.report_error(
                error_type="EXCESSIVE_META_GUARD_BREACHES",
                severity="CRITICAL",
                details={
                    "message": "Meta guard breach count above threshold",
                    "count": metrics["meta_guard_breaches"],
                    "threshold": self.thresholds["meta_guard_breach_count"]
                },
                context={
                    "session_duration": metrics["duration_seconds"]
                }
            ))
        
        return reports
    
    def _check_resource_documentation_metrics(self) -> List[ErrorReport]:
        """Check resource and documentation metrics against thresholds."""
        reports = []
        metrics = self.session_metrics
        
        # Resource footprint change
        if "resource_footprint_change" in metrics and metrics["resource_footprint_change"] > self.thresholds["resource_footprint_change"]:
            reports.append(self.report_error(
                error_type="HIGH_RESOURCE_GROWTH",
                severity="MEDIUM",
                details={
                    "message": "Resource footprint growth above threshold",
                    "change": metrics["resource_footprint_change"],
                    "threshold": self.thresholds["resource_footprint_change"]
                },
                context={
                    "memory_start": metrics["memory_usage_start_mb"],
                    "memory_current": metrics["memory_usage_current_mb"],
                    "cpu_start": metrics["cpu_usage_start_pct"],
                    "cpu_current": metrics["cpu_usage_current_pct"]
                }
            ))
        
        # Documentation completeness delta
        if "documentation_completeness_delta" in metrics and metrics["documentation_completeness_delta"] < self.thresholds["documentation_completeness_delta"]:
            reports.append(self.report_error(
                error_type="DECLINING_DOCUMENTATION",
                severity="LOW",
                details={
                    "message": "Documentation completeness declining",
                    "delta": metrics["documentation_completeness_delta"],
                    "threshold": self.thresholds["documentation_completeness_delta"]
                },
                context={
                    "doc_ratio_start": metrics["doc_lines_start"] / metrics["code_lines_start"],
                    "doc_ratio_current": metrics["doc_lines_current"] / metrics["code_lines_current"]
                }
            ))
        
        # Aggregate confidence score
        if "aggregate_confidence_score" in metrics and metrics["aggregate_confidence_score"] < self.thresholds["aggregate_confidence_score"]:
            reports.append(self.report_error(
                error_type="LOW_CONFIDENCE_SCORE",
                severity="HIGH",
                details={
                    "message": "Aggregate confidence score below threshold",
                    "score": metrics["aggregate_confidence_score"],
                    "threshold": self.thresholds["aggregate_confidence_score"]
                },
                context={
                    "session_id": metrics["session_id"],
                    "duration_seconds": metrics["duration_seconds"]
                }
            ))
        
        return reports
    
    def _update_history(self) -> None:
        """Update the metric history with current values."""
        for metric in self._get_all_metric_names():
            if metric in self.session_metrics:
                self.history[metric].append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "value": self.session_metrics[metric]
                })
    
    def _get_threshold(self, name: str, default: Any) -> Any:
        """Get a threshold value from config or use default."""
        if self.config and "thresholds" in self.config and name in self.config["thresholds"]:
            return self.config["thresholds"][name]
        return default
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of the current metrics."""
        return {
            "session_id": self.session_metrics["session_id"],
            "duration_seconds": self.session_metrics["duration_seconds"],
            "bug_detection_recall": self.session_metrics.get("bug_detection_recall", 0),
            "bug_fix_yield": self.session_metrics.get("bug_fix_yield", 0),
            "test_pass_ratio": self.session_metrics.get("test_pass_ratio", 0),
            "energy_reduction_pct": self.session_metrics.get("energy_reduction_pct", 0),
            "mttr": self.session_metrics.get("mttr", 0),
            "lyapunov_descent_consistency": self.session_metrics.get("lyapunov_descent_consistency", 0),
            "aggregate_confidence_score": self.session_metrics.get("aggregate_confidence_score", 0)
        }
    
    def get_lyapunov_table(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the Lyapunov history table with all metrics."""
        return self.history
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the sensor."""
        return {
            "sensor_id": self.sensor_id,
            "component_name": self.component_name,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "session_id": self.session_metrics["session_id"],
            "metrics_count": len(self._get_all_metric_names()),
            "metrics_summary": self.get_metrics_summary()
        }
