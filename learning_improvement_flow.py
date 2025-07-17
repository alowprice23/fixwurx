#!/usr/bin/env python3
"""
learning_improvement_flow.py
────────────────────────────
Implements the learning and improvement flow for the FixWurx system.

This module provides the core flow for learning from fix results and improving
the system's performance over time. It analyzes verification results, updates
neural weights, identifies patterns, and enhances the system's ability to
detect and fix bugs more effectively in the future.
"""

import os
import sys
import json
import logging
import time
import uuid
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta

# Import core components
from triangulation_engine import TriangulationEngine
from neural_matrix_core import NeuralMatrix
from meta_agent import MetaAgent
from storage_manager import StorageManager
from neural_matrix.core.validation import NeuralMatrixValidator
from optimization.mttr_optimizer import MTTROptimizer
from learning.pattern_recognition import PatternRecognizer
from learning.weight_adjustment import WeightAdjuster
from learning.success_tracking import SuccessTracker

# Configure logging
logger = logging.getLogger("LearningImprovementFlow")

class LearningImprovementFlow:
    """
    Implements the learning and improvement flow for the FixWurx system.
    
    This class orchestrates the entire learning and improvement process, from
    analyzing verification results to updating neural weights and identifying
    patterns to enhance the system's performance over time.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the learning and improvement flow.
        
        Args:
            config: Configuration for the learning and improvement flow.
        """
        self.config = config or {}
        self.triangulation_engine = TriangulationEngine()
        self.neural_matrix = NeuralMatrix()
        self.meta_agent = MetaAgent()
        self.storage_manager = StorageManager()
        self.neural_validator = NeuralMatrixValidator()
        self.mttr_optimizer = MTTROptimizer()
        self.pattern_recognizer = PatternRecognizer()
        self.weight_adjuster = WeightAdjuster()
        self.success_tracker = SuccessTracker()
        
        # Initialize state
        self.current_learning_id = None
        self.current_context = {}
        self.learning_sessions = []
        
        # Configure learning settings
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.pattern_threshold = self.config.get("pattern_threshold", 0.7)
        self.weight_decay = self.config.get("weight_decay", 0.001)
        self.backpropagation_epochs = self.config.get("backpropagation_epochs", 5)
        
        # Initialize metrics tracking
        self.metrics = {
            "success_rate_history": [],
            "mttr_history": [],
            "coverage_history": [],
            "pattern_discovery_history": []
        }
        
        logger.info("Learning Improvement Flow initialized")
    
    def start_learning(self, 
                      verification_id: Optional[str] = None, 
                      learning_options: Dict[str, Any] = None) -> str:
        """
        Start the learning and improvement process based on verification results.
        
        Args:
            verification_id: ID of the verification to learn from, or None to learn from all recent verifications.
            learning_options: Options for the learning process.
            
        Returns:
            Learning ID for the learning process.
        """
        learning_options = learning_options or {}
        
        # Generate a learning ID
        timestamp = int(time.time())
        learning_id = f"learn_{timestamp}_{str(uuid.uuid4())[:8]}"
        self.current_learning_id = learning_id
        
        # Initialize learning context
        self.current_context = {
            "learning_id": learning_id,
            "verification_id": verification_id,
            "start_time": timestamp,
            "options": learning_options,
            "status": "started",
            "metrics_collected": 0,
            "patterns_discovered": 0,
            "weight_updates": 0
        }
        
        logger.info(f"Starting learning and improvement process {learning_id}")
        
        # Trigger the learning flow
        if verification_id:
            # Learn from a specific verification
            verification = self.storage_manager.get_verification(verification_id)
            if not verification:
                raise ValueError(f"Verification with ID {verification_id} not found")
            self._execute_learning_flow([verification], learning_options)
        else:
            # Learn from all recent verifications
            time_window = learning_options.get("time_window", 7 * 24 * 60 * 60)  # Default: 7 days
            recent_verifications = self._get_recent_verifications(time_window)
            self._execute_learning_flow(recent_verifications, learning_options)
        
        return learning_id
    
    def _get_recent_verifications(self, time_window: int) -> List[Dict[str, Any]]:
        """
        Get recent verifications within the specified time window.
        
        Args:
            time_window: Time window in seconds.
            
        Returns:
            List of recent verifications.
        """
        # Get current time
        current_time = int(time.time())
        cutoff_time = current_time - time_window
        
        # Get all verifications
        all_verifications = self.storage_manager.get_all_verifications()
        
        # Filter by time
        recent_verifications = [
            v for v in all_verifications 
            if v.get("timestamp", 0) >= cutoff_time
        ]
        
        logger.info(f"Found {len(recent_verifications)} recent verifications within the time window")
        
        return recent_verifications
    
    def _execute_learning_flow(self, 
                             verifications: List[Dict[str, Any]], 
                             learning_options: Dict[str, Any]) -> None:
        """
        Execute the learning and improvement flow.
        
        Args:
            verifications: List of verifications to learn from.
            learning_options: Options for the learning process.
        """
        try:
            # Phase 1: Collect metrics
            logger.info("Phase 1: Collect metrics")
            metrics = self._collect_metrics(verifications)
            
            # Phase 2: Identify patterns
            logger.info("Phase 2: Identify patterns")
            patterns = self._identify_patterns(verifications, metrics)
            
            # Phase 3: Update neural weights
            logger.info("Phase 3: Update neural weights")
            weight_updates = self._update_neural_weights(verifications, patterns)
            
            # Phase 4: Optimize MTTR
            logger.info("Phase 4: Optimize MTTR")
            mttr_optimizations = self._optimize_mttr(metrics)
            
            # Phase 5: Generate learning report
            logger.info("Phase 5: Generate learning report")
            learning_report = self._generate_learning_report(
                metrics, patterns, weight_updates, mttr_optimizations
            )
            
            # Update context
            self.current_context["status"] = "completed"
            self.current_context["end_time"] = int(time.time())
            self.current_context["metrics_collected"] = len(metrics.get("success_rates", []))
            self.current_context["patterns_discovered"] = len(patterns)
            self.current_context["weight_updates"] = len(weight_updates)
            self.current_context["metrics"] = metrics
            self.current_context["patterns"] = patterns
            self.current_context["weight_updates"] = weight_updates
            self.current_context["mttr_optimizations"] = mttr_optimizations
            self.current_context["learning_report"] = learning_report
            
            # Store learning results
            learning_result = {
                "learning_id": self.current_learning_id,
                "timestamp": int(time.time()),
                "metrics": metrics,
                "patterns": patterns,
                "weight_updates": weight_updates,
                "mttr_optimizations": mttr_optimizations,
                "learning_report": learning_report
            }
            self.learning_sessions.append(learning_result)
            
            # Store learning in storage manager
            self.storage_manager.store_learning(learning_result)
            
            # Notify the Meta Agent
            self.meta_agent.notify_learning_complete(learning_result)
            
            # Update historical metrics
            self._update_historical_metrics(metrics)
            
            logger.info(f"Learning and improvement process {self.current_learning_id} completed with {len(patterns)} patterns discovered and {len(weight_updates)} weight updates")
            
        except Exception as e:
            logger.error(f"Error in learning and improvement flow: {e}")
            self.current_context["status"] = "failed"
            self.current_context["error"] = str(e)
            self.current_context["end_time"] = int(time.time())
            raise
    
    def _collect_metrics(self, verifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collect metrics from verifications.
        
        Args:
            verifications: List of verifications to collect metrics from.
            
        Returns:
            Collected metrics.
        """
        # Initialize metrics
        metrics = {
            "success_rates": [],
            "failure_rates": [],
            "mttrs": [],
            "code_coverage": [],
            "regression_rates": [],
            "bug_types": {},
            "fix_strategies": {},
            "verification_times": []
        }
        
        # Process each verification
        for verification in verifications:
            # Get implementation and plan associated with this verification
            implementation_id = verification.get("implementation_id")
            implementation = self.storage_manager.get_implementation(implementation_id) if implementation_id else None
            
            plan_id = implementation.get("plan_id") if implementation else None
            plan = self.storage_manager.get_plan(plan_id) if plan_id else None
            
            # Calculate success rate
            statistics = verification.get("verification_report", {}).get("statistics", {})
            total_tests = statistics.get("total_tests", 0)
            passed_tests = statistics.get("passed_tests", 0)
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            # Calculate failure rate
            failed_tests = statistics.get("failed_tests", 0) + statistics.get("error_tests", 0)
            failure_rate = failed_tests / total_tests if total_tests > 0 else 0
            
            # Calculate MTTR (Mean Time To Resolution)
            mttr = self._calculate_mttr(implementation, plan) if implementation and plan else None
            
            # Calculate code coverage
            avg_coverage = statistics.get("average_coverage", 0)
            
            # Calculate regression rate
            has_regressions = verification.get("verification_report", {}).get("has_regressions", False)
            regression_count = verification.get("verification_report", {}).get("regression_count", 0)
            regression_rate = regression_count / total_tests if total_tests > 0 else 0
            
            # Collect bug types
            if plan:
                for path in plan.get("paths", []):
                    bug_id = path.get("bug_id")
                    bug_type = path.get("bug_type", "unknown")
                    
                    if bug_type not in metrics["bug_types"]:
                        metrics["bug_types"][bug_type] = 0
                    
                    metrics["bug_types"][bug_type] += 1
            
            # Collect fix strategies
            if implementation:
                for result in implementation.get("results", []):
                    strategy = result.get("strategy", "unknown")
                    
                    if strategy not in metrics["fix_strategies"]:
                        metrics["fix_strategies"][strategy] = 0
                    
                    metrics["fix_strategies"][strategy] += 1
            
            # Calculate verification time
            start_time = verification.get("start_time", 0)
            end_time = verification.get("end_time", 0)
            verification_time = end_time - start_time if end_time > start_time else 0
            
            # Add metrics to collections
            metrics["success_rates"].append(success_rate)
            metrics["failure_rates"].append(failure_rate)
            if mttr is not None:
                metrics["mttrs"].append(mttr)
            metrics["code_coverage"].append(avg_coverage)
            metrics["regression_rates"].append(regression_rate)
            metrics["verification_times"].append(verification_time)
        
        # Calculate aggregate metrics
        metrics["avg_success_rate"] = sum(metrics["success_rates"]) / len(metrics["success_rates"]) if metrics["success_rates"] else 0
        metrics["avg_failure_rate"] = sum(metrics["failure_rates"]) / len(metrics["failure_rates"]) if metrics["failure_rates"] else 0
        metrics["avg_mttr"] = sum(metrics["mttrs"]) / len(metrics["mttrs"]) if metrics["mttrs"] else 0
        metrics["avg_code_coverage"] = sum(metrics["code_coverage"]) / len(metrics["code_coverage"]) if metrics["code_coverage"] else 0
        metrics["avg_regression_rate"] = sum(metrics["regression_rates"]) / len(metrics["regression_rates"]) if metrics["regression_rates"] else 0
        metrics["avg_verification_time"] = sum(metrics["verification_times"]) / len(metrics["verification_times"]) if metrics["verification_times"] else 0
        
        # Sort bug types and fix strategies by frequency
        metrics["bug_types"] = dict(sorted(metrics["bug_types"].items(), key=lambda x: x[1], reverse=True))
        metrics["fix_strategies"] = dict(sorted(metrics["fix_strategies"].items(), key=lambda x: x[1], reverse=True))
        
        return metrics
    
    def _calculate_mttr(self, implementation: Dict[str, Any], plan: Dict[str, Any]) -> float:
        """
        Calculate Mean Time To Resolution (MTTR) for a fix.
        
        Args:
            implementation: Implementation data.
            plan: Solution plan data.
            
        Returns:
            MTTR in seconds.
        """
        # Get plan creation time
        plan_creation_time = plan.get("created_at", 0)
        
        # Get implementation completion time
        implementation_completion_time = implementation.get("end_time", 0)
        
        # Calculate MTTR
        mttr = implementation_completion_time - plan_creation_time
        
        return max(0, mttr)  # Ensure non-negative
    
    def _identify_patterns(self, 
                          verifications: List[Dict[str, Any]], 
                          metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify patterns in verification results.
        
        Args:
            verifications: List of verifications to analyze.
            metrics: Collected metrics.
            
        Returns:
            List of identified patterns.
        """
        patterns = []
        
        # Get pattern recognition threshold
        threshold = self.config.get("pattern_threshold", self.pattern_threshold)
        
        # Use pattern recognizer to find patterns
        raw_patterns = self.pattern_recognizer.find_patterns(verifications, threshold)
        
        # Process and enrich patterns
        for pattern in raw_patterns:
            # Calculate confidence
            confidence = pattern.get("confidence", 0)
            
            # Only include patterns with confidence above threshold
            if confidence >= threshold:
                # Enrich pattern with additional information
                enriched_pattern = self._enrich_pattern(pattern, verifications, metrics)
                patterns.append(enriched_pattern)
        
        # Sort patterns by confidence
        patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        
        return patterns
    
    def _enrich_pattern(self, 
                       pattern: Dict[str, Any], 
                       verifications: List[Dict[str, Any]], 
                       metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a pattern with additional information.
        
        Args:
            pattern: Pattern to enrich.
            verifications: List of verifications.
            metrics: Collected metrics.
            
        Returns:
            Enriched pattern.
        """
        # Get pattern type
        pattern_type = pattern.get("type")
        
        # Enrich based on pattern type
        if pattern_type == "bug_fix_correlation":
            # Correlate bug types with successful fix strategies
            bug_type = pattern.get("bug_type")
            fix_strategy = pattern.get("fix_strategy")
            
            # Count occurrences and success rate
            total_occurrences = 0
            successful_occurrences = 0
            
            for verification in verifications:
                implementation_id = verification.get("implementation_id")
                implementation = self.storage_manager.get_implementation(implementation_id) if implementation_id else None
                
                if implementation:
                    plan_id = implementation.get("plan_id")
                    plan = self.storage_manager.get_plan(plan_id) if plan_id else None
                    
                    if plan:
                        # Check if this plan contains the bug type
                        for path in plan.get("paths", []):
                            if path.get("bug_type") == bug_type:
                                # Check if this implementation used the fix strategy
                                for result in implementation.get("results", []):
                                    if result.get("strategy") == fix_strategy:
                                        total_occurrences += 1
                                        
                                        # Check if verification passed
                                        if verification.get("verification_report", {}).get("status") == "passed":
                                            successful_occurrences += 1
            
            # Calculate success rate
            success_rate = successful_occurrences / total_occurrences if total_occurrences > 0 else 0
            
            # Add to pattern
            pattern["total_occurrences"] = total_occurrences
            pattern["successful_occurrences"] = successful_occurrences
            pattern["success_rate"] = success_rate
            
        elif pattern_type == "regression_predictor":
            # Identify factors that predict regressions
            factor = pattern.get("factor")
            threshold = pattern.get("threshold")
            
            # Count occurrences and regression rate
            total_occurrences = 0
            regression_occurrences = 0
            
            for verification in verifications:
                # Check if factor is present and above threshold
                factor_value = self._extract_factor_value(verification, factor)
                
                if factor_value is not None and factor_value >= threshold:
                    total_occurrences += 1
                    
                    # Check if verification had regressions
                    if verification.get("verification_report", {}).get("has_regressions", False):
                        regression_occurrences += 1
            
            # Calculate regression rate
            regression_rate = regression_occurrences / total_occurrences if total_occurrences > 0 else 0
            
            # Add to pattern
            pattern["total_occurrences"] = total_occurrences
            pattern["regression_occurrences"] = regression_occurrences
            pattern["regression_rate"] = regression_rate
            
        elif pattern_type == "coverage_impact":
            # Correlate code coverage with fix success
            coverage_range = pattern.get("coverage_range", [0, 0])
            
            # Count occurrences and success rate
            total_occurrences = 0
            successful_occurrences = 0
            
            for verification in verifications:
                # Get coverage
                avg_coverage = verification.get("verification_report", {}).get("statistics", {}).get("average_coverage", 0)
                
                if coverage_range[0] <= avg_coverage <= coverage_range[1]:
                    total_occurrences += 1
                    
                    # Check if verification passed
                    if verification.get("verification_report", {}).get("status") == "passed":
                        successful_occurrences += 1
            
            # Calculate success rate
            success_rate = successful_occurrences / total_occurrences if total_occurrences > 0 else 0
            
            # Add to pattern
            pattern["total_occurrences"] = total_occurrences
            pattern["successful_occurrences"] = successful_occurrences
            pattern["success_rate"] = success_rate
        
        # Add general metadata
        pattern["discovery_time"] = int(time.time())
        pattern["learning_id"] = self.current_learning_id
        
        return pattern
    
    def _extract_factor_value(self, verification: Dict[str, Any], factor: str) -> Optional[float]:
        """
        Extract a factor value from a verification.
        
        Args:
            verification: Verification to extract from.
            factor: Factor to extract.
            
        Returns:
            Factor value or None if not found.
        """
        # Handle different factor types
        if factor == "code_complexity":
            # Extract from quality results
            quality_results = verification.get("quality_results", {})
            complexity_results = quality_results.get("complexity_results", [])
            
            if complexity_results:
                # Calculate average complexity
                total_complexity = 0
                count = 0
                
                for result in complexity_results:
                    avg_complexity = result.get("average_complexity", 0)
                    if avg_complexity > 0:
                        total_complexity += avg_complexity
                        count += 1
                
                return total_complexity / count if count > 0 else None
            
        elif factor == "test_coverage":
            # Extract from verification report
            return verification.get("verification_report", {}).get("statistics", {}).get("average_coverage", None)
            
        elif factor == "code_churn":
            # Extract from implementation
            implementation_id = verification.get("implementation_id")
            implementation = self.storage_manager.get_implementation(implementation_id) if implementation_id else None
            
            if implementation:
                # Calculate code churn
                total_lines_changed = 0
                
                for result in implementation.get("results", []):
                    for change in result.get("changes", []):
                        lines_changed = change.get("lines_changed", 0)
                        total_lines_changed += lines_changed
                
                return total_lines_changed
        
        return None
    
    def _update_neural_weights(self, 
                              verifications: List[Dict[str, Any]], 
                              patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update neural weights based on verification results and patterns.
        
        Args:
            verifications: List of verifications.
            patterns: List of identified patterns.
            
        Returns:
            List of weight updates.
        """
        # Initialize weight updates
        weight_updates = []
        
        # Process each verification for direct feedback
        for verification in verifications:
            # Extract success/failure information
            status = verification.get("verification_report", {}).get("status")
            success = status == "passed"
            partial = status == "partial"
            failure = status == "failed"
            
            # Get implementation and plan
            implementation_id = verification.get("implementation_id")
            implementation = self.storage_manager.get_implementation(implementation_id) if implementation_id else None
            
            plan_id = implementation.get("plan_id") if implementation else None
            plan = self.storage_manager.get_plan(plan_id) if plan_id else None
            
            if plan:
                # Update weights for each path in the plan
                for path in plan.get("paths", []):
                    path_id = path.get("id")
                    
                    # Find corresponding implementation result
                    result = None
                    if implementation:
                        for r in implementation.get("results", []):
                            if r.get("path_id") == path_id:
                                result = r
                                break
                    
                    # Only update weights if we have a result
                    if result:
                        result_status = result.get("status")
                        
                        # Create feedback data
                        feedback = {
                            "path_id": path_id,
                            "bug_id": path.get("bug_id"),
                            "bug_type": path.get("bug_type", "unknown"),
                            "path_success": result_status == "success",
                            "verification_success": success,
                            "verification_partial": partial,
                            "verification_failure": failure,
                            "neural_insights": path.get("neural_insights", {})
                        }
                        
                        # Apply feedback to neural matrix
                        update = self.weight_adjuster.apply_feedback(feedback)
                        
                        if update:
                            weight_updates.append(update)
        
        # Process patterns for indirect feedback
        for pattern in patterns:
            # Convert pattern to feedback
            feedback = self._pattern_to_feedback(pattern)
            
            if feedback:
                # Apply feedback to neural matrix
                update = self.weight_adjuster.apply_pattern_feedback(feedback)
                
                if update:
                    weight_updates.append(update)
        
        # Perform backpropagation
        if weight_updates:
            for _ in range(self.backpropagation_epochs):
                backprop_updates = self.neural_matrix.backpropagate(weight_updates, self.learning_rate, self.weight_decay)
                
                if backprop_updates:
                    weight_updates.extend(backprop_updates)
        
        # Validate updated neural matrix
        validation_result = self.neural_validator.validate_matrix()
        
        if not validation_result.get("valid", False):
            logger.warning(f"Neural matrix validation failed: {validation_result.get('reason')}")
            
            # Rollback changes if validation fails
            self.neural_matrix.rollback_changes()
            
            # Add rollback info to weight updates
            for update in weight_updates:
                update["rolled_back"] = True
            
            weight_updates.append({
                "type": "rollback",
                "reason": validation_result.get("reason"),
                "timestamp": int(time.time())
            })
        else:
            # Commit changes
            self.neural_matrix.commit_changes()
            
            # Update success tracking
            self.success_tracker.update_success_rates(weight_updates)
        
        return weight_updates
    
    def _pattern_to_feedback(self, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert a pattern to neural feedback.
        
        Args:
            pattern: Pattern to convert.
            
        Returns:
            Feedback data or None if not applicable.
        """
        # Get pattern type
        pattern_type = pattern.get("type")
        
        # Convert based on pattern type
        if pattern_type == "bug_fix_correlation":
            return {
                "type": "correlation",
                "bug_type": pattern.get("bug_type"),
                "fix_strategy": pattern.get("fix_strategy"),
                "success_rate": pattern.get("success_rate", 0),
                "confidence": pattern.get("confidence", 0)
            }
        elif pattern_type == "regression_predictor":
            return {
                "type": "predictor",
                "factor": pattern.get("factor"),
                "threshold": pattern.get("threshold"),
                "regression_rate": pattern.get("regression_rate", 0),
                "confidence": pattern.get("confidence", 0)
            }
        elif pattern_type == "coverage_impact":
            return {
                "type": "impact",
                "factor": "coverage",
                "range": pattern.get("coverage_range", [0, 0]),
                "success_rate": pattern.get("success_rate", 0),
                "confidence": pattern.get("confidence", 0)
            }
        
        return None
    
    def _optimize_mttr(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize Mean Time To Resolution (MTTR).
        
        Args:
            metrics: Collected metrics.
            
        Returns:
            MTTR optimization results.
        """
        # Get MTTR data
        mttrs = metrics.get("mttrs", [])
        
        if not mttrs:
            return {
                "optimized": False,
                "reason": "No MTTR data available"
            }
        
        # Calculate current average MTTR
        avg_mttr = sum(mttrs) / len(mttrs)
        
        # Get historical MTTR data
        historical_mttrs = self.metrics.get("mttr_history", [])
        
        # Perform optimization
        optimization_result = self.mttr_optimizer.optimize(
            mttrs, historical_mttrs, metrics.get("bug_types", {}), metrics.get("fix_strategies", {})
        )
        
        # Apply optimizations
        if optimization_result.get("optimizations"):
            for optimization in optimization_result.get("optimizations", []):
                optimization_type = optimization.get("type")
                
                if optimization_type == "path_selection":
                    # Update path selection weights in neural matrix
                    selection_weights = optimization.get("weights", {})
                    self.neural_matrix.update_path_selection_weights(selection_weights)
                
                elif optimization_type == "resource_allocation":
                    # Update resource allocation in resource manager
                    allocations = optimization.get("allocations", {})
                    self.resource_manager.update_allocations(allocations)
                
                elif optimization_type == "parallelization":
                    # Update parallelization settings
                    parallel_settings = optimization.get("settings", {})
                    self.triangulation_engine.update_parallelization(parallel_settings)
        
        # Calculate projected MTTR
        projected_mttr = optimization_result.get("projected_mttr", avg_mttr)
        
        # Calculate improvement
        improvement = (avg_mttr - projected_mttr) / avg_mttr if avg_mttr > 0 else 0
        
        # Create result
        result = {
            "optimized": True,
            "current_mttr": avg_mttr,
            "projected_mttr": projected_mttr,
            "improvement": improvement,
            "optimizations": optimization_result.get("optimizations", [])
        }
        
        return result
    
    def _generate_learning_report(self, 
                                metrics: Dict[str, Any], 
                                patterns: List[Dict[str, Any]], 
                                weight_updates: List[Dict[str, Any]], 
                                mttr_optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a learning report.
        
        Args:
            metrics: Collected metrics.
            patterns: Identified patterns.
            weight_updates: Neural weight updates.
            mttr_optimizations: MTTR optimization results.
            
        Returns:
            Learning report.
        """
        # Calculate system improvements
        improvements = {
            "success_rate": self._calculate_improvement("success_rate_history", metrics.get("avg_success_rate", 0)),
            "mttr": self._calculate_improvement("mttr_history", metrics.get("avg_mttr", 0), lower_is_better=True),
            "code_coverage": self._calculate_improvement("coverage_history", metrics.get("avg_code_coverage", 0)),
            "pattern_discovery": self._calculate_improvement("pattern_discovery_history", len(patterns))
        }
        
        # Generate top patterns
        top_patterns = patterns[:min(5, len(patterns))] if patterns else []
        
        # Generate top weight updates
        top_weight_updates = sorted(
            [w for w in weight_updates if not w.get("rolled_back", False)],
            key=lambda w: abs(w.get("delta", 0)),
            reverse=True
        )[:min(5, len(weight_updates))]
        
        # Generate MTTR optimization summary
        mttr_optimization_summary = {
            "optimized": mttr_optimizations.get("optimized", False),
            "current_mttr": mttr_optimizations.get("current_mttr", 0),
            "projected_mttr": mttr_optimizations.get("projected_mttr", 0),
            "improvement": mttr_optimizations.get("improvement", 0),
            "optimization_count": len(mttr_optimizations.get("optimizations", []))
        }
        
        # Create learning report
        learning_report = {
            "learning_id": self.current_learning_id,
            "timestamp": int(time.time()),
            "metrics_summary": {
                "success_rate": metrics.get("avg_success_rate", 0),
                "failure_rate": metrics.get("avg_failure_rate", 0),
                "mttr": metrics.get("avg_mttr", 0),
                "code_coverage": metrics.get("avg_code_coverage", 0),
                "regression_rate": metrics.get("avg_regression_rate", 0),
                "verification_time": metrics.get("avg_verification_time", 0)
            },
            "improvements": improvements,
            "top_patterns": top_patterns,
            "top_weight_updates": top_weight_updates,
            "mttr_optimization": mttr_optimization_summary,
            "recommendations": self._generate_recommendations(
                metrics, patterns, weight_updates, mttr_optimizations, improvements
            )
        }
        
        return learning_report
    
    def _calculate_improvement(self, 
                             metric_key: str, 
                             current_value: float, 
                             lower_is_better: bool = False) -> float:
        """
        Calculate improvement for a metric compared to historical values.
        
        Args:
            metric_key: Key for the metric in historical data.
            current_value: Current value of the metric.
            lower_is_better: Whether lower values are better (e.g., for MTTR).
            
        Returns:
            Improvement as a fraction (negative means degradation).
        """
        # Get historical values
        historical_values = self.metrics.get(metric_key, [])
        
        if not historical_values:
            return 0.0  # No historical data for comparison
        
        # Calculate average of historical values
        avg_historical = sum(historical_values) / len(historical_values)
        
        # Calculate improvement
        if avg_historical == 0:
            return 0.0  # Avoid division by zero
        
        if lower_is_better:
            # For metrics where lower values are better (e.g., MTTR)
            improvement = (avg_historical - current_value) / avg_historical
        else:
            # For metrics where higher values are better (e.g., success rate)
            improvement = (current_value - avg_historical) / avg_historical
        
        return improvement
    
    def _update_historical_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update historical metrics with new values.
        
        Args:
            metrics: New metrics to add to history.
        """
        # Limit history length
        max_history_length = self.config.get("max_history_length", 100)
        
        # Update success rate history
        self.metrics["success_rate_history"].append(metrics.get("avg_success_rate", 0))
        if len(self.metrics["success_rate_history"]) > max_history_length:
            self.metrics["success_rate_history"] = self.metrics["success_rate_history"][-max_history_length:]
        
        # Update MTTR history
        self.metrics["mttr_history"].append(metrics.get("avg_mttr", 0))
        if len(self.metrics["mttr_history"]) > max_history_length:
            self.metrics["mttr_history"] = self.metrics["mttr_history"][-max_history_length:]
        
        # Update coverage history
        self.metrics["coverage_history"].append(metrics.get("avg_code_coverage", 0))
        if len(self.metrics["coverage_history"]) > max_history_length:
            self.metrics["coverage_history"] = self.metrics["coverage_history"][-max_history_length:]
        
        # Update pattern discovery history
        self.metrics["pattern_discovery_history"].append(len(metrics.get("patterns", [])))
        if len(self.metrics["pattern_discovery_history"]) > max_history_length:
            self.metrics["pattern_discovery_history"] = self.metrics["pattern_discovery_history"][-max_history_length:]
    
    def _generate_recommendations(self, 
                                metrics: Dict[str, Any], 
                                patterns: List[Dict[str, Any]], 
                                weight_updates: List[Dict[str, Any]], 
                                mttr_optimizations: Dict[str, Any], 
                                improvements: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on learning results.
        
        Args:
            metrics: Collected metrics.
            patterns: Identified patterns.
            weight_updates: Neural weight updates.
            mttr_optimizations: MTTR optimization results.
            improvements: Calculated improvements.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        
        # Add recommendations based on patterns
        for pattern in patterns:
            pattern_type = pattern.get("type")
            
            if pattern_type == "bug_fix_correlation":
                bug_type = pattern.get("bug_type")
                fix_strategy = pattern.get("fix_strategy")
                success_rate = pattern.get("success_rate", 0)
                
                if success_rate > 0.8:
                    recommendations.append({
                        "type": "strategy",
                        "priority": "high",
                        "title": f"Prefer {fix_strategy} for {bug_type} bugs",
                        "description": f"This strategy has a {success_rate:.1%} success rate for this bug type.",
                        "confidence": pattern.get("confidence", 0)
                    })
            elif pattern_type == "regression_predictor":
                factor = pattern.get("factor")
                threshold = pattern.get("threshold")
                regression_rate = pattern.get("regression_rate", 0)
                
                if regression_rate > 0.6:
                    recommendations.append({
                        "type": "caution",
                        "priority": "high",
                        "title": f"Be cautious with changes where {factor} > {threshold}",
                        "description": f"Changes with this characteristic have a {regression_rate:.1%} regression rate.",
                        "confidence": pattern.get("confidence", 0)
                    })
            elif pattern_type == "coverage_impact":
                coverage_range = pattern.get("coverage_range", [0, 0])
                success_rate = pattern.get("success_rate", 0)
                
                if success_rate > 0.8:
                    recommendations.append({
                        "type": "best_practice",
                        "priority": "medium",
                        "title": f"Aim for test coverage between {coverage_range[0]:.1%} and {coverage_range[1]:.1%}",
                        "description": f"This coverage range has a {success_rate:.1%} success rate for fix verification.",
                        "confidence": pattern.get("confidence", 0)
                    })
        
        # Add recommendations based on metrics
        if metrics.get("avg_code_coverage", 0) < 0.5:
            recommendations.append({
                "type": "improvement",
                "priority": "medium",
                "title": "Improve test coverage",
                "description": "Current test coverage is below 50%. Increasing test coverage can help detect bugs earlier and improve fix verification.",
                "confidence": 0.9
            })
        
        if metrics.get("avg_regression_rate", 0) > 0.2:
            recommendations.append({
                "type": "improvement",
                "priority": "high",
                "title": "Focus on reducing regressions",
                "description": "The current regression rate is over 20%. Improving test coverage and implementing more thorough verification can help reduce regressions.",
                "confidence": 0.9
            })
        
        # Add recommendations based on MTTR optimizations
        if mttr_optimizations.get("optimized", False) and mttr_optimizations.get("improvement", 0) > 0.1:
            recommendations.append({
                "type": "optimization",
                "priority": "medium",
                "title": "Apply MTTR optimizations",
                "description": f"MTTR optimizations can reduce resolution time by {mttr_optimizations.get('improvement', 0):.1%}.",
                "confidence": 0.8
            })
        
        # Add recommendations based on improvements
        for metric, improvement in improvements.items():
            if improvement < -0.1:  # Degradation
                recommendations.append({
                    "type": "alert",
                    "priority": "high",
                    "title": f"{metric.replace('_', ' ').title()} is degrading",
                    "description": f"There has been a {-improvement:.1%} degradation in {metric.replace('_', ' ')}. Review recent changes to identify the cause.",
                    "confidence": 0.8
                })
        
        # Sort recommendations by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 3))
        
        return recommendations
    
    def get_learning_by_id(self, learning_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a learning session by its ID.
        
        Args:
            learning_id: Learning ID.
            
        Returns:
            Learning session data or None if not found.
        """
        if learning_id == self.current_learning_id and self.current_context:
            return self.current_context
        
        # Try to find in previous learning sessions
        for session in self.learning_sessions:
            if session.get("learning_id") == learning_id:
                return session
        
        # Try to retrieve from storage
        return self.storage_manager.get_learning(learning_id)
    
    def get_learning_status(self) -> Dict[str, Any]:
        """
        Get the status of the current learning process.
        
        Returns:
            Learning status data.
        """
        return self.current_context
    
    def save_learning_report(self, 
                           output_path: Optional[str] = None, 
                           format: str = "json") -> str:
        """
        Save the learning report to a file.
        
        Args:
            output_path: Path to save the report. If None, a default path is used.
            format: Report format (json or html).
            
        Returns:
            Path to the saved report.
        """
        if not self.current_context.get("learning_report"):
            raise ValueError("No learning report available to save")
        
        # Create default output path if not provided
        if not output_path:
            timestamp = self.current_context.get("start_time", int(time.time()))
            filename = f"learning_report_{self.current_learning_id}.{format}"
            output_path = os.path.join(os.getcwd(), filename)
        
        # Save report in the specified format
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.current_context["learning_report"], f, indent=2)
        elif format == "html":
            # Generate HTML report
            html_report = self._generate_html_report(self.current_context["learning_report"])
            
            with open(output_path, "w") as f:
                f.write(html_report)
        else:
            raise ValueError(f"Unsupported report format: {format}")
        
        logger.info(f"Learning report saved to {output_path}")
        
        return output_path
    
    def _generate_html_report(self, learning_report: Dict[str, Any]) -> str:
        """
        Generate an HTML report from the learning report.
        
        Args:
            learning_report: Learning report data.
            
        Returns:
            HTML report as a string.
        """
        # Generate HTML report
        # This is a simplified implementation
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Learning Report: {learning_report.get('learning_id')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .recommendation-high {{ background-color: #f8d7da; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .recommendation-medium {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .recommendation-low {{ background-color: #d1ecf1; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .chart {{ height: 200px; background-color: #f8f9fa; margin: 20px 0; padding: 10px; }}
            </style>
        </head>
        <body>
            <h1>Learning and Improvement Report</h1>
            
            <h2>Summary</h2>
            <table>
                <tr><th>Learning ID</th><td>{learning_report.get('learning_id')}</td></tr>
                <tr><th>Timestamp</th><td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(learning_report.get('timestamp', 0)))}</td></tr>
            </table>
            
            <h2>Metrics Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Current Value</th>
                    <th>Improvement</th>
                </tr>
                <tr>
                    <td>Success Rate</td>
                    <td>{learning_report.get('metrics_summary', {}).get('success_rate', 0):.1%}</td>
                    <td class="{'positive' if learning_report.get('improvements', {}).get('success_rate', 0) >= 0 else 'negative'}">
                        {learning_report.get('improvements', {}).get('success_rate', 0):.1%}
                    </td>
                </tr>
                <tr>
                    <td>Mean Time To Resolution (MTTR)</td>
                    <td>{learning_report.get('metrics_summary', {}).get('mttr', 0):.1f} seconds</td>
                    <td class="{'positive' if learning_report.get('improvements', {}).get('mttr', 0) >= 0 else 'negative'}">
                        {learning_report.get('improvements', {}).get('mttr', 0):.1%}
                    </td>
                </tr>
                <tr>
                    <td>Code Coverage</td>
                    <td>{learning_report.get('metrics_summary', {}).get('code_coverage', 0):.1%}</td>
                    <td class="{'positive' if learning_report.get('improvements', {}).get('code_coverage', 0) >= 0 else 'negative'}">
                        {learning_report.get('improvements', {}).get('code_coverage', 0):.1%}
                    </td>
                </tr>
                <tr>
                    <td>Regression Rate</td>
                    <td>{learning_report.get('metrics_summary', {}).get('regression_rate', 0):.1%}</td>
                    <td>N/A</td>
                </tr>
            </table>
            
            <h2>MTTR Optimization</h2>
            <table>
                <tr><th>Current MTTR</th><td>{learning_report.get('mttr_optimization', {}).get('current_mttr', 0):.1f} seconds</td></tr>
                <tr><th>Projected MTTR</th><td>{learning_report.get('mttr_optimization', {}).get('projected_mttr', 0):.1f} seconds</td></tr>
                <tr><th>Improvement</th><td class="{'positive' if learning_report.get('mttr_optimization', {}).get('improvement', 0) >= 0 else 'negative'}">
                    {learning_report.get('mttr_optimization', {}).get('improvement', 0):.1%}
                </td></tr>
                <tr><th>Optimizations Applied</th><td>{learning_report.get('mttr_optimization', {}).get('optimization_count', 0)}</td></tr>
            </table>
            
            <h2>Top Patterns</h2>
        """
        
        # Add top patterns
        for pattern in learning_report.get('top_patterns', []):
            pattern_type = pattern.get('type', '')
            confidence = pattern.get('confidence', 0)
            
            html += f"""
            <div>
                <h3>{pattern_type.replace('_', ' ').title()} Pattern (Confidence: {confidence:.1%})</h3>
                <table>
            """
            
            for key, value in pattern.items():
                if key not in ['type', 'learning_id', 'discovery_time']:
                    html += f"<tr><th>{key.replace('_', ' ').title()}</th><td>{value}</td></tr>"
            
            html += """
                </table>
            </div>
            """
        
        html += """
            <h2>Recommendations</h2>
        """
        
        # Add recommendations
        for recommendation in learning_report.get('recommendations', []):
            priority = recommendation.get('priority', 'low')
            title = recommendation.get('title', 'Untitled Recommendation')
            description = recommendation.get('description', 'No description available.')
            confidence = recommendation.get('confidence', 0)
            
            html += f"""
            <div class="recommendation-{priority}">
                <h3>{title}</h3>
                <p>{description}</p>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                <p><strong>Priority:</strong> {priority.title()}</p>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html

# Main entry point
def learn_from_verification(verification_id: Optional[str] = None, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main function to learn and improve based on verification results.
    
    Args:
        verification_id: ID of the verification to learn from, or None to learn from all recent verifications.
        options: Learning options.
        
    Returns:
        Learning report.
    """
    flow = LearningImprovementFlow()
    learning_id = flow.start_learning(verification_id, options)
    
    # Wait for learning to complete
    while flow.get_learning_status()["status"] not in ["completed", "failed"]:
        time.sleep(0.1)
    
    # Get learning results
    learning_status = flow.get_learning_status()
    
    if learning_status["status"] == "failed":
        logger.error(f"Learning failed: {learning_status.get('error', 'Unknown error')}")
        raise RuntimeError(f"Learning failed: {learning_status.get('error', 'Unknown error')}")
    
    return learning_status["learning_report"]

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python learning_improvement_flow.py <verification_id> [options_json]")
        print("       python learning_improvement_flow.py --all [options_json]")
        sys.exit(1)
    
    verification_id = None
    if sys.argv[1] != "--all":
        verification_id = sys.argv[1]
    
    # Parse options if provided
    options = {}
    options_index = 2 if verification_id else 2
    if len(sys.argv) > options_index:
        try:
            options = json.loads(sys.argv[options_index])
        except json.JSONDecodeError:
            print("Error: options must be a valid JSON string")
            sys.exit(1)
    
    # Run learning and improvement
    try:
        report = learn_from_verification(verification_id, options)
        
        # Create output path
        output_path = options.get("output_path", f"learning_report_{int(time.time())}.json")
        
        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Learning report saved to {output_path}")
        
        # Print summary
        print("\nSummary:")
        print(f"Success Rate: {report['metrics_summary']['success_rate']:.1%}")
        print(f"MTTR: {report['metrics_summary']['mttr']:.1f} seconds")
        print(f"Code Coverage: {report['metrics_summary']['code_coverage']:.1%}")
        print(f"Patterns Discovered: {len(report.get('top_patterns', []))}")
        print(f"Recommendations: {len(report.get('recommendations', []))}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
