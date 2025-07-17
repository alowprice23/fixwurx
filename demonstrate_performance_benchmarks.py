#!/usr/bin/env python3
"""
FixWurx Auditor Performance Benchmarks Demonstration

This script demonstrates the comprehensive performance benchmarking system
for the auditor. It shows how the 20 quantitative metrics are tracked and
analyzed during debugging sessions to provide a complete view of the auditor's
performance in finding, fixing, and certifying defects.
"""

import os
import logging
import json
import datetime
import time
import random
import math
from typing import Dict, Any, List

# Import sensor components
from error_report import ErrorReport
from sensor_base import ErrorSensor
from sensor_registry import SensorRegistry, create_sensor_registry
from sensor_manager import SensorManager
from performance_benchmark_sensor import PerformanceBenchmarkSensor
from sensor_llm_bridge import SensorLLMBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [BenchmarkDemo] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('benchmark_demo')


class SimulatedDebuggingSession:
    """Simulates a debugging session with metrics for benchmarking."""
    
    def __init__(self, bugs_count=10, difficulty_level=0.7):
        """
        Initialize a simulated debugging session.
        
        Args:
            bugs_count: Number of bugs to simulate
            difficulty_level: Difficulty level (0-1), higher means harder bugs
        """
        self.bugs_count = bugs_count
        self.difficulty_level = difficulty_level
        self.detected_bugs = 0
        self.resolved_bugs = 0
        self.iteration = 0
        self.diagnose_iterations = []
        self.repair_iterations = []
        self.lyapunov_history = []
        self.energy_start = 100.0 * difficulty_level * bugs_count
        self.energy_current = self.energy_start
        self.lambda_current = 0.99
        self.proof_coverage_start = 0.4
        self.proof_coverage_current = self.proof_coverage_start
        self.residual_risk_start = 0.05
        self.residual_risk_current = self.residual_risk_start
        self.tests_total = 50
        self.tests_passed = int(self.tests_total * (1 - difficulty_level))
        self.patches_applied = 0
        self.new_failures = 0
        self.tokens_used = 0
        self.agent_messages = 0
        self.meta_guard_breaches = 0
        self.hallucination_lines = 0
        self.total_generated_lines = 100
        self.memory_usage_mb = 150.0
        self.cpu_usage_pct = 20.0
        self.doc_lines = 200
        self.code_lines = 1000
        self.auditor_failed = False
        self.auditor_passed = False
        self.redundant_ast_nodes = int(200 * difficulty_level)
        self.total_ast_nodes = 2000
        
        # Special events in the session
        self.has_added_regression = False
        self.has_fixed_critical = False
        self.has_improved_coverage = False
        
        logger.info(f"Initialized debugging session with {bugs_count} bugs at difficulty {difficulty_level}")
    
    def get_initial_metrics(self) -> Dict[str, Any]:
        """Get initial metrics for the benchmark sensor."""
        return {
            "total_known_bugs": self.bugs_count,
            "energy_start": self.energy_start,
            "energy_current": self.energy_start,
            "lambda_current": self.lambda_current,
            "proof_coverage_start": self.proof_coverage_start,
            "proof_coverage_current": self.proof_coverage_start,
            "residual_risk_start": self.residual_risk_start,
            "residual_risk_current": self.residual_risk_start,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "tokens_used_start": self.tokens_used,
            "tokens_used_current": self.tokens_used,
            "memory_usage_start_mb": self.memory_usage_mb,
            "memory_usage_current_mb": self.memory_usage_mb,
            "cpu_usage_start_pct": self.cpu_usage_pct,
            "cpu_usage_current_pct": self.cpu_usage_pct,
            "doc_lines_start": self.doc_lines,
            "code_lines_start": self.code_lines,
            "doc_lines_current": self.doc_lines,
            "code_lines_current": self.code_lines,
            "redundant_ast_nodes": self.redundant_ast_nodes,
            "total_ast_nodes": self.total_ast_nodes,
            "lines_without_obligation": self.hallucination_lines,
            "total_generated_lines": self.total_generated_lines
        }
    
    def simulate_iteration(self) -> Dict[str, Any]:
        """
        Simulate an iteration of the debugging session.
        
        Returns:
            Updated metrics
        """
        self.iteration += 1
        updates = {}
        
        # Detect bugs
        if self.detected_bugs < self.bugs_count:
            detection_probability = 0.2 - (0.1 * self.difficulty_level)
            if random.random() < detection_probability:
                self.detected_bugs += 1
                updates["bug_detected"] = True
                diagnose_iters = max(1, int(random.normalvariate(
                    5 + (10 * self.difficulty_level),
                    2
                )))
                self.diagnose_iterations.append(diagnose_iters)
                updates["diagnose_iteration"] = True
                updates["iteration_count"] = diagnose_iters
                logger.info(f"Detected bug {self.detected_bugs}/{self.bugs_count} "
                           f"after {diagnose_iters} iterations")
        
        # Resolve bugs
        if self.detected_bugs > self.resolved_bugs:
            resolution_probability = 0.15 - (0.1 * self.difficulty_level)
            if random.random() < resolution_probability:
                self.resolved_bugs += 1
                updates["bug_resolved"] = True
                repair_iters = max(1, int(random.normalvariate(
                    8 + (15 * self.difficulty_level),
                    3
                )))
                self.repair_iterations.append(repair_iters)
                updates["repair_iteration"] = True
                updates["iteration_count"] = repair_iters
                
                # Apply patch
                self.patches_applied += 1
                updates["patch_applied"] = True
                
                # Possible regression
                if random.random() < 0.2 * self.difficulty_level and not self.has_added_regression:
                    self.new_failures += 1
                    updates["new_failures"] = 1
                    self.tests_passed -= 1
                    self.has_added_regression = True
                    logger.info(f"Patch introduced a regression! Tests now: {self.tests_passed}/{self.tests_total}")
                
                logger.info(f"Resolved bug {self.resolved_bugs}/{self.detected_bugs} "
                           f"after {repair_iters} iterations")
        
        # Update energy and convergence
        energy_reduction = (0.05 + (0.03 * random.random())) * (self.resolved_bugs / max(1, self.bugs_count))
        self.energy_current = max(0, self.energy_current * (1 - energy_reduction))
        self.lambda_current = max(0.9, self.lambda_current * (1 - 0.01 * energy_reduction))
        
        # Special event: improve test coverage 
        if self.iteration % 10 == 0 and self.tests_passed < self.tests_total:
            improvement = random.randint(1, 2)
            self.tests_passed = min(self.tests_total, self.tests_passed + improvement)
            logger.info(f"Improved test coverage: {self.tests_passed}/{self.tests_total}")
        
        # Special event: improve proof coverage
        if self.iteration % 15 == 0 and not self.has_improved_coverage:
            self.proof_coverage_current += 0.05 + (0.05 * random.random())
            self.residual_risk_current = self.residual_risk_start * (1 - (self.proof_coverage_current - self.proof_coverage_start))
            self.has_improved_coverage = True
            logger.info(f"Improved proof coverage to {self.proof_coverage_current:.2f}, "
                       f"residual risk: {self.residual_risk_current:.4f}")
        
        # Update resource usage
        self.tokens_used += int(random.normalvariate(500, 100))
        self.agent_messages += random.randint(1, 3)
        self.memory_usage_mb += random.normalvariate(0.5, 0.2)
        self.cpu_usage_pct += random.normalvariate(0.2, 0.1)
        
        # Special event: hallucination
        if random.random() < 0.05 * self.difficulty_level:
            hallucination_count = random.randint(1, 5)
            self.hallucination_lines += hallucination_count
            self.total_generated_lines += hallucination_count + random.randint(5, 10)
            logger.info(f"Detected {hallucination_count} lines of hallucinated code")
        
        # Special event: meta guard breach
        if random.random() < 0.02 * self.difficulty_level and self.meta_guard_breaches < 2:
            self.meta_guard_breaches += 1
            updates["meta_guard_breach"] = True
            logger.info(f"Meta guard breach detected! Total: {self.meta_guard_breaches}")
        
        # Special event: documentation update
        if self.iteration % 20 == 0:
            doc_change = int(random.normalvariate(10, 5))
            code_change = int(random.normalvariate(20, 8))
            self.doc_lines += doc_change
            self.code_lines += code_change
            logger.info(f"Updated documentation: {self.doc_lines} doc lines, {self.code_lines} code lines")
        
        # Update Lyapunov function
        is_decreasing = random.random() < (0.95 - (0.1 * self.difficulty_level))
        self.lyapunov_history.append(is_decreasing)
        updates["lyapunov_update"] = True
        updates["decreasing"] = is_decreasing
        
        # Auditor status changes
        if self.resolved_bugs >= int(self.bugs_count * 0.5) and not self.auditor_failed:
            self.auditor_failed = True
            updates["auditor_fail"] = True
            logger.info("Auditor status: FAIL (transitioning to assessment)")
        
        if self.resolved_bugs >= int(self.bugs_count * 0.8) and self.auditor_failed and not self.auditor_passed:
            self.auditor_passed = True
            updates["auditor_pass"] = True
            logger.info("Auditor status: PASS (requirements met)")
        
        # Add current metrics
        updates.update({
            "detected_bugs": self.detected_bugs,
            "resolved_bugs": self.resolved_bugs,
            "energy_current": self.energy_current,
            "lambda_current": self.lambda_current,
            "convergence_iter_count": self.iteration,
            "proof_coverage_current": self.proof_coverage_current,
            "residual_risk_current": self.residual_risk_current,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "tokens_used_current": self.tokens_used,
            "agent_messages": self.agent_messages,
            "memory_usage_current_mb": self.memory_usage_mb,
            "cpu_usage_current_pct": self.cpu_usage_pct,
            "doc_lines_current": self.doc_lines,
            "code_lines_current": self.code_lines,
            "redundant_ast_nodes": self.redundant_ast_nodes,
            "total_ast_nodes": self.total_ast_nodes,
            "lines_without_obligation": self.hallucination_lines,
            "total_generated_lines": self.total_generated_lines
        })
        
        return updates


def create_benchmark_system():
    """Create the benchmark system."""
    # Create directories for sensor data
    os.makedirs("auditor_data/sensors", exist_ok=True)
    
    # Create registry and manager
    registry = create_sensor_registry({
        "sensors": {
            "storage_path": "auditor_data/sensors"
        }
    })
    
    manager = SensorManager(registry, {
        "sensors_enabled": True,
        "collection_interval_seconds": 5
    })
    
    # Create benchmark sensor with thresholds
    benchmark_sensor = PerformanceBenchmarkSensor(config={
        "thresholds": {
            "bug_detection_recall": 0.7,
            "bug_fix_yield": 0.6,
            "mttd": 12,
            "mttr": 25,
            "energy_reduction_pct": 0.5,
            "test_pass_ratio": 0.9,
            "regression_introduction_rate": 0.15,
            "hallucination_rate": 0.1,
            "lyapunov_descent_consistency": 0.9,
            "meta_guard_breach_count": 1
        }
    })
    registry.register_sensor(benchmark_sensor)
    
    # Create LLM bridge
    bridge = SensorLLMBridge(registry)
    
    return registry, manager, bridge, benchmark_sensor


def run_simulated_session(session, benchmark_sensor, iterations=50):
    """Run a simulated debugging session."""
    print("\n=== Running Simulated Debugging Session ===\n")
    
    # Initialize benchmark sensor with initial metrics
    initial_metrics = session.get_initial_metrics()
    reports = benchmark_sensor.monitor(initial_metrics)
    
    # Run iterations
    for i in range(1, iterations + 1):
        print(f"\nIteration {i}/{iterations}")
        print("-" * 50)
        
        # Simulate an iteration
        updates = session.simulate_iteration()
        
        # Update benchmark sensor
        reports = benchmark_sensor.monitor(updates)
        
        # Check for errors
        if reports:
            print(f"\nBenchmark Sensor detected {len(reports)} issues:")
            for j, report in enumerate(reports):
                print(f"  {j+1}. {report.error_type} - {report.severity}")
                print(f"     {report.details.get('message', 'No details')}")
        
        # Show some metrics every 10 iterations
        if i % 10 == 0 or i == iterations:
            metrics = benchmark_sensor.get_metrics_summary()
            print("\nCurrent Metrics:")
            print(f"  Bug Detection Recall: {metrics['bug_detection_recall']:.2f}")
            print(f"  Bug Fix Yield: {metrics['bug_fix_yield']:.2f}")
            print(f"  Test Pass Ratio: {metrics['test_pass_ratio']:.2f}")
            print(f"  Energy Reduction: {metrics['energy_reduction_pct']:.2f}")
            print(f"  Mean Time to Repair: {metrics['mttr']:.1f} iterations")
            print(f"  Lyapunov Descent Consistency: {metrics['lyapunov_descent_consistency']:.2f}")
            print(f"  Aggregate Confidence Score: {metrics['aggregate_confidence_score']:.2f}")
        
        # Pause between iterations
        time.sleep(0.2)
    
    return benchmark_sensor.session_metrics


def demonstrate_performance_metrics(metrics):
    """Demonstrate performance metrics analysis."""
    print("\n=== Performance Metrics Analysis ===\n")
    
    # Detection and fix metrics
    print("Bug Detection and Fix Metrics:")
    print(f"  Bug Detection Recall: {metrics.get('bug_detection_recall', 0):.2f}")
    print(f"    ({metrics.get('detected_bugs', 0)}/{metrics.get('total_known_bugs', 0)} bugs detected)")
    
    print(f"  Bug Fix Yield: {metrics.get('bug_fix_yield', 0):.2f}")
    print(f"    ({metrics.get('resolved_bugs', 0)}/{metrics.get('detected_bugs', 0)} detected bugs fixed)")
    
    print(f"  Mean Time to Diagnose: {metrics.get('mttd', 0):.1f} iterations")
    print(f"  Mean Time to Repair: {metrics.get('mttr', 0):.1f} iterations")
    
    # Energy and convergence metrics
    print("\nEnergy and Convergence Metrics:")
    print(f"  Convergence Iterations: {metrics.get('convergence_iter_count', 0)}")
    print(f"  Energy Reduction: {metrics.get('energy_reduction_pct', 0):.2f}")
    print(f"    (From {metrics.get('energy_start', 0):.1f} to {metrics.get('energy_current', 0):.1f})")
    
    print(f"  Proof Coverage Delta: {metrics.get('proof_coverage_delta', 0):.2f}")
    print(f"    (From {metrics.get('proof_coverage_start', 0):.2f} to {metrics.get('proof_coverage_current', 0):.2f})")
    
    print(f"  Residual Risk Improvement: {metrics.get('residual_risk_improvement', 0):.4f}")
    print(f"    (From {metrics.get('residual_risk_start', 0):.4f} to {metrics.get('residual_risk_current', 0):.4f})")
    
    # Test and quality metrics
    print("\nTest and Quality Metrics:")
    print(f"  Test Pass Ratio: {metrics.get('test_pass_ratio', 0):.2f}")
    print(f"    ({metrics.get('tests_passed', 0)}/{metrics.get('tests_total', 0)} tests passing)")
    
    print(f"  Regression Introduction Rate: {metrics.get('regression_introduction_rate', 0):.2f}")
    print(f"    ({metrics.get('new_failures', 0)} regressions in {metrics.get('patches_applied', 0)} patches)")
    
    print(f"  Duplicate Module Ratio: {metrics.get('duplicate_module_ratio', 0):.2f}")
    print(f"    ({metrics.get('redundant_ast_nodes', 0)}/{metrics.get('total_ast_nodes', 0)} AST nodes redundant)")
    
    # Agent and resource metrics
    print("\nAgent and Resource Metrics:")
    print(f"  Agent Coordination Overhead: {metrics.get('agent_coordination_overhead', 0):.1f} messages/fix")
    print(f"  Token Per Fix Efficiency: {metrics.get('token_per_fix_efficiency', 0):.1f} tokens/fix")
    print(f"  Hallucination Rate: {metrics.get('hallucination_rate', 0):.2f}")
    print(f"    ({metrics.get('lines_without_obligation', 0)}/{metrics.get('total_generated_lines', 0)} lines without obligation)")
    
    # Time and stability metrics
    print("\nTime and Stability Metrics:")
    print(f"  Certainty Gap Closure Time: {metrics.get('certainty_gap_closure_time', 0):.1f} seconds")
    print(f"  Lyapunov Descent Consistency: {metrics.get('lyapunov_descent_consistency', 0):.2f}")
    print(f"    ({metrics.get('lyapunov_decreasing_iterations', 0)}/{metrics.get('total_iterations', 0)} iterations decreasing)")
    print(f"  Meta Guard Breach Count: {metrics.get('meta_guard_breaches', 0)}")
    
    # Resource and documentation metrics
    print("\nResource and Documentation Metrics:")
    print(f"  Resource Footprint Change: {metrics.get('resource_footprint_change', 0):.2f}")
    print(f"    (Memory: {metrics.get('memory_usage_start_mb', 0):.1f}MB → {metrics.get('memory_usage_current_mb', 0):.1f}MB)")
    print(f"    (CPU: {metrics.get('cpu_usage_start_pct', 0):.1f}% → {metrics.get('cpu_usage_current_pct', 0):.1f}%)")
    
    doc_ratio_start = metrics.get('doc_lines_start', 0) / max(1, metrics.get('code_lines_start', 1))
    doc_ratio_current = metrics.get('doc_lines_current', 0) / max(1, metrics.get('code_lines_current', 1))
    
    print(f"  Documentation Completeness Delta: {metrics.get('documentation_completeness_delta', 0):.2f}")
    print(f"    (Doc/Code ratio: {doc_ratio_start:.2f} → {doc_ratio_current:.2f})")
    
    # Aggregate confidence
    print(f"\nAggregate Confidence Score: {metrics.get('aggregate_confidence_score', 0):.2f}")


def demonstrate_benchmark_integration():
    """Demonstrate the benchmark integration with the auditor."""
    print("\n=== Demonstrating Benchmark Integration with Auditor ===\n")
    
    print("The benchmark system integrates with the auditor's sensor framework to:")
    print("  1. Track 20 quantitative benchmarks in real-time")
    print("  2. Generate error reports when metrics fall below thresholds")
    print("  3. Provide insights into debugging efficiency and effectiveness")
    print("  4. Store time-series data in the Lyapunov history table")
    print("  5. Calculate an aggregate confidence score for the system")
    
    print("\nThese benchmarks provide a 360° view of the auditor's performance in:")
    print("  • Finding bugs (detection recall, MTTD)")
    print("  • Fixing issues (fix yield, MTTR, regression rate)")
    print("  • Certifying correctness (proof coverage, test pass ratio)")
    
    print("\nThe system also tracks:")
    print("  • Resource efficiency (tokens per fix, memory footprint)")
    print("  • Agent coordination (message overhead)")
    print("  • Self-awareness (meta guard breaches)")
    print("  • Documentation completeness")
    
    print("\nBy tracking these metrics, the auditor can:")
    print("  1. Identify inefficiencies in its debugging process")
    print("  2. Recognize when it's introducing new problems")
    print("  3. Ensure convergence toward correctness")
    print("  4. Maintain a balanced approach to debugging")


def main():
    """Main demonstration function."""
    print("\n" + "="*60)
    print("AUDITOR PERFORMANCE BENCHMARKING DEMONSTRATION")
    print("="*60 + "\n")
    
    print("Initializing benchmark system...")
    registry, manager, bridge, benchmark_sensor = create_benchmark_system()
    
    # Create simulated debugging session
    print("Creating simulated debugging session...")
    session = SimulatedDebuggingSession(bugs_count=15, difficulty_level=0.6)
    
    # Run session
    metrics = run_simulated_session(session, benchmark_sensor, iterations=30)
    
    # Demonstrate performance analysis
    demonstrate_performance_metrics(metrics)
    
    # Demonstrate integration
    demonstrate_benchmark_integration()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60 + "\n")
    
    print("The auditor now has comprehensive performance monitoring capabilities")
    print("through its integrated benchmark sensor framework. It can track and")
    print("analyze 20 key performance indicators that provide a complete view of")
    print("its debugging efficiency and effectiveness.")
    
    return 0


if __name__ == "__main__":
    exit(main())
