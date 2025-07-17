"""
FixWurx Auditor Benchmarking System Demonstration

This script demonstrates how the benchmarking system integrates with the auditor system.
It runs benchmarks on key auditor components, analyzes performance trends, and integrates
with the error reporting system to provide a comprehensive performance overview.
"""

import os
import sys
import time
import logging
import datetime
import random
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [BenchmarkDemo] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('benchmark_demo')

# Import required components
from benchmarking_system import BenchmarkingSystem, BenchmarkConfig
from auditor import Auditor, ObligationLedger, EnergyCalculator, ProofMetrics, MetaAwareness
from sensor_registry import ErrorReport, create_sensor_registry
from time_series_database import TimeSeriesDatabase
from component_sensors import (
    ObligationLedgerSensor, EnergyCalculatorSensor, ProofMetricsSensor,
    MetaAwarenessSensor
)
from graph_database import GraphDatabase

class AuditorBenchmarkDemo:
    """Demonstrates benchmarking system integration with the auditor."""
    
    def __init__(self):
        """Initialize the benchmark demonstration."""
        # Create directories
        os.makedirs("auditor_data/benchmarks", exist_ok=True)
        
        # Create benchmarking system
        self.benchmarking = BenchmarkingSystem(
            storage_path="auditor_data/benchmarks",
            time_series_db_path="auditor_data/time_series"
        )
        
        # Create auditor components for benchmarking
        self.obligation_ledger = ObligationLedger()
        self.energy_calculator = EnergyCalculator()
        self.proof_metrics = ProofMetrics()
        self.meta_awareness = MetaAwareness()
        
        # Create sensor registry and components
        self.registry, self.sensor_manager = create_sensor_registry()
        
        logger.info("Benchmark demonstration initialized")
    
    def run_obligation_ledger_benchmark(self, rule_count: int = 20, obligations_per_rule: int = 10):
        """
        Benchmark the ObligationLedger component.
        
        Args:
            rule_count: Number of rules to create
            obligations_per_rule: Number of obligations per rule
        """
        logger.info(f"Running ObligationLedger benchmark with {rule_count} rules")
        
        # Create benchmark function
        def benchmark_obligation_ledger(rule_count: int, obligations_per_rule: int):
            # Create test data
            delta_rules = []
            for i in range(rule_count):
                rule = {
                    "pattern": f"obligation_{i}",
                    "transforms_to": [f"derived_{i}_{j}" for j in range(obligations_per_rule)],
                    "priority": random.randint(1, 10)
                }
                delta_rules.append(rule)
            
            # Set rules
            self.obligation_ledger.set_delta_rules(delta_rules)
            
            # Generate initial obligations
            initial_obligations = set(f"obligation_{i}" for i in range(rule_count))
            
            # Apply rules and measure performance
            start_time = time.time()
            closure = self.obligation_ledger.compute_closure(initial_obligations)
            end_time = time.time()
            
            # Return metrics
            return {
                "metrics": {
                    "rule_count": rule_count,
                    "initial_obligation_count": len(initial_obligations),
                    "final_obligation_count": len(closure),
                    "throughput": len(closure) / (end_time - start_time) if end_time > start_time else 0
                }
            }
        
        # Create benchmark configuration
        config = BenchmarkConfig(
            name=f"obligation_ledger_benchmark_{rule_count}",
            target="ObligationLedger",
            benchmark_type="PERFORMANCE",
            test_scenario=f"Apply {rule_count} rules to generate obligations",
            iterations=5,
            warmup_iterations=1,
            function=benchmark_obligation_ledger,
            function_args={"rule_count": rule_count, "obligations_per_rule": obligations_per_rule},
            metrics=["execution_time", "cpu_usage", "memory_usage", "throughput"]
        )
        
        # Run benchmark
        result = self.benchmarking.run_benchmark(config)
        
        # Print results
        print(f"\n=== OBLIGATION LEDGER BENCHMARK RESULTS ===")
        if result.success and 'execution_time' in result.statistics:
            print(f"Execution Time: {result.statistics['execution_time']['mean']:.4f} seconds")
            print(f"CPU Usage: {result.statistics['cpu_usage']['mean']:.2f}%")
            print(f"Memory Usage: {result.statistics['memory_usage']['mean']:.2f} MB")
            if "throughput" in result.statistics:
                print(f"Throughput: {result.statistics['throughput']['mean']:.2f} obligations/second")
        else:
            print(f"Benchmark failed: {result.error}")
        
        return result
    
    def run_energy_calculator_benchmark(self, matrix_size: int = 100):
        """
        Benchmark the EnergyCalculator component.
        
        Args:
            matrix_size: Size of the energy matrix to calculate
        """
        logger.info(f"Running EnergyCalculator benchmark with matrix size {matrix_size}")
        
        # Create benchmark function
        def benchmark_energy_calculator(matrix_size: int):
            # Set up energy calculator with test data
            self.energy_calculator.initialize_matrix(matrix_size)
            
            # Run energy minimization and measure performance
            start_time = time.time()
            result = self.energy_calculator.minimize_energy(max_iterations=50)
            end_time = time.time()
            
            # Return metrics
            return {
                "metrics": {
                    "matrix_size": matrix_size,
                    "iterations": result.get("iterations", 0),
                    "final_energy": result.get("final_energy", 0),
                    "energy_delta": result.get("energy_delta", 0),
                    "throughput": result.get("iterations", 0) / (end_time - start_time) if end_time > start_time else 0
                }
            }
        
        # Create benchmark configuration
        config = BenchmarkConfig(
            name=f"energy_calculator_benchmark_{matrix_size}",
            target="EnergyCalculator",
            benchmark_type="PERFORMANCE",
            test_scenario=f"Minimize energy for {matrix_size}x{matrix_size} matrix",
            iterations=5,
            warmup_iterations=1,
            function=benchmark_energy_calculator,
            function_args={"matrix_size": matrix_size},
            metrics=["execution_time", "cpu_usage", "memory_usage", "throughput"]
        )
        
        # Run benchmark
        result = self.benchmarking.run_benchmark(config)
        
        # Print results
        print(f"\n=== ENERGY CALCULATOR BENCHMARK RESULTS ===")
        if result.success and 'execution_time' in result.statistics:
            print(f"Execution Time: {result.statistics['execution_time']['mean']:.4f} seconds")
            print(f"CPU Usage: {result.statistics['cpu_usage']['mean']:.2f}%")
            print(f"Memory Usage: {result.statistics['memory_usage']['mean']:.2f} MB")
            if "throughput" in result.statistics:
                print(f"Throughput: {result.statistics['throughput']['mean']:.2f} iterations/second")
        else:
            print(f"Benchmark failed: {result.error}")
        
        return result
    
    def run_proof_metrics_benchmark(self, proof_count: int = 1000):
        """
        Benchmark the ProofMetrics component.
        
        Args:
            proof_count: Number of proofs to generate
        """
        logger.info(f"Running ProofMetrics benchmark with {proof_count} proofs")
        
        # Create benchmark function
        def benchmark_proof_metrics(proof_count: int):
            # Generate test data
            self.proof_metrics.initialize(proof_count)
            
            # Run verification and measure performance
            start_time = time.time()
            metrics = self.proof_metrics.verify_all()
            end_time = time.time()
            
            # Return metrics
            return {
                "metrics": {
                    "proof_count": proof_count,
                    "verified_count": metrics.get("verified", 0),
                    "coverage": metrics.get("coverage", 0),
                    "residual_bug_probability": metrics.get("p_bug", 0),
                    "throughput": proof_count / (end_time - start_time) if end_time > start_time else 0
                }
            }
        
        # Create benchmark configuration
        config = BenchmarkConfig(
            name=f"proof_metrics_benchmark_{proof_count}",
            target="ProofMetrics",
            benchmark_type="PERFORMANCE",
            test_scenario=f"Verify {proof_count} proofs",
            iterations=5,
            warmup_iterations=1,
            function=benchmark_proof_metrics,
            function_args={"proof_count": proof_count},
            metrics=["execution_time", "cpu_usage", "memory_usage", "throughput"]
        )
        
        # Run benchmark
        result = self.benchmarking.run_benchmark(config)
        
        # Print results
        print(f"\n=== PROOF METRICS BENCHMARK RESULTS ===")
        if result.success and 'execution_time' in result.statistics:
            print(f"Execution Time: {result.statistics['execution_time']['mean']:.4f} seconds")
            print(f"CPU Usage: {result.statistics['cpu_usage']['mean']:.2f}%")
            print(f"Memory Usage: {result.statistics['memory_usage']['mean']:.2f} MB")
            if "throughput" in result.statistics:
                print(f"Throughput: {result.statistics['throughput']['mean']:.2f} proofs/second")
        else:
            print(f"Benchmark failed: {result.error}")
        
        return result
    
    def run_meta_awareness_benchmark(self, dimension: int = 100):
        """
        Benchmark the MetaAwareness component.
        
        Args:
            dimension: Dimension of the awareness space
        """
        logger.info(f"Running MetaAwareness benchmark with dimension {dimension}")
        
        # Create benchmark function
        def benchmark_meta_awareness(dimension: int):
            # Initialize meta-awareness with test data
            self.meta_awareness.initialize(dimension)
            
            # Run reflection and measure performance
            start_time = time.time()
            results = self.meta_awareness.reflect(iterations=10)
            end_time = time.time()
            
            # Return metrics
            return {
                "metrics": {
                    "dimension": dimension,
                    "phi_value": results.get("phi", 0),
                    "drift": results.get("drift", 0),
                    "perturbation": results.get("perturbation", 0),
                    "throughput": 10 / (end_time - start_time) if end_time > start_time else 0
                }
            }
        
        # Create benchmark configuration
        config = BenchmarkConfig(
            name=f"meta_awareness_benchmark_{dimension}",
            target="MetaAwareness",
            benchmark_type="PERFORMANCE",
            test_scenario=f"Reflect with dimension {dimension}",
            iterations=5,
            warmup_iterations=1,
            function=benchmark_meta_awareness,
            function_args={"dimension": dimension},
            metrics=["execution_time", "cpu_usage", "memory_usage", "throughput"]
        )
        
        # Run benchmark
        result = self.benchmarking.run_benchmark(config)
        
        # Print results
        print(f"\n=== META AWARENESS BENCHMARK RESULTS ===")
        if result.success and 'execution_time' in result.statistics:
            print(f"Execution Time: {result.statistics['execution_time']['mean']:.4f} seconds")
            print(f"CPU Usage: {result.statistics['cpu_usage']['mean']:.2f}%")
            print(f"Memory Usage: {result.statistics['memory_usage']['mean']:.2f} MB")
            if "throughput" in result.statistics:
                print(f"Throughput: {result.statistics['throughput']['mean']:.2f} iterations/second")
        else:
            print(f"Benchmark failed: {result.error}")
        
        return result
    
    def run_sensor_integration_benchmark(self, sensor_count: int = 4, error_count: int = 100):
        """
        Benchmark the sensor integration with error reporting.
        
        Args:
            sensor_count: Number of sensors to create
            error_count: Number of errors to generate
        """
        logger.info(f"Running sensor integration benchmark with {sensor_count} sensors and {error_count} errors")
        
        # Create benchmark function
        def benchmark_sensor_integration(sensor_count: int, error_count: int):
            # Create sensors
            sensors = []
            for i in range(sensor_count):
                component_name = f"TestComponent{i}"
                sensor = ObligationLedgerSensor(component_name, {"enabled": True})
                self.registry.register_sensor(sensor)
                sensors.append(sensor)
            
            # Generate errors
            start_time = time.time()
            
            for i in range(error_count):
                sensor = random.choice(sensors)
                error_type = random.choice(["TEST_ERROR_1", "TEST_ERROR_2", "TEST_ERROR_3"])
                severity = random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
                
                sensor.report_error(
                    error_type=error_type,
                    severity=severity,
                    details={"message": f"Test error {i}"},
                    context={"test_id": i}
                )
            
            # Collect errors
            collected = self.sensor_manager.collect_errors(force=True)
            
            end_time = time.time()
            
            # Return metrics
            return {
                "metrics": {
                    "sensor_count": sensor_count,
                    "error_count": error_count,
                    "collected_count": len(collected),
                    "throughput": error_count / (end_time - start_time) if end_time > start_time else 0
                }
            }
        
        # Create benchmark configuration
        config = BenchmarkConfig(
            name=f"sensor_integration_benchmark_{sensor_count}_{error_count}",
            target="SensorRegistry",
            benchmark_type="PERFORMANCE",
            test_scenario=f"Generate and collect {error_count} errors from {sensor_count} sensors",
            iterations=5,
            warmup_iterations=1,
            function=benchmark_sensor_integration,
            function_args={"sensor_count": sensor_count, "error_count": error_count},
            metrics=["execution_time", "cpu_usage", "memory_usage", "throughput"]
        )
        
        # Run benchmark
        result = self.benchmarking.run_benchmark(config)
        
        # Print results
        print(f"\n=== SENSOR INTEGRATION BENCHMARK RESULTS ===")
        if result.success and 'execution_time' in result.statistics:
            print(f"Execution Time: {result.statistics['execution_time']['mean']:.4f} seconds")
            print(f"CPU Usage: {result.statistics['cpu_usage']['mean']:.2f}%")
            print(f"Memory Usage: {result.statistics['memory_usage']['mean']:.2f} MB")
            if "throughput" in result.statistics:
                print(f"Throughput: {result.statistics['throughput']['mean']:.2f} errors/second")
        else:
            print(f"Benchmark failed: {result.error}")
        
        return result
    
    def run_time_series_integration_benchmark(self, point_count: int = 1000):
        """
        Benchmark the time series database integration.
        
        Args:
            point_count: Number of data points to generate
        """
        logger.info(f"Running time series integration benchmark with {point_count} points")
        
        # Create benchmark function
        def benchmark_time_series(point_count: int):
            # Create time series database
            ts_db = TimeSeriesDatabase("auditor_data/time_series")
            
            # Create time series
            series_name = f"benchmark_test_{int(time.time())}"
            ts_db.create_time_series(
                name=series_name,
                description="Benchmark test series",
                unit="count"
            )
            
            # Generate data points
            start_time = time.time()
            
            base_time = datetime.datetime.now() - datetime.timedelta(days=point_count // 24)
            for i in range(point_count):
                timestamp = base_time + datetime.timedelta(hours=i)
                values = {
                    "value1": random.random() * 100,
                    "value2": random.random() * 200,
                    "value3": random.random() * 300
                }
                
                ts_db.add_point(
                    series_name=series_name,
                    timestamp=timestamp,
                    values=values
                )
            
            # Analyze trends
            trend = ts_db.get_trend_analysis(series_name, "value1")
            
            end_time = time.time()
            
            # Return metrics
            return {
                "metrics": {
                    "point_count": point_count,
                    "trend": trend["trend"],
                    "slope": trend["slope"],
                    "throughput": point_count / (end_time - start_time) if end_time > start_time else 0
                }
            }
        
        # Create benchmark configuration
        config = BenchmarkConfig(
            name=f"time_series_benchmark_{point_count}",
            target="TimeSeriesDatabase",
            benchmark_type="PERFORMANCE",
            test_scenario=f"Generate and analyze {point_count} time series points",
            iterations=3,  # Fewer iterations as this is I/O intensive
            warmup_iterations=1,
            function=benchmark_time_series,
            function_args={"point_count": point_count},
            metrics=["execution_time", "cpu_usage", "memory_usage", "throughput"]
        )
        
        # Run benchmark
        result = self.benchmarking.run_benchmark(config)
        
        # Print results
        print(f"\n=== TIME SERIES INTEGRATION BENCHMARK RESULTS ===")
        if result.success and 'execution_time' in result.statistics:
            print(f"Execution Time: {result.statistics['execution_time']['mean']:.4f} seconds")
            print(f"CPU Usage: {result.statistics['cpu_usage']['mean']:.2f}%")
            print(f"Memory Usage: {result.statistics['memory_usage']['mean']:.2f} MB")
            if "throughput" in result.statistics:
                print(f"Throughput: {result.statistics['throughput']['mean']:.2f} points/second")
        else:
            print(f"Benchmark failed: {result.error}")
        
        return result
    
    def generate_benchmark_report(self):
        """Generate a comprehensive benchmark report."""
        logger.info("Generating benchmark report")
        
        # Generate report
        report = self.benchmarking.generate_report()
        
        # Print report summary
        print("\n=== BENCHMARK REPORT SUMMARY ===")
        print(f"Generated at: {report['generated_at']}")
        print(f"Targets: {', '.join(report['targets'].keys())}")
        
        # Print recent benchmarks
        print("\nRecent Benchmarks:")
        for bench in report['recent_benchmarks']:
            print(f"- {bench['benchmark_id']} ({bench['target']}): {'✓' if bench['success'] else '✗'}")
        
        # Print target details
        print("\nTarget Details:")
        for target, details in report['targets'].items():
            print(f"\n{target}:")
            print(f"  Latest benchmark: {details['latest_benchmark']}")
            print(f"  Latest timestamp: {details['latest_timestamp']}")
            
            # Print metric trends
            if 'trends' in details:
                print("  Metric Trends:")
                for metric, trend in details['trends'].items():
                    print(f"    {metric}: {trend['trend']} (Slope: {trend.get('slope', 'N/A')})")
        
        return report
    
    def run_all_benchmarks(self):
        """Run all benchmarks and generate a report."""
        print("\n*** AUDITOR BENCHMARKING SYSTEM DEMONSTRATION ***\n")
        
        # Run component benchmarks
        print("\nStep 1: Running component benchmarks...")
        self.run_obligation_ledger_benchmark(rule_count=15)
        self.run_energy_calculator_benchmark(matrix_size=80)
        self.run_proof_metrics_benchmark(proof_count=800)
        self.run_meta_awareness_benchmark(dimension=80)
        
        # Run integration benchmarks
        print("\nStep 2: Running integration benchmarks...")
        self.run_sensor_integration_benchmark(sensor_count=4, error_count=80)
        self.run_time_series_integration_benchmark(point_count=500)
        
        # Generate report
        print("\nStep 3: Generating comprehensive benchmark report...")
        self.generate_benchmark_report()
        
        # Print conclusion
        print("\n*** BENCHMARKING DEMONSTRATION COMPLETE ***\n")
        print("The benchmarking system has been successfully integrated with the auditor system.")
        print("It provides performance measurements for all key components and integrations,")
        print("tracks performance trends over time, and generates comprehensive reports.")


if __name__ == "__main__":
    # Run the demonstration
    demo = AuditorBenchmarkDemo()
    demo.run_all_benchmarks()
