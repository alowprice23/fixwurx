"""
FixWurx Benchmarking System

This module implements a comprehensive benchmarking system for the Auditor agent
to measure and track the performance of system components over time. It provides
functionality for running benchmarks, storing results, and analyzing performance trends.

See docs/auditor_agent_specification.md for full specification.
"""

import os
import json
import logging
import datetime
import time
import uuid
import subprocess
import statistics
import platform
import psutil
import yaml
import multiprocessing
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Import time series database for storing benchmark results
from time_series_database import TimeSeriesDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [Benchmark] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('benchmarking_system')


class BenchmarkConfig:
    """
    Configuration for a benchmark.
    
    Defines the parameters for running a benchmark, including the target,
    test scenario, environment settings, and measurement parameters.
    """
    
    def __init__(self, 
                name: str,
                target: str,
                benchmark_type: str,
                test_scenario: str = "",
                iterations: int = 5,
                warmup_iterations: int = 1,
                timeout_seconds: int = 60,
                environment: Dict = None,
                metrics: List[str] = None,
                command: str = None,
                function: Callable = None,
                function_args: Dict = None):
        """
        Initialize a benchmark configuration.
        
        Args:
            name: Name of the benchmark
            target: The target component to benchmark
            benchmark_type: Type of benchmark (PERFORMANCE, RELIABILITY, SCALABILITY)
            test_scenario: Description of the test scenario
            iterations: Number of iterations to run
            warmup_iterations: Number of warmup iterations (not included in results)
            timeout_seconds: Maximum time per iteration
            environment: Environment details
            metrics: List of metrics to measure
            command: Shell command to execute for the benchmark
            function: Python function to call for the benchmark
            function_args: Arguments to pass to the function
        """
        self.name = name
        self.target = target
        self.benchmark_type = benchmark_type
        self.test_scenario = test_scenario
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.timeout_seconds = timeout_seconds
        self.environment = environment or {}
        self.metrics = metrics or ["execution_time", "cpu_usage", "memory_usage"]
        self.command = command
        self.function = function
        self.function_args = function_args or {}
        
        # Validate configuration
        if not command and not function:
            raise ValueError("Either command or function must be provided")
    
    def to_dict(self) -> Dict:
        """Convert benchmark configuration to dictionary representation"""
        result = {
            "name": self.name,
            "target": self.target,
            "benchmark_type": self.benchmark_type,
            "test_scenario": self.test_scenario,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "timeout_seconds": self.timeout_seconds,
            "environment": self.environment,
            "metrics": self.metrics
        }
        
        if self.command:
            result["command"] = self.command
        
        # Cannot serialize function, so just note if it's present
        if self.function:
            result["has_function"] = True
            result["function_args"] = self.function_args
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BenchmarkConfig':
        """Create benchmark configuration from dictionary representation"""
        return cls(
            name=data["name"],
            target=data["target"],
            benchmark_type=data["benchmark_type"],
            test_scenario=data.get("test_scenario", ""),
            iterations=data.get("iterations", 5),
            warmup_iterations=data.get("warmup_iterations", 1),
            timeout_seconds=data.get("timeout_seconds", 60),
            environment=data.get("environment", {}),
            metrics=data.get("metrics", ["execution_time", "cpu_usage", "memory_usage"]),
            command=data.get("command"),
            function=None,  # Function cannot be deserialized
            function_args=data.get("function_args", {})
        )


class BenchmarkResult:
    """
    Result of a benchmark run.
    
    Contains the measurements, statistics, and metadata for a benchmark run.
    """
    
    def __init__(self, 
                benchmark_id: str,
                config: BenchmarkConfig,
                start_time: datetime.datetime,
                end_time: datetime.datetime,
                system_info: Dict,
                measurements: List[Dict],
                statistics: Dict,
                success: bool,
                error: str = None):
        """
        Initialize a benchmark result.
        
        Args:
            benchmark_id: Unique identifier for the benchmark run
            config: The benchmark configuration
            start_time: Start time of the benchmark run
            end_time: End time of the benchmark run
            system_info: Information about the system where the benchmark was run
            measurements: List of individual measurements
            statistics: Statistics calculated from the measurements
            success: Whether the benchmark run was successful
            error: Error message if the benchmark run failed
        """
        self.benchmark_id = benchmark_id
        self.config = config
        self.start_time = start_time
        self.end_time = end_time
        self.system_info = system_info
        self.measurements = measurements
        self.statistics = statistics
        self.success = success
        self.error = error
    
    def to_dict(self) -> Dict:
        """Convert benchmark result to dictionary representation"""
        return {
            "benchmark_id": self.benchmark_id,
            "config": self.config.to_dict(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "system_info": self.system_info,
            "measurements": self.measurements,
            "statistics": self.statistics,
            "success": self.success,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BenchmarkResult':
        """Create benchmark result from dictionary representation"""
        try:
            start_time = datetime.datetime.fromisoformat(data["start_time"])
            end_time = datetime.datetime.fromisoformat(data["end_time"])
            
            return cls(
                benchmark_id=data["benchmark_id"],
                config=BenchmarkConfig.from_dict(data["config"]),
                start_time=start_time,
                end_time=end_time,
                system_info=data["system_info"],
                measurements=data["measurements"],
                statistics=data["statistics"],
                success=data["success"],
                error=data.get("error")
            )
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to create BenchmarkResult from dict: {e}")
            
            # Return a default result
            return cls(
                benchmark_id="unknown",
                config=BenchmarkConfig(
                    name="unknown",
                    target="unknown",
                    benchmark_type="unknown",
                    command="echo 'default'"
                ),
                start_time=datetime.datetime.now(),
                end_time=datetime.datetime.now(),
                system_info={},
                measurements=[],
                statistics={},
                success=False,
                error=str(e)
            )


class BenchmarkingSystem:
    """
    Benchmarking system implementation for the Auditor agent.
    
    Provides functionality for running benchmarks, storing results, and
    analyzing performance trends over time.
    """
    
    def __init__(self, storage_path: str, time_series_db_path: str = None):
        """
        Initialize the benchmarking system.
        
        Args:
            storage_path: Path to the storage directory for benchmark results
            time_series_db_path: Path to the time series database for storing trends
        """
        self.storage_path = storage_path
        self.time_series_db_path = time_series_db_path or os.path.join(storage_path, "time_series")
        self.time_series_db = TimeSeriesDatabase(self.time_series_db_path)
        
        # Ensure storage paths exist
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.time_series_db_path, exist_ok=True)
    
    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run a benchmark based on the provided configuration.
        
        Args:
            config: The benchmark configuration
            
        Returns:
            The benchmark result
        """
        benchmark_id = f"BENCH-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"
        
        logger.info(f"Starting benchmark {config.name} (ID: {benchmark_id})")
        
        # Record start time
        start_time = datetime.datetime.now()
        
        # Get system info
        system_info = self._get_system_info()
        
        # Initialize measurements and statistics
        measurements = []
        success = True
        error = None
        
        try:
            # Run warmup iterations
            logger.info(f"Running {config.warmup_iterations} warmup iterations")
            for i in range(config.warmup_iterations):
                self._run_iteration(config, warmup=True)
            
            # Run benchmark iterations
            logger.info(f"Running {config.iterations} benchmark iterations")
            for i in range(config.iterations):
                measurement = self._run_iteration(config)
                measurements.append(measurement)
            
            # Calculate statistics
            statistics = self._calculate_statistics(measurements)
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            success = False
            error = str(e)
            statistics = {}
        
        # Record end time
        end_time = datetime.datetime.now()
        
        # Create result
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            config=config,
            start_time=start_time,
            end_time=end_time,
            system_info=system_info,
            measurements=measurements,
            statistics=statistics,
            success=success,
            error=error
        )
        
        # Store result
        self._store_result(result)
        
        # Update time series database
        self._update_time_series(result)
        
        logger.info(f"Benchmark {config.name} completed in {(end_time - start_time).total_seconds():.2f} seconds")
        
        return result
    
    def _run_iteration(self, config: BenchmarkConfig, warmup: bool = False) -> Dict:
        """
        Run a single iteration of the benchmark.
        
        Args:
            config: The benchmark configuration
            warmup: Whether this is a warmup iteration
            
        Returns:
            Dictionary with the measurement results
        """
        measurement = {
            "timestamp": datetime.datetime.now().isoformat(),
            "iteration": "warmup" if warmup else f"iteration-{len(self.measurements) if hasattr(self, 'measurements') else 0 + 1}"
        }
        
        # Start monitoring resources
        initial_cpu = psutil.cpu_percent(interval=None)
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run the benchmark
        start_time = time.time()
        
        if config.command:
            # Run command
            result = self._run_command(config.command, config.timeout_seconds)
            measurement["command_output"] = result.get("stdout", "")
            measurement["command_error"] = result.get("stderr", "")
            measurement["command_exit_code"] = result.get("exit_code", -1)
            
        elif config.function:
            # Call function
            try:
                result = config.function(**config.function_args)
                measurement["function_result"] = str(result)
            except Exception as e:
                measurement["function_error"] = str(e)
        
        end_time = time.time()
        
        # Measure resources
        final_cpu = psutil.cpu_percent(interval=None)
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Record measurements
        measurement["execution_time"] = end_time - start_time
        measurement["cpu_usage"] = final_cpu - initial_cpu
        measurement["memory_usage"] = final_memory - initial_memory
        
        # Add custom metrics if function returned them
        if isinstance(result, dict) and "metrics" in result:
            for key, value in result["metrics"].items():
                measurement[key] = value
        
        return measurement
    
    def _run_command(self, command: str, timeout_seconds: int) -> Dict:
        """
        Run a shell command and capture its output.
        
        Args:
            command: The command to run
            timeout_seconds: Timeout in seconds
            
        Returns:
            Dictionary with command output, error, and exit code
        """
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": process.returncode
            }
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": -1,
                "timeout": True
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "error": str(e)
            }
    
    def _calculate_statistics(self, measurements: List[Dict]) -> Dict:
        """
        Calculate statistics from the measurements.
        
        Args:
            measurements: List of measurements
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Get all metric names
        metric_names = set()
        for measurement in measurements:
            metric_names.update(k for k in measurement.keys() if isinstance(measurement[k], (int, float)))
        
        # Calculate statistics for each metric
        for metric in metric_names:
            values = [m[metric] for m in measurements if metric in m]
            
            if not values:
                continue
            
            stats[metric] = {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0
            }
        
        return stats
    
    def _get_system_info(self) -> Dict:
        """
        Get information about the system.
        
        Returns:
            Dictionary with system information
        """
        cpu_info = {}
        
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_info["model"] = line.split(":")[1].strip()
                            break
            except Exception:
                cpu_info["model"] = platform.processor()
        else:
            cpu_info["model"] = platform.processor()
        
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "platform_release": platform.release(),
            "cpu": cpu_info,
            "cpu_count": multiprocessing.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }
    
    def _store_result(self, result: BenchmarkResult) -> None:
        """
        Store the benchmark result.
        
        Args:
            result: The benchmark result
        """
        try:
            # Create result directory if it doesn't exist
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Write result to file
            file_path = os.path.join(self.storage_path, f"{result.benchmark_id}.yaml")
            
            with open(file_path, 'w') as f:
                yaml.dump(result.to_dict(), f, default_flow_style=False)
            
            logger.info(f"Stored benchmark result to {file_path}")
        except Exception as e:
            logger.error(f"Failed to store benchmark result: {e}")
    
    def _update_time_series(self, result: BenchmarkResult) -> None:
        """
        Update the time series database with the benchmark result.
        
        Args:
            result: The benchmark result
        """
        try:
            # Get time series for the target and metrics
            target = result.config.target
            series_name = f"benchmark_{target}"
            
            # Create time series if it doesn't exist
            self.time_series_db.create_time_series(
                name=series_name,
                description=f"Benchmark results for {target}",
                unit="various"
            )
            
            # Add data point for each metric in the statistics
            metrics = {}
            
            for metric, stats in result.statistics.items():
                metrics[f"{metric}_mean"] = stats["mean"]
                metrics[f"{metric}_min"] = stats["min"]
                metrics[f"{metric}_max"] = stats["max"]
            
            # Add the data point
            self.time_series_db.add_point(
                series_name=series_name,
                timestamp=result.end_time,
                values=metrics
            )
            
            logger.info(f"Updated time series for {target}")
        except Exception as e:
            logger.error(f"Failed to update time series: {e}")
    
    def get_benchmark_results(self, target: str = None, benchmark_type: str = None, 
                            limit: int = 10) -> List[BenchmarkResult]:
        """
        Get benchmark results matching the specified criteria.
        
        Args:
            target: Filter by target component
            benchmark_type: Filter by benchmark type
            limit: Maximum number of results to return
            
        Returns:
            List of benchmark results
        """
        try:
            # List all result files
            files = [f for f in os.listdir(self.storage_path) if f.startswith("BENCH-") and f.endswith(".yaml")]
            
            # Sort by timestamp (newest first)
            files.sort(reverse=True)
            
            results = []
            
            for file in files[:limit * 2]:  # Get more than needed to allow for filtering
                try:
                    file_path = os.path.join(self.storage_path, file)
                    
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    # Apply filters
                    if target and data["config"]["target"] != target:
                        continue
                    
                    if benchmark_type and data["config"]["benchmark_type"] != benchmark_type:
                        continue
                    
                    # Create result object
                    result = BenchmarkResult.from_dict(data)
                    results.append(result)
                    
                    # Stop if we have enough results
                    if len(results) >= limit:
                        break
                        
                except Exception as e:
                    logger.error(f"Failed to load benchmark result from {file}: {e}")
            
            return results
        except Exception as e:
            logger.error(f"Failed to get benchmark results: {e}")
            return []
    
    def compare_benchmarks(self, result1: BenchmarkResult, result2: BenchmarkResult) -> Dict:
        """
        Compare two benchmark results.
        
        Args:
            result1: First benchmark result
            result2: Second benchmark result
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "benchmark1_id": result1.benchmark_id,
            "benchmark2_id": result2.benchmark_id,
            "metrics": {}
        }
        
        # Get common metrics
        metrics = set(result1.statistics.keys()) & set(result2.statistics.keys())
        
        for metric in metrics:
            stat1 = result1.statistics[metric]
            stat2 = result2.statistics[metric]
            
            mean1 = stat1["mean"]
            mean2 = stat2["mean"]
            
            # Calculate percent change
            if mean1 != 0:
                percent_change = ((mean2 - mean1) / mean1) * 100
            else:
                percent_change = float('inf') if mean2 > 0 else 0
            
            # Determine if the change is an improvement or regression
            is_improvement = False
            
            # For execution time, lower is better
            if metric == "execution_time":
                is_improvement = percent_change < 0
            # For throughput, higher is better
            elif "throughput" in metric:
                is_improvement = percent_change > 0
            # For resource usage, lower is better
            else:
                is_improvement = percent_change < 0
            
            comparison["metrics"][metric] = {
                "mean1": mean1,
                "mean2": mean2,
                "absolute_change": mean2 - mean1,
                "percent_change": percent_change,
                "is_improvement": is_improvement,
                "significant": abs(percent_change) > 5  # Consider changes > 5% significant
            }
        
        return comparison
    
    def get_trend_analysis(self, target: str, metric: str) -> Dict:
        """
        Perform trend analysis on a metric for a target.
        
        Args:
            target: The target component
            metric: The metric to analyze
            
        Returns:
            Dictionary with trend analysis results
        """
        series_name = f"benchmark_{target}"
        metric_name = f"{metric}_mean"  # Use mean value for trend analysis
        
        return self.time_series_db.get_trend_analysis(series_name, metric_name)
    
    def generate_report(self, target: str = None, benchmark_type: str = None) -> Dict:
        """
        Generate a comprehensive report on benchmark results.
        
        Args:
            target: Filter by target component
            benchmark_type: Filter by benchmark type
            
        Returns:
            Dictionary with the report
        """
        report = {
            "generated_at": datetime.datetime.now().isoformat(),
            "targets": {},
            "overall_trends": {},
            "recent_benchmarks": []
        }
        
        try:
            # Get recent benchmark results
            results = self.get_benchmark_results(target, benchmark_type, limit=20)
            
            # Group by target
            targets = {}
            for result in results:
                target = result.config.target
                if target not in targets:
                    targets[target] = []
                targets[target].append(result)
            
            # Process each target
            for target, target_results in targets.items():
                # Sort by timestamp
                target_results.sort(key=lambda r: r.end_time)
                
                # Get latest result
                latest = target_results[-1] if target_results else None
                
                # Get baseline result (first one)
                baseline = target_results[0] if target_results else None
                
                target_report = {
                    "latest_benchmark": latest.benchmark_id if latest else None,
                    "latest_timestamp": latest.end_time.isoformat() if latest else None,
                    "metrics": {},
                    "trends": {}
                }
                
                # Process metrics
                if latest and baseline:
                    comparison = self.compare_benchmarks(baseline, latest)
                    target_report["metrics"] = comparison["metrics"]
                
                # Process trends
                if latest:
                    for metric in latest.statistics.keys():
                        trend = self.get_trend_analysis(target, metric)
                        target_report["trends"][metric] = trend
                
                report["targets"][target] = target_report
            
            # Add recent benchmarks
            for result in results[:10]:
                report["recent_benchmarks"].append({
                    "benchmark_id": result.benchmark_id,
                    "target": result.config.target,
                    "type": result.config.benchmark_type,
                    "timestamp": result.end_time.isoformat(),
                    "success": result.success
                })
            
            return report
        except Exception as e:
            logger.error(f"Failed to generate benchmark report: {e}")
            return report


# Example usage
if __name__ == "__main__":
    # Create benchmarking system
    benchmarking = BenchmarkingSystem("benchmark_data")
    
    # Create a benchmark configuration
    config = BenchmarkConfig(
        name="example_benchmark",
        target="example_component",
        benchmark_type="PERFORMANCE",
        command="echo 'Hello, World!' && sleep 1",
        iterations=3
    )
    
    # Run the benchmark
    result = benchmarking.run_benchmark(config)
    
    # Get trend analysis
    trend = benchmarking.get_trend_analysis("example_component", "execution_time")
    print(f"Execution time trend: {trend['trend']} (slope: {trend['slope']})")
    
    # Generate report
    report = benchmarking.generate_report()
    print(f"Report targets: {list(report['targets'].keys())}")
