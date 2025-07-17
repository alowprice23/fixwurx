#!/usr/bin/env python3
"""
test_horizontal_scaling.py
━━━━━━━━━━━━━━━━━━━━━━━━━
Test script demonstrating horizontal scaling capabilities in FixWurx.

This script showcases:
1. Dynamic worker node allocation via ScalingCoordinator
2. Intelligent load balancing across worker nodes
3. Cluster-aware resource management
4. Fault tolerance with worker node failures

Together, these components enable horizontal scaling for processing
multiple bugs in parallel across a cluster of worker nodes.
"""

import time
import logging
import threading
import random
import argparse
from typing import Dict, List, Any
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_scaling")

# Import the components
from resource_manager import ResourceManager
from scaling_coordinator import ScalingCoordinator, WorkerState
from load_balancer import LoadBalancer, BalancingStrategy
from resource_manager_extension import ClusterResourceManager

# Simulated bugs for testing
SIMULATED_BUGS = [
    "BUG-1001", "BUG-1002", "BUG-1003", "BUG-1004", "BUG-1005",
    "BUG-1006", "BUG-1007", "BUG-1008", "BUG-1009", "BUG-1010"
]

# Test constants
DEFAULT_TEST_DURATION_SEC = 60
DEFAULT_BUG_PROCESSING_TIME_SEC = 10
DEFAULT_WORKER_FAILURE_PROBABILITY = 0.05  # 5% chance of worker failure per tick


def simulate_bug_processing(
    bug_id: str,
    processing_time_sec: float,
    resource_manager: ClusterResourceManager
) -> None:
    """
    Simulate processing a bug for a certain amount of time.
    
    Args:
        bug_id: The bug ID to process
        processing_time_sec: How long to process the bug
        resource_manager: The cluster resource manager
    """
    # Get the worker for this bug
    worker_id = resource_manager.get_worker_for_bug(bug_id)
    if not worker_id:
        logger.error(f"Bug {bug_id} not assigned to any worker")
        return
        
    logger.info(f"Processing bug {bug_id} on worker {worker_id} for {processing_time_sec:.1f} seconds")
    
    # Simulate processing time
    time.sleep(processing_time_sec)
    
    # Free resources
    resource_manager.free_agents(bug_id)
    logger.info(f"Completed processing bug {bug_id}")


def run_scaling_test(
    test_duration_sec: int = DEFAULT_TEST_DURATION_SEC,
    bug_processing_time_sec: float = DEFAULT_BUG_PROCESSING_TIME_SEC,
    worker_failure_probability: float = DEFAULT_WORKER_FAILURE_PROBABILITY,
    load_balancing_strategy: BalancingStrategy = BalancingStrategy.WEIGHTED_CAPACITY,
    enable_worker_failures: bool = True,
    log_metrics_interval_sec: int = 5
) -> Dict[str, Any]:
    """
    Run the horizontal scaling test.
    
    Args:
        test_duration_sec: How long to run the test (seconds)
        bug_processing_time_sec: How long each bug takes to process (seconds)
        worker_failure_probability: Probability of worker failure per tick
        load_balancing_strategy: Strategy to use for load balancing
        enable_worker_failures: Whether to simulate worker failures
        log_metrics_interval_sec: How often to log metrics (seconds)
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Starting horizontal scaling test with strategy: {load_balancing_strategy.value}")
    logger.info(f"Test will run for {test_duration_sec} seconds")
    
    # Create state directory
    state_dir = Path(".test_scaling_state")
    state_dir.mkdir(exist_ok=True)
    
    # Initialize components
    base_resource_manager = ResourceManager()
    
    scaling_coordinator = ScalingCoordinator(
        min_workers=1,
        max_workers=5,
        scaling_interval_sec=5,
        resource_manager=base_resource_manager,
        state_path=str(state_dir / "scaling_state.json")
    )
    
    load_balancer = LoadBalancer(
        strategy=load_balancing_strategy,
        scaling_coordinator=scaling_coordinator,
        health_check_interval_sec=5
    )
    
    cluster_manager = ClusterResourceManager(
        base_resource_manager=base_resource_manager,
        scaling_coordinator=scaling_coordinator,
        load_balancer=load_balancer,
        sync_interval_sec=2
    )
    
    # Start components
    scaling_coordinator.start()
    load_balancer.start()
    cluster_manager.start_sync()
    
    # Test state
    start_time = time.time()
    end_time = start_time + test_duration_sec
    active_bugs = set()
    completed_bugs = set()
    failed_bugs = set()
    processing_threads = {}
    metrics_history = []
    
    # Last metrics log time
    last_metrics_time = start_time
    
    try:
        # Main test loop
        while time.time() < end_time:
            current_time = time.time()
            
            # Check if we should log metrics
            if current_time - last_metrics_time >= log_metrics_interval_sec:
                metrics = log_test_metrics(
                    scaling_coordinator, 
                    load_balancer, 
                    cluster_manager,
                    active_bugs,
                    completed_bugs,
                    failed_bugs
                )
                metrics_history.append(metrics)
                last_metrics_time = current_time
            
            # Simulate worker failures (if enabled)
            if enable_worker_failures:
                simulate_worker_failures(scaling_coordinator, worker_failure_probability)
            
            # Clean up completed bug threads
            for bug_id in list(processing_threads.keys()):
                if not processing_threads[bug_id].is_alive():
                    processing_threads.pop(bug_id, None)
            
            # Try to allocate new bugs if we have capacity
            if len(active_bugs) < len(SIMULATED_BUGS) and cluster_manager.can_allocate():
                # Find a bug that isn't active or completed
                available_bugs = [
                    bug_id for bug_id in SIMULATED_BUGS
                    if bug_id not in active_bugs and bug_id not in completed_bugs
                ]
                
                if available_bugs:
                    bug_id = available_bugs[0]
                    
                    # Allocate resources
                    success = cluster_manager.allocate(bug_id)
                    
                    if success:
                        # Start processing thread
                        active_bugs.add(bug_id)
                        
                        # Randomize processing time slightly
                        processing_time = bug_processing_time_sec * (0.8 + 0.4 * random.random())
                        
                        thread = threading.Thread(
                            target=simulate_bug_processing,
                            args=(bug_id, processing_time, cluster_manager),
                            daemon=True,
                            name=f"bug-{bug_id}"
                        )
                        processing_threads[bug_id] = thread
                        thread.start()
                        
                        logger.info(f"Started processing bug {bug_id}")
                    else:
                        logger.warning(f"Failed to allocate resources for bug {bug_id}")
            
            # Check for completed bugs (no longer in cluster manager and not in failed)
            for bug_id in list(active_bugs):
                if not cluster_manager.has(bug_id):
                    active_bugs.remove(bug_id)
                    if bug_id in processing_threads and not processing_threads[bug_id].is_alive():
                        completed_bugs.add(bug_id)
                        logger.info(f"Bug {bug_id} completed successfully")
                    else:
                        failed_bugs.add(bug_id)
                        logger.warning(f"Bug {bug_id} failed or was reassigned")
            
            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        # Stop components
        cluster_manager.stop_sync()
        load_balancer.stop()
        scaling_coordinator.stop()
        
        # Final metrics
        final_metrics = log_test_metrics(
            scaling_coordinator, 
            load_balancer, 
            cluster_manager,
            active_bugs,
            completed_bugs,
            failed_bugs,
            is_final=True
        )
        metrics_history.append(final_metrics)
        
        # Calculate test results
        test_duration = time.time() - start_time
        
        results = {
            "test_duration_sec": test_duration,
            "strategy": load_balancing_strategy.value,
            "bugs_completed": len(completed_bugs),
            "bugs_failed": len(failed_bugs),
            "bugs_active": len(active_bugs),
            "peak_worker_count": max(m["worker_count"] for m in metrics_history),
            "final_metrics": final_metrics,
            "metrics_history": metrics_history
        }
        
        # Write results to file
        results_file = state_dir / f"test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Test results saved to {results_file}")
        
        # Clean up state directory
        try:
            import shutil
            shutil.rmtree(state_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up state directory: {e}")
        
        return results


def simulate_worker_failures(
    scaling_coordinator: ScalingCoordinator,
    failure_probability: float
) -> None:
    """
    Simulate random worker failures.
    
    Args:
        scaling_coordinator: The scaling coordinator
        failure_probability: Probability of worker failure (0.0-1.0)
    """
    # Skip if probability is zero
    if failure_probability <= 0:
        return
        
    # Get all workers except local
    workers = [
        worker_id for worker_id in scaling_coordinator.workers
        if worker_id != scaling_coordinator.local_worker_id
    ]
    
    # Skip if no remote workers
    if not workers:
        return
        
    # Check each worker for failure
    for worker_id in workers:
        if random.random() < failure_probability:
            worker = scaling_coordinator.workers.get(worker_id)
            if worker and worker.state == WorkerState.READY:
                logger.warning(f"Simulating failure of worker {worker_id}")
                # In a real implementation, we would trigger a failure
                # For simulation, just remove the worker
                scaling_coordinator.workers.pop(worker_id, None)
                return  # Only fail one worker at a time


def log_test_metrics(
    scaling_coordinator: ScalingCoordinator,
    load_balancer: LoadBalancer,
    cluster_manager: ClusterResourceManager,
    active_bugs: set,
    completed_bugs: set,
    failed_bugs: set,
    is_final: bool = False
) -> Dict[str, Any]:
    """
    Log current test metrics.
    
    Args:
        scaling_coordinator: The scaling coordinator
        load_balancer: The load balancer
        cluster_manager: The cluster resource manager
        active_bugs: Set of currently active bugs
        completed_bugs: Set of completed bugs
        failed_bugs: Set of failed bugs
        is_final: Whether this is the final metrics log
        
    Returns:
        Dictionary with metrics
    """
    # Gather metrics
    worker_count = len(scaling_coordinator.workers)
    cluster_capacity = scaling_coordinator.get_cluster_capacity()
    allocation_status = cluster_manager.get_allocation_status()
    balancer_metrics = load_balancer.get_balancer_metrics()
    
    # Create metrics dict
    metrics = {
        "timestamp": time.time(),
        "worker_count": worker_count,
        "cluster_capacity": cluster_capacity,
        "total_agents": allocation_status["total_agents"],
        "free_agents": allocation_status["free_agents"],
        "active_bugs": len(active_bugs),
        "completed_bugs": len(completed_bugs),
        "failed_bugs": len(failed_bugs),
        "total_bugs": len(active_bugs) + len(completed_bugs) + len(failed_bugs),
        "balancer_metrics": balancer_metrics
    }
    
    # Log the metrics
    prefix = "FINAL METRICS:" if is_final else "METRICS:"
    logger.info(f"{prefix} Workers={worker_count}, Active={len(active_bugs)}, Completed={len(completed_bugs)}, Failed={len(failed_bugs)}")
    logger.info(f"  Resource usage: {allocation_status['total_agents']} total agents, {allocation_status['free_agents']} free agents")
    
    return metrics


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test horizontal scaling in FixWurx")
    parser.add_argument(
        "--duration", type=int, default=DEFAULT_TEST_DURATION_SEC,
        help=f"Test duration in seconds (default: {DEFAULT_TEST_DURATION_SEC})"
    )
    parser.add_argument(
        "--processing-time", type=float, default=DEFAULT_BUG_PROCESSING_TIME_SEC,
        help=f"Bug processing time in seconds (default: {DEFAULT_BUG_PROCESSING_TIME_SEC})"
    )
    parser.add_argument(
        "--failure-prob", type=float, default=DEFAULT_WORKER_FAILURE_PROBABILITY,
        help=f"Worker failure probability (default: {DEFAULT_WORKER_FAILURE_PROBABILITY})"
    )
    parser.add_argument(
        "--no-failures", action="store_true",
        help="Disable worker failures"
    )
    parser.add_argument(
        "--strategy", type=str, default="weighted_capacity",
        choices=["round_robin", "least_connections", "weighted_capacity", "random"],
        help="Load balancing strategy (default: weighted_capacity)"
    )
    parser.add_argument(
        "--metrics-interval", type=int, default=5,
        help="Metrics logging interval in seconds (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Convert strategy string to enum
    strategy_map = {
        "round_robin": BalancingStrategy.ROUND_ROBIN,
        "least_connections": BalancingStrategy.LEAST_CONNECTIONS,
        "weighted_capacity": BalancingStrategy.WEIGHTED_CAPACITY,
        "random": BalancingStrategy.RANDOM
    }
    strategy = strategy_map[args.strategy]
    
    # Run the test
    run_scaling_test(
        test_duration_sec=args.duration,
        bug_processing_time_sec=args.processing_time,
        worker_failure_probability=args.failure_prob,
        load_balancing_strategy=strategy,
        enable_worker_failures=not args.no_failures,
        log_metrics_interval_sec=args.metrics_interval
    )


if __name__ == "__main__":
    main()
