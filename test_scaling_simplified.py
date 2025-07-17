#!/usr/bin/env python3
"""
test_scaling_simplified.py
━━━━━━━━━━━━━━━━━━━━━━━━━
A simplified test for the horizontal scaling capabilities in FixWurx.
This script runs a basic test of the scaling coordinator, load balancer,
and cluster resource manager without requiring external dependencies.
"""

import time
import logging
import random
import threading
from typing import Dict, List, Any, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("scaling_test")

# Import the components we need to test
from resource_manager import ResourceManager, AGENTS_PER_BUG
from scaling_coordinator import ScalingCoordinator, WorkerState, WorkerNode
from load_balancer import LoadBalancer, BalancingStrategy
from resource_manager_extension import ClusterResourceManager

# Test constants
TEST_DURATION_SEC = 20
BUG_PROCESSING_TIME_SEC = 2
SIMULATED_BUGS = ["BUG-001", "BUG-002", "BUG-003", "BUG-004", "BUG-005"]


def simulate_worker_task(bug_id: str, processing_time: float, resource_manager: ClusterResourceManager):
    """Simulate a worker processing a bug."""
    worker_id = resource_manager.get_worker_for_bug(bug_id)
    logger.info(f"Processing bug {bug_id} on worker {worker_id}")
    
    # Simulate processing time
    time.sleep(processing_time)
    
    # Free resources
    resource_manager.free_agents(bug_id)
    logger.info(f"Completed processing bug {bug_id}")


def run_simple_test():
    """Run a simple test of the horizontal scaling system."""
    logger.info("Starting simplified horizontal scaling test")
    
    # Initialize components
    base_resource_manager = ResourceManager(total_agents=9)  # Start with 9 agents
    
    # Initialize scaling coordinator with no external dependencies
    scaling_coordinator = ScalingCoordinator(
        min_workers=1,
        max_workers=3,
        scaling_interval_sec=2,
        resource_manager=base_resource_manager
    )
    
    # Initialize load balancer
    load_balancer = LoadBalancer(
        strategy=BalancingStrategy.WEIGHTED_CAPACITY,
        scaling_coordinator=scaling_coordinator,
        health_check_interval_sec=1
    )
    
    # Initialize cluster resource manager
    cluster_manager = ClusterResourceManager(
        base_resource_manager=base_resource_manager,
        scaling_coordinator=scaling_coordinator,
        load_balancer=load_balancer,
        sync_interval_sec=1
    )
    
    # Start components
    scaling_coordinator.start()
    load_balancer.start()
    cluster_manager.start_sync()
    
    # Test state
    active_bugs = set()
    completed_bugs = set()
    processing_threads = {}
    
    try:
        logger.info("Initial state:")
        logger.info(f"  Total agents: {cluster_manager.cluster_state.total_agents}")
        logger.info(f"  Free agents: {cluster_manager.cluster_state.free_agents}")
        logger.info(f"  Workers: {len(scaling_coordinator.workers)}")
        
        # Manually add some simulated workers for testing
        logger.info("Adding simulated worker nodes...")
        for i in range(2):
            worker_id = f"simulated-worker-{i+1}"
            scaling_coordinator.workers[worker_id] = WorkerNode(
                worker_id=worker_id,
                hostname=f"worker-{i+1}",
                port=8000 + i,
                state=WorkerState.READY,
                total_agents=9,
                free_agents=9
            )
        
        # Wait for state to propagate
        time.sleep(2)
        
        logger.info("Updated state after adding workers:")
        logger.info(f"  Total agents: {cluster_manager.cluster_state.total_agents}")
        logger.info(f"  Free agents: {cluster_manager.cluster_state.free_agents}")
        logger.info(f"  Workers: {len(scaling_coordinator.workers)}")
        
        start_time = time.time()
        end_time = start_time + TEST_DURATION_SEC
        
        logger.info(f"Starting bug processing test (duration: {TEST_DURATION_SEC}s)...")
        
        # Main test loop
        while time.time() < end_time:
            # Clean up completed threads
            for bug_id in list(processing_threads.keys()):
                thread = processing_threads[bug_id]
                if not thread.is_alive():
                    processing_threads.pop(bug_id, None)
            
            # Try to allocate new bugs if capacity available
            if len(active_bugs) < len(SIMULATED_BUGS) and cluster_manager.can_allocate():
                # Find an available bug
                available_bugs = [
                    bug_id for bug_id in SIMULATED_BUGS
                    if bug_id not in active_bugs and bug_id not in completed_bugs
                ]
                
                if available_bugs:
                    bug_id = available_bugs[0]
                    
                    # Try to allocate resources
                    success = cluster_manager.allocate(bug_id)
                    
                    if success:
                        # Start processing
                        active_bugs.add(bug_id)
                        
                        # Randomize processing time
                        processing_time = BUG_PROCESSING_TIME_SEC * (0.8 + 0.4 * random.random())
                        
                        # Create thread
                        thread = threading.Thread(
                            target=simulate_worker_task,
                            args=(bug_id, processing_time, cluster_manager),
                            daemon=True
                        )
                        processing_threads[bug_id] = thread
                        thread.start()
                        
                        logger.info(f"Started bug {bug_id} processing")
                    else:
                        logger.warning(f"Failed to allocate resources for bug {bug_id}")
            
            # Check for completed bugs
            for bug_id in list(active_bugs):
                if not cluster_manager.has(bug_id):
                    active_bugs.remove(bug_id)
                    if bug_id in processing_threads and not processing_threads[bug_id].is_alive():
                        completed_bugs.add(bug_id)
                        logger.info(f"Bug {bug_id} completed successfully")
            
            # Log current state every few seconds
            if int(time.time()) % 5 == 0:
                worker_info = []
                for worker_id, worker in scaling_coordinator.workers.items():
                    metrics = load_balancer.worker_metrics.get(worker_id, None)
                    if metrics:
                        worker_info.append(
                            f"{worker_id}: {metrics.current_load}/{metrics.capacity} used"
                        )
                    else:
                        worker_info.append(f"{worker_id}: no metrics")
                
                logger.info(f"Current state: Active={len(active_bugs)}, Completed={len(completed_bugs)}")
                logger.info(f"  Workers: {', '.join(worker_info)}")
                logger.info(f"  Resource usage: {cluster_manager.cluster_state.total_agents - cluster_manager.cluster_state.free_agents}/{cluster_manager.cluster_state.total_agents}")
                
                # Only log once per second
                time.sleep(1)
            else:
                # Small sleep to avoid busy waiting
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        # Stop components
        cluster_manager.stop_sync()
        load_balancer.stop()
        scaling_coordinator.stop()
        
        # Final state
        logger.info("\nFinal state:")
        logger.info(f"  Active bugs: {len(active_bugs)}")
        logger.info(f"  Completed bugs: {len(completed_bugs)}")
        logger.info(f"  Total bugs processed: {len(active_bugs) + len(completed_bugs)}")
        logger.info(f"  Workers: {len(scaling_coordinator.workers)}")
        
        # Bug assignments
        logger.info("\nBug assignments:")
        for bug_id, worker_id in cluster_manager.bug_assignments.items():
            logger.info(f"  {bug_id} -> {worker_id}")
        
        # Worker metrics
        logger.info("\nWorker metrics:")
        for metrics in load_balancer.get_worker_metrics():
            logger.info(f"  {metrics['worker_id']}: {metrics['current_load']}/{metrics['capacity']} used, health={metrics['health_score']:.2f}")


if __name__ == "__main__":
    run_simple_test()
