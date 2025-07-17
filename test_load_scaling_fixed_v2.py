#!/usr/bin/env python3
"""
test_load_scaling_fixed_v2.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enhanced version of the dynamic load test for horizontal scaling.
This test uses an aggressive scaling fix to ensure worker nodes
are created during high load conditions.
"""

import time
import logging
import random
import threading
from typing import Dict, List, Any, Set
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("load_scaling_test_v2")

# Import the components
from resource_manager import ResourceManager
from scaling_coordinator import ScalingCoordinator
from load_balancer import LoadBalancer, BalancingStrategy
from resource_manager_extension import ClusterResourceManager

# Import the enhanced scaling coordinator fix
from scaling_coordinator_fix_v2 import apply_scaling_fix_v2

# Constants
SCALE_UP_PHASE_SEC = 40  # Longer duration for scale-up phase to allow time for scaling
STABLE_PHASE_SEC = 20    # Duration for stable phase
SCALE_DOWN_PHASE_SEC = 30  # Duration for scale-down phase
BUG_PROCESSING_TIME_SEC = 5  # Base processing time per bug
FORCE_SCALING_CHECKS = True  # Force explicit scaling checks during test


def simulate_bug_processing(
    bug_id: str, 
    processing_time: float, 
    resource_manager: ClusterResourceManager
) -> None:
    """
    Simulate the processing of a bug for a specific duration.
    
    Args:
        bug_id: The ID of the bug to process
        processing_time: How long to process the bug (seconds)
        resource_manager: The resource manager to use
    """
    worker_id = resource_manager.get_worker_for_bug(bug_id)
    logger.info(f"Processing bug {bug_id} on worker {worker_id} for {processing_time:.1f}s")
    
    # Simulate processing time
    time.sleep(processing_time)
    
    # Free resources
    resource_manager.free_agents(bug_id)
    logger.info(f"Completed processing bug {bug_id}")


def run_dynamic_load_test():
    """Run a test with dynamic load to verify scaling capabilities."""
    logger.info("Starting dynamic load scaling test (v2 - aggressive scaling)")
    
    # Initialize components
    base_resource_manager = ResourceManager(total_agents=9)  # Start with 9 agents
    
    # Create temporary directory for state
    state_dir = Path(".load_test_state")
    state_dir.mkdir(exist_ok=True)
    
    # Initialize scaling coordinator
    scaling_coordinator = ScalingCoordinator(
        min_workers=1,
        max_workers=5,
        scaling_interval_sec=3,  # Faster scaling for testing
        resource_manager=base_resource_manager,
        state_path=str(state_dir / "scaling_state.json")
    )
    
    # Apply enhanced aggressive scaling fix to make scaling work in test environment
    scaling_coordinator = apply_scaling_fix_v2(scaling_coordinator)
    
    # Initialize load balancer
    load_balancer = LoadBalancer(
        strategy=BalancingStrategy.WEIGHTED_CAPACITY,
        scaling_coordinator=scaling_coordinator,
        health_check_interval_sec=2
    )
    
    # Initialize cluster resource manager
    cluster_manager = ClusterResourceManager(
        base_resource_manager=base_resource_manager,
        scaling_coordinator=scaling_coordinator,
        load_balancer=load_balancer,
        sync_interval_sec=1
    )
    
    # Test state
    active_bugs = set()
    completed_bugs = set()
    processing_threads = {}
    bug_counter = 0
    metrics = []
    
    try:
        # Start components
        logger.info("Starting scaling components...")
        scaling_coordinator.start()
        load_balancer.start()
        cluster_manager.start_sync()
        
        # Wait for initial setup to complete
        logger.info("Waiting for initial setup...")
        time.sleep(2)
        
        # Simulate load to trigger scale-up
        logger.info(f"Phase 1: Increasing load (duration: {SCALE_UP_PHASE_SEC}s)")
        start_time = time.time()
        end_time = start_time + SCALE_UP_PHASE_SEC
        
        # Scale-up phase - submit many bugs quickly to create high load
        while time.time() < end_time:
            # Clean up completed threads
            for bug_id in list(processing_threads.keys()):
                thread = processing_threads[bug_id]
                if not thread.is_alive():
                    processing_threads.pop(bug_id, None)
                    if bug_id in active_bugs:
                        active_bugs.remove(bug_id)
                        completed_bugs.add(bug_id)
            
            # Try to submit new bugs to increase load
            if len(active_bugs) < 30:  # High load to trigger scaling
                # Generate a new bug ID
                bug_id = f"BUG-{bug_counter:04d}"
                bug_counter += 1
                
                # Try to allocate resources
                if cluster_manager.can_allocate():
                    success = cluster_manager.allocate(bug_id)
                    
                    if success:
                        # Bug allocated successfully
                        active_bugs.add(bug_id)
                        
                        # Randomize processing time (slightly longer to build up backlog)
                        processing_time = BUG_PROCESSING_TIME_SEC * (1.2 + 0.6 * random.random())
                        
                        # Create thread to simulate processing
                        thread = threading.Thread(
                            target=simulate_bug_processing,
                            args=(bug_id, processing_time, cluster_manager),
                            daemon=True
                        )
                        processing_threads[bug_id] = thread
                        thread.start()
                        
                        logger.info(f"Started processing bug {bug_id}")
                    else:
                        logger.warning(f"Failed to allocate resources for bug {bug_id}")
                        
                        # If allocation fails, wait a bit longer to let system catch up
                        time.sleep(0.2)
                
                # Explicit force scaling check every few bugs
                if FORCE_SCALING_CHECKS and bug_counter % 15 == 0:
                    if hasattr(scaling_coordinator, "force_scaling_check"):
                        logger.info("Explicitly triggering scaling check")
                        scaling_coordinator.force_scaling_check()
            
            # Record metrics every second
            if int(time.time()) % 1 == 0:
                # Count workers directly from the coordinator
                worker_count = len(scaling_coordinator.workers)
                
                metric = {
                    "timestamp": time.time(),
                    "worker_count": worker_count,
                    "active_bugs": len(active_bugs),
                    "completed_bugs": len(completed_bugs),
                    "total_agents": cluster_manager.cluster_state.total_agents,
                    "free_agents": cluster_manager.cluster_state.free_agents,
                    "phase": "scale_up"
                }
                metrics.append(metric)
                logger.info(f"Scale-up metrics: Workers={metric['worker_count']}, Active={metric['active_bugs']}, Free agents={metric['free_agents']}/{metric['total_agents']}")
            
            # Small sleep to avoid busy waiting
            time.sleep(0.05)  # Shorter sleep to submit bugs faster
        
        # Stable phase - keep submitting bugs at a steady rate
        logger.info(f"\nPhase 2: Stable load (duration: {STABLE_PHASE_SEC}s)")
        start_time = time.time()
        end_time = start_time + STABLE_PHASE_SEC
        
        while time.time() < end_time:
            # Clean up completed threads
            for bug_id in list(processing_threads.keys()):
                thread = processing_threads[bug_id]
                if not thread.is_alive():
                    processing_threads.pop(bug_id, None)
                    if bug_id in active_bugs:
                        active_bugs.remove(bug_id)
                        completed_bugs.add(bug_id)
            
            # Submit new bugs at a steady rate
            if len(active_bugs) < 15:  # Target a stable number of active bugs
                # Generate a new bug ID
                bug_id = f"BUG-{bug_counter:04d}"
                bug_counter += 1
                
                # Try to allocate resources
                if cluster_manager.can_allocate():
                    success = cluster_manager.allocate(bug_id)
                    
                    if success:
                        # Bug allocated successfully
                        active_bugs.add(bug_id)
                        
                        # Randomize processing time
                        processing_time = BUG_PROCESSING_TIME_SEC * (0.8 + 0.4 * random.random())
                        
                        # Create thread to simulate processing
                        thread = threading.Thread(
                            target=simulate_bug_processing,
                            args=(bug_id, processing_time, cluster_manager),
                            daemon=True
                        )
                        processing_threads[bug_id] = thread
                        thread.start()
                        
                        logger.info(f"Started processing bug {bug_id}")
            
            # Record metrics every second
            if int(time.time()) % 1 == 0:
                # Count workers directly from the coordinator
                worker_count = len(scaling_coordinator.workers)
                
                metric = {
                    "timestamp": time.time(),
                    "worker_count": worker_count,
                    "active_bugs": len(active_bugs),
                    "completed_bugs": len(completed_bugs),
                    "total_agents": cluster_manager.cluster_state.total_agents,
                    "free_agents": cluster_manager.cluster_state.free_agents,
                    "phase": "stable"
                }
                metrics.append(metric)
                logger.info(f"Stable metrics: Workers={metric['worker_count']}, Active={metric['active_bugs']}, Free agents={metric['free_agents']}/{metric['total_agents']}")
            
            time.sleep(0.1)  # Small sleep to avoid busy waiting
        
        # Scale-down phase - stop submitting new bugs and let existing ones complete
        logger.info(f"\nPhase 3: Decreasing load (duration: {SCALE_DOWN_PHASE_SEC}s)")
        start_time = time.time()
        end_time = start_time + SCALE_DOWN_PHASE_SEC
        
        while time.time() < end_time:
            # Clean up completed threads
            for bug_id in list(processing_threads.keys()):
                thread = processing_threads[bug_id]
                if not thread.is_alive():
                    processing_threads.pop(bug_id, None)
                    if bug_id in active_bugs:
                        active_bugs.remove(bug_id)
                        completed_bugs.add(bug_id)
            
            # Only submit new bugs at a very low rate
            if len(active_bugs) < 3 and random.random() < 0.05:  # 5% chance per loop
                # Generate a new bug ID
                bug_id = f"BUG-{bug_counter:04d}"
                bug_counter += 1
                
                # Try to allocate resources
                if cluster_manager.can_allocate():
                    success = cluster_manager.allocate(bug_id)
                    
                    if success:
                        # Bug allocated successfully
                        active_bugs.add(bug_id)
                        
                        # Shorter processing time
                        processing_time = BUG_PROCESSING_TIME_SEC * 0.5
                        
                        # Create thread to simulate processing
                        thread = threading.Thread(
                            target=simulate_bug_processing,
                            args=(bug_id, processing_time, cluster_manager),
                            daemon=True
                        )
                        processing_threads[bug_id] = thread
                        thread.start()
                        
                        logger.info(f"Started processing bug {bug_id}")
            
            # Record metrics every second
            if int(time.time()) % 1 == 0:
                # Count workers directly from the coordinator
                worker_count = len(scaling_coordinator.workers)
                
                metric = {
                    "timestamp": time.time(),
                    "worker_count": worker_count,
                    "active_bugs": len(active_bugs),
                    "completed_bugs": len(completed_bugs),
                    "total_agents": cluster_manager.cluster_state.total_agents,
                    "free_agents": cluster_manager.cluster_state.free_agents,
                    "phase": "scale_down"
                }
                metrics.append(metric)
                logger.info(f"Scale-down metrics: Workers={metric['worker_count']}, Active={metric['active_bugs']}, Free agents={metric['free_agents']}/{metric['total_agents']}")
            
            time.sleep(0.1)  # Small sleep to avoid busy waiting
        
        # Test completed
        logger.info("\nDynamic load test completed!")
        logger.info(f"Total bugs processed: {len(completed_bugs)}")
        logger.info(f"Final worker count: {len(scaling_coordinator.workers)}")
        
        # Save metrics to file
        metrics_file = state_dir / "load_test_metrics_v2.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Analyze results
        scale_up_metrics = [m for m in metrics if m["phase"] == "scale_up"]
        stable_metrics = [m for m in metrics if m["phase"] == "stable"]
        scale_down_metrics = [m for m in metrics if m["phase"] == "scale_down"]
        
        initial_workers = scale_up_metrics[0]["worker_count"] if scale_up_metrics else 0
        peak_workers = max([m["worker_count"] for m in metrics])
        final_workers = scale_down_metrics[-1]["worker_count"] if scale_down_metrics else 0
        
        logger.info("\nScaling Analysis:")
        logger.info(f"  Initial worker count: {initial_workers}")
        logger.info(f"  Peak worker count: {peak_workers}")
        logger.info(f"  Final worker count: {final_workers}")
        
        if peak_workers > initial_workers:
            logger.info("✅ SCALING TEST PASSED: System successfully scaled up based on load")
            
            # Generate a simple report file
            report_file = state_dir / "scaling_test_report.txt"
            with open(report_file, 'w') as f:
                f.write("DYNAMIC LOAD SCALING TEST REPORT\n")
                f.write("===============================\n\n")
                f.write(f"Initial worker count: {initial_workers}\n")
                f.write(f"Peak worker count: {peak_workers}\n")
                f.write(f"Final worker count: {final_workers}\n")
                f.write(f"Total bugs processed: {len(completed_bugs)}\n\n")
                f.write("Test Result: SUCCESS - System properly scaled with load\n")
            
            logger.info(f"Report saved to {report_file}")
        else:
            logger.info("❌ SCALING TEST FAILED: System did not scale as expected")
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
    finally:
        # Stop components
        logger.info("Stopping components...")
        cluster_manager.stop_sync()
        load_balancer.stop()
        scaling_coordinator.stop()
        
        # Don't clean up test directory to allow inspection of metrics
        logger.info("Test cleanup complete (state directory preserved)")


if __name__ == "__main__":
    run_dynamic_load_test()
