#!/usr/bin/env python3
"""
test_scaling_minimal.py
━━━━━━━━━━━━━━━━━━━━━━━
A minimal test for the horizontal scaling capabilities in FixWurx.
"""

import time
import logging
import random
import threading
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("scaling_test")

# Import the components we need to test
from resource_manager import ResourceManager
from scaling_coordinator import ScalingCoordinator 
from load_balancer import LoadBalancer, BalancingStrategy
from resource_manager_extension import ClusterResourceManager

# Test constants
SIMULATED_BUGS = ["BUG-001", "BUG-002", "BUG-003"]


def run_minimal_test():
    """Run a minimal test of the horizontal scaling system."""
    logger.info("Starting minimal horizontal scaling test")
    
    # Initialize components
    base_resource_manager = ResourceManager(total_agents=9)  # Start with 9 agents
    
    # Initialize scaling coordinator
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
    
    try:
        # Start components
        logger.info("Starting scaling components...")
        scaling_coordinator.start()
        load_balancer.start()
        cluster_manager.start_sync()
        
        # Log initial state
        logger.info("Initial state:")
        logger.info(f"  Total agents: {cluster_manager.cluster_state.total_agents}")
        logger.info(f"  Free agents: {cluster_manager.cluster_state.free_agents}")
        logger.info(f"  Workers: {len(scaling_coordinator.workers)}")
        
        # Wait for components to initialize
        logger.info("Waiting for components to initialize...")
        time.sleep(3)
        
        # Log updated state
        logger.info("Updated state:")
        logger.info(f"  Total agents: {cluster_manager.cluster_state.total_agents}")
        logger.info(f"  Free agents: {cluster_manager.cluster_state.free_agents}")
        logger.info(f"  Workers: {len(scaling_coordinator.workers)}")
        
        # Try to allocate some bugs
        logger.info("Allocating bugs...")
        for bug_id in SIMULATED_BUGS:
            if cluster_manager.can_allocate():
                success = cluster_manager.allocate(bug_id)
                if success:
                    worker_id = cluster_manager.get_worker_for_bug(bug_id)
                    logger.info(f"Bug {bug_id} allocated to worker {worker_id}")
                else:
                    logger.warning(f"Failed to allocate bug {bug_id}")
            else:
                logger.warning("Cannot allocate more bugs - insufficient resources")
                break
        
        # Log allocation status
        logger.info("Allocation status:")
        allocation_status = cluster_manager.get_allocation_status()
        logger.info(f"  Total agents: {allocation_status['total_agents']}")
        logger.info(f"  Free agents: {allocation_status['free_agents']}")
        logger.info(f"  Active bugs: {allocation_status['active_bugs']}")
        
        # Free some bugs
        logger.info("Freeing bugs...")
        for bug_id in list(cluster_manager.bug_assignments.keys())[:2]:
            cluster_manager.free_agents(bug_id)
            logger.info(f"Freed bug {bug_id}")
        
        # Final allocation status
        logger.info("Final allocation status:")
        allocation_status = cluster_manager.get_allocation_status()
        logger.info(f"  Total agents: {allocation_status['total_agents']}")
        logger.info(f"  Free agents: {allocation_status['free_agents']}")
        logger.info(f"  Active bugs: {allocation_status['active_bugs']}")
        
        # Test successful
        logger.info("Minimal test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Stop components
        logger.info("Stopping components...")
        cluster_manager.stop_sync()
        load_balancer.stop()
        scaling_coordinator.stop()
        logger.info("Test cleanup complete")


if __name__ == "__main__":
    run_minimal_test()
