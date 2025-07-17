"""
core/scaling_coordinator_fix.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixes for the scaling_coordinator.py module to ensure proper horizontal scaling behavior.
This patch enables proper worker node creation in test environments.
"""

import time
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scaling_fix")

def apply_scaling_fix(scaling_coordinator):
    """
    Apply fixes to the scaling coordinator to ensure it correctly scales in test environments.
    
    Args:
        scaling_coordinator: The ScalingCoordinator instance to fix
    """
    # Store the original scale_up method
    original_scale_up = scaling_coordinator._scale_up
    
    # Define a patched scale_up method
    def patched_scale_up():
        """Patched version of _scale_up that works in test environments without Docker."""
        logger.info("Scaling up: creating new worker node (patched method)")
        
        try:
            # Generate a unique worker ID
            worker_id = f"worker-{int(time.time())}"
            
            # Create a new worker with default configuration
            from scaling_coordinator import WorkerNode, WorkerState
            
            new_worker = WorkerNode(
                worker_id=worker_id,
                hostname=f"worker-{len(scaling_coordinator.workers) + 1}",
                port=8000 + len(scaling_coordinator.workers),
                state=WorkerState.STARTING,
                total_agents=9,  # Default agent pool size
                free_agents=9
            )
            
            # Add to worker registry
            scaling_coordinator.workers[worker_id] = new_worker
            logger.info(f"Added new worker node: {worker_id}")
            
            # In a real implementation, we would wait for the worker to report ready
            # For simulation, just mark it ready after a delay
            time.sleep(1)
            new_worker.state = WorkerState.READY
            
            return True
        except Exception as e:
            logger.error(f"Failed to scale up: {e}")
            return False
    
    # Replace the method
    scaling_coordinator._scale_up = patched_scale_up
    
    # Adjust scaling thresholds to make scaling more aggressive in tests
    scaling_coordinator.SCALE_UP_THRESHOLD = 0.6  # Scale up at 60% utilization
    scaling_coordinator.SCALE_DOWN_THRESHOLD = 0.2  # Scale down at 20% utilization
    
    # Reduce scaling interval for tests
    scaling_coordinator.scaling_interval_sec = 2
    
    logger.info("Applied scaling fixes to coordinator")
    
    return scaling_coordinator
