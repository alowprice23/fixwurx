#!/usr/bin/env python3
"""
resource_manager_extension.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compatibility layer for the extended resource manager for cluster operations
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ClusterResourceManager")

class ClusterResourceManager:
    """
    Manages resources across a cluster of worker nodes.
    """
    
    def __init__(self, 
                base_resource_manager, 
                scaling_coordinator, 
                load_balancer, 
                sync_interval_sec: int = 5):
        """Initialize the cluster resource manager."""
        self.base_resource_manager = base_resource_manager
        self.scaling_coordinator = scaling_coordinator
        self.load_balancer = load_balancer
        self.sync_interval_sec = sync_interval_sec
        self.cluster_profile = {}
        self.running = False
        self.thread = None
        
        logger.info("ClusterResourceManager initialized")
        
    def start(self):
        """Start the cluster resource manager."""
        if not self.running:
            self.running = True
            # Initialize the cluster profile
            self._update_cluster_profile()
            logger.info("ClusterResourceManager started")
            
    def start_sync(self):
        """Start the cluster resource manager with synchronization."""
        self.start()
        # Perform initial synchronization
        self._update_cluster_profile()
        
    def stop(self):
        """Stop the cluster resource manager."""
        if self.running:
            self.running = False
            logger.info("ClusterResourceManager stopped")
            
    def _update_cluster_profile(self):
        """Update the cluster resource profile."""
        # Get worker status from scaling coordinator
        workers = self.scaling_coordinator.workers
        
        # Build the cluster profile
        self.cluster_profile = {
            "total_workers": len(workers),
            "ready_workers": sum(1 for w in workers.values() if isinstance(w, dict) and w.get("state") == "ready"),
            "total_resources": sum(w.get("total_agents", 0) for w in workers.values() if isinstance(w, dict)),
            "used_resources": sum(w.get("total_agents", 0) - w.get("free_agents", 0) 
                              for w in workers.values() if isinstance(w, dict)),
            "last_updated": time.time()
        }
        
        logger.debug(f"Updated cluster profile: {self.cluster_profile}")
        
    def get_cluster_profile(self) -> Dict[str, Any]:
        """Get the current cluster resource profile."""
        if not self.cluster_profile or time.time() - self.cluster_profile.get("last_updated", 0) > self.sync_interval_sec:
            self._update_cluster_profile()
            
        return self.cluster_profile
