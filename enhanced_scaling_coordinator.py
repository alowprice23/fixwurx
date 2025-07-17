#!/usr/bin/env python3
"""
enhanced_scaling_coordinator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enhanced horizontal scaling coordinator for FixWurx.
"""

import time
import logging
import threading
import json
from typing import Dict, List, Any, Optional
from enum import Enum, auto
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EnhancedScalingCoordinator")

class WorkerState(Enum):
    """State of a worker node."""
    IDLE = auto()
    READY = auto()
    BUSY = auto()
    UNAVAILABLE = auto()

class DeploymentMode(Enum):
    """Deployment mode for the scaling coordinator."""
    PRODUCTION = auto()
    SIMULATION = auto()

class EnhancedWorkerNode:
    """
    Represents a worker node in the enhanced scaling system.
    Provides detailed worker information and capabilities.
    """
    
    def __init__(self, worker_id: str, capacity: int = 10, state: WorkerState = WorkerState.READY):
        """Initialize an enhanced worker node."""
        self.worker_id = worker_id
        self.state = state
        self.total_agents = capacity
        self.free_agents = capacity
        self.active_bugs = []
        self.last_heartbeat = time.time()
        self.affinities = {}
        self.languages = {}
        self.metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "network_usage": 0.0
        }
        self.capabilities = []
    
    def update_heartbeat(self):
        """Update the worker's heartbeat timestamp."""
        self.last_heartbeat = time.time()
    
    def is_healthy(self) -> bool:
        """Check if the worker is healthy based on heartbeat."""
        return (time.time() - self.last_heartbeat) < 30  # 30 seconds timeout
    
    def add_bug(self, bug_id: str) -> bool:
        """
        Add a bug to the worker's active bugs.
        
        Args:
            bug_id: The ID of the bug to add
            
        Returns:
            True if the bug was added, False if the worker is full
        """
        if self.free_agents > 0:
            self.active_bugs.append(bug_id)
            self.free_agents -= 1
            return True
        return False
    
    def remove_bug(self, bug_id: str) -> bool:
        """
        Remove a bug from the worker's active bugs.
        
        Args:
            bug_id: The ID of the bug to remove
            
        Returns:
            True if the bug was removed, False if the bug was not found
        """
        if bug_id in self.active_bugs:
            self.active_bugs.remove(bug_id)
            self.free_agents += 1
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert worker to dictionary for serialization."""
        return {
            "worker_id": self.worker_id,
            "state": self.state.name,
            "total_agents": self.total_agents,
            "free_agents": self.free_agents,
            "active_bugs": self.active_bugs,
            "last_heartbeat": self.last_heartbeat,
            "affinities": self.affinities,
            "languages": self.languages,
            "metrics": self.metrics,
            "capabilities": self.capabilities
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedWorkerNode':
        """Create a worker from a dictionary."""
        worker = cls(
            worker_id=data["worker_id"],
            capacity=data["total_agents"],
            state=WorkerState[data["state"]]
        )
        worker.free_agents = data["free_agents"]
        worker.active_bugs = data["active_bugs"]
        worker.last_heartbeat = data["last_heartbeat"]
        worker.affinities = data.get("affinities", {})
        worker.languages = data.get("languages", {})
        worker.metrics = data.get("metrics", {"cpu_usage": 0.0, "memory_usage": 0.0, "network_usage": 0.0})
        worker.capabilities = data.get("capabilities", [])
        return worker

class EnhancedScalingCoordinator:
    """
    Coordinates enhanced horizontal scaling of worker nodes.
    """
    
    def __init__(self, config: Dict[str, Any], resource_manager, advanced_load_balancer, resource_optimizer, state_path: str):
        """Initialize the enhanced scaling coordinator."""
        self.config = config
        self.resource_manager = resource_manager
        self.load_balancer = advanced_load_balancer
        self.optimizer = resource_optimizer
        self.state_path = Path(state_path)
        
        self.workers: Dict[str, Any] = {}
        self.local_worker_id: Optional[str] = None
        self.in_burst_mode = False
        
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._scaling_loop, daemon=True)
        
        self._load_state()
        self._register_local_worker()

    def start(self):
        """Start the scaling coordinator."""
        self._thread.start()
        logger.info("Enhanced Scaling Coordinator started.")

    def stop(self):
        """Stop the scaling coordinator."""
        self._stop_event.set()
        self._thread.join()
        self._save_state()
        logger.info("Enhanced Scaling Coordinator stopped.")

    def _scaling_loop(self):
        """Main scaling loop."""
        while not self._stop_event.is_set():
            self.discover_workers()
            self.check_worker_health()
            self.scale()
            time.sleep(self.config.get("sync_interval_sec", 5))

    def _register_local_worker(self):
        """Register the local worker."""
        self.local_worker_id = f"{self.config.get('worker_prefix', 'worker-')}{int(time.time())}"
        self.workers[self.local_worker_id] = {
            "id": self.local_worker_id,
            "state": WorkerState.READY,
            "last_heartbeat": time.time(),
            "active_bugs": [],
            "total_agents": self.config.get("min_workers", 1) * 10, # Mock capacity
        }
        logger.info(f"Registered local worker: {self.local_worker_id}")

    def discover_workers(self):
        """Discover worker nodes."""
        if self.config.get("discovery_method") == "static":
            for worker_addr in self.config.get("static_workers", []):
                if worker_addr not in self.workers:
                    worker_id = f"{self.config.get('worker_prefix', 'worker-')}{worker_addr}"
                    self.workers[worker_id] = {
                        "id": worker_id,
                        "state": WorkerState.READY,
                        "last_heartbeat": time.time(),
                        "active_bugs": [],
                        "total_agents": 10, # Mock capacity
                    }
                    logger.info(f"Discovered static worker: {worker_id}")

    def check_worker_health(self):
        """Check the health of worker nodes."""
        now = time.time()
        for worker_id, worker in list(self.workers.items()):
            if now - worker["last_heartbeat"] > self.config.get("heartbeat_timeout_sec", 15):
                worker["state"] = WorkerState.UNAVAILABLE
                logger.warning(f"Worker {worker_id} is unavailable.")

    def scale(self):
        """Scale workers up or down based on utilization."""
        metrics = self.optimizer.get_optimization_metrics()
        usage_ratio = metrics["current_usage_ratio"]
        
        if usage_ratio > self.config.get("scale_up_threshold", 0.8):
            self._scale_up()
        elif usage_ratio < self.config.get("scale_down_threshold", 0.3):
            self._scale_down()
            
        if metrics["in_burst_mode"] and not self.in_burst_mode:
            self._enter_burst_mode()
        elif not metrics["in_burst_mode"] and self.in_burst_mode:
            self._exit_burst_mode()

    def _scale_up(self):
        """Scale up the number of workers."""
        if len(self.workers) < self.config.get("max_workers", 5):
            if self.config.get("deployment_mode") == "simulation":
                self._scale_up_simulation()
            else:
                # Production scaling logic would go here
                pass

    def _scale_up_simulation(self):
        """Simulate scaling up a worker."""
        worker_id = f"{self.config.get('worker_prefix', 'sim-')}{int(time.time())}"
        self.workers[worker_id] = {
            "id": worker_id,
            "state": WorkerState.READY,
            "last_heartbeat": time.time(),
            "active_bugs": [],
            "total_agents": 10, # Mock capacity
        }
        logger.info(f"Scaled up simulation worker: {worker_id}")

    def _scale_down(self):
        """Scale down the number of workers."""
        if len(self.workers) > self.config.get("min_workers", 1):
            # Find an idle worker to remove
            for worker_id, worker in list(self.workers.items()):
                if worker["state"] == WorkerState.READY and not worker["active_bugs"] and worker_id != self.local_worker_id:
                    del self.workers[worker_id]
                    logger.info(f"Scaled down worker: {worker_id}")
                    break

    def _enter_burst_mode(self):
        """Enter burst mode."""
        self.in_burst_mode = True
        for worker in self.workers.values():
            worker["total_agents"] = int(worker["total_agents"] * self.config.get("burst_factor", 1.5))
        logger.info("Entered burst mode.")

    def _exit_burst_mode(self):
        """Exit burst mode."""
        self.in_burst_mode = False
        for worker in self.workers.values():
            worker["total_agents"] = int(worker["total_agents"] / self.config.get("burst_factor", 1.5))
        logger.info("Exited burst mode.")

    def register_bug_assignment(self, bug_id: str):
        """Register a bug assignment to the local worker."""
        if self.local_worker_id:
            self.workers[self.local_worker_id]["active_bugs"].append(bug_id)

    def register_bug_completion(self, bug_id: str, success: bool, tokens_used: int):
        """Register a bug completion on the local worker."""
        if self.local_worker_id and bug_id in self.workers[self.local_worker_id]["active_bugs"]:
            self.workers[self.local_worker_id]["active_bugs"].remove(bug_id)

    def get_worker_for_bug(self, bug_id: str) -> Optional[str]:
        """Get the worker assigned to a bug."""
        for worker_id, worker in self.workers.items():
            if bug_id in worker["active_bugs"]:
                return worker_id
        return None

    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get scaling metrics."""
        active_workers = sum(1 for w in self.workers.values() if w["state"] != WorkerState.UNAVAILABLE)
        return {
            "worker_count": len(self.workers),
            "active_workers": active_workers,
            "in_burst_mode": self.in_burst_mode,
        }

    def _save_state(self):
        """Save the coordinator's state to a file."""
        try:
            with open(self.state_path, "w") as f:
                # Convert enums to strings for JSON serialization
                state = {
                    "workers": {
                        worker_id: {**data, "state": data["state"].name}
                        for worker_id, data in self.workers.items()
                    },
                    "local_worker_id": self.local_worker_id,
                    "in_burst_mode": self.in_burst_mode,
                }
                json.dump(state, f)
            logger.info(f"Saved scaling state to {self.state_path}")
        except Exception as e:
            logger.error(f"Failed to save scaling state: {e}")

    def _load_state(self):
        """Load the coordinator's state from a file."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    state = json.load(f)
                    # Convert state strings back to enums
                    self.workers = {
                        worker_id: {**data, "state": WorkerState[data["state"]]}
                        for worker_id, data in state.get("workers", {}).items()
                    }
                    self.local_worker_id = state.get("local_worker_id")
                    self.in_burst_mode = state.get("in_burst_mode", False)
                logger.info(f"Loaded scaling state from {self.state_path}")
            except Exception as e:
                logger.error(f"Failed to load scaling state: {e}")
