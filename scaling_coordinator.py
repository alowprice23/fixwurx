"""
core/scaling_coordinator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
Implements horizontal scaling for FixWurx execution, allowing dynamic adjustment 
of agent resources based on workload and system capacity.

Key capabilities:
1. Dynamic worker node allocation and management
2. Cluster-aware resource allocation
3. Load balancing across multiple worker nodes
4. Automatic scaling based on workload metrics

Integrates with ResourceManager and ParallelExecutor to provide seamless 
horizontal scaling for bug processing.
"""

import asyncio
import logging
import time
import os
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import socket
from pathlib import Path
import threading

# Optional third-party imports (with graceful fallbacks)
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

try:
    import docker
    HAVE_DOCKER = True
except ImportError:
    HAVE_DOCKER = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scaling_coordinator")

# Constants
DEFAULT_MIN_WORKERS = 1
DEFAULT_MAX_WORKERS = 8
DEFAULT_WORKER_MEMORY_MB = 2048
DEFAULT_WORKER_CPU_CORES = 2
DEFAULT_SCALING_INTERVAL_SEC = 60
DEFAULT_SCALE_UP_THRESHOLD = 0.8  # 80% resource utilization
DEFAULT_SCALE_DOWN_THRESHOLD = 0.3  # 30% resource utilization

# Configuration from environment variables (with defaults)
MIN_WORKERS = int(os.environ.get("FIXWURX_MIN_WORKERS", DEFAULT_MIN_WORKERS))
MAX_WORKERS = int(os.environ.get("FIXWURX_MAX_WORKERS", DEFAULT_MAX_WORKERS))
WORKER_MEMORY_MB = int(os.environ.get("FIXWURX_WORKER_MEMORY_MB", DEFAULT_WORKER_MEMORY_MB))
WORKER_CPU_CORES = int(os.environ.get("FIXWURX_WORKER_CPU_CORES", DEFAULT_WORKER_CPU_CORES))
SCALING_INTERVAL_SEC = int(os.environ.get("FIXWURX_SCALING_INTERVAL_SEC", DEFAULT_SCALING_INTERVAL_SEC))
SCALE_UP_THRESHOLD = float(os.environ.get("FIXWURX_SCALE_UP_THRESHOLD", DEFAULT_SCALE_UP_THRESHOLD))
SCALE_DOWN_THRESHOLD = float(os.environ.get("FIXWURX_SCALE_DOWN_THRESHOLD", DEFAULT_SCALE_DOWN_THRESHOLD))


class WorkerState(Enum):
    """Possible states for a worker node."""
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    STOPPING = "stopping"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class WorkerNode:
    """Represents a worker node in the cluster."""
    worker_id: str
    hostname: str
    port: int
    state: WorkerState = WorkerState.STARTING
    total_agents: int = 0
    free_agents: int = 0
    cpu_cores: int = WORKER_CPU_CORES
    memory_mb: int = WORKER_MEMORY_MB
    last_heartbeat: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def is_available(self) -> bool:
        """Check if this worker is available to handle new tasks."""
        return self.state == WorkerState.READY and self.free_agents > 0
    
    def utilization(self) -> float:
        """Calculate resource utilization as a percentage (0.0-1.0)."""
        if self.total_agents == 0:
            return 0.0
        return 1.0 - (self.free_agents / self.total_agents)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "port": self.port,
            "state": self.state.value,
            "total_agents": self.total_agents,
            "free_agents": self.free_agents,
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "last_heartbeat": self.last_heartbeat,
            "metrics": self.metrics
        }


class ScalingCoordinator:
    """
    Manages horizontal scaling of FixWurx worker nodes based on current load and capacity.
    
    This class:
    1. Maintains a registry of active worker nodes
    2. Monitors system load and resource utilization
    3. Makes scaling decisions based on configured thresholds
    4. Coordinates with resource manager to adjust agent pool size
    """
    
    def __init__(
        self,
        min_workers: int = MIN_WORKERS,
        max_workers: int = MAX_WORKERS,
        scaling_interval_sec: int = SCALING_INTERVAL_SEC,
        resource_manager=None,
        state_path: Optional[str] = None
    ):
        """
        Initialize the scaling coordinator.
        
        Args:
            min_workers: Minimum number of worker nodes to maintain
            max_workers: Maximum number of worker nodes allowed
            scaling_interval_sec: Interval between scaling decisions (seconds)
            resource_manager: ResourceManager instance to coordinate with
            state_path: Path to store state information (if None, use in-memory only)
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scaling_interval_sec = scaling_interval_sec
        self.resource_manager = resource_manager
        self.state_path = Path(state_path) if state_path else None
        
        # Worker node registry
        self.workers: Dict[str, WorkerNode] = {}
        self.local_worker_id = self._generate_worker_id()
        
        # Scaling state
        self.scaling_active = False
        self.last_scaling_time = 0
        self.scaling_lock = threading.Lock()
        self.scaling_thread = None
        
        # Initialize local worker node
        self._register_local_worker()
        
        # Load saved state if available
        if self.state_path and self.state_path.exists():
            self._load_state()
    
    def _generate_worker_id(self) -> str:
        """Generate a unique worker ID based on hostname and timestamp."""
        hostname = socket.gethostname()
        timestamp = int(time.time())
        return f"worker-{hostname}-{timestamp}"
    
    def _register_local_worker(self) -> None:
        """Register this instance as a local worker node."""
        # Determine agent capacity based on system resources
        total_agents = self._calculate_local_capacity()
        
        local_worker = WorkerNode(
            worker_id=self.local_worker_id,
            hostname=socket.gethostname(),
            port=0,  # Local worker doesn't need port
            state=WorkerState.READY,
            total_agents=total_agents,
            free_agents=total_agents
        )
        
        self.workers[self.local_worker_id] = local_worker
        logger.info(f"Registered local worker node: {self.local_worker_id} with {total_agents} agents")
        
        # If we have a resource manager, update its capacity
        if self.resource_manager:
            self.resource_manager.update_total_agents(total_agents)
    
    def _calculate_local_capacity(self) -> int:
        """Calculate how many agents the local system can support based on resources."""
        # Start with default minimum capacity
        capacity = 3  # Minimum of 3 agents (1 bug)
        
        # If psutil is available, calculate based on system resources
        if HAVE_PSUTIL:
            # CPU cores - each agent needs ~0.5 cores
            cpu_count = psutil.cpu_count(logical=True)
            cpu_capacity = max(3, cpu_count * 2)
            
            # Memory - each agent needs ~512MB
            mem_info = psutil.virtual_memory()
            mem_capacity = max(3, int(mem_info.total / (512 * 1024 * 1024)))
            
            # Use the limiting factor
            capacity = min(cpu_capacity, mem_capacity)
        
        return capacity
    
    def _save_state(self) -> None:
        """Save current cluster state to persistent storage."""
        if not self.state_path:
            return
            
        try:
            # Create directory if it doesn't exist
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert worker objects to serializable dictionaries
            workers_dict = {
                worker_id: worker.as_dict() 
                for worker_id, worker in self.workers.items()
            }
            
            state = {
                "timestamp": time.time(),
                "workers": workers_dict,
                "local_worker_id": self.local_worker_id
            }
            
            # Write to file
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.debug(f"Saved scaling state to {self.state_path}")
        except Exception as e:
            logger.error(f"Failed to save scaling state: {e}")
    
    def _load_state(self) -> None:
        """Load cluster state from persistent storage."""
        if not self.state_path or not self.state_path.exists():
            return
            
        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)
            
            # Process workers (excluding local worker which we already initialized)
            workers_dict = state.get("workers", {})
            for worker_id, worker_data in workers_dict.items():
                if worker_id != self.local_worker_id:
                    self.workers[worker_id] = WorkerNode(
                        worker_id=worker_id,
                        hostname=worker_data["hostname"],
                        port=worker_data["port"],
                        state=WorkerState(worker_data["state"]),
                        total_agents=worker_data["total_agents"],
                        free_agents=worker_data["free_agents"],
                        cpu_cores=worker_data["cpu_cores"],
                        memory_mb=worker_data["memory_mb"],
                        last_heartbeat=worker_data["last_heartbeat"],
                        metrics=worker_data.get("metrics", {})
                    )
            
            logger.info(f"Loaded scaling state with {len(self.workers)} workers")
        except Exception as e:
            logger.error(f"Failed to load scaling state: {e}")
    
    def start(self) -> None:
        """Start the scaling coordinator."""
        with self.scaling_lock:
            if self.scaling_active:
                return
                
            self.scaling_active = True
            self.scaling_thread = threading.Thread(
                target=self._scaling_loop,
                daemon=True,
                name="scaling-coordinator"
            )
            self.scaling_thread.start()
            logger.info("Scaling coordinator started")
    
    def stop(self) -> None:
        """Stop the scaling coordinator."""
        with self.scaling_lock:
            if not self.scaling_active:
                return
                
            self.scaling_active = False
            if self.scaling_thread:
                self.scaling_thread.join(timeout=5)
                self.scaling_thread = None
            
            # Save final state
            self._save_state()
            logger.info("Scaling coordinator stopped")
    
    def _scaling_loop(self) -> None:
        """Main loop for scaling decisions."""
        while self.scaling_active:
            try:
                # Check for dead workers (no heartbeat in 2x interval)
                self._cleanup_dead_workers()
                
                # Make scaling decisions
                self._evaluate_scaling()
                
                # Save current state
                self._save_state()
                
                # Sleep until next interval
                time.sleep(self.scaling_interval_sec)
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(5)  # Shorter sleep on error
    
    def _cleanup_dead_workers(self) -> None:
        """Remove workers that haven't sent a heartbeat recently."""
        now = time.time()
        timeout = self.scaling_interval_sec * 2
        
        dead_workers = [
            worker_id for worker_id, worker in self.workers.items()
            if worker_id != self.local_worker_id and 
            (now - worker.last_heartbeat) > timeout
        ]
        
        for worker_id in dead_workers:
            logger.warning(f"Removing dead worker: {worker_id}")
            self.workers.pop(worker_id, None)
    
    def _evaluate_scaling(self) -> None:
        """Evaluate current load and make scaling decisions."""
        # Skip if we're already at min/max workers
        current_workers = len(self.workers)
        if current_workers >= self.max_workers and current_workers <= self.min_workers:
            return
            
        # Calculate cluster-wide utilization
        total_agents = sum(w.total_agents for w in self.workers.values())
        free_agents = sum(w.free_agents for w in self.workers.values())
        
        if total_agents == 0:
            utilization = 0.0
        else:
            utilization = 1.0 - (free_agents / total_agents)
        
        # Determine if we should scale
        now = time.time()
        if now - self.last_scaling_time < self.scaling_interval_sec:
            return  # Too soon since last scaling action
            
        if utilization >= SCALE_UP_THRESHOLD and current_workers < self.max_workers:
            self._scale_up()
            self.last_scaling_time = now
        elif utilization <= SCALE_DOWN_THRESHOLD and current_workers > self.min_workers:
            self._scale_down()
            self.last_scaling_time = now
    
    def _scale_up(self) -> None:
        """Add a new worker node to the cluster."""
        if not HAVE_DOCKER:
            logger.warning("Docker not available, cannot scale up worker nodes")
            return
            
        try:
            logger.info("Scaling up: starting new worker node")
            
            # Here we would use Docker API to start a new container
            # For now, just simulate adding a new worker
            worker_id = f"worker-{int(time.time())}"
            
            # In a real implementation, we would:
            # 1. Start a new container with the FixWurx image
            # 2. Wait for it to start and register
            # 3. Update our worker registry
            
            # For simulation, create a dummy worker
            new_worker = WorkerNode(
                worker_id=worker_id,
                hostname=f"worker-{len(self.workers) + 1}",
                port=8000 + len(self.workers),
                state=WorkerState.STARTING,
                total_agents=9,  # Default agent pool size
                free_agents=9
            )
            
            self.workers[worker_id] = new_worker
            logger.info(f"Added new worker node: {worker_id}")
            
            # In a real implementation, we would wait for the worker to report ready
            # For simulation, just mark it ready after a delay
            time.sleep(2)
            new_worker.state = WorkerState.READY
        except Exception as e:
            logger.error(f"Failed to scale up: {e}")
    
    def _scale_down(self) -> None:
        """Remove a worker node from the cluster."""
        if len(self.workers) <= 1:
            return  # Don't remove the last worker
            
        try:
            logger.info("Scaling down: removing worker node")
            
            # Find the worker with the lowest utilization (excluding local)
            workers_by_utilization = sorted(
                [w for w_id, w in self.workers.items() if w_id != self.local_worker_id],
                key=lambda w: w.utilization()
            )
            
            if not workers_by_utilization:
                return
                
            worker_to_remove = workers_by_utilization[0]
            worker_id = worker_to_remove.worker_id
            
            # In a real implementation, we would:
            # 1. Notify the worker to finish current tasks and shut down
            # 2. Wait for it to complete or force-stop after timeout
            # 3. Remove it from our registry
            
            # For simulation, just remove it
            self.workers.pop(worker_id, None)
            logger.info(f"Removed worker node: {worker_id}")
        except Exception as e:
            logger.error(f"Failed to scale down: {e}")
    
    def update_local_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update metrics for the local worker node."""
        if self.local_worker_id in self.workers:
            self.workers[self.local_worker_id].metrics = metrics
            self.workers[self.local_worker_id].last_heartbeat = time.time()
    
    def update_agent_counts(self, total_agents: int, free_agents: int) -> None:
        """Update agent counts for the local worker node."""
        if self.local_worker_id in self.workers:
            self.workers[self.local_worker_id].total_agents = total_agents
            self.workers[self.local_worker_id].free_agents = free_agents
    
    def get_cluster_capacity(self) -> int:
        """Get the total agent capacity across all active worker nodes."""
        return sum(
            worker.total_agents 
            for worker in self.workers.values() 
            if worker.state in (WorkerState.READY, WorkerState.BUSY)
        )
    
    def get_worker_status(self) -> List[Dict[str, Any]]:
        """Get status information for all worker nodes."""
        return [worker.as_dict() for worker in self.workers.values()]
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get metrics about the current scaling state."""
        total_agents = sum(w.total_agents for w in self.workers.values())
        free_agents = sum(w.free_agents for w in self.workers.values())
        
        return {
            "worker_count": len(self.workers),
            "total_agent_capacity": total_agents,
            "free_agents": free_agents,
            "utilization": 1.0 - (free_agents / total_agents) if total_agents > 0 else 0.0,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "scaling_interval_sec": self.scaling_interval_sec
        }
