"""
core/load_balancer.py
━━━━━━━━━━━━━━━━━━━
Implements intelligent load balancing for distributing bug processing workloads
across multiple worker nodes in a horizontally scaled FixWurx deployment.

Key capabilities:
1. Worker-aware task distribution
2. Multiple load balancing strategies (round-robin, least-loaded, weighted)
3. Health-check based routing
4. Automatic failover for worker node failures
5. Dynamic worker registration and deregistration

Integrates with ScalingCoordinator to obtain worker node information and
with ParallelExecutor to distribute task execution.
"""

import asyncio
import logging
import time
import random
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
import threading
from dataclasses import dataclass, field
import heapq

# Local imports
# We'll import these dynamically to avoid circular imports
# from scaling_coordinator import ScalingCoordinator, WorkerNode, WorkerState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("load_balancer")


class BalancingStrategy(Enum):
    """Load balancing strategies supported by the balancer."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_CAPACITY = "weighted_capacity"
    RANDOM = "random"


@dataclass
class RoutingMetrics:
    """Metrics for a single worker node used in routing decisions."""
    worker_id: str
    capacity: int
    current_load: int
    health_score: float = 1.0  # 0.0-1.0, where 1.0 is fully healthy
    response_time_ms: int = 0
    last_used: float = 0.0

    def available_capacity(self) -> int:
        """Calculate available capacity on this worker."""
        return max(0, self.capacity - self.current_load)
    
    def load_percentage(self) -> float:
        """Calculate load as percentage of capacity."""
        if self.capacity == 0:
            return 1.0  # Fully loaded if capacity is zero
        return self.current_load / self.capacity
    
    def routing_score(self) -> float:
        """Calculate an overall routing score (higher is better for routing)."""
        # Start with available capacity
        score = self.available_capacity()
        
        # Adjust by health
        score *= self.health_score
        
        # Penalize for response time (slower = lower score)
        response_factor = 1.0
        if self.response_time_ms > 0:
            response_factor = 1.0 / (1.0 + (self.response_time_ms / 1000.0))
        score *= response_factor
        
        return score


class LoadBalancer:
    """
    Distributes bug processing tasks across multiple worker nodes based on
    capacity, load, and health status.
    
    This class:
    1. Maintains metrics for all worker nodes
    2. Makes routing decisions based on configured strategy
    3. Handles worker health checks and failover
    4. Coordinates with scaling system to adjust for changes in the cluster
    """
    
    def __init__(
        self, 
        strategy: BalancingStrategy = BalancingStrategy.WEIGHTED_CAPACITY,
        scaling_coordinator=None,
        health_check_interval_sec: int = 30
    ):
        """
        Initialize the load balancer.
        
        Args:
            strategy: The load balancing strategy to use
            scaling_coordinator: ScalingCoordinator instance to coordinate with
            health_check_interval_sec: Interval between worker health checks
        """
        self.strategy = strategy
        self.scaling_coordinator = scaling_coordinator
        self.health_check_interval_sec = health_check_interval_sec
        
        # Worker metrics for routing decisions
        self.worker_metrics: Dict[str, RoutingMetrics] = {}
        
        # Round-robin state
        self.rr_index = 0
        
        # Health check state
        self.health_check_active = False
        self.health_check_lock = threading.Lock()
        self.health_check_thread = None
        
        # Task distribution counters
        self.tasks_routed = 0
        self.tasks_failed = 0
        self.task_latencies_ms = []  # For calculating averages
        
        # Initialize from scaling coordinator if available
        if self.scaling_coordinator:
            self._sync_with_scaling_coordinator()
    
    def start(self) -> None:
        """Start the load balancer health check system."""
        with self.health_check_lock:
            if self.health_check_active:
                return
                
            self.health_check_active = True
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
                name="load-balancer-health"
            )
            self.health_check_thread.start()
            logger.info(f"Load balancer started with {self.strategy.value} strategy")
    
    def stop(self) -> None:
        """Stop the load balancer health check system."""
        with self.health_check_lock:
            if not self.health_check_active:
                return
                
            self.health_check_active = False
            if self.health_check_thread:
                self.health_check_thread.join(timeout=5)
                self.health_check_thread = None
            
            logger.info("Load balancer stopped")
    
    def _health_check_loop(self) -> None:
        """Main loop for health checking worker nodes."""
        while self.health_check_active:
            try:
                # Sync with scaling coordinator to get latest worker information
                if self.scaling_coordinator:
                    self._sync_with_scaling_coordinator()
                
                # Perform health checks on all workers
                self._check_worker_health()
                
                # Sleep until next interval
                time.sleep(self.health_check_interval_sec)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(5)  # Shorter sleep on error
    
    def _sync_with_scaling_coordinator(self) -> None:
        """Synchronize worker information from scaling coordinator."""
        try:
            # Get all worker nodes from scaling coordinator
            worker_status = self.scaling_coordinator.get_worker_status()
            
            # Update our metrics with new workers, remove gone workers
            current_workers = set()
            
            for worker_data in worker_status:
                worker_id = worker_data["worker_id"]
                current_workers.add(worker_id)
                
                # Skip workers that aren't ready
                if worker_data["state"] not in ("ready", "busy"):
                    if worker_id in self.worker_metrics:
                        del self.worker_metrics[worker_id]
                    continue
                
                # Update or create metrics for this worker
                if worker_id in self.worker_metrics:
                    # Update existing metrics
                    self.worker_metrics[worker_id].capacity = worker_data["total_agents"]
                    self.worker_metrics[worker_id].current_load = (
                        worker_data["total_agents"] - worker_data["free_agents"]
                    )
                else:
                    # Create new metrics
                    self.worker_metrics[worker_id] = RoutingMetrics(
                        worker_id=worker_id,
                        capacity=worker_data["total_agents"],
                        current_load=worker_data["total_agents"] - worker_data["free_agents"],
                        health_score=1.0,
                        response_time_ms=0,
                        last_used=time.time()
                    )
            
            # Remove workers that are no longer in the cluster
            for worker_id in list(self.worker_metrics.keys()):
                if worker_id not in current_workers:
                    del self.worker_metrics[worker_id]
            
            logger.debug(f"Synced with scaling coordinator: {len(self.worker_metrics)} active workers")
        except Exception as e:
            logger.error(f"Failed to sync with scaling coordinator: {e}")
    
    def _check_worker_health(self) -> None:
        """Perform health checks on all worker nodes."""
        # In a real implementation, we would:
        # 1. Send a health check request to each worker
        # 2. Measure response time
        # 3. Check for errors or timeouts
        # 4. Update health scores accordingly
        
        # For simulation, we'll just:
        # 1. Check if worker is reachable 
        # 2. Check if it has capacity
        # 3. Generate a simulated health score
        
        for worker_id, metrics in self.worker_metrics.items():
            try:
                # Simulate health check
                # In a real implementation, send an HTTP request and measure response time
                
                # Simulate response time (10-200ms)
                response_time = random.randint(10, 200)
                metrics.response_time_ms = response_time
                
                # Calculate health score based on response time
                # 0-50ms: 1.0, 50-100ms: 0.9, 100-150ms: 0.8, 150-200ms: 0.7
                health_score = 1.0 - (response_time / 1000.0)
                metrics.health_score = max(0.7, health_score)
                
                logger.debug(f"Health check for {worker_id}: {metrics.health_score:.2f} ({response_time}ms)")
            except Exception as e:
                # If health check fails, reduce health score
                metrics.health_score = max(0.0, metrics.health_score - 0.2)
                logger.warning(f"Health check failed for {worker_id}: {e}")
    
    def select_worker(self, task_id: str, requirements: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Select the best worker for a given task based on the current strategy.
        
        Args:
            task_id: Unique identifier for the task
            requirements: Optional requirements for worker selection
                          (e.g., minimum free capacity, specific capabilities)
        
        Returns:
            worker_id of the selected worker, or None if no suitable worker is found
        """
        if not self.worker_metrics:
            logger.warning("No workers available for routing")
            return None
        
        selected_worker_id = None
        
        # Apply the selected strategy
        if self.strategy == BalancingStrategy.ROUND_ROBIN:
            selected_worker_id = self._select_round_robin()
        elif self.strategy == BalancingStrategy.LEAST_CONNECTIONS:
            selected_worker_id = self._select_least_connections()
        elif self.strategy == BalancingStrategy.WEIGHTED_CAPACITY:
            selected_worker_id = self._select_weighted_capacity()
        elif self.strategy == BalancingStrategy.RANDOM:
            selected_worker_id = self._select_random()
        else:
            # Default to weighted capacity
            selected_worker_id = self._select_weighted_capacity()
        
        if selected_worker_id:
            # Update metrics for the selected worker
            metrics = self.worker_metrics[selected_worker_id]
            metrics.current_load += 1
            metrics.last_used = time.time()
            self.tasks_routed += 1
        else:
            self.tasks_failed += 1
            
        return selected_worker_id
    
    def _select_round_robin(self) -> Optional[str]:
        """Select a worker using round-robin strategy."""
        if not self.worker_metrics:
            return None
            
        # Get sorted list of worker IDs
        worker_ids = sorted(self.worker_metrics.keys())
        
        # Check each worker in order from current index
        for _ in range(len(worker_ids)):
            # Update round-robin index
            self.rr_index = (self.rr_index + 1) % len(worker_ids)
            worker_id = worker_ids[self.rr_index]
            
            # Check if this worker has capacity
            metrics = self.worker_metrics[worker_id]
            if metrics.available_capacity() > 0 and metrics.health_score > 0.5:
                return worker_id
        
        # If we got here, no workers had capacity
        return None
    
    def _select_least_connections(self) -> Optional[str]:
        """Select the worker with the fewest active connections."""
        if not self.worker_metrics:
            return None
            
        min_load = float('inf')
        selected_id = None
        
        for worker_id, metrics in self.worker_metrics.items():
            # Skip unhealthy workers
            if metrics.health_score <= 0.5:
                continue
                
            # Check if this worker has less load than current minimum
            if metrics.current_load < min_load and metrics.available_capacity() > 0:
                min_load = metrics.current_load
                selected_id = worker_id
        
        return selected_id
    
    def _select_weighted_capacity(self) -> Optional[str]:
        """Select a worker based on weighted capacity and health."""
        if not self.worker_metrics:
            return None
            
        best_score = -1.0
        selected_id = None
        
        for worker_id, metrics in self.worker_metrics.items():
            # Skip workers with no capacity
            if metrics.available_capacity() <= 0:
                continue
                
            # Calculate routing score
            score = metrics.routing_score()
            
            # Check if this worker has a better score
            if score > best_score:
                best_score = score
                selected_id = worker_id
        
        return selected_id
    
    def _select_random(self) -> Optional[str]:
        """Select a random worker with available capacity."""
        if not self.worker_metrics:
            return None
            
        # Filter to workers with capacity and reasonable health
        available_workers = [
            worker_id for worker_id, metrics in self.worker_metrics.items()
            if metrics.available_capacity() > 0 and metrics.health_score > 0.5
        ]
        
        if not available_workers:
            return None
            
        # Select random worker from available ones
        return random.choice(available_workers)
    
    def release_worker(self, worker_id: str) -> None:
        """
        Release a worker after task completion.
        
        Args:
            worker_id: ID of the worker to release
        """
        if worker_id in self.worker_metrics:
            # Decrement load counter
            self.worker_metrics[worker_id].current_load = max(
                0, self.worker_metrics[worker_id].current_load - 1
            )
    
    def register_task_latency(self, worker_id: str, latency_ms: int) -> None:
        """
        Register the latency of a completed task for metrics.
        
        Args:
            worker_id: ID of the worker that processed the task
            latency_ms: Task processing latency in milliseconds
        """
        # Update response time for the worker
        if worker_id in self.worker_metrics:
            # Use exponential moving average
            alpha = 0.3  # Weight for new observation
            old_time = self.worker_metrics[worker_id].response_time_ms
            new_time = old_time * (1 - alpha) + latency_ms * alpha
            self.worker_metrics[worker_id].response_time_ms = int(new_time)
        
        # Track overall latencies (keep last 100)
        self.task_latencies_ms.append(latency_ms)
        if len(self.task_latencies_ms) > 100:
            self.task_latencies_ms.pop(0)
    
    def get_balancer_metrics(self) -> Dict[str, Any]:
        """Get metrics about the load balancer's operation."""
        # Calculate average latency
        avg_latency = 0
        if self.task_latencies_ms:
            avg_latency = sum(self.task_latencies_ms) / len(self.task_latencies_ms)
            
        # Count available workers
        available_workers = sum(
            1 for metrics in self.worker_metrics.values()
            if metrics.available_capacity() > 0 and metrics.health_score > 0.5
        )
        
        return {
            "strategy": self.strategy.value,
            "worker_count": len(self.worker_metrics),
            "available_workers": available_workers,
            "tasks_routed": self.tasks_routed,
            "tasks_failed": self.tasks_failed,
            "average_latency_ms": avg_latency,
            "health_check_interval_sec": self.health_check_interval_sec
        }
    
    def get_worker_metrics(self) -> List[Dict[str, Any]]:
        """Get detailed metrics for all worker nodes."""
        return [
            {
                "worker_id": metrics.worker_id,
                "capacity": metrics.capacity,
                "current_load": metrics.current_load,
                "available_capacity": metrics.available_capacity(),
                "load_percentage": metrics.load_percentage(),
                "health_score": metrics.health_score,
                "response_time_ms": metrics.response_time_ms,
                "last_used": metrics.last_used
            }
            for metrics in self.worker_metrics.values()
        ]
    
    def set_balancing_strategy(self, strategy: BalancingStrategy) -> None:
        """
        Change the current load balancing strategy.
        
        Args:
            strategy: The new strategy to use
        """
        self.strategy = strategy
        logger.info(f"Load balancing strategy changed to {strategy.value}")
