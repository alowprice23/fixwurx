#!/usr/bin/env python3
"""
async_load_balancer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Async event-driven load balancer for distributing tasks across worker nodes.
Implements the design from LOAD_BALANCER_MASTER_PLAN.md.
"""

import asyncio
import time
import logging
import hashlib
import random
from typing import Dict, List, Any, Optional, Set, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AsyncLoadBalancer")

class State(Enum):
    """Load balancer states."""
    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()

class BalancingStrategy(Enum):
    """Available load balancing strategies."""
    ROUND_ROBIN = auto()
    LEAST_CONNECTIONS = auto()
    WEIGHTED_CAPACITY = auto()
    CONSISTENT_HASH = auto()

class AffinityType(Enum):
    """Types of worker affinities."""
    CPU_INTENSIVE = auto()
    MEMORY_INTENSIVE = auto()
    IO_INTENSIVE = auto()
    NETWORK_INTENSIVE = auto()

class LoadBalancerError(Exception):
    """Base class for load balancer errors."""
    pass

class InvalidStateTransitionError(LoadBalancerError):
    """Error raised when an invalid state transition is attempted."""
    pass

class LoadBalancerNotRunningError(LoadBalancerError):
    """Error raised when operations are attempted while not running."""
    pass

class CircuitBreakerOpenError(LoadBalancerError):
    """Error raised when the circuit breaker is open."""
    pass

@dataclass
class Event:
    """Event for the event loop."""
    event_type: str
    data: Any

@dataclass
class Worker:
    """Worker node information."""
    id: str
    capacity: int = 0
    current_load: int = 0
    health_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    affinities: Dict[str, float] = field(default_factory=dict)
    languages: Dict[str, float] = field(default_factory=dict)

@dataclass
class BugRouting:
    """Bug routing information."""
    bug_id: str
    worker_id: str
    first_routed_at: float
    last_routed_at: float
    success_count: int = 0
    failure_count: int = 0
    total_processing_time_ms: int = 0
    
    @property
    def total_attempts(self) -> int:
        """Get the total number of attempts."""
        return self.success_count + self.failure_count
    
    @property
    def success_rate(self) -> float:
        """Get the success rate."""
        if self.total_attempts == 0:
            return 1.0
        return self.success_count / self.total_attempts
    
    @property
    def avg_processing_time_ms(self) -> float:
        """Get the average processing time in milliseconds."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_attempts

class StateManager:
    """
    Manages the state lifecycle of the load balancer.
    """
    def __init__(self):
        self.current_state = State.CREATED
        self.observers = set()
        self._lock = asyncio.Lock()
        
    async def transition_to(self, new_state: State):
        """Transition to a new state with validation."""
        async with self._lock:
            # Validate the transition
            if not self._is_valid_transition(self.current_state, new_state):
                raise InvalidStateTransitionError(
                    f"Cannot transition from {self.current_state} to {new_state}"
                )
                
            # Perform the transition
            old_state = self.current_state
            self.current_state = new_state
            
            # Notify observers
            await self._notify_observers(old_state, new_state)
        
    def _is_valid_transition(self, from_state: State, to_state: State) -> bool:
        """Check if a state transition is valid."""
        valid_transitions = {
            State.CREATED: {State.STARTING, State.ERROR},
            State.STARTING: {State.RUNNING, State.ERROR},
            State.RUNNING: {State.STOPPING, State.ERROR},
            State.STOPPING: {State.STOPPED, State.ERROR},
            State.STOPPED: {State.STARTING},
            State.ERROR: {State.STOPPING}
        }
        return to_state in valid_transitions.get(from_state, set())
        
    async def _notify_observers(self, old_state: State, new_state: State):
        """Notify all observers of the state change."""
        for observer in self.observers:
            await observer.on_state_change(old_state, new_state)
            
    def add_observer(self, observer):
        """Add an observer for state changes."""
        self.observers.add(observer)
        
    def remove_observer(self, observer):
        """Remove an observer for state changes."""
        self.observers.discard(observer)

class EventLoop:
    """
    Asynchronous event processing loop.
    """
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        self.running = False
        self.event_queue = asyncio.Queue()
        self.handlers = {}
        
    async def start(self):
        """Start the event loop."""
        if self.running:
            return
            
        self.running = True
        
        if not self.test_mode:
            # Start the event processing task
            asyncio.create_task(self._process_events())
        
    async def stop(self):
        """Stop the event loop."""
        self.running = False
        
        if not self.test_mode:
            # Add a sentinel event to unblock the queue
            await self.event_queue.put(None)
            
    async def publish(self, event_type: str, event_data: Any):
        """Publish an event to the loop."""
        event = Event(event_type=event_type, data=event_data)
        
        if self.test_mode:
            # In test mode, process events immediately
            await self._handle_event(event)
        else:
            # In normal mode, queue events for processing
            await self.event_queue.put(event)
            
    def register_handler(self, event_type: str, handler: Callable):
        """Register a handler for an event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
            
    async def _process_events(self):
        """Process events from the queue."""
        while self.running:
            # Get the next event
            event = await self.event_queue.get()
            
            # Check for sentinel
            if event is None:
                break
                
            # Process the event
            try:
                await self._handle_event(event)
            except Exception as e:
                # Log the error but don't stop processing
                logger.error(f"Error handling event: {e}")
                
            # Mark the event as done
            self.event_queue.task_done()
            
    async def _handle_event(self, event: Event):
        """Handle a single event."""
        handlers = self.handlers.get(event.event_type, [])
        for handler in handlers:
            await handler(event)

class ConsistentHashRing:
    """Consistent hash ring for stable distribution."""
    
    def __init__(self, replicas: int = 100):
        """Initialize the consistent hash ring."""
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        self.node_keys = {}
    
    def add_node(self, node_id: str):
        """Add a node to the ring."""
        # Check if node already exists
        if node_id in self.node_keys:
            return
        
        # Create hash keys for this node
        node_keys = []
        for i in range(self.replicas):
            key = self._hash(f"{node_id}:{i}")
            self.ring[key] = node_id
            node_keys.append(key)
        
        # Store the keys for this node
        self.node_keys[node_id] = node_keys
        
        # Re-sort the keys
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node_id: str):
        """Remove a node from the ring."""
        if node_id not in self.node_keys:
            return
        
        # Remove this node's keys
        for key in self.node_keys[node_id]:
            if key in self.ring:
                del self.ring[key]
        
        # Clean up the node's keys
        del self.node_keys[node_id]
        
        # Re-sort the keys
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """Get the node for a key."""
        if not self.ring:
            return None
        
        # Hash the key
        hash_key = self._hash(key)
        
        # Find the node responsible for this key
        for ring_key in self.sorted_keys:
            if hash_key <= ring_key:
                return self.ring[ring_key]
        
        # If we get here, we've wrapped around the ring
        return self.ring[self.sorted_keys[0]] if self.sorted_keys else None
    
    def _hash(self, key: str) -> int:
        """Hash a key to an integer."""
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the circuit breaker."""
        self.failure_threshold = config.get("circuit_breaker_failure_threshold", 5)
        self.reset_timeout_sec = config.get("circuit_breaker_reset_timeout_sec", 30)
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()
        
    def is_open(self) -> bool:
        """Check if the circuit breaker is open."""
        return self.state == "OPEN"
    
    async def record_success(self):
        """Record a successful operation."""
        async with self._lock:
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED state after success in HALF_OPEN state")
    
    async def record_failure(self):
        """Record a failed operation."""
        async with self._lock:
            self.last_failure_time = time.time()
            
            if self.state == "OPEN":
                # Already open, nothing to do
                return
                
            if self.state == "HALF_OPEN":
                # Failed in half-open state, go back to open
                self.state = "OPEN"
                logger.warning("Circuit breaker returned to OPEN state after failure in HALF_OPEN state")
                return
                
            # In closed state, increment failure count
            self.failure_count += 1
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker tripped to OPEN state after {self.failure_count} failures")
    
    async def check_state(self):
        """Check and potentially update the circuit breaker state."""
        async with self._lock:
            if self.state == "OPEN":
                # Check if it's time to try again
                if time.time() - self.last_failure_time >= self.reset_timeout_sec:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker transitioned to HALF_OPEN state to test recovery")

class BulkheadManager:
    """Provides isolation between components using bulkheads."""
    
    def __init__(self):
        """Initialize the bulkhead manager."""
        self.bulkheads = {}
        
    def configure_bulkhead(self, name: str, max_concurrent: int = 10, queue_size: int = 10):
        """Configure a bulkhead with specified limits."""
        self.bulkheads[name] = asyncio.Semaphore(max_concurrent)
        
    @asynccontextmanager
    async def get_bulkhead(self, name: str):
        """Get a bulkhead for a specific operation."""
        if name not in self.bulkheads:
            self.configure_bulkhead(name)
            
        semaphore = self.bulkheads[name]
        
        try:
            await semaphore.acquire()
            yield
        finally:
            semaphore.release()

class MetricsCollector:
    """Collects and processes performance metrics."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = {}
        self.timers = {}
        
    async def start(self):
        """Start the metrics collector."""
        pass
        
    async def stop(self):
        """Stop the metrics collector."""
        pass
        
    @asynccontextmanager
    async def measure(self, operation: str):
        """Measure the execution time of an operation."""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = (time.time() - start_time) * 1000  # in ms
            await self.record_metric(operation, execution_time)
            
    async def record_metric(self, metric_name: str, value: float):
        """Record a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            
        self.metrics[metric_name].append(value)
        
        # Keep only recent metrics (e.g., last 1000)
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]
            
    def get_metric_average(self, metric_name: str) -> float:
        """Get the average value of a metric."""
        values = self.metrics.get(metric_name, [])
        return sum(values) / max(1, len(values))
        
    def get_metric_percentile(self, metric_name: str, percentile: float) -> float:
        """Get a percentile value of a metric."""
        values = sorted(self.metrics.get(metric_name, []))
        if not values:
            return 0.0
            
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]

class HealthMonitor:
    """Monitors system and worker health."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the health monitor."""
        self.check_interval_sec = config.get("health_check_interval_sec", 5)
        self.health_checks = {}
        self.running = False
        self._task = None
        
    async def start(self):
        """Start the health monitor."""
        if self.running:
            return
            
        self.running = True
        self._task = asyncio.create_task(self._health_check_loop())
        
    async def stop(self):
        """Stop the health monitor."""
        self.running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_func
        
    async def _health_check_loop(self):
        """Main health check loop."""
        try:
            while self.running:
                await self._run_health_checks()
                await asyncio.sleep(self.check_interval_sec)
        except asyncio.CancelledError:
            logger.info("Health check loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in health check loop: {e}")
            
    async def _run_health_checks(self):
        """Run all registered health checks."""
        for name, check_func in self.health_checks.items():
            try:
                await check_func()
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")

class WorkerRegistry:
    """Manages worker information and health."""
    
    def __init__(self):
        """Initialize the worker registry."""
        self.workers = {}
        self.bug_assignments = {}
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the worker registry."""
        pass
        
    async def add_worker(self, worker: Worker):
        """Add a worker to the registry."""
        async with self._lock:
            self.workers[worker.id] = worker
            self.bug_assignments[worker.id] = set()
            
    async def remove_worker(self, worker_id: str):
        """Remove a worker from the registry."""
        async with self._lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                
            if worker_id in self.bug_assignments:
                del self.bug_assignments[worker_id]
                
    async def get_worker(self, worker_id: str) -> Optional[Worker]:
        """Get a worker by ID."""
        async with self._lock:
            return self.workers.get(worker_id)
            
    async def get_all_workers(self) -> List[Worker]:
        """Get all workers."""
        async with self._lock:
            return list(self.workers.values())
            
    async def get_healthy_workers(self) -> List[Worker]:
        """Get all healthy workers."""
        async with self._lock:
            return [w for w in self.workers.values() if w.health_score > 0]
            
    async def record_assignment(self, bug_id: str, worker_id: str):
        """Record a bug assignment to a worker."""
        async with self._lock:
            if worker_id in self.bug_assignments:
                self.bug_assignments[worker_id].add(bug_id)

class StrategyEngine:
    """Pluggable routing strategies engine."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy engine."""
        self.config = config
        self.current_strategy = config.get("default_strategy", BalancingStrategy.ROUND_ROBIN)
        self.last_worker_index = 0
        self.hash_ring = ConsistentHashRing(replicas=config.get("hash_replicas", 100))
        
    def get_strategy(self) -> BalancingStrategy:
        """Get the current active strategy."""
        return self.current_strategy
        
    def set_strategy(self, strategy: BalancingStrategy):
        """Set the current active strategy."""
        self.current_strategy = strategy
        
    async def select_worker(self, workers: List[Worker], task_id: str, requirements: Optional[Dict[str, Any]] = None) -> Optional[Worker]:
        """Select a worker for a task using the current strategy."""
        if not workers:
            return None
            
        requirements = requirements or {}
        
        # Use the appropriate strategy
        if self.current_strategy == BalancingStrategy.ROUND_ROBIN:
            return await self._select_round_robin(workers)
        elif self.current_strategy == BalancingStrategy.LEAST_CONNECTIONS:
            return await self._select_least_connections(workers)
        elif self.current_strategy == BalancingStrategy.WEIGHTED_CAPACITY:
            return await self._select_weighted_capacity(workers)
        elif self.current_strategy == BalancingStrategy.CONSISTENT_HASH:
            return await self._select_consistent_hash(workers, task_id)
        else:
            # Default to round robin
            return await self._select_round_robin(workers)
            
    async def _select_round_robin(self, workers: List[Worker]) -> Worker:
        """Select a worker using round robin strategy."""
        self.last_worker_index = (self.last_worker_index + 1) % len(workers)
        return workers[self.last_worker_index]
        
    async def _select_least_connections(self, workers: List[Worker]) -> Worker:
        """Select a worker with the fewest connections."""
        return min(workers, key=lambda w: w.current_load)
        
    async def _select_weighted_capacity(self, workers: List[Worker]) -> Worker:
        """Select a worker based on available capacity."""
        # Find workers with capacity
        available_workers = [w for w in workers if w.capacity > w.current_load]
        
        if not available_workers:
            # Fall back to least connections if no capacity
            return await self._select_least_connections(workers)
            
        # Calculate weights based on available capacity
        weights = []
        for worker in available_workers:
            available = max(0, worker.capacity - worker.current_load)
            weights.append(available / max(1, worker.capacity))
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            
        # Select worker based on weights
        rand = random.random()
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand <= cumulative:
                return available_workers[i]
                
        # Fallback
        return available_workers[-1]
        
    async def _select_consistent_hash(self, workers: List[Worker], task_id: str) -> Worker:
        """Select a worker using consistent hashing."""
        # Ensure all workers are in the hash ring
        for worker in workers:
            self.hash_ring.add_node(worker.id)
            
        # Get the worker ID from the hash ring
        worker_id = self.hash_ring.get_node(task_id)
        
        # Find the worker object
        for worker in workers:
            if worker.id == worker_id:
                return worker
                
        # Fallback to round robin if worker not found
        return await self._select_round_robin(workers)

class AsyncLoadBalancer:
    """
    Async event-driven load balancer implementation.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, test_mode: bool = False):
        """Initialize the async load balancer."""
        self.config = config or {}
        self.test_mode = test_mode
        
        # Create core components
        self.state_manager = StateManager()
        self.event_loop = EventLoop(test_mode=test_mode)
        self.worker_registry = WorkerRegistry()
        self.strategy_engine = StrategyEngine(self.config)
        self.health_monitor = HealthMonitor(self.config)
        self.metrics_collector = MetricsCollector()
        self.circuit_breaker = CircuitBreaker(self.config)
        self.bulkhead_manager = BulkheadManager()
        
        # Bug routing info
        self.bug_routing = {}
        
        # Register for state changes
        self.state_manager.add_observer(self)
        
    async def on_state_change(self, old_state: State, new_state: State):
        """Handle state changes."""
        logger.info(f"State changed from {old_state} to {new_state}")
        
    async def start(self):
        """Start the load balancer asynchronously."""
        await self.state_manager.transition_to(State.STARTING)
        await self.event_loop.start()
        await self.worker_registry.initialize()
        await self.health_monitor.start()
        await self.metrics_collector.start()
        await self.state_manager.transition_to(State.RUNNING)
        logger.info("Async Load Balancer started")
        
    async def stop(self):
        """Stop the load balancer gracefully."""
        await self.state_manager.transition_to(State.STOPPING)
        await self.health_monitor.stop()
        await self.metrics_collector.stop()
        await self.event_loop.stop()
        await self.state_manager.transition_to(State.STOPPED)
        logger.info("Async Load Balancer stopped")
        
    async def select_worker(self, task_id: str, requirements: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Select a worker for a task asynchronously."""
        if self.state_manager.current_state != State.RUNNING:
            raise LoadBalancerNotRunningError("Load balancer is not running")
            
        if self.circuit_breaker.is_open():
            raise CircuitBreakerOpenError("Circuit breaker is open")
            
        requirements = requirements or {}
        
        # Use bulkhead to isolate this operation
        async with self.bulkhead_manager.get_bulkhead("worker_selection"):
            # Record metrics for this operation
            async with self.metrics_collector.measure("worker_selection"):
                # Check for sticky routing
                if self.config.get("enable_sticky_routing", True) and not requirements.get("bypass_sticky", False):
                    worker_id = await self._check_sticky_routing(task_id)
                    if worker_id:
                        return worker_id
                
                # Get healthy workers
                workers = await self.worker_registry.get_healthy_workers()
                if not workers:
                    return None
                    
                # Select a worker using the strategy
                worker = await self.strategy_engine.select_worker(workers, task_id, requirements)
                if not worker:
                    return None
                    
                # Record the assignment
                await self._record_assignment(task_id, worker.id)
                
                return worker.id
                
    async def _check_sticky_routing(self, task_id: str) -> Optional[str]:
        """Check if a task should use sticky routing."""
        if task_id in self.bug_routing:
            routing = self.bug_routing[task_id]
            worker_id = routing.worker_id
            
            # Check if the worker is still healthy
            worker = await self.worker_registry.get_worker(worker_id)
            if worker and worker.health_score > 0:
                # Update last routed time
                routing.last_routed_at = time.time()
                return worker_id
                
        return None
        
    async def _record_assignment(self, task_id: str, worker_id: str):
        """Record a task assignment."""
        current_time = time.time()
        
        # Record in worker registry
        await self.worker_registry.record_assignment(task_id, worker_id)
        
        # Update bug routing
        if task_id in self.bug_routing:
            routing = self.bug_routing[task_id]
            
            # If worker changed, update
            if routing.worker_id != worker_id:
                routing.worker_id = worker_id
                
            # Update last routed time
            routing.last_routed_at = current_time
        else:
            # Create new routing entry
            self.bug_routing[task_id] = BugRouting(
                bug_id=task_id,
                worker_id=worker_id,
                first_routed_at=current_time,
                last_routed_at=current_time
            )
            
    async def register_task_completion(self, task_id: str, worker_id: str, success: bool, processing_time_ms: int):
        """Register a task completion."""
        if task_id in self.bug_routing:
            routing = self.bug_routing[task_id]
            
            if success:
                routing.success_count += 1
                await self.circuit_breaker.record_success()
            else:
                routing.failure_count += 1
                await self.circuit_breaker.record_failure()
                
            routing.total_processing_time_ms += processing_time_ms
            
    async def add_worker(self, worker_id: str, capacity: int = 10, metadata: Optional[Dict[str, Any]] = None):
        """Add a worker to the load balancer."""
        worker = Worker(
            id=worker_id,
            capacity=capacity,
            metadata=metadata or {}
        )
        await self.worker_registry.add_worker(worker)
        logger.info(f"Added worker {worker_id} to load balancer")
        
    async def remove_worker(self, worker_id: str):
        """Remove a worker from the load balancer."""
        await self.worker_registry.remove_worker(worker_id)
        logger.info(f"Removed worker {worker_id} from load balancer")
        
    async def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the load balancer."""
        workers = await self.worker_registry.get_all_workers()
        
        return {
            "state": self.state_manager.current_state.name,
            "worker_count": len(workers),
            "healthy_worker_count": len([w for w in workers if w.health_score > 0]),
            "active_tasks_count": len(self.bug_routing),
            "current_strategy": self.strategy_engine.current_strategy.name,
            "circuit_breaker_state": self.circuit_breaker.state,
            "test_mode": self.test_mode,
            "avg_worker_selection_time_ms": self.metrics_collector.get_metric_average("worker_selection"),
            "p95_worker_selection_time_ms": self.metrics_collector.get_metric_percentile("worker_selection", 95)
        }
