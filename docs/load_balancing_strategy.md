# FixWurx Load Balancing Strategy

## Overview

The FixWurx load balancing system distributes bug processing workloads across multiple worker nodes in a horizontally scaled deployment. It employs advanced routing strategies to optimize resource utilization, ensure fault tolerance, and maximize processing efficiency.

## Key Components

### 1. Basic Load Balancer

The base `LoadBalancer` class provides core load balancing functionality:

- Multiple balancing strategies (round-robin, least-connections, weighted-capacity)
- Health check monitoring of worker nodes
- Basic routing metrics collection
- Worker capacity tracking

### 2. Advanced Load Balancer

The `AdvancedLoadBalancer` extends the base functionality with:

- **Consistent hashing** for stable bug-to-worker assignment
- **Sticky bug routing** to maintain cache and context locality
- **Dynamic strategy selection** based on cluster conditions
- **Affinity-based routing** for specialized worker capabilities
- **Predictive load balancing** using historical usage patterns

## Load Balancing Strategies

### Round Robin (BalancingStrategy.ROUND_ROBIN)

Distributes tasks evenly across all available workers in a cyclical fashion.

**Best for:** Uniform worker capabilities and evenly sized tasks.

### Least Connections (BalancingStrategy.LEAST_CONNECTIONS)

Routes tasks to the worker with the fewest active connections.

**Best for:** Variable task durations and preventing worker overload.

### Weighted Capacity (BalancingStrategy.WEIGHTED_CAPACITY)

Routes tasks based on worker capacity, load, health, and response time.

**Best for:** Heterogeneous clusters with varying worker capacities.

### Dynamic Strategy Selection

The advanced load balancer can automatically select the optimal strategy based on:
- Current cluster load
- Load distribution uniformity
- Worker capacity variability
- Cluster size

## Advanced Features

### Consistent Hashing

Provides stable task-to-worker mapping even as the cluster scales up or down.

**Benefits:**
- Minimizes redistribution when workers are added/removed
- Ensures predictable routing for related tasks
- Improves cache locality and efficiency

### Sticky Bug Routing

Ensures the same worker consistently handles a specific bug, maintaining context.

**Benefits:**
- Preserves cached bug context
- Reduces duplicate work
- Improves efficiency with specialized knowledge

### Affinity-Based Routing

Routes tasks to workers based on their demonstrated proficiency with specific:
- Programming languages
- Task types (CPU-intensive, memory-intensive, I/O-intensive)

**Benefits:**
- Leverages worker specialization
- Improves processing efficiency
- Adapts to emergent worker strengths

### Predictive Load Balancing

Uses historical data to predict future worker load and routes accordingly.

**Benefits:**
- Anticipates load spikes
- Prevents hotspots
- Optimizes resource allocation proactively

## Configuration Options

The load balancer can be configured through the system configuration:

```yaml
load_balancing:
  # Basic settings
  strategy: "weighted_capacity"  # round_robin, least_connections, weighted_capacity, random
  health_check_interval_sec: 30
  
  # Advanced settings
  hash_replicas: 100                # Virtual nodes per physical node for consistent hashing
  sticky_bugs: true                 # Enable sticky bug routing
  sticky_expiration_sec: 3600       # How long to maintain sticky assignments
  enable_affinity_routing: true     # Enable routing based on worker affinities
  affinity_weight: 0.3              # Influence of affinity in routing decisions (0.0-1.0)
  enable_predictive_routing: true   # Enable predictive load-based routing
  prediction_weight: 0.2            # Influence of load predictions (0.0-1.0)
  auto_strategy_selection: false    # Automatically select optimal strategy
  strategy_update_interval_sec: 300 # How often to re-evaluate strategy
```

## Integration with Other Components

The load balancer integrates with:

- **ScalingCoordinator**: Obtains worker status and metrics
- **ResourceAllocationOptimizer**: Gets load predictions and optimization metrics
- **ParallelExecutor**: Distributes tasks to selected workers
- **SystemMonitor**: Reports load balancing metrics

## Usage Example

```python
from advanced_load_balancer import AdvancedLoadBalancer, BalancingStrategy

# Create load balancer
balancer = AdvancedLoadBalancer(
    strategy=BalancingStrategy.WEIGHTED_CAPACITY,
    scaling_coordinator=scaling_coordinator,
    resource_optimizer=resource_optimizer,
    config=config.get("load_balancing", {})
)

# Start the balancer
balancer.start()

# Select a worker for a task with specific requirements
worker_id = balancer.select_worker(
    task_id="bug-123",
    requirements={
        "language": "python",
        "affinity_type": "memory_intensive",
        "sticky": True
    }
)

# After task completion, register the result for learning
balancer.register_task_completion(
    task_id="bug-123",
    worker_id=worker_id,
    success=True,
    processing_time_ms=250
)

# Get advanced metrics
metrics = balancer.get_advanced_metrics()
```

## Performance Considerations

- The load balancer is designed to be lightweight with O(1) worker selection time in most cases
- Consistent hashing provides O(log n) lookup time complexity
- Health checks are performed asynchronously to avoid blocking routing decisions
- Metrics collection has minimal overhead with configurable sampling rates
