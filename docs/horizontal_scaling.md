# Horizontal Scaling in FixWurx

This document provides an overview of the horizontal scaling capabilities implemented in FixWurx, enabling distributed bug processing across multiple worker nodes.

## Components Overview

The horizontal scaling system consists of four main components:

1. **ScalingCoordinator** (`scaling_coordinator.py`)
2. **LoadBalancer** (`load_balancer.py`) 
3. **ClusterResourceManager** (`resource_manager_extension.py`)
4. **Test Suite** (`test_minimal.py`, `test_load_scaling.py`, etc.)

These components work together to provide seamless scaling of bug processing across multiple worker nodes while maintaining the core invariants of the FixWurx system.

## Component Details

### ScalingCoordinator

The ScalingCoordinator manages the lifecycle of worker nodes in the cluster:

- Tracks active worker nodes and their capabilities
- Monitors resource utilization to make scaling decisions
- Handles worker node health checks and failure detection
- Automatically scales up or down based on workload

Key capabilities:
- Dynamic worker registration and deregistration
- Resource-aware scaling decisions
- Configurable scaling thresholds and policies
- State persistence for recovery

### LoadBalancer

The LoadBalancer distributes bug processing tasks across worker nodes:

- Supports multiple load balancing strategies
- Tracks worker health and performance metrics
- Handles automatic failover for failed workers
- Optimizes resource utilization across the cluster

Available strategies:
- **Round Robin**: Simple rotation through available workers
- **Least Connections**: Send to worker with fewest active tasks
- **Weighted Capacity**: Balance based on worker capacity and health
- **Random**: Randomly select from available workers

### ClusterResourceManager

The ClusterResourceManager extends the base ResourceManager to support cluster operations:

- Maintains a global view of resources across all worker nodes
- Coordinates resource allocation across distributed workers
- Preserves core ResourceManager invariants in a distributed context
- Provides transparent integration with existing FixWurx components

Key benefits:
- Unified resource management interface
- Consistent allocation guarantees
- Transparent worker assignment

## Usage Example

```python
# Initialize components
base_resource_manager = ResourceManager(total_agents=9)

# Create a scaling coordinator
scaling_coordinator = ScalingCoordinator(
    min_workers=1,
    max_workers=5,
    scaling_interval_sec=60,
    resource_manager=base_resource_manager
)

# Create a load balancer
load_balancer = LoadBalancer(
    strategy=BalancingStrategy.WEIGHTED_CAPACITY,
    scaling_coordinator=scaling_coordinator
)

# Create a cluster resource manager
cluster_manager = ClusterResourceManager(
    base_resource_manager=base_resource_manager,
    scaling_coordinator=scaling_coordinator,
    load_balancer=load_balancer
)

# Start the components
scaling_coordinator.start()
load_balancer.start()
cluster_manager.start_sync()

# Use cluster_manager instead of base_resource_manager
# for all resource allocation operations
if cluster_manager.can_allocate():
    success = cluster_manager.allocate("BUG-123")
    if success:
        worker_id = cluster_manager.get_worker_for_bug("BUG-123")
        print(f"Bug allocated to worker: {worker_id}")
```

## Configuration Options

### Scaling Coordinator

- `min_workers`: Minimum number of worker nodes to maintain
- `max_workers`: Maximum number of worker nodes allowed
- `scaling_interval_sec`: Time between scaling decisions
- `state_path`: Path to persist coordinator state

### Load Balancer

- `strategy`: Load balancing strategy to use
- `health_check_interval_sec`: Interval between worker health checks

### Cluster Resource Manager

- `sync_interval_sec`: Interval for synchronizing cluster state

## Testing Horizontal Scaling

Several test scripts are provided to validate the horizontal scaling implementation:

1. `test_scaling_minimal.py`: Basic functionality test
2. `test_load_scaling.py`: Dynamic workload test to verify scaling up/down
3. `test_horizontal_scaling.py`: Comprehensive test with all features

To run the tests:

```
python test_scaling_minimal.py
python test_load_scaling.py
```

## Operational Benefits

The horizontal scaling system provides several key operational benefits:

1. **Increased throughput**: Process more bugs in parallel across multiple workers
2. **Improved resilience**: Automatic failover on worker node failures
3. **Cost efficiency**: Scale resources up/down based on actual workload
4. **Flexible deployment**: Support for heterogeneous worker nodes with different capabilities
5. **Operational simplicity**: Unified interface that preserves core ResourceManager semantics

## Integration with Existing Components

The horizontal scaling system integrates seamlessly with existing FixWurx components:

- **Parallel Executor**: Enhanced to support distributed execution
- **Scheduler**: Coordinates with ClusterResourceManager for bug assignments
- **System Monitor**: Tracks metrics across all worker nodes
- **Triangulation Engine**: Works transparently across distributed workers
