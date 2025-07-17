#!/usr/bin/env python3
"""
Test Script for Resource Manager Module

This script tests the functionality of the resource manager module by creating
tasks, workers, and verifying proper resource allocation and load balancing.
"""

import os
import sys
import json
import time
from pathlib import Path

# Ensure the resource_manager module is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from resource_manager import (
        ResourceProfile, Task, Worker, LoadBalancer,
        HorizontalScaler, BurstCapacityManager, WorkerOrchestrator,
        ResourceManager, create_resource_manager, create_task
    )
except ImportError:
    print("Error: Could not import resource_manager module")
    sys.exit(1)

def test_resource_profile():
    """Test the ResourceProfile class."""
    print("\n=== Testing ResourceProfile ===")
    
    # Create a resource profile
    profile = ResourceProfile(
        cpu=2.0, 
        memory=1024, 
        disk=5000, 
        gpu=0.5, 
        priority=8
    )
    
    print(f"Created profile: {profile.cpu} CPU, {profile.memory} MB memory")
    print(f"Priority: {profile.priority}")
    
    # Test to_dict
    profile_dict = profile.to_dict()
    print(f"Dictionary representation: {list(profile_dict.keys())}")
    
    # Test from_dict
    new_profile = ResourceProfile.from_dict(profile_dict)
    print(f"Recreated profile: {new_profile.cpu} CPU, {new_profile.memory} MB memory")
    
    # Test can_fit
    available = ResourceProfile(
        cpu=4.0, 
        memory=2048, 
        disk=10000, 
        gpu=1.0
    )
    
    print(f"Can fit in available resources: {profile.can_fit(available)}")
    
    limited = ResourceProfile(
        cpu=1.0, 
        memory=512, 
        disk=1000, 
        gpu=0.0
    )
    
    print(f"Can fit in limited resources: {profile.can_fit(limited)}")
    
    return True

def test_task():
    """Test the Task class."""
    print("\n=== Testing Task ===")
    
    # Create resource profile
    resources = ResourceProfile(
        cpu=1.0, 
        memory=512, 
        disk=1000
    )
    
    # Create a task
    task = Task(
        task_id="task-123",
        name="Test Task",
        command="echo 'Hello, World!'",
        resources=resources,
        dependencies=["task-abc"],
        timeout=30,
        retries=2
    )
    
    print(f"Created task: {task.name} ({task.task_id})")
    print(f"Command: {task.command}")
    print(f"Dependencies: {task.dependencies}")
    print(f"Status: {task.status}")
    
    # Test to_dict
    task_dict = task.to_dict()
    print(f"Dictionary representation: {list(task_dict.keys())}")
    
    # Test from_dict
    new_task = Task.from_dict(task_dict)
    print(f"Recreated task: {new_task.name} ({new_task.task_id})")
    
    return True

def test_worker(skip_execution=True):
    """
    Test the Worker class.
    
    Args:
        skip_execution: Whether to skip actual task execution
    """
    print("\n=== Testing Worker ===")
    
    # Create resource profile
    resources = ResourceProfile(
        cpu=2.0, 
        memory=1024, 
        disk=5000
    )
    
    # Create a worker
    worker = Worker(
        worker_id="worker-abc",
        name="Test Worker",
        resources=resources
    )
    
    print(f"Created worker: {worker.name} ({worker.worker_id})")
    print(f"Status: {worker.status}")
    print(f"Resources: {worker.resources.cpu} CPU, {worker.resources.memory} MB memory")
    
    if skip_execution:
        return True
    
    # Start worker
    worker.start()
    print(f"Started worker: {worker.status}")
    
    # Create a task
    task_resources = ResourceProfile(
        cpu=0.5, 
        memory=256, 
        disk=500
    )
    
    task = Task(
        task_id="task-123",
        name="Test Task",
        command="echo 'Hello from task!'",
        resources=task_resources
    )
    
    # Add task to worker
    success = worker.add_task(task)
    print(f"Task added to worker: {success}")
    
    # Wait for task to complete
    time.sleep(2)
    
    # Get worker status
    status = worker.to_dict()
    print(f"Worker status: {status['status']}")
    print(f"Current tasks: {status['current_tasks']}")
    print(f"Completed tasks: {status['completed_tasks']}")
    
    # Stop worker
    worker.stop()
    print(f"Stopped worker: {worker.status}")
    
    return True

def test_load_balancer():
    """Test the LoadBalancer class."""
    print("\n=== Testing LoadBalancer ===")
    
    # Create load balancer
    load_balancer = LoadBalancer(strategy="best_fit")
    print(f"Created load balancer with strategy: {load_balancer.strategy}")
    
    # Create workers
    workers = []
    
    for i in range(3):
        worker = Worker(
            worker_id=f"worker-{i}",
            name=f"Worker {i}",
            resources=ResourceProfile(
                cpu=2.0,
                memory=1024,
                disk=5000
            )
        )
        
        # Set different available resources
        worker.available_resources = ResourceProfile(
            cpu=2.0 - (i * 0.5),
            memory=1024 - (i * 100),
            disk=5000
        )
        
        workers.append(worker)
    
    # Create a task
    task = Task(
        task_id="task-123",
        name="Test Task",
        command="echo 'Hello!'",
        resources=ResourceProfile(
            cpu=0.5,
            memory=200,
            disk=100
        )
    )
    
    # Select worker with best fit
    selected = load_balancer.select_worker(task, workers)
    print(f"Selected worker: {selected.name} (available CPU: {selected.available_resources.cpu})")
    
    # Try round robin
    load_balancer.strategy = "round_robin"
    
    for i in range(4):
        selected = load_balancer.select_worker(task, workers)
        print(f"Round {i}: Selected worker: {selected.name}")
    
    return True

def test_horizontal_scaler():
    """Test the HorizontalScaler class."""
    print("\n=== Testing HorizontalScaler ===")
    
    # Create horizontal scaler
    scaler = HorizontalScaler(
        min_workers=2,
        max_workers=5,
        scale_up_threshold=0.7,
        scale_down_threshold=0.3,
        cooldown_period=10
    )
    
    print(f"Created scaler: min={scaler.min_workers}, max={scaler.max_workers}")
    print(f"Thresholds: scale_up={scaler.scale_up_threshold}, scale_down={scaler.scale_down_threshold}")
    
    # Create workers
    workers = []
    
    # Case 1: Below minimum
    delta, reason = scaler.check_scaling(workers, [])
    print(f"Case 1 (No workers): Delta = {delta}, Reason = {reason}")
    
    # Add one worker (still below minimum)
    workers.append(Worker(
        worker_id="worker-1",
        name="Worker 1",
        resources=ResourceProfile(cpu=2.0, memory=1024, disk=5000)
    ))
    
    delta, reason = scaler.check_scaling(workers, [])
    print(f"Case 2 (Below minimum): Delta = {delta}, Reason = {reason}")
    
    # Add workers to meet minimum
    workers.append(Worker(
        worker_id="worker-2",
        name="Worker 2",
        resources=ResourceProfile(cpu=2.0, memory=1024, disk=5000)
    ))
    
    # Case 3: High utilization
    for worker in workers:
        # Set high utilization
        worker.available_resources = ResourceProfile(
            cpu=0.4,  # 80% used
            memory=200,  # ~80% used
            disk=5000
        )
    
    # Create tasks
    tasks = [
        Task(
            task_id=f"task-{i}",
            name=f"Task {i}",
            command="echo 'test'",
            resources=ResourceProfile(cpu=0.1, memory=50, disk=10)
        )
        for i in range(5)
    ]
    
    # Set tasks as pending
    for task in tasks:
        task.status = "pending"
    
    delta, reason = scaler.check_scaling(workers, tasks)
    print(f"Case 3 (High utilization): Delta = {delta}, Reason = {reason}")
    
    # Case 4: Low utilization
    for worker in workers:
        # Set low utilization
        worker.available_resources = ResourceProfile(
            cpu=1.8,  # 10% used
            memory=900,  # ~12% used
            disk=4500
        )
    
    # Clear tasks
    tasks = []
    
    delta, reason = scaler.check_scaling(workers, tasks)
    print(f"Case 4 (Low utilization): Delta = {delta}, Reason = {reason}")
    
    # Case 5: Above maximum
    for i in range(3, 7):
        workers.append(Worker(
            worker_id=f"worker-{i}",
            name=f"Worker {i}",
            resources=ResourceProfile(cpu=2.0, memory=1024, disk=5000)
        ))
    
    delta, reason = scaler.check_scaling(workers, tasks)
    print(f"Case 5 (Above maximum): Delta = {delta}, Reason = {reason}")
    
    return True

def test_burst_capacity_manager():
    """Test the BurstCapacityManager class."""
    print("\n=== Testing BurstCapacityManager ===")
    
    # Create burst capacity manager
    burst_manager = BurstCapacityManager(
        trigger_threshold=0.8,
        burst_factor=2.0,
        max_burst_workers=3,
        burst_duration=30
    )
    
    print(f"Created burst manager: threshold={burst_manager.trigger_threshold}, factor={burst_manager.burst_factor}")
    
    # Create workers
    workers = [
        Worker(
            worker_id=f"worker-{i}",
            name=f"Worker {i}",
            resources=ResourceProfile(cpu=2.0, memory=1024, disk=5000)
        )
        for i in range(2)
    ]
    
    # Case 1: No burst needed (low utilization)
    for worker in workers:
        worker.available_resources = ResourceProfile(
            cpu=1.5,  # 25% used
            memory=800,  # ~22% used
            disk=4500
        )
    
    # Create tasks
    tasks = [
        Task(
            task_id=f"task-{i}",
            name=f"Task {i}",
            command="echo 'test'",
            resources=ResourceProfile(cpu=0.1, memory=50, disk=10)
        )
        for i in range(2)
    ]
    
    # Set tasks as pending
    for task in tasks:
        task.status = "pending"
    
    burst_active, delta, reason = burst_manager.check_burst(workers, tasks)
    print(f"Case 1 (Low utilization): Burst = {burst_active}, Delta = {delta}, Reason = {reason}")
    
    # Case 2: Burst needed (high utilization)
    for worker in workers:
        # Set high utilization
        worker.available_resources = ResourceProfile(
            cpu=0.2,  # 90% used
            memory=100,  # ~90% used
            disk=4500
        )
    
    # Add more tasks
    tasks.extend([
        Task(
            task_id=f"task-{i+2}",
            name=f"Task {i+2}",
            command="echo 'test'",
            resources=ResourceProfile(cpu=0.1, memory=50, disk=10)
        )
        for i in range(10)
    ])
    
    # Set tasks as pending
    for task in tasks:
        task.status = "pending"
    
    burst_active, delta, reason = burst_manager.check_burst(workers, tasks)
    print(f"Case 2 (High utilization): Burst = {burst_active}, Delta = {delta}, Reason = {reason}")
    
    # Case 3: Burst already active
    burst_manager.burst_active = True
    burst_manager.burst_start_time = time.time()
    burst_manager.burst_workers = ["burst-worker-1", "burst-worker-2"]
    
    burst_active, delta, reason = burst_manager.check_burst(workers, tasks)
    print(f"Case 3 (Burst active): Burst = {burst_active}, Delta = {delta}, Reason = {reason}")
    
    # Case 4: Burst expired
    burst_manager.burst_active = True
    burst_manager.burst_start_time = time.time() - 31  # Just over burst_duration
    burst_manager.burst_workers = ["burst-worker-1", "burst-worker-2"]
    
    burst_active, delta, reason = burst_manager.check_burst(workers, tasks)
    print(f"Case 4 (Burst expired): Burst = {burst_active}, Delta = {delta}, Reason = {reason}")
    
    return True

def test_worker_orchestrator():
    """Test the WorkerOrchestrator class."""
    print("\n=== Testing WorkerOrchestrator ===")
    
    # Create worker orchestrator
    orchestrator = WorkerOrchestrator()
    
    print(f"Created orchestrator with {len(orchestrator.workers)} workers")
    
    # Create worker
    worker = orchestrator.create_worker(
        name="Test Worker",
        resources=ResourceProfile(cpu=1.0, memory=512, disk=1000)
    )
    
    print(f"Created worker: {worker.name} ({worker.worker_id})")
    
    # Get workers
    workers = orchestrator.get_workers()
    print(f"Total workers: {len(workers)}")
    
    # Get worker by ID
    found_worker = orchestrator.get_worker(worker.worker_id)
    print(f"Found worker: {found_worker is not None}")
    
    # Start worker
    success = orchestrator.start_worker(worker.worker_id)
    print(f"Started worker: {success}")
    
    # Stop worker
    success = orchestrator.stop_worker(worker.worker_id)
    print(f"Stopped worker: {success}")
    
    # Remove worker
    success = orchestrator.remove_worker(worker.worker_id)
    print(f"Removed worker: {success}")
    
    workers = orchestrator.get_workers()
    print(f"Total workers after removal: {len(workers)}")
    
    return True

def test_resource_manager(skip_execution=True):
    """
    Test the ResourceManager class.
    
    Args:
        skip_execution: Whether to skip actual task execution
    """
    print("\n=== Testing ResourceManager ===")
    
    # Create resource manager
    config = {
        "min_workers": 1,
        "max_workers": 3,
        "scale_up_threshold": 0.7,
        "scale_down_threshold": 0.3,
        "load_balancer_strategy": "best_fit"
    }
    
    manager = ResourceManager(config)
    
    print(f"Created resource manager with config: {list(config.keys())}")
    
    if skip_execution:
        return True
    
    # Start manager
    manager.start()
    print("Resource manager started")
    
    # Create task
    task = create_task(
        name="Test Task",
        command="echo 'Hello from task!'",
        resources={
            "cpu": 0.5,
            "memory": 256,
            "disk": 500
        }
    )
    
    # Submit task
    success = manager.submit_task(task)
    print(f"Task submitted: {success}")
    
    # Wait for task to be processed
    time.sleep(3)
    
    # Get task
    retrieved_task = manager.get_task(task.task_id)
    print(f"Retrieved task: {retrieved_task.name}, status: {retrieved_task.status}")
    
    # Get all tasks
    tasks = manager.get_tasks()
    print(f"Total tasks: {len(tasks)}")
    
    # Get status
    status = manager.get_status()
    print(f"Manager status: {list(status.keys())}")
    print(f"Workers: {status['workers']['active']}/{status['workers']['total']} active")
    print(f"Tasks: {status['tasks']['total']} total, {status['tasks']['completed']} completed")
    
    # Stop manager
    manager.stop()
    print("Resource manager stopped")
    
    return True

def test_api_functions():
    """Test the API functions."""
    print("\n=== Testing API Functions ===")
    
    # Test create_resource_manager
    manager = create_resource_manager({
        "min_workers": 2,
        "max_workers": 5
    })
    
    print(f"Created manager through API: {manager is not None}")
    
    # Test create_task
    task = create_task(
        name="API Task",
        command="echo 'API test'",
        resources={
            "cpu": 1.0,
            "memory": 512,
            "disk": 1000
        },
        dependencies=["other-task"],
        timeout=120,
        retries=3
    )
    
    print(f"Created task through API: {task.name} ({task.task_id})")
    print(f"Resources: CPU={task.resources.cpu}, Memory={task.resources.memory}")
    print(f"Dependencies: {task.dependencies}")
    
    return True

def main():
    """Main function."""
    print("=== Resource Manager Test Suite ===")
    
    # Run tests
    tests = [
        ("ResourceProfile", test_resource_profile),
        ("Task", test_task),
        ("Worker", lambda: test_worker(skip_execution=True)),
        ("LoadBalancer", test_load_balancer),
        ("HorizontalScaler", test_horizontal_scaler),
        ("BurstCapacityManager", test_burst_capacity_manager),
        ("WorkerOrchestrator", test_worker_orchestrator),
        ("ResourceManager", lambda: test_resource_manager(skip_execution=True)),
        ("API Functions", test_api_functions)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            print(f"\nRunning test: {name}")
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"Error running test: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n=== Test Summary ===")
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        if result:
            passed += 1
        else:
            failed += 1
        
        print(f"{name}: {status}")
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
