#!/usr/bin/env python3
"""
Resource Manager Module

This module provides resource allocation, load balancing, horizontal scaling,
burst capacity management, and worker orchestration for the FixWurx system.
"""

import os
import sys
import json
import logging
import time
import multiprocessing
import threading
import queue
import uuid
import subprocess
import psutil
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Constants
AGENTS_PER_BUG = 3  # Default number of agents allocated per bug

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("resource_manager.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ResourceManager")

class ResourceProfile:
    """
    Resource profile for a task or worker.
    """
    
    def __init__(self, cpu: float = 1.0, memory: int = 512, disk: int = 100, 
                 gpu: float = 0.0, priority: int = 1):
        """
        Initialize resource profile.
        
        Args:
            cpu: CPU cores required (can be fractional)
            memory: Memory required in MB
            disk: Disk space required in MB
            gpu: GPU cores required (can be fractional)
            priority: Priority level (1-10, higher is more important)
        """
        self.cpu = cpu
        self.memory = memory
        self.disk = disk
        self.gpu = gpu
        self.priority = max(1, min(10, priority))  # Clamp to 1-10
        
        logger.debug(f"Created resource profile: CPU={cpu}, Memory={memory}MB, Disk={disk}MB, GPU={gpu}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "cpu": self.cpu,
            "memory": self.memory,
            "disk": self.disk,
            "gpu": self.gpu,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceProfile':
        """
        Create a resource profile from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Resource profile
        """
        return cls(
            cpu=data.get("cpu", 1.0),
            memory=data.get("memory", 512),
            disk=data.get("disk", 100),
            gpu=data.get("gpu", 0.0),
            priority=data.get("priority", 1)
        )
    
    def can_fit(self, available: 'ResourceProfile') -> bool:
        """
        Check if this profile can fit within available resources.
        
        Args:
            available: Available resources
            
        Returns:
            Whether this profile can fit
        """
        return (
            self.cpu <= available.cpu and
            self.memory <= available.memory and
            self.disk <= available.disk and
            self.gpu <= available.gpu
        )

class Task:
    """
    A task to be executed with resource requirements.
    """
    
    def __init__(self, task_id: str, name: str, command: str, 
                 resources: ResourceProfile, dependencies: List[str] = None,
                 timeout: int = 3600, retries: int = 0, callback: Callable = None):
        """
        Initialize task.
        
        Args:
            task_id: Unique task identifier
            name: Task name
            command: Command to execute
            resources: Resource requirements
            dependencies: List of task IDs that must complete before this task
            timeout: Timeout in seconds
            retries: Number of retries if task fails
            callback: Callback function to call when task completes
        """
        self.task_id = task_id
        self.name = name
        self.command = command
        self.resources = resources
        self.dependencies = dependencies or []
        self.timeout = timeout
        self.retries = retries
        self.callback = callback
        
        self.status = "pending"  # pending, running, completed, failed
        self.result = None
        self.start_time = None
        self.end_time = None
        self.worker_id = None
        self.retry_count = 0
        
        logger.debug(f"Created task {task_id}: {name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "task_id": self.task_id,
            "name": self.name,
            "command": self.command,
            "resources": self.resources.to_dict(),
            "dependencies": self.dependencies,
            "timeout": self.timeout,
            "retries": self.retries,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "worker_id": self.worker_id,
            "retry_count": self.retry_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """
        Create a task from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Task
        """
        resources = ResourceProfile.from_dict(data.get("resources", {}))
        
        task = cls(
            task_id=data.get("task_id", str(uuid.uuid4())),
            name=data.get("name", "Unnamed Task"),
            command=data.get("command", ""),
            resources=resources,
            dependencies=data.get("dependencies", []),
            timeout=data.get("timeout", 3600),
            retries=data.get("retries", 0)
        )
        
        task.status = data.get("status", "pending")
        task.start_time = data.get("start_time")
        task.end_time = data.get("end_time")
        task.worker_id = data.get("worker_id")
        task.retry_count = data.get("retry_count", 0)
        
        return task

class Worker:
    """
    A worker that executes tasks.
    """
    
    def __init__(self, worker_id: str, name: str, resources: ResourceProfile,
                 status: str = "idle", task_queue: queue.Queue = None):
        """
        Initialize worker.
        
        Args:
            worker_id: Unique worker identifier
            name: Worker name
            resources: Available resources
            status: Initial status (idle, busy, offline)
            task_queue: Task queue
        """
        self.worker_id = worker_id
        self.name = name
        self.resources = resources
        self.status = status
        self.task_queue = task_queue or queue.Queue()
        self.available_resources = ResourceProfile(
            cpu=resources.cpu,
            memory=resources.memory,
            disk=resources.disk,
            gpu=resources.gpu
        )
        self.current_tasks = {}  # task_id -> Task
        self.completed_tasks = []
        self.failed_tasks = []
        self.last_heartbeat = time.time()
        
        self.thread = threading.Thread(target=self._worker_loop)
        self.thread.daemon = True
        
        logger.debug(f"Created worker {worker_id}: {name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "worker_id": self.worker_id,
            "name": self.name,
            "resources": self.resources.to_dict(),
            "available_resources": self.available_resources.to_dict(),
            "status": self.status,
            "current_tasks": len(self.current_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "last_heartbeat": self.last_heartbeat
        }
    
    def start(self) -> None:
        """
        Start the worker thread.
        """
        if not self.thread.is_alive():
            self.thread = threading.Thread(target=self._worker_loop)
            self.thread.daemon = True
            self.thread.start()
            logger.info(f"Worker {self.worker_id} started")
    
    def stop(self) -> None:
        """
        Stop the worker thread.
        """
        self.status = "offline"
        logger.info(f"Worker {self.worker_id} stopped")
    
    def add_task(self, task: Task) -> bool:
        """
        Add a task to the worker's queue.
        
        Args:
            task: Task to add
            
        Returns:
            Whether the task was added
        """
        if self.status == "offline":
            logger.warning(f"Cannot add task to offline worker {self.worker_id}")
            return False
        
        if not task.resources.can_fit(self.available_resources):
            logger.warning(f"Task {task.task_id} does not fit in worker {self.worker_id} resources")
            return False
        
        self.task_queue.put(task)
        logger.info(f"Task {task.task_id} added to worker {self.worker_id} queue")
        return True
    
    def heartbeat(self) -> Dict[str, Any]:
        """
        Send a heartbeat to update worker status.
        
        Returns:
            Worker status
        """
        self.last_heartbeat = time.time()
        
        # Update available resources based on system metrics
        if self.status != "offline":
            try:
                cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
                memory_usage = psutil.virtual_memory().percent / 100.0
                disk_usage = psutil.disk_usage('/').percent / 100.0
                
                self.available_resources = ResourceProfile(
                    cpu=self.resources.cpu * (1.0 - cpu_usage),
                    memory=self.resources.memory * (1.0 - memory_usage),
                    disk=self.resources.disk * (1.0 - disk_usage),
                    gpu=self.resources.gpu
                )
            except Exception as e:
                logger.error(f"Error updating worker resources: {e}")
        
        return self.to_dict()
    
    def _worker_loop(self) -> None:
        """
        Main worker loop.
        """
        self.status = "idle"
        
        while self.status != "offline":
            try:
                # Check for new tasks
                try:
                    task = self.task_queue.get(block=True, timeout=1)
                    self._execute_task(task)
                except queue.Empty:
                    # No tasks available
                    pass
                
                # Update status based on current tasks
                if self.current_tasks:
                    self.status = "busy"
                else:
                    self.status = "idle"
                
                # Clean up completed tasks
                self._cleanup_tasks()
                
                # Send heartbeat
                self.heartbeat()
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)
    
    def _execute_task(self, task: Task) -> None:
        """
        Execute a task.
        
        Args:
            task: Task to execute
        """
        # Check if we have enough resources
        if not task.resources.can_fit(self.available_resources):
            logger.warning(f"Task {task.task_id} does not fit in worker {self.worker_id} resources")
            self.task_queue.put(task)  # Put back in queue
            return
        
        # Update task status
        task.status = "running"
        task.start_time = time.time()
        task.worker_id = self.worker_id
        
        # Reserve resources
        self.available_resources.cpu -= task.resources.cpu
        self.available_resources.memory -= task.resources.memory
        self.available_resources.disk -= task.resources.disk
        self.available_resources.gpu -= task.resources.gpu
        
        # Add to current tasks
        self.current_tasks[task.task_id] = task
        
        # Run task in a separate thread
        thread = threading.Thread(target=self._run_task, args=(task,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Task {task.task_id} started on worker {self.worker_id}")
    
    def _run_task(self, task: Task) -> None:
        """
        Run a task and capture the result.
        
        Args:
            task: Task to run
        """
        try:
            # Execute command
            process = subprocess.Popen(
                task.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            try:
                stdout, _ = process.communicate(timeout=task.timeout)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, _ = process.communicate()
                exit_code = -1
                logger.warning(f"Task {task.task_id} timed out after {task.timeout} seconds")
            
            # Update task status
            task.end_time = time.time()
            
            if exit_code == 0:
                task.status = "completed"
                self.completed_tasks.append(task.task_id)
            else:
                # Check if we should retry
                if task.retry_count < task.retries:
                    task.retry_count += 1
                    task.status = "pending"
                    logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.retries})")
                    self.task_queue.put(task)
                else:
                    task.status = "failed"
                    self.failed_tasks.append(task.task_id)
            
            # Store result
            task.result = {
                "exit_code": exit_code,
                "output": stdout,
                "duration": task.end_time - task.start_time
            }
            
            # Call callback if provided
            if task.callback:
                try:
                    task.callback(task)
                except Exception as e:
                    logger.error(f"Error in task callback: {e}")
            
            logger.info(f"Task {task.task_id} completed with status {task.status}")
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            task.end_time = time.time()
            task.status = "failed"
            task.result = {
                "exit_code": -1,
                "output": str(e),
                "duration": task.end_time - task.start_time
            }
            self.failed_tasks.append(task.task_id)
    
    def _cleanup_tasks(self) -> None:
        """
        Clean up completed and failed tasks.
        """
        for task_id in list(self.current_tasks.keys()):
            task = self.current_tasks[task_id]
            
            if task.status in ["completed", "failed"]:
                # Release resources
                self.available_resources.cpu += task.resources.cpu
                self.available_resources.memory += task.resources.memory
                self.available_resources.disk += task.resources.disk
                self.available_resources.gpu += task.resources.gpu
                
                # Remove from current tasks
                del self.current_tasks[task_id]

class LoadBalancer:
    """
    Load balancer for distributing tasks among workers.
    """
    
    def __init__(self, strategy: str = "best_fit"):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy (best_fit, round_robin, random)
        """
        self.strategy = strategy
        self.last_worker_index = -1
        
        logger.debug(f"Created load balancer with strategy {strategy}")
    
    def select_worker(self, task: Task, workers: List[Worker]) -> Optional[Worker]:
        """
        Select a worker for a task.
        
        Args:
            task: Task to assign
            workers: Available workers
            
        Returns:
            Selected worker, or None if no suitable worker found
        """
        # Filter idle or busy workers that can accommodate the task
        available_workers = [
            w for w in workers
            if w.status != "offline" and task.resources.can_fit(w.available_resources)
        ]
        
        if not available_workers:
            logger.warning(f"No available workers for task {task.task_id}")
            return None
        
        # Apply load balancing strategy
        if self.strategy == "best_fit":
            # Choose worker with least available resources that can fit the task
            return min(
                available_workers,
                key=lambda w: (
                    w.available_resources.cpu - task.resources.cpu,
                    w.available_resources.memory - task.resources.memory
                )
            )
        elif self.strategy == "round_robin":
            # Choose workers in circular order
            self.last_worker_index = (self.last_worker_index + 1) % len(available_workers)
            return available_workers[self.last_worker_index]
        elif self.strategy == "random":
            # Choose a random worker
            import random
            return random.choice(available_workers)
        else:
            # Default to best fit
            return min(
                available_workers,
                key=lambda w: (
                    w.available_resources.cpu - task.resources.cpu,
                    w.available_resources.memory - task.resources.memory
                )
            )

class HorizontalScaler:
    """
    Horizontal scaler for scaling workers based on demand.
    """
    
    def __init__(self, min_workers: int = 1, max_workers: int = 10,
                 scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.2,
                 cooldown_period: int = 60):
        """
        Initialize horizontal scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            scale_up_threshold: Resource utilization threshold to scale up (0.0-1.0)
            scale_down_threshold: Resource utilization threshold to scale down (0.0-1.0)
            cooldown_period: Cooldown period in seconds between scaling actions
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.last_scale_time = 0
        
        logger.debug(f"Created horizontal scaler: min={min_workers}, max={max_workers}")
    
    def check_scaling(self, workers: List[Worker], tasks: List[Task]) -> Tuple[int, str]:
        """
        Check if scaling is needed.
        
        Args:
            workers: Current workers
            tasks: Current tasks
            
        Returns:
            Tuple of (number of workers to add/remove, reason)
        """
        # Skip if in cooldown period
        if time.time() - self.last_scale_time < self.cooldown_period:
            return 0, "In cooldown period"
        
        # Count active workers
        active_workers = [w for w in workers if w.status != "offline"]
        
        # If below minimum, scale up to minimum
        if len(active_workers) < self.min_workers:
            return self.min_workers - len(active_workers), "Below minimum workers"
        
        # If above maximum, scale down to maximum
        if len(active_workers) > self.max_workers:
            return len(active_workers) - self.max_workers, "Above maximum workers"
        
        # Calculate current utilization
        if not active_workers:
            return 0, "No active workers"
        
        total_cpu = sum(w.resources.cpu for w in active_workers)
        total_memory = sum(w.resources.memory for w in active_workers)
        available_cpu = sum(w.available_resources.cpu for w in active_workers)
        available_memory = sum(w.available_resources.memory for w in active_workers)
        
        if total_cpu == 0 or total_memory == 0:
            return 0, "No resources available"
        
        cpu_utilization = 1.0 - (available_cpu / total_cpu)
        memory_utilization = 1.0 - (available_memory / total_memory)
        
        # Use higher of CPU or memory utilization
        utilization = max(cpu_utilization, memory_utilization)
        
        # Check pending tasks
        pending_tasks = [t for t in tasks if t.status == "pending"]
        
        # Scale up if high utilization and below max workers
        if utilization >= self.scale_up_threshold and len(active_workers) < self.max_workers:
            self.last_scale_time = time.time()
            return 1, f"High utilization ({utilization:.2f}) and {len(pending_tasks)} pending tasks"
        
        # Scale down if low utilization, no pending tasks, and above min workers
        if utilization <= self.scale_down_threshold and len(pending_tasks) == 0 and len(active_workers) > self.min_workers:
            self.last_scale_time = time.time()
            return -1, f"Low utilization ({utilization:.2f}) and no pending tasks"
        
        return 0, f"No scaling needed, utilization: {utilization:.2f}"

class BurstCapacityManager:
    """
    Burst capacity manager for handling sudden increases in load.
    """
    
    def __init__(self, trigger_threshold: float = 0.9, burst_factor: float = 2.0,
                 max_burst_workers: int = 5, burst_duration: int = 300):
        """
        Initialize burst capacity manager.
        
        Args:
            trigger_threshold: Utilization threshold to trigger burst (0.0-1.0)
            burst_factor: Factor to multiply capacity by during burst
            max_burst_workers: Maximum number of burst workers
            burst_duration: Duration of burst in seconds
        """
        self.trigger_threshold = trigger_threshold
        self.burst_factor = burst_factor
        self.max_burst_workers = max_burst_workers
        self.burst_duration = burst_duration
        self.burst_active = False
        self.burst_start_time = 0
        self.burst_workers = []
        
        logger.debug(f"Created burst capacity manager: threshold={trigger_threshold}, factor={burst_factor}")
    
    def check_burst(self, workers: List[Worker], tasks: List[Task]) -> Tuple[bool, int, str]:
        """
        Check if burst capacity is needed.
        
        Args:
            workers: Current workers
            tasks: Current tasks
            
        Returns:
            Tuple of (burst active, number of workers to add/remove, reason)
        """
        # Count active workers
        active_workers = [w for w in workers if w.status != "offline"]
        
        # Calculate current utilization
        if not active_workers:
            return self.burst_active, 0, "No active workers"
        
        total_cpu = sum(w.resources.cpu for w in active_workers)
        total_memory = sum(w.resources.memory for w in active_workers)
        available_cpu = sum(w.available_resources.cpu for w in active_workers)
        available_memory = sum(w.available_resources.memory for w in active_workers)
        
        if total_cpu == 0 or total_memory == 0:
            return self.burst_active, 0, "No resources available"
        
        cpu_utilization = 1.0 - (available_cpu / total_cpu)
        memory_utilization = 1.0 - (available_memory / total_memory)
        
        # Use higher of CPU or memory utilization
        utilization = max(cpu_utilization, memory_utilization)
        
        # Check pending tasks
        pending_tasks = [t for t in tasks if t.status == "pending"]
        
        # Check if burst is already active
        if self.burst_active:
            # Check if burst should be deactivated
            if time.time() - self.burst_start_time >= self.burst_duration:
                self.burst_active = False
                return False, -len(self.burst_workers), "Burst duration expired"
            
            # Keep burst active
            return True, 0, f"Burst active for {int(time.time() - self.burst_start_time)} seconds"
        
        # Check if burst should be activated
        if utilization >= self.trigger_threshold and len(pending_tasks) > 0:
            # Calculate number of burst workers needed
            current_capacity = sum(w.resources.cpu for w in active_workers)
            required_capacity = current_capacity * self.burst_factor
            additional_capacity = required_capacity - current_capacity
            
            # Assume each worker has the same capacity as average of existing workers
            avg_worker_capacity = current_capacity / len(active_workers)
            additional_workers = int(additional_capacity / avg_worker_capacity)
            
            # Clamp to max burst workers
            additional_workers = min(additional_workers, self.max_burst_workers)
            
            if additional_workers > 0:
                self.burst_active = True
                self.burst_start_time = time.time()
                self.burst_workers = []  # Will be filled by ResourceManager
                return True, additional_workers, f"Burst activated due to high utilization ({utilization:.2f}) and {len(pending_tasks)} pending tasks"
        
        return False, 0, f"No burst needed, utilization: {utilization:.2f}"

class WorkerOrchestrator:
    """
    Worker orchestrator for managing worker lifecycle.
    """
    
    def __init__(self, worker_configs: List[Dict[str, Any]] = None):
        """
        Initialize worker orchestrator.
        
        Args:
            worker_configs: List of worker configurations
        """
        self.worker_configs = worker_configs or []
        self.workers = {}
        
        logger.debug(f"Created worker orchestrator with {len(self.worker_configs)} worker configs")
    
    def create_worker(self, name: str = None, resources: ResourceProfile = None) -> Worker:
        """
        Create a new worker.
        
        Args:
            name: Worker name (or None for auto-generated)
            resources: Worker resources (or None for default)
            
        Returns:
            Created worker
        """
        worker_id = str(uuid.uuid4())
        name = name or f"Worker-{worker_id[:8]}"
        
        if not resources:
            # Use default resources based on system
            cpu_count = multiprocessing.cpu_count()
            total_memory = psutil.virtual_memory().total // (1024 * 1024)  # MB
            total_disk = psutil.disk_usage('/').total // (1024 * 1024)  # MB
            
            resources = ResourceProfile(
                cpu=max(1, cpu_count // 2),
                memory=max(512, total_memory // 4),
                disk=max(1024, total_disk // 10),
                gpu=0.0
            )
        
        worker = Worker(worker_id, name, resources)
        self.workers[worker_id] = worker
        
        logger.info(f"Created worker {worker_id}: {name} with {resources.cpu} CPU, {resources.memory} MB memory")
        
        return worker
    
    def start_worker(self, worker_id: str) -> bool:
        """
        Start a worker.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Whether the worker was started
        """
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found")
            return False
        
        worker = self.workers[worker_id]
        worker.start()
        return True
    
    def stop_worker(self, worker_id: str) -> bool:
        """
        Stop a worker.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Whether the worker was stopped
        """
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found")
            return False
        
        worker = self.workers[worker_id]
        worker.stop()
        return True
    
    def remove_worker(self, worker_id: str) -> bool:
        """
        Remove a worker.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Whether the worker was removed
        """
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found")
            return False
        
        worker = self.workers[worker_id]
        worker.stop()
        del self.workers[worker_id]
        return True
    
    def get_workers(self) -> List[Worker]:
        """
        Get all workers.
        
        Returns:
            List of workers
        """
        return list(self.workers.values())
    
    def get_worker(self, worker_id: str) -> Optional[Worker]:
        """
        Get a worker by ID.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Worker, or None if not found
        """
        return self.workers.get(worker_id)

class ResourceManager:
    """
    Main resource manager class that combines all resource management components.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize resource manager.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Initialize components
        self.orchestrator = WorkerOrchestrator(self.config.get("worker_configs", []))
        self.load_balancer = LoadBalancer(self.config.get("load_balancer_strategy", "best_fit"))
        self.horizontal_scaler = HorizontalScaler(
            min_workers=self.config.get("min_workers", 1),
            max_workers=self.config.get("max_workers", 10),
            scale_up_threshold=self.config.get("scale_up_threshold", 0.8),
            scale_down_threshold=self.config.get("scale_down_threshold", 0.2),
            cooldown_period=self.config.get("cooldown_period", 60)
        )
        self.burst_manager = BurstCapacityManager(
            trigger_threshold=self.config.get("burst_trigger_threshold", 0.9),
            burst_factor=self.config.get("burst_factor", 2.0),
            max_burst_workers=self.config.get("max_burst_workers", 5),
            burst_duration=self.config.get("burst_duration", 300)
        )
        
        # Task management
        self.tasks = {}  # task_id -> Task
        self.task_queue = queue.PriorityQueue()  # Priority queue for tasks
        
        # Create initial workers
        self._create_initial_workers()
        
        # Start management thread
        self.thread = threading.Thread(target=self._management_loop)
        self.thread.daemon = True
        self.running = False
        
        logger.info("Resource manager initialized")
    
    def _create_initial_workers(self) -> None:
        """
        Create initial workers based on configuration.
        """
        min_workers = self.config.get("min_workers", 1)
        
        for i in range(min_workers):
            worker = self.orchestrator.create_worker()
            worker.start()
    
    def start(self) -> None:
        """
        Start the resource manager.
        """
        if not self.running:
            self.running = True
            self.thread.start()
            logger.info("Resource manager started")
    
    def stop(self) -> None:
        """
        Stop the resource manager.
        """
        self.running = False
        
        # Stop all workers
        for worker in self.orchestrator.get_workers():
            worker.stop()
        
        logger.info("Resource manager stopped")
    
    def submit_task(self, task: Task) -> bool:
        """
        Submit a task for execution.
        
        Args:
            task: Task to submit
            
        Returns:
            Whether the task was submitted successfully
        """
        # Check if task already exists
        if task.task_id in self.tasks:
            logger.warning(f"Task {task.task_id} already exists")
            return False
        
        # Add task to registry
        self.tasks[task.task_id] = task
        
        # Add to priority queue
        # Priority is determined by task priority (higher value = higher priority)
        self.task_queue.put((-task.resources.priority, task.task_id, task))
        
        logger.info(f"Task {task.task_id} submitted")
        return True
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task, or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_tasks(self) -> List[Task]:
        """
        Get all tasks.
        
        Returns:
            List of tasks
        """
        return list(self.tasks.values())
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Whether the task was cancelled
        """
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found")
            return False
        
        task = self.tasks[task_id]
        
        # If task is running, set status to cancelled
        if task.status == "running":
            task.status = "cancelled"
        # If task is pending, remove from queue (harder to do with a priority queue)
        elif task.status == "pending":
            task.status = "cancelled"
        
        logger.info(f"Task {task_id} cancelled")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get resource manager status.
        
        Returns:
            Status information
        """
        workers = self.orchestrator.get_workers()
        active_workers = [w for w in workers if w.status != "offline"]
        
        # Count tasks by status
        pending_tasks = [t for t in self.tasks.values() if t.status == "pending"]
        running_tasks = [t for t in self.tasks.values() if t.status == "running"]
        completed_tasks = [t for t in self.tasks.values() if t.status == "completed"]
        failed_tasks = [t for t in self.tasks.values() if t.status == "failed"]
        
        # Calculate utilization
        total_cpu = sum(w.resources.cpu for w in active_workers) if active_workers else 0
        total_memory = sum(w.resources.memory for w in active_workers) if active_workers else 0
        available_cpu = sum(w.available_resources.cpu for w in active_workers) if active_workers else 0
        available_memory = sum(w.available_resources.memory for w in active_workers) if active_workers else 0
        
        cpu_utilization = 1.0 - (available_cpu / total_cpu) if total_cpu > 0 else 0.0
        memory_utilization = 1.0 - (available_memory / total_memory) if total_memory > 0 else 0.0
        
        return {
            "workers": {
                "total": len(workers),
                "active": len(active_workers),
                "idle": len([w for w in active_workers if w.status == "idle"]),
                "busy": len([w for w in active_workers if w.status == "busy"])
            },
            "tasks": {
                "total": len(self.tasks),
                "pending": len(pending_tasks),
                "running": len(running_tasks),
                "completed": len(completed_tasks),
                "failed": len(failed_tasks)
            },
            "resources": {
                "cpu_utilization": cpu_utilization,
                "memory_utilization": memory_utilization,
                "total_cpu": total_cpu,
                "total_memory": total_memory,
                "available_cpu": available_cpu,
                "available_memory": available_memory
            },
            "burst": {
                "active": self.burst_manager.burst_active,
                "burst_workers": len(self.burst_manager.burst_workers)
            }
        }
    
    def _management_loop(self) -> None:
        """
        Main resource manager loop.
        """
        while self.running:
            try:
                # Check for scaling
                delta, reason = self.horizontal_scaler.check_scaling(
                    self.orchestrator.get_workers(), 
                    list(self.tasks.values())
                )
                
                if delta > 0:
                    # Scale up
                    logger.info(f"Scaling up by {delta} workers: {reason}")
                    for i in range(delta):
                        worker = self.orchestrator.create_worker()
                        worker.start()
                elif delta < 0:
                    # Scale down
                    logger.info(f"Scaling down by {abs(delta)} workers: {reason}")
                    
                    # Get idle workers
                    idle_workers = [
                        w for w in self.orchestrator.get_workers()
                        if w.status == "idle"
                    ]
                    
                    # Remove up to abs(delta) idle workers
                    for i in range(min(abs(delta), len(idle_workers))):
                        self.orchestrator.remove_worker(idle_workers[i].worker_id)
                
                # Check for burst capacity
                burst_active, burst_delta, burst_reason = self.burst_manager.check_burst(
                    self.orchestrator.get_workers(), 
                    list(self.tasks.values())
                )
                
                if burst_delta > 0:
                    # Add burst workers
                    logger.info(f"Adding {burst_delta} burst workers: {burst_reason}")
                    for i in range(burst_delta):
                        worker = self.orchestrator.create_worker(name=f"Burst-Worker-{i}")
                        worker.start()
                        self.burst_manager.burst_workers.append(worker.worker_id)
                elif burst_delta < 0:
                    # Remove burst workers
                    logger.info(f"Removing {abs(burst_delta)} burst workers: {burst_reason}")
                    for worker_id in self.burst_manager.burst_workers:
                        self.orchestrator.remove_worker(worker_id)
                    self.burst_manager.burst_workers = []
                
                # Process pending tasks
                self._assign_tasks()
                
                # Sleep briefly
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in management loop: {e}")
                time.sleep(1)
    
    def _assign_tasks(self) -> None:
        """
        Assign pending tasks to workers.
        """
        # Get available workers
        workers = self.orchestrator.get_workers()
        available_workers = [w for w in workers if w.status != "offline"]
        
        if not available_workers:
            return
        
        # Get pending tasks
        pending_tasks = []
        
        # Try to get tasks from queue without removing them
        while not self.task_queue.empty():
            try:
                # Get the next task from the queue
                _, task_id, task = self.task_queue.get(block=False)
                
                # Skip if task is not pending
                if task.status != "pending":
                    continue
                
                # Check if all dependencies are satisfied
                dependencies_satisfied = True
                for dep_id in task.dependencies:
                    if dep_id not in self.tasks:
                        logger.warning(f"Dependency {dep_id} not found for task {task.task_id}")
                        dependencies_satisfied = False
                        break
                    
                    dep_task = self.tasks[dep_id]
                    if dep_task.status != "completed":
                        dependencies_satisfied = False
                        break
                
                if dependencies_satisfied:
                    pending_tasks.append(task)
                else:
                    # Put back in queue
                    self.task_queue.put((-task.resources.priority, task.task_id, task))
                
            except queue.Empty:
                break
        
        # Assign tasks to workers
        for task in pending_tasks:
            # Select worker
            worker = self.load_balancer.select_worker(task, available_workers)
            
            if worker:
                # Assign task to worker
                if worker.add_task(task):
                    logger.info(f"Task {task.task_id} assigned to worker {worker.worker_id}")
                else:
                    # Put back in queue
                    self.task_queue.put((-task.resources.priority, task.task_id, task))
            else:
                # No suitable worker found, put back in queue
                self.task_queue.put((-task.resources.priority, task.task_id, task))


# API Functions

def create_resource_manager(config: Dict[str, Any] = None) -> ResourceManager:
    """
    Create and initialize a resource manager.
    
    Args:
        config: Configuration options
        
    Returns:
        Initialized resource manager
    """
    manager = ResourceManager(config)
    return manager

def create_task(name: str, command: str, resources: Dict[str, Any] = None,
               dependencies: List[str] = None, timeout: int = 3600, 
               retries: int = 0) -> Task:
    """
    Create a task for execution.
    
    Args:
        name: Task name
        command: Command to execute
        resources: Resource requirements
        dependencies: List of task IDs that must complete before this task
        timeout: Timeout in seconds
        retries: Number of retries if task fails
        
    Returns:
        Created task
    """
    task_id = str(uuid.uuid4())
    resource_profile = ResourceProfile.from_dict(resources or {})
    
    task = Task(
        task_id=task_id,
        name=name,
        command=command,
        resources=resource_profile,
        dependencies=dependencies,
        timeout=timeout,
        retries=retries
    )
    
    return task

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Resource Manager")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--min-workers", type=int, default=1, help="Minimum number of workers")
    parser.add_argument("--max-workers", type=int, default=10, help="Maximum number of workers")
    parser.add_argument("--status", action="store_true", help="Show resource manager status")
    parser.add_argument("--submit", help="Submit a task (command)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    # Override configuration with command-line arguments
    if args.min_workers:
        config["min_workers"] = args.min_workers
    if args.max_workers:
        config["max_workers"] = args.max_workers
    
    # Create resource manager
    manager = create_resource_manager(config)
    manager.start()
    
    try:
        if args.status:
            # Show status
            status = manager.get_status()
            
            print("\nResource Manager Status:")
            print(f"Workers: {status['workers']['active']}/{status['workers']['total']} active")
            print(f"Tasks: {status['tasks']['total']} total, {status['tasks']['pending']} pending, {status['tasks']['running']} running")
            print(f"CPU: {status['resources']['cpu_utilization']:.2f} utilization")
            print(f"Memory: {status['resources']['memory_utilization']:.2f} utilization")
            
        elif args.submit:
            # Submit a task
            task = create_task("Command Line Task", args.submit)
            
            if manager.submit_task(task):
                print(f"Task submitted: {task.task_id}")
            else:
                print("Failed to submit task")
        else:
            # Run interactively
            print("Resource Manager started. Press Ctrl+C to stop.")
            
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Resource Manager...")
    finally:
        manager.stop()
