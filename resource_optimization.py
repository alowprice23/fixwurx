#!/usr/bin/env python3
"""
resource_optimization.py
───────────────────────
Implements comprehensive resource optimization for the FixWurx platform.

This module provides advanced resource management, allocation, and optimization
capabilities for the system, including dynamic scaling, performance monitoring,
and resource usage tracking.
"""

import os
import sys
import time
import json
import yaml
import logging
import threading
import multiprocessing
import psutil
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field

# Import system configuration
from system_configuration import get_config, ConfigurationError

# Configure logging
logger = logging.getLogger("ResourceOptimization")

# Default values for resource limits
DEFAULT_MAX_CPU_PERCENT = 80
DEFAULT_MAX_MEMORY_PERCENT = 75
DEFAULT_MAX_DISK_PERCENT = 90
DEFAULT_MAX_THREADS = multiprocessing.cpu_count() * 2
DEFAULT_MAX_PROCESSES = multiprocessing.cpu_count()
DEFAULT_THREAD_POOL_SIZE = multiprocessing.cpu_count() * 4
DEFAULT_PROCESS_POOL_SIZE = multiprocessing.cpu_count()

# Constants for optimization strategies
STRATEGY_PERFORMANCE = "performance"
STRATEGY_BALANCED = "balanced"
STRATEGY_CONSERVATIVE = "conservative"
STRATEGY_MINIMAL = "minimal"
STRATEGY_CUSTOM = "custom"

@dataclass
class ResourceUsage:
    """Class for tracking resource usage."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used: int = 0
    memory_total: int = 0
    disk_percent: float = 0.0
    disk_used: int = 0
    disk_total: int = 0
    threads_active: int = 0
    processes_active: int = 0
    network_sent_bytes: int = 0
    network_recv_bytes: int = 0
    timestamp: float = field(default_factory=time.time)
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert resource usage to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used": self.memory_used,
            "memory_total": self.memory_total,
            "disk_percent": self.disk_percent,
            "disk_used": self.disk_used,
            "disk_total": self.disk_total,
            "threads_active": self.threads_active,
            "processes_active": self.processes_active,
            "network_sent_bytes": self.network_sent_bytes,
            "network_recv_bytes": self.network_recv_bytes,
            "timestamp": self.timestamp,
            "io_read_bytes": self.io_read_bytes,
            "io_write_bytes": self.io_write_bytes
        }

@dataclass
class ResourceLimits:
    """Class for defining resource limits."""
    max_cpu_percent: float = DEFAULT_MAX_CPU_PERCENT
    max_memory_percent: float = DEFAULT_MAX_MEMORY_PERCENT
    max_disk_percent: float = DEFAULT_MAX_DISK_PERCENT
    max_threads: int = DEFAULT_MAX_THREADS
    max_processes: int = DEFAULT_MAX_PROCESSES
    thread_pool_size: int = DEFAULT_THREAD_POOL_SIZE
    process_pool_size: int = DEFAULT_PROCESS_POOL_SIZE
    io_throttle_threshold: float = 0.8  # Throttle IO when disk usage > 80%
    network_throttle_threshold: float = 0.9  # Throttle network when usage > 90%
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert resource limits to dictionary."""
        return {
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_percent": self.max_memory_percent,
            "max_disk_percent": self.max_disk_percent,
            "max_threads": self.max_threads,
            "max_processes": self.max_processes,
            "thread_pool_size": self.thread_pool_size,
            "process_pool_size": self.process_pool_size,
            "io_throttle_threshold": self.io_throttle_threshold,
            "network_throttle_threshold": self.network_throttle_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceLimits':
        """Create resource limits from dictionary."""
        return cls(
            max_cpu_percent=data.get("max_cpu_percent", DEFAULT_MAX_CPU_PERCENT),
            max_memory_percent=data.get("max_memory_percent", DEFAULT_MAX_MEMORY_PERCENT),
            max_disk_percent=data.get("max_disk_percent", DEFAULT_MAX_DISK_PERCENT),
            max_threads=data.get("max_threads", DEFAULT_MAX_THREADS),
            max_processes=data.get("max_processes", DEFAULT_MAX_PROCESSES),
            thread_pool_size=data.get("thread_pool_size", DEFAULT_THREAD_POOL_SIZE),
            process_pool_size=data.get("process_pool_size", DEFAULT_PROCESS_POOL_SIZE),
            io_throttle_threshold=data.get("io_throttle_threshold", 0.8),
            network_throttle_threshold=data.get("network_throttle_threshold", 0.9)
        )

@dataclass
class ComponentResource:
    """Class for tracking component resource allocation."""
    component_id: str
    component_type: str
    cpu_allocation: float = 0.0  # Percentage of total CPU
    memory_allocation: float = 0.0  # Percentage of total memory
    disk_allocation: float = 0.0  # Percentage of total disk
    thread_allocation: int = 0
    process_allocation: int = 0
    priority: int = 0  # 0-100, higher is more important
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert component resource to dictionary."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "cpu_allocation": self.cpu_allocation,
            "memory_allocation": self.memory_allocation,
            "disk_allocation": self.disk_allocation,
            "thread_allocation": self.thread_allocation,
            "process_allocation": self.process_allocation,
            "priority": self.priority,
            "active": self.active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentResource':
        """Create component resource from dictionary."""
        return cls(
            component_id=data["component_id"],
            component_type=data["component_type"],
            cpu_allocation=data.get("cpu_allocation", 0.0),
            memory_allocation=data.get("memory_allocation", 0.0),
            disk_allocation=data.get("disk_allocation", 0.0),
            thread_allocation=data.get("thread_allocation", 0),
            process_allocation=data.get("process_allocation", 0),
            priority=data.get("priority", 0),
            active=data.get("active", True)
        )

class ResourceOptimizationError(Exception):
    """Exception raised for resource optimization errors."""
    pass

class ResourceOptimizer:
    """
    Resource optimization and management for the FixWurx platform.
    
    This class provides resource monitoring, allocation, and optimization
    capabilities for the system, ensuring efficient use of system resources.
    """
    
    def __init__(self):
        """Initialize the resource optimizer."""
        self._config = get_config()
        self._limits = self._load_resource_limits()
        self._components: Dict[str, ComponentResource] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=self._limits.thread_pool_size)
        self._process_pool = ProcessPoolExecutor(max_workers=self._limits.process_pool_size)
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._resource_history: List[ResourceUsage] = []
        self._max_history_size = 1000
        self._history_lock = threading.Lock()
        self._optimization_strategy = self._get_optimization_strategy()
        self._optimization_interval = 10  # seconds
        self._optimizing = False
        self._component_lock = threading.Lock()
        
        # Initialize system resource usage
        self._current_usage = self._get_resource_usage()
        
        # Start monitoring thread
        self._start_monitoring()
        
        logger.info("Resource optimizer initialized")
    
    def _load_resource_limits(self) -> ResourceLimits:
        """
        Load resource limits from configuration.
        
        Returns:
            ResourceLimits object.
        """
        # Get limits from configuration
        resource_section = self._config.get_section("resources")
        limits_dict = {
            "max_cpu_percent": resource_section.get("max_cpu_percent", DEFAULT_MAX_CPU_PERCENT),
            "max_memory_percent": resource_section.get("max_memory_percent", DEFAULT_MAX_MEMORY_PERCENT),
            "max_disk_percent": resource_section.get("max_disk_percent", DEFAULT_MAX_DISK_PERCENT),
            "max_threads": resource_section.get("max_threads", DEFAULT_MAX_THREADS),
            "max_processes": resource_section.get("max_processes", DEFAULT_MAX_PROCESSES),
            "thread_pool_size": resource_section.get("thread_pool_size", DEFAULT_THREAD_POOL_SIZE),
            "process_pool_size": resource_section.get("process_pool_size", DEFAULT_PROCESS_POOL_SIZE),
            "io_throttle_threshold": resource_section.get("io_throttle_threshold", 0.8),
            "network_throttle_threshold": resource_section.get("network_throttle_threshold", 0.9)
        }
        
        return ResourceLimits.from_dict(limits_dict)
    
    def _get_optimization_strategy(self) -> str:
        """
        Get the optimization strategy from configuration.
        
        Returns:
            Optimization strategy.
        """
        resource_section = self._config.get_section("resources")
        strategy = resource_section.get("optimization_strategy", STRATEGY_BALANCED)
        
        if strategy not in [STRATEGY_PERFORMANCE, STRATEGY_BALANCED, 
                           STRATEGY_CONSERVATIVE, STRATEGY_MINIMAL, 
                           STRATEGY_CUSTOM]:
            logger.warning(f"Unknown optimization strategy: {strategy}. Using balanced.")
            return STRATEGY_BALANCED
        
        return strategy
    
    def _start_monitoring(self) -> None:
        """Start the resource monitoring thread."""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            logger.warning("Resource monitoring already started")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True,
            name="ResourceMonitoringThread"
        )
        self._monitoring_thread.start()
        logger.info("Resource monitoring started")
    
    def _stop_monitoring_thread(self) -> None:
        """Stop the resource monitoring thread."""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=2.0)
            logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self) -> None:
        """Monitor system resources and optimize if needed."""
        last_optimization_time = time.time()
        
        while not self._stop_monitoring.is_set():
            try:
                # Get current resource usage
                self._current_usage = self._get_resource_usage()
                
                # Add to history
                with self._history_lock:
                    self._resource_history.append(self._current_usage)
                    if len(self._resource_history) > self._max_history_size:
                        self._resource_history.pop(0)
                
                # Log resource usage
                logger.debug(f"CPU: {self._current_usage.cpu_percent:.1f}%, "
                            f"Memory: {self._current_usage.memory_percent:.1f}%, "
                            f"Disk: {self._current_usage.disk_percent:.1f}%")
                
                # Check if optimization is needed
                current_time = time.time()
                if (current_time - last_optimization_time) >= self._optimization_interval:
                    self._optimize_resources()
                    last_optimization_time = current_time
            
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
            
            # Sleep before next check
            time.sleep(1.0)
    
    def _get_resource_usage(self) -> ResourceUsage:
        """
        Get current system resource usage.
        
        Returns:
            ResourceUsage object.
        """
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used
        memory_total = memory.total
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used = disk.used
        disk_total = disk.total
        
        # Get thread and process count
        threads_active = threading.active_count()
        processes_active = len(psutil.pids())
        
        # Get network statistics
        network = psutil.net_io_counters()
        network_sent_bytes = network.bytes_sent
        network_recv_bytes = network.bytes_recv
        
        # Get IO statistics
        io = psutil.disk_io_counters()
        io_read_bytes = io.read_bytes if io else 0
        io_write_bytes = io.write_bytes if io else 0
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used=memory_used,
            memory_total=memory_total,
            disk_percent=disk_percent,
            disk_used=disk_used,
            disk_total=disk_total,
            threads_active=threads_active,
            processes_active=processes_active,
            network_sent_bytes=network_sent_bytes,
            network_recv_bytes=network_recv_bytes,
            timestamp=time.time(),
            io_read_bytes=io_read_bytes,
            io_write_bytes=io_write_bytes
        )
    
    def _optimize_resources(self) -> None:
        """
        Optimize resource allocation based on current usage and strategy.
        """
        if self._optimizing:
            return
        
        self._optimizing = True
        
        try:
            # Check if resources are over limits
            cpu_over = self._current_usage.cpu_percent > self._limits.max_cpu_percent
            memory_over = self._current_usage.memory_percent > self._limits.max_memory_percent
            disk_over = self._current_usage.disk_percent > self._limits.max_disk_percent
            threads_over = self._current_usage.threads_active > self._limits.max_threads
            
            if cpu_over or memory_over or disk_over or threads_over:
                logger.warning(f"Resource limits exceeded: CPU={cpu_over}, "
                              f"Memory={memory_over}, Disk={disk_over}, "
                              f"Threads={threads_over}")
                
                # Apply resource limits based on strategy
                self._apply_resource_limits()
            
            # Re-allocate resources to components based on priority
            self._allocate_resources_to_components()
            
        finally:
            self._optimizing = False
    
    def _apply_resource_limits(self) -> None:
        """
        Apply resource limits based on current usage and strategy.
        """
        # Determine actions based on optimization strategy
        if self._optimization_strategy == STRATEGY_PERFORMANCE:
            # Performance strategy: Allow high resource usage
            pass  # No additional limits
            
        elif self._optimization_strategy == STRATEGY_BALANCED:
            # Balanced strategy: Apply moderate limits
            if self._current_usage.cpu_percent > self._limits.max_cpu_percent:
                # Reduce thread pool size temporarily
                new_size = max(2, int(self._limits.thread_pool_size * 0.8))
                self._adjust_thread_pool(new_size)
            
            if self._current_usage.memory_percent > self._limits.max_memory_percent:
                # Release memory if possible
                self._release_memory()
            
        elif self._optimization_strategy == STRATEGY_CONSERVATIVE:
            # Conservative strategy: Apply stricter limits
            if self._current_usage.cpu_percent > self._limits.max_cpu_percent * 0.9:
                # Reduce thread pool size more aggressively
                new_size = max(2, int(self._limits.thread_pool_size * 0.6))
                self._adjust_thread_pool(new_size)
            
            if self._current_usage.memory_percent > self._limits.max_memory_percent * 0.9:
                # Release memory more aggressively
                self._release_memory()
                
            # Pause low priority components
            self._pause_low_priority_components()
            
        elif self._optimization_strategy == STRATEGY_MINIMAL:
            # Minimal strategy: Apply strict limits
            if self._current_usage.cpu_percent > self._limits.max_cpu_percent * 0.8:
                # Reduce thread pool size drastically
                new_size = max(1, int(self._limits.thread_pool_size * 0.4))
                self._adjust_thread_pool(new_size)
            
            if self._current_usage.memory_percent > self._limits.max_memory_percent * 0.8:
                # Release memory aggressively
                self._release_memory()
                
            # Pause all non-essential components
            self._pause_non_essential_components()
            
        elif self._optimization_strategy == STRATEGY_CUSTOM:
            # Custom strategy: Apply user-defined limits
            # This would be implemented based on specific requirements
            pass
    
    def _adjust_thread_pool(self, new_size: int) -> None:
        """
        Adjust the thread pool size.
        
        Args:
            new_size: New thread pool size.
        """
        if new_size == self._thread_pool._max_workers:
            return
        
        # Create new thread pool with adjusted size
        old_pool = self._thread_pool
        self._thread_pool = ThreadPoolExecutor(max_workers=new_size)
        
        # Shutdown old pool (after pending tasks complete)
        old_pool.shutdown(wait=False)
        
        logger.info(f"Adjusted thread pool size to {new_size}")
    
    def _release_memory(self) -> None:
        """Release memory if possible."""
        # This is a simplistic implementation
        # In a real system, this would use more sophisticated memory management
        import gc
        gc.collect()
        logger.info("Released memory through garbage collection")
    
    def _pause_low_priority_components(self) -> None:
        """Pause low priority components to free resources."""
        with self._component_lock:
            # Sort components by priority (ascending)
            sorted_components = sorted(
                self._components.values(),
                key=lambda c: c.priority
            )
            
            # Pause lowest priority components until we're under resource limits
            for component in sorted_components:
                if component.active and component.priority < 50:  # Low priority threshold
                    component.active = False
                    logger.info(f"Paused low priority component: {component.component_id}")
                    
                    # Check if we're now under resource limits
                    if (self._current_usage.cpu_percent < self._limits.max_cpu_percent and
                        self._current_usage.memory_percent < self._limits.max_memory_percent):
                        break
    
    def _pause_non_essential_components(self) -> None:
        """Pause all non-essential components to free resources."""
        with self._component_lock:
            # Sort components by priority (ascending)
            sorted_components = sorted(
                self._components.values(),
                key=lambda c: c.priority
            )
            
            # Pause all but highest priority components
            for component in sorted_components:
                if component.active and component.priority < 80:  # High priority threshold
                    component.active = False
                    logger.info(f"Paused non-essential component: {component.component_id}")
    
    def _allocate_resources_to_components(self) -> None:
        """
        Allocate resources to components based on priority.
        """
        with self._component_lock:
            if not self._components:
                return
            
            # Sort components by priority (descending)
            sorted_components = sorted(
                self._components.values(),
                key=lambda c: c.priority,
                reverse=True
            )
            
            # Active components only
            active_components = [c for c in sorted_components if c.active]
            
            if not active_components:
                return
            
            # Calculate total priority points
            total_priority = sum(c.priority for c in active_components)
            
            if total_priority == 0:
                # Equal allocation if all priorities are 0
                equal_share = 1.0 / len(active_components)
                for component in active_components:
                    component.cpu_allocation = equal_share * 100
                    component.memory_allocation = equal_share * 100
                return
            
            # Allocate CPU and memory based on priority
            for component in active_components:
                # Calculate share based on priority
                priority_share = component.priority / total_priority
                
                # Allocate CPU
                component.cpu_allocation = priority_share * 100
                
                # Allocate memory
                component.memory_allocation = priority_share * 100
                
                # Allocate threads based on CPU allocation
                component.thread_allocation = max(
                    1, 
                    int(priority_share * self._limits.max_threads)
                )
                
                # Allocate processes based on CPU allocation
                component.process_allocation = max(
                    1,
                    int(priority_share * self._limits.max_processes)
                )
                
                logger.debug(f"Allocated resources to {component.component_id}: "
                           f"CPU={component.cpu_allocation:.1f}%, "
                           f"Memory={component.memory_allocation:.1f}%, "
                           f"Threads={component.thread_allocation}")
    
    def register_component(self, component_id: str, component_type: str, 
                          priority: int = 50) -> None:
        """
        Register a component for resource management.
        
        Args:
            component_id: Unique identifier for the component.
            component_type: Type of component.
            priority: Priority level (0-100, higher is more important).
        """
        with self._component_lock:
            if component_id in self._components:
                logger.warning(f"Component {component_id} already registered")
                return
            
            # Validate priority
            priority = max(0, min(100, priority))
            
            # Create component resource
            component = ComponentResource(
                component_id=component_id,
                component_type=component_type,
                priority=priority
            )
            
            # Add to components
            self._components[component_id] = component
            
            logger.info(f"Registered component {component_id} with priority {priority}")
            
            # Re-allocate resources
            self._allocate_resources_to_components()
    
    def unregister_component(self, component_id: str) -> None:
        """
        Unregister a component from resource management.
        
        Args:
            component_id: Component identifier.
        """
        with self._component_lock:
            if component_id not in self._components:
                logger.warning(f"Component {component_id} not registered")
                return
            
            # Remove component
            del self._components[component_id]
            
            logger.info(f"Unregistered component {component_id}")
            
            # Re-allocate resources
            self._allocate_resources_to_components()
    
    def set_component_priority(self, component_id: str, priority: int) -> None:
        """
        Set the priority of a component.
        
        Args:
            component_id: Component identifier.
            priority: Priority level (0-100, higher is more important).
        """
        with self._component_lock:
            if component_id not in self._components:
                logger.warning(f"Component {component_id} not registered")
                return
            
            # Validate priority
            priority = max(0, min(100, priority))
            
            # Update priority
            self._components[component_id].priority = priority
            
            logger.info(f"Set component {component_id} priority to {priority}")
            
            # Re-allocate resources
            self._allocate_resources_to_components()
    
    def set_component_active(self, component_id: str, active: bool) -> None:
        """
        Set the active state of a component.
        
        Args:
            component_id: Component identifier.
            active: Whether the component is active.
        """
        with self._component_lock:
            if component_id not in self._components:
                logger.warning(f"Component {component_id} not registered")
                return
            
            # Update active state
            self._components[component_id].active = active
            
            logger.info(f"Set component {component_id} active state to {active}")
            
            # Re-allocate resources
            self._allocate_resources_to_components()
    
    def get_component_resources(self, component_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the resource allocation for a component.
        
        Args:
            component_id: Component identifier.
            
        Returns:
            Resource allocation dictionary or None if component not found.
        """
        with self._component_lock:
            if component_id not in self._components:
                logger.warning(f"Component {component_id} not registered")
                return None
            
            return self._components[component_id].to_dict()
    
    def get_all_component_resources(self) -> Dict[str, Dict[str, Any]]:
        """
        Get resource allocations for all components.
        
        Returns:
            Dictionary mapping component IDs to resource allocations.
        """
        with self._component_lock:
            return {
                component_id: component.to_dict()
                for component_id, component in self._components.items()
            }
    
    def get_system_resource_usage(self) -> Dict[str, Any]:
        """
        Get current system resource usage.
        
        Returns:
            Dictionary with current resource usage.
        """
        return self._current_usage.to_dict()
    
    def get_resource_history(self, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Get historical resource usage.
        
        Args:
            limit: Maximum number of history entries to return (0 for all).
            
        Returns:
            List of resource usage dictionaries.
        """
        with self._history_lock:
            history = self._resource_history.copy()
            
            if limit > 0 and limit < len(history):
                history = history[-limit:]
            
            return [usage.to_dict() for usage in history]
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """
        Get current resource limits.
        
        Returns:
            Dictionary with current resource limits.
        """
        return self._limits.to_dict()
    
    def set_resource_limits(self, limits: Dict[str, Any]) -> None:
        """
        Set resource limits.
        
        Args:
            limits: Dictionary with resource limits.
        """
        try:
            # Create new resource limits
            new_limits = ResourceLimits.from_dict(limits)
            
            # Update limits
            self._limits = new_limits
            
            # Update thread pool size if needed
            if new_limits.thread_pool_size != self._thread_pool._max_workers:
                self._adjust_thread_pool(new_limits.thread_pool_size)
            
            logger.info("Updated resource limits")
            
            # Re-allocate resources
            self._allocate_resources_to_components()
            
        except Exception as e:
            raise ResourceOptimizationError(f"Error setting resource limits: {e}")
    
    def set_optimization_strategy(self, strategy: str) -> None:
        """
        Set the optimization strategy.
        
        Args:
            strategy: Optimization strategy (performance, balanced, conservative, minimal, custom).
        """
        if strategy not in [STRATEGY_PERFORMANCE, STRATEGY_BALANCED, 
                           STRATEGY_CONSERVATIVE, STRATEGY_MINIMAL, 
                           STRATEGY_CUSTOM]:
            raise ResourceOptimizationError(f"Unknown optimization strategy: {strategy}")
        
        self._optimization_strategy = strategy
        logger.info(f"Set optimization strategy to {strategy}")
    
    def set_optimization_interval(self, interval: float) -> None:
        """
        Set the optimization interval.
        
        Args:
            interval: Optimization interval in seconds.
        """
        if interval <= 0:
            raise ResourceOptimizationError("Optimization interval must be positive")
        
        self._optimization_interval = interval
        logger.info(f"Set optimization interval to {interval} seconds")
    
    def submit_thread_task(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Submit a task to the thread pool.
        
        Args:
            fn: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
            
        Returns:
            Future object for the task.
        """
        return self._thread_pool.submit(fn, *args, **kwargs)
    
    def submit_process_task(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Submit a task to the process pool.
        
        Args:
            fn: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
            
        Returns:
            Future object for the task.
        """
        return self._process_pool.submit(fn, *args, **kwargs)
    
    def shutdown(self) -> None:
        """Shutdown the resource optimizer."""
        # Stop monitoring thread
        self._stop_monitoring_thread()
        
        # Shutdown thread pool
        self._thread_pool.shutdown()
        
        # Shutdown process pool
        self._process_pool.shutdown()
        
        logger.info("Resource optimizer shutdown")

# Global instance
resource_optimizer = ResourceOptimizer()

def get_resource_optimizer() -> ResourceOptimizer:
    """
    Get the global ResourceOptimizer instance.
    
    Returns:
        Global ResourceOptimizer instance.
    """
    return resource_optimizer

def initialize_resource_optimizer() -> None:
    """
    Initialize the global ResourceOptimizer.
    """
    global resource_optimizer
    resource_optimizer = ResourceOptimizer()

def configure_resources(
    max_cpu_percent: Optional[float] = None,
    max_memory_percent: Optional[float] = None,
    max_threads: Optional[int] = None,
    optimization_strategy: Optional[str] = None
) -> None:
    """
    Configure the resource optimizer with the given parameters.
    
    Args:
        max_cpu_percent: Maximum CPU usage percentage.
        max_memory_percent: Maximum memory usage percentage.
        max_threads: Maximum number of threads.
        optimization_strategy: Optimization strategy.
    """
    limits = resource_optimizer.get_resource_limits()
    
    if max_cpu_percent is not None:
        limits["max_cpu_percent"] = max_cpu_percent
    
    if max_memory_percent is not None:
        limits["max_memory_percent"] = max_memory_percent
    
    if max_threads is not None:
        limits["max_threads"] = max_threads
        limits["thread_pool_size"] = max(1, max_threads // 2)
    
    resource_optimizer.set_resource_limits(limits)
    
    if optimization_strategy is not None:
        resource_optimizer.set_optimization_strategy(optimization_strategy)

def create_component(
    component_id: str,
    component_type: str,
    priority: int = 50
) -> None:
    """
    Register a component with the resource optimizer.
    
    Args:
        component_id: Unique identifier for the component.
        component_type: Type of component.
        priority: Priority level (0-100, higher is more important).
    """
    resource_optimizer.register_component(component_id, component_type, priority)

def remove_component(component_id: str) -> None:
    """
    Unregister a component from the resource optimizer.
    
    Args:
        component_id: Component identifier.
    """
    resource_optimizer.unregister_component(component_id)

def change_component_priority(component_id: str, priority: int) -> None:
    """
    Change the priority of a component.
    
    Args:
        component_id: Component identifier.
        priority: New priority level (0-100).
    """
    resource_optimizer.set_component_priority(component_id, priority)

def get_system_resources() -> Dict[str, Any]:
    """
    Get current system resource usage.
    
    Returns:
        Dictionary with current resource usage.
    """
    return resource_optimizer.get_system_resource_usage()

def shutdown_resources() -> None:
    """
    Shutdown the resource optimizer.
    """
    resource_optimizer.shutdown()

def main():
    """Command-line interface for resource optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FixWurx Resource Optimization")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show resource status")
    status_parser.add_argument("--component", help="Component ID to show status for")
    status_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    
    # Configure command
    configure_parser = subparsers.add_parser("configure", help="Configure resource limits")
    configure_parser.add_argument("--cpu", type=float, help="Maximum CPU percentage")
    configure_parser.add_argument("--memory", type=float, help="Maximum memory percentage")
    configure_parser.add_argument("--threads", type=int, help="Maximum threads")
    configure_parser.add_argument("--strategy", choices=[
        STRATEGY_PERFORMANCE, STRATEGY_BALANCED, STRATEGY_CONSERVATIVE, STRATEGY_MINIMAL
    ], help="Optimization strategy")
    
    # Component commands
    component_parser = subparsers.add_parser("component", help="Manage components")
    component_subparsers = component_parser.add_subparsers(dest="component_command", help="Component command")
    
    # Register component
    register_parser = component_subparsers.add_parser("register", help="Register a component")
    register_parser.add_argument("id", help="Component ID")
    register_parser.add_argument("type", help="Component type")
    register_parser.add_argument("--priority", type=int, default=50, help="Component priority (0-100)")
    
    # Unregister component
    unregister_parser = component_subparsers.add_parser("unregister", help="Unregister a component")
    unregister_parser.add_argument("id", help="Component ID")
    
    # Set component priority
    priority_parser = component_subparsers.add_parser("priority", help="Set component priority")
    priority_parser.add_argument("id", help="Component ID")
    priority_parser.add_argument("priority", type=int, help="Priority level (0-100)")
    
    # Set component active state
    active_parser = component_subparsers.add_parser("active", help="Set component active state")
    active_parser.add_argument("id", help="Component ID")
    active_parser.add_argument("state", choices=["true", "false"], help="Active state")
    
    # List components
    list_parser = component_subparsers.add_parser("list", help="List components")
    list_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    
    # History command
    history_parser = subparsers.add_parser("history", help="Show resource history")
    history_parser.add_argument("--limit", type=int, default=10, help="Maximum entries to show")
    history_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "status":
        if args.component:
            resources = resource_optimizer.get_component_resources(args.component)
            if resources:
                if args.format == "json":
                    print(json.dumps(resources, indent=2))
                else:
                    print(f"Component: {args.component}")
                    for key, value in resources.items():
                        print(f"  {key}: {value}")
            else:
                print(f"Component {args.component} not found")
        else:
            usage = resource_optimizer.get_system_resource_usage()
            limits = resource_optimizer.get_resource_limits()
            
            if args.format == "json":
                print(json.dumps({
                    "usage": usage,
                    "limits": limits
                }, indent=2))
            else:
                print("System Resource Usage:")
                print(f"  CPU: {usage['cpu_percent']:.1f}% (limit: {limits['max_cpu_percent']:.1f}%)")
                print(f"  Memory: {usage['memory_percent']:.1f}% (limit: {limits['max_memory_percent']:.1f}%)")
                print(f"  Disk: {usage['disk_percent']:.1f}% (limit: {limits['max_disk_percent']:.1f}%)")
                print(f"  Threads: {usage['threads_active']} (limit: {limits['max_threads']})")
                print(f"  Processes: {usage['processes_active']} (limit: {limits['max_processes']})")
    
    elif args.command == "configure":
        changes = {}
        
        if args.cpu:
            changes["max_cpu_percent"] = args.cpu
        
        if args.memory:
            changes["max_memory_percent"] = args.memory
        
        if args.threads:
            changes["max_threads"] = args.threads
            changes["thread_pool_size"] = max(1, args.threads // 2)
        
        if changes:
            limits = resource_optimizer.get_resource_limits()
            limits.update(changes)
            resource_optimizer.set_resource_limits(limits)
            print("Resource limits updated")
        
        if args.strategy:
            resource_optimizer.set_optimization_strategy(args.strategy)
            print(f"Optimization strategy set to {args.strategy}")
    
    elif args.command == "component":
        if args.component_command == "register":
            resource_optimizer.register_component(args.id, args.type, args.priority)
            print(f"Registered component {args.id}")
        
        elif args.component_command == "unregister":
            resource_optimizer.unregister_component(args.id)
            print(f"Unregistered component {args.id}")
        
        elif args.component_command == "priority":
            resource_optimizer.set_component_priority(args.id, args.priority)
            print(f"Set component {args.id} priority to {args.priority}")
        
        elif args.component_command == "active":
            active = args.state.lower() == "true"
            resource_optimizer.set_component_active(args.id, active)
            print(f"Set component {args.id} active state to {active}")
        
        elif args.component_command == "list":
            components = resource_optimizer.get_all_component_resources()
            
            if args.format == "json":
                print(json.dumps(components, indent=2))
            else:
                print("Components:")
                for component_id, resources in components.items():
                    print(f"\n{component_id}:")
                    for key, value in resources.items():
                        if key != "component_id":
                            print(f"  {key}: {value}")
    
    elif args.command == "history":
        history = resource_optimizer.get_resource_history(args.limit)
        
        if args.format == "json":
            print(json.dumps(history, indent=2))
        else:
            print("Resource History:")
            for i, entry in enumerate(history):
                timestamp = datetime.datetime.fromtimestamp(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n{i+1}. {timestamp}")
                print(f"  CPU: {entry['cpu_percent']:.1f}%")
                print(f"  Memory: {entry['memory_percent']:.1f}%")
                print(f"  Disk: {entry['disk_percent']:.1f}%")
                print(f"  Threads: {entry['threads_active']}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
