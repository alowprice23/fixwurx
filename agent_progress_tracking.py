#!/usr/bin/env python3
"""
agent_progress_tracking.py
─────────────────────────
Progress tracking system for long-running agent operations.

This module provides functionality to track and report progress of long-running
operations performed by agents in the FixWurx shell environment.
"""

import os
import sys
import json
import time
import logging
import threading
import shlex
import argparse
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime

# Configure logging
logger = logging.getLogger("AgentProgressTracking")

# Import communication system
import agent_communication_system

# Global instance
_instance = None

class ProgressTracker:
    """
    Progress tracking system for long-running agent operations.
    
    This class provides functionality to:
    1. Create and manage progress trackers for long-running operations
    2. Update progress with detailed status information
    3. Generate visualizations of progress (text-based)
    4. Track multiple concurrent operations
    5. Integrate with the agent communication system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Progress Tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        
        # Get communication system
        self.communication_system = agent_communication_system.get_instance()
        
        # Progress tracking data
        self.trackers = {}
        self.active_tasks = {}
        
        # Thread lock for thread safety
        self._lock = threading.Lock()
        
        # Thread for background progress tracking
        self._stop_tracking = threading.Event()
        self._tracking_thread = None
        
        # Register global instance
        global _instance
        _instance = self
        
        logger.info("Progress Tracker initialized")
        
        # Start background tracking
        self._start_background_tracking()
    
    def _start_background_tracking(self) -> None:
        """Start background progress tracking thread."""
        if self._tracking_thread is None:
            self._stop_tracking.clear()
            self._tracking_thread = threading.Thread(
                target=self._tracking_loop,
                daemon=True,
                name="ProgressTrackingThread"
            )
            self._tracking_thread.start()
            logger.info("Background progress tracking started")
    
    def _tracking_loop(self) -> None:
        """Background tracking loop."""
        try:
            while not self._stop_tracking.is_set():
                # Check for timed-out trackers
                self._check_timeouts()
                
                # Sleep for a bit
                time.sleep(1.0)
        except Exception as e:
            logger.error(f"Error in progress tracking loop: {e}")
    
    def _check_timeouts(self) -> None:
        """Check for timed-out progress trackers."""
        with self._lock:
            current_time = time.time()
            timed_out_trackers = []
            
            for tracker_id, tracker_data in self.trackers.items():
                # Skip completed or failed trackers
                if tracker_data["status"] not in ("in_progress", "paused"):
                    continue
                
                # Check for timeout (no update in 5 minutes)
                last_update = tracker_data.get("last_update", 0)
                if current_time - last_update > 300:  # 5 minutes
                    timed_out_trackers.append(tracker_id)
            
            # Update timed-out trackers
            for tracker_id in timed_out_trackers:
                tracker_data = self.trackers[tracker_id]
                tracker_data["status"] = "stalled"
                
                # Notify via communication system
                agent_id = tracker_data["agent_id"]
                task_id = tracker_data["task_id"]
                description = tracker_data["description"]
                
                if self.communication_system and self.communication_system.agent_is_active(agent_id):
                    self.communication_system.speak_warning(
                        agent_id=agent_id,
                        warning_message=f"Task stalled: {description} (no updates in 5 minutes)",
                        session_id=tracker_data.get("session_id")
                    )
    
    def start_task(self, agent_id: str, task_id: str, description: str, 
                  total_steps: int = 100, session_id: Optional[str] = None) -> str:
        """
        Start tracking a task.
        
        Args:
            agent_id: The agent's ID
            task_id: Task identifier
            description: Task description
            total_steps: Total number of steps
            session_id: Optional session ID
            
        Returns:
            Tracker ID
        """
        if not self.enabled:
            return ""
        
        with self._lock:
            # Create tracker ID
            tracker_id = f"{agent_id}_{task_id}_{int(time.time())}"
            
            # Create tracker data
            self.trackers[tracker_id] = {
                "tracker_id": tracker_id,
                "agent_id": agent_id,
                "task_id": task_id,
                "description": description,
                "total_steps": total_steps,
                "current_step": 0,
                "percentage": 0,
                "status": "in_progress",
                "started_at": time.time(),
                "last_update": time.time(),
                "estimated_completion": None,
                "status_message": "Starting task...",
                "session_id": session_id
            }
            
            # Add to active tasks for the agent
            if agent_id not in self.active_tasks:
                self.active_tasks[agent_id] = []
            self.active_tasks[agent_id].append(tracker_id)
            
            # Notify via communication system
            if self.communication_system and self.communication_system.agent_is_active(agent_id):
                progress_tracker_id = self.communication_system.start_progress(
                    agent_id=agent_id,
                    task_id=task_id,
                    description=description,
                    total_steps=total_steps,
                    session_id=session_id
                )
            
            logger.info(f"Started tracking task: {description} (Agent: {agent_id}, Task ID: {task_id})")
            return tracker_id
    
    def update_task(self, tracker_id: str, current_step: int, 
                   status_message: Optional[str] = None) -> None:
        """
        Update task progress.
        
        Args:
            tracker_id: Tracker ID
            current_step: Current step
            status_message: Optional status message
        """
        if not self.enabled or not tracker_id:
            return
        
        with self._lock:
            if tracker_id not in self.trackers:
                logger.warning(f"Unknown tracker ID: {tracker_id}")
                return
            
            tracker_data = self.trackers[tracker_id]
            
            # Skip if tracker is not in progress
            if tracker_data["status"] not in ("in_progress", "paused"):
                return
            
            # Update tracker data
            total_steps = tracker_data["total_steps"]
            previous_step = tracker_data["current_step"]
            
            # Ensure step doesn't exceed total
            current_step = min(current_step, total_steps)
            
            # Update step and percentage
            tracker_data["current_step"] = current_step
            tracker_data["percentage"] = int((current_step / total_steps) * 100)
            
            # Update status message if provided
            if status_message:
                tracker_data["status_message"] = status_message
            
            # Update last update time
            tracker_data["last_update"] = time.time()
            
            # Calculate estimated completion time
            if current_step > previous_step and current_step < total_steps:
                elapsed = time.time() - tracker_data["started_at"]
                rate = current_step / elapsed if elapsed > 0 else 0
                
                if rate > 0:
                    remaining_steps = total_steps - current_step
                    remaining_time = remaining_steps / rate
                    tracker_data["estimated_completion"] = time.time() + remaining_time
            
            # Notify via communication system
            agent_id = tracker_data["agent_id"]
            if self.communication_system and self.communication_system.agent_is_active(agent_id):
                self.communication_system.update_progress(
                    tracker_id=tracker_id,
                    current_step=current_step,
                    status_message=status_message
                )
    
    def complete_task(self, tracker_id: str, success: bool = True, 
                     completion_message: Optional[str] = None) -> None:
        """
        Complete a task.
        
        Args:
            tracker_id: Tracker ID
            success: Whether the task completed successfully
            completion_message: Optional completion message
        """
        if not self.enabled or not tracker_id:
            return
        
        with self._lock:
            if tracker_id not in self.trackers:
                logger.warning(f"Unknown tracker ID: {tracker_id}")
                return
            
            tracker_data = self.trackers[tracker_id]
            
            # Skip if tracker is already completed or failed
            if tracker_data["status"] in ("completed", "failed"):
                return
            
            # Update tracker data
            tracker_data["status"] = "completed" if success else "failed"
            tracker_data["current_step"] = tracker_data["total_steps"] if success else tracker_data["current_step"]
            tracker_data["percentage"] = 100 if success else tracker_data["percentage"]
            
            if completion_message:
                tracker_data["status_message"] = completion_message
            else:
                tracker_data["status_message"] = "Task completed successfully" if success else "Task failed"
            
            # Update completion time
            tracker_data["completed_at"] = time.time()
            
            # Remove from active tasks
            agent_id = tracker_data["agent_id"]
            if agent_id in self.active_tasks and tracker_id in self.active_tasks[agent_id]:
                self.active_tasks[agent_id].remove(tracker_id)
            
            # Notify via communication system
            if self.communication_system and self.communication_system.agent_is_active(agent_id):
                self.communication_system.complete_progress(
                    tracker_id=tracker_id,
                    success=success,
                    completion_message=completion_message
                )
            
            logger.info(f"Task {tracker_data['description']} {'completed successfully' if success else 'failed'}")
    
    def pause_task(self, tracker_id: str, pause_message: Optional[str] = None) -> bool:
        """
        Pause a task.
        
        Args:
            tracker_id: Tracker ID
            pause_message: Optional pause message
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not tracker_id:
            return False
        
        with self._lock:
            if tracker_id not in self.trackers:
                logger.warning(f"Unknown tracker ID: {tracker_id}")
                return False
            
            tracker_data = self.trackers[tracker_id]
            
            # Skip if tracker is not in progress
            if tracker_data["status"] != "in_progress":
                return False
            
            # Update tracker data
            tracker_data["status"] = "paused"
            
            if pause_message:
                tracker_data["status_message"] = pause_message
            else:
                tracker_data["status_message"] = "Task paused"
            
            # Update last update time
            tracker_data["last_update"] = time.time()
            
            # Notify via communication system
            agent_id = tracker_data["agent_id"]
            if self.communication_system and self.communication_system.agent_is_active(agent_id):
                self.communication_system.speak_info(
                    agent_id=agent_id,
                    info_message=f"Task paused: {tracker_data['description']}",
                    session_id=tracker_data.get("session_id")
                )
            
            logger.info(f"Task {tracker_data['description']} paused")
            return True
    
    def resume_task(self, tracker_id: str, resume_message: Optional[str] = None) -> bool:
        """
        Resume a paused task.
        
        Args:
            tracker_id: Tracker ID
            resume_message: Optional resume message
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not tracker_id:
            return False
        
        with self._lock:
            if tracker_id not in self.trackers:
                logger.warning(f"Unknown tracker ID: {tracker_id}")
                return False
            
            tracker_data = self.trackers[tracker_id]
            
            # Skip if tracker is not paused
            if tracker_data["status"] != "paused":
                return False
            
            # Update tracker data
            tracker_data["status"] = "in_progress"
            
            if resume_message:
                tracker_data["status_message"] = resume_message
            else:
                tracker_data["status_message"] = "Task resumed"
            
            # Update last update time
            tracker_data["last_update"] = time.time()
            
            # Notify via communication system
            agent_id = tracker_data["agent_id"]
            if self.communication_system and self.communication_system.agent_is_active(agent_id):
                self.communication_system.speak_info(
                    agent_id=agent_id,
                    info_message=f"Task resumed: {tracker_data['description']}",
                    session_id=tracker_data.get("session_id")
                )
            
            logger.info(f"Task {tracker_data['description']} resumed")
            return True
    
    def get_task_status(self, tracker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        
        Args:
            tracker_id: Tracker ID
            
        Returns:
            Task status dictionary or None if not found
        """
        with self._lock:
            if tracker_id not in self.trackers:
                return None
            
            return self.trackers[tracker_id].copy()
    
    def get_agent_tasks(self, agent_id: str, include_completed: bool = False) -> List[Dict[str, Any]]:
        """
        Get all tasks for an agent.
        
        Args:
            agent_id: Agent ID
            include_completed: Whether to include completed tasks
            
        Returns:
            List of task status dictionaries
        """
        tasks = []
        
        with self._lock:
            # Get active tasks for the agent
            active_trackers = self.active_tasks.get(agent_id, [])
            
            # Add active tasks
            for tracker_id in active_trackers:
                if tracker_id in self.trackers:
                    tasks.append(self.trackers[tracker_id].copy())
            
            # Add completed tasks if requested
            if include_completed:
                for tracker_id, tracker_data in self.trackers.items():
                    if tracker_data["agent_id"] == agent_id and tracker_id not in active_trackers:
                        tasks.append(tracker_data.copy())
        
        # Sort by start time (newest first)
        tasks.sort(key=lambda x: x.get("started_at", 0), reverse=True)
        
        return tasks
    
    def get_all_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all active tasks.
        
        Returns:
            List of task status dictionaries
        """
        tasks = []
        
        with self._lock:
            for agent_id, tracker_ids in self.active_tasks.items():
                for tracker_id in tracker_ids:
                    if tracker_id in self.trackers:
                        tasks.append(self.trackers[tracker_id].copy())
        
        # Sort by start time (newest first)
        tasks.sort(key=lambda x: x.get("started_at", 0), reverse=True)
        
        return tasks
    
    def shutdown(self) -> None:
        """Shutdown the progress tracker."""
        # Stop background tracking
        self._stop_tracking.set()
        if self._tracking_thread and self._tracking_thread.is_alive():
            self._tracking_thread.join(timeout=2.0)
        
        logger.info("Progress Tracker shutdown")


def get_instance(config: Optional[Dict[str, Any]] = None) -> ProgressTracker:
    """
    Get the singleton instance of the Progress Tracker.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ProgressTracker instance
    """
    global _instance
    if _instance is None:
        _instance = ProgressTracker(config)
    return _instance


# Command handler for progress tracking
def progress_command(args: str) -> int:
    """
    Command handler for progress tracking.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Manage progress tracking")
    parser.add_argument("action", choices=["start", "update", "complete", "pause", "resume", "list", "show"],
                        help="Action to perform")
    parser.add_argument("--agent", "-a", help="Agent ID")
    parser.add_argument("--task", "-t", help="Task ID")
    parser.add_argument("--tracker", "-r", help="Tracker ID")
    parser.add_argument("--description", "-d", help="Task description")
    parser.add_argument("--steps", "-s", type=int, default=100, help="Total steps")
    parser.add_argument("--step", "-p", type=int, help="Current step")
    parser.add_argument("--message", "-m", help="Status message")
    parser.add_argument("--success", action="store_true", help="Task completed successfully")
    parser.add_argument("--fail", action="store_true", help="Task failed")
    parser.add_argument("--all", "-A", action="store_true", help="Include all tasks")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get progress tracker
    progress_tracker = get_instance()
    
    # Perform action
    if cmd_args.action == "start":
        # Start tracking a task
        if not cmd_args.agent:
            print("Error: Agent ID is required")
            return 1
        
        if not cmd_args.task:
            print("Error: Task ID is required")
            return 1
        
        if not cmd_args.description:
            print("Error: Task description is required")
            return 1
        
        tracker_id = progress_tracker.start_task(
            agent_id=cmd_args.agent,
            task_id=cmd_args.task,
            description=cmd_args.description,
            total_steps=cmd_args.steps
        )
        
        print(f"Started tracking task: {cmd_args.description}")
        print(f"Tracker ID: {tracker_id}")
        return 0
    
    elif cmd_args.action == "update":
        # Update task progress
        if not cmd_args.tracker:
            print("Error: Tracker ID is required")
            return 1
        
        if cmd_args.step is None:
            print("Error: Current step is required")
            return 1
        
        progress_tracker.update_task(
            tracker_id=cmd_args.tracker,
            current_step=cmd_args.step,
            status_message=cmd_args.message
        )
        
        print(f"Updated task progress: {cmd_args.step} steps")
        return 0
    
    elif cmd_args.action == "complete":
        # Complete a task
        if not cmd_args.tracker:
            print("Error: Tracker ID is required")
            return 1
        
        success = not cmd_args.fail
        
        progress_tracker.complete_task(
            tracker_id=cmd_args.tracker,
            success=success,
            completion_message=cmd_args.message
        )
        
        print(f"Task marked as {'completed' if success else 'failed'}")
        return 0
    
    elif cmd_args.action == "pause":
        # Pause a task
        if not cmd_args.tracker:
            print("Error: Tracker ID is required")
            return 1
        
        success = progress_tracker.pause_task(
            tracker_id=cmd_args.tracker,
            pause_message=cmd_args.message
        )
        
        if success:
            print("Task paused")
            return 0
        else:
            print("Failed to pause task")
            return 1
    
    elif cmd_args.action == "resume":
        # Resume a task
        if not cmd_args.tracker:
            print("Error: Tracker ID is required")
            return 1
        
        success = progress_tracker.resume_task(
            tracker_id=cmd_args.tracker,
            resume_message=cmd_args.message
        )
        
        if success:
            print("Task resumed")
            return 0
        else:
            print("Failed to resume task")
            return 1
    
    elif cmd_args.action == "list":
        # List tasks
        if cmd_args.agent:
            # List tasks for a specific agent
            tasks = progress_tracker.get_agent_tasks(
                agent_id=cmd_args.agent,
                include_completed=cmd_args.all
            )
            
            print(f"\nTasks for Agent: {cmd_args.agent}")
        else:
            # List all active tasks
            tasks = progress_tracker.get_all_active_tasks()
            
            print("\nActive Tasks:")
        
        if not tasks:
            print("No tasks found")
            return 0
        
        # Print task information
        print("-" * 80)
        print(f"{'Agent':<10} {'Status':<10} {'Progress':<10} {'Description':<30} {'Tracker ID':<20}")
        print("-" * 80)
        
        for task in tasks:
            agent_id = task.get("agent_id", "")
            status = task.get("status", "")
            percentage = task.get("percentage", 0)
            description = task.get("description", "")
            tracker_id = task.get("tracker_id", "")
            
            print(f"{agent_id:<10} {status:<10} {percentage:>3}%{' '*6} {description[:30]:<30} {tracker_id:<20}")
        
        return 0
    
    elif cmd_args.action == "show":
        # Show task details
        if not cmd_args.tracker:
            print("Error: Tracker ID is required")
            return 1
        
        task = progress_tracker.get_task_status(cmd_args.tracker)
        
        if not task:
            print(f"Task not found: {cmd_args.tracker}")
            return 1
        
        # Print task details
        print(f"\nTask Details: {task.get('description', '')}")
        print("-" * 80)
        
        # Format timestamps
        started_at = datetime.fromtimestamp(task.get("started_at", 0)).strftime("%Y-%m-%d %H:%M:%S")
        last_update = datetime.fromtimestamp(task.get("last_update", 0)).strftime("%Y-%m-%d %H:%M:%S")
        
        estimated_completion = task.get("estimated_completion")
        if estimated_completion:
            estimated_completion = datetime.fromtimestamp(estimated_completion).strftime("%Y-%m-%d %H:%M:%S")
        
        completed_at = task.get("completed_at")
        if completed_at:
            completed_at = datetime.fromtimestamp(completed_at).strftime("%Y-%m-%d %H:%M:%S")
        
        # Print basic information
        print(f"Agent ID:       {task.get('agent_id', '')}")
        print(f"Task ID:        {task.get('task_id', '')}")
        print(f"Tracker ID:     {task.get('tracker_id', '')}")
        print(f"Status:         {task.get('status', '')}")
        print(f"Progress:       {task.get('current_step', 0)} / {task.get('total_steps', 0)} steps ({task.get('percentage', 0)}%)")
        print(f"Status Message: {task.get('status_message', '')}")
        print(f"Started At:     {started_at}")
        print(f"Last Update:    {last_update}")
        
        if estimated_completion and task.get("status") == "in_progress":
            print(f"Est. Completion: {estimated_completion}")
        
        if completed_at:
            print(f"Completed At:    {completed_at}")
            
            # Calculate duration
            duration = task.get("completed_at", 0) - task.get("started_at", 0)
            print(f"Duration:        {duration:.2f} seconds")
        
        return 0
    
    else:
        print(f"Unknown action: {cmd_args.action}")
        return 1


def register_progress_commands(registry) -> None:
    """
    Register progress tracking commands with the shell environment.
    
    Args:
        registry: Component registry
    """
    logger.info("Registering progress tracking commands")
    
    # Initialize progress tracker
    config = {
        "enabled": True
    }
    
    progress_tracker = get_instance(config)
    registry.register_component("progress_tracker", progress_tracker)
    
    # Register progress tracking commands
    registry.register_command_handler("agent:progress", progress_command, "agent")
    
    # Register helpful aliases
    registry.register_alias("tasks", "agent:progress list")
    
    logger.info("Progress tracking commands registered successfully")
