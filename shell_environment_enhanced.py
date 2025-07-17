#!/usr/bin/env python3
"""
Enhanced Shell Environment

This module extends the shell environment with advanced features:
- Pipeline support for command chaining
- Command output redirection
- Background task execution
"""

import os
import sys
import time
import threading
import queue
import io
import contextlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger("EnhancedShell")

class CommandPipeline:
    """
    Handles command pipelines, redirection, and background execution.
    """
    
    def __init__(self, registry):
        """
        Initialize the command pipeline.
        
        Args:
            registry: Component registry
        """
        self.registry = registry
        self.background_tasks = {}
        self.next_task_id = 1
        self.bg_task_lock = threading.Lock()
    
    def parse_command_line(self, line: str) -> Dict[str, Any]:
        """
        Parse a command line, detecting pipelines, redirection, and background execution.
        
        Args:
            line: Command line
            
        Returns:
            Dictionary with parsed command information
        """
        result = {
            "background": False,
            "pipeline": [],
            "redirection": None,
            "append": False,
            "original_line": line
        }
        
        # Check for background execution (must be at the end)
        if line.rstrip().endswith(" &"):
            result["background"] = True
            line = line.rstrip()[:-2].rstrip()
        
        # Check for redirection (must be before we split pipelines)
        if " > " in line:
            parts = line.rsplit(" > ", 1)
            line = parts[0]
            result["redirection"] = parts[1].strip()
            result["append"] = False
        elif " >> " in line:
            parts = line.rsplit(" >> ", 1)
            line = parts[0]
            result["redirection"] = parts[1].strip()
            result["append"] = True
        
        # Split by pipe symbol for command pipeline
        if " | " in line:
            commands = [cmd.strip() for cmd in line.split(" | ")]
            result["pipeline"] = commands
        else:
            # Single command (still treated as a pipeline with one command)
            result["pipeline"] = [line.strip()]
        
        return result
    
    def execute_pipeline(self, parsed_command: Dict[str, Any]) -> Tuple[int, str]:
        """
        Execute a command pipeline.
        
        Args:
            parsed_command: Parsed command information
            
        Returns:
            Tuple of (exit_code, output)
        """
        pipeline = parsed_command["pipeline"]
        if not pipeline:
            return 1, "Empty command"
        
        current_input = None
        exit_code = 0
        
        # Execute each command in the pipeline
        for i, cmd in enumerate(pipeline):
            is_last = (i == len(pipeline) - 1)
            
            # Capture command output
            with io.StringIO() as buffer, contextlib.redirect_stdout(buffer):
                # Execute the command
                exit_code = self._execute_single_command(cmd, input_data=current_input)
                
                # Get the output
                output = buffer.getvalue()
            
            # If this is the last command and we have redirection, we'll handle
            # the output outside this loop
            if is_last and parsed_command["redirection"]:
                break
            
            # If not the last command, use output as input for the next command
            if not is_last:
                current_input = output
        
        # Handle redirection
        if parsed_command["redirection"]:
            file_path = parsed_command["redirection"]
            mode = 'a' if parsed_command["append"] else 'w'
            
            try:
                with open(file_path, mode) as f:
                    f.write(output)
                return exit_code, f"Output redirected to {file_path}"
            except Exception as e:
                return 1, f"Error redirecting output: {e}"
        
        return exit_code, output
    
    def _execute_single_command(self, cmd: str, input_data: Optional[str] = None) -> int:
        """
        Execute a single command.
        
        Args:
            cmd: Command to execute
            input_data: Optional input data
            
        Returns:
            Exit code
        """
        # If we have input data, we need to temporarily replace stdin
        if input_data is not None:
            # Save original stdin
            original_stdin = sys.stdin
            
            # Create a StringIO object with the input data
            # Ensure input_data ends with a newline to avoid issues with line counting
            if not input_data.endswith('\n'):
                input_data += '\n'
            
            string_io = io.StringIO(input_data)
            
            # Replace stdin with our StringIO object
            sys.stdin = string_io
        
        try:
            # Split command and arguments
            parts = cmd.split(maxsplit=1)
            if not parts:
                return 0
            
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Find the command handler
            handler_info = self.registry.get_command_handler(command)
            
            if handler_info:
                # Execute the command handler
                handler = handler_info["handler"]
                exit_code = handler(args)
                
                # Log command execution
                component = handler_info.get("component", "unknown")
                logger.info(f"Executed command: {command} (component: {component}, exit_code: {exit_code})")
                
                return exit_code
            else:
                print(f"Unknown command: {command}")
                return 1
        except Exception as e:
            print(f"Error executing command: {e}")
            logger.error(f"Error executing command: {e}")
            return 1
        finally:
            # Restore original stdin if we replaced it
            if input_data is not None:
                sys.stdin = original_stdin
    
    def execute_in_background(self, parsed_command: Dict[str, Any]) -> int:
        """
        Execute a command in the background.
        
        Args:
            parsed_command: Parsed command information
            
        Returns:
            Task ID
        """
        with self.bg_task_lock:
            task_id = self.next_task_id
            self.next_task_id += 1
        
        # Create a thread to execute the command
        thread = threading.Thread(
            target=self._background_task_worker,
            args=(task_id, parsed_command),
            daemon=True
        )
        
        # Start the thread
        thread.start()
        
        # Store task info
        with self.bg_task_lock:
            self.background_tasks[task_id] = {
                "id": task_id,
                "command": parsed_command["original_line"],
                "status": "running",
                "started": time.time(),
                "thread": thread,
                "result": None
            }
        
        return task_id
    
    def _background_task_worker(self, task_id: int, parsed_command: Dict[str, Any]) -> None:
        """
        Worker function for background tasks.
        
        Args:
            task_id: Task ID
            parsed_command: Parsed command information
        """
        try:
            # Execute the command
            exit_code, output = self.execute_pipeline(parsed_command)
            
            # Update task info
            with self.bg_task_lock:
                if task_id in self.background_tasks:
                    self.background_tasks[task_id].update({
                        "status": "completed",
                        "completed": time.time(),
                        "exit_code": exit_code,
                        "output": output
                    })
        except Exception as e:
            # Update task info with error
            with self.bg_task_lock:
                if task_id in self.background_tasks:
                    self.background_tasks[task_id].update({
                        "status": "failed",
                        "completed": time.time(),
                        "error": str(e)
                    })
    
    def get_background_tasks(self) -> List[Dict[str, Any]]:
        """
        Get a list of background tasks.
        
        Returns:
            List of background task information
        """
        with self.bg_task_lock:
            return list(self.background_tasks.values())
    
    def get_background_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific background task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task information or None if not found
        """
        with self.bg_task_lock:
            return self.background_tasks.get(task_id)
    
    def terminate_background_task(self, task_id: int) -> bool:
        """
        Terminate a background task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was terminated, False otherwise
        """
        with self.bg_task_lock:
            if task_id not in self.background_tasks:
                return False
            
            task = self.background_tasks[task_id]
            
            # Can't terminate completed or failed tasks
            if task["status"] not in ("running", "paused"):
                return False
            
            # Mark task as terminated
            task.update({
                "status": "terminated",
                "completed": time.time()
            })
            
            # Note: We can't actually terminate a thread in Python,
            # but we can mark it as terminated. The thread will continue
            # to run but we'll ignore its result.
            
            return True
    
    def cleanup_completed_tasks(self, max_age: float = 3600.0) -> int:
        """
        Remove completed tasks older than max_age seconds.
        
        Args:
            max_age: Maximum age in seconds
            
        Returns:
            Number of tasks removed
        """
        removed = 0
        current_time = time.time()
        
        with self.bg_task_lock:
            task_ids = list(self.background_tasks.keys())
            
            for task_id in task_ids:
                task = self.background_tasks[task_id]
                
                if task["status"] in ("completed", "failed", "terminated"):
                    completed_time = task.get("completed", current_time)
                    
                    if current_time - completed_time > max_age:
                        del self.background_tasks[task_id]
                        removed += 1
        
        return removed

# Command handlers for background tasks

def bg_command(args: str) -> int:
    """
    Manage background tasks.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Manage background tasks")
    parser.add_argument("action", choices=["list", "status", "output", "kill", "cleanup"], 
                       help="Action to perform")
    parser.add_argument("task_id", nargs="?", type=int, help="Task ID (for status, output, kill)")
    parser.add_argument("--all", action="store_true", help="Apply to all tasks (for kill, cleanup)")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get pipeline from registry
    registry = sys.modules.get("__main__").registry
    command_pipeline = registry.get_component("command_pipeline")
    
    if not command_pipeline:
        print("Error: Command pipeline not available")
        return 1
    
    # Perform action
    if cmd_args.action == "list":
        tasks = command_pipeline.get_background_tasks()
        
        if not tasks:
            print("No background tasks")
            return 0
        
        print("\nBackground Tasks:")
        print("-" * 60)
        
        for task in sorted(tasks, key=lambda t: t["id"]):
            task_id = task["id"]
            status = task["status"]
            command = task["command"]
            started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task["started"]))
            
            # Format duration
            if status in ("completed", "failed", "terminated") and "completed" in task:
                duration = task["completed"] - task["started"]
                duration_str = f"{duration:.2f}s"
            else:
                duration = time.time() - task["started"]
                duration_str = f"{duration:.2f}s (running)"
            
            print(f"  {task_id}: [{status}] {command}")
            print(f"     Started: {started}, Duration: {duration_str}")
            
            if status == "failed" and "error" in task:
                print(f"     Error: {task['error']}")
            
            if status == "completed" and "exit_code" in task:
                print(f"     Exit Code: {task['exit_code']}")
    
    elif cmd_args.action == "status":
        if not cmd_args.task_id:
            print("Error: Task ID required for status action")
            return 1
        
        task = command_pipeline.get_background_task(cmd_args.task_id)
        
        if not task:
            print(f"Error: Task {cmd_args.task_id} not found")
            return 1
        
        print(f"\nTask {cmd_args.task_id} Status:")
        print("-" * 60)
        
        status = task["status"]
        command = task["command"]
        started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task["started"]))
        
        # Format duration
        if status in ("completed", "failed", "terminated") and "completed" in task:
            duration = task["completed"] - task["started"]
            duration_str = f"{duration:.2f}s"
        else:
            duration = time.time() - task["started"]
            duration_str = f"{duration:.2f}s (running)"
        
        print(f"Status: {status}")
        print(f"Command: {command}")
        print(f"Started: {started}")
        print(f"Duration: {duration_str}")
        
        if status == "failed" and "error" in task:
            print(f"Error: {task['error']}")
        
        if status == "completed" and "exit_code" in task:
            print(f"Exit Code: {task['exit_code']}")
    
    elif cmd_args.action == "output":
        if not cmd_args.task_id:
            print("Error: Task ID required for output action")
            return 1
        
        task = command_pipeline.get_background_task(cmd_args.task_id)
        
        if not task:
            print(f"Error: Task {cmd_args.task_id} not found")
            return 1
        
        print(f"\nTask {cmd_args.task_id} Output:")
        print("-" * 60)
        
        if "output" in task:
            print(task["output"])
        else:
            print("No output available")
    
    elif cmd_args.action == "kill":
        if not cmd_args.task_id and not cmd_args.all:
            print("Error: Task ID or --all required for kill action")
            return 1
        
        if cmd_args.all:
            # Kill all running tasks
            tasks = command_pipeline.get_background_tasks()
            running_tasks = [t for t in tasks if t["status"] in ("running", "paused")]
            
            if not running_tasks:
                print("No running tasks to kill")
                return 0
            
            killed = 0
            for task in running_tasks:
                if command_pipeline.terminate_background_task(task["id"]):
                    killed += 1
            
            print(f"Killed {killed} tasks")
        else:
            # Kill specific task
            if command_pipeline.terminate_background_task(cmd_args.task_id):
                print(f"Task {cmd_args.task_id} terminated")
            else:
                print(f"Error: Could not terminate task {cmd_args.task_id}")
                return 1
    
    elif cmd_args.action == "cleanup":
        # Cleanup completed tasks
        removed = command_pipeline.cleanup_completed_tasks()
        print(f"Removed {removed} completed tasks")
    
    return 0
