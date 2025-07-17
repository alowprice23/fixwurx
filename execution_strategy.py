#!/usr/bin/env python3
"""
execution_strategy.py
────────────────────
Execution strategy for implementing the three-strike rule and handling blockers.

This module provides a comprehensive execution strategy that combines blocker detection
and structured response protocols to implement the complete three-strike rule for
handling stuck operations.
"""

import os
import sys
import time
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum

# Internal imports
from shell_environment import register_command, emit_event, EventType
from meta_agent import request_agent_task
from blocker_detection import BlockerType, BlockerSeverity, BlockerDetector
from response_protocol import ResponseStrategy, ResponseStage, ResponseProtocol

# Configure logging
logger = logging.getLogger("ExecutionStrategy")
handler = logging.FileHandler("execution_strategy.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ExecutionStatus(Enum):
    """Status of an execution."""
    PENDING = "pending"           # Not yet started
    RUNNING = "running"           # Currently running
    BLOCKED = "blocked"           # Blocked by an issue
    RESOLVING = "resolving"       # Attempting to resolve a blocker
    RESOLVED = "resolved"         # Blocker resolved
    COMPLETED = "completed"       # Successfully completed
    FAILED = "failed"             # Failed to complete
    ABORTED = "aborted"           # Aborted due to unresolvable blocker

class ExecutionStrategy:
    """
    Execution strategy for the three-strike rule.
    
    Ties together blocker detection and response protocols to create a complete
    anti-stuck system that can recover from issues automatically.
    """
    
    def __init__(self):
        """Initialize the execution strategy."""
        self.executions = {}
        self.blocker_detector = BlockerDetector()
        self.response_protocol = ResponseProtocol()
        self.max_attempts = 3  # Three-strike rule
        
        # Register commands
        try:
            register_command("execute_with_strategy", self.execute_with_strategy_command, 
                            "Execute an operation with the three-strike rule")
            register_command("check_execution", self.check_execution_command,
                            "Check the status of an execution")
            register_command("list_executions", self.list_executions_command,
                            "List all executions")
            logger.info("Execution strategy commands registered")
        except Exception as e:
            logger.error(f"Failed to register commands: {e}")
    
    def execute_with_strategy(self, 
                              operation_func: Callable, 
                              operation_args: Dict[str, Any], 
                              operation_id: Optional[str] = None, 
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute an operation with the three-strike rule.
        
        Args:
            operation_func: Function to execute
            operation_args: Arguments to pass to the function
            operation_id: Optional ID for the operation (generated if not provided)
            context: Optional additional context for blocker detection
            
        Returns:
            Execution result
        """
        # Generate operation ID if not provided
        if not operation_id:
            operation_id = f"op_{int(time.time())}_{id(operation_func)}"
        
        # Initialize context if not provided
        if not context:
            context = {}
        
        # Add operation ID to context
        context["operation_id"] = operation_id
        
        # Initialize execution record
        execution = {
            "operation_id": operation_id,
            "start_time": time.time(),
            "status": ExecutionStatus.PENDING.value,
            "attempts": 0,
            "blockers": [],
            "responses": [],
            "results": [],
            "final_result": None,
            "end_time": None
        }
        
        self.executions[operation_id] = execution
        
        # Execute with three-strike rule
        return self._execute_with_retries(operation_func, operation_args, execution, context)
    
    def _execute_with_retries(self, 
                             operation_func: Callable, 
                             operation_args: Dict[str, Any], 
                             execution: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation with retries based on the three-strike rule.
        
        Args:
            operation_func: Function to execute
            operation_args: Arguments to pass to the function
            execution: Execution record
            context: Context for blocker detection
            
        Returns:
            Execution result
        """
        operation_id = execution["operation_id"]
        max_attempts = self.max_attempts
        current_attempt = 0
        
        while current_attempt < max_attempts:
            # Increment attempt counter
            current_attempt += 1
            execution["attempts"] = current_attempt
            execution["status"] = ExecutionStatus.RUNNING.value
            
            # Update context with attempt information
            context["attempt"] = current_attempt
            context["execution_states"] = execution.get("execution_states", [])
            context["progress_history"] = execution.get("progress_history", [])
            
            # Record start time for this attempt
            attempt_start_time = time.time()
            context["operation_start_time"] = attempt_start_time
            
            # Log attempt
            logger.info(f"Executing operation {operation_id}, attempt {current_attempt}/{max_attempts}")
            
            # Execute operation
            try:
                # Update status
                execution["status"] = ExecutionStatus.RUNNING.value
                
                # Execute operation
                result = operation_func(**operation_args)
                
                # Record result
                execution["results"].append({
                    "attempt": current_attempt,
                    "result": result,
                    "success": True,
                    "error": None,
                    "elapsed_time": time.time() - attempt_start_time
                })
                
                # Operation succeeded
                execution["status"] = ExecutionStatus.COMPLETED.value
                execution["final_result"] = result
                execution["end_time"] = time.time()
                
                # Log success
                logger.info(f"Operation {operation_id} completed successfully on attempt {current_attempt}")
                
                # Return result
                return {
                    "operation_id": operation_id,
                    "status": ExecutionStatus.COMPLETED.value,
                    "result": result,
                    "attempts": current_attempt,
                    "execution": execution
                }
                
            except Exception as e:
                # Record error
                error_info = {
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
                
                execution["results"].append({
                    "attempt": current_attempt,
                    "result": None,
                    "success": False,
                    "error": error_info,
                    "elapsed_time": time.time() - attempt_start_time
                })
                
                # Update context with error information
                context["operation_type"] = operation_func.__name__
                context["elapsed_time"] = time.time() - attempt_start_time
                context["errors"] = context.get("errors", []) + [str(e)]
                
                # Detect blockers
                logger.info(f"Detecting blockers for operation {operation_id}, attempt {current_attempt}")
                blocker_result = self.blocker_detector.detect_blockers(context)
                
                if blocker_result.get("has_blockers", False):
                    # Get blockers
                    blockers = blocker_result.get("blockers", [])
                    
                    # Record blockers
                    execution["blockers"].extend(blockers)
                    execution["status"] = ExecutionStatus.BLOCKED.value
                    
                    # Log blockers
                    for blocker in blockers:
                        logger.warning(f"Blocker detected for operation {operation_id}: {blocker.get('message', 'Unknown blocker')}")
                    
                    # Check if this is the last attempt
                    if current_attempt >= max_attempts:
                        # Last attempt, abort
                        execution["status"] = ExecutionStatus.ABORTED.value
                        execution["end_time"] = time.time()
                        
                        # Log failure
                        logger.error(f"Operation {operation_id} aborted after {current_attempt} attempts")
                        
                        # Return failure result
                        return {
                            "operation_id": operation_id,
                            "status": ExecutionStatus.ABORTED.value,
                            "result": None,
                            "attempts": current_attempt,
                            "execution": execution,
                            "error": f"Operation aborted after {current_attempt} attempts due to unresolved blockers"
                        }
                    
                    # Generate response for each blocker
                    for blocker in blockers:
                        # Generate response
                        logger.info(f"Generating response for blocker in operation {operation_id}, attempt {current_attempt}")
                        response = self.response_protocol.generate_response(blocker, operation_id, current_attempt)
                        
                        # Record response
                        execution["responses"].append(response)
                        execution["status"] = ExecutionStatus.RESOLVING.value
                        
                        # Log response
                        logger.info(f"Response for operation {operation_id}, attempt {current_attempt}: {response.get('strategy', 'Unknown strategy')}")
                        
                        # Execute response actions
                        self._execute_response_actions(response, context)
                    
                    # Update status to indicate resolution is complete
                    execution["status"] = ExecutionStatus.RESOLVED.value
                    
                else:
                    # No blockers detected, but operation failed
                    logger.warning(f"Operation {operation_id} failed but no blockers detected, retrying...")
                    
                    # Simulate basic retry response
                    response = {
                        "operation_id": operation_id,
                        "attempt": current_attempt,
                        "stage": ResponseStage.INITIAL.value if current_attempt == 1 else ResponseStage.SECONDARY.value if current_attempt == 2 else ResponseStage.FINAL.value,
                        "strategy": ResponseStrategy.RETRY.value,
                        "actions": ["Wait for a short period", "Retry the operation"],
                        "expected_outcome": "Operation succeeds on retry",
                        "timestamp": time.time()
                    }
                    
                    # Record response
                    execution["responses"].append(response)
                    
                    # Wait before retrying
                    time.sleep(2)  # Basic wait before retry
            
            # Delay between attempts
            if current_attempt < max_attempts:
                time.sleep(1)  # Additional delay between attempts
        
        # If we reach here, all attempts have failed
        execution["status"] = ExecutionStatus.FAILED.value
        execution["end_time"] = time.time()
        
        # Log failure
        logger.error(f"Operation {operation_id} failed after {max_attempts} attempts")
        
        # Return failure result
        return {
            "operation_id": operation_id,
            "status": ExecutionStatus.FAILED.value,
            "result": None,
            "attempts": max_attempts,
            "execution": execution,
            "error": f"Operation failed after {max_attempts} attempts"
        }
    
    def _execute_response_actions(self, response: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Execute the actions in a response.
        
        Args:
            response: Response to execute
            context: Context for execution
        """
        operation_id = response.get("operation_id", "unknown")
        strategy = response.get("strategy", "unknown")
        actions = response.get("actions", [])
        
        logger.info(f"Executing {len(actions)} actions for operation {operation_id}, strategy {strategy}")
        
        for i, action in enumerate(actions, 1):
            logger.info(f"Executing action {i}/{len(actions)}: {action}")
            
            # Here you would implement the actual action execution
            # For now, we'll just simulate waiting for actions to complete
            time.sleep(0.5)  # Simulate action execution
            
            # For human intervention, ask the agent
            if strategy == ResponseStrategy.HUMAN_INTERVENTION.value:
                self._request_human_intervention(operation_id, action, context)
    
    def _request_human_intervention(self, operation_id: str, action: str, context: Dict[str, Any]) -> None:
        """
        Request human intervention using an agent.
        
        Args:
            operation_id: ID of the operation
            action: Action requiring human intervention
            context: Context for execution
        """
        # Prepare prompt for agent
        prompt = f"""
        Human intervention required for operation {operation_id}:
        
        Action needed: {action}
        
        Context:
        {json.dumps(context, indent=2)}
        
        Please provide detailed instructions on what human intervention is needed.
        """
        
        try:
            # Request intervention from agent
            result = request_agent_task("human_intervention", prompt, timeout=30)
            
            if result.get("success", False):
                logger.info(f"Human intervention request for operation {operation_id}: {result.get('message', 'No message')}")
            else:
                logger.warning(f"Failed to request human intervention for operation {operation_id}")
        except Exception as e:
            logger.error(f"Error requesting human intervention for operation {operation_id}: {e}")
    
    def execute_with_strategy_command(self, args: str) -> int:
        """
        Handle the execute_with_strategy command.
        
        Args:
            args: Command arguments (function_name args_json)
            
        Returns:
            Exit code
        """
        try:
            arg_parts = args.strip().split(' ', 1)
            
            if len(arg_parts) < 1:
                print("Error: Function name required.")
                print("Usage: execute_with_strategy <function_name> [args_json]")
                return 1
            
            function_name = arg_parts[0]
            args_json = arg_parts[1] if len(arg_parts) > 1 else "{}"
            
            # Parse arguments JSON
            try:
                operation_args = json.loads(args_json)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON arguments: {args_json}")
                return 1
            
            # Get function from globals
            if function_name in globals():
                operation_func = globals()[function_name]
                if not callable(operation_func):
                    print(f"Error: {function_name} is not callable.")
                    return 1
            else:
                # Try to get from built-in functions
                try:
                    operation_func = getattr(__builtins__, function_name)
                except (AttributeError, TypeError):
                    print(f"Error: Function {function_name} not found.")
                    return 1
            
            # Execute operation with strategy
            print(f"Executing {function_name} with three-strike rule...")
            result = self.execute_with_strategy(operation_func, operation_args)
            
            # Print result
            if result["status"] == ExecutionStatus.COMPLETED.value:
                print(f"Operation completed successfully after {result['attempts']} attempt(s).")
                print(f"Result: {result['result']}")
                return 0
            else:
                print(f"Operation {result['status']} after {result['attempts']} attempt(s).")
                if "error" in result:
                    print(f"Error: {result['error']}")
                return 1
                
        except Exception as e:
            print(f"Error executing operation: {e}")
            logger.error(f"Error executing operation: {e}")
            return 1
    
    def check_execution_command(self, args: str) -> int:
        """
        Handle the check_execution command.
        
        Args:
            args: Command arguments (operation_id)
            
        Returns:
            Exit code
        """
        try:
            operation_id = args.strip()
            
            if not operation_id:
                print("Error: Operation ID required.")
                print("Usage: check_execution <operation_id>")
                return 1
            
            if operation_id in self.executions:
                execution = self.executions[operation_id]
                
                print(f"Execution {operation_id}:")
                print(f"  Status: {execution['status']}")
                print(f"  Attempts: {execution['attempts']}")
                print(f"  Start time: {time.ctime(execution['start_time'])}")
                
                if execution.get("end_time"):
                    print(f"  End time: {time.ctime(execution['end_time'])}")
                    print(f"  Duration: {execution['end_time'] - execution['start_time']:.2f} seconds")
                
                if execution.get("blockers"):
                    print(f"\nBlockers ({len(execution['blockers'])}):")
                    for i, blocker in enumerate(execution['blockers'], 1):
                        print(f"  {i}. {blocker.get('message', 'Unknown blocker')}")
                        print(f"     Type: {blocker.get('type', 'Unknown')}")
                        print(f"     Severity: {blocker.get('severity', 'Unknown')}")
                
                if execution.get("responses"):
                    print(f"\nResponses ({len(execution['responses'])}):")
                    for i, response in enumerate(execution['responses'], 1):
                        print(f"  {i}. Strategy: {response.get('strategy', 'Unknown')}")
                        print(f"     Stage: {response.get('stage', 'Unknown')}")
                        print(f"     Actions: {len(response.get('actions', []))}")
                
                if execution.get("final_result") is not None:
                    print(f"\nFinal result: {execution['final_result']}")
                
                return 0
            else:
                print(f"No execution found with ID {operation_id}.")
                return 1
                
        except Exception as e:
            print(f"Error checking execution: {e}")
            logger.error(f"Error checking execution: {e}")
            return 1
    
    def list_executions_command(self, args: str) -> int:
        """
        Handle the list_executions command.
        
        Args:
            args: Command arguments (optional filter)
            
        Returns:
            Exit code
        """
        try:
            status_filter = args.strip() if args.strip() else None
            
            executions = self.executions.values()
            
            if status_filter:
                try:
                    status_filter = ExecutionStatus(status_filter).value
                    executions = [e for e in executions if e.get("status") == status_filter]
                except ValueError:
                    print(f"Error: Invalid status filter: {status_filter}")
                    print("Valid statuses:")
                    for status in ExecutionStatus:
                        print(f"  {status.value}")
                    return 1
            
            print(f"Found {len(executions)} executions:")
            
            for execution in sorted(executions, key=lambda e: e.get("start_time", 0), reverse=True):
                operation_id = execution.get("operation_id", "unknown")
                status = execution.get("status", "unknown")
                attempts = execution.get("attempts", 0)
                start_time = time.ctime(execution.get("start_time", 0))
                
                print(f"{operation_id}: {status}, {attempts} attempt(s), started at {start_time}")
            
            return 0
                
        except Exception as e:
            print(f"Error listing executions: {e}")
            logger.error(f"Error listing executions: {e}")
            return 1


# Initialize execution strategy
execution_strategy = ExecutionStrategy()
logger.info("Execution strategy initialized")

# Example test function for the execution strategy
def test_function(succeed_after: int = 3, sleep_time: float = 1.0, error_message: str = "Simulated error") -> str:
    """
    Test function that fails a certain number of times before succeeding.
    
    Args:
        succeed_after: Number of attempts before succeeding (0 to always succeed)
        sleep_time: Time to sleep in seconds
        error_message: Error message to throw
        
    Returns:
        Success message
    """
    # Access the execution context
    if "attempt" in globals():
        attempt = globals()["attempt"]
    else:
        # Get attempt from context if available
        context = globals().get("context", {})
        attempt = context.get("attempt", 1)
    
    # Sleep to simulate work
    time.sleep(sleep_time)
    
    # Fail if we haven't reached the succeed_after count
    if succeed_after > 0 and attempt <= succeed_after:
        raise Exception(f"{error_message} (attempt {attempt}/{succeed_after})")
    
    # Otherwise succeed
    return f"Operation completed successfully on attempt {attempt}"
