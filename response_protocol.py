#!/usr/bin/env python3
"""
response_protocol.py
───────────────────
Structured response protocol for handling blockers during execution.

This module provides a framework for systematic responses to blockers detected
during system execution, implementing the three-strike rule and ensuring the
system can recover from or work around detected issues.
"""

import os
import sys
import time
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum

# Internal imports
from shell_environment import register_command, emit_event, EventType
from meta_agent import request_agent_task
from blocker_detection import BlockerType, BlockerSeverity

# Configure logging
logger = logging.getLogger("ResponseProtocol")
handler = logging.FileHandler("response_protocol.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ResponseStrategy(Enum):
    """Types of response strategies for blockers."""
    RETRY = "retry"                 # Simply retry the operation
    WORKAROUND = "workaround"       # Try an alternative approach
    RESOURCE_ADJUSTMENT = "resource_adjustment"  # Adjust resource allocation
    DEPENDENCY_RESOLUTION = "dependency_resolution"  # Resolve dependency issues
    ACCESS_VERIFICATION = "access_verification"      # Verify file/resource access
    TIMEOUT_EXTENSION = "timeout_extension"  # Extend timeout limits
    ALTERNATIVE_APPROACH = "alternative_approach"  # Try a completely different approach
    HUMAN_INTERVENTION = "human_intervention"  # Request human help
    ABORT = "abort"                 # Abort the operation

class ResponseStage(Enum):
    """Stages of the response protocol."""
    INITIAL = "initial"             # First attempt at resolution
    SECONDARY = "secondary"         # Second attempt with more aggressive measures
    FINAL = "final"                 # Final attempt before giving up
    EMERGENCY = "emergency"         # Emergency measures when all else fails

class ResponseProtocol:
    """
    Structured response protocol for handling blockers.
    
    Implements the three-strike rule:
    1. First strike: Simple retry or basic workaround
    2. Second strike: More aggressive workaround or alternative approach
    3. Third strike: Comprehensive solution or human intervention
    """
    
    def __init__(self):
        """Initialize the response protocol."""
        self.response_history = {}
        self.strategy_mapping = self._create_strategy_mapping()
        
        # Register commands
        try:
            register_command("generate_response", self.generate_response_command, 
                            "Generate a response to a blocker")
            register_command("list_responses", self.list_responses_command,
                            "List responses for a given operation")
            logger.info("Response protocol commands registered")
        except Exception as e:
            logger.error(f"Failed to register commands: {e}")
    
    def _create_strategy_mapping(self) -> Dict:
        """
        Create a mapping from blocker types and severities to response strategies.
        
        Returns:
            Dictionary mapping (type, severity, stage) to strategy
        """
        mapping = {}
        
        # Dependency blockers
        mapping[(BlockerType.DEPENDENCY.value, BlockerSeverity.LOW.value, ResponseStage.INITIAL.value)] = ResponseStrategy.RETRY.value
        mapping[(BlockerType.DEPENDENCY.value, BlockerSeverity.LOW.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.DEPENDENCY_RESOLUTION.value
        mapping[(BlockerType.DEPENDENCY.value, BlockerSeverity.LOW.value, ResponseStage.FINAL.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        
        mapping[(BlockerType.DEPENDENCY.value, BlockerSeverity.MEDIUM.value, ResponseStage.INITIAL.value)] = ResponseStrategy.DEPENDENCY_RESOLUTION.value
        mapping[(BlockerType.DEPENDENCY.value, BlockerSeverity.MEDIUM.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        mapping[(BlockerType.DEPENDENCY.value, BlockerSeverity.MEDIUM.value, ResponseStage.FINAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        
        mapping[(BlockerType.DEPENDENCY.value, BlockerSeverity.HIGH.value, ResponseStage.INITIAL.value)] = ResponseStrategy.DEPENDENCY_RESOLUTION.value
        mapping[(BlockerType.DEPENDENCY.value, BlockerSeverity.HIGH.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.DEPENDENCY.value, BlockerSeverity.HIGH.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        mapping[(BlockerType.DEPENDENCY.value, BlockerSeverity.FATAL.value, ResponseStage.INITIAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.DEPENDENCY.value, BlockerSeverity.FATAL.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ABORT.value
        mapping[(BlockerType.DEPENDENCY.value, BlockerSeverity.FATAL.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        # Permission blockers
        mapping[(BlockerType.PERMISSION.value, BlockerSeverity.LOW.value, ResponseStage.INITIAL.value)] = ResponseStrategy.RETRY.value
        mapping[(BlockerType.PERMISSION.value, BlockerSeverity.LOW.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ACCESS_VERIFICATION.value
        mapping[(BlockerType.PERMISSION.value, BlockerSeverity.LOW.value, ResponseStage.FINAL.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        
        mapping[(BlockerType.PERMISSION.value, BlockerSeverity.MEDIUM.value, ResponseStage.INITIAL.value)] = ResponseStrategy.ACCESS_VERIFICATION.value
        mapping[(BlockerType.PERMISSION.value, BlockerSeverity.MEDIUM.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        mapping[(BlockerType.PERMISSION.value, BlockerSeverity.MEDIUM.value, ResponseStage.FINAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        
        mapping[(BlockerType.PERMISSION.value, BlockerSeverity.HIGH.value, ResponseStage.INITIAL.value)] = ResponseStrategy.ACCESS_VERIFICATION.value
        mapping[(BlockerType.PERMISSION.value, BlockerSeverity.HIGH.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.PERMISSION.value, BlockerSeverity.HIGH.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        mapping[(BlockerType.PERMISSION.value, BlockerSeverity.FATAL.value, ResponseStage.INITIAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.PERMISSION.value, BlockerSeverity.FATAL.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ABORT.value
        mapping[(BlockerType.PERMISSION.value, BlockerSeverity.FATAL.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        # Resource blockers
        mapping[(BlockerType.RESOURCE.value, BlockerSeverity.LOW.value, ResponseStage.INITIAL.value)] = ResponseStrategy.RETRY.value
        mapping[(BlockerType.RESOURCE.value, BlockerSeverity.LOW.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.RESOURCE_ADJUSTMENT.value
        mapping[(BlockerType.RESOURCE.value, BlockerSeverity.LOW.value, ResponseStage.FINAL.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        
        mapping[(BlockerType.RESOURCE.value, BlockerSeverity.MEDIUM.value, ResponseStage.INITIAL.value)] = ResponseStrategy.RESOURCE_ADJUSTMENT.value
        mapping[(BlockerType.RESOURCE.value, BlockerSeverity.MEDIUM.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        mapping[(BlockerType.RESOURCE.value, BlockerSeverity.MEDIUM.value, ResponseStage.FINAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        
        mapping[(BlockerType.RESOURCE.value, BlockerSeverity.HIGH.value, ResponseStage.INITIAL.value)] = ResponseStrategy.RESOURCE_ADJUSTMENT.value
        mapping[(BlockerType.RESOURCE.value, BlockerSeverity.HIGH.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.RESOURCE.value, BlockerSeverity.HIGH.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        mapping[(BlockerType.RESOURCE.value, BlockerSeverity.FATAL.value, ResponseStage.INITIAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.RESOURCE.value, BlockerSeverity.FATAL.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ABORT.value
        mapping[(BlockerType.RESOURCE.value, BlockerSeverity.FATAL.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        # Timeout blockers
        mapping[(BlockerType.TIMEOUT.value, BlockerSeverity.LOW.value, ResponseStage.INITIAL.value)] = ResponseStrategy.RETRY.value
        mapping[(BlockerType.TIMEOUT.value, BlockerSeverity.LOW.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.TIMEOUT_EXTENSION.value
        mapping[(BlockerType.TIMEOUT.value, BlockerSeverity.LOW.value, ResponseStage.FINAL.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        
        mapping[(BlockerType.TIMEOUT.value, BlockerSeverity.MEDIUM.value, ResponseStage.INITIAL.value)] = ResponseStrategy.TIMEOUT_EXTENSION.value
        mapping[(BlockerType.TIMEOUT.value, BlockerSeverity.MEDIUM.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        mapping[(BlockerType.TIMEOUT.value, BlockerSeverity.MEDIUM.value, ResponseStage.FINAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        
        mapping[(BlockerType.TIMEOUT.value, BlockerSeverity.HIGH.value, ResponseStage.INITIAL.value)] = ResponseStrategy.TIMEOUT_EXTENSION.value
        mapping[(BlockerType.TIMEOUT.value, BlockerSeverity.HIGH.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.TIMEOUT.value, BlockerSeverity.HIGH.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        mapping[(BlockerType.TIMEOUT.value, BlockerSeverity.FATAL.value, ResponseStage.INITIAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.TIMEOUT.value, BlockerSeverity.FATAL.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ABORT.value
        mapping[(BlockerType.TIMEOUT.value, BlockerSeverity.FATAL.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        # Logic blockers
        mapping[(BlockerType.LOGIC.value, BlockerSeverity.LOW.value, ResponseStage.INITIAL.value)] = ResponseStrategy.RETRY.value
        mapping[(BlockerType.LOGIC.value, BlockerSeverity.LOW.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.WORKAROUND.value
        mapping[(BlockerType.LOGIC.value, BlockerSeverity.LOW.value, ResponseStage.FINAL.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        
        mapping[(BlockerType.LOGIC.value, BlockerSeverity.MEDIUM.value, ResponseStage.INITIAL.value)] = ResponseStrategy.WORKAROUND.value
        mapping[(BlockerType.LOGIC.value, BlockerSeverity.MEDIUM.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        mapping[(BlockerType.LOGIC.value, BlockerSeverity.MEDIUM.value, ResponseStage.FINAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        
        mapping[(BlockerType.LOGIC.value, BlockerSeverity.HIGH.value, ResponseStage.INITIAL.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        mapping[(BlockerType.LOGIC.value, BlockerSeverity.HIGH.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.LOGIC.value, BlockerSeverity.HIGH.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        mapping[(BlockerType.LOGIC.value, BlockerSeverity.FATAL.value, ResponseStage.INITIAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.LOGIC.value, BlockerSeverity.FATAL.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ABORT.value
        mapping[(BlockerType.LOGIC.value, BlockerSeverity.FATAL.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        # External blockers
        mapping[(BlockerType.EXTERNAL.value, BlockerSeverity.LOW.value, ResponseStage.INITIAL.value)] = ResponseStrategy.RETRY.value
        mapping[(BlockerType.EXTERNAL.value, BlockerSeverity.LOW.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.WORKAROUND.value
        mapping[(BlockerType.EXTERNAL.value, BlockerSeverity.LOW.value, ResponseStage.FINAL.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        
        mapping[(BlockerType.EXTERNAL.value, BlockerSeverity.MEDIUM.value, ResponseStage.INITIAL.value)] = ResponseStrategy.WORKAROUND.value
        mapping[(BlockerType.EXTERNAL.value, BlockerSeverity.MEDIUM.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        mapping[(BlockerType.EXTERNAL.value, BlockerSeverity.MEDIUM.value, ResponseStage.FINAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        
        mapping[(BlockerType.EXTERNAL.value, BlockerSeverity.HIGH.value, ResponseStage.INITIAL.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        mapping[(BlockerType.EXTERNAL.value, BlockerSeverity.HIGH.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.EXTERNAL.value, BlockerSeverity.HIGH.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        mapping[(BlockerType.EXTERNAL.value, BlockerSeverity.FATAL.value, ResponseStage.INITIAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.EXTERNAL.value, BlockerSeverity.FATAL.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ABORT.value
        mapping[(BlockerType.EXTERNAL.value, BlockerSeverity.FATAL.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        # Unknown blockers
        mapping[(BlockerType.UNKNOWN.value, BlockerSeverity.LOW.value, ResponseStage.INITIAL.value)] = ResponseStrategy.RETRY.value
        mapping[(BlockerType.UNKNOWN.value, BlockerSeverity.LOW.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.WORKAROUND.value
        mapping[(BlockerType.UNKNOWN.value, BlockerSeverity.LOW.value, ResponseStage.FINAL.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        
        mapping[(BlockerType.UNKNOWN.value, BlockerSeverity.MEDIUM.value, ResponseStage.INITIAL.value)] = ResponseStrategy.WORKAROUND.value
        mapping[(BlockerType.UNKNOWN.value, BlockerSeverity.MEDIUM.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        mapping[(BlockerType.UNKNOWN.value, BlockerSeverity.MEDIUM.value, ResponseStage.FINAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        
        mapping[(BlockerType.UNKNOWN.value, BlockerSeverity.HIGH.value, ResponseStage.INITIAL.value)] = ResponseStrategy.ALTERNATIVE_APPROACH.value
        mapping[(BlockerType.UNKNOWN.value, BlockerSeverity.HIGH.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.UNKNOWN.value, BlockerSeverity.HIGH.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        mapping[(BlockerType.UNKNOWN.value, BlockerSeverity.FATAL.value, ResponseStage.INITIAL.value)] = ResponseStrategy.HUMAN_INTERVENTION.value
        mapping[(BlockerType.UNKNOWN.value, BlockerSeverity.FATAL.value, ResponseStage.SECONDARY.value)] = ResponseStrategy.ABORT.value
        mapping[(BlockerType.UNKNOWN.value, BlockerSeverity.FATAL.value, ResponseStage.FINAL.value)] = ResponseStrategy.ABORT.value
        
        # Emergency stage always aborts
        for blocker_type in BlockerType:
            for severity in BlockerSeverity:
                mapping[(blocker_type.value, severity.value, ResponseStage.EMERGENCY.value)] = ResponseStrategy.ABORT.value
        
        return mapping
    
    def generate_response(self, blocker: Dict[str, Any], operation_id: str, attempt: int = 1) -> Dict[str, Any]:
        """
        Generate a response to a blocker.
        
        Args:
            blocker: Blocker information
            operation_id: ID of the operation
            attempt: Attempt number (1-based)
            
        Returns:
            Response information
        """
        # Determine stage based on attempt
        if attempt <= 1:
            stage = ResponseStage.INITIAL.value
        elif attempt == 2:
            stage = ResponseStage.SECONDARY.value
        elif attempt == 3:
            stage = ResponseStage.FINAL.value
        else:
            stage = ResponseStage.EMERGENCY.value
        
        # Get blocker type and severity
        blocker_type = blocker.get("type", BlockerType.UNKNOWN.value)
        blocker_severity = blocker.get("severity", BlockerSeverity.MEDIUM.value)
        
        # Determine strategy
        key = (blocker_type, blocker_severity, stage)
        strategy = self.strategy_mapping.get(key, ResponseStrategy.RETRY.value)
        
        # Get specific response from agent
        response_details = self._get_response_details(blocker, strategy)
        
        # Create response
        response = {
            "operation_id": operation_id,
            "blocker": blocker,
            "attempt": attempt,
            "stage": stage,
            "strategy": strategy,
            "actions": response_details.get("actions", []),
            "expected_outcome": response_details.get("expected_outcome", "Unknown"),
            "fallback": response_details.get("fallback", None),
            "timestamp": time.time()
        }
        
        # Store response in history
        if operation_id not in self.response_history:
            self.response_history[operation_id] = []
        
        self.response_history[operation_id].append(response)
        
        # Log response
        logger.info(f"Generated response for operation {operation_id}, attempt {attempt}: {strategy}")
        
        return response
    
    def _get_response_details(self, blocker: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """
        Get detailed response actions for a blocker and strategy.
        
        Uses an LLM agent to generate specific actions to take.
        
        Args:
            blocker: Blocker information
            strategy: Response strategy
            
        Returns:
            Response details including actions and expected outcome
        """
        # Prepare prompt for agent
        prompt = f"""
        Generate a detailed response for the following blocker using the {strategy} strategy:
        
        {json.dumps(blocker, indent=2)}
        
        Your response should include:
        1. A list of specific actions to take to resolve the blocker
        2. The expected outcome of these actions
        3. A fallback plan if these actions fail
        
        Return your response as a structured object.
        """
        
        try:
            # Request response from agent
            result = request_agent_task("response_generation", prompt, timeout=30)
            
            if result.get("success", False) and result.get("response_details"):
                return result["response_details"]
            
            # Fallback to basic response if agent fails
            return self._generate_basic_response(blocker, strategy)
        except Exception as e:
            logger.error(f"Error getting response details from agent: {e}")
            return self._generate_basic_response(blocker, strategy)
    
    def _generate_basic_response(self, blocker: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """
        Generate a basic response without using an agent.
        
        Args:
            blocker: Blocker information
            strategy: Response strategy
            
        Returns:
            Basic response details
        """
        actions = []
        expected_outcome = "Resolve the blocker"
        fallback = None
        
        # Generate basic actions based on strategy
        if strategy == ResponseStrategy.RETRY.value:
            actions.append("Wait for a short period")
            actions.append("Retry the operation with the same parameters")
            expected_outcome = "Operation succeeds on retry"
            fallback = "Try with modified parameters"
            
        elif strategy == ResponseStrategy.WORKAROUND.value:
            actions.append("Identify alternative method to accomplish the task")
            actions.append("Implement workaround using available functionality")
            expected_outcome = "Operation succeeds through alternative method"
            fallback = "Try a completely different approach"
            
        elif strategy == ResponseStrategy.RESOURCE_ADJUSTMENT.value:
            actions.append("Reduce resource usage of other components")
            actions.append("Increase allocation to the blocked operation")
            actions.append("Try operation with increased resources")
            expected_outcome = "Operation succeeds with adequate resources"
            fallback = "Try operation with minimum required resources"
            
        elif strategy == ResponseStrategy.DEPENDENCY_RESOLUTION.value:
            actions.append("Identify missing or incompatible dependencies")
            actions.append("Install or update required dependencies")
            actions.append("Retry operation with dependencies satisfied")
            expected_outcome = "Operation succeeds with proper dependencies"
            fallback = "Try alternative libraries or versions"
            
        elif strategy == ResponseStrategy.ACCESS_VERIFICATION.value:
            actions.append("Verify file and resource access paths")
            actions.append("Check if resources exist and are accessible")
            actions.append("Retry operation with verified access paths")
            expected_outcome = "Operation succeeds with proper access"
            fallback = "Try alternative resources or paths"
            
        elif strategy == ResponseStrategy.TIMEOUT_EXTENSION.value:
            actions.append("Increase timeout threshold")
            actions.append("Retry operation with extended timeout")
            expected_outcome = "Operation completes with extended time"
            fallback = "Try breaking operation into smaller chunks"
            
        elif strategy == ResponseStrategy.ALTERNATIVE_APPROACH.value:
            actions.append("Research alternative algorithms or methods")
            actions.append("Implement completely different approach")
            actions.append("Validate new approach meets requirements")
            expected_outcome = "Operation succeeds through different method"
            fallback = "Simplify requirements and try again"
            
        elif strategy == ResponseStrategy.HUMAN_INTERVENTION.value:
            actions.append("Generate detailed report of the issue")
            actions.append("Request human intervention with specific needs")
            actions.append("Wait for human input before proceeding")
            expected_outcome = "Human resolves blocking issue"
            fallback = "Proceed with limited functionality if possible"
            
        elif strategy == ResponseStrategy.ABORT.value:
            actions.append("Clean up any partial state changes")
            actions.append("Log detailed information about the failure")
            actions.append("Abort operation and report failure")
            expected_outcome = "Operation aborted cleanly"
            fallback = "Emergency shutdown procedure"
        
        return {
            "actions": actions,
            "expected_outcome": expected_outcome,
            "fallback": fallback
        }
    
    def generate_response_command(self, args: str) -> int:
        """
        Handle the generate_response command.
        
        Args:
            args: Command arguments (blocker_file operation_id attempt)
            
        Returns:
            Exit code
        """
        try:
            arg_parts = args.strip().split()
            
            if len(arg_parts) < 2:
                print("Error: Insufficient arguments.")
                print("Usage: generate_response <blocker_file> <operation_id> [attempt]")
                return 1
            
            blocker_file = arg_parts[0]
            operation_id = arg_parts[1]
            attempt = int(arg_parts[2]) if len(arg_parts) > 2 and arg_parts[2].isdigit() else 1
            
            # Load blocker from file
            try:
                with open(blocker_file, 'r') as f:
                    blocker = json.load(f)
            except Exception as e:
                print(f"Error loading blocker file: {e}")
                return 1
            
            # Generate response
            response = self.generate_response(blocker, operation_id, attempt)
            
            # Print response
            print(f"Response Strategy: {response['strategy']}")
            print(f"Stage: {response['stage']}")
            
            print("\nActions:")
            for i, action in enumerate(response['actions'], 1):
                print(f"  {i}. {action}")
            
            print(f"\nExpected Outcome: {response['expected_outcome']}")
            
            if response.get('fallback'):
                print(f"Fallback: {response['fallback']}")
            
            # Save response to file
            output_file = f"response_{operation_id}_{attempt}.json"
            with open(output_file, 'w') as f:
                json.dump(response, f, indent=2)
            
            print(f"\nResponse saved to {output_file}")
            
            return 0
                
        except Exception as e:
            print(f"Error generating response: {e}")
            logger.error(f"Error generating response: {e}")
            return 1
    
    def list_responses_command(self, args: str) -> int:
        """
        Handle the list_responses command.
        
        Args:
            args: Command arguments (operation_id)
            
        Returns:
            Exit code
        """
        try:
            operation_id = args.strip()
            
            if not operation_id:
                print("Error: Operation ID required.")
                print("Usage: list_responses <operation_id>")
                return 1
            
            if operation_id in self.response_history:
                responses = self.response_history[operation_id]
                print(f"Found {len(responses)} responses for operation {operation_id}:")
                
                for i, response in enumerate(responses, 1):
                    print(f"\nResponse {i} (Attempt {response['attempt']}):")
                    print(f"  Strategy: {response['strategy']}")
                    print(f"  Stage: {response['stage']}")
                    print("  Actions:")
                    for j, action in enumerate(response['actions'], 1):
                        print(f"    {j}. {action}")
                    print(f"  Expected Outcome: {response['expected_outcome']}")
                    if response.get('fallback'):
                        print(f"  Fallback: {response['fallback']}")
                
                return 0
            else:
                print(f"No responses found for operation {operation_id}.")
                return 1
                
        except Exception as e:
            print(f"Error listing responses: {e}")
            logger.error(f"Error listing responses: {e}")
            return 1


# Initialize response protocol
response_protocol = ResponseProtocol()
logger.info("Response protocol initialized")
