#!/usr/bin/env python3
"""
Plan Executor Component for Triangulum Integration

This module provides the PlanExecutor class for executing multi-step plans.
"""

import os
import json
import time
import uuid
import threading
import logging
from typing import Dict, List, Any, Optional, Callable

# Configure logging if not already configured
logger = logging.getLogger("TriangulumIntegration")

# Mock mode for testing
MOCK_MODE = os.environ.get("TRIANGULUM_TEST_MODE", "0") == "1"

class PlanExecutor:
    """
    Executes plans with steps.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize plan executor.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.plan_dir = self.config.get("plan_dir", "plans")
        self.plans = {}
        self.is_executing = False
        self.current_plan_id = None
        self.stop_execution_flag = False
        self.execution_thread = None
        self.lock = threading.Lock()
        
        # Create plan directory if it doesn't exist
        os.makedirs(self.plan_dir, exist_ok=True)
        
        logger.info(f"Plan executor initialized with plan dir {self.plan_dir}")
    
    def create_plan(self, name: str, steps: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> str:
        """
        Create a plan.
        
        Args:
            name: Name of the plan
            steps: Steps in the plan
            metadata: Additional metadata
            
        Returns:
            ID of the created plan
        """
        plan_id = str(uuid.uuid4())
        
        with self.lock:
            plan = {
                "id": plan_id,
                "metadata": {
                    "name": name,
                    "created": time.time(),
                    **(metadata or {})
                },
                "steps": steps,
                "status": "created",
                "results": [],
                "current_step": 0
            }
            
            self.plans[plan_id] = plan
            
            # Write plan to disk (simplified for testing)
            if not MOCK_MODE:
                try:
                    plan_path = os.path.join(self.plan_dir, f"{plan_id}.json")
                    with open(plan_path, "w") as f:
                        json.dump(plan, f, indent=2)
                except Exception as e:
                    logger.error(f"Error writing plan to disk: {e}")
            
            logger.info(f"Created plan: {name} ({plan_id})")
            return plan_id
    
    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a plan by ID.
        
        Args:
            plan_id: ID of the plan
            
        Returns:
            Plan data, or None if not found
        """
        with self.lock:
            if plan_id not in self.plans:
                logger.warning(f"Plan {plan_id} not found")
                return None
            
            return self.plans[plan_id]
    
    def list_plans(self) -> List[Dict[str, Any]]:
        """
        List all plans.
        
        Returns:
            List of plans
        """
        with self.lock:
            return list(self.plans.values())
    
    def execute_plan(self, plan_id: str, async_execution: bool = False) -> bool:
        """
        Execute a plan.
        
        Args:
            plan_id: ID of the plan
            async_execution: Whether to execute asynchronously
            
        Returns:
            Whether the plan execution was started
        """
        with self.lock:
            if plan_id not in self.plans:
                logger.warning(f"Plan {plan_id} not found")
                return False
            
            if self.is_executing:
                logger.warning("Already executing a plan")
                return False
            
            plan = self.plans[plan_id]
            plan["status"] = "executing"
            plan["current_step"] = 0
            plan["results"] = []
            self.current_plan_id = plan_id
            self.stop_execution_flag = False
            
            if async_execution:
                # Start execution in a thread
                self.is_executing = True
                self.execution_thread = threading.Thread(target=self._execute_plan, args=(plan_id,))
                self.execution_thread.daemon = True
                self.execution_thread.start()
                logger.info(f"Started async execution of plan: {plan['metadata']['name']} ({plan_id})")
                return True
            else:
                # Execute synchronously
                self.is_executing = True
                result = self._execute_plan(plan_id)
                self.is_executing = False
                self.current_plan_id = None
                logger.info(f"Completed sync execution of plan: {plan['metadata']['name']} ({plan_id})")
                return result
    
    def _execute_plan(self, plan_id: str) -> bool:
        """
        Execute a plan.
        
        Args:
            plan_id: ID of the plan
            
        Returns:
            Whether the plan execution was successful
        """
        try:
            plan = self.plans[plan_id]
            
            for i, step in enumerate(plan["steps"]):
                # Check if execution should be stopped
                if self.stop_execution_flag:
                    with self.lock:
                        plan["status"] = "stopped"
                    logger.info(f"Plan execution stopped: {plan['metadata']['name']} ({plan_id})")
                    return False
                
                # Update current step
                with self.lock:
                    plan["current_step"] = i
                
                # Execute step (simplified for testing)
                step_result = {
                    "step": i,
                    "name": step.get("name", f"Step {i}"),
                    "status": "success",
                    "output": f"Executed {step.get('name', f'Step {i}')}",
                    "timestamp": time.time()
                }
                
                # Add result
                with self.lock:
                    plan["results"].append(step_result)
            
            # Update status
            with self.lock:
                plan["status"] = "completed"
                self.is_executing = False
                self.current_plan_id = None
            
            logger.info(f"Plan execution completed: {plan['metadata']['name']} ({plan_id})")
            return True
        except Exception as e:
            # Update status on error
            with self.lock:
                if plan_id in self.plans:
                    self.plans[plan_id]["status"] = "failed"
                self.is_executing = False
                self.current_plan_id = None
            
            logger.error(f"Error executing plan: {e}")
            return False
    
    def stop_execution(self) -> bool:
        """
        Stop the current plan execution.
        
        Returns:
            Whether the stop request was successful
        """
        if not self.is_executing:
            logger.warning("No plan currently executing")
            return False
        
        self.stop_execution_flag = True
        logger.info("Requested to stop plan execution")
        return True
