#!/usr/bin/env python3
"""
Triangulation Engine

This module implements the core execution engine for bug fixing, providing
deterministic phase transitions, path-based execution, and bug state tracking.

The Triangulation Engine is the central component of the FixWurx system that
orchestrates the bug fixing process through multiple phases and execution paths.
"""

# Re-export MetricBus for backward compatibility
from metrics_bus import MetricBus, get_metric_bus, publish_metric

import os
import sys
import json
import time
import uuid
import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(".triangulum/triangulation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("TriangulationEngine")

# Import agent system components
try:
    from agent_commands import (
        _planner, _meta, _bugs,
        plan_generate_command, plan_select_command,
        observe_command, analyze_command, verify_command
    )
except ImportError:
    logger.warning("Agent system not available, using placeholder implementations")
    _planner = None
    _meta = None
    _bugs = {}

# Phase and state definitions
class FixPhase(Enum):
    """Fix phases in the Triangulation Engine."""
    INITIALIZE = "initialize"
    ANALYZE = "analyze"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    LEARN = "learn"
    COMPLETE = "complete"
    FAILED = "failed"
    ABANDONED = "abandoned"

class BugState:
    """Bug state tracker."""
    def __init__(
        self,
        bug_id: str,
        title: str,
        description: Optional[str] = None,
        severity: str = "medium",
        metadata: Dict[str, Any] = None
    ):
        self.bug_id = bug_id
        self.title = title
        self.description = description
        self.severity = severity
        self.metadata = metadata or {}
        self.phase = FixPhase.INITIALIZE
        self.status = "pending"
        self.created_at = time.time()
        self.updated_at = time.time()
        self.path_id = None
        self.active_path = None
        self.phase_history = []
        self.fix_attempts = []
        self.results = {}
    
    def update_phase(self, phase: FixPhase, details: Dict[str, Any] = None) -> None:
        """Update the phase of the bug."""
        self.phase = phase
        self.updated_at = time.time()
        
        # Add to phase history
        self.phase_history.append({
            "phase": phase.value,
            "timestamp": self.updated_at,
            "details": details or {}
        })
        
        # Update status based on phase
        if phase == FixPhase.COMPLETE:
            self.status = "fixed"
        elif phase == FixPhase.FAILED:
            self.status = "failed"
        elif phase == FixPhase.ABANDONED:
            self.status = "abandoned"
        else:
            self.status = "in_progress"
        
        logger.info(f"Bug {self.bug_id} phase updated to {phase.value}")
    
    def add_fix_attempt(self, attempt_data: Dict[str, Any]) -> None:
        """Add a fix attempt to the bug history."""
        attempt_data["timestamp"] = time.time()
        self.fix_attempts.append(attempt_data)
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "bug_id": self.bug_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "phase": self.phase.value,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "path_id": self.path_id,
            "phase_history": self.phase_history,
            "fix_attempts": self.fix_attempts,
            "results": self.results,
            "metadata": self.metadata
        }

class ExecutionPath:
    """Represents a path for fixing a bug."""
    def __init__(
        self,
        path_id: str,
        bug_id: str,
        steps: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None
    ):
        self.path_id = path_id
        self.bug_id = bug_id
        self.steps = steps
        self.metadata = metadata or {}
        self.current_step = 0
        self.status = "pending"
        self.created_at = time.time()
        self.updated_at = time.time()
        self.results = {}
    
    def advance_step(self) -> bool:
        """Advance to the next step in the path."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.updated_at = time.time()
            return True
        
        return False
    
    def get_current_step(self) -> Optional[Dict[str, Any]]:
        """Get the current step in the path."""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        
        return None
    
    def mark_complete(self, success: bool, details: Dict[str, Any] = None) -> None:
        """Mark the execution path as complete."""
        self.status = "successful" if success else "failed"
        self.updated_at = time.time()
        
        if details:
            self.results.update(details)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path_id": self.path_id,
            "bug_id": self.bug_id,
            "steps": self.steps,
            "current_step": self.current_step,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "results": self.results,
            "metadata": self.metadata
        }

class TriangulationEngine:
    """
    Core execution engine for bug fixing.
    
    The TriangulationEngine orchestrates the bug fixing process by:
    1. Managing bug state and execution paths
    2. Implementing deterministic phase transitions
    3. Executing fix steps through specialized agents
    4. Tracking results and learning from outcomes
    """
    
    def __init__(self, config: Dict[str, Any] = None, resource_manager=None):
        """
        Initialize the Triangulation Engine.
        
        Args:
            config: Engine configuration
        """
        self.config = config or {}
        self.resource_manager = resource_manager
        self.bugs = {}  # bug_id -> BugState
        self.paths = {}  # path_id -> ExecutionPath
        self.active_executions = {}  # execution_id -> (bug_id, path_id)
        
        # Create triangulum directory if it doesn't exist
        self.triangulum_dir = Path(".triangulum")
        self.triangulum_dir.mkdir(exist_ok=True)
        
        # Load state if available
        self.state_file = self.triangulum_dir / "engine_state.json"
        self.load_state()
        
        # Initialize locks
        self._bug_lock = threading.RLock()
        self._path_lock = threading.RLock()
        self._execution_lock = threading.RLock()
        
        # Start execution thread if configured
        self.auto_execute = self.config.get("auto_execute", False)
        self._shutdown = threading.Event()
        self._execution_thread = None
        
        if self.auto_execute:
            self.start_execution_thread()
        
        logger.info("Triangulation Engine initialized")
    
    def start_execution_thread(self) -> None:
        """Start the execution thread."""
        if self._execution_thread is not None and self._execution_thread.is_alive():
            logger.warning("Execution thread already running")
            return
        
        self._shutdown.clear()
        self._execution_thread = threading.Thread(target=self._execution_loop)
        self._execution_thread.daemon = True
        self._execution_thread.start()
        logger.info("Execution thread started")
    
    def stop_execution_thread(self) -> None:
        """Stop the execution thread."""
        if self._execution_thread is None or not self._execution_thread.is_alive():
            logger.warning("Execution thread not running")
            return
        
        self._shutdown.set()
        self._execution_thread.join(timeout=5.0)
        logger.info("Execution thread stopped")
    
    def _execution_loop(self) -> None:
        """Main execution loop for automatically processing bugs."""
        while not self._shutdown.is_set():
            try:
                # Process pending bugs
                self._process_pending_bugs()
                
                # Process active executions
                self._process_active_executions()
                
                # Save state
                self.save_state()
                
                # Wait for next cycle
                self._shutdown.wait(self.config.get("execution_interval", 5.0))
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                # Brief pause before continuing
                time.sleep(1.0)
    
    def _process_pending_bugs(self) -> None:
        """Process bugs in the INITIALIZE phase."""
        with self._bug_lock:
            for bug_id, bug in list(self.bugs.items()):
                if bug.phase == FixPhase.INITIALIZE:
                    # Start fixing the bug
                    self.start_bug_fix(bug_id)
    
    def _process_active_executions(self) -> None:
        """Process active executions."""
        with self._execution_lock:
            for exec_id, (bug_id, path_id) in list(self.active_executions.items()):
                try:
                    # Check if bug and path exist
                    if bug_id not in self.bugs or path_id not in self.paths:
                        logger.warning(f"Bug {bug_id} or path {path_id} not found for execution {exec_id}")
                        del self.active_executions[exec_id]
                        continue
                    
                    # Get bug and path
                    bug = self.bugs[bug_id]
                    path = self.paths[path_id]
                    
                    # Process based on bug phase
                    if bug.phase == FixPhase.ANALYZE:
                        self._execute_analyze_phase(bug, path, exec_id)
                    elif bug.phase == FixPhase.PLAN:
                        self._execute_plan_phase(bug, path, exec_id)
                    elif bug.phase == FixPhase.IMPLEMENT:
                        self._execute_implement_phase(bug, path, exec_id)
                    elif bug.phase == FixPhase.VERIFY:
                        self._execute_verify_phase(bug, path, exec_id)
                    elif bug.phase == FixPhase.LEARN:
                        self._execute_learn_phase(bug, path, exec_id)
                    elif bug.phase in (FixPhase.COMPLETE, FixPhase.FAILED, FixPhase.ABANDONED):
                        # Remove from active executions
                        del self.active_executions[exec_id]
                except Exception as e:
                    logger.error(f"Error processing execution {exec_id}: {e}")
    
    def _execute_analyze_phase(self, bug: BugState, path: ExecutionPath, exec_id: str) -> None:
        """Execute the ANALYZE phase for a bug."""
        # Get the current step
        step = path.get_current_step()
        if not step or step.get("phase") != FixPhase.ANALYZE.value:
            logger.warning(f"Invalid step for ANALYZE phase: {step}")
            return
        
        # Execute the analyze step
        try:
            # In a real implementation, this would use more sophisticated analysis
            logger.info(f"Executing ANALYZE phase for bug {bug.bug_id}")
            
            # Use Observer Agent to analyze the bug
            observe_command(f"analyze {bug.bug_id}")
            
            # Update bug and path
            bug.update_phase(FixPhase.PLAN, {
                "execution_id": exec_id,
                "result": "Analysis completed successfully"
            })
            
            path.advance_step()
            path.updated_at = time.time()
        except Exception as e:
            logger.error(f"Error in ANALYZE phase for bug {bug.bug_id}: {e}")
            bug.update_phase(FixPhase.FAILED, {
                "execution_id": exec_id,
                "error": str(e)
            })
    
    def _execute_plan_phase(self, bug: BugState, path: ExecutionPath, exec_id: str) -> None:
        """Execute the PLAN phase for a bug."""
        # Get the current step
        step = path.get_current_step()
        if not step or step.get("phase") != FixPhase.PLAN.value:
            logger.warning(f"Invalid step for PLAN phase: {step}")
            return
        
        # Execute the plan step
        try:
            logger.info(f"Executing PLAN phase for bug {bug.bug_id}")
            
            # Use Planner Agent to generate and select solution paths
            plan_generate_command(bug.bug_id)
            plan_select_command(bug.bug_id)
            
            # Update bug and path
            bug.update_phase(FixPhase.IMPLEMENT, {
                "execution_id": exec_id,
                "result": "Planning completed successfully"
            })
            
            path.advance_step()
            path.updated_at = time.time()
        except Exception as e:
            logger.error(f"Error in PLAN phase for bug {bug.bug_id}: {e}")
            bug.update_phase(FixPhase.FAILED, {
                "execution_id": exec_id,
                "error": str(e)
            })
    
    def _execute_implement_phase(self, bug: BugState, path: ExecutionPath, exec_id: str) -> None:
        """Execute the IMPLEMENT phase for a bug."""
        # Get the current step
        step = path.get_current_step()
        if not step or step.get("phase") != FixPhase.IMPLEMENT.value:
            logger.warning(f"Invalid step for IMPLEMENT phase: {step}")
            return
        
        # Execute the implement step
        try:
            logger.info(f"Executing IMPLEMENT phase for bug {bug.bug_id}")
            
            # Use Analyst Agent to generate a patch
            analyze_command(f"patch {bug.bug_id}")
            
            # Update bug and path
            bug.update_phase(FixPhase.VERIFY, {
                "execution_id": exec_id,
                "result": "Implementation completed successfully"
            })
            
            path.advance_step()
            path.updated_at = time.time()
        except Exception as e:
            logger.error(f"Error in IMPLEMENT phase for bug {bug.bug_id}: {e}")
            bug.update_phase(FixPhase.FAILED, {
                "execution_id": exec_id,
                "error": str(e)
            })
    
    def _execute_verify_phase(self, bug: BugState, path: ExecutionPath, exec_id: str) -> None:
        """Execute the VERIFY phase for a bug."""
        # Get the current step
        step = path.get_current_step()
        if not step or step.get("phase") != FixPhase.VERIFY.value:
            logger.warning(f"Invalid step for VERIFY phase: {step}")
            return
        
        # Execute the verify step
        try:
            logger.info(f"Executing VERIFY phase for bug {bug.bug_id}")
            
            # Use Verifier Agent to test the fix
            verify_command(f"test {bug.bug_id}")
            
            # Update bug and path
            bug.update_phase(FixPhase.LEARN, {
                "execution_id": exec_id,
                "result": "Verification completed successfully"
            })
            
            path.advance_step()
            path.updated_at = time.time()
        except Exception as e:
            logger.error(f"Error in VERIFY phase for bug {bug.bug_id}: {e}")
            bug.update_phase(FixPhase.FAILED, {
                "execution_id": exec_id,
                "error": str(e)
            })
    
    def _execute_learn_phase(self, bug: BugState, path: ExecutionPath, exec_id: str) -> None:
        """Execute the LEARN phase for a bug."""
        # Get the current step
        step = path.get_current_step()
        if not step or step.get("phase") != FixPhase.LEARN.value:
            logger.warning(f"Invalid step for LEARN phase: {step}")
            return
        
        # Execute the learn step
        try:
            logger.info(f"Executing LEARN phase for bug {bug.bug_id}")
            
            # In a real implementation, this would update neural weights
            # and store successful patterns for future reference
            
            # Record the successful fix
            path.mark_complete(True, {
                "execution_id": exec_id,
                "message": "Fix completed successfully"
            })
            
            # Update bug
            bug.update_phase(FixPhase.COMPLETE, {
                "execution_id": exec_id,
                "result": "Bug fixed successfully"
            })
            
            logger.info(f"Bug {bug.bug_id} fixed successfully")
        except Exception as e:
            logger.error(f"Error in LEARN phase for bug {bug.bug_id}: {e}")
            bug.update_phase(FixPhase.FAILED, {
                "execution_id": exec_id,
                "error": str(e)
            })
    
    def register_bug(
        self,
        bug_id: str,
        title: str,
        description: Optional[str] = None,
        severity: str = "medium",
        metadata: Dict[str, Any] = None
    ) -> BugState:
        """
        Register a bug with the engine.
        
        Args:
            bug_id: ID of the bug
            title: Bug title
            description: Bug description
            severity: Bug severity
            metadata: Additional metadata
            
        Returns:
            BugState: The registered bug state
        """
        with self._bug_lock:
            # Check if bug already exists
            if bug_id in self.bugs:
                logger.warning(f"Bug {bug_id} already registered")
                return self.bugs[bug_id]
            
            # Create new bug state
            bug = BugState(
                bug_id=bug_id,
                title=title,
                description=description,
                severity=severity,
                metadata=metadata
            )
            
            # Store in bugs dictionary
            self.bugs[bug_id] = bug
            
            # Save state
            self.save_state()
            
            logger.info(f"Bug {bug_id} registered: {title}")
            return bug
    
    def start_bug_fix(self, bug_id: str) -> Optional[str]:
        """
        Start fixing a bug.
        
        Args:
            bug_id: ID of the bug to fix
            
        Returns:
            str: Execution ID if successful, None otherwise
        """
        with self._bug_lock:
            # Check if bug exists
            if bug_id not in self.bugs:
                logger.error(f"Bug {bug_id} not found")
                return None
            
            bug = self.bugs[bug_id]
            
            # Check if bug is already being fixed
            if bug.phase not in (FixPhase.INITIALIZE, FixPhase.FAILED):
                logger.warning(f"Bug {bug_id} is already being fixed (phase: {bug.phase.value})")
                return None
            
            # Create a default execution path
            path_id = f"{bug_id}-path-{str(uuid.uuid4())[:8]}"
            
            with self._path_lock:
                path = ExecutionPath(
                    path_id=path_id,
                    bug_id=bug_id,
                    steps=[
                        {
                            "phase": FixPhase.ANALYZE.value,
                            "description": f"Analyze bug {bug_id}",
                            "params": {"bug_id": bug_id}
                        },
                        {
                            "phase": FixPhase.PLAN.value,
                            "description": f"Plan fix for bug {bug_id}",
                            "params": {"bug_id": bug_id}
                        },
                        {
                            "phase": FixPhase.IMPLEMENT.value,
                            "description": f"Implement fix for bug {bug_id}",
                            "params": {"bug_id": bug_id}
                        },
                        {
                            "phase": FixPhase.VERIFY.value,
                            "description": f"Verify fix for bug {bug_id}",
                            "params": {"bug_id": bug_id}
                        },
                        {
                            "phase": FixPhase.LEARN.value,
                            "description": f"Learn from fix for bug {bug_id}",
                            "params": {"bug_id": bug_id}
                        }
                    ]
                )
                
                self.paths[path_id] = path
            
            # Update bug state
            bug.path_id = path_id
            bug.active_path = path_id
            bug.update_phase(FixPhase.ANALYZE)
            
            # Create execution ID
            execution_id = f"exec-{str(uuid.uuid4())[:8]}"
            
            with self._execution_lock:
                self.active_executions[execution_id] = (bug_id, path_id)
            
            # Save state
            self.save_state()
            
            logger.info(f"Started fixing bug {bug_id} with execution ID {execution_id}")
            return execution_id
    
    def get_bug_status(self, bug_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a bug.
        
        Args:
            bug_id: ID of the bug
            
        Returns:
            Dict[str, Any]: Bug status if found, None otherwise
        """
        with self._bug_lock:
            if bug_id not in self.bugs:
                return None
            
            return self.bugs[bug_id].to_dict()
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an execution.
        
        Args:
            execution_id: ID of the execution
            
        Returns:
            Dict[str, Any]: Execution status if found, None otherwise
        """
        with self._execution_lock:
            if execution_id not in self.active_executions:
                return None
            
            bug_id, path_id = self.active_executions[execution_id]
            
            with self._bug_lock, self._path_lock:
                if bug_id not in self.bugs or path_id not in self.paths:
                    return None
                
                return {
                    "execution_id": execution_id,
                    "bug": self.bugs[bug_id].to_dict(),
                    "path": self.paths[path_id].to_dict()
                }
    
    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active execution.
        
        Args:
            execution_id: ID of the execution to cancel
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self._execution_lock:
            if execution_id not in self.active_executions:
                logger.warning(f"Execution {execution_id} not found")
                return False
            
            bug_id, path_id = self.active_executions[execution_id]
            
            with self._bug_lock, self._path_lock:
                if bug_id in self.bugs:
                    self.bugs[bug_id].update_phase(FixPhase.ABANDONED, {
                        "execution_id": execution_id,
                        "reason": "Execution cancelled"
                    })
                
                if path_id in self.paths:
                    self.paths[path_id].mark_complete(False, {
                        "execution_id": execution_id,
                        "reason": "Execution cancelled"
                    })
            
            # Remove from active executions
            del self.active_executions[execution_id]
            
            # Save state
            self.save_state()
            
            logger.info(f"Execution {execution_id} cancelled")
            return True
    
    def load_state(self) -> None:
        """Load engine state from disk."""
        try:
            if not self.state_file.exists():
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Load bugs
            for bug_data in state.get("bugs", []):
                bug_id = bug_data.get("bug_id")
                if not bug_id:
                    continue
                
                bug = BugState(
                    bug_id=bug_id,
                    title=bug_data.get("title", ""),
                    description=bug_data.get("description"),
                    severity=bug_data.get("severity", "medium"),
                    metadata=bug_data.get("metadata", {})
                )
                
                bug.phase = FixPhase(bug_data.get("phase", FixPhase.INITIALIZE.value))
                bug.status = bug_data.get("status", "pending")
                bug.created_at = bug_data.get("created_at", time.time())
                bug.updated_at = bug_data.get("updated_at", time.time())
                bug.path_id = bug_data.get("path_id")
                bug.active_path = bug_data.get("active_path")
                bug.phase_history = bug_data.get("phase_history", [])
                bug.fix_attempts = bug_data.get("fix_attempts", [])
                bug.results = bug_data.get("results", {})
                
                self.bugs[bug_id] = bug
            
            # Load paths
            for path_data in state.get("paths", []):
                path_id = path_data.get("path_id")
                if not path_id:
                    continue
                
                bug_id = path_data.get("bug_id")
                if not bug_id:
                    continue
                
                path = ExecutionPath(
                    path_id=path_id,
                    bug_id=bug_id,
                    steps=path_data.get("steps", []),
                    metadata=path_data.get("metadata", {})
                )
                
                path.current_step = path_data.get("current_step", 0)
                path.status = path_data.get("status", "pending")
                path.created_at = path_data.get("created_at", time.time())
                path.updated_at = path_data.get("updated_at", time.time())
                path.results = path_data.get("results", {})
                
                self.paths[path_id] = path
            
            # Load active executions
            for exec_id, (bug_id, path_id) in state.get("active_executions", {}).items():
                self.active_executions[exec_id] = (bug_id, path_id)
            
            logger.info(f"Loaded state: {len(self.bugs)} bugs, {len(self.paths)} paths, {len(self.active_executions)} executions")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    def save_state(self) -> None:
        """Save engine state to disk."""
        try:
            state = {
                "bugs": [bug.to_dict() for bug in self.bugs.values()],
                "paths": [path.to_dict() for path in self.paths.values()],
                "active_executions": self.active_executions
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.debug("State saved")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "bugs": {
                "total": len(self.bugs),
                "by_phase": {phase.value: len([b for b in self.bugs.values() if b.phase == phase]) for phase in FixPhase},
                "by_status": {status: len([b for b in self.bugs.values() if b.status == status]) for status in set(b.status for b in self.bugs.values())}
            },
            "paths": {
                "total": len(self.paths),
                "by_status": {status: len([p for p in self.paths.values() if p.status == status]) for status in set(p.status for p in self.paths.values())}
            },
            "executions": {
                "active": len(self.active_executions)
            }
        }

# Singleton instance for the Triangulation Engine
_engine = None

def get_engine(config: Dict[str, Any] = None) -> TriangulationEngine:
    """
    Get the singleton instance of the Triangulation Engine.
    
    Args:
        config: Engine configuration (used only if engine is not initialized)
        
    Returns:
        TriangulationEngine: The engine instance
    """
    global _engine
    
    if _engine is None:
        _engine = TriangulationEngine(config)
    
    return _engine

# API Functions for command handlers

def register_bug(
    bug_id: str,
    title: str,
    description: Optional[str] = None,
    severity: str = "medium",
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Register a bug with the engine.
    
    Args:
        bug_id: ID of the bug
        title: Bug title
        description: Bug description
        severity: Bug severity
        metadata: Additional metadata
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        engine = get_engine()
        bug = engine.register_bug(
            bug_id=bug_id,
            title=title,
            description=description,
            severity=severity,
            metadata=metadata
        )
        
        return {
            "success": True,
            "bug_id": bug_id,
            "bug": bug.to_dict()
        }
    except Exception as e:
        logger.error(f"Error registering bug: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def start_bug_fix(bug_id: str) -> Dict[str, Any]:
    """
    Start fixing a bug.
    
    Args:
        bug_id: ID of the bug to fix
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        engine = get_engine()
        execution_id = engine.start_bug_fix(bug_id)
        
        if execution_id is None:
            return {
                "success": False,
                "error": f"Failed to start fixing bug {bug_id}"
            }
        
        return {
            "success": True,
            "bug_id": bug_id,
            "execution_id": execution_id
        }
    except Exception as e:
        logger.error(f"Error starting bug fix: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_bug_status(bug_id: str) -> Dict[str, Any]:
    """
    Get the status of a bug.
    
    Args:
        bug_id: ID of the bug
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        engine = get_engine()
        status = engine.get_bug_status(bug_id)
        
        if status is None:
            return {
                "success": False,
                "error": f"Bug {bug_id} not found"
            }
        
        return {
            "success": True,
            "bug_id": bug_id,
            "status": status
        }
    except Exception as e:
        logger.error(f"Error getting bug status: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_execution_status(execution_id: str) -> Dict[str, Any]:
    """
    Get the status of an execution.
    
    Args:
        execution_id: ID of the execution
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        engine = get_engine()
        status = engine.get_execution_status(execution_id)
        
        if status is None:
            return {
                "success": False,
                "error": f"Execution {execution_id} not found"
            }
        
        return {
            "success": True,
            "execution_id": execution_id,
            "status": status
        }
    except Exception as e:
        logger.error(f"Error getting execution status: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def cancel_execution(execution_id: str) -> Dict[str, Any]:
    """
    Cancel an active execution.
    
    Args:
        execution_id: ID of the execution to cancel
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        engine = get_engine()
        success = engine.cancel_execution(execution_id)
        
        if not success:
            return {
                "success": False,
                "error": f"Failed to cancel execution {execution_id}"
            }
        
        return {
            "success": True,
            "execution_id": execution_id,
            "message": "Execution cancelled successfully"
        }
    except Exception as e:
        logger.error(f"Error cancelling execution: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_engine_stats() -> Dict[str, Any]:
    """
    Get engine statistics.
    
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        engine = get_engine()
        stats = engine.get_stats()
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting engine stats: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Triangulation Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Register bug command
    register_parser = subparsers.add_parser("register", help="Register a bug")
    register_parser.add_argument("bug_id", help="Bug ID")
    register_parser.add_argument("title", help="Bug title")
    register_parser.add_argument("--description", help="Bug description")
    register_parser.add_argument("--severity", default="medium", help="Bug severity")
    
    # Start fix command
    start_parser = subparsers.add_parser("start", help="Start fixing a bug")
    start_parser.add_argument("bug_id", help="Bug ID")
    
    # Get bug status command
    status_parser = subparsers.add_parser("status", help="Get bug status")
    status_parser.add_argument("bug_id", help="Bug ID")
    
    # Get execution status command
    exec_parser = subparsers.add_parser("execution", help="Get execution status")
    exec_parser.add_argument("execution_id", help="Execution ID")
    
    # Cancel execution command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel an execution")
    cancel_parser.add_argument("execution_id", help="Execution ID")
    
    # Get stats command
    stats_parser = subparsers.add_parser("stats", help="Get engine statistics")
    
    args = parser.parse_args()
    
    if args.command == "register":
        result = register_bug(
            args.bug_id,
            args.title,
            args.description,
            args.severity
        )
    elif args.command == "start":
        result = start_bug_fix(args.bug_id)
    elif args.command == "status":
        result = get_bug_status(args.bug_id)
    elif args.command == "execution":
        result = get_execution_status(args.execution_id)
    elif args.command == "cancel":
        result = cancel_execution(args.execution_id)
    elif args.command == "stats":
        result = get_engine_stats()
    else:
        parser.print_help()
        sys.exit(1)
    
    # Print result as JSON
    print(json.dumps(result, indent=2, default=str))
