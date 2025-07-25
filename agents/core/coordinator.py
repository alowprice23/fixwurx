"""
agents/agent_coordinator.py
───────────────────────────
Orchestration layer that drives agent interactions using the Planner as the root coordinator.
Implements the enhanced agent handoff protocol and path-based execution flow.

Key properties
──────────────
▪ Pure-async (`await`-friendly) – fits both Scheduler and ParallelExecutor.  
▪ Planner-driven – executes solution paths generated by the planner agent.
▪ Path-based – follows dynamic execution paths rather than a fixed sequence.
▪ Fallback-capable – implements recovery strategies when primary approaches fail.
▪ Metrics-tracked – reports performance metrics back to the planner.

External dependencies
─────────────────────
Only Python std-lib + specialized agents, state_machine, and planner_agent.
No network access happens here; AutoGen handles that inside each agent.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set

from agents.specialized.specialized_agents import (
    ObserverAgent,
    AnalystAgent,
    VerifierAgent
)
from planner_agent import PlannerAgent
from state_machine import Phase
from triangulation_engine import TriangulationEngine
from data_structures import BugState, PlannerPath

# Setup logging
logger = logging.getLogger("agent_coordinator")

# ───────────────────────────────────────────────────────────────────────────────
# Constants for agent handoff protocol
# ───────────────────────────────────────────────────────────────────────────────
class HandoffStatus:
    """Status codes for agent handoffs."""
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    FALLBACK = "fallback"
    ERROR = "error"

# ───────────────────────────────────────────────────────────────────────────────
# Dataclass storing the in-flight artefacts for one bug
# ───────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class _Artefacts:
    # Traditional artifacts
    observer_report: Optional[str] = None  # JSON string
    patch_bundle: Optional[str] = None     # unified diff
    first_fail_seen: bool = False
    completed: bool = False
    
    # Planner-specific artifacts
    solution_paths: List[Dict[str, Any]] = field(default_factory=list)  # List of solution paths
    current_path_id: Optional[str] = None  # ID of the currently active path
    fallbacks_used: int = 0  # Count of fallbacks used for the current bug
    
    # Enhanced execution tracking
    execution_history: List[Dict[str, Any]] = field(default_factory=list)  # History of actions and results
    current_action_index: int = 0  # Index of the current action in the path
    
    # Metrics
    start_time: float = field(default_factory=time.time)  # Start time of bug resolution
    action_times: Dict[str, float] = field(default_factory=dict)  # Time spent on each action
    handoff_counts: Dict[str, int] = field(default_factory=dict)  # Count of handoffs between agents
    error_counts: Dict[str, int] = field(default_factory=dict)  # Count of errors by type
    
    # Additional context for agents
    context: Dict[str, Any] = field(default_factory=dict)  # Shared context for agents
    
    def record_action_start(self, action_type: str) -> None:
        """Record the start time of an action."""
        self.action_times[f"{action_type}_start"] = time.time()
    
    def record_action_end(self, action_type: str) -> None:
        """Record the end time of an action and calculate duration."""
        start_key = f"{action_type}_start"
        if start_key in self.action_times:
            start_time = self.action_times[start_key]
            duration = time.time() - start_time
            self.action_times[f"{action_type}_duration"] = duration
    
    def record_handoff(self, from_agent: str, to_agent: str, status: str) -> None:
        """Record a handoff between agents."""
        handoff_key = f"{from_agent}_to_{to_agent}"
        self.handoff_counts[handoff_key] = self.handoff_counts.get(handoff_key, 0) + 1
        
        # Record in execution history
        self.execution_history.append({
            "type": "handoff",
            "from_agent": from_agent,
            "to_agent": to_agent,
            "status": status,
            "timestamp": time.time()
        })
    
    def record_error(self, error_type: str, details: str) -> None:
        """Record an error."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Record in execution history
        self.execution_history.append({
            "type": "error",
            "error_type": error_type,
            "details": details,
            "timestamp": time.time()
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for the current bug resolution process."""
        return {
            "total_duration": time.time() - self.start_time,
            "action_times": {k: v for k, v in self.action_times.items() if k.endswith("_duration")},
            "handoff_counts": self.handoff_counts,
            "error_counts": self.error_counts,
            "fallbacks_used": self.fallbacks_used,
            "solution_paths_count": len(self.solution_paths),
            "actions_executed": self.current_action_index
        }
    
    def get_current_action(self) -> Optional[Dict[str, Any]]:
        """Get the current action from the active solution path."""
        if not self.solution_paths or not self.current_path_id:
            return None
        
        # Find the current path
        current_path = next(
            (p for p in self.solution_paths if p.get("path_id") == self.current_path_id),
            None
        )
        
        if not current_path or "actions" not in current_path:
            return None
        
        actions = current_path.get("actions", [])
        if self.current_action_index >= len(actions):
            return None
        
        return actions[self.current_action_index]
    
    def advance_to_next_action(self) -> Optional[Dict[str, Any]]:
        """Advance to the next action in the current path."""
        self.current_action_index += 1
        return self.get_current_action()
    
    def add_to_context(self, key: str, value: Any) -> None:
        """Add a value to the shared context."""
        self.context[key] = value
    
    def get_from_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared context."""
        return self.context.get(key, default)


# ───────────────────────────────────────────────────────────────────────────────
# AgentCoordinator
# ───────────────────────────────────────────────────────────────────────────────
class AgentCoordinator:
    """
    Advanced coordinator that orchestrates agent interactions using the Planner.
    
    Implements the enhanced agent handoff protocol and path-based execution.
    One coordinator is instantiated per live bug.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize the AgentCoordinator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize agents
        self._observer = ObserverAgent()
        self._analyst = AnalystAgent()
        self._verifier = VerifierAgent()
        self._planner = PlannerAgent()  # Root agent that coordinates others

        # Track artifacts
        self._art = _Artefacts()
        
        # Track state
        self._last_phase: Optional[Phase] = None
        self._family_tree_updated = False
        self._agents_initialized = False
        
        # Configuration options
        self.use_enhanced_handoff = self.config.get("use_enhanced_handoff", True)
        self.collect_metrics = self.config.get("collect_metrics", True)
        self.max_retry_attempts = self.config.get("max_retry_attempts", 3)
        self.retry_count = 0
        
        # Agent registry
        self._agent_registry = {
            "observer": self._observer,
            "analyst": self._analyst,
            "verifier": self._verifier,
            "planner": self._planner
        }
        
        logger.info("AgentCoordinator initialized with enhanced handoff protocol")

    # ==========================================================================
    # Main coordination loop
    # ==========================================================================
    async def coordinate_tick(self, engine: TriangulationEngine) -> None:
        """
        Main coordination entry point called by the scheduler.
        
        Args:
            engine: TriangulationEngine instance containing exactly one bug
        """
        bug = engine.bugs[0]  # by contract there's exactly one

        # Edge detection: do stuff only if phase changed since last call
        phase_changed = bug.phase is not self._last_phase
        self._last_phase = bug.phase
        if not phase_changed:
            return
        
        # Initialize agents if not already done
        if not self._agents_initialized:
            await self._initialize_agents(bug)
        
        # Setup family tree if not already done
        if not self._family_tree_updated:
            await self._setup_family_tree(bug)
        
        if self.use_enhanced_handoff:
            # Enhanced path-based execution
            await self._execute_path_based(bug)
        else:
            # Traditional phase-based execution
            await self._execute_phase_based(bug)
            
        # Report metrics if collection is enabled
        if self.collect_metrics and (bug.phase == Phase.DONE or bug.phase == Phase.ESCALATE):
            await self._report_metrics(bug)

    # ==========================================================================
    # Enhanced path-based execution
    # ==========================================================================
    async def _execute_path_based(self, bug) -> None:
        """
        Execute bug resolution using the planner-generated path.
        
        This is the enhanced execution flow that follows the dynamic path
        generated by the planner instead of a fixed phase sequence.
        """
        # Generate solution paths if needed
        if not self._art.solution_paths and bug.phase == Phase.REPRO:
            await self._run_planner_for_paths(bug)
        
        # Select a path if none is active
        if self._art.solution_paths and not self._art.current_path_id:
            await self._select_solution_path(bug)
        
        # Get the current action
        current_action = self._art.get_current_action()
        
        # If we have a current action, execute it
        if current_action:
            action_type = current_action.get("type")
            agent_type = current_action.get("agent")
            
            # Record start time for metrics
            if self.collect_metrics:
                self._art.record_action_start(action_type)
            
            # Execute the action based on type
            if action_type == "analyze" and agent_type == "observer":
                success = await self._execute_analysis_action(bug, current_action)
            elif action_type == "patch" and agent_type == "analyst":
                success = await self._execute_patch_action(bug, current_action)
            elif action_type == "verify" and agent_type == "verifier":
                success = await self._execute_verify_action(bug, current_action)
            else:
                # Unknown action or agent type
                logger.warning(f"Unknown action: {action_type} or agent: {agent_type}")
                self._art.record_error("unknown_action", f"Unknown action: {action_type} or agent: {agent_type}")
                success = False
            
            # Record end time for metrics
            if self.collect_metrics:
                self._art.record_action_end(action_type)
            
            # Advance to the next action if successful
            if success:
                self._art.advance_to_next_action()
                self.retry_count = 0  # Reset retry count
            else:
                # Handle failure
                if self.retry_count < self.max_retry_attempts:
                    # Retry the action
                    self.retry_count += 1
                    logger.info(f"Retrying action {action_type} (attempt {self.retry_count}/{self.max_retry_attempts})")
                else:
                    # Try fallback if available
                    self.retry_count = 0
                    await self._try_fallback_path(bug)
        
        # If we've run out of actions in the current path, check if we need to try a fallback
        elif self._art.current_path_id and not self._art.completed:
            # All actions completed but not marked as complete, try fallback
            await self._try_fallback_path(bug)
        
        # Respect the phase transitions from the state machine
        # This ensures compatibility with the existing engine
        self._sync_with_phase(bug)

    # ==========================================================================
    # Traditional phase-based execution
    # ==========================================================================
    async def _execute_phase_based(self, bug) -> None:
        """
        Execute bug resolution using the traditional phase-based approach.
        
        This is the original execution flow that follows the fixed
        REPRO -> PATCH -> VERIFY sequence based on the bug's phase.
        """
        # First, let the planner generate solution paths if needed
        if not self._art.solution_paths and bug.phase == Phase.REPRO:
            await self._run_planner_for_paths(bug)
            
        # Select a solution path if one isn't active
        if self._art.solution_paths and not self._art.current_path_id:
            await self._select_solution_path(bug)

        # REPRO phase
        if bug.phase == Phase.REPRO and not self._art.observer_report:
            if self.collect_metrics:
                self._art.record_action_start("analyze")
                
            await self._run_observer(bug)
            
            if self.collect_metrics:
                self._art.record_action_end("analyze")

        # PATCH phase
        elif bug.phase == Phase.PATCH and self._art.observer_report:
            # Analyst may be called twice: initial patch, then refined patch
            if self.collect_metrics:
                self._art.record_action_start("patch")
                
            await self._run_analyst(bug)
            
            if self.collect_metrics:
                self._art.record_action_end("patch")

        # VERIFY phase
        elif bug.phase == Phase.VERIFY and self._art.patch_bundle:
            if self.collect_metrics:
                self._art.record_action_start("verify")
                
            await self._run_verifier(bug)
            
            if self.collect_metrics:
                self._art.record_action_end("verify")
            
        # If verification failed and there are fallbacks, try a different approach
        if bug.phase == Phase.VERIFY and self._art.first_fail_seen and not self._art.completed:
            await self._try_fallback_path(bug)

    # ==========================================================================
    # Sync with state machine phase
    # ==========================================================================
    def _sync_with_phase(self, bug) -> None:
        """
        Synchronize the coordinator state with the bug's phase.
        
        This ensures compatibility with the state machine transitions.
        """
        # If we've completed the observer report, make sure we're in PATCH phase
        if self._art.observer_report and bug.phase == Phase.REPRO:
            bug.phase = Phase.PATCH
            
        # If we've completed the patch bundle, make sure we're in VERIFY phase
        if self._art.patch_bundle and bug.phase == Phase.PATCH:
            bug.phase = Phase.VERIFY
            
        # If we've completed verification, move to DONE phase
        if self._art.completed and bug.phase == Phase.VERIFY:
            bug.phase = Phase.DONE

    # ==========================================================================
    # Agent initialization and setup
    # ==========================================================================
    async def _initialize_agents(self, bug) -> None:
        """Initialize all agents with necessary context."""
        # Initialize the bug state in each agent
        for agent_name, agent in self._agent_registry.items():
            # Set agent-specific context
            prompt = textwrap.dedent(
                f"""
                INITIALIZATION
                -------------
                You are working on bug ID: {bug.id}
                Your role: {agent_name.capitalize()}
                
                When working with this bug, remember your specific responsibilities 
                and coordinate with other agents as directed by the planner.
                
                Respond with "INITIALIZED" if you understand your role.
                """
            )
            
            # Ask the agent to initialize
            try:
                response = await agent.ask(prompt)
                if "INITIALIZED" in response:
                    logger.info(f"Agent {agent_name} initialized successfully")
                else:
                    logger.warning(f"Agent {agent_name} initialization response: {response}")
            except Exception as e:
                logger.error(f"Error initializing agent {agent_name}: {e}")
                self._art.record_error("initialization_error", f"Agent {agent_name}: {str(e)}")
        
        self._agents_initialized = True

    # ==========================================================================
    # Setup family tree
    # ==========================================================================
    async def _setup_family_tree(self, bug) -> None:
        """Initialize the agent family tree."""
        prompt = textwrap.dedent(
            f"""
            Initialize the family tree for bug {bug.id}.
            
            Create a family tree with:
            - Planner as the root agent
            - Observer, Analyst, and Verifier as children
            
            Please output the agent registration confirmation in JSON format.
            """
        )
        reply = await self._planner.ask(prompt)
        try:
            response = json.loads(reply)
            if response.get("status") == "success":
                self._family_tree_updated = True
        except json.JSONDecodeError:
            # Even if JSON parsing fails, we'll mark it as updated to avoid retrying
            self._family_tree_updated = True

    # ==========================================================================
    # Enhanced agent handoff protocol methods
    # ==========================================================================
    async def _agent_handoff(self, 
                            from_agent: str, 
                            to_agent: str, 
                            handoff_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Implement the enhanced agent handoff protocol.
        
        Args:
            from_agent: Source agent type (e.g., "observer")
            to_agent: Target agent type (e.g., "analyst")
            handoff_data: Data to be passed to the target agent
            
        Returns:
            Tuple of (status, result) where status is one of HandoffStatus values
        """
        # Record the handoff for metrics
        if self.collect_metrics:
            self._art.record_handoff(from_agent, to_agent, "initiated")
        
        # Get the target agent
        target_agent = self._agent_registry.get(to_agent)
        if not target_agent:
            error_msg = f"Target agent {to_agent} not found in registry"
            logger.error(error_msg)
            self._art.record_error("handoff_error", error_msg)
            return HandoffStatus.ERROR, {"error": error_msg}
        
        # Construct the handoff prompt
        handoff_context = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "bug_id": handoff_data.get("bug_id", "unknown"),
            "timestamp": time.time()
        }
        
        # Add to the shared context
        for key, value in handoff_data.items():
            self._art.add_to_context(key, value)
        
        # Get any specific context needed for this handoff
        additional_context = self._get_handoff_context(from_agent, to_agent)
        
        # Construct the prompt
        prompt = textwrap.dedent(
            f"""
            HANDOFF: {from_agent.upper()} → {to_agent.upper()}
            ────────────────────────────────────
            
            CONTEXT
            • Bug ID: {handoff_data.get('bug_id', 'unknown')}
            • Handoff type: {handoff_data.get('handoff_type', 'standard')}
            • Action: {handoff_data.get('action', 'unknown')}
            
            {additional_context}
            
            DATA
            ────
            {json.dumps(handoff_data.get('data', {}), indent=2)}
            
            YOUR TASK
            ─────────
            {handoff_data.get('instructions', 'Process the handoff data and respond accordingly.')}
            
            Include a "status" field in your response with one of: SUCCESS, FAILURE, RETRY, FALLBACK.
            """
        )
        
        # Send the handoff to the target agent
        try:
            response = await target_agent.ask(prompt)
            
            # Parse the response
            status = HandoffStatus.SUCCESS  # Default status
            result = {"raw_response": response}
            
            # Try to extract a JSON status if available
            try:
                # Check if the response contains a JSON object
                if "{" in response and "}" in response:
                    # Extract the JSON part
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    json_str = response[json_start:json_end]
                    
                    # Parse the JSON
                    parsed = json.loads(json_str)
                    if "status" in parsed:
                        status_str = parsed["status"].upper()
                        if hasattr(HandoffStatus, status_str):
                            status = getattr(HandoffStatus, status_str)
                        result.update(parsed)
                    else:
                        # If there's a parseable JSON but no status, include the parsed data
                        result.update(parsed)
            except json.JSONDecodeError:
                # If we can't parse JSON, just keep the raw response
                pass
            
            # Record the handoff result for metrics
            if self.collect_metrics:
                self._art.record_handoff(from_agent, to_agent, status)
            
            return status, result
            
        except Exception as e:
            error_msg = f"Error during handoff from {from_agent} to {to_agent}: {str(e)}"
            logger.error(error_msg)
            self._art.record_error("handoff_error", error_msg)
            
            # Record the failed handoff for metrics
            if self.collect_metrics:
                self._art.record_handoff(from_agent, to_agent, HandoffStatus.ERROR)
                
            return HandoffStatus.ERROR, {"error": error_msg}
    
    def _get_handoff_context(self, from_agent: str, to_agent: str) -> str:
        """
        Get additional context for specific handoff types.
        
        Args:
            from_agent: Source agent type
            to_agent: Target agent type
            
        Returns:
            String with additional context for the handoff
        """
        # Observer to Analyst handoff
        if from_agent == "observer" and to_agent == "analyst":
            return textwrap.dedent(
                """
                ANALYST GUIDELINES
                • Use the observer's report to understand the bug
                • Generate a patch that fixes the issue
                • Keep changes minimal and focused
                """
            )
        
        # Analyst to Verifier handoff
        elif from_agent == "analyst" and to_agent == "verifier":
            return textwrap.dedent(
                """
                VERIFIER GUIDELINES
                • Apply the patch in a clean environment
                • Run tests to validate the fix
                • Report any issues or edge cases
                """
            )
        
        # Verifier to Planner handoff
        elif from_agent == "verifier" and to_agent == "planner":
            return textwrap.dedent(
                """
                PLANNER GUIDELINES
                • Analyze the verification results
                • Decide if the solution is complete or needs refinement
                • Select appropriate next steps or fallback strategies
                """
            )
        
        # Default context
        return ""

    # ==========================================================================
    # Enhanced action execution methods
    # ==========================================================================
    async def _execute_analysis_action(self, bug, action: Dict[str, Any]) -> bool:
        """
        Execute an analysis action using the Observer agent.
        
        Args:
            bug: The bug being processed
            action: The action definition
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing analysis action for bug {bug.id}")
        
        # Get action-specific parameters
        depth = action.get("parameters", {}).get("depth", "full")
        
        # Prepare handoff data
        handoff_data = {
            "bug_id": bug.id,
            "handoff_type": "analysis",
            "action": action.get("description", "Analyze bug"),
            "data": {
                "bug_id": bug.id,
                "phase": bug.phase.name,
                "timer": bug.timer,
                "depth": depth
            },
            "instructions": textwrap.dedent(
                f"""
                Please analyze bug {bug.id} with {depth} depth.
                
                Return a JSON object with the following fields:
                - summary: Brief description of the bug
                - repro_steps: Steps to reproduce the bug
                - evidence: Log excerpts or errors (≤120 chars each)
                - root_cause: Your assessment of the root cause
                - complexity: Estimated complexity (low, medium, high)
                
                Include "status": "SUCCESS" if you were able to analyze the bug,
                or "status": "FAILURE" with an explanation if you couldn't.
                """
            )
        }
        
        # Execute the handoff
        status, result = await self._agent_handoff("planner", "observer", handoff_data)
        
        # Process the result
        if status == HandoffStatus.SUCCESS:
            # Extract the observer report
            try:
                if "raw_response" in result:
                    # Try to extract JSON from the raw response
                    raw = result["raw_response"]
                    if "{" in raw and "}" in raw:
                        json_start = raw.find("{")
                        json_end = raw.rfind("}") + 1
                        json_str = raw[json_start:json_end]
                        observer_report = json.loads(json_str)
                    else:
                        observer_report = result
                else:
                    observer_report = result
                
                # Store the observer report
                self._art.observer_report = json.dumps(observer_report)
                
                # Update the shared context
                self._art.add_to_context("observer_report", observer_report)
                self._art.add_to_context("analysis_completed", True)
                
                # Notify the planner of the analysis result
                await self._notify_planner_of_progress(bug, "analysis", True, observer_report)
                
                return True
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing observer report: {e}")
                self._art.record_error("json_error", f"Observer report parsing: {str(e)}")
                return False
        else:
            # Analysis failed
            logger.warning(f"Analysis action failed with status {status}")
            error_details = result.get("error", "Unknown error")
            self._art.record_error("analysis_error", error_details)
            
            # Notify the planner of the failure
            await self._notify_planner_of_progress(bug, "analysis", False, {"error": error_details})
            
            return False

    async def _execute_patch_action(self, bug, action: Dict[str, Any]) -> bool:
        """
        Execute a patch action using the Analyst agent.
        
        Args:
            bug: The bug being processed
            action: The action definition
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing patch action for bug {bug.id}")
        
        # Ensure we have an observer report
        if not self._art.observer_report:
            logger.error("Cannot execute patch action without observer report")
            self._art.record_error("sequence_error", "No observer report available for patch action")
            return False
        
        # Get action-specific parameters
        conservative = action.get("parameters", {}).get("conservative", False)
        
        # Prepare handoff data
        handoff_data = {
            "bug_id": bug.id,
            "handoff_type": "patch_generation",
            "action": action.get("description", "Generate patch"),
            "data": {
                "observer_report": json.loads(self._art.observer_report) if isinstance(self._art.observer_report, str) else self._art.observer_report,
                "conservative": conservative
            },
            "instructions": textwrap.dedent(
                f"""
                Please generate a patch for bug {bug.id} based on the observer report.
                
                {"Use a conservative approach, prioritizing stability over completeness." if conservative else ""}
                
                Return a unified diff patch that fixes the bug. The patch should:
                - Not modify more than 5 files
                - Change no more than 120 lines total
                - Avoid touching generated or vendor folders
                
                Include "status": "SUCCESS" if you created a patch,
                or "status": "FAILURE" with an explanation if you couldn't.
                """
            )
        }
        
        # Execute the handoff
        status, result = await self._agent_handoff("observer", "analyst", handoff_data)
        
        # Process the result
        if status == HandoffStatus.SUCCESS:
            # Extract the patch bundle
            raw_response = result.get("raw_response", "")
            
            # Find the patch in the raw response
            if "diff --git" in raw_response:
                # Extract the diff
                patch_start = raw_response.find("diff --git")
