#!/usr/bin/env python3
"""
state_machine.py
────────────────
State machine for managing the bug resolution process.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

# Constants
AGENTS_PER_BUG = 3
TICKS_PER_PHASE = 10
MAX_PATH_ATTEMPTS = 3

class Phase(Enum):
    """
    Defines the phases of the bug resolution process.
    """
    TRIAGE = auto()
    REPRO = auto()
    ANALYZE = auto()
    PATCH = auto()
    VERIFY = auto()
    DONE = auto()
    ESCALATE = auto()
    
    # New planner-specific phases
    WAIT = auto()      # Waiting for resources
    PLAN = auto()      # Planning solution paths
    PATH_EXEC = auto() # Executing a solution path
    PATH_VERIFY = auto() # Verifying a solution path
    FALLBACK = auto()  # Using fallback path
    
    @property
    def planner_phase(self) -> bool:
        """Whether this is a planner-specific phase."""
        return self in {Phase.PLAN, Phase.PATH_EXEC, Phase.PATH_VERIFY, Phase.FALLBACK}
    
    @property
    def terminal(self) -> bool:
        """Whether this is a terminal phase (no further processing)."""
        return self in {Phase.DONE, Phase.ESCALATE}
    
    @property
    def active(self) -> bool:
        """Whether this phase is actively consuming agents."""
        return self not in {Phase.WAIT, Phase.DONE, Phase.ESCALATE}
    
    @property
    def timed(self) -> bool:
        """Whether this phase has a timer."""
        return self not in {Phase.WAIT, Phase.DONE, Phase.ESCALATE}


@dataclass
class Bug:
    """
    Represents a bug in the system.
    """
    id: str
    phase: Phase = Phase.TRIAGE
    timer: int = 0
    path_id: str = ""
    fallback_path_id: str = ""
    path_attempt: int = 0
    path_success: bool = False
    
    def has_path(self) -> bool:
        """Whether this bug has a solution path."""
        return bool(self.path_id)
    
    def has_fallback(self) -> bool:
        """Whether this bug has a fallback path."""
        return bool(self.fallback_path_id)


class StateMachine:
    """
    State machine for managing the bug resolution process.
    """
    
    def __init__(self, initial_phase: Phase = Phase.TRIAGE):
        """Initialize the state machine."""
        self.phase = initial_phase
    
    def transition_to(self, new_phase: Phase) -> None:
        """
        Transition to a new phase.
        
        Args:
            new_phase: The new phase to transition to
        """
        self.phase = new_phase


def _check_path_outcome(bug: Bug) -> bool:
    """
    Check if a path execution was successful.
    This is a simple deterministic implementation for testing.
    
    Args:
        bug: The bug to check
        
    Returns:
        True if the path execution was successful
    """
    # For testing: first attempt always succeeds, others depend on path_success flag
    if bug.path_attempt == 0:
        return True
    return bug.path_success


def transition_bug(bug: Bug, free_agents: int) -> Tuple[Bug, int]:
    """
    Transition a bug to its next phase based on current state.
    
    Args:
        bug: The bug to transition
        free_agents: Number of free agents available
        
    Returns:
        Tuple of (new bug state, agent delta)
    """
    # Agent delta tracks how many agents are consumed (-) or released (+)
    agent_delta = 0
    
    # If the bug is in a timed phase and the timer expired
    if bug.phase.timed and bug.timer <= 0:
        # Handle phase transitions
        if bug.phase == Phase.PLAN:
            # PLAN → PATH_EXEC if we have a path, otherwise REPRO
            if bug.has_path():
                return Bug(
                    id=bug.id,
                    phase=Phase.PATH_EXEC,
                    timer=TICKS_PER_PHASE,
                    path_id=bug.path_id,
                    fallback_path_id=bug.fallback_path_id,
                    path_attempt=bug.path_attempt
                ), 0
            else:
                # No path, fall back to traditional flow
                return Bug(
                    id=bug.id,
                    phase=Phase.REPRO,
                    timer=TICKS_PER_PHASE,
                    path_attempt=bug.path_attempt
                ), 0
                
        elif bug.phase == Phase.PATH_EXEC:
            # PATH_EXEC → PATH_VERIFY
            return Bug(
                id=bug.id,
                phase=Phase.PATH_VERIFY,
                timer=TICKS_PER_PHASE,
                path_id=bug.path_id,
                fallback_path_id=bug.fallback_path_id,
                path_attempt=bug.path_attempt
            ), 0
            
        elif bug.phase == Phase.PATH_VERIFY:
            # Check if path execution was successful
            if _check_path_outcome(bug):
                # PATH_VERIFY → DONE (success)
                return Bug(
                    id=bug.id,
                    phase=Phase.DONE,
                    path_id=bug.path_id,
                    path_attempt=bug.path_attempt,
                    path_success=True
                ), +AGENTS_PER_BUG  # Release agents
                
            # Path execution failed
            elif bug.has_fallback():
                # PATH_VERIFY → FALLBACK
                return Bug(
                    id=bug.id,
                    phase=Phase.FALLBACK,
                    timer=TICKS_PER_PHASE,
                    path_id=bug.fallback_path_id,  # Switch to fallback path
                    path_attempt=bug.path_attempt + 1
                ), 0
                
            elif bug.path_attempt < MAX_PATH_ATTEMPTS - 1:
                # PATH_VERIFY → PLAN (retry)
                return Bug(
                    id=bug.id,
                    phase=Phase.PLAN,
                    timer=TICKS_PER_PHASE,
                    path_id="",  # Clear path for new planning
                    path_attempt=bug.path_attempt + 1
                ), 0
                
            else:
                # PATH_VERIFY → ESCALATE (exceeded retries)
                return Bug(
                    id=bug.id,
                    phase=Phase.ESCALATE,
                    path_attempt=bug.path_attempt
                ), +AGENTS_PER_BUG  # Release agents
                
        elif bug.phase == Phase.FALLBACK:
            # FALLBACK → PATH_VERIFY
            return Bug(
                id=bug.id,
                phase=Phase.PATH_VERIFY,
                timer=TICKS_PER_PHASE,
                path_id=bug.path_id,
                path_attempt=bug.path_attempt
            ), 0
            
        # Handle traditional flow phase transitions
        elif bug.phase == Phase.REPRO:
            # REPRO → PATCH
            return Bug(
                id=bug.id,
                phase=Phase.PATCH,
                timer=TICKS_PER_PHASE,
                path_attempt=bug.path_attempt
            ), 0
            
        elif bug.phase == Phase.PATCH:
            # PATCH → VERIFY
            return Bug(
                id=bug.id,
                phase=Phase.VERIFY,
                timer=TICKS_PER_PHASE,
                path_attempt=bug.path_attempt
            ), 0
            
        elif bug.phase == Phase.VERIFY:
            # VERIFY → DONE
            return Bug(
                id=bug.id,
                phase=Phase.DONE,
                path_attempt=bug.path_attempt
            ), +AGENTS_PER_BUG  # Release agents
    
    # Handle waiting bugs
    elif bug.phase == Phase.WAIT:
        # WAIT → PLAN if we have enough agents
        if free_agents >= AGENTS_PER_BUG:
            return Bug(
                id=bug.id,
                phase=Phase.PLAN,
                timer=TICKS_PER_PHASE
            ), -AGENTS_PER_BUG  # Consume agents
    
    # Default: no transition, just return the bug as is
    return bug, 0


def tick(bugs: List[Bug], free_agents: int) -> Tuple[List[Bug], int]:
    """
    Process a single tick for all bugs.
    
    Args:
        bugs: List of bugs to process
        free_agents: Number of free agents available
        
    Returns:
        Tuple of (new bug list, new free agent count)
    """
    new_bugs = []
    
    for bug in bugs:
        if bug.phase.timed:
            # Decrement timer
            bug = Bug(
                id=bug.id,
                phase=bug.phase,
                timer=max(0, bug.timer - 1),
                path_id=bug.path_id,
                fallback_path_id=bug.fallback_path_id,
                path_attempt=bug.path_attempt,
                path_success=bug.path_success
            )
        
        # Try to transition the bug
        new_bug, agent_delta = transition_bug(bug, free_agents)
        
        # Update free agent count
        free_agents += agent_delta
        
        # Add the new bug to the list
        new_bugs.append(new_bug)
    
    return new_bugs, free_agents
