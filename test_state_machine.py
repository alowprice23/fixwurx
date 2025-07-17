#!/usr/bin/env python3
"""
test_state_machine.py
────────────────────
Unit tests for state machine enhancements with planner integration.

Tests focus on:
1. New planner-specific phases
2. Transitions between planner phases
3. Path execution and verification
4. Fallback mechanism
5. Integration with existing phase system
"""

import unittest
from dataclasses import replace
from state_machine import (
    Bug, Phase, tick, transition_bug,
    AGENTS_PER_BUG, TICKS_PER_PHASE
)

class TestPlannerPhases(unittest.TestCase):
    """Test the planner-specific phases in the state machine."""
    
    def test_phase_classification(self):
        """Test phase classification properties."""
        # Test planner_phase property
        self.assertTrue(Phase.PLAN.planner_phase)
        self.assertTrue(Phase.PATH_EXEC.planner_phase)
        self.assertTrue(Phase.PATH_VERIFY.planner_phase)
        self.assertTrue(Phase.FALLBACK.planner_phase)
        
        # Test non-planner phases
        self.assertFalse(Phase.REPRO.planner_phase)
        self.assertFalse(Phase.PATCH.planner_phase)
        self.assertFalse(Phase.VERIFY.planner_phase)
        self.assertFalse(Phase.WAIT.planner_phase)
        
        # Test terminal phases
        self.assertTrue(Phase.DONE.terminal)
        self.assertTrue(Phase.ESCALATE.terminal)
        self.assertFalse(Phase.PLAN.terminal)
        
        # Test active phases (consuming agents)
        for phase in [Phase.PLAN, Phase.PATH_EXEC, Phase.PATH_VERIFY, Phase.FALLBACK]:
            self.assertTrue(phase.active)
            
        # Test timed phases
        for phase in [Phase.PLAN, Phase.PATH_EXEC, Phase.PATH_VERIFY, Phase.FALLBACK]:
            self.assertTrue(phase.timed)
    
    def test_bug_path_helpers(self):
        """Test bug path helper methods."""
        # Bug with no path
        bug = Bug(id="bug-1", phase=Phase.PLAN)
        self.assertFalse(bug.has_path())
        self.assertFalse(bug.has_fallback())
        
        # Bug with path
        bug = Bug(id="bug-1", phase=Phase.PLAN, path_id="path-1")
        self.assertTrue(bug.has_path())
        self.assertFalse(bug.has_fallback())
        
        # Bug with fallback
        bug = Bug(id="bug-1", phase=Phase.PLAN, path_id="path-1", fallback_path_id="fallback-1")
        self.assertTrue(bug.has_path())
        self.assertTrue(bug.has_fallback())


class TestPlannerTransitions(unittest.TestCase):
    """Test transitions between planner phases."""
    
    def test_wait_to_plan_transition(self):
        """Test transition from WAIT to PLAN phase."""
        bug = Bug(id="bug-1", phase=Phase.WAIT)
        free_agents = AGENTS_PER_BUG
        
        next_bug, delta = transition_bug(bug, free_agents)
        
        # Should transition to PLAN and consume agents
        self.assertEqual(next_bug.phase, Phase.PLAN)
        self.assertEqual(next_bug.timer, TICKS_PER_PHASE)
        self.assertEqual(delta, -AGENTS_PER_BUG)
    
    def test_plan_to_path_exec_transition(self):
        """Test transition from PLAN to PATH_EXEC phase."""
        # Bug with path ready for execution
        bug = Bug(id="bug-1", phase=Phase.PLAN, path_id="path-1", timer=0)
        free_agents = AGENTS_PER_BUG
        
        next_bug, delta = transition_bug(bug, free_agents)
        
        # Should transition to PATH_EXEC
        self.assertEqual(next_bug.phase, Phase.PATH_EXEC)
        self.assertEqual(next_bug.timer, TICKS_PER_PHASE)
        self.assertEqual(delta, 0)  # No change in agents
        
        # Bug without path should fall back to REPRO
        bug = Bug(id="bug-1", phase=Phase.PLAN, timer=0)
        next_bug, delta = transition_bug(bug, free_agents)
        
        self.assertEqual(next_bug.phase, Phase.REPRO)
        self.assertEqual(next_bug.timer, TICKS_PER_PHASE)
    
    def test_path_exec_to_path_verify_transition(self):
        """Test transition from PATH_EXEC to PATH_VERIFY phase."""
        bug = Bug(id="bug-1", phase=Phase.PATH_EXEC, path_id="path-1", timer=0)
        free_agents = AGENTS_PER_BUG
        
        next_bug, delta = transition_bug(bug, free_agents)
        
        # Should transition to PATH_VERIFY
        self.assertEqual(next_bug.phase, Phase.PATH_VERIFY)
        self.assertEqual(next_bug.timer, TICKS_PER_PHASE)
        self.assertEqual(delta, 0)  # No change in agents
    
    def test_path_verify_success_transition(self):
        """Test successful path verification transition to DONE."""
        # Successful path (path_success flag)
        bug = Bug(id="bug-1", phase=Phase.PATH_VERIFY, path_id="path-1", timer=0, path_success=True)
        free_agents = AGENTS_PER_BUG
        
        next_bug, delta = transition_bug(bug, free_agents)
        
        # Should transition to DONE and release agents
        self.assertEqual(next_bug.phase, Phase.DONE)
        self.assertEqual(next_bug.timer, 0)
        self.assertEqual(delta, +AGENTS_PER_BUG)
        
        # First attempt success (determined by _check_path_outcome)
        bug = Bug(id="bug-1", phase=Phase.PATH_VERIFY, path_id="path-1", timer=0, path_attempt=0)
        next_bug, delta = transition_bug(bug, free_agents)
        
        # Should transition to DONE and release agents
        self.assertEqual(next_bug.phase, Phase.DONE)
        self.assertEqual(delta, +AGENTS_PER_BUG)
    
    def test_path_verify_failure_with_fallback(self):
        """Test failed path verification with fallback available."""
        bug = Bug(
            id="bug-1", 
            phase=Phase.PATH_VERIFY, 
            path_id="path-1", 
            fallback_path_id="fallback-1",
            timer=0, 
            path_attempt=1  # Second attempt will fail according to _check_path_outcome
        )
        free_agents = AGENTS_PER_BUG
        
        next_bug, delta = transition_bug(bug, free_agents)
        
        # Should transition to FALLBACK
        self.assertEqual(next_bug.phase, Phase.FALLBACK)
        self.assertEqual(next_bug.timer, TICKS_PER_PHASE)
        self.assertEqual(next_bug.path_id, "fallback-1")
        self.assertEqual(next_bug.path_attempt, 2)
        self.assertEqual(delta, 0)  # No change in agents
    
    def test_path_verify_failure_retry_planning(self):
        """Test failed path verification with no fallback but retry allowed."""
        bug = Bug(
            id="bug-1", 
            phase=Phase.PATH_VERIFY, 
            path_id="path-1",
            timer=0, 
            path_attempt=1  # Second attempt will fail
        )
        free_agents = AGENTS_PER_BUG
        
        next_bug, delta = transition_bug(bug, free_agents)
        
        # Should transition back to PLAN for another attempt
        self.assertEqual(next_bug.phase, Phase.PLAN)
        self.assertEqual(next_bug.timer, TICKS_PER_PHASE)
        self.assertEqual(next_bug.path_id, "")  # Path cleared for new planning
        self.assertEqual(next_bug.path_attempt, 2)
        self.assertEqual(delta, 0)  # No change in agents
    
    def test_path_verify_failure_escalate(self):
        """Test failed path verification with no retry available."""
        bug = Bug(
            id="bug-1", 
            phase=Phase.PATH_VERIFY, 
            path_id="path-1",
            timer=0, 
            path_attempt=2  # Exceeds PROMOTION_LIMIT
        )
        free_agents = AGENTS_PER_BUG
        
        next_bug, delta = transition_bug(bug, free_agents)
        
        # Should transition to ESCALATE and release agents
        self.assertEqual(next_bug.phase, Phase.ESCALATE)
        self.assertEqual(next_bug.timer, 0)
        self.assertEqual(delta, +AGENTS_PER_BUG)
    
    def test_fallback_to_path_verify_transition(self):
        """Test transition from FALLBACK to PATH_VERIFY."""
        bug = Bug(id="bug-1", phase=Phase.FALLBACK, path_id="fallback-1", timer=0)
        free_agents = AGENTS_PER_BUG
        
        next_bug, delta = transition_bug(bug, free_agents)
        
        # Should transition to PATH_VERIFY
        self.assertEqual(next_bug.phase, Phase.PATH_VERIFY)
        self.assertEqual(next_bug.timer, TICKS_PER_PHASE)
        self.assertEqual(delta, 0)  # No change in agents


class TestGlobalTick(unittest.TestCase):
    """Test the global tick function with planner phases."""
    
    def test_planner_phase_transitions(self):
        """Test that planner phases transition correctly."""
        # Create a bug in PLAN phase with a path
        bug = Bug(id="bug-1", phase=Phase.PLAN, timer=0, path_id="path-1")
        free_agents = AGENTS_PER_BUG
        
        # Execute transition
        next_bug, delta = transition_bug(bug, free_agents)
        
        # Should transition to PATH_EXEC
        self.assertEqual(next_bug.phase, Phase.PATH_EXEC)
        self.assertEqual(next_bug.timer, TICKS_PER_PHASE)
        self.assertEqual(delta, 0)  # No agent change in this transition
        
        # Test that WAIT phase bugs can transition to PLAN when resources are available
        bug = Bug(id="bug-2", phase=Phase.WAIT)
        next_bug, delta = transition_bug(bug, free_agents)
        self.assertEqual(next_bug.phase, Phase.PLAN)
        self.assertEqual(delta, -AGENTS_PER_BUG)  # Agents allocated
    
    def test_timer_countdown(self):
        """Test timer countdown for planner phases."""
        # Create bug in PLAN phase with timer
        bug = Bug(id="bug-1", phase=Phase.PLAN, timer=TICKS_PER_PHASE)
        
        # Execute tick
        new_bugs, _ = tick([bug], AGENTS_PER_BUG)
        
        # Timer should decrement
        self.assertEqual(new_bugs[0].timer, TICKS_PER_PHASE - 1)
        self.assertEqual(new_bugs[0].phase, Phase.PLAN)
    
    def test_full_path_execution_cycle(self):
        """Test a full cycle of planner-based path execution using direct transitions."""
        # Create a bug that was already allocated agents (in PLAN phase)
        bug = Bug(id="bug-1", phase=Phase.PLAN, timer=0, path_id="path-1")
        free_agents = 0  # Already consumed by this bug
        
        # 1. PLAN → PATH_EXEC
        next_bug, delta = transition_bug(bug, free_agents)
        self.assertEqual(next_bug.phase, Phase.PATH_EXEC)
        self.assertEqual(next_bug.timer, TICKS_PER_PHASE)
        self.assertEqual(delta, 0)  # No agent change
        
        # 2. PATH_EXEC → PATH_VERIFY
        next_bug = replace(next_bug, timer=0)  # Simulate timer expiring
        next_bug, delta = transition_bug(next_bug, free_agents)
        self.assertEqual(next_bug.phase, Phase.PATH_VERIFY)
        self.assertEqual(next_bug.timer, TICKS_PER_PHASE)
        self.assertEqual(delta, 0)  # No agent change
        
        # 3. PATH_VERIFY → DONE (success case)
        next_bug = replace(next_bug, timer=0)  # Simulate timer expiring
        next_bug, delta = transition_bug(next_bug, free_agents + delta)
        self.assertEqual(next_bug.phase, Phase.DONE)
        self.assertEqual(delta, AGENTS_PER_BUG)  # Agents released
    
    def test_integration_with_existing_phases(self):
        """Test integration with existing phase system."""
        # Create a bug that will follow traditional flow
        bug = Bug(id="bug-1", phase=Phase.PLAN, timer=0)  # No path_id
        free_agents = AGENTS_PER_BUG
        
        # PLAN → REPRO (fallback to traditional flow due to no path)
        bugs, free = tick([bug], free_agents)
        self.assertEqual(bugs[0].phase, Phase.REPRO)
        
        # Let REPRO finish
        for _ in range(TICKS_PER_PHASE):
            bugs, free = tick(bugs, free)
        
        # REPRO → PATCH
        self.assertEqual(bugs[0].phase, Phase.PATCH)
        
        # This shows we can fall back to the traditional flow when needed


if __name__ == "__main__":
    unittest.main()
