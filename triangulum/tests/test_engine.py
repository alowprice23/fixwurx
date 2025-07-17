#!/usr/bin/env python3
"""
Test script for the Triangulation Engine.

This script demonstrates the core functionality of the Triangulation Engine,
including bug registration, execution paths, and phase transitions.
"""

import sys
import json
import time
from pathlib import Path

from triangulation_engine import (
    register_bug, start_bug_fix, get_bug_status,
    get_execution_status, cancel_execution, get_engine_stats,
    get_engine, FixPhase
)

def test_triangulation_engine():
    """Test the core functionality of the Triangulation Engine."""
    print("\n=== Triangulation Engine Test ===\n")
    
    # Initialize engine
    engine = get_engine()
    print("Engine initialized")
    
    # Register a test bug
    bug_id = "test-bug-1"
    title = "Test Bug for Triangulation Engine"
    description = "This is a test bug to demonstrate the Triangulation Engine"
    
    result = register_bug(bug_id, title, description, "high")
    
    if result.get("success", False):
        print(f"Bug '{bug_id}' registered successfully")
    else:
        print(f"Error registering bug: {result.get('error', 'Unknown error')}")
        return 1
    
    # Start fixing the bug
    result = start_bug_fix(bug_id)
    
    if result.get("success", False):
        execution_id = result.get("execution_id")
        print(f"Started fixing bug '{bug_id}' with execution ID: {execution_id}")
    else:
        print(f"Error starting bug fix: {result.get('error', 'Unknown error')}")
        return 1
    
    # Check bug status (should be in ANALYZE phase now)
    result = get_bug_status(bug_id)
    
    if result.get("success", False):
        status = result.get("status", {})
        print(f"Bug status: {status.get('status')}")
        print(f"Bug phase: {status.get('phase')}")
    else:
        print(f"Error getting bug status: {result.get('error', 'Unknown error')}")
        return 1
    
    # Check execution status
    result = get_execution_status(execution_id)
    
    if result.get("success", False):
        status = result.get("status", {})
        bug = status.get("bug", {})
        path = status.get("path", {})
        
        print(f"\nExecution Status:")
        print(f"Bug ID: {bug.get('bug_id')}")
        print(f"Bug Status: {bug.get('status')}")
        print(f"Bug Phase: {bug.get('phase')}")
        print(f"Path ID: {path.get('path_id')}")
        print(f"Current Step: {path.get('current_step') + 1}/{len(path.get('steps', []))}")
    else:
        print(f"Error getting execution status: {result.get('error', 'Unknown error')}")
        return 1
    
    # Wait for the execution to progress through phases
    print("\nWaiting for execution to progress (simulated)...")
    time.sleep(1)
    
    # Since we don't have the actual agent implementations, we'll simulate
    # the engine progressing through phases by directly updating the bug's phase
    if bug_id in engine.bugs:
        bug = engine.bugs[bug_id]
        
        # Move through the phases
        phases = [
            FixPhase.PLAN,
            FixPhase.IMPLEMENT,
            FixPhase.VERIFY,
            FixPhase.LEARN,
            FixPhase.COMPLETE
        ]
        
        for phase in phases:
            bug.update_phase(phase, {
                "execution_id": execution_id,
                "simulated": True,
                "timestamp": time.time()
            })
            
            print(f"Bug moved to phase: {phase.value}")
            time.sleep(0.5)
    
    # Get final bug status
    result = get_bug_status(bug_id)
    
    if result.get("success", False):
        status = result.get("status", {})
        print(f"\nFinal Bug Status:")
        print(f"Status: {status.get('status')}")
        print(f"Phase: {status.get('phase')}")
        
        if status.get("phase_history"):
            print("\nPhase History:")
            for i, phase in enumerate(status.get("phase_history", []), 1):
                timestamp = time.ctime(phase.get("timestamp", 0))
                print(f"  {i}. {phase.get('phase')} at {timestamp}")
    else:
        print(f"Error getting bug status: {result.get('error', 'Unknown error')}")
        return 1
    
    # Get engine stats
    result = get_engine_stats()
    
    if result.get("success", False):
        stats = result.get("stats", {})
        print("\nEngine Statistics:")
        print(f"Total Bugs: {stats.get('bugs', {}).get('total', 0)}")
        print(f"Total Paths: {stats.get('paths', {}).get('total', 0)}")
        print(f"Active Executions: {stats.get('executions', {}).get('active', 0)}")
    else:
        print(f"Error getting engine stats: {result.get('error', 'Unknown error')}")
        return 1
    
    print("\n=== Triangulation Engine Test Completed Successfully ===")
    return 0

if __name__ == "__main__":
    sys.exit(test_triangulation_engine())
