#!/usr/bin/env python3
"""
Enhanced Triangulum Commands Module

This module provides enhanced command handlers for the Triangulation Engine,
enabling bug tracking, path-based execution, and phase transitions.
"""

import os
import sys
import json
import time
import logging
import argparse
import shlex
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Import triangulation engine
try:
    from triangulation_engine import (
        register_bug, start_bug_fix, get_bug_status,
        get_execution_status, cancel_execution, get_engine_stats,
        get_engine, FixPhase
    )
except ImportError:
    print("Error: Triangulation Engine not available")
    sys.exit(1)

logger = logging.getLogger("TriangulationCommands")

def register_triangulation_commands(registry):
    """
    Register enhanced Triangulation command handlers with the component registry.
    
    Args:
        registry: Component registry instance
    """
    # Check if a Triangulation component already exists
    triangulum = registry.get_component("triangulum")
    if not triangulum:
        # Create a placeholder resource manager
        from resource_manager import ResourceManager
        triangulum = ResourceManager()
        registry.register_component("triangulum", triangulum)
    
    # Register command handlers for Triangulation Engine
    registry.register_command_handler("fix", fix_command, "triangulum")
    registry.register_command_handler("bug", bug_command, "triangulum")
    registry.register_command_handler("path", path_command, "triangulum")
    registry.register_command_handler("execution", execution_command, "triangulum")
    registry.register_command_handler("phase", phase_command, "triangulum")
    
    # Register prefixed versions
    registry.register_command_handler("triangulum:fix", fix_command, "triangulum")
    registry.register_command_handler("triangulum:bug", bug_command, "triangulum")
    registry.register_command_handler("triangulum:path", path_command, "triangulum")
    registry.register_command_handler("triangulum:execution", execution_command, "triangulum")
    registry.register_command_handler("triangulum:phase", phase_command, "triangulum")
    
    # Register aliases
    registry.register_alias("bugs", "bug list")
    registry.register_alias("executions", "execution list")
    registry.register_alias("paths", "path list")
    
    logger.info("Enhanced Triangulation commands registered")

def fix_command(args: str) -> int:
    """
    Fix a bug or manage fixes.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Fix a bug or manage fixes")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Start action
    start_parser = subparsers.add_parser("start", help="Start fixing a bug")
    start_parser.add_argument("bug_id", help="Bug ID")
    
    # Status action
    status_parser = subparsers.add_parser("status", help="Check fix status")
    status_parser.add_argument("bug_id", help="Bug ID")
    
    # Cancel action
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a fix")
    cancel_parser.add_argument("execution_id", help="Execution ID")
    
    # Stats action
    subparsers.add_parser("stats", help="Show fix statistics")
    
    # Parse arguments
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Initialize engine
    engine = get_engine()
    
    # Execute appropriate action
    if cmd_args.action == "start":
        result = start_bug_fix(cmd_args.bug_id)
        
        if result.get("success", False):
            print(f"Started fixing bug {cmd_args.bug_id}")
            print(f"Execution ID: {result.get('execution_id')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    
    elif cmd_args.action == "status":
        result = get_bug_status(cmd_args.bug_id)
        
        if result.get("success", False):
            status = result.get("status", {})
            
            print(f"\nBug {cmd_args.bug_id} - {status.get('title', 'Untitled')}")
            print("=" * 60)
            print(f"Status: {status.get('status', 'Unknown')}")
            print(f"Phase: {status.get('phase', 'Unknown')}")
            print(f"Severity: {status.get('severity', 'Unknown')}")
            
            if status.get("description"):
                print(f"\nDescription:")
                print(status.get("description"))
            
            if status.get("phase_history"):
                print("\nPhase History:")
                for i, phase in enumerate(status.get("phase_history", []), 1):
                    timestamp = time.ctime(phase.get("timestamp", 0))
                    print(f"  {i}. {phase.get('phase', 'Unknown')} at {timestamp}")
                    
                    details = phase.get("details", {})
                    if details:
                        for key, value in details.items():
                            print(f"     {key}: {value}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    
    elif cmd_args.action == "cancel":
        result = cancel_execution(cmd_args.execution_id)
        
        if result.get("success", False):
            print(f"Execution {cmd_args.execution_id} cancelled successfully")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    
    elif cmd_args.action == "stats":
        result = get_engine_stats()
        
        if result.get("success", False):
            stats = result.get("stats", {})
            
            print("\nTriangulation Engine Statistics:")
            print("=" * 60)
            
            bug_stats = stats.get("bugs", {})
            print(f"Total Bugs: {bug_stats.get('total', 0)}")
            
            phase_stats = bug_stats.get("by_phase", {})
            if phase_stats:
                print("\nBugs by Phase:")
                for phase, count in phase_stats.items():
                    if count > 0:
                        print(f"  {phase}: {count}")
            
            status_stats = bug_stats.get("by_status", {})
            if status_stats:
                print("\nBugs by Status:")
                for status, count in status_stats.items():
                    if count > 0:
                        print(f"  {status}: {count}")
            
            path_stats = stats.get("paths", {})
            print(f"\nTotal Paths: {path_stats.get('total', 0)}")
            
            path_status_stats = path_stats.get("by_status", {})
            if path_status_stats:
                print("\nPaths by Status:")
                for status, count in path_status_stats.items():
                    if count > 0:
                        print(f"  {status}: {count}")
            
            execution_stats = stats.get("executions", {})
            print(f"\nActive Executions: {execution_stats.get('active', 0)}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0

def bug_command(args: str) -> int:
    """
    Manage bugs in the Triangulation Engine.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Manage bugs")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Register action
    register_parser = subparsers.add_parser("register", help="Register a new bug")
    register_parser.add_argument("bug_id", help="Bug ID")
    register_parser.add_argument("title", help="Bug title")
    register_parser.add_argument("--description", help="Bug description")
    register_parser.add_argument("--severity", default="medium", 
                              choices=["critical", "high", "medium", "low"],
                              help="Bug severity")
    
    # List action
    list_parser = subparsers.add_parser("list", help="List bugs")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--phase", help="Filter by phase")
    
    # Show action
    show_parser = subparsers.add_parser("show", help="Show bug details")
    show_parser.add_argument("bug_id", help="Bug ID")
    
    # Parse arguments
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Initialize engine
    engine = get_engine()
    
    # Execute appropriate action
    if cmd_args.action == "register":
        result = register_bug(
            cmd_args.bug_id,
            cmd_args.title,
            cmd_args.description,
            cmd_args.severity
        )
        
        if result.get("success", False):
            print(f"Bug {cmd_args.bug_id} registered: {cmd_args.title}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    
    elif cmd_args.action == "list":
        # Get all bugs
        bugs = {}
        
        # For now, we'll just iterate through the engine's bugs dictionary
        for bug_id, bug in engine.bugs.items():
            # Apply filters
            if cmd_args.status and bug.status != cmd_args.status:
                continue
            
            if cmd_args.phase and bug.phase.value != cmd_args.phase:
                continue
            
            bugs[bug_id] = bug
        
        if not bugs:
            print("No bugs found")
            return 0
        
        print("\nBugs:")
        print("=" * 60)
        print(f"{'ID':<15} {'Status':<12} {'Phase':<15} {'Title'}")
        print("-" * 60)
        
        for bug_id, bug in sorted(bugs.items()):
            print(f"{bug_id:<15} {bug.status:<12} {bug.phase.value:<15} {bug.title}")
    
    elif cmd_args.action == "show":
        result = get_bug_status(cmd_args.bug_id)
        
        if result.get("success", False):
            status = result.get("status", {})
            
            print(f"\nBug {cmd_args.bug_id} - {status.get('title', 'Untitled')}")
            print("=" * 60)
            print(f"Status: {status.get('status', 'Unknown')}")
            print(f"Phase: {status.get('phase', 'Unknown')}")
            print(f"Severity: {status.get('severity', 'Unknown')}")
            print(f"Created: {time.ctime(status.get('created_at', 0))}")
            print(f"Updated: {time.ctime(status.get('updated_at', 0))}")
            
            if status.get("description"):
                print(f"\nDescription:")
                print(status.get("description"))
            
            if status.get("path_id"):
                print(f"\nActive Path: {status.get('path_id')}")
            
            if status.get("phase_history"):
                print("\nPhase History:")
                for i, phase in enumerate(status.get("phase_history", []), 1):
                    timestamp = time.ctime(phase.get("timestamp", 0))
                    print(f"  {i}. {phase.get('phase', 'Unknown')} at {timestamp}")
            
            if status.get("fix_attempts"):
                print("\nFix Attempts:")
                for i, attempt in enumerate(status.get("fix_attempts", []), 1):
                    timestamp = time.ctime(attempt.get("timestamp", 0))
                    print(f"  {i}. Attempt at {timestamp}")
                    
                    for key, value in attempt.items():
                        if key != "timestamp":
                            print(f"     {key}: {value}")
            
            if status.get("results"):
                print("\nResults:")
                for key, value in status.get("results", {}).items():
                    print(f"  {key}: {value}")
            
            if status.get("metadata"):
                print("\nMetadata:")
                for key, value in status.get("metadata", {}).items():
                    print(f"  {key}: {value}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0

def path_command(args: str) -> int:
    """
    Manage execution paths in the Triangulation Engine.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Manage execution paths")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # List action
    list_parser = subparsers.add_parser("list", help="List paths")
    list_parser.add_argument("--bug-id", help="Filter by bug ID")
    list_parser.add_argument("--status", help="Filter by status")
    
    # Show action
    show_parser = subparsers.add_parser("show", help="Show path details")
    show_parser.add_argument("path_id", help="Path ID")
    
    # Parse arguments
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Initialize engine
    engine = get_engine()
    
    # Execute appropriate action
    if cmd_args.action == "list":
        # Get all paths
        paths = {}
        
        # For now, we'll just iterate through the engine's paths dictionary
        for path_id, path in engine.paths.items():
            # Apply filters
            if cmd_args.bug_id and path.bug_id != cmd_args.bug_id:
                continue
            
            if cmd_args.status and path.status != cmd_args.status:
                continue
            
            paths[path_id] = path
        
        if not paths:
            print("No paths found")
            return 0
        
        print("\nExecution Paths:")
        print("=" * 60)
        print(f"{'ID':<25} {'Bug ID':<15} {'Status':<12} {'Current Step'}")
        print("-" * 60)
        
        for path_id, path in sorted(paths.items()):
            print(f"{path_id:<25} {path.bug_id:<15} {path.status:<12} {path.current_step}/{len(path.steps)}")
    
    elif cmd_args.action == "show":
        # Get path details
        if cmd_args.path_id not in engine.paths:
            print(f"Error: Path {cmd_args.path_id} not found")
            return 1
        
        path = engine.paths[cmd_args.path_id]
        
        print(f"\nExecution Path: {cmd_args.path_id}")
        print("=" * 60)
        print(f"Bug ID: {path.bug_id}")
        print(f"Status: {path.status}")
        print(f"Current Step: {path.current_step + 1}/{len(path.steps)}")
        print(f"Created: {time.ctime(path.created_at)}")
        print(f"Updated: {time.ctime(path.updated_at)}")
        
        if path.steps:
            print("\nSteps:")
            for i, step in enumerate(path.steps):
                marker = " *" if i == path.current_step else ""
                print(f"  {i + 1}.{marker} {step.get('phase', 'Unknown')} - {step.get('description', 'No description')}")
                
                params = step.get("params", {})
                if params:
                    for key, value in params.items():
                        print(f"     {key}: {value}")
        
        if path.results:
            print("\nResults:")
            for key, value in path.results.items():
                print(f"  {key}: {value}")
        
        if path.metadata:
            print("\nMetadata:")
            for key, value in path.metadata.items():
                print(f"  {key}: {value}")
    
    else:
        parser.print_help()
        return 1
    
    return 0

def execution_command(args: str) -> int:
    """
    Manage executions in the Triangulation Engine.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Manage executions")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # List action
    list_parser = subparsers.add_parser("list", help="List executions")
    
    # Show action
    show_parser = subparsers.add_parser("show", help="Show execution details")
    show_parser.add_argument("execution_id", help="Execution ID")
    
    # Cancel action
    cancel_parser = subparsers.add_parser("cancel", help="Cancel an execution")
    cancel_parser.add_argument("execution_id", help="Execution ID")
    
    # Parse arguments
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Initialize engine
    engine = get_engine()
    
    # Execute appropriate action
    if cmd_args.action == "list":
        # Get all executions
        executions = engine.active_executions
        
        if not executions:
            print("No active executions found")
            return 0
        
        print("\nActive Executions:")
        print("=" * 60)
        print(f"{'ID':<20} {'Bug ID':<15} {'Path ID'}")
        print("-" * 60)
        
        for exec_id, (bug_id, path_id) in sorted(executions.items()):
            print(f"{exec_id:<20} {bug_id:<15} {path_id}")
    
    elif cmd_args.action == "show":
        result = get_execution_status(cmd_args.execution_id)
        
        if result.get("success", False):
            status = result.get("status", {})
            
            print(f"\nExecution: {cmd_args.execution_id}")
            print("=" * 60)
            
            bug = status.get("bug", {})
            print(f"Bug ID: {bug.get('bug_id', 'Unknown')}")
            print(f"Bug Title: {bug.get('title', 'Unknown')}")
            print(f"Bug Status: {bug.get('status', 'Unknown')}")
            print(f"Bug Phase: {bug.get('phase', 'Unknown')}")
            
            path = status.get("path", {})
            print(f"\nPath ID: {path.get('path_id', 'Unknown')}")
            print(f"Path Status: {path.get('status', 'Unknown')}")
            print(f"Current Step: {path.get('current_step', 0) + 1}/{len(path.get('steps', []))}")
            
            steps = path.get("steps", [])
            if steps:
                print("\nSteps:")
                for i, step in enumerate(steps):
                    marker = " *" if i == path.get("current_step", 0) else ""
                    print(f"  {i + 1}.{marker} {step.get('phase', 'Unknown')} - {step.get('description', 'No description')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    
    elif cmd_args.action == "cancel":
        result = cancel_execution(cmd_args.execution_id)
        
        if result.get("success", False):
            print(f"Execution {cmd_args.execution_id} cancelled successfully")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0

def phase_command(args: str) -> int:
    """
    View and manage phases in the Triangulation Engine.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="View and manage phases")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # List action
    list_parser = subparsers.add_parser("list", help="List phases")
    
    # Show action
    show_parser = subparsers.add_parser("show", help="Show phase details")
    show_parser.add_argument("phase", help="Phase name")
    
    # Stats action
    stats_parser = subparsers.add_parser("stats", help="Show phase statistics")
    
    # Parse arguments
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Execute appropriate action
    if cmd_args.action == "list":
        print("\nTriangulation Engine Phases:")
        print("=" * 60)
        
        for phase in FixPhase:
            print(f"{phase.value} - {phase.name}")
    
    elif cmd_args.action == "show":
        # Check if phase exists
        try:
            phase = FixPhase(cmd_args.phase)
        except ValueError:
            print(f"Error: Phase {cmd_args.phase} not found")
            return 1
        
        print(f"\nPhase: {phase.value} ({phase.name})")
        print("=" * 60)
        
        # Get phase description
        if phase == FixPhase.INITIALIZE:
            print("Description: Initial state of a bug, before any processing begins")
            print("Follows: None (starting phase)")
            print("Followed by: ANALYZE")
        elif phase == FixPhase.ANALYZE:
            print("Description: Analyzing the bug to understand its root cause")
            print("Follows: INITIALIZE")
            print("Followed by: PLAN")
        elif phase == FixPhase.PLAN:
            print("Description: Planning the solution path for the bug")
            print("Follows: ANALYZE")
            print("Followed by: IMPLEMENT")
        elif phase == FixPhase.IMPLEMENT:
            print("Description: Implementing the fix for the bug")
            print("Follows: PLAN")
            print("Followed by: VERIFY")
        elif phase == FixPhase.VERIFY:
            print("Description: Verifying that the fix resolves the bug")
            print("Follows: IMPLEMENT")
            print("Followed by: LEARN")
        elif phase == FixPhase.LEARN:
            print("Description: Learning from the fix process to improve future fixes")
            print("Follows: VERIFY")
            print("Followed by: COMPLETE")
        elif phase == FixPhase.COMPLETE:
            print("Description: Bug has been successfully fixed")
            print("Follows: LEARN")
            print("Followed by: None (terminal phase)")
        elif phase == FixPhase.FAILED:
            print("Description: Bug fix process has failed")
            print("Follows: Any phase (on failure)")
            print("Followed by: None (terminal phase)")
        elif phase == FixPhase.ABANDONED:
            print("Description: Bug fix process has been abandoned")
            print("Follows: Any phase (on abandonment)")
            print("Followed by: None (terminal phase)")
        
        # Get bugs in this phase
        engine = get_engine()
        bugs_in_phase = [bug for bug in engine.bugs.values() if bug.phase == phase]
        
        if bugs_in_phase:
            print(f"\nBugs in {phase.value} phase:")
            for bug in bugs_in_phase:
                print(f"  {bug.bug_id} - {bug.title}")
    
    elif cmd_args.action == "stats":
        engine = get_engine()
        
        print("\nPhase Statistics:")
        print("=" * 60)
        
        # Count bugs by phase
        phase_counts = {}
        for phase in FixPhase:
            phase_counts[phase] = len([bug for bug in engine.bugs.values() if bug.phase == phase])
        
        for phase, count in phase_counts.items():
            print(f"{phase.value}: {count} bugs")
        
        # Calculate phase transition stats
        transitions = {}
        for bug in engine.bugs.values():
            history = bug.phase_history
            
            for i in range(1, len(history)):
                from_phase = history[i-1]["phase"]
                to_phase = history[i]["phase"]
                key = f"{from_phase} -> {to_phase}"
                
                if key not in transitions:
                    transitions[key] = 0
                
                transitions[key] += 1
        
        if transitions:
            print("\nPhase Transitions:")
            for transition, count in sorted(transitions.items()):
                print(f"{transition}: {count}")
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    # Simple CLI for testing
    print("Triangulation Commands Module")
    print("This module should be imported by the shell environment")
    print("Run 'python triangulation_engine.py' for direct engine access")
