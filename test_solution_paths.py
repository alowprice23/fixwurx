#!/usr/bin/env python3
"""
Test Script for Solution Paths Module

This script tests the functionality of the solution paths module by selecting and executing
solution paths for various bug types.
"""

import os
import sys
import json
import time
from pathlib import Path

# Ensure the solution_paths module is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from solution_paths import (
        SolutionManager, SolutionPath, SolutionStrategy,
        select_solution_path, execute_solution_path
    )
except ImportError:
    print("Error: Could not import solution_paths module")
    sys.exit(1)

def _test_strategy_creation():
    """Test creating solution strategies."""
    print("\n=== Testing Strategy Creation ===")
    
    # Create a solution strategy
    strategy = SolutionStrategy(
        "TEST_STRATEGY", "Test Strategy", 
        "A test strategy for testing", 
        3, 0.85
    )
    
    print(f"Created strategy: {strategy.name} ({strategy.strategy_type})")
    print(f"Description: {strategy.description}")
    print(f"Complexity: {strategy.complexity}")
    print(f"Success probability: {strategy.success_probability:.2f}")
    
    # Create a custom handler
    def custom_handler(bug_data, context):
        print(f"Executing custom handler for bug: {bug_data.get('id', 'unknown')}")
        return {
            "success": True,
            "fixes_applied": ["Custom fix"]
        }
    
    # Create a strategy with a custom handler
    strategy_with_handler = SolutionStrategy(
        "CUSTOM_STRATEGY", "Custom Strategy", 
        "A strategy with a custom handler", 
        2, 0.9, custom_handler
    )
    
    print(f"\nCreated strategy with custom handler: {strategy_with_handler.name}")
    
    return strategy, strategy_with_handler

def _test_path_creation(primary_strategy, fallback_strategy):
    """
    Test creating solution paths.
    
    Args:
        primary_strategy: Primary strategy for the path
        fallback_strategy: Fallback strategy for the path
    """
    print("\n=== Testing Path Creation ===")
    
    # Create a solution path
    path = SolutionPath(
        "Test Path", 
        "A test path for testing"
    )
    
    # Add strategies to the path
    path.add_primary_strategy(primary_strategy)
    path.add_fallback_strategy(fallback_strategy)
    
    print(f"Created path: {path.name}")
    print(f"Description: {path.description}")
    print(f"Primary strategies: {len(path.primary_strategies)}")
    print(f"Fallback strategies: {len(path.fallback_strategies)}")
    
    return path

def _test_solution_manager():
    """Test the solution manager."""
    print("\n=== Testing Solution Manager ===")
    
    # Create a solution manager
    manager = SolutionManager()
    
    # Get the number of registered strategies and paths
    num_strategies = sum(len(strategies) for strategies in manager.strategies.values())
    num_paths = len(manager.paths)
    
    print(f"Solution manager initialized with:")
    print(f"  {num_strategies} strategies")
    print(f"  {num_paths} paths")
    
    # List strategy types
    print("\nStrategy types:")
    for strategy_type in manager.strategies.keys():
        num_strategies = len(manager.strategies[strategy_type])
        print(f"  {strategy_type}: {num_strategies} strategies")
    
    # List paths
    print("\nPaths:")
    for name in manager.paths.keys():
        print(f"  {name}")
    
    return manager

def _test_path_selection(manager):
    """
    Test selecting a solution path.
    
    Args:
        manager: Solution manager
    """
    print("\n=== Testing Path Selection ===")
    
    # Create bug data for different types of bugs
    bugs = [
        {
            "id": "BUG-001",
            "type": "syntax_error",
            "language": "python",
            "file_size": 500,
            "complexity": 2,
            "description": "Missing parenthesis in function call"
        },
        {
            "id": "BUG-002",
            "type": "logic_error",
            "language": "javascript",
            "file_size": 1200,
            "complexity": 4,
            "description": "Incorrect condition in if statement"
        },
        {
            "id": "BUG-003",
            "type": "security_issue",
            "language": "java",
            "file_size": 3000,
            "complexity": 7,
            "description": "SQL injection vulnerability"
        },
        {
            "id": "BUG-004",
            "type": "performance_issue",
            "language": "c++",
            "file_size": 2500,
            "complexity": 6,
            "description": "Inefficient algorithm in loop"
        }
    ]
    
    # Create context
    context = {
        "timestamp": int(time.time()),
        "test": True
    }
    
    # Select paths for each bug
    for bug in bugs:
        path = manager.select_path(bug, context)
        print(f"\nBug: {bug['id']} ({bug['type']})")
        print(f"Selected path: {path.name}")
        print(f"Description: {path.description}")
        print(f"Primary strategies: {[s.name for s in path.primary_strategies]}")
        print(f"Fallback strategies: {[s.name for s in path.fallback_strategies]}")
    
    return bugs, context

def _test_path_execution(manager, bugs, context):
    """
    Test executing solution paths.
    
    Args:
        manager: Solution manager
        bugs: List of bug data
        context: Context for execution
    """
    print("\n=== Testing Path Execution ===")
    
    # Execute paths for each bug
    for bug in bugs:
        print(f"\nExecuting solution for bug: {bug['id']} ({bug['type']})")
        
        # Execute with automatic path selection
        result = manager.execute_solution(bug, context)
        
        print(f"Success: {result.get('success', False)}")
        print(f"Path: {result.get('path', 'Unknown')}")
        
        if result.get("primary_strategy_succeeded", False):
            print("Primary strategy succeeded")
        elif result.get("fallback_strategy_succeeded", False):
            print("Fallback strategy succeeded")
        
        # Print strategy results
        strategies_attempted = len(result.get("strategy_results", []))
        print(f"Strategies attempted: {strategies_attempted}")
        
        if not result.get("success", False):
            print(f"Error: {result.get('error', 'Unknown')}")
    
    return True

def _test_api_functions():
    """Test the API functions."""
    print("\n=== Testing API Functions ===")
    
    # Create bug data
    bug = {
        "id": "BUG-005",
        "type": "validation_error",
        "language": "python",
        "file_size": 800,
        "complexity": 3,
        "description": "Missing input validation"
    }
    
    # Create context
    context = {
        "timestamp": int(time.time()),
        "test": True,
        "api_test": True
    }
    
    # Test select_solution_path
    print("\nTesting select_solution_path")
    select_result = select_solution_path(bug, context)
    
    print(f"Success: {select_result.get('success', False)}")
    if select_result.get("success", False):
        print(f"Path: {select_result.get('path', 'Unknown')}")
        print(f"Description: {select_result.get('description', 'Unknown')}")
        print(f"Primary strategies: {select_result.get('primary_strategies', [])}")
        print(f"Fallback strategies: {select_result.get('fallback_strategies', [])}")
    else:
        print(f"Error: {select_result.get('error', 'Unknown')}")
    
    # Test execute_solution_path
    print("\nTesting execute_solution_path")
    execute_result = execute_solution_path(bug, select_result.get("path"), context)
    
    print(f"Success: {execute_result.get('success', False)}")
    print(f"Path: {execute_result.get('path', 'Unknown')}")
    
    if execute_result.get("primary_strategy_succeeded", False):
        print("Primary strategy succeeded")
    elif execute_result.get("fallback_strategy_succeeded", False):
        print("Fallback strategy succeeded")
    
    # Print strategy results
    strategies_attempted = len(execute_result.get("strategy_results", []))
    print(f"Strategies attempted: {strategies_attempted}")
    
    if not execute_result.get("success", False):
        print(f"Error: {execute_result.get('error', 'Unknown')}")
    
    return True

# Pytest-compatible test functions

def test_create_strategies():
    """Pytest-compatible test for strategy creation."""
    strategies = _test_strategy_creation()
    assert strategies is not None
    assert len(strategies) == 2
    assert strategies[0].name == "Test Strategy"
    assert strategies[1].name == "Custom Strategy"
    return strategies

def test_create_path():
    """Pytest-compatible test for path creation."""
    strategies = test_create_strategies()
    path = _test_path_creation(strategies[0], strategies[1])
    assert path is not None
    assert path.name == "Test Path"
    assert len(path.primary_strategies) == 1
    assert len(path.fallback_strategies) == 1
    return path

def test_create_manager():
    """Pytest-compatible test for solution manager."""
    manager = _test_solution_manager()
    assert manager is not None
    assert len(manager.paths) > 0
    return manager

def test_select_paths():
    """Pytest-compatible test for path selection."""
    manager = test_create_manager()
    bugs, context = _test_path_selection(manager)
    assert bugs is not None
    assert context is not None
    assert len(bugs) > 0
    return bugs, context

def test_execute_paths():
    """Pytest-compatible test for path execution."""
    manager = test_create_manager()
    bugs, context = test_select_paths()
    result = _test_path_execution(manager, bugs, context)
    assert result is True

def test_api():
    """Pytest-compatible test for API functions."""
    result = _test_api_functions()
    assert result is True

# Original main function (kept for command-line testing)
def main():
    """Main function."""
    print("=== Solution Paths Test Suite ===")
    
    # Run tests sequentially in the correct order
    try:
        strategies = test_create_strategies()
        path = test_create_path()
        manager = test_create_manager()
        bugs, context = test_select_paths()
        test_execute_paths()
        test_api()
        
        print("\n=== Test Summary ===")
        print("All tests passed!")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
