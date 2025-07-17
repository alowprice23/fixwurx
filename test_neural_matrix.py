#!/usr/bin/env python3
"""
Test script for the Neural Matrix.

This script demonstrates the core functionality of the Neural Matrix,
including pattern recognition, learning from historical fixes, and
adaptive solution path selection.
"""

import sys
import json
import time
import uuid
from pathlib import Path

from neural_matrix_core import (
    register_bug, find_similar_bugs, find_solution_patterns,
    optimize_solution_paths, learn_from_fix, get_matrix_stats,
    get_matrix
)

def test_neural_matrix():
    """Test the core functionality of the Neural Matrix."""
    print("\n=== Neural Matrix Test ===\n")
    
    # Initialize matrix
    matrix = get_matrix()
    print("Matrix initialized")
    
    # Create test bugs
    bugs = [
        {
            "bug_id": "bug-1",
            "title": "Application crashes on startup",
            "description": "The application crashes immediately after launching with a segmentation fault.",
            "severity": "critical"
        },
        {
            "bug_id": "bug-2",
            "title": "Login form fails validation",
            "description": "When submitting the login form with valid credentials, the form fails validation and displays an error message.",
            "severity": "high"
        },
        {
            "bug_id": "bug-3",
            "title": "Database connection timeout",
            "description": "The application fails to connect to the database with a timeout error.",
            "severity": "high"
        },
        {
            "bug_id": "bug-4",
            "title": "Application crashes during startup",
            "description": "The application crashes during the initialization phase with an access violation error.",
            "severity": "critical"
        }
    ]
    
    # Register bugs
    for bug in bugs:
        result = register_bug(bug)
        
        if result.get("success", False):
            print(f"Bug '{bug['bug_id']}' registered successfully")
        else:
            print(f"Error registering bug: {result.get('error', 'Unknown error')}")
            return 1
    
    # Create test solutions
    solutions = [
        {
            "solution_id": "bug-1-solution-1",
            "bug_id": "bug-1",
            "steps": [
                {"type": "analyze", "description": "Analyze crash dump"},
                {"type": "implement", "description": "Fix memory allocation"},
                {"type": "verify", "description": "Verify fix by running application"}
            ],
            "fallbacks": [],
            "execution_time": 1800,  # 30 minutes
            "successful": True
        },
        {
            "solution_id": "bug-2-solution-1",
            "bug_id": "bug-2",
            "steps": [
                {"type": "analyze", "description": "Analyze form validation"},
                {"type": "implement", "description": "Fix validation logic"},
                {"type": "verify", "description": "Verify fix by submitting form"}
            ],
            "fallbacks": [],
            "execution_time": 900,  # 15 minutes
            "successful": True
        },
        {
            "solution_id": "bug-3-solution-1",
            "bug_id": "bug-3",
            "steps": [
                {"type": "analyze", "description": "Analyze database connection"},
                {"type": "implement", "description": "Increase timeout value"},
                {"type": "verify", "description": "Verify fix by connecting to database"}
            ],
            "fallbacks": [],
            "execution_time": 1200,  # 20 minutes
            "successful": False
        },
        {
            "solution_id": "bug-3-solution-2",
            "bug_id": "bug-3",
            "steps": [
                {"type": "analyze", "description": "Analyze database configuration"},
                {"type": "implement", "description": "Fix connection string"},
                {"type": "verify", "description": "Verify fix by connecting to database"}
            ],
            "fallbacks": [
                {"type": "fallback", "description": "Retry with increased timeout"}
            ],
            "execution_time": 2400,  # 40 minutes
            "successful": True
        }
    ]
    
    # Learn from solutions
    for solution in solutions:
        bug = next((b for b in bugs if b["bug_id"] == solution["bug_id"]), None)
        
        if bug:
            result = learn_from_fix(bug, solution)
            
            if result.get("success", False):
                print(f"Learned from solution '{solution['solution_id']}' (success: {solution['successful']})")
            else:
                print(f"Error learning from solution: {result.get('error', 'Unknown error')}")
                return 1
    
    # Test finding similar bugs
    test_bug = {
        "bug_id": "test-bug-1",
        "title": "Application crashes at launch",
        "description": "The application crashes immediately when launched with an access violation.",
        "severity": "critical"
    }
    
    result = find_similar_bugs(test_bug, threshold=0.5)
    
    if result.get("success", False):
        similar_bugs = result.get("similar_bugs", [])
        
        print(f"\nFound {len(similar_bugs)} similar bugs to '{test_bug['bug_id']}':")
        for bug_id, similarity in similar_bugs:
            print(f"  {bug_id}: {similarity:.2f} similarity")
    else:
        print(f"Error finding similar bugs: {result.get('error', 'Unknown error')}")
        return 1
    
    # Test finding solution patterns
    result = find_solution_patterns(test_bug)
    
    if result.get("success", False):
        patterns = result.get("patterns", [])
        
        print(f"\nFound {len(patterns)} solution patterns for '{test_bug['bug_id']}':")
        for solution_id, score in patterns:
            print(f"  {solution_id}: {score:.2f} confidence")
    else:
        print(f"Error finding solution patterns: {result.get('error', 'Unknown error')}")
        return 1
    
    # Test optimizing solution paths
    test_paths = [
        {
            "path_id": "path-1",
            "bug_id": "test-bug-1",
            "steps": [
                {"type": "analyze", "description": "Analyze crash dump"},
                {"type": "implement", "description": "Fix memory allocation"},
                {"type": "verify", "description": "Verify fix by running application"}
            ],
            "fallbacks": [],
            "score": 0.7
        },
        {
            "path_id": "path-2",
            "bug_id": "test-bug-1",
            "steps": [
                {"type": "analyze", "description": "Analyze application logs"},
                {"type": "implement", "description": "Fix initialization code"},
                {"type": "verify", "description": "Verify fix by running application"}
            ],
            "fallbacks": [
                {"type": "fallback", "description": "Try alternative initialization"}
            ],
            "score": 0.6
        }
    ]
    
    result = optimize_solution_paths(test_bug, test_paths)
    
    if result.get("success", False):
        optimized_paths = result.get("paths", [])
        
        print(f"\nOptimized solution paths for '{test_bug['bug_id']}':")
        for path in optimized_paths:
            print(f"  {path['path_id']}: {path.get('neural_score', 0):.2f} neural score (base score: {path['score']:.2f})")
    else:
        print(f"Error optimizing solution paths: {result.get('error', 'Unknown error')}")
        return 1
    
    # Get matrix stats
    result = get_matrix_stats()
    
    if result.get("success", False):
        stats = result.get("stats", {})
        
        print("\nNeural Matrix Statistics:")
        print(f"  Bugs: {stats.get('bugs', 0)}")
        print(f"  Solutions: {stats.get('solutions', 0)}")
        print(f"  Patterns: {stats.get('patterns', 0)}")
        print(f"  Learning Rate: {stats.get('learning_rate', 0)}")
        print(f"  Similarity Threshold: {stats.get('similarity_threshold', 0)}")
    else:
        print(f"Error getting matrix stats: {result.get('error', 'Unknown error')}")
        return 1
    
    print("\n=== Neural Matrix Test Completed Successfully ===")
    return 0

if __name__ == "__main__":
    sys.exit(test_neural_matrix())
