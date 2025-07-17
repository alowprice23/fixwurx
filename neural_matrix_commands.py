#!/usr/bin/env python3
"""
Neural Matrix Commands Module

This module registers command handlers for the Neural Matrix system within the shell environment,
enabling pattern recognition, learning from historical fixes, and adaptive solution path selection.
"""

import os
import sys
import json
import time
import logging
import argparse
import shlex
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("NeuralMatrixCommands")

def register_neural_matrix_commands(registry):
    """
    Register Neural Matrix command handlers with the component registry.
    
    Args:
        registry: Component registry instance
    """
    try:
        # Register command handlers
        registry.register_command_handler("neural:register", register_command, "neural_matrix")
        registry.register_command_handler("neural:similar", similar_command, "neural_matrix")
        registry.register_command_handler("neural:patterns", patterns_command, "neural_matrix")
        registry.register_command_handler("neural:optimize", optimize_command, "neural_matrix")
        registry.register_command_handler("neural:learn", learn_command, "neural_matrix")
        registry.register_command_handler("neural:stats", stats_command, "neural_matrix")
        
        # Register aliases
        registry.register_alias("nreg", "neural:register")
        registry.register_alias("nsim", "neural:similar")
        registry.register_alias("npat", "neural:patterns")
        registry.register_alias("nopt", "neural:optimize")
        registry.register_alias("nlearn", "neural:learn")
        registry.register_alias("nstats", "neural:stats")
        
        logger.info("Neural Matrix commands registered")
    except Exception as e:
        logger.error(f"Error registering Neural Matrix commands: {e}")

def register_command(args: str) -> int:
    """
    Register a bug with the Neural Matrix.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Register a bug with the Neural Matrix")
    parser.add_argument("bug_id", help="Bug ID")
    parser.add_argument("title", help="Bug title")
    parser.add_argument("--description", help="Bug description")
    parser.add_argument("--severity", default="medium", 
                      choices=["critical", "high", "medium", "low"],
                      help="Bug severity")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import neural matrix
    try:
        from neural_matrix_core import register_bug
    except ImportError:
        print("Error: Neural Matrix not available")
        return 1
    
    # Register bug
    result = register_bug({
        "bug_id": cmd_args.bug_id,
        "title": cmd_args.title,
        "description": cmd_args.description,
        "severity": cmd_args.severity
    })
    
    if result.get("success", False):
        print(f"Bug {cmd_args.bug_id} registered with Neural Matrix")
    else:
        print(f"Error registering bug: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

def similar_command(args: str) -> int:
    """
    Find bugs similar to the given bug.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Find similar bugs")
    parser.add_argument("bug_id", help="Bug ID")
    parser.add_argument("--title", help="Bug title")
    parser.add_argument("--description", help="Bug description")
    parser.add_argument("--severity", default="medium", 
                      choices=["critical", "high", "medium", "low"],
                      help="Bug severity")
    parser.add_argument("--threshold", type=float, help="Similarity threshold")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import neural matrix
    try:
        from neural_matrix_core import find_similar_bugs
    except ImportError:
        print("Error: Neural Matrix not available")
        return 1
    
    # Find similar bugs
    result = find_similar_bugs({
        "bug_id": cmd_args.bug_id,
        "title": cmd_args.title,
        "description": cmd_args.description,
        "severity": cmd_args.severity
    }, cmd_args.threshold)
    
    if result.get("success", False):
        similar_bugs = result.get("similar_bugs", [])
        
        if similar_bugs:
            print(f"\nFound {len(similar_bugs)} similar bugs:")
            print("-" * 60)
            
            for bug_id, similarity in similar_bugs:
                print(f"  {bug_id}: {similarity:.2f} similarity")
        else:
            print("No similar bugs found")
    else:
        print(f"Error finding similar bugs: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

def patterns_command(args: str) -> int:
    """
    Find solution patterns for a bug.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Find solution patterns")
    parser.add_argument("bug_id", help="Bug ID")
    parser.add_argument("--title", help="Bug title")
    parser.add_argument("--description", help="Bug description")
    parser.add_argument("--severity", default="medium", 
                      choices=["critical", "high", "medium", "low"],
                      help="Bug severity")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import neural matrix
    try:
        from neural_matrix_core import find_solution_patterns
    except ImportError:
        print("Error: Neural Matrix not available")
        return 1
    
    # Find solution patterns
    result = find_solution_patterns({
        "bug_id": cmd_args.bug_id,
        "title": cmd_args.title,
        "description": cmd_args.description,
        "severity": cmd_args.severity
    })
    
    if result.get("success", False):
        patterns = result.get("patterns", [])
        
        if patterns:
            print(f"\nFound {len(patterns)} solution patterns:")
            print("-" * 60)
            
            for solution_id, score in patterns:
                print(f"  {solution_id}: {score:.2f} confidence")
        else:
            print("No solution patterns found")
    else:
        print(f"Error finding solution patterns: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

def optimize_command(args: str) -> int:
    """
    Optimize solution paths for a bug.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Optimize solution paths")
    parser.add_argument("bug_id", help="Bug ID")
    parser.add_argument("--paths", help="Path to JSON file containing solution paths")
    parser.add_argument("--title", help="Bug title")
    parser.add_argument("--description", help="Bug description")
    parser.add_argument("--severity", default="medium", 
                      choices=["critical", "high", "medium", "low"],
                      help="Bug severity")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import neural matrix
    try:
        from neural_matrix_core import optimize_solution_paths
    except ImportError:
        print("Error: Neural Matrix not available")
        return 1
    
    # Load paths from file
    paths = []
    if cmd_args.paths:
        try:
            with open(cmd_args.paths, 'r') as f:
                paths = json.load(f)
        except Exception as e:
            print(f"Error loading paths file: {e}")
            return 1
    else:
        # Use example paths
        paths = [
            {
                "path_id": "path-1",
                "bug_id": cmd_args.bug_id,
                "steps": [
                    {"type": "analyze", "description": "Analyze bug"},
                    {"type": "implement", "description": "Fix issue"},
                    {"type": "verify", "description": "Verify fix"}
                ],
                "fallbacks": [],
                "score": 0.7
            },
            {
                "path_id": "path-2",
                "bug_id": cmd_args.bug_id,
                "steps": [
                    {"type": "analyze", "description": "Analyze bug"},
                    {"type": "plan", "description": "Plan solution"},
                    {"type": "implement", "description": "Implement fix"},
                    {"type": "verify", "description": "Verify fix"}
                ],
                "fallbacks": [
                    {"type": "fallback", "description": "Try alternative approach"}
                ],
                "score": 0.6
            }
        ]
    
    # Optimize solution paths
    result = optimize_solution_paths({
        "bug_id": cmd_args.bug_id,
        "title": cmd_args.title,
        "description": cmd_args.description,
        "severity": cmd_args.severity
    }, paths)
    
    if result.get("success", False):
        optimized_paths = result.get("paths", [])
        
        if optimized_paths:
            print(f"\nOptimized solution paths:")
            print("-" * 60)
            
            for path in optimized_paths:
                print(f"  {path['path_id']}: {path.get('neural_score', 0):.2f} neural score (base score: {path['score']:.2f})")
                
                # Print steps
                print("  Steps:")
                for i, step in enumerate(path.get("steps", []), 1):
                    print(f"    {i}. {step.get('type')}: {step.get('description')}")
                
                # Print fallbacks
                fallbacks = path.get("fallbacks", [])
                if fallbacks:
                    print("  Fallbacks:")
                    for i, fallback in enumerate(fallbacks, 1):
                        print(f"    {i}. {fallback.get('description')}")
                
                print()
        else:
            print("No optimized paths found")
    else:
        print(f"Error optimizing solution paths: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

def learn_command(args: str) -> int:
    """
    Learn from a successful or failed fix.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Learn from a fix")
    parser.add_argument("bug_id", help="Bug ID")
    parser.add_argument("solution_id", help="Solution ID")
    parser.add_argument("--successful", action="store_true", help="Whether the fix was successful")
    parser.add_argument("--steps", help="Path to JSON file containing solution steps")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import neural matrix
    try:
        from neural_matrix_core import learn_from_fix
    except ImportError:
        print("Error: Neural Matrix not available")
        return 1
    
    # Load steps from file
    steps = []
    if cmd_args.steps:
        try:
            with open(cmd_args.steps, 'r') as f:
                steps = json.load(f)
        except Exception as e:
            print(f"Error loading steps file: {e}")
            return 1
    else:
        # Use example steps
        steps = [
            {"type": "analyze", "description": "Analyze bug"},
            {"type": "implement", "description": "Fix issue"},
            {"type": "verify", "description": "Verify fix"}
        ]
    
    # Learn from fix
    result = learn_from_fix(
        {
            "bug_id": cmd_args.bug_id,
            "title": f"Bug {cmd_args.bug_id}",
            "description": f"Description for bug {cmd_args.bug_id}",
            "severity": "medium"
        },
        {
            "solution_id": cmd_args.solution_id,
            "bug_id": cmd_args.bug_id,
            "steps": steps,
            "fallbacks": [],
            "execution_time": 1800,  # 30 minutes
            "successful": cmd_args.successful
        }
    )
    
    if result.get("success", False):
        print(f"Learned from fix: bug={cmd_args.bug_id}, solution={cmd_args.solution_id}, successful={cmd_args.successful}")
    else:
        print(f"Error learning from fix: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

def stats_command(args: str) -> int:
    """
    Get Neural Matrix statistics.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    # Import neural matrix
    try:
        from neural_matrix_core import get_matrix_stats
    except ImportError:
        print("Error: Neural Matrix not available")
        return 1
    
    # Get stats
    result = get_matrix_stats()
    
    if result.get("success", False):
        stats = result.get("stats", {})
        
        print("\nNeural Matrix Statistics:")
        print("-" * 60)
        
        print(f"Bugs: {stats.get('bugs', 0)}")
        print(f"Solutions: {stats.get('solutions', 0)}")
        print(f"Patterns: {stats.get('patterns', 0)}")
        print(f"Learning Rate: {stats.get('learning_rate', 0)}")
        print(f"Similarity Threshold: {stats.get('similarity_threshold', 0)}")
        print(f"Pattern Count: {stats.get('pattern_count', 0)}")
        print(f"Weight Decay: {stats.get('weight_decay', 0)}")
    else:
        print(f"Error getting matrix stats: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    print("Neural Matrix Commands Module")
    print("This module should be imported by the shell environment")
