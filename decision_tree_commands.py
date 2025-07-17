#!/usr/bin/env python3
"""
Decision Tree Commands

This module provides commands for the decision tree to be registered with the shell.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(".triangulum/decision_tree_commands.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("DecisionTreeCommands")

# Import API functions from integration module
from decision_tree_integration import (
    identify_bug, generate_solution_paths,
    select_best_solution_path, fix_bug
)

def bug_identify_command(shell, args):
    """
    Identify a bug in code.
    
    Usage: bug_identify <file> [--language <language>]
    """
    if not args:
        shell.print_error("No file specified")
        shell.print("Usage: bug_identify <file> [--language <language>]")
        return
    
    file_path = args[0]
    
    # Check if language is specified
    language = "py"
    if len(args) > 1 and args[1] == "--language" and len(args) > 2:
        language = args[2]
    
    # Check if file exists
    if not os.path.exists(file_path):
        shell.print_error(f"File {file_path} not found")
        return
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        code_content = f.read()
    
    # Identify bug
    shell.print(f"Identifying bug in {file_path} (language: {language})...")
    result = identify_bug(code_content, language)
    
    if result.get("success", False):
        bug_id = result.get("bug_id")
        bug_info = result.get("bug_info", {})
        
        shell.print(f"Identified bug: {bug_id}")
        shell.print(f"Bug type: {bug_info.get('bug_type')}")
        shell.print(f"Severity: {bug_info.get('severity')}")
        shell.print(f"Complexity: {bug_info.get('complexity')}")
        
        # Save bug_id for later use
        shell.set_variable("last_bug_id", bug_id)
        
        return bug_id
    else:
        shell.print_error(f"Error identifying bug: {result.get('error')}")
        return None

def bug_generate_paths_command(shell, args):
    """
    Generate solution paths for a bug.
    
    Usage: bug_generate_paths <bug_id>
    """
    if not args:
        # Try to get bug_id from shell variable
        bug_id = shell.get_variable("last_bug_id")
        if not bug_id:
            shell.print_error("No bug ID specified or found in last_bug_id variable")
            shell.print("Usage: bug_generate_paths <bug_id>")
            return
    else:
        bug_id = args[0]
    
    # Generate solution paths
    shell.print(f"Generating solution paths for bug {bug_id}...")
    result = generate_solution_paths(bug_id)
    
    if result.get("success", False):
        paths = result.get("paths", [])
        
        shell.print(f"Generated {len(paths)} solution paths for bug {bug_id}")
        
        # Print details of each path
        for i, path in enumerate(paths):
            shell.print(f"\nPath {i+1} - {path.get('path_id')}")
            shell.print(f"  Approach: {path.get('approach')}")
            shell.print(f"  Confidence: {path.get('confidence')}")
            shell.print(f"  Estimated Time: {path.get('estimated_time')} minutes")
            shell.print(f"  Estimated Complexity: {path.get('estimated_complexity')}")
            shell.print(f"  Estimated Success Rate: {path.get('estimated_success_rate', 0) * 100:.1f}%")
            shell.print(f"  Score: {path.get('score', 0) * 100:.0f}")
        
        # Save paths for later use
        shell.set_variable("solution_paths", paths)
        
        return paths
    else:
        shell.print_error(f"Error generating solution paths: {result.get('error')}")
        return None

def bug_select_path_command(shell, args):
    """
    Select the best solution path.
    
    Usage: bug_select_path [path_index]
    """
    # Get paths from shell variable
    paths = shell.get_variable("solution_paths")
    if not paths:
        shell.print_error("No solution paths found. Run bug_generate_paths first.")
        return
    
    # Check if path index is specified
    if args and args[0].isdigit():
        index = int(args[0]) - 1
        if 0 <= index < len(paths):
            best_path = paths[index]
            shell.print(f"Selected path {index+1}: {best_path.get('path_id')}")
        else:
            shell.print_error(f"Invalid path index: {index+1}. Must be between 1 and {len(paths)}.")
            return
    else:
        # Select best path
        shell.print("Selecting best solution path...")
        result = select_best_solution_path(paths)
        
        if result.get("success", False):
            best_path = result.get("path")
            shell.print(f"Selected best path: {best_path.get('path_id')}")
            shell.print(f"  Approach: {best_path.get('approach')}")
            shell.print(f"  Confidence: {best_path.get('confidence')}")
            shell.print(f"  Score: {best_path.get('score', 0) * 100:.0f}")
        else:
            shell.print_error(f"Error selecting best path: {result.get('error')}")
            return None
    
    # Save best path for later use
    shell.set_variable("best_path", best_path)
    
    return best_path

def bug_fix_command(shell, args):
    """
    Fix a bug in code.
    
    Usage: bug_fix <file> [--language <language>]
    """
    if not args:
        shell.print_error("No file specified")
        shell.print("Usage: bug_fix <file> [--language <language>]")
        return
    
    file_path = args[0]
    
    # Check if language is specified
    language = "py"
    if len(args) > 1 and args[1] == "--language" and len(args) > 2:
        language = args[2]
    
    # Check if file exists
    if not os.path.exists(file_path):
        shell.print_error(f"File {file_path} not found")
        return
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        code_content = f.read()
    
    # Fix bug
    shell.print(f"Fixing bug in {file_path} (language: {language})...")
    result = fix_bug(code_content, language)
    
    if result.get("success", False):
        bug_id = result.get("bug_id")
        patch_id = result.get("patch_id")
        verification_id = result.get("verification_id")
        verification_result = result.get("verification_result")
        
        shell.print(f"Successfully fixed bug {bug_id}")
        shell.print(f"Patch ID: {patch_id}")
        shell.print(f"Verification ID: {verification_id}")
        shell.print(f"Verification Result: {verification_result}")
        shell.print(f"Duration: {result.get('duration', 0):.2f} seconds")
        
        # Save results for later use
        shell.set_variable("last_bug_id", bug_id)
        shell.set_variable("last_patch_id", patch_id)
        shell.set_variable("last_verification_id", verification_id)
        
        return result
    else:
        shell.print_error(f"Error fixing bug: {result.get('error')}")
        
        # Print partial results if available
        if "bug_id" in result:
            shell.print(f"Bug ID: {result.get('bug_id')}")
            shell.set_variable("last_bug_id", result.get('bug_id'))
        
        if "path" in result:
            path = result.get("path", {})
            shell.print(f"Selected Path: {path.get('path_id')}")
            shell.print(f"  Approach: {path.get('approach')}")
        
        if "patch_id" in result:
            shell.print(f"Patch ID: {result.get('patch_id')}")
            shell.set_variable("last_patch_id", result.get('patch_id'))
        
        return result

def bug_demo_command(shell, args):
    """
    Run the decision tree demo.
    
    Usage: bug_demo
    """
    shell.print("Running decision tree demo...")
    
    # Create sample buggy file
    sample_code = """
def calculate_average(numbers):
    total = 0
    for number in numbers:
        total += number
    return total / len(numbers)  # Bug: Potential division by zero

def process_data(data):
    result = []
    for item in data:
        if item > 0:
            # Bug: Using append incorrectly
            result.append = item * 2
    return result

def recursive_function(n):
    if n <= 0:
        return 1
    # Bug: No base case for n=1, will cause infinite recursion
    return n * recursive_function(n-1)

def get_value(dictionary, key):
    # Bug: No key error handling
    return dictionary[key]

# Bug: Unused import
import os

# Bug: Bare except
try:
    x = 1 / 0
except:
    pass
"""
    
    file_path = "sample_demo.py"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(sample_code)
    
    shell.print(f"Created sample file: {file_path}")
    
    # Step 1: Identify bug
    shell.print("\n===== STEP 1: IDENTIFY BUG =====\n")
    bug_id = bug_identify_command(shell, [file_path])
    
    if not bug_id:
        shell.print_error("Bug identification failed, cannot continue demo")
        return
    
    # Step 2: Generate solution paths
    shell.print("\n===== STEP 2: GENERATE SOLUTION PATHS =====\n")
    paths = bug_generate_paths_command(shell, [bug_id])
    
    if not paths:
        shell.print_error("Solution path generation failed, cannot continue demo")
        return
    
    # Step 3: Select best path
    shell.print("\n===== STEP 3: SELECT BEST PATH =====\n")
    best_path = bug_select_path_command(shell, [])
    
    if not best_path:
        shell.print_error("Path selection failed, cannot continue demo")
        return
    
    # Step 4: Fix bug
    shell.print("\n===== STEP 4: FIX BUG =====\n")
    result = bug_fix_command(shell, [file_path])
    
    if result and result.get("success", False):
        shell.print("\nDemo completed successfully!")
    else:
        shell.print("\nDemo completed with errors.")

def register_commands(shell):
    """
    Register decision tree commands with the shell.
    
    Args:
        shell: Shell instance
    """
    shell.register_command("bug_identify", bug_identify_command, "Identify a bug in code")
    shell.register_command("bug_generate_paths", bug_generate_paths_command, "Generate solution paths for a bug")
    shell.register_command("bug_select_path", bug_select_path_command, "Select the best solution path")
    shell.register_command("bug_fix", bug_fix_command, "Fix a bug in code")
    shell.register_command("bug_demo", bug_demo_command, "Run the decision tree demo")

def register(registry):
    """
    Register decision tree commands with the shell registry.
    This function is called by the shell when loading modules.
    """
    from logging import getLogger
    logger = getLogger("DecisionTreeCommands")
    
    # Register commands
    register_commands(registry)
    
    # Create necessary directories
    import os
    os.makedirs(".triangulum/results", exist_ok=True)
    os.makedirs(".triangulum/patches", exist_ok=True)
    os.makedirs(".triangulum/verification_results", exist_ok=True)
    os.makedirs(".triangulum/logs", exist_ok=True)
    
    logger.info("Decision tree commands registered")
    return True
