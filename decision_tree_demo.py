#!/usr/bin/env python3
"""
Decision Tree Demo

This script demonstrates the use of the decision tree logic for bug fixing.
It provides examples of how to use the various components to identify, fix,
and verify bug fixes.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Import integration module
from decision_tree_integration import (
    DecisionTreeIntegration, identify_bug, generate_solution_paths,
    select_best_solution_path, fix_bug
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("decision_tree_demo.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("DecisionTreeDemo")

# Sample buggy code with various issues
SAMPLE_BUGGY_CODE = """
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

# Sample buggy JavaScript code
SAMPLE_JS_BUGGY_CODE = """
function calculateAverage(numbers) {
    let total = 0;
    for (let i = 0; i < numbers.length; i++) {
        total += numbers[i];
    }
    return total / numbers.length;  // Bug: Potential division by zero
}

function processData(data) {
    const result = [];
    for (let i = 0; i < data.length; i++) {
        if (data[i] > 0) {
            // Bug: Using push incorrectly
            result.push = data[i] * 2;
        }
    }
    return result;
}

// Bug: Console.log left in code
console.log('Debug message');

// Bug: Eval usage
function executeCode(code) {
    return eval(code);
}

// Bug: No error handling
function getValue(obj, key) {
    return obj[key];
}

// Bug: No try-catch
const value = 1 / 0;
"""

def demonstrate_bug_identification():
    """Demonstrate bug identification."""
    print("\n===== DEMONSTRATING BUG IDENTIFICATION =====\n")
    
    # Identify bugs in Python code
    result_py = identify_bug(SAMPLE_BUGGY_CODE, "py")
    
    if result_py.get("success", False):
        bug_id = result_py.get("bug_id")
        bug_info = result_py.get("bug_info", {})
        
        print(f"Identified bug: {bug_id}")
        print(f"Bug type: {bug_info.get('bug_type')}")
        print(f"Severity: {bug_info.get('severity')}")
        print(f"Complexity: {bug_info.get('complexity')}")
        
        # Print analysis summary
        analysis = bug_info.get("analysis", {})
        summary = analysis.get("summary", {})
        
        if summary:
            print("\nAnalysis Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
        
        # Print issue counts
        issue_count = summary.get("issue_count", {})
        if issue_count:
            print("\nIssue Counts:")
            for issue_type, count in issue_count.items():
                print(f"  {issue_type}: {count}")
    else:
        print(f"Error identifying bug: {result_py.get('error')}")
    
    # Identify bugs in JavaScript code
    result_js = identify_bug(SAMPLE_JS_BUGGY_CODE, "js")
    
    if result_js.get("success", False):
        bug_id = result_js.get("bug_id")
        bug_info = result_js.get("bug_info", {})
        
        print(f"\nIdentified bug: {bug_id}")
        print(f"Bug type: {bug_info.get('bug_type')}")
        print(f"Severity: {bug_info.get('severity')}")
        print(f"Complexity: {bug_info.get('complexity')}")
    else:
        print(f"Error identifying bug: {result_js.get('error')}")

def demonstrate_solution_path_generation(bug_id):
    """
    Demonstrate solution path generation.
    
    Args:
        bug_id: ID of the bug to generate solution paths for
    """
    print("\n===== DEMONSTRATING SOLUTION PATH GENERATION =====\n")
    
    # Generate solution paths
    result = generate_solution_paths(bug_id)
    
    if result.get("success", False):
        paths = result.get("paths", [])
        
        print(f"Generated {len(paths)} solution paths for bug {bug_id}")
        
        # Print details of each path
        for i, path in enumerate(paths):
            print(f"\nPath {i+1} - {path.get('path_id')}")
            print(f"  Approach: {path.get('approach')}")
            print(f"  Confidence: {path.get('confidence')}")
            print(f"  Estimated Time: {path.get('estimated_time')} minutes")
            print(f"  Estimated Complexity: {path.get('estimated_complexity')}")
            print(f"  Estimated Success Rate: {path.get('estimated_success_rate', 0) * 100:.1f}%")
            print(f"  Score: {path.get('score', 0) * 100:.1f}")
            
            # Print steps
            steps = path.get("steps", [])
            if steps:
                print("\n  Steps:")
                for j, step in enumerate(steps):
                    print(f"    {j+1}. {step.get('description')}")
        
        # Select best path
        best_path_result = select_best_solution_path(paths)
        
        if best_path_result.get("success", False):
            best_path = best_path_result.get("path", {})
            print(f"\nSelected best path: {best_path.get('path_id')}")
            print(f"  Approach: {best_path.get('approach')}")
            print(f"  Score: {best_path.get('score', 0) * 100:.1f}")
        else:
            print(f"Error selecting best path: {best_path_result.get('error')}")
    else:
        print(f"Error generating solution paths: {result.get('error')}")
    
    return result.get("paths", [])

def demonstrate_full_bug_fixing(code, language):
    """
    Demonstrate the full bug fixing process.
    
    Args:
        code: Code to fix
        language: Programming language of the code
    """
    print("\n===== DEMONSTRATING FULL BUG FIXING PROCESS =====\n")
    
    # Create sample file with buggy code
    file_path = f"sample_buggy.{language}"
    with open(file_path, 'w') as f:
        f.write(code)
    
    print(f"Created sample file: {file_path}")
    
    # Fix the bug
    result = fix_bug(code, language)
    
    if result.get("success", False):
        bug_id = result.get("bug_id")
        patch_id = result.get("patch_id")
        verification_id = result.get("verification_id")
        verification_result = result.get("verification_result")
        
        print(f"Successfully fixed bug {bug_id}")
        print(f"Patch ID: {patch_id}")
        print(f"Verification ID: {verification_id}")
        print(f"Verification Result: {verification_result}")
        print(f"Duration: {result.get('duration', 0):.2f} seconds")
    else:
        print(f"Error fixing bug: {result.get('error')}")
        
        # Print partial results if available
        if "bug_id" in result:
            print(f"Bug ID: {result.get('bug_id')}")
        
        if "path" in result:
            path = result.get("path", {})
            print(f"Selected Path: {path.get('path_id')}")
            print(f"  Approach: {path.get('approach')}")
        
        if "patch_id" in result:
            print(f"Patch ID: {result.get('patch_id')}")

def demonstrate_programmatic_usage():
    """Demonstrate programmatic usage of the decision tree integration."""
    print("\n===== DEMONSTRATING PROGRAMMATIC USAGE =====\n")
    
    # Initialize integration
    integration = DecisionTreeIntegration()
    
    # Identify bug
    bug_info = integration.identify_bug(SAMPLE_BUGGY_CODE, "py")
    bug_id = bug_info["bug_id"]
    
    print(f"Identified bug: {bug_id}")
    
    # Generate solution paths
    paths = integration.generate_solution_paths(bug_info)
    
    print(f"Generated {len(paths)} solution paths")
    
    # Select best path
    best_path = integration.select_best_solution_path(paths)
    
    if best_path:
        path_id = best_path["path_id"]
        print(f"Selected best path: {path_id}")
        
        # Define changes
        changes = [{
            "file_path": "sample_fixed.py",
            "operation": "add",
            "content": """
def calculate_average(numbers):
    if not numbers:
        return 0
    total = 0
    for number in numbers:
        total += number
    return total / len(numbers)

def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

def recursive_function(n):
    if n <= 0:
        return 1
    if n == 1:
        return 1
    return n * recursive_function(n-1)

def get_value(dictionary, key):
    return dictionary.get(key)

try:
    x = 1 / 0
except ZeroDivisionError:
    pass
"""
        }]
        
        # Generate patch
        patch_result = integration.generate_patch(bug_id, path_id, best_path, changes)
        
        if patch_result.get("success", False):
            patch_id = patch_result.get("patch_id")
            print(f"Generated patch: {patch_id}")
            
            # Apply patch
            apply_result = integration.apply_patch(patch_id)
            
            if apply_result.get("success", False):
                print("Applied patch successfully")
                
                # Verify fix
                verify_result = integration.verify_fix(patch_id, bug_id)
                
                if verify_result.get("success", False):
                    verification_id = verify_result.get("verification_id")
                    results = verify_result.get("results", {})
                    overall_result = results.get("overall_result")
                    
                    print(f"Verification ID: {verification_id}")
                    print(f"Overall Result: {overall_result}")
                else:
                    print(f"Error verifying fix: {verify_result.get('error')}")
            else:
                print(f"Error applying patch: {apply_result.get('error')}")
        else:
            print(f"Error generating patch: {patch_result.get('error')}")
    else:
        print("No suitable solution path found")

def main():
    """Main function."""
    print("Decision Tree Demo\n")
    
    # Demonstrate bug identification
    demonstrate_bug_identification()
    
    # Get the bug ID from the identification demo
    results_dir = Path(".triangulum/results")
    if results_dir.exists():
        # Find the most recent bug info file
        bug_files = list(results_dir.glob("bug-*.json"))
        if bug_files:
            bug_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            bug_id = bug_files[0].stem
            
            # Demonstrate solution path generation
            paths = demonstrate_solution_path_generation(bug_id)
            
            # Only demonstrate full bug fixing if paths were generated
            if paths:
                demonstrate_full_bug_fixing(SAMPLE_BUGGY_CODE, "py")
    
    # Demonstrate programmatic usage
    demonstrate_programmatic_usage()

if __name__ == "__main__":
    main()
