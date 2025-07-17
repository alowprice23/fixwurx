#!/usr/bin/env python3
"""
Test Script for Scope Filter Module

This script tests the functionality of the scope filter module by running various filtering operations
on the current directory.
"""

import os
import sys
import json
from pathlib import Path

# Ensure the scope_filter module is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from scope_filter import ScopeFilter, filter_directory, detect_patterns, analyze_content, entropy_analysis
except ImportError:
    print("Error: Could not import scope_filter module")
    sys.exit(1)

def test_extension_filtering():
    """Test filtering by file extension."""
    print("\n=== Testing Extension Filtering ===")
    
    # Configure filter to include only Python files
    config = {
        "include_extensions": [".py"]
    }
    
    # Create filter
    filter = ScopeFilter(config)
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find Python files
    python_files = filter.find_files(current_dir, recursive=False)
    
    print(f"Found {len(python_files)} Python files in current directory")
    for i, file in enumerate(python_files[:5], 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    if len(python_files) > 5:
        print(f"  ... and {len(python_files) - 5} more")
    
    # Configure filter to exclude Python files
    config = {
        "exclude_extensions": [".py"]
    }
    
    # Create filter
    filter = ScopeFilter(config)
    
    # Find non-Python files
    non_python_files = filter.find_files(current_dir, recursive=False)
    
    print(f"\nFound {len(non_python_files)} non-Python files in current directory")
    for i, file in enumerate(non_python_files[:5], 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    if len(non_python_files) > 5:
        print(f"  ... and {len(non_python_files) - 5} more")
    
    return True

def test_pattern_filtering():
    """Test filtering by file name pattern."""
    print("\n=== Testing Pattern Filtering ===")
    
    # Configure filter to include only files with "test" in the name
    config = {
        "include_patterns": ["*test*"]
    }
    
    # Create filter
    filter = ScopeFilter(config)
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find test files
    test_files = filter.find_files(current_dir, recursive=False)
    
    print(f"Found {len(test_files)} test files in current directory")
    for i, file in enumerate(test_files[:5], 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    if len(test_files) > 5:
        print(f"  ... and {len(test_files) - 5} more")
    
    # Configure filter to exclude files with "test" in the name
    config = {
        "exclude_patterns": ["*test*"]
    }
    
    # Create filter
    filter = ScopeFilter(config)
    
    # Find non-test files
    non_test_files = filter.find_files(current_dir, recursive=False)
    
    print(f"\nFound {len(non_test_files)} non-test files in current directory")
    for i, file in enumerate(non_test_files[:5], 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    if len(non_test_files) > 5:
        print(f"  ... and {len(non_test_files) - 5} more")
    
    return True

def test_entropy_analysis():
    """Test entropy analysis."""
    print("\n=== Testing Entropy Analysis ===")
    
    # Configure filter to include only Python files
    config = {
        "include_extensions": [".py"]
    }
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Perform entropy analysis
    result = entropy_analysis(current_dir, config)
    
    if result.get("success", False):
        file_count = result.get("file_count", 0)
        avg_entropy = result.get("average_entropy", 0)
        max_entropy = result.get("max_entropy", 0)
        min_entropy = result.get("min_entropy", 0)
        high_entropy_files = result.get("high_entropy_files", [])
        
        print(f"Analyzed {file_count} Python files")
        print(f"Average entropy: {avg_entropy:.4f}")
        print(f"Maximum entropy: {max_entropy:.4f}")
        print(f"Minimum entropy: {min_entropy:.4f}")
        print(f"Files with high entropy: {len(high_entropy_files)}")
        
        if high_entropy_files:
            print("\nTop 5 high entropy files:")
            for i, file in enumerate(high_entropy_files[:5], 1):
                entropy = result.get("results", {}).get(file, 0)
                print(f"  {i}. {os.path.basename(file)}: {entropy:.4f}")
        
        # Get highest entropy file
        highest_entropy_file = max(result.get("results", {}).items(), key=lambda x: x[1], default=(None, 0))
        if highest_entropy_file[0]:
            print(f"\nHighest entropy file: {os.path.basename(highest_entropy_file[0])}: {highest_entropy_file[1]:.4f}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return False
    
    return True

def test_bug_pattern_detection():
    """Test bug pattern detection."""
    print("\n=== Testing Bug Pattern Detection ===")
    
    # Configure filter to include only Python files
    config = {
        "include_extensions": [".py"]
    }
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Perform pattern detection
    result = detect_patterns(current_dir, config)
    
    if result.get("success", False):
        file_count = result.get("file_count", 0)
        high_severity_files = result.get("high_severity_files", [])
        
        print(f"Analyzed {file_count} Python files")
        print(f"Files with high severity patterns: {len(high_severity_files)}")
        
        if high_severity_files:
            print("\nHigh severity files:")
            for i, file in enumerate(high_severity_files[:5], 1):
                pattern_result = result.get("results", {}).get(file, {})
                print(f"  {i}. {os.path.basename(file)}")
                print(f"     Patterns: {pattern_result.get('patterns_detected', 0)}")
                print(f"     Types: {', '.join(pattern_result.get('pattern_types', []))}")
        
        # Get file with most patterns
        most_patterns_file = None
        max_patterns = 0
        
        for file, file_result in result.get("results", {}).items():
            patterns = file_result.get("patterns_detected", 0)
            if patterns > max_patterns:
                max_patterns = patterns
                most_patterns_file = file
        
        if most_patterns_file:
            pattern_result = result.get("results", {}).get(most_patterns_file, {})
            print(f"\nFile with most patterns: {os.path.basename(most_patterns_file)}")
            print(f"  Patterns: {pattern_result.get('patterns_detected', 0)}")
            print(f"  Types: {', '.join(pattern_result.get('pattern_types', []))}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return False
    
    return True

def test_content_analysis():
    """Test content analysis."""
    print("\n=== Testing Content Analysis ===")
    
    # Configure filter to include only Python files
    config = {
        "include_extensions": [".py"]
    }
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Perform content analysis
    result = analyze_content(current_dir, config)
    
    if result.get("success", False):
        file_count = result.get("file_count", 0)
        high_complexity_files = result.get("high_complexity_files", [])
        
        print(f"Analyzed {file_count} Python files")
        print(f"Files with high complexity: {len(high_complexity_files)}")
        
        if high_complexity_files:
            print("\nHigh complexity files:")
            for i, file in enumerate(high_complexity_files[:5], 1):
                content_result = result.get("results", {}).get(file, {})
                print(f"  {i}. {os.path.basename(file)}")
                print(f"     Lines: {content_result.get('lines', 0)}")
                print(f"     Code lines: {content_result.get('code_lines', 0)}")
                print(f"     Complexity: {content_result.get('complexity_estimate', 'unknown')}")
        
        # Get file with most code lines
        most_lines_file = None
        max_lines = 0
        
        for file, file_result in result.get("results", {}).items():
            lines = file_result.get("code_lines", 0)
            if lines > max_lines:
                max_lines = lines
                most_lines_file = file
        
        if most_lines_file:
            content_result = result.get("results", {}).get(most_lines_file, {})
            print(f"\nLargest file: {os.path.basename(most_lines_file)}")
            print(f"  Total lines: {content_result.get('lines', 0)}")
            print(f"  Code lines: {content_result.get('code_lines', 0)}")
            print(f"  Comment lines: {content_result.get('comment_lines', 0)}")
            print(f"  Blank lines: {content_result.get('blank_lines', 0)}")
            print(f"  Complexity: {content_result.get('complexity_estimate', 'unknown')}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return False
    
    return True

def main():
    """Main function."""
    print("=== Scope Filter Test Suite ===")
    
    # Run tests
    tests = [
        ("Extension filtering", test_extension_filtering),
        ("Pattern filtering", test_pattern_filtering),
        ("Entropy analysis", test_entropy_analysis),
        ("Bug pattern detection", test_bug_pattern_detection),
        ("Content analysis", test_content_analysis)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            print(f"\nRunning test: {name}")
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"Error running test: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n=== Test Summary ===")
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        if result:
            passed += 1
        else:
            failed += 1
        
        print(f"{name}: {status}")
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
