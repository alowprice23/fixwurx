#!/usr/bin/env python3
"""
Scope Filter Commands Module

This module registers command handlers for the scope filtering system within the shell environment,
enabling intelligent filtering and scope reduction for code analysis.
"""

import os
import sys
import json
import logging
import argparse
import shlex
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("ScopeFilterCommands")

def register_scope_commands(registry):
    """
    Register scope filter command handlers with the component registry.
    
    Args:
        registry: Component registry instance
    """
    try:
        # Register command handlers
        registry.register_command_handler("filter", filter_command, "scope_filter")
        registry.register_command_handler("patterns", patterns_command, "scope_filter")
        registry.register_command_handler("entropy", entropy_command, "scope_filter")
        registry.register_command_handler("analyze-content", analyze_content_command, "scope_filter")
        
        # Register prefixed versions
        registry.register_command_handler("scope:filter", filter_command, "scope_filter")
        registry.register_command_handler("scope:patterns", patterns_command, "scope_filter")
        registry.register_command_handler("scope:entropy", entropy_command, "scope_filter")
        registry.register_command_handler("scope:analyze", analyze_content_command, "scope_filter")
        
        # Register aliases
        registry.register_alias("file-filter", "scope:filter")
        registry.register_alias("find-patterns", "scope:patterns")
        registry.register_alias("entropy-analysis", "scope:entropy")
        
        logger.info("Scope filter commands registered")
    except Exception as e:
        logger.error(f"Error registering scope filter commands: {e}")

def filter_command(args: str) -> int:
    """
    Filter files in a directory based on various criteria.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Filter files in a directory")
    parser.add_argument("directory", help="Directory to filter")
    parser.add_argument("--extensions", "-e", nargs="+", help="File extensions to include")
    parser.add_argument("--exclude-extensions", "-x", nargs="+", help="File extensions to exclude")
    parser.add_argument("--patterns", "-p", nargs="+", help="File name patterns to include")
    parser.add_argument("--exclude-patterns", "-xp", nargs="+", help="File name patterns to exclude")
    parser.add_argument("--content", "-c", nargs="+", help="Content patterns to match")
    parser.add_argument("--check-content", action="store_true", help="Check file content for matches")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import scope filter module
    try:
        from scope_filter import filter_directory
    except ImportError:
        print("Error: Scope filter module not available")
        return 1
    
    # Create configuration
    config = {}
    
    if cmd_args.extensions:
        config["include_extensions"] = [f".{ext.lstrip('.')}" for ext in cmd_args.extensions]
    
    if cmd_args.exclude_extensions:
        config["exclude_extensions"] = [f".{ext.lstrip('.')}" for ext in cmd_args.exclude_extensions]
    
    if cmd_args.patterns:
        config["include_patterns"] = cmd_args.patterns
    
    if cmd_args.exclude_patterns:
        config["exclude_patterns"] = cmd_args.exclude_patterns
    
    if cmd_args.content:
        config["content_patterns"] = cmd_args.content
    
    # Filter directory
    print(f"Filtering directory: {cmd_args.directory}")
    result = filter_directory(cmd_args.directory, config)
    
    if result.get("success", False):
        matching_files = result.get("matching_files", [])
        file_count = result.get("file_count", 0)
        
        print(f"\nFound {file_count} matching files:")
        for i, file in enumerate(matching_files[:10], 1):  # Show first 10 files
            print(f"  {i}. {file}")
        
        if len(matching_files) > 10:
            print(f"  ... and {len(matching_files) - 10} more")
        
        # Save results to file if requested
        if cmd_args.output:
            try:
                with open(cmd_args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to {cmd_args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")
                return 1
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

def patterns_command(args: str) -> int:
    """
    Detect bug patterns in files within a directory.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Detect bug patterns in files")
    parser.add_argument("directory", help="Directory to analyze")
    parser.add_argument("--extensions", "-e", nargs="+", help="File extensions to include")
    parser.add_argument("--exclude-extensions", "-x", nargs="+", help="File extensions to exclude")
    parser.add_argument("--patterns", "-p", nargs="+", help="File name patterns to include")
    parser.add_argument("--exclude-patterns", "-xp", nargs="+", help="File name patterns to exclude")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import scope filter module
    try:
        from scope_filter import detect_patterns
    except ImportError:
        print("Error: Scope filter module not available")
        return 1
    
    # Create configuration
    config = {}
    
    if cmd_args.extensions:
        config["include_extensions"] = [f".{ext.lstrip('.')}" for ext in cmd_args.extensions]
    
    if cmd_args.exclude_extensions:
        config["exclude_extensions"] = [f".{ext.lstrip('.')}" for ext in cmd_args.exclude_extensions]
    
    if cmd_args.patterns:
        config["include_patterns"] = cmd_args.patterns
    
    if cmd_args.exclude_patterns:
        config["exclude_patterns"] = cmd_args.exclude_patterns
    
    # Detect patterns
    print(f"Detecting bug patterns in directory: {cmd_args.directory}")
    result = detect_patterns(cmd_args.directory, config)
    
    if result.get("success", False):
        file_count = result.get("file_count", 0)
        high_severity_count = result.get("high_severity_count", 0)
        
        print(f"\nAnalyzed {file_count} files")
        print(f"Found {high_severity_count} high severity files")
        
        # Show high severity files
        high_severity_files = result.get("high_severity_files", [])
        if high_severity_files:
            print("\nHigh severity files:")
            for i, file in enumerate(high_severity_files[:10], 1):  # Show first 10 files
                print(f"  {i}. {file}")
            
            if len(high_severity_files) > 10:
                print(f"  ... and {len(high_severity_files) - 10} more")
        
        # Save results to file if requested
        if cmd_args.output:
            try:
                with open(cmd_args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to {cmd_args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")
                return 1
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

def entropy_command(args: str) -> int:
    """
    Perform entropy-based analysis on files within a directory.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Perform entropy analysis on files")
    parser.add_argument("directory", help="Directory to analyze")
    parser.add_argument("--extensions", "-e", nargs="+", help="File extensions to include")
    parser.add_argument("--exclude-extensions", "-x", nargs="+", help="File extensions to exclude")
    parser.add_argument("--min-entropy", type=float, default=0.0, help="Minimum entropy threshold")
    parser.add_argument("--max-entropy", type=float, default=1.0, help="Maximum entropy threshold")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import scope filter module
    try:
        from scope_filter import entropy_analysis
    except ImportError:
        print("Error: Scope filter module not available")
        return 1
    
    # Create configuration
    config = {}
    
    if cmd_args.extensions:
        config["include_extensions"] = [f".{ext.lstrip('.')}" for ext in cmd_args.extensions]
    
    if cmd_args.exclude_extensions:
        config["exclude_extensions"] = [f".{ext.lstrip('.')}" for ext in cmd_args.exclude_extensions]
    
    # Perform entropy analysis
    print(f"Performing entropy analysis on directory: {cmd_args.directory}")
    result = entropy_analysis(cmd_args.directory, config)
    
    if result.get("success", False):
        file_count = result.get("file_count", 0)
        avg_entropy = result.get("average_entropy", 0)
        max_entropy = result.get("max_entropy", 0)
        min_entropy = result.get("min_entropy", 0)
        high_entropy_count = result.get("high_entropy_count", 0)
        
        print(f"\nAnalyzed {file_count} files")
        print(f"Average entropy: {avg_entropy:.4f}")
        print(f"Maximum entropy: {max_entropy:.4f}")
        print(f"Minimum entropy: {min_entropy:.4f}")
        print(f"Files with high entropy (> 0.7): {high_entropy_count}")
        
        # Show high entropy files
        high_entropy_files = result.get("high_entropy_files", [])
        if high_entropy_files:
            print("\nHigh entropy files:")
            for i, file in enumerate(high_entropy_files[:10], 1):  # Show first 10 files
                print(f"  {i}. {file}")
            
            if len(high_entropy_files) > 10:
                print(f"  ... and {len(high_entropy_files) - 10} more")
        
        # Save results to file if requested
        if cmd_args.output:
            try:
                with open(cmd_args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to {cmd_args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")
                return 1
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

def analyze_content_command(args: str) -> int:
    """
    Perform content analysis on files within a directory.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Perform content analysis on files")
    parser.add_argument("directory", help="Directory to analyze")
    parser.add_argument("--extensions", "-e", nargs="+", help="File extensions to include")
    parser.add_argument("--exclude-extensions", "-x", nargs="+", help="File extensions to exclude")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import scope filter module
    try:
        from scope_filter import analyze_content
    except ImportError:
        print("Error: Scope filter module not available")
        return 1
    
    # Create configuration
    config = {}
    
    if cmd_args.extensions:
        config["include_extensions"] = [f".{ext.lstrip('.')}" for ext in cmd_args.extensions]
    
    if cmd_args.exclude_extensions:
        config["exclude_extensions"] = [f".{ext.lstrip('.')}" for ext in cmd_args.exclude_extensions]
    
    # Perform content analysis
    print(f"Performing content analysis on directory: {cmd_args.directory}")
    result = analyze_content(cmd_args.directory, config)
    
    if result.get("success", False):
        file_count = result.get("file_count", 0)
        high_complexity_count = result.get("high_complexity_count", 0)
        
        print(f"\nAnalyzed {file_count} files")
        print(f"Files with high complexity: {high_complexity_count}")
        
        # Show high complexity files
        high_complexity_files = result.get("high_complexity_files", [])
        if high_complexity_files:
            print("\nHigh complexity files:")
            for i, file in enumerate(high_complexity_files[:10], 1):  # Show first 10 files
                print(f"  {i}. {file}")
            
            if len(high_complexity_files) > 10:
                print(f"  ... and {len(high_complexity_files) - 10} more")
        
        # Save results to file if requested
        if cmd_args.output:
            try:
                with open(cmd_args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to {cmd_args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")
                return 1
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    print("Scope Filter Commands Module")
    print("This module should be imported by the shell environment")
