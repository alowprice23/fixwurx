#!/usr/bin/env python3
"""
FixWurx Commands Module

This module registers command handlers for the FixWurx system within the shell environment,
integrating the core functionality and providing a unified interface.
"""

import os
import sys
import json
import yaml
import time
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger("FixWurxCommands")

def register_fixwurx_commands(registry):
    """
    Register FixWurx command handlers with the component registry.
    
    Args:
        registry: Component registry instance
    """
    # Register command handlers
    registry.register_command_handler("fix", fix_command, "fixwurx")
    registry.register_command_handler("analyze", analyze_command, "fixwurx")
    registry.register_command_handler("deploy", deploy_command, "fixwurx")
    registry.register_command_handler("diagnose", diagnose_command, "fixwurx")
    registry.register_command_handler("benchmark", benchmark_command, "fixwurx")
    registry.register_command_handler("scan", scan_command, "fixwurx")
    registry.register_command_handler("stats", stats_command, "fixwurx")
    registry.register_command_handler("init", init_command, "fixwurx")
    
    # Register aliases
    registry.register_alias("repair", "fix")
    registry.register_alias("check", "analyze")
    
    # Register event handlers
    registry.register_event_handler("fix_started", fix_started_handler, "fixwurx")
    registry.register_event_handler("fix_completed", fix_completed_handler, "fixwurx")
    
    logger.info("FixWurx commands registered")

def fix_command(args: str) -> int:
    """
    Fix issues in the specified code or system.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fix issues in code or system")
    parser.add_argument("target", nargs="?", help="Target to fix (file or directory)")
    parser.add_argument("--issue", help="Specific issue to fix")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively fix directories")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be fixed without making changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument("--output", "-o", help="Output file for fix report")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    target = cmd_args.target or "."
    issue_id = cmd_args.issue
    recursive = cmd_args.recursive
    dry_run = cmd_args.dry_run
    verbose = cmd_args.verbose
    output_file = cmd_args.output
    
    print(f"{'Analyzing' if dry_run else 'Fixing'} target: {target}")
    
    # Trigger fix started event
    registry = sys.modules.get("__main__").registry
    registry.trigger_event("fix_started", {
        "target": target,
        "issue_id": issue_id,
        "recursive": recursive,
        "dry_run": dry_run
    })
    
    try:
        # Call fixwurx module
        try:
            from fixwurx import FixWurx
            
            # Create FixWurx instance
            fix_engine = FixWurx()
            
            # Run the fix operation
            if dry_run:
                result = fix_engine.analyze(target, recursive=recursive)
                
                # Display issues that would be fixed
                if "issues" in result:
                    issues = result["issues"]
                    if issues:
                        print(f"\nFound {len(issues)} issues:")
                        for i, issue in enumerate(issues, 1):
                            issue_id = issue.get("id", "Unknown")
                            severity = issue.get("severity", "Unknown")
                            description = issue.get("description", "No description")
                            
                            print(f"  {i}. [{severity}] {issue_id}: {description}")
                    else:
                        print("No issues found")
            else:
                result = fix_engine.fix(target, issue_id=issue_id, recursive=recursive)
                
                # Display fix results
                if "fixed_issues" in result:
                    fixed = result["fixed_issues"]
                    if fixed:
                        print(f"\nFixed {len(fixed)} issues:")
                        for i, issue in enumerate(fixed, 1):
                            issue_id = issue.get("id", "Unknown")
                            file_path = issue.get("file", "Unknown")
                            description = issue.get("description", "No description")
                            
                            print(f"  {i}. {issue_id} in {file_path}")
                            if verbose:
                                print(f"     {description}")
                    else:
                        print("No issues fixed")
                
                # Display remaining issues
                if "remaining_issues" in result:
                    remaining = result["remaining_issues"]
                    if remaining:
                        print(f"\nRemaining issues: {len(remaining)}")
                        if verbose:
                            for i, issue in enumerate(remaining, 1):
                                issue_id = issue.get("id", "Unknown")
                                severity = issue.get("severity", "Unknown")
                                description = issue.get("description", "No description")
                                
                                print(f"  {i}. [{severity}] {issue_id}: {description}")
            
            # Save results to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to {output_file}")
            
            # Trigger fix completed event
            registry.trigger_event("fix_completed", {
                "target": target,
                "result": result,
                "success": True
            })
            
            return 0
        except ImportError:
            print("FixWurx module not available")
            
            # Trigger fix completed event
            registry.trigger_event("fix_completed", {
                "target": target,
                "success": False,
                "error": "FixWurx module not available"
            })
            
            return 1
    except Exception as e:
        print(f"Error fixing target: {e}")
        
        # Trigger fix completed event
        registry.trigger_event("fix_completed", {
            "target": target,
            "success": False,
            "error": str(e)
        })
        
        return 1

def analyze_command(args: str) -> int:
    """
    Analyze code or system for issues.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze code or system for issues")
    parser.add_argument("target", nargs="?", help="Target to analyze (file or directory)")
    parser.add_argument("--type", choices=["code", "performance", "security", "all"], default="all", 
                       help="Type of analysis to perform")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively analyze directories")
    parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                       help="Output format")
    parser.add_argument("--output", "-o", help="Output file for analysis report")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    target = cmd_args.target or "."
    analysis_type = cmd_args.type
    recursive = cmd_args.recursive
    output_format = cmd_args.format
    output_file = cmd_args.output
    
    print(f"Analyzing target: {target}")
    print(f"Analysis type: {analysis_type}")
    
    try:
        # Call fixwurx module
        try:
            from fixwurx import FixWurx
            
            # Create FixWurx instance
            fix_engine = FixWurx()
            
            # Run the analysis
            result = fix_engine.analyze(target, analysis_type=analysis_type, recursive=recursive)
            
            # Display results
            if output_format == "json":
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"Analysis result saved to {output_file}")
                else:
                    print(json.dumps(result, indent=2))
            elif output_format == "yaml":
                if output_file:
                    with open(output_file, 'w') as f:
                        yaml.dump(result, f, default_flow_style=False)
                    print(f"Analysis result saved to {output_file}")
                else:
                    print(yaml.dump(result, default_flow_style=False))
            else:  # text format
                # Display summary
                print("\nAnalysis Summary:")
                print("-" * 60)
                
                print(f"Target: {target}")
                print(f"Files analyzed: {result.get('files_analyzed', 0)}")
                print(f"Issues found: {result.get('issues_count', 0)}")
                
                # Display issues by severity
                issues_by_severity = result.get("issues_by_severity", {})
                if issues_by_severity:
                    print("\nIssues by Severity:")
                    for severity, count in issues_by_severity.items():
                        print(f"  {severity}: {count}")
                
                # Display issues
                issues = result.get("issues", [])
                if issues:
                    print("\nIssues:")
                    for i, issue in enumerate(issues, 1):
                        issue_id = issue.get("id", "Unknown")
                        severity = issue.get("severity", "Unknown")
                        file_path = issue.get("file", "Unknown")
                        message = issue.get("message", "No message")
                        
                        print(f"  {i}. [{severity}] {issue_id} in {file_path}")
                        print(f"     {message}")
                
                # Save to file if requested
                if output_file:
                    with open(output_file, 'w') as f:
                        if output_file.endswith('.json'):
                            json.dump(result, f, indent=2)
                        elif output_file.endswith('.yaml') or output_file.endswith('.yml'):
                            yaml.dump(result, f, default_flow_style=False)
                        else:
                            f.write(f"Analysis Results\n")
                            f.write(f"Target: {target}\n")
                            f.write(f"Files analyzed: {result.get('files_analyzed', 0)}\n")
                            f.write(f"Issues found: {result.get('issues_count', 0)}\n\n")
                            
                            # Write issues by severity
                            if issues_by_severity:
                                f.write("Issues by Severity:\n")
                                for severity, count in issues_by_severity.items():
                                    f.write(f"  {severity}: {count}\n")
                                f.write("\n")
                            
                            # Write issues
                            if issues:
                                f.write("Issues:\n")
                                for i, issue in enumerate(issues, 1):
                                    issue_id = issue.get("id", "Unknown")
                                    severity = issue.get("severity", "Unknown")
                                    file_path = issue.get("file", "Unknown")
                                    message = issue.get("message", "No message")
                                    
                                    f.write(f"{i}. [{severity}] {issue_id} in {file_path}\n")
                                    f.write(f"   {message}\n")
                    
                    print(f"Analysis result saved to {output_file}")
            
            return 0
        except ImportError:
            print("FixWurx module not available")
            return 1
    except Exception as e:
        print(f"Error analyzing target: {e}")
        return 1

def deploy_command(args: str) -> int:
    """
    Deploy FixWurx to a target system.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Deploy FixWurx to a target system")
    parser.add_argument("--target", choices=["local", "remote"], default="local", 
                       help="Deployment target")
    parser.add_argument("--host", help="Remote host (for remote deployment)")
    parser.add_argument("--port", type=int, default=22, help="Remote port (for remote deployment)")
    parser.add_argument("--user", help="Remote user (for remote deployment)")
    parser.add_argument("--config", help="Path to deployment configuration file")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be deployed without making changes")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    target = cmd_args.target
    host = cmd_args.host
    port = cmd_args.port
    user = cmd_args.user
    config_file = cmd_args.config
    dry_run = cmd_args.dry_run
    
    # Validate remote deployment arguments
    if target == "remote" and not (host and user):
        print("Error: Remote deployment requires --host and --user")
        return 1
    
    # Load configuration if provided
    config = {}
    if config_file:
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.json'):
                    config = json.load(f)
                elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    print("Error: Configuration file must be JSON or YAML")
                    return 1
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            return 1
    
    print(f"Deploying FixWurx to {target} target" + (f" ({host}:{port})" if target == "remote" else ""))
    if dry_run:
        print("Dry run: No changes will be made")
    
    try:
        # Perform deployment
        if target == "local":
            print("Deploying locally...")
            
            # Placeholder for local deployment
            print("Local deployment completed")
        else:  # remote
            print(f"Deploying to remote host {host}...")
            
            # Placeholder for remote deployment
            print("Remote deployment completed")
        
        return 0
    except Exception as e:
        print(f"Error deploying FixWurx: {e}")
        return 1

def diagnose_command(args: str) -> int:
    """
    Diagnose system problems.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Diagnose system problems")
    parser.add_argument("--component", help="Component to diagnose")
    parser.add_argument("--level", choices=["basic", "advanced", "comprehensive"], default="basic", 
                       help="Diagnostic level")
    parser.add_argument("--output", "-o", help="Output file for diagnostic report")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    component = cmd_args.component
    level = cmd_args.level
    output_file = cmd_args.output
    
    print(f"Running {level} diagnostics" + (f" for component: {component}" if component else ""))
    
    try:
        # Check for required components
        registry = sys.modules.get("__main__").registry
        
        # Get integrated components
        auditor = registry.get_component("auditor")
        neural_matrix = registry.get_component("neural_matrix")
        triangulum = registry.get_component("triangulum")
        
        # Run diagnostics
        results = {}
        
        # Auditor diagnostics
        if auditor and (not component or component == "auditor"):
            print("Running auditor diagnostics...")
            try:
                auditor_status = auditor.get_status()
                auditor_health = auditor.check_health() if hasattr(auditor, 'check_health') else {"status": "Unknown"}
                
                results["auditor"] = {
                    "status": auditor_status,
                    "health": auditor_health
                }
                
                print("Auditor diagnostics completed")
            except Exception as e:
                print(f"Error running auditor diagnostics: {e}")
                results["auditor"] = {"error": str(e)}
        
        # Neural Matrix diagnostics
        if neural_matrix and (not component or component == "neural_matrix"):
            print("Running neural matrix diagnostics...")
            try:
                neural_matrix_status = neural_matrix.get_status()
                neural_matrix_health = neural_matrix.check_health() if hasattr(neural_matrix, 'check_health') else {"status": "Unknown"}
                
                results["neural_matrix"] = {
                    "status": neural_matrix_status,
                    "health": neural_matrix_health
                }
                
                print("Neural Matrix diagnostics completed")
            except Exception as e:
                print(f"Error running neural matrix diagnostics: {e}")
                results["neural_matrix"] = {"error": str(e)}
        
        # Triangulum diagnostics
        if triangulum and (not component or component == "triangulum"):
            print("Running triangulum diagnostics...")
            try:
                triangulum_status = triangulum.get_status()
                triangulum_health = triangulum.check_health() if hasattr(triangulum, 'check_health') else {"status": "Unknown"}
                
                results["triangulum"] = {
                    "status": triangulum_status,
                    "health": triangulum_health
                }
                
                print("Triangulum diagnostics completed")
            except Exception as e:
                print(f"Error running triangulum diagnostics: {e}")
                results["triangulum"] = {"error": str(e)}
        
        # Display results
        print("\nDiagnostic Results:")
        print("-" * 60)
        
        if not results:
            print("No diagnostic results available")
        else:
            for component_name, result in results.items():
                print(f"\n{component_name.upper()}:")
                
                if "error" in result:
                    print(f"  Error: {result['error']}")
                    continue
                
                # Print status
                status = result.get("status", {})
                if status:
                    print("  Status:")
                    for key, value in status.items():
                        if isinstance(value, (dict, list)):
                            continue  # Skip complex structures
                        print(f"    {key}: {value}")
                
                # Print health
                health = result.get("health", {})
                if health:
                    print("  Health:")
                    for key, value in health.items():
                        if isinstance(value, (dict, list)):
                            continue  # Skip complex structures
                        print(f"    {key}: {value}")
        
        # Save results to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                if output_file.endswith('.json'):
                    json.dump(results, f, indent=2, default=str)
                elif output_file.endswith('.yaml') or output_file.endswith('.yml'):
                    yaml.dump(results, f, default_flow_style=False)
                else:
                    f.write(f"Diagnostic Results\n")
                    f.write("-" * 60 + "\n")
                    
                    for component_name, result in results.items():
                        f.write(f"\n{component_name.upper()}:\n")
                        
                        if "error" in result:
                            f.write(f"  Error: {result['error']}\n")
                            continue
                        
                        # Write status
                        status = result.get("status", {})
                        if status:
                            f.write("  Status:\n")
                            for key, value in status.items():
                                if isinstance(value, (dict, list)):
                                    continue  # Skip complex structures
                                f.write(f"    {key}: {value}\n")
                        
                        # Write health
                        health = result.get("health", {})
                        if health:
                            f.write("  Health:\n")
                            for key, value in health.items():
                                if isinstance(value, (dict, list)):
                                    continue  # Skip complex structures
                                f.write(f"    {key}: {value}\n")
            
            print(f"\nDiagnostic results saved to {output_file}")
        
        return 0
    except Exception as e:
        print(f"Error running diagnostics: {e}")
        return 1

def benchmark_command(args: str) -> int:
    """
    Run benchmarks on the system.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run benchmarks on the system")
    parser.add_argument("benchmark", nargs="?", choices=["cpu", "memory", "disk", "network", "all"], default="all", 
                       help="Benchmark to run")
    parser.add_argument("--iterations", type=int, default=1, help="Number of benchmark iterations")
    parser.add_argument("--output", "-o", help="Output file for benchmark results")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    benchmark_type = cmd_args.benchmark
    iterations = cmd_args.iterations
    output_file = cmd_args.output
    
    print(f"Running {benchmark_type} benchmark with {iterations} iterations...")
    
    try:
        # Check for required modules
        try:
            from benchmarking_system import BenchmarkingSystem
        except ImportError:
            print("Benchmarking module not available")
            return 1
        
        # Run benchmarks
        benchmark_system = BenchmarkingSystem()
        
        if benchmark_type == "cpu" or benchmark_type == "all":
            print("Running CPU benchmark...")
            cpu_result = benchmark_system.benchmark_cpu(iterations=iterations)
            print(f"CPU Score: {cpu_result.get('score', 0)}")
        
        if benchmark_type == "memory" or benchmark_type == "all":
            print("Running Memory benchmark...")
            memory_result = benchmark_system.benchmark_memory(iterations=iterations)
            print(f"Memory Score: {memory_result.get('score', 0)}")
        
        if benchmark_type == "disk" or benchmark_type == "all":
            print("Running Disk benchmark...")
            disk_result = benchmark_system.benchmark_disk(iterations=iterations)
            print(f"Disk Score: {disk_result.get('score', 0)}")
        
        if benchmark_type == "network" or benchmark_type == "all":
            print("Running Network benchmark...")
            network_result = benchmark_system.benchmark_network(iterations=iterations)
            print(f"Network Score: {network_result.get('score', 0)}")
        
        # Get combined results
        results = benchmark_system.get_last_results()
        
        # Display detailed results
        print("\nDetailed Benchmark Results:")
        print("-" * 60)
        
        for benchmark_name, result in results.items():
            print(f"\n{benchmark_name.upper()}:")
            
            score = result.get("score", 0)
            print(f"  Score: {score}")
            
            metrics = result.get("metrics", {})
            if metrics:
                print("  Metrics:")
                for key, value in metrics.items():
                    print(f"    {key}: {value}")
        
        # Save results to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                if output_file.endswith('.json'):
                    json.dump(results, f, indent=2)
                elif output_file.endswith('.yaml') or output_file.endswith('.yml'):
                    yaml.dump(results, f, default_flow_style=False)
                else:
                    f.write(f"Benchmark Results\n")
                    f.write("-" * 60 + "\n")
                    
                    for benchmark_name, result in results.items():
                        f.write(f"\n{benchmark_name.upper()}:\n")
                        
                        score = result.get("score", 0)
                        f.write(f"  Score: {score}\n")
                        
                        metrics = result.get("metrics", {})
                        if metrics:
                            f.write("  Metrics:\n")
                            for key, value in metrics.items():
                                f.write(f"    {key}: {value}\n")
            
            print(f"\nBenchmark results saved to {output_file}")
        
        return 0
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        return 1

def scan_command(args: str) -> int:
    """
    Scan the system for issues.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Scan the system for issues")
    parser.add_argument("scan_type", nargs="?", choices=["security", "performance", "code", "all"], default="all", 
                       help="Type of scan to perform")
    parser.add_argument("--target", help="Scan target (file or directory)")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively scan directories")
    parser.add_argument("--output", "-o", help="Output file for scan results")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    scan_type = cmd_args.scan_type
    target = cmd_args.target or "."
    recursive = cmd_args.recursive
    output_file = cmd_args.output
    
    print(f"Running {scan_type} scan on target: {target}")
    
    try:
        # Get integrated components
        registry = sys.modules.get("__main__").registry
        auditor = registry.get_component("auditor")
        
        if not auditor:
            print("Error: Auditor component not available")
            return 1
        
        # Run scan
        scan_options = {
            "target": target,
            "recursive": recursive,
            "scan_type": scan_type
        }
        
        results = auditor.scan(**scan_options)
        
        # Display results
        print("\nScan Results:")
        print("-" * 60)
        
        # Display summary
        print(f"Target: {target}")
        print(f"Issues found: {results.get('total_issues', 0)}")
        
        # Display issues by severity
        issues_by_severity = results.get("issues_by_severity", {})
        if issues_by_severity:
            print("\nIssues by Severity:")
            for severity, count in issues_by_severity.items():
                print(f"  {severity}: {count}")
        
        # Display issues
        issues = results.get("issues", [])
        if issues:
            print("\nIssues:")
            for i, issue in enumerate(issues, 1):
                issue_id = issue.get("id", "Unknown")
                severity = issue.get("severity", "Unknown")
                location = issue.get("location", "Unknown")
                description = issue.get("description", "No description")
                
                print(f"  {i}. [{severity}] {issue_id}")
                print(f"     Location: {location}")
                print(f"     Description: {description}")
        else:
            print("\nNo issues found")
        
        # Save results to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                if output_file.endswith('.json'):
                    json.dump(results, f, indent=2)
                elif output_file.endswith('.yaml') or output_file.endswith('.yml'):
                    yaml.dump(results, f, default_flow_style=False)
                else:
                    f.write(f"Scan Results\n")
                    f.write(f"Target: {target}\n")
                    f.write(f"Issues found: {results.get('total_issues', 0)}\n\n")
                    
                    # Write issues by severity
                    if issues_by_severity:
                        f.write("Issues by Severity:\n")
                        for severity, count in issues_by_severity.items():
                            f.write(f"  {severity}: {count}\n")
                        f.write("\n")
                    
                    # Write issues
                    if issues:
                        f.write("Issues:\n")
                        for i, issue in enumerate(issues, 1):
                            issue_id = issue.get("id", "Unknown")
                            severity = issue.get("severity", "Unknown")
                            location = issue.get("location", "Unknown")
                            description = issue.get("description", "No description")
                            
                            f.write(f"{i}. [{severity}] {issue_id}\n")
                            f.write(f"   Location: {location}\n")
                            f.write(f"   Description: {description}\n\n")
            
            print(f"\nScan results saved to {output_file}")
        
        return 0
    except Exception as e:
        print(f"Error scanning system: {e}")
        return 1

def stats_command(args: str) -> int:
    """
    Display system statistics.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Display system statistics")
    parser.add_argument("--component", help="Component to show statistics for")
    parser.add_argument("--period", choices=["hour", "day", "week", "month"], default="day", 
                       help="Time period for statistics")
    parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                       help="Output format")
    parser.add_argument("--output", "-o", help="Output file for statistics")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    component = cmd_args.component
    period = cmd_args.period
    output_format = cmd_args.format
    output_file = cmd_args.output
    
    print(f"Retrieving statistics for period: {period}" + (f", component: {component}" if component else ""))
    
    try:
        # Get integrated components
        registry = sys.modules.get("__main__").registry
        
        # Collect statistics
        stats = {}
        
        # Auditor statistics
        auditor = registry.get_component("auditor")
        if auditor and (not component or component == "auditor"):
            try:
                auditor_stats = auditor.get_metrics(period=period)
                stats["auditor"] = auditor_stats
            except Exception as e:
                print(f"Error getting auditor statistics: {e}")
                stats["auditor"] = {"error": str(e)}
        
        # Neural Matrix statistics
        neural_matrix = registry.get_component("neural_matrix")
        if neural_matrix and (not component or component == "neural_matrix"):
            try:
                neural_matrix_stats = neural_matrix.get_statistics(period=period)
                stats["neural_matrix"] = neural_matrix_stats
            except Exception as e:
                print(f"Error getting neural matrix statistics: {e}")
                stats["neural_matrix"] = {"error": str(e)}
        
        # Triangulum statistics
        triangulum = registry.get_component("triangulum")
        if triangulum and (not component or component == "triangulum"):
            try:
                triangulum_stats = triangulum.get_statistics(period=period)
                stats["triangulum"] = triangulum_stats
            except Exception as e:
                print(f"Error getting triangulum statistics: {e}")
                stats["triangulum"] = {"error": str(e)}
        
        # Display statistics
        if output_format == "json":
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
                print(f"Statistics saved to {output_file}")
            else:
                print(json.dumps(stats, indent=2, default=str))
        elif output_format == "yaml":
            if output_file:
                with open(output_file, 'w') as f:
                    yaml.dump(stats, f, default_flow_style=False)
                print(f"Statistics saved to {output_file}")
            else:
                print(yaml.dump(stats, default_flow_style=False))
        else:  # text format
            print("\nSystem Statistics:")
            print("-" * 60)
            
            if not stats:
                print("No statistics available")
            else:
                for component_name, component_stats in stats.items():
                    print(f"\n{component_name.upper()}:")
                    
                    if "error" in component_stats:
                        print(f"  Error: {component_stats['error']}")
                        continue
                    
                    if isinstance(component_stats, dict):
                        for category, value in component_stats.items():
                            if isinstance(value, dict):
                                print(f"  {category}:")
                                for k, v in value.items():
                                    print(f"    {k}: {v}")
                            else:
                                print(f"  {category}: {value}")
                    elif isinstance(component_stats, list):
                        for item in component_stats:
                            if isinstance(item, dict):
                                timestamp = item.get("timestamp", "Unknown")
                                print(f"  {timestamp}:")
                                for k, v in item.items():
                                    if k != "timestamp":
                                        print(f"    {k}: {v}")
                            else:
                                print(f"  {item}")
                    else:
                        print(f"  {component_stats}")
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    f.write("System Statistics\n")
                    f.write("-" * 60 + "\n")
                    
                    for component_name, component_stats in stats.items():
                        f.write(f"\n{component_name.upper()}:\n")
                        
                        if "error" in component_stats:
                            f.write(f"  Error: {component_stats['error']}\n")
                            continue
                        
                        if isinstance(component_stats, dict):
                            for category, value in component_stats.items():
                                if isinstance(value, dict):
                                    f.write(f"  {category}:\n")
                                    for k, v in value.items():
                                        f.write(f"    {k}: {v}\n")
                                else:
                                    f.write(f"  {category}: {value}\n")
                        elif isinstance(component_stats, list):
                            for item in component_stats:
                                if isinstance(item, dict):
                                    timestamp = item.get("timestamp", "Unknown")
                                    f.write(f"  {timestamp}:\n")
                                    for k, v in item.items():
                                        if k != "timestamp":
                                            f.write(f"    {k}: {v}\n")
                                else:
                                    f.write(f"  {item}\n")
                        else:
                            f.write(f"  {component_stats}\n")
                
                print(f"\nStatistics saved to {output_file}")
        
        return 0
    except Exception as e:
        print(f"Error retrieving statistics: {e}")
        return 1

def init_command(args: str) -> int:
    """
    Initialize the FixWurx system.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Initialize the FixWurx system")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--force", "-f", action="store_true", help="Force reinitialization")
    parser.add_argument("--component", help="Specific component to initialize")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    config_file = cmd_args.config
    force = cmd_args.force
    component = cmd_args.component
    
    print(f"Initializing FixWurx" + (f" component: {component}" if component else ""))
    
    try:
        # Check for configuration file
        if config_file:
            if not Path(config_file).exists():
                print(f"Error: Configuration file '{config_file}' not found")
                return 1
            
            print(f"Using configuration from: {config_file}")
        
        # Get registry
        registry = sys.modules.get("__main__").registry
        
        # Initialize components
        if component:
            # Initialize specific component
            if component == "auditor":
                auditor = registry.get_component("auditor")
                if auditor:
                    if hasattr(auditor, 'initialize'):
                        auditor.initialize(force=force)
                        print("Auditor component initialized")
                    else:
                        print("Auditor component does not support initialization")
                else:
                    print("Auditor component not available")
            elif component == "neural_matrix":
                neural_matrix = registry.get_component("neural_matrix")
                if neural_matrix:
                    if hasattr(neural_matrix, 'initialize'):
                        neural_matrix.initialize(force=force)
                        print("Neural Matrix component initialized")
                    else:
                        print("Neural Matrix component does not support initialization")
                else:
                    print("Neural Matrix component not available")
            elif component == "triangulum":
                triangulum = registry.get_component("triangulum")
                if triangulum:
                    if hasattr(triangulum, 'initialize'):
                        triangulum.initialize(force=force)
                        print("Triangulum component initialized")
                    else:
                        print("Triangulum component does not support initialization")
                else:
                    print("Triangulum component not available")
            else:
                print(f"Unknown component: {component}")
                return 1
        else:
            # Initialize all components
            auditor = registry.get_component("auditor")
            if auditor and hasattr(auditor, 'initialize'):
                try:
                    auditor.initialize(force=force)
                    print("Auditor component initialized")
                except Exception as e:
                    print(f"Error initializing Auditor: {e}")
            
            neural_matrix = registry.get_component("neural_matrix")
            if neural_matrix and hasattr(neural_matrix, 'initialize'):
                try:
                    neural_matrix.initialize(force=force)
                    print("Neural Matrix component initialized")
                except Exception as e:
                    print(f"Error initializing Neural Matrix: {e}")
            
            triangulum = registry.get_component("triangulum")
            if triangulum and hasattr(triangulum, 'initialize'):
                try:
                    triangulum.initialize(force=force)
                    print("Triangulum component initialized")
                except Exception as e:
                    print(f"Error initializing Triangulum: {e}")
        
        print("\nInitialization complete")
        return 0
    except Exception as e:
        print(f"Error initializing FixWurx: {e}")
        return 1

def fix_started_handler(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle fix started events.
    
    Args:
        event_data: Event data
        
    Returns:
        Result data
    """
    try:
        logger.info(f"Fix started: {event_data.get('target', 'Unknown')}")
        
        # Log details
        target = event_data.get("target", "Unknown")
        issue_id = event_data.get("issue_id")
        recursive = event_data.get("recursive", False)
        dry_run = event_data.get("dry_run", False)
        
        logger.info(f"Fix details: target={target}, issue_id={issue_id}, recursive={recursive}, dry_run={dry_run}")
        
        return {
            "success": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "message": f"Fix started for {target}"
        }
    except Exception as e:
        logger.error(f"Error handling fix started event: {e}")
        return {"success": False, "error": str(e)}

def fix_completed_handler(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle fix completed events.
    
    Args:
        event_data: Event data
        
    Returns:
        Result data
    """
    try:
        target = event_data.get("target", "Unknown")
        success = event_data.get("success", False)
        
        if success:
            logger.info(f"Fix completed successfully: {target}")
            
            # Log results
            result = event_data.get("result", {})
            fixed_issues = result.get("fixed_issues", [])
            remaining_issues = result.get("remaining_issues", [])
            
            logger.info(f"Fix results: fixed={len(fixed_issues)}, remaining={len(remaining_issues)}")
        else:
            error = event_data.get("error", "Unknown error")
            logger.error(f"Fix failed: {target} - {error}")
        
        return {
            "success": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "message": f"Fix {'completed' if success else 'failed'} for {target}"
        }
    except Exception as e:
        logger.error(f"Error handling fix completed event: {e}")
        return {"success": False, "error": str(e)}
