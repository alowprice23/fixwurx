#!/usr/bin/env python3
"""
Bug Detection Commands Module

This module registers command handlers for the bug detection system within the shell environment,
enabling AI-powered code analysis and bug detection.
"""

import os
import sys
import json
import logging
import argparse
import shlex
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("BugDetectionCommands")

def register_bug_commands(registry):
    """
    Register bug detection command handlers with the component registry.
    
    Args:
        registry: Component registry instance
    """
    try:
        # Register command handlers
        registry.register_command_handler("detect", detect_command, "bug_detection")
        registry.register_command_handler("analyze", analyze_command, "bug_detection")
        registry.register_command_handler("fix", fix_command, "bug_detection")
        registry.register_command_handler("scan", scan_command, "bug_detection")
        
        # Register prefixed versions
        registry.register_command_handler("bugs:detect", detect_command, "bug_detection")
        registry.register_command_handler("bugs:analyze", analyze_command, "bug_detection")
        registry.register_command_handler("bugs:fix", fix_command, "bug_detection")
        registry.register_command_handler("bugs:scan", scan_command, "bug_detection")
        
        # Register aliases
        registry.register_alias("find-bugs", "detect")
        registry.register_alias("code-analyze", "analyze")
        registry.register_alias("auto-fix", "fix --auto")
        registry.register_alias("scan-dir", "scan")
        
        logger.info("Bug detection commands registered")
    except Exception as e:
        logger.error(f"Error registering bug detection commands: {e}")

def detect_command(args: str) -> int:
    """
    Detect bugs in a file or directory.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Detect bugs in a file or directory")
    parser.add_argument("path", help="File or directory to analyze")
    parser.add_argument("--pattern", help="File pattern to match (for directory analysis)")
    parser.add_argument("--recursive", "-r", action="store_true", help="Analyze subdirectories recursively")
    parser.add_argument("--confidence", "-c", type=float, default=0.7, help="Minimum confidence threshold")
    parser.add_argument("--severity", "-s", choices=["critical", "high", "medium", "low"], 
                       help="Minimum severity level")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import bug detection module
    try:
        from bug_detection import analyze_code, detect_bugs_in_directory
    except ImportError:
        print("Error: Bug detection module not available")
        return 1
    
    # Create configuration
    config = {
        "confidence_threshold": cmd_args.confidence,
    }
    
    if cmd_args.severity:
        severity_levels = ["critical", "high", "medium", "low"]
        severity_index = severity_levels.index(cmd_args.severity)
        config["severity_levels"] = severity_levels[:severity_index + 1]
    
    # Analyze file or directory
    path = Path(cmd_args.path)
    
    if path.is_file():
        # Analyze single file
        print(f"Analyzing file: {cmd_args.path}")
        result = analyze_code(cmd_args.path, config=config)
        
        if result.get("success", False):
            analysis = result["analysis"]
            print(f"\nAnalysis results:")
            print(f"Found {analysis.get('total_issues', 0)} issues")
            print(f"Highest severity: {analysis.get('highest_severity', 'none')}")
            
            for i, issue in enumerate(analysis.get("issues", []), 1):
                print(f"\nIssue {i}:")
                print(f"  Description: {issue.get('description', 'Unknown')}")
                print(f"  Location: {issue.get('location', 'Unknown')}")
                print(f"  Severity: {issue.get('severity', 'Unknown')}")
                print(f"  Confidence: {issue.get('confidence', 0):.2f}")
                print(f"  Fix: {issue.get('fix', 'Unknown')}")
            
            # Save results to file if requested
            if cmd_args.output:
                try:
                    with open(cmd_args.output, 'w') as f:
                        json.dump(analysis, f, indent=2)
                    print(f"\nResults saved to {cmd_args.output}")
                except Exception as e:
                    print(f"Error saving results: {e}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    elif path.is_dir():
        # Analyze directory
        print(f"Analyzing directory: {cmd_args.path}")
        print(f"Pattern: {cmd_args.pattern or '*'}")
        print(f"Recursive: {cmd_args.recursive}")
        
        result = detect_bugs_in_directory(
            cmd_args.path, 
            cmd_args.pattern, 
            cmd_args.recursive, 
            config
        )
        
        if result.get("success", False):
            results = result["results"]
            print(f"\nAnalysis of directory {cmd_args.path}:")
            print(f"Analyzed {result.get('analyzed_files', 0)} of {result.get('total_files', 0)} files")
            
            total_issues = sum(analysis.get("total_issues", 0) for analysis in results.values())
            print(f"Found {total_issues} issues in total")
            
            # Show files with issues
            files_with_issues = [
                (file_path, analysis) 
                for file_path, analysis in results.items() 
                if analysis.get("total_issues", 0) > 0
            ]
            
            if files_with_issues:
                print("\nFiles with issues:")
                for file_path, analysis in files_with_issues:
                    print(f"  {file_path}: {analysis.get('total_issues', 0)} issues ({analysis.get('highest_severity', 'none')})")
            else:
                print("\nNo issues found in any files")
            
            # Save results to file if requested
            if cmd_args.output:
                try:
                    with open(cmd_args.output, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"\nResults saved to {cmd_args.output}")
                except Exception as e:
                    print(f"Error saving results: {e}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    else:
        print(f"Error: {cmd_args.path} is not a valid file or directory")
        return 1
    
    return 0

def analyze_command(args: str) -> int:
    """
    Analyze a file for bugs with detailed output.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze a file for bugs")
    parser.add_argument("file", help="File to analyze")
    parser.add_argument("--focus", "-f", help="Function or section to focus on")
    parser.add_argument("--confidence", "-c", type=float, default=0.7, help="Minimum confidence threshold")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed analysis")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import bug detection module
    try:
        from bug_detection import analyze_code
    except ImportError:
        print("Error: Bug detection module not available")
        return 1
    
    # Create configuration
    config = {
        "confidence_threshold": cmd_args.confidence,
    }
    
    # Analyze file
    print(f"Analyzing file: {cmd_args.file}")
    if cmd_args.focus:
        print(f"Focusing on: {cmd_args.focus}")
    
    result = analyze_code(cmd_args.file, cmd_args.focus, config)
    
    if result.get("success", False):
        analysis = result["analysis"]
        print(f"\nAnalysis results for {cmd_args.file}:")
        print(f"Language: {analysis.get('language', 'unknown')}")
        print(f"Issues found: {analysis.get('total_issues', 0)}")
        print(f"Highest severity: {analysis.get('highest_severity', 'none')}")
        print(f"Analysis time: {analysis.get('analysis_time', 0):.2f} seconds")
        
        if analysis.get("issues"):
            print("\nIssues:")
            for i, issue in enumerate(analysis["issues"], 1):
                print(f"\nIssue {i}:")
                print(f"  Description: {issue.get('description', 'Unknown')}")
                print(f"  Location: {issue.get('location', 'Unknown')}")
                print(f"  Severity: {issue.get('severity', 'Unknown')}")
                print(f"  Confidence: {issue.get('confidence', 0):.2f}")
                
                if cmd_args.detailed:
                    print(f"  Fix: {issue.get('fix', 'Unknown')}")
        
        # Ask if user wants to apply fixes
        if analysis.get("total_issues", 0) > 0:
            print("\nWould you like to apply the suggested fixes? (y/n)")
            print("Type 'fix' to apply fixes to this file")
        
        # Save results to file if requested
        if cmd_args.output:
            try:
                with open(cmd_args.output, 'w') as f:
                    json.dump(analysis, f, indent=2)
                print(f"\nResults saved to {cmd_args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

def fix_command(args: str) -> int:
    """
    Apply fixes to a file based on analysis.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Apply fixes to a file")
    parser.add_argument("file", help="File to fix")
    parser.add_argument("--auto", "-a", action="store_true", help="Apply fixes automatically without confirmation")
    parser.add_argument("--focus", "-f", help="Function or section to focus on")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import bug detection module
    try:
        from bug_detection import analyze_code, apply_suggested_fixes
    except ImportError:
        print("Error: Bug detection module not available")
        return 1
    
    # First analyze the file
    print(f"Analyzing file: {cmd_args.file}")
    result = analyze_code(cmd_args.file, cmd_args.focus)
    
    if result.get("success", False):
        analysis = result["analysis"]
        print(f"\nFound {analysis.get('total_issues', 0)} issues to fix")
        
        if analysis.get("total_issues", 0) == 0:
            print("No issues to fix")
            return 0
        
        # Apply fixes with or without confirmation
        if cmd_args.auto:
            print("Automatically applying fixes...")
            fix_result = apply_suggested_fixes(result, True)
        else:
            # Show issues and ask for confirmation
            print("\nIssues to fix:")
            for i, issue in enumerate(analysis.get("issues", []), 1):
                print(f"\nIssue {i}:")
                print(f"  Description: {issue.get('description', 'Unknown')}")
                print(f"  Location: {issue.get('location', 'Unknown')}")
                print(f"  Severity: {issue.get('severity', 'Unknown')}")
                print(f"  Fix: {issue.get('fix', 'Unknown')}")
            
            confirmation = input("\nApply these fixes? (y/n): ").lower()
            if confirmation != 'y':
                print("Fix operation cancelled")
                return 0
            
            print("Applying fixes...")
            fix_result = apply_suggested_fixes(result, False)
        
        # Report result
        if fix_result.get("success", False):
            print(f"\nSuccessfully applied fixes to {cmd_args.file}")
            print(f"Fixed {fix_result.get('issues_fixed', 0)} issues")
            print(f"Backup saved to {cmd_args.file}.bak")
        else:
            print(f"\nFailed to apply fixes: {fix_result.get('error', 'Unknown error')}")
            return 1
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

def scan_command(args: str) -> int:
    """
    Scan a directory for bugs and generate a report.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Scan a directory for bugs")
    parser.add_argument("directory", help="Directory to scan")
    parser.add_argument("--pattern", "-p", help="File pattern to match (e.g., '*.py')")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan subdirectories recursively")
    parser.add_argument("--output", "-o", help="Output file for report (HTML or JSON format)")
    parser.add_argument("--format", choices=["html", "json"], default="html", help="Report format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Import bug detection module
    try:
        from bug_detection import detect_bugs_in_directory
    except ImportError:
        print("Error: Bug detection module not available")
        return 1
    
    # Scan directory
    print(f"Scanning directory: {cmd_args.directory}")
    print(f"Pattern: {cmd_args.pattern or '*'}")
    print(f"Recursive: {cmd_args.recursive}")
    
    result = detect_bugs_in_directory(
        cmd_args.directory, 
        cmd_args.pattern, 
        cmd_args.recursive
    )
    
    if result.get("success", False):
        results = result["results"]
        total_files = result.get("total_files", 0)
        analyzed_files = result.get("analyzed_files", 0)
        
        # Calculate statistics
        total_issues = sum(analysis.get("total_issues", 0) for analysis in results.values())
        files_with_issues = sum(1 for analysis in results.values() if analysis.get("total_issues", 0) > 0)
        
        # Count issues by severity
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        for analysis in results.values():
            for issue in analysis.get("issues", []):
                severity = issue.get("severity", "low")
                if severity in severity_counts:
                    severity_counts[severity] += 1
        
        # Print summary
        print(f"\nScan Results:")
        print(f"Analyzed {analyzed_files} of {total_files} files")
        print(f"Found {total_issues} issues in {files_with_issues} files")
        print("\nIssues by severity:")
        for severity, count in severity_counts.items():
            print(f"  {severity}: {count}")
        
        # Generate report
        if cmd_args.output:
            try:
                if cmd_args.format == "json":
                    # JSON report
                    with open(cmd_args.output, 'w') as f:
                        json.dump({
                            "scan_results": results,
                            "statistics": {
                                "total_files": total_files,
                                "analyzed_files": analyzed_files,
                                "total_issues": total_issues,
                                "files_with_issues": files_with_issues,
                                "severity_counts": severity_counts
                            }
                        }, f, indent=2)
                else:
                    # HTML report
                    generate_html_report(
                        cmd_args.output,
                        results,
                        {
                            "total_files": total_files,
                            "analyzed_files": analyzed_files,
                            "total_issues": total_issues,
                            "files_with_issues": files_with_issues,
                            "severity_counts": severity_counts,
                            "directory": cmd_args.directory,
                            "pattern": cmd_args.pattern,
                            "recursive": cmd_args.recursive
                        }
                    )
                
                print(f"\nReport saved to {cmd_args.output}")
            except Exception as e:
                print(f"Error generating report: {e}")
                return 1
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

def generate_html_report(output_file: str, results: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """
    Generate an HTML report from scan results.
    
    Args:
        output_file: Output file path
        results: Scan results
        stats: Statistics
    """
    # Generate simple HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bug Detection Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background-color: #f2f2f2; text-align: left; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .critical {{ color: #d9534f; font-weight: bold; }}
            .high {{ color: #f0ad4e; font-weight: bold; }}
            .medium {{ color: #5bc0de; }}
            .low {{ color: #5cb85c; }}
            .summary {{ background-color: #eef; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Bug Detection Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Directory: {stats.get('directory', 'Unknown')}</p>
            <p>Pattern: {stats.get('pattern', '*')}</p>
            <p>Recursive: {stats.get('recursive', False)}</p>
            <p>Files analyzed: {stats.get('analyzed_files', 0)} of {stats.get('total_files', 0)}</p>
            <p>Total issues: {stats.get('total_issues', 0)} in {stats.get('files_with_issues', 0)} files</p>
            
            <h3>Issues by severity:</h3>
            <ul>
                <li class="critical">Critical: {stats.get('severity_counts', {}).get('critical', 0)}</li>
                <li class="high">High: {stats.get('severity_counts', {}).get('high', 0)}</li>
                <li class="medium">Medium: {stats.get('severity_counts', {}).get('medium', 0)}</li>
                <li class="low">Low: {stats.get('severity_counts', {}).get('low', 0)}</li>
            </ul>
        </div>
        
        <h2>Detailed Results</h2>
    """
    
    # Add file details
    for file_path, analysis in results.items():
        if analysis.get("total_issues", 0) > 0:
            html += f"""
            <h3>File: {file_path}</h3>
            <p>Language: {analysis.get('language', 'Unknown')}</p>
            <p>Issues: {analysis.get('total_issues', 0)}</p>
            <p>Highest severity: {analysis.get('highest_severity', 'none')}</p>
            
            <table>
                <tr>
                    <th>Description</th>
                    <th>Location</th>
                    <th>Severity</th>
                    <th>Confidence</th>
                    <th>Fix</th>
                </tr>
            """
            
            for issue in analysis.get("issues", []):
                severity = issue.get("severity", "low")
                html += f"""
                <tr>
                    <td>{issue.get('description', 'Unknown')}</td>
                    <td>{issue.get('location', 'Unknown')}</td>
                    <td class="{severity}">{severity}</td>
                    <td>{issue.get('confidence', 0):.2f}</td>
                    <td>{issue.get('fix', 'Unknown')}</td>
                </tr>
                """
            
            html += "</table>"
    
    html += """
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html)

if __name__ == "__main__":
    print("Bug Detection Commands Module")
    print("This module should be imported by the shell environment")
