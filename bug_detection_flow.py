#!/usr/bin/env python3
"""
bug_detection_flow.py
─────────────────────
Implements the bug detection flow for the FixWurx system.

This module provides the core flow for detecting bugs in code, including
file scanning, analysis, classification, and reporting. It integrates with
various components of the system including the agent system, triangulation
engine, and neural matrix.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

# Import core components
from triangulation_engine import TriangulationEngine
from neural_matrix_core import NeuralMatrix
from meta_agent import MetaAgent
from scope_filter import ScopeFilter
from bug_detection import BugDetector

# Configure logging
logger = logging.getLogger("BugDetectionFlow")

class BugDetectionFlow:
    """
    Implements the bug detection flow for the FixWurx system.
    
    This class orchestrates the entire bug detection process, from scanning
    files to classifying and reporting detected bugs. It serves as the main
    entry point for the bug detection subsystem.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the bug detection flow.
        
        Args:
            config: Configuration for the bug detection flow.
        """
        self.config = config or {}
        self.triangulation_engine = TriangulationEngine()
        self.neural_matrix = NeuralMatrix()
        self.meta_agent = MetaAgent()
        self.scope_filter = ScopeFilter()
        self.bug_detector = BugDetector()
        
        # Initialize state
        self.current_scan_id = None
        self.current_context = {}
        self.detected_bugs = []
        
        logger.info("Bug Detection Flow initialized")
    
    def start_detection(self, 
                        target_path: str, 
                        scan_options: Dict[str, Any] = None) -> str:
        """
        Start the bug detection process on a specified path.
        
        Args:
            target_path: Path to scan for bugs.
            scan_options: Options for the scan.
            
        Returns:
            Scan ID for the detection process.
        """
        scan_options = scan_options or {}
        
        # Generate a scan ID
        timestamp = int(time.time())
        scan_id = f"scan_{timestamp}_{os.path.basename(target_path)}"
        self.current_scan_id = scan_id
        
        # Initialize scan context
        self.current_context = {
            "scan_id": scan_id,
            "target_path": target_path,
            "start_time": timestamp,
            "options": scan_options,
            "status": "started",
            "files_scanned": 0,
            "bugs_detected": 0
        }
        
        logger.info(f"Starting bug detection scan {scan_id} on {target_path}")
        
        # Trigger the detection flow
        self._execute_detection_flow(target_path, scan_options)
        
        return scan_id
    
    def _execute_detection_flow(self, 
                               target_path: str, 
                               scan_options: Dict[str, Any]) -> None:
        """
        Execute the bug detection flow.
        
        Args:
            target_path: Path to scan for bugs.
            scan_options: Options for the scan.
        """
        try:
            # Phase 1: Scope filtering
            logger.info("Phase 1: Scope filtering")
            filtered_files = self._filter_scope(target_path, scan_options)
            
            # Phase 2: Bug detection
            logger.info("Phase 2: Bug detection")
            detected_issues = self._detect_bugs(filtered_files, scan_options)
            
            # Phase 3: Classification and prioritization
            logger.info("Phase 3: Classification and prioritization")
            classified_bugs = self._classify_bugs(detected_issues)
            
            # Phase 4: Neural analysis
            logger.info("Phase 4: Neural analysis")
            enriched_bugs = self._neural_analysis(classified_bugs)
            
            # Phase 5: Report generation
            logger.info("Phase 5: Report generation")
            report = self._generate_report(enriched_bugs)
            
            # Update context
            self.current_context["status"] = "completed"
            self.current_context["end_time"] = int(time.time())
            self.current_context["bugs_detected"] = len(enriched_bugs)
            self.current_context["report"] = report
            
            # Store detected bugs
            self.detected_bugs = enriched_bugs
            
            logger.info(f"Bug detection scan {self.current_scan_id} completed with {len(enriched_bugs)} bugs detected")
            
        except Exception as e:
            logger.error(f"Error in bug detection flow: {e}")
            self.current_context["status"] = "failed"
            self.current_context["error"] = str(e)
            self.current_context["end_time"] = int(time.time())
            raise
    
    def _filter_scope(self, 
                     target_path: str, 
                     scan_options: Dict[str, Any]) -> List[str]:
        """
        Filter the scope of the scan to focus on relevant files.
        
        Args:
            target_path: Path to scan.
            scan_options: Options for the scan.
            
        Returns:
            List of file paths to scan.
        """
        # Configure scope filter
        exclude_patterns = scan_options.get("exclude_patterns", [])
        include_extensions = scan_options.get("include_extensions", [])
        
        # Apply entropy-driven scope reduction if enabled
        use_entropy = scan_options.get("use_entropy", True)
        entropy_threshold = scan_options.get("entropy_threshold", 0.7)
        
        # Get filtered files
        filtered_files = self.scope_filter.filter_files(
            target_path, 
            exclude_patterns=exclude_patterns,
            include_extensions=include_extensions,
            use_entropy=use_entropy,
            entropy_threshold=entropy_threshold
        )
        
        # Update context
        self.current_context["files_scanned"] = len(filtered_files)
        
        return filtered_files
    
    def _detect_bugs(self, 
                    file_paths: List[str], 
                    scan_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect bugs in the provided files.
        
        Args:
            file_paths: List of file paths to scan.
            scan_options: Options for the scan.
            
        Returns:
            List of detected issues.
        """
        detected_issues = []
        
        for file_path in file_paths:
            # Check if the file is a valid target
            if not os.path.isfile(file_path):
                continue
            
            # Detect bugs in the file
            file_issues = self.bug_detector.detect_bugs(file_path, scan_options)
            
            if file_issues:
                detected_issues.extend(file_issues)
        
        return detected_issues
    
    def _classify_bugs(self, 
                      detected_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify and prioritize detected bugs.
        
        Args:
            detected_issues: List of detected issues.
            
        Returns:
            List of classified bugs.
        """
        classified_bugs = []
        
        for issue in detected_issues:
            # Classify the bug using the triangulation engine
            classification = self.triangulation_engine.classify_bug(issue)
            
            # Add classification to the issue
            issue["classification"] = classification
            
            # Prioritize the bug
            priority = self._calculate_priority(issue)
            issue["priority"] = priority
            
            classified_bugs.append(issue)
        
        # Sort bugs by priority
        classified_bugs.sort(key=lambda x: x["priority"], reverse=True)
        
        return classified_bugs
    
    def _calculate_priority(self, issue: Dict[str, Any]) -> float:
        """
        Calculate the priority of a bug.
        
        Args:
            issue: Bug issue data.
            
        Returns:
            Priority score (higher is more important).
        """
        # Base priority from classification
        base_priority = issue["classification"].get("severity", 0.5)
        
        # Adjust for impact
        impact = issue["classification"].get("impact", 0.5)
        
        # Adjust for confidence
        confidence = issue["classification"].get("confidence", 0.8)
        
        # Calculate weighted priority
        priority = (base_priority * 0.4) + (impact * 0.4) + (confidence * 0.2)
        
        return priority
    
    def _neural_analysis(self, 
                        classified_bugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform neural analysis on classified bugs.
        
        Args:
            classified_bugs: List of classified bugs.
            
        Returns:
            List of bugs with neural analysis.
        """
        enriched_bugs = []
        
        for bug in classified_bugs:
            # Get neural insights
            neural_insights = self.neural_matrix.analyze_bug(bug)
            
            # Add neural insights to the bug
            bug["neural_insights"] = neural_insights
            
            # Get similar bugs from history
            similar_bugs = self.neural_matrix.find_similar_bugs(bug)
            
            if similar_bugs:
                bug["similar_bugs"] = similar_bugs
                
            # Get solution recommendations
            solution_recommendations = self.neural_matrix.recommend_solutions(bug)
            
            if solution_recommendations:
                bug["solution_recommendations"] = solution_recommendations
            
            enriched_bugs.append(bug)
        
        return enriched_bugs
    
    def _generate_report(self, 
                        enriched_bugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a report of the detected bugs.
        
        Args:
            enriched_bugs: List of bugs with neural analysis.
            
        Returns:
            Bug detection report.
        """
        # Calculate statistics
        bug_count = len(enriched_bugs)
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        for bug in enriched_bugs:
            severity = bug["classification"].get("severity_level", "medium")
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Generate summary
        summary = {
            "scan_id": self.current_scan_id,
            "target_path": self.current_context["target_path"],
            "start_time": self.current_context["start_time"],
            "end_time": int(time.time()),
            "files_scanned": self.current_context["files_scanned"],
            "bugs_detected": bug_count,
            "severity_counts": severity_counts
        }
        
        # Generate report
        report = {
            "summary": summary,
            "bugs": enriched_bugs
        }
        
        # Notify the Meta Agent
        self.meta_agent.notify_bug_detection_complete(report)
        
        return report
    
    def get_bug_by_id(self, bug_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a bug by its ID.
        
        Args:
            bug_id: Bug ID.
            
        Returns:
            Bug data or None if not found.
        """
        for bug in self.detected_bugs:
            if bug.get("id") == bug_id:
                return bug
        
        return None
    
    def get_scan_status(self) -> Dict[str, Any]:
        """
        Get the status of the current scan.
        
        Returns:
            Scan status data.
        """
        return self.current_context
    
    def save_report(self, 
                   output_path: str = None, 
                   format: str = "json") -> str:
        """
        Save the bug detection report to a file.
        
        Args:
            output_path: Path to save the report. If None, a default path is used.
            format: Report format (json or html).
            
        Returns:
            Path to the saved report.
        """
        if not self.current_context.get("report"):
            raise ValueError("No report available to save")
        
        # Create default output path if not provided
        if not output_path:
            timestamp = self.current_context["start_time"]
            filename = f"bug_detection_report_{self.current_scan_id}.{format}"
            output_path = os.path.join(os.getcwd(), filename)
        
        # Save report in the specified format
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.current_context["report"], f, indent=2)
        elif format == "html":
            # Generate HTML report
            html_report = self._generate_html_report(self.current_context["report"])
            
            with open(output_path, "w") as f:
                f.write(html_report)
        else:
            raise ValueError(f"Unsupported report format: {format}")
        
        logger.info(f"Bug detection report saved to {output_path}")
        
        return output_path
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """
        Generate an HTML report from the bug detection data.
        
        Args:
            report: Bug detection report data.
            
        Returns:
            HTML report as a string.
        """
        # Generate HTML report
        # This is a simplified implementation
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bug Detection Report: {report['summary']['scan_id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .severity-critical {{ background-color: #ffdddd; }}
                .severity-high {{ background-color: #ffffcc; }}
                .severity-medium {{ background-color: #e6f3ff; }}
                .severity-low {{ background-color: #f0f0f0; }}
            </style>
        </head>
        <body>
            <h1>Bug Detection Report</h1>
            
            <h2>Summary</h2>
            <table>
                <tr><th>Scan ID</th><td>{report['summary']['scan_id']}</td></tr>
                <tr><th>Target Path</th><td>{report['summary']['target_path']}</td></tr>
                <tr><th>Start Time</th><td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['summary']['start_time']))}</td></tr>
                <tr><th>End Time</th><td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['summary']['end_time']))}</td></tr>
                <tr><th>Files Scanned</th><td>{report['summary']['files_scanned']}</td></tr>
                <tr><th>Bugs Detected</th><td>{report['summary']['bugs_detected']}</td></tr>
            </table>
            
            <h2>Severity Distribution</h2>
            <table>
                <tr>
                    <th>Critical</th>
                    <th>High</th>
                    <th>Medium</th>
                    <th>Low</th>
                </tr>
                <tr>
                    <td>{report['summary']['severity_counts']['critical']}</td>
                    <td>{report['summary']['severity_counts']['high']}</td>
                    <td>{report['summary']['severity_counts']['medium']}</td>
                    <td>{report['summary']['severity_counts']['low']}</td>
                </tr>
            </table>
            
            <h2>Detected Bugs</h2>
        """
        
        # Add bug details
        for i, bug in enumerate(report['bugs']):
            severity = bug['classification'].get('severity_level', 'medium')
            html += f"""
            <div class="bug severity-{severity}">
                <h3>Bug #{i+1}: {bug.get('title', 'Untitled Bug')}</h3>
                <table>
                    <tr><th>ID</th><td>{bug.get('id', 'N/A')}</td></tr>
                    <tr><th>File</th><td>{bug.get('file_path', 'N/A')}</td></tr>
                    <tr><th>Line</th><td>{bug.get('line_number', 'N/A')}</td></tr>
                    <tr><th>Severity</th><td>{severity}</td></tr>
                    <tr><th>Description</th><td>{bug.get('description', 'N/A')}</td></tr>
                </table>
            """
            
            # Add solution recommendations if available
            if 'solution_recommendations' in bug:
                html += """
                <h4>Solution Recommendations</h4>
                <ul>
                """
                
                for rec in bug['solution_recommendations']:
                    html += f"<li>{rec.get('description', 'N/A')}</li>"
                
                html += "</ul>"
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html

# Main entry point
def detect_bugs(target_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main function to detect bugs in a codebase.
    
    Args:
        target_path: Path to the code to analyze.
        options: Detection options.
        
    Returns:
        Detection report.
    """
    flow = BugDetectionFlow()
    scan_id = flow.start_detection(target_path, options)
    
    # Wait for scan to complete
    while flow.get_scan_status()["status"] not in ["completed", "failed"]:
        time.sleep(0.1)
    
    # Get scan results
    scan_status = flow.get_scan_status()
    
    if scan_status["status"] == "failed":
        logger.error(f"Bug detection failed: {scan_status.get('error', 'Unknown error')}")
        raise RuntimeError(f"Bug detection failed: {scan_status.get('error', 'Unknown error')}")
    
    return scan_status["report"]

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python bug_detection_flow.py <target_path> [options_json]")
        sys.exit(1)
    
    target_path = sys.argv[1]
    
    # Parse options if provided
    options = {}
    if len(sys.argv) > 2:
        try:
            options = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            print("Error: options must be a valid JSON string")
            sys.exit(1)
    
    # Run bug detection
    try:
        report = detect_bugs(target_path, options)
        
        # Create output path
        output_path = options.get("output_path", f"bug_detection_report_{int(time.time())}.json")
        
        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Bug detection report saved to {output_path}")
        
        # Print summary
        print("\nSummary:")
        print(f"Files scanned: {report['summary']['files_scanned']}")
        print(f"Bugs detected: {report['summary']['bugs_detected']}")
        print("Severity distribution:")
        for severity, count in report['summary']['severity_counts'].items():
            print(f"  {severity}: {count}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
