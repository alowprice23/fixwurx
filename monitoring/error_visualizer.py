"""
monitoring/error_visualizer.py
───────────────────────────────
Enhanced visualization and analysis tools for error logs.

This module provides:
1. Advanced trend analysis for error occurrences
2. Severity distribution visualization
3. Component-based error grouping
4. Time-series error pattern detection
5. Export capabilities for various formats

Usage:
    ```python
    from monitoring.error_visualizer import ErrorVisualizer
    
    # Create visualizer with error log
    visualizer = ErrorVisualizer(error_log)
    
    # Get error trends
    trends = visualizer.get_error_trends(days=7)
    
    # Get severity distribution
    distribution = visualizer.get_severity_distribution()
    
    # Export to various formats
    visualizer.export_to_csv("error_report.csv")
    visualizer.export_to_html("error_report.html")
    ```
"""

import time
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict

from monitoring.error_log import ErrorLog, ErrorSeverity


class ErrorVisualizer:
    """
    Enhanced visualization and analysis tools for error logs.
    
    Provides advanced metrics, trend analysis, and export capabilities
    for error log data.
    """
    
    def __init__(self, error_log: ErrorLog):
        """
        Initialize the error visualizer.
        
        Args:
            error_log: The error log to visualize
        """
        self.error_log = error_log
    
    def get_error_trends(
        self,
        days: int = 7,
        group_by: str = "day",
        min_severity: Optional[str] = None
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Get error trends over time.
        
        Args:
            days: Number of days to analyze
            group_by: Grouping interval ('hour', 'day', 'week')
            min_severity: Minimum severity to include
            
        Returns:
            Dictionary mapping components to time series data
        """
        # Set up time parameters
        end_time = time.time()
        start_time = end_time - (days * 86400)  # 86400 seconds in a day
        
        # Query errors in the time range
        errors = self.error_log.query(
            min_severity=min_severity,
            start_time=start_time,
            end_time=end_time
        )
        
        # Group by component and time
        component_trends = defaultdict(lambda: defaultdict(int))
        
        for error in errors:
            # Get timestamp and component
            timestamp = error.get("timestamp", 0)
            component = error.get("component", "unknown")
            
            # Convert to datetime
            dt = datetime.fromtimestamp(timestamp)
            
            # Create time bucket based on group_by
            if group_by == "hour":
                bucket = dt.strftime("%Y-%m-%d %H:00")
            elif group_by == "week":
                # Start of the week (Monday)
                start_of_week = dt - timedelta(days=dt.weekday())
                bucket = start_of_week.strftime("%Y-%m-%d")
            else:  # Default to day
                bucket = dt.strftime("%Y-%m-%d")
            
            # Increment count
            component_trends[component][bucket] += 1
        
        # Convert to list format for easier plotting
        result = {}
        for component, trends in component_trends.items():
            # Sort by time bucket
            sorted_trends = sorted(trends.items())
            result[component] = sorted_trends
        
        return result
    
    def get_severity_distribution(
        self,
        component: Optional[str] = None,
        days: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Get distribution of errors by severity.
        
        Args:
            component: Optional component filter
            days: Optional time range in days
            
        Returns:
            Dictionary mapping severity levels to counts
        """
        # Set up time parameters
        time_filter = {}
        if days:
            time_filter["start_time"] = time.time() - (days * 86400)
        
        # Query errors
        errors = self.error_log.query(
            component=component,
            **time_filter
        )
        
        # Count by severity
        severity_counts = defaultdict(int)
        for error in errors:
            severity = error.get("severity", "UNKNOWN")
            severity_counts[severity] += 1
        
        return dict(severity_counts)
    
    def get_component_distribution(
        self,
        min_severity: Optional[str] = None,
        days: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Get distribution of errors by component.
        
        Args:
            min_severity: Optional minimum severity filter
            days: Optional time range in days
            
        Returns:
            Dictionary mapping components to counts
        """
        # Set up time parameters
        time_filter = {}
        if days:
            time_filter["start_time"] = time.time() - (days * 86400)
        
        # Query errors
        errors = self.error_log.query(
            min_severity=min_severity,
            **time_filter
        )
        
        # Count by component
        component_counts = defaultdict(int)
        for error in errors:
            component = error.get("component", "unknown")
            component_counts[component] += 1
        
        return dict(component_counts)
    
    def get_error_patterns(
        self,
        min_severity: Optional[str] = None,
        min_occurrences: int = 2,
        days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns in error messages.
        
        Args:
            min_severity: Optional minimum severity filter
            min_occurrences: Minimum number of occurrences to consider a pattern
            days: Optional time range in days
            
        Returns:
            List of detected patterns with counts
        """
        # Set up time parameters
        time_filter = {}
        if days:
            time_filter["start_time"] = time.time() - (days * 86400)
        
        # Query errors
        errors = self.error_log.query(
            min_severity=min_severity,
            **time_filter
        )
        
        # Group similar messages
        message_groups = defaultdict(list)
        for error in errors:
            # Use the first 50 chars of the message as a simple pattern key
            message = error.get("message", "")
            pattern_key = message[:50].lower()
            message_groups[pattern_key].append(error)
        
        # Filter by minimum occurrences
        patterns = []
        for pattern_key, group in message_groups.items():
            if len(group) >= min_occurrences:
                patterns.append({
                    "pattern": pattern_key,
                    "count": len(group),
                    "examples": group[:3],  # Include up to 3 examples
                    "components": list(set(e.get("component", "unknown") for e in group)),
                    "severity_levels": list(set(e.get("severity", "UNKNOWN") for e in group))
                })
        
        # Sort by count (descending)
        patterns.sort(key=lambda p: p["count"], reverse=True)
        
        return patterns
    
    def get_error_summary(
        self,
        days: int = 7,
        min_severity: Optional[str] = "WARNING"
    ) -> Dict[str, Any]:
        """
        Get a comprehensive error summary.
        
        Args:
            days: Number of days to analyze
            min_severity: Minimum severity to include
            
        Returns:
            Dictionary with summary information
        """
        # Set up time parameters
        end_time = time.time()
        start_time = end_time - (days * 86400)
        
        # Query errors in the time range
        errors = self.error_log.query(
            min_severity=min_severity,
            start_time=start_time,
            end_time=end_time
        )
        
        # Early return if no errors
        if not errors:
            return {
                "total_errors": 0,
                "days_analyzed": days,
                "severity_distribution": {},
                "component_distribution": {},
                "most_recent": None,
                "error_rate_per_day": 0,
                "peak_day": None,
                "peak_day_count": 0
            }
        
        # Calculate basic stats
        severity_distribution = self.get_severity_distribution(days=days)
        component_distribution = self.get_component_distribution(
            min_severity=min_severity, 
            days=days
        )
        
        # Get most recent error
        most_recent = max(errors, key=lambda e: e.get("timestamp", 0))
        
        # Calculate error rate per day
        error_rate = len(errors) / days if days > 0 else 0
        
        # Find peak day
        day_counts = defaultdict(int)
        peak_day = None
        peak_day_count = 0
        
        for error in errors:
            timestamp = error.get("timestamp", 0)
            dt = datetime.fromtimestamp(timestamp)
            day = dt.strftime("%Y-%m-%d")
            day_counts[day] += 1
            
            if day_counts[day] > peak_day_count:
                peak_day = day
                peak_day_count = day_counts[day]
        
        return {
            "total_errors": len(errors),
            "days_analyzed": days,
            "severity_distribution": severity_distribution,
            "component_distribution": component_distribution,
            "most_recent": most_recent,
            "error_rate_per_day": error_rate,
            "peak_day": peak_day,
            "peak_day_count": peak_day_count
        }
    
    def export_to_csv(self, filepath: str, days: Optional[int] = None) -> bool:
        """
        Export error logs to CSV file.
        
        Args:
            filepath: Path to output CSV file
            days: Optional number of days to include
            
        Returns:
            True if export was successful
        """
        try:
            # Set up time parameters
            time_filter = {}
            if days:
                time_filter["start_time"] = time.time() - (days * 86400)
            
            # Query errors
            errors = self.error_log.query(**time_filter)
            
            # Write to CSV
            with open(filepath, 'w', newline='') as csvfile:
                # Define fields
                fieldnames = [
                    'timestamp', 'datetime', 'severity', 'component', 
                    'message', 'context'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for error in errors:
                    # Format context as string
                    context_str = json.dumps(error.get("context", {}))
                    
                    # Write row
                    writer.writerow({
                        'timestamp': error.get("timestamp", ""),
                        'datetime': error.get("datetime", ""),
                        'severity': error.get("severity", ""),
                        'component': error.get("component", ""),
                        'message': error.get("message", ""),
                        'context': context_str
                    })
            
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    def export_to_html(
        self,
        filepath: str,
        days: Optional[int] = None,
        min_severity: Optional[str] = None,
        include_summary: bool = True
    ) -> bool:
        """
        Export error logs to HTML report.
        
        Args:
            filepath: Path to output HTML file
            days: Optional number of days to include
            min_severity: Minimum severity to include
            include_summary: Whether to include summary statistics
            
        Returns:
            True if export was successful
        """
        try:
            # Set up time parameters
            time_filter = {}
            if days:
                time_filter["start_time"] = time.time() - (days * 86400)
            
            # Query errors
            errors = self.error_log.query(min_severity=min_severity, **time_filter)
            
            # Generate summary if requested
            summary = None
            if include_summary:
                summary = self.get_error_summary(
                    days=days or 7,
                    min_severity=min_severity
                )
            
            # Generate HTML
            html = self._generate_html_report(errors, summary)
            
            # Write to file
            with open(filepath, 'w') as f:
                f.write(html)
            
            return True
        except Exception as e:
            print(f"Error exporting to HTML: {e}")
            return False
    
    def _generate_html_report(
        self,
        errors: List[Dict[str, Any]],
        summary: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate HTML report from error data.
        
        Args:
            errors: List of error entries
            summary: Optional summary statistics
            
        Returns:
            HTML string
        """
        # Start HTML document
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Error Log Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .summary {
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                }
                .summary-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 10px;
                }
                .summary-item {
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                .summary-value {
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                .summary-label {
                    font-size: 14px;
                    color: #666;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    text-align: left;
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .severity-DEBUG { color: #6c757d; }
                .severity-INFO { color: #17a2b8; }
                .severity-WARNING { color: #f39c12; background-color: rgba(243, 156, 18, 0.1); }
                .severity-ERROR { color: #e74c3c; background-color: rgba(231, 76, 60, 0.1); }
                .severity-CRITICAL { color: white; background-color: #e74c3c; }
                .timestamp {
                    font-size: 12px;
                    color: #666;
                }
                .context {
                    font-family: monospace;
                    font-size: 12px;
                    white-space: pre-wrap;
                    background-color: #f8f9fa;
                    padding: 8px;
                    border-radius: 3px;
                    max-height: 100px;
                    overflow-y: auto;
                }
            </style>
        </head>
        <body>
            <h1>Error Log Report</h1>
            <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        """
        
        # Add summary section if provided
        if summary:
            html += """
            <div class="summary">
                <h2>Summary</h2>
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-value">""" + str(summary.get("total_errors", 0)) + """</div>
                        <div class="summary-label">Total Errors</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">""" + str(round(summary.get("error_rate_per_day", 0), 1)) + """</div>
                        <div class="summary-label">Errors per Day</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">""" + str(summary.get("days_analyzed", 0)) + """</div>
                        <div class="summary-label">Days Analyzed</div>
                    </div>
            """
            
            # Add peak day if available
            if summary.get("peak_day"):
                html += """
                    <div class="summary-item">
                        <div class="summary-value">""" + str(summary.get("peak_day_count", 0)) + """</div>
                        <div class="summary-label">Peak Day: """ + str(summary.get("peak_day", "")) + """</div>
                    </div>
                """
            
            html += """
                </div>
                
                <h3>Severity Distribution</h3>
                <table>
                    <tr>
                        <th>Severity</th>
                        <th>Count</th>
                    </tr>
            """
            
            # Add severity distribution
            for severity, count in summary.get("severity_distribution", {}).items():
                html += f"""
                    <tr>
                        <td class="severity-{severity}">{severity}</td>
                        <td>{count}</td>
                    </tr>
                """
            
            html += """
                </table>
                
                <h3>Component Distribution</h3>
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Count</th>
                    </tr>
            """
            
            # Add component distribution
            for component, count in sorted(
                summary.get("component_distribution", {}).items(),
                key=lambda x: x[1],
                reverse=True
            ):
                html += f"""
                    <tr>
                        <td>{component}</td>
                        <td>{count}</td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        # Add error log table
        html += """
            <h2>Error Log Entries</h2>
            <table>
                <tr>
                    <th>Timestamp</th>
                    <th>Severity</th>
                    <th>Component</th>
                    <th>Message</th>
                    <th>Context</th>
                </tr>
        """
        
        # Add error entries
        for error in errors:
            # Format timestamp
            timestamp = error.get("timestamp", 0)
            dt = datetime.fromtimestamp(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Format severity with class
            severity = error.get("severity", "UNKNOWN")
            severity_html = f'<span class="severity-{severity}">{severity}</span>'
            
            # Format context as JSON
            context = error.get("context", {})
            context_html = f'<div class="context">{json.dumps(context, indent=2)}</div>' if context else ''
            
            # Add table row
            html += f"""
                <tr>
                    <td class="timestamp">{formatted_time}</td>
                    <td>{severity_html}</td>
                    <td>{error.get("component", "unknown")}</td>
                    <td>{error.get("message", "")}</td>
                    <td>{context_html}</td>
                </tr>
            """
        
        # Close table and HTML document
        html += """
            </table>
        </body>
        </html>
        """
        
        return html


# Example usage
if __name__ == "__main__":
    from monitoring.error_log import ErrorLog, ErrorSeverity
    
    # Create a sample error log
    error_log = ErrorLog()
    
    # Add some sample errors
    for i in range(10):
        # Add varied errors
        if i % 5 == 0:
            error_log.critical(f"Critical error {i}", component="system")
        elif i % 4 == 0:
            error_log.error(f"Error {i}", component="database")
        elif i % 3 == 0:
            error_log.warning(f"Warning {i}", component="network")
        elif i % 2 == 0:
            error_log.info(f"Info {i}", component="ui")
        else:
            error_log.debug(f"Debug {i}", component="misc")
    
    # Create visualizer
    visualizer = ErrorVisualizer(error_log)
    
    # Get and print summary
    summary = visualizer.get_error_summary()
    print("Error Summary:")
    print(f"- Total errors: {summary['total_errors']}")
    print(f"- By severity: {summary['severity_distribution']}")
    
    # Export to HTML
    if visualizer.export_to_html("error_report.html"):
        print("Exported HTML report to error_report.html")
