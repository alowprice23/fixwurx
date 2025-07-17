"""
Demonstration of Time Series Database Integration with Auditor

This script demonstrates how the time series database works with the auditor and 
error reporting system, showing:
1. How error data is stored in time series format
2. How to query historical error trends
3. How time series data enhances error reporting
4. How to analyze error patterns over time
"""

import os
import sys
import datetime
import random
import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [TimeSeriesDemo] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('time_series_demo')

# Import required components
from sensor_registry import ErrorReport, SensorRegistry, create_sensor_registry
from component_sensors import (
    ObligationLedgerSensor, EnergyCalculatorSensor, ProofMetricsSensor,
    MetaAwarenessSensor
)
from time_series_database import TimeSeriesDatabase
from llm_integrations import LLMManager

# Mock LLM Manager for demo
class MockLLMManager(LLMManager):
    def __init__(self):
        logger.info("Initialized MockLLMManager")
    
    def chat(self, role: str, content: str, task_type: str = None, 
            complexity: str = None):
        # Just a simple mock response class
        class MockResponse:
            def __init__(self, text):
                self.text = text
                self.tokens = 0
                self.latency_ms = 0
                self.cost_usd = 0
                self.provider = "mock"
                self.model = "mock"
        
        return MockResponse("This is a mock LLM response.")

class TimeSeriesErrorDemonstration:
    """Demonstrates time series integration with error reporting."""
    
    def __init__(self):
        """Initialize the demonstration."""
        # Create time series database
        self.ts_db = TimeSeriesDatabase("auditor_data/time_series")
        
        # Create registry and sensor manager
        self.registry, self.sensor_manager = create_sensor_registry()
        
        # Create a mock LLM manager
        self.llm_manager = MockLLMManager()
        
        # Create time series for errors
        self.error_ts = self.ts_db.create_time_series(
            name="error_metrics",
            description="Error metrics from the auditor",
            unit="count"
        )
        
        self.component_error_ts = self.ts_db.create_time_series(
            name="component_errors",
            description="Errors by component",
            unit="count"
        )
        
        self.severity_ts = self.ts_db.create_time_series(
            name="error_severity",
            description="Error severity metrics",
            unit="count"
        )
        
        # Create some sensors
        self.sensors = {
            "obligation": ObligationLedgerSensor("ObligationLedger", {"enabled": True}),
            "energy": EnergyCalculatorSensor("EnergyCalculator", {"enabled": True}),
            "proof": ProofMetricsSensor("ProofMetrics", {"enabled": True}),
            "meta": MetaAwarenessSensor("MetaAwareness", {"enabled": True})
        }
        
        # Register sensors
        for sensor in self.sensors.values():
            self.registry.register_sensor(sensor)
        
        logger.info("Demonstration initialized")
    
    def generate_historical_errors(self, days: int = 7, errors_per_day: int = 5):
        """
        Generate historical error data.
        
        Args:
            days: Number of days of historical data
            errors_per_day: Average number of errors per day
        """
        logger.info(f"Generating {days} days of historical error data")
        
        # Error types by component
        error_types = {
            "ObligationLedger": ["CIRCULAR_DEPENDENCIES", "MISSING_OBLIGATIONS", "RULE_APPLICATION_FAILURE"],
            "EnergyCalculator": ["ENERGY_DIVERGENCE", "LAMBDA_EXCEEDS_THRESHOLD", "NEGATIVE_GRADIENT"],
            "ProofMetrics": ["COVERAGE_BELOW_THRESHOLD", "HIGH_BUG_PROBABILITY", "INSUFFICIENT_VERIFICATION"],
            "MetaAwareness": ["EXCESSIVE_DRIFT", "EXCESSIVE_PERTURBATION", "PHI_INCREASE"]
        }
        
        # Severity levels
        severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        severity_weights = [0.4, 0.3, 0.2, 0.1]  # Probability weights
        
        # Generate historical data
        now = datetime.datetime.now()
        
        # Time series data
        daily_error_counts = {}
        component_error_counts = {"ObligationLedger": 0, "EnergyCalculator": 0, "ProofMetrics": 0, "MetaAwareness": 0}
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        
        # Generate errors for each day
        for day in range(days):
            # Calculate date
            date = now - datetime.timedelta(days=days-day)
            date_str = date.strftime("%Y-%m-%d")
            daily_error_counts[date_str] = 0
            
            # Generate random number of errors for this day
            num_errors = max(1, int(random.gauss(errors_per_day, errors_per_day / 3)))
            
            for _ in range(num_errors):
                # Random time within the day
                error_time = date + datetime.timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )
                
                # Random component
                component = random.choice(list(error_types.keys()))
                
                # Random error type for this component
                error_type = random.choice(error_types[component])
                
                # Random severity based on weights
                severity = random.choices(severities, severity_weights)[0]
                
                # Generate error report
                sensor_id = f"{component.lower()}_sensor"
                error = ErrorReport(
                    sensor_id=sensor_id,
                    component_name=component,
                    error_type=error_type,
                    severity=severity,
                    details={"message": f"Test {error_type} error"},
                    context={"source_file": f"{component.lower()}.py"}
                )
                
                # Add to registry
                self.registry.error_reports.append(error)
                
                # Update counts
                daily_error_counts[date_str] += 1
                component_error_counts[component] += 1
                severity_counts[severity] += 1
            
            # Add daily data to time series
            self.ts_db.add_point(
                series_name="error_metrics",
                timestamp=date,
                values={"total_errors": daily_error_counts[date_str]}
            )
            
            # Add component data to time series (normalize to percentage)
            total = sum(component_error_counts.values())
            component_percentages = {k: (v / total) * 100 for k, v in component_error_counts.items()}
            
            self.ts_db.add_point(
                series_name="component_errors",
                timestamp=date,
                values=component_percentages
            )
            
            # Add severity data to time series (normalize to percentage)
            total = sum(severity_counts.values())
            severity_percentages = {k: (v / total) * 100 for k, v in severity_counts.items()}
            
            self.ts_db.add_point(
                series_name="error_severity",
                timestamp=date,
                values=severity_percentages
            )
        
        logger.info(f"Generated {sum(daily_error_counts.values())} historical errors")
    
    def generate_real_time_error(self, component: str, error_type: str, severity: str):
        """
        Generate a real-time error and update time series.
        
        Args:
            component: Component name
            error_type: Error type
            severity: Error severity
        """
        sensor_id = f"{component.lower()}_sensor"
        sensor = self.sensors.get(component.lower())
        
        if not sensor:
            logger.error(f"Sensor for component {component} not found")
            return None
        
        # Generate error report
        error = sensor.report_error(
            error_type=error_type,
            severity=severity,
            details={"message": f"Real-time {error_type} error"},
            context={"source_file": f"{component.lower()}.py"}
        )
        
        # Collect errors from sensors
        new_errors = self.sensor_manager.collect_errors(force=True)
        
        # Update time series
        now = datetime.datetime.now()
        
        # Get current daily count
        date_str = now.strftime("%Y-%m-%d")
        
        # Get latest value
        latest = self.ts_db.get_latest_metric_value("error_metrics", "total_errors")
        daily_count = 1
        if latest and latest[0].strftime("%Y-%m-%d") == date_str:
            daily_count = latest[1] + 1
        
        # Update error metrics
        self.ts_db.add_point(
            series_name="error_metrics",
            timestamp=now,
            values={"total_errors": daily_count}
        )
        
        # Update component counts
        components = ["ObligationLedger", "EnergyCalculator", "ProofMetrics", "MetaAwareness"]
        component_counts = {c: 0 for c in components}
        
        # Count errors by component
        for e in self.registry.error_reports:
            if e.component_name in component_counts:
                component_counts[e.component_name] += 1
        
        # Calculate percentages
        total = sum(component_counts.values())
        component_percentages = {k: (v / total) * 100 for k, v in component_counts.items()}
        
        # Update component errors time series
        self.ts_db.add_point(
            series_name="component_errors",
            timestamp=now,
            values=component_percentages
        )
        
        # Update severity counts
        severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        severity_counts = {s: 0 for s in severities}
        
        # Count errors by severity
        for e in self.registry.error_reports:
            if e.severity in severity_counts:
                severity_counts[e.severity] += 1
        
        # Calculate percentages
        total = sum(severity_counts.values())
        severity_percentages = {k: (v / total) * 100 for k, v in severity_counts.items()}
        
        # Update severity time series
        self.ts_db.add_point(
            series_name="error_severity",
            timestamp=now,
            values=severity_percentages
        )
        
        logger.info(f"Generated real-time error: {error_type} in {component} with {severity} severity")
        
        return error
    
    def analyze_error_trends(self):
        """Analyze error trends from time series data."""
        logger.info("Analyzing error trends")
        
        # Get trend analysis for total errors
        trend = self.ts_db.get_trend_analysis("error_metrics", "total_errors")
        
        print("\n=== ERROR TREND ANALYSIS ===")
        print(f"Total Errors Trend: {trend['trend']}")
        print(f"Slope: {trend['slope']:.4f}")
        print(f"Mean: {trend['mean']:.2f} errors per day")
        print(f"Standard Deviation: {trend['std']:.2f}")
        print(f"Min: {trend['min']:.2f}, Max: {trend['max']:.2f}")
        
        # Analyze component trends
        components = ["ObligationLedger", "EnergyCalculator", "ProofMetrics", "MetaAwareness"]
        print("\n=== COMPONENT ERROR TRENDS ===")
        
        for component in components:
            trend = self.ts_db.get_trend_analysis("component_errors", component)
            print(f"{component}: {trend['trend']} (Mean: {trend['mean']:.2f}%)")
        
        # Analyze severity trends
        severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        print("\n=== SEVERITY TRENDS ===")
        
        for severity in severities:
            trend = self.ts_db.get_trend_analysis("error_severity", severity)
            print(f"{severity}: {trend['trend']} (Mean: {trend['mean']:.2f}%)")
    
    def visualize_error_trends(self):
        """Visualize error trends using matplotlib."""
        try:
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
            
            # Get time series data
            error_metrics = self.ts_db.get_time_series("error_metrics").to_dataframe()
            component_errors = self.ts_db.get_time_series("component_errors").to_dataframe()
            severity_errors = self.ts_db.get_time_series("error_severity").to_dataframe()
            
            # Plot daily error counts
            ax1.plot(error_metrics.index, error_metrics["total_errors"], 'o-', color='blue')
            ax1.set_title('Daily Error Counts')
            ax1.set_ylabel('Number of Errors')
            ax1.grid(True)
            
            # Plot component percentages
            component_errors.plot(kind='area', stacked=True, ax=ax2, colormap='viridis')
            ax2.set_title('Error Distribution by Component')
            ax2.set_ylabel('Percentage')
            ax2.set_ylim(0, 100)
            ax2.grid(True)
            
            # Plot severity percentages
            severity_errors.plot(kind='area', stacked=True, ax=ax3, colormap='RdYlGn_r')
            ax3.set_title('Error Distribution by Severity')
            ax3.set_ylabel('Percentage')
            ax3.set_ylim(0, 100)
            ax3.grid(True)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig('error_trends.png')
            logger.info("Saved error trend visualization to 'error_trends.png'")
            
            print("\nVisualization saved to 'error_trends.png'")
        except Exception as e:
            logger.error(f"Failed to visualize error trends: {e}")
    
    def enhance_error_with_time_series(self, error: ErrorReport):
        """
        Enhance an error report with time series data.
        
        Args:
            error: The error report to enhance
        """
        # Get component history
        component = error.component_name
        component_values = self.ts_db.get_metric_values("component_errors", component)
        
        # Get severity history
        severity = error.severity
        severity_values = self.ts_db.get_metric_values("error_severity", severity)
        
        # Calculate trends
        component_trend = self.ts_db.get_trend_analysis("component_errors", component)
        severity_trend = self.ts_db.get_trend_analysis("error_severity", severity)
        
        # Enhance error report with time series context
        if not error.context:
            error.context = {}
        
        error.context["time_series_data"] = {
            "component_trend": component_trend["trend"],
            "component_mean": component_trend["mean"],
            "severity_trend": severity_trend["trend"],
            "severity_mean": severity_trend["mean"]
        }
        
        # Add impact based on time series data
        error.impact = {
            "severity": severity,
            "scope": "multi_component" if component_trend["mean"] > 30 else "single_component",
            "affected_components": [component],
            "affected_functionality": [],
            "user_impact": f"This error is part of a {component_trend['trend'].lower()} trend in {component} errors",
            "system_impact": f"This {severity} error is part of a {severity_trend['trend'].lower()} trend in {severity} errors"
        }
        
        # Add historical context to related errors
        # In a real system, we would look up actual related errors
        error.related_errors = []
        
        # Generate recommendations based on trend analysis
        recommendations = []
        
        if component_trend["trend"] == "INCREASING":
            recommendations.append(f"Investigate rising trend in {component} errors (up by {component_trend['slope']:.4f})")
        
        if severity_trend["trend"] == "INCREASING" and severity in ["HIGH", "CRITICAL"]:
            recommendations.append(f"Prioritize fixing {severity} errors as they show an increasing trend")
        
        if component_trend["mean"] > 40:
            recommendations.append(f"Consider audit of {component} as it accounts for {component_trend['mean']:.2f}% of all errors")
        
        error.recommendations = recommendations
        
        logger.info(f"Enhanced error report with time series data: {error.error_id}")
        
        return error
    
    def report_error_with_time_series(self):
        """Generate a sample error report enhanced with time series data."""
        # Generate a real-time error
        error = self.generate_real_time_error("energy", "ENERGY_DIVERGENCE", "HIGH")
        
        if not error:
            logger.error("Failed to generate error")
            return
        
        # Enhance with time series data
        enhanced_error = self.enhance_error_with_time_series(error)
        
        # Display enhanced error report
        print("\n=== ENHANCED ERROR REPORT WITH TIME SERIES DATA ===")
        print(f"Error ID: {enhanced_error.error_id}")
        print(f"Component: {enhanced_error.component_name}")
        print(f"Type: {enhanced_error.error_type}")
        print(f"Severity: {enhanced_error.severity}")
        print(f"Timestamp: {enhanced_error.timestamp}")
        
        print("\nTime Series Context:")
        ts_data = enhanced_error.context.get("time_series_data", {})
        print(f"Component Trend: {ts_data.get('component_trend')} (Mean: {ts_data.get('component_mean', 0):.2f}%)")
        print(f"Severity Trend: {ts_data.get('severity_trend')} (Mean: {ts_data.get('severity_mean', 0):.2f}%)")
        
        print("\nImpact:")
        impact = enhanced_error.impact or {}
        print(f"Scope: {impact.get('scope', 'N/A')}")
        print(f"User Impact: {impact.get('user_impact', 'N/A')}")
        print(f"System Impact: {impact.get('system_impact', 'N/A')}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(enhanced_error.recommendations or [], 1):
            print(f"{i}. {rec}")
    
    def run_demo(self):
        """Run the full demonstration."""
        print("\n*** TIME SERIES DATABASE INTEGRATION WITH AUDITOR DEMONSTRATION ***\n")
        
        # Step 1: Generate historical error data
        print("Step 1: Generating historical error data...")
        self.generate_historical_errors(days=14, errors_per_day=8)
        
        # Step 2: Analyze error trends
        print("\nStep 2: Analyzing error trends from time series data...")
        self.analyze_error_trends()
        
        # Step 3: Visualize error trends
        print("\nStep 3: Visualizing error trends...")
        self.visualize_error_trends()
        
        # Step 4: Show error reporting with time series enhancement
        print("\nStep 4: Demonstrating error reporting with time series enhancement...")
        self.report_error_with_time_series()
        
        print("\n*** DEMONSTRATION COMPLETE ***\n")


if __name__ == "__main__":
    # Run the demonstration
    demo = TimeSeriesErrorDemonstration()
    demo.run_demo()
