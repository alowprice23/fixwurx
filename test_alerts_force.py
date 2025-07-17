"""
test_alerts_force.py
────────────────────────────
Script to force trigger alerts in the FixWurx system by sending extreme metric values.
This provides a clear demonstration of the alert threshold system.
"""

import time
import random
import argparse
import sys
import asyncio
from datetime import datetime

# Try to import from monitoring modules
try:
    from monitoring.alerts import (
        AlertConfig, 
        AlertCondition, 
        AlertSeverity, 
        default_manager,
        get_active_alerts,
        get_alert_history
    )
except ImportError:
    print("Error: Could not import monitoring modules.")
    print("Make sure you're running this script from the FixWurx root directory.")
    sys.exit(1)

def log(message):
    """Log a message with timestamp."""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")

def print_box(title, content):
    """Print content in a box."""
    width = 80
    print("┌" + "─" * (width - 2) + "┐")
    print(f"│ {title.center(width - 4)} │")
    print("├" + "─" * (width - 2) + "┤")
    for line in content:
        print(f"│ {line.ljust(width - 4)} │")
    print("└" + "─" * (width - 2) + "┘")
    print()

def print_alert_configs():
    """Print current alert configurations."""
    configs = default_manager.get_configs()
    content = []
    
    for config in configs:
        line = f"{config['name']} - {config['condition']} {config['threshold']} ({config['severity'].upper()})"
        content.append(line)
    
    print_box("📋 ALERT CONFIGURATIONS", content)

def print_active_alerts():
    """Print currently active alerts."""
    alerts = get_active_alerts()
    content = []
    
    if not alerts:
        content.append("No active alerts")
    else:
        for alert in alerts:
            line = f"[{alert['severity'].upper()}] {alert['name']} - {alert['metric_name']}={alert['value']:.3f}"
            content.append(line)
    
    print_box("🚨 ACTIVE ALERTS", content)

def trigger_alert(metric_name, value, threshold_direction):
    """
    Send a metric value that will trigger an alert.
    
    Args:
        metric_name: Name of the metric to send
        value: Value to send
        threshold_direction: 'above' or 'below' to indicate which alerts to trigger
    """
    from monitoring.alerts import evaluate_metric
    
    log(f"TRIGGERING ALERT TEST: {metric_name}={value} ({threshold_direction})")
    
    # Evaluate the metric against alert thresholds
    alerts = evaluate_metric(metric_name, value, {"source": "force_test"})
    
    if alerts:
        log(f"🚨 ALERT TRIGGERED! {len(alerts)} alerts:")
        for alert in alerts:
            severity_marker = "❗❗❗" if alert.config.severity == "critical" else "⚠️" if alert.config.severity == "warning" else "ℹ️"
            log(f"  {severity_marker} {alert.config.severity.upper()}: {alert.config.name}")
            log(f"    Description: {alert.config.description}")
            log(f"    Metric: {metric_name}={value}, Threshold: {alert.config.condition} {alert.config.threshold}")
    else:
        log(f"No alerts triggered for {metric_name}={value}")
    
    # Wait a moment for the alert to be processed
    time.sleep(1)
    
    # Show current active alerts
    print_active_alerts()
    print()

async def run_alert_tests():
    """Run a series of tests to trigger different alerts."""
    log("Starting alert system tests...")
    print()
    
    # Print current alert configurations
    print_alert_configs()
    
    # Test cases that should trigger alerts
    test_cases = [
        # System health critical (below 0.1)
        ("system.health", 0.05, "below"),
        
        # System health warning (below 0.6)
        ("system.health", 0.5, "below"),
        
        # High error rate (above 0.05 or 5%)
        ("system.error_rate", 0.08, "above"),
        
        # No active agents
        ("system.active_agents", 0.0, "equals"),
        
        # Entropy reduction stalled
        ("entropy.reduction_rate", 0.005, "below"),
    ]
    
    # Run each test case
    for metric_name, value, direction in test_cases:
        trigger_alert(metric_name, value, direction)
        
        # Brief pause between tests
        await asyncio.sleep(2)
    
    # Show final summary
    print_active_alerts()
    
    # Show alert history
    history = get_alert_history()
    content = []
    
    if not history:
        content.append("No alert history")
    else:
        for alert in history[:10]:  # Show up to 10 most recent
            status = alert['status'].upper()
            line = f"[{alert['severity'].upper()}] {alert['name']} - {status}"
            content.append(line)
    
    print_box("📜 ALERT HISTORY", content)
    
    log("Alert system tests complete.")

async def async_main():
    """Async entry point."""
    await run_alert_tests()

def main():
    """Main entry point that runs the async function."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
