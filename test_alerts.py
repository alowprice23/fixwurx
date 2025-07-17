"""
test_alerts.py
────────────────────────────
Script to generate test metrics that trigger alerts in the FixWurx system.
This allows for demonstration and validation of the alert threshold system.
"""

import time
import random
import argparse
import sys
import asyncio
from datetime import datetime

# Try to import from monitoring modules
try:
    from monitoring.dashboard import MetricsBus as OriginalMetricsBus
    from monitoring.alerts import AlertConfig, AlertCondition, AlertSeverity, default_manager
except ImportError:
    print("Error: Could not import monitoring modules.")
    print("Make sure you're running this script from the FixWurx root directory.")
    sys.exit(1)

# Create a testing version of MetricsBus that doesn't rely on asyncio.create_task
class TestMetricsBus:
    """Simplified version of MetricsBus for testing."""
    
    def __init__(self):
        """Initialize metrics bus without asyncio task."""
        self._history = []
        log("Created test metrics bus")
        
    def send(self, name, value, tags=None, metric_type="SYSTEM"):
        """Send a metric and check for alerts."""
        log(f"Sending metric: {name}={value}")
        
        # Create a simplified record
        rec = {
            "ts": time.time(),
            "name": name,
            "value": value,
            "type": metric_type,
            "tags": tags or {},
        }
        self._history.append(rec)
        
        # Check for alerts
        from monitoring.alerts import evaluate_metric, get_active_alerts
        alerts = evaluate_metric(name, value, tags)
        
        if alerts:
            log(f"🚨 ALERT! Triggered {len(alerts)} alerts:")
            for alert in alerts:
                log(f"  - {alert.config.severity.upper()}: {alert.config.name} - {alert.config.description}")
                log(f"    Metric: {name}={value}, Threshold: {alert.config.condition} {alert.config.threshold}")
            
        # Periodically show all active alerts
        if random.random() < 0.2:  # About 20% of the time
            active = get_active_alerts()
            if active:
                log(f"📊 Currently {len(active)} active alerts:")
                for a in active:
                    log(f"  - {a['severity'].upper()}: {a['name']} ({a['status']})")

def log(message):
    """Log a message with timestamp."""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")

async def generate_test_metrics(bus, duration=60, interval=5):
    """
    Generate test metrics that will trigger alerts.
    
    Args:
        bus: The metrics bus to send metrics to
        duration: How long to run the test in seconds
        interval: Seconds between metric updates
    """
    log("Starting metric generation...")
    
    # Track start time
    start_time = time.time()
    
    # Create a sequence of system health values that will trigger alerts
    health_sequence = [
        # Start with good health
        (0.9, 10),   # (value, seconds to maintain)
        # Drop to warning level
        (0.5, 20),   # Should trigger warning alert after 2 minutes (which we accelerate)
        # Recover briefly
        (0.8, 5),
        # Drop to critical
        (0.05, 15),  # Should trigger critical alert after 1 minute
        # Recover to warning
        (0.4, 10),
    ]
    
    # Create error rate sequence
    error_sequence = [
        (0.01, 15),  # Low error rate
        (0.06, 15),  # High error rate > 5% (should trigger critical alert)
        (0.02, 15),  # Return to normal
        (0.08, 15),  # Very high error rate
    ]
    
    # Track current position in sequences
    health_idx = 0
    health_time = 0
    error_idx = 0
    error_time = 0
    
    # Generate metrics until duration expires
    while time.time() - start_time < duration:
        # Get current health target and error rate target
        health_target, health_duration = health_sequence[health_idx]
        error_target, error_duration = error_sequence[error_idx]
        
        # Add small random fluctuation
        health = max(0, min(1, health_target + random.uniform(-0.05, 0.05)))
        error_rate = max(0, min(1, error_target + random.uniform(-0.005, 0.005)))
        
        # Generate random agent count (0-5)
        active_agents = random.randint(0, 5)
        
        # Generate entropy reduction rate
        if random.random() < 0.2:  # 20% chance of stalled entropy
            entropy_reduction = 0.005  # Very low reduction rate
        else:
            entropy_reduction = random.uniform(0.05, 0.2)  # Normal reduction rate
        
        # Send metrics
        log(f"Sending metrics - Health: {health:.2f}, Error Rate: {error_rate:.3f}, "
            f"Active Agents: {active_agents}, Entropy Reduction: {entropy_reduction:.3f}")
        
        bus.send("system.health", health, {"source": "test_script"})
        bus.send("system.error_rate", error_rate, {"source": "test_script"})
        bus.send("system.active_agents", float(active_agents), {"source": "test_script"})
        bus.send("entropy.reduction_rate", entropy_reduction, {"source": "test_script"})
        
        # Update sequence positions
        health_time += interval
        error_time += interval
        
        if health_time >= health_duration:
            health_time = 0
            health_idx = (health_idx + 1) % len(health_sequence)
            
        if error_time >= error_duration:
            error_time = 0
            error_idx = (error_idx + 1) % len(error_sequence)
        
        # Sleep for interval
        log(f"Waiting {interval} seconds...")
        await asyncio.sleep(interval)
    
    log("Metric generation complete.")

async def async_main():
    """Async entry point."""
    parser = argparse.ArgumentParser(description="Generate test metrics to trigger alerts")
    parser.add_argument("--duration", type=int, default=120, 
                        help="Duration in seconds to run the test")
    parser.add_argument("--interval", type=int, default=5,
                        help="Interval in seconds between metric updates")
    args = parser.parse_args()
    
    # Create metrics bus
    bus = TestMetricsBus()
    log("Metrics bus created")
    
    # Run test metrics generator
    await generate_test_metrics(bus, args.duration, args.interval)
    
    log("Test complete. Check the dashboard for alerts.")

def main():
    """Main entry point that runs the async function."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
