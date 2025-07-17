# FixWurx Alert System

The FixWurx Alert System provides real-time monitoring and notification capabilities for critical system metrics. This document describes the alert system architecture, configuration, and usage.

## Architecture

The alert system consists of the following components:

1. **Alert Manager**: Core component responsible for managing alert configurations, evaluating metrics against thresholds, and maintaining alert status.

2. **Dashboard Integration**: The alert system is integrated with the FixWurx Dashboard, providing a user interface for viewing and managing alerts.

3. **Metrics Integration**: The system monitor's metrics bus is connected to the alert system, allowing metric values to be evaluated against alert thresholds.

4. **Configuration**: Alert thresholds and settings are configurable via the `system_config.yaml` file.

## Alert Types and Severity Levels

Alerts can have one of three severity levels:

- **Info**: Informational alerts that don't require immediate action
- **Warning**: Alerts that indicate potential issues that should be addressed
- **Critical**: High-priority alerts that require immediate attention

## Default Alert Thresholds

The system comes with several pre-configured alert thresholds:

| Alert | Metric | Condition | Threshold | Severity | Duration |
|-------|--------|-----------|-----------|----------|----------|
| System Health Critical | system.health | Below | 0.1 | Critical | 60s |
| System Health Warning | system.health | Below | 0.6 | Warning | 120s |
| No Active Agents | system.active_agents | Equals | 0 | Warning | 300s |
| High Error Rate | system.error_rate | Above | 0.05 | Critical | 60s |
| Entropy Reduction Stalled | entropy.reduction_rate | Below | 0.01 | Warning | 1800s |
| Agent Error | agent.status | Below | 0 | Warning | 60s |

## Alert System Features

### Real-time Monitoring
- Continuous evaluation of metrics against thresholds
- Configurable duration requirements (alerts trigger only when conditions persist)
- Alert cooldown periods to prevent alert storms

### Alert Management
- Acknowledge alerts to indicate they are being addressed
- Resolve alerts when issues are fixed
- Alert history for tracking past incidents

### User Interface
- Active alerts dashboard with severity indicators
- Alert configuration management
- Historical alert view with filtering

## Using the Alert System

### Viewing Alerts
1. Open the FixWurx Dashboard at http://localhost:8081/
2. Navigate to the "Alerts" tab to view active alerts

### Creating Custom Alert Configurations
1. In the Alerts page, click "New Alert"
2. Fill in the alert details:
   - Name and description
   - Metric name to monitor
   - Condition (above, below, equals)
   - Threshold value
   - Severity level
   - Duration (how long condition must persist before alerting)
3. Click "Save" to activate the alert

### Managing Alerts
- **Acknowledge**: Click the "Acknowledge" button to indicate you're addressing the alert
- **Resolve**: Click the "Resolve" button once the issue is fixed
- **Filter History**: Use the dropdown on the Alert History section to filter by severity

## Testing the Alert System

A test script is included to simulate metrics and trigger alerts:

```bash
python test_alerts.py --duration 120 --interval 5
```

This will generate test metrics that will trigger various alerts based on the default thresholds.

## Integration with External Systems

The alert system can be integrated with external notification systems via webhooks:

1. Configure webhook endpoints in `system_config.yaml`:
   ```yaml
   alerts:
     webhook-endpoints:
       - "http://your-notification-service/api/hook"
   ```

2. Implement a webhook handler that processes the alert payload:
   ```json
   {
     "alert": {
       "id": "system_health_critical",
       "name": "System Health Critical",
       "severity": "critical",
       "metric_name": "system.health",
       "value": 0.05,
       "threshold": 0.1,
       "status": "active"
     },
     "timestamp": 1688571036.789
   }
   ```

## Customizing Alert Thresholds

Alert thresholds can be customized in the `system_config.yaml` file:

```yaml
alerts:
  critical-thresholds:
    error-rate: 0.05             # 5% error rate triggers critical alert
    system-health: 0.1           # Health score below 0.1 triggers critical alert
    active-agents: 0             # No active agents triggers warning alert
    entropy-reduction: 0.01      # Entropy reduction below 0.01 triggers warning
```

## Future Enhancements

Future versions of the alert system will include:
- Alert escalation workflows
- More complex condition types (rate of change, correlation)
- Alert templates for common monitoring scenarios
- Integration with ticketing systems
- Mobile notifications
