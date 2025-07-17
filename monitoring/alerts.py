"""
monitoring/alerts.py
────────────────────────────
Alert system for the FixWurx dashboard that monitors metrics,
evaluates thresholds, and triggers notifications for critical events.

Highlights
──────────
• Configurable alert thresholds for all metric types
• Multi-level severity (info, warning, critical)
• Alert history and status tracking
• Notification delivery via webhooks and UI
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("alerts")

# ---------------------------------------------------------------------------—
# Data Models
# ---------------------------------------------------------------------------—
class AlertSeverity(str, Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    """Status of an alert."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

class AlertCondition(str, Enum):
    """Types of alert conditions."""
    ABOVE = "above"
    BELOW = "below"
    EQUALS = "equals"
    CHANGES = "changes"
    RATE_OF_CHANGE = "rate_of_change"

class AlertConfig:
    """Configuration for an alert threshold."""
    
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        metric_name: str,
        condition: AlertCondition,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        duration_seconds: int = 0,  # 0 means immediate alert
        tags: Optional[Dict[str, str]] = None,
        enabled: bool = True,
        cooldown_seconds: int = 300,  # 5 minutes between repeated alerts
    ):
        self.id = id
        self.name = name
        self.description = description
        self.metric_name = metric_name
        self.condition = condition if isinstance(condition, AlertCondition) else AlertCondition(condition)
        self.threshold = threshold
        self.severity = severity if isinstance(severity, AlertSeverity) else AlertSeverity(severity)
        self.duration_seconds = duration_seconds
        self.tags = tags or {}
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metric_name": self.metric_name,
            "condition": self.condition.value if isinstance(self.condition, Enum) else self.condition,
            "threshold": self.threshold,
            "severity": self.severity.value if isinstance(self.severity, Enum) else self.severity,
            "duration_seconds": self.duration_seconds,
            "tags": self.tags,
            "enabled": self.enabled,
            "cooldown_seconds": self.cooldown_seconds,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertConfig":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            metric_name=data["metric_name"],
            condition=data["condition"],
            threshold=data["threshold"],
            severity=data["severity"],
            duration_seconds=data.get("duration_seconds", 0),
            tags=data.get("tags", {}),
            enabled=data.get("enabled", True),
            cooldown_seconds=data.get("cooldown_seconds", 300),
        )


class Alert:
    """An active or historical alert."""
    
    def __init__(
        self,
        config: AlertConfig,
        value: float,
        status: AlertStatus = AlertStatus.ACTIVE,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
        resolved_at: Optional[float] = None,
        acknowledged_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.value = value
        self.status = status if isinstance(status, AlertStatus) else AlertStatus(status)
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or self.created_at
        self.resolved_at = resolved_at
        self.acknowledged_by = acknowledged_by
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.config.id,
            "name": self.config.name,
            "description": self.config.description,
            "metric_name": self.config.metric_name,
            "condition": self.config.condition.value if isinstance(self.config.condition, Enum) else self.config.condition,
            "threshold": self.config.threshold,
            "value": self.value,
            "severity": self.config.severity.value if isinstance(self.config.severity, Enum) else self.config.severity,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "acknowledged_by": self.acknowledged_by,
            "tags": self.config.tags,
            "metadata": self.metadata,
        }
        
    def acknowledge(self, user: str) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = user
        self.updated_at = time.time()
        
    def resolve(self) -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.updated_at = time.time()
        self.resolved_at = self.updated_at


class AlertManager:
    """
    Manages alert configurations, evaluates metrics against thresholds,
    and handles alert lifecycle.
    """
    
    def __init__(self, history_size: int = 1000) -> None:
        """Initialize alert manager."""
        self._configs: Dict[str, AlertConfig] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._history_size = history_size
        self._last_triggered: Dict[str, float] = {}
        self._metric_windows: Dict[str, List[Tuple[float, float]]] = {}  # (timestamp, value) pairs
        self._notification_handlers: List[Callable[[Alert], None]] = []
        
        # Default system alert configurations
        self._add_default_configs()
        
    def _add_default_configs(self) -> None:
        """Add default alert configurations."""
        defaults = [
            AlertConfig(
                id="system_health_critical",
                name="System Health Critical",
                description="System health is in critical state",
                metric_name="system.health",
                condition=AlertCondition.BELOW,
                threshold=0.1,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=60,  # Must be critical for 1 minute
            ),
            AlertConfig(
                id="system_health_warning",
                name="System Health Warning",
                description="System health is in warning state",
                metric_name="system.health",
                condition=AlertCondition.BELOW,
                threshold=0.6,
                severity=AlertSeverity.WARNING,
                duration_seconds=120,  # Must be warning for 2 minutes
            ),
            AlertConfig(
                id="no_active_agents",
                name="No Active Agents",
                description="No agents are currently active in the system",
                metric_name="system.active_agents",
                condition=AlertCondition.EQUALS,
                threshold=0,
                severity=AlertSeverity.WARNING,
                duration_seconds=300,  # Must be 0 for 5 minutes
            ),
            AlertConfig(
                id="high_error_rate",
                name="High Error Rate",
                description="System is experiencing a high error rate",
                metric_name="system.error_rate",
                condition=AlertCondition.ABOVE,
                threshold=0.05,  # 5% error rate
                severity=AlertSeverity.CRITICAL,
                duration_seconds=60,
            ),
            AlertConfig(
                id="entropy_reduction_stalled",
                name="Entropy Reduction Stalled",
                description="Entropy reduction has stalled",
                metric_name="entropy.reduction_rate",
                condition=AlertCondition.BELOW,
                threshold=0.01,  # Very low reduction rate
                severity=AlertSeverity.WARNING,
                duration_seconds=1800,  # 30 minutes of stalled progress
            ),
            AlertConfig(
                id="agent_error",
                name="Agent Error",
                description="An agent is in error state",
                metric_name="agent.status",
                condition=AlertCondition.BELOW,
                threshold=0,  # Error status is -1.0
                severity=AlertSeverity.WARNING,
                duration_seconds=60,
            ),
        ]
        
        for config in defaults:
            self._configs[config.id] = config
    
    def add_config(self, config: AlertConfig) -> None:
        """Add or update an alert configuration."""
        self._configs[config.id] = config
        
    def remove_config(self, config_id: str) -> bool:
        """Remove an alert configuration."""
        if config_id in self._configs:
            del self._configs[config_id]
            return True
        return False
        
    def get_configs(self) -> List[Dict[str, Any]]:
        """Get all alert configurations."""
        return [config.to_dict() for config in self._configs.values()]
        
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [alert.to_dict() for alert in self._active_alerts.values()]
        
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        return [alert.to_dict() for alert in self._alert_history[-limit:]]
    
    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a notification handler that will be called when an alert is triggered."""
        self._notification_handlers.append(handler)
        
    def evaluate_metric(
        self,
        name: str,
        value: float,
        timestamp: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[Alert]:
        """
        Evaluate a metric against all configurations and trigger alerts as needed.
        
        Returns any new alerts that were triggered.
        """
        timestamp = timestamp or time.time()
        tags = tags or {}
        new_alerts = []
        
        # Update the metric window for this metric
        if name not in self._metric_windows:
            self._metric_windows[name] = []
        
        self._metric_windows[name].append((timestamp, value))
        
        # Prune old values (keep last 24 hours)
        cutoff = timestamp - 86400
        self._metric_windows[name] = [(ts, val) for ts, val in self._metric_windows[name] if ts >= cutoff]
        
        # Evaluate each config that applies to this metric
        for config in self._configs.values():
            if not config.enabled:
                continue
                
            if config.metric_name != name:
                # Skip configs that don't apply to this metric
                continue
                
            # Check if tags match (if config has tags)
            if config.tags and not all(tags.get(k) == v for k, v in config.tags.items()):
                continue
                
            # Check if we're in cooldown for this alert
            last_triggered = self._last_triggered.get(config.id, 0)
            if timestamp - last_triggered < config.cooldown_seconds:
                continue
                
            # Check if the condition is met for the required duration
            if self._check_condition(config, name, value, timestamp):
                # Create a new alert
                alert = Alert(
                    config=config,
                    value=value,
                    status=AlertStatus.ACTIVE,
                    created_at=timestamp,
                    metadata={"tags": tags}
                )
                
                # Add to active alerts and history
                self._active_alerts[config.id] = alert
                self._alert_history.append(alert)
                
                # Trim history if needed
                if len(self._alert_history) > self._history_size:
                    self._alert_history = self._alert_history[-self._history_size:]
                
                # Update last triggered time
                self._last_triggered[config.id] = timestamp
                
                # Call notification handlers
                for handler in self._notification_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Error in notification handler: {e}")
                
                new_alerts.append(alert)
        
        return new_alerts
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id].acknowledge(user)
            return True
        return False
        
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.resolve()
            
            # Move to history
            del self._active_alerts[alert_id]
            return True
        return False
    
    def _check_condition(
        self, 
        config: AlertConfig, 
        metric_name: str, 
        current_value: float, 
        timestamp: float
    ) -> bool:
        """
        Check if the condition is met for the required duration.
        
        For duration_seconds > 0, the condition must be consistently true
        for the entire duration.
        """
        # Check the current value first
        if not self._evaluate_condition(config.condition, current_value, config.threshold):
            return False
            
        # If no duration requirement, we're done
        if config.duration_seconds <= 0:
            return True
            
        # Otherwise, check if condition has been true for the required duration
        if metric_name not in self._metric_windows:
            return False
            
        # Find all data points within the duration window
        start_time = timestamp - config.duration_seconds
        window = [(ts, val) for ts, val in self._metric_windows[metric_name] if ts >= start_time]
        
        # If we don't have enough data points, we can't confirm the duration
        if not window:
            return False
            
        # Check if condition was true for all data points in the window
        return all(self._evaluate_condition(config.condition, value, config.threshold) for _, value in window)
    
    def _evaluate_condition(self, condition: AlertCondition, value: float, threshold: float) -> bool:
        """Evaluate a simple condition."""
        if condition == AlertCondition.ABOVE:
            return value > threshold
        elif condition == AlertCondition.BELOW:
            return value < threshold
        elif condition == AlertCondition.EQUALS:
            return abs(value - threshold) < 1e-6  # Float comparison with small epsilon
        elif condition == AlertCondition.CHANGES:
            # This would require historical context, handled differently
            return False
        elif condition == AlertCondition.RATE_OF_CHANGE:
            # This would require historical context, handled differently
            return False
        else:
            return False


class WebhookNotifier:
    """Sends alert notifications to webhook endpoints."""
    
    def __init__(self, endpoints: List[str]) -> None:
        """Initialize with a list of webhook URLs."""
        self.endpoints = endpoints
        
    async def notify(self, alert: Alert) -> None:
        """Send a notification for an alert."""
        import aiohttp
        
        payload = {
            "alert": alert.to_dict(),
            "timestamp": time.time(),
        }
        
        async with aiohttp.ClientSession() as session:
            for endpoint in self.endpoints:
                try:
                    async with session.post(
                        endpoint, 
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status >= 400:
                            logger.error(f"Webhook notification failed: {response.status}")
                except Exception as e:
                    logger.error(f"Webhook notification error: {e}")


# Singleton instance that can be imported
default_manager = AlertManager()

# Helper functions that operate on the default manager
def add_config(config: AlertConfig) -> None:
    """Add an alert configuration to the default manager."""
    default_manager.add_config(config)
    
def evaluate_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> List[Alert]:
    """Evaluate a metric against all configurations in the default manager."""
    return default_manager.evaluate_metric(name, value, tags=tags)

def get_active_alerts() -> List[Dict[str, Any]]:
    """Get all active alerts from the default manager."""
    return default_manager.get_active_alerts()

def get_alert_history(limit: int = 100) -> List[Dict[str, Any]]:
    """Get alert history from the default manager."""
    return default_manager.get_alert_history(limit)
