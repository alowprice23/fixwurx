"""
monitoring/dashboard.py
────────────────────────────
Optimized dashboard for the FixWurx system with comprehensive metrics visualization,
agent status monitoring, entropy tracking, and system health indicators.

Highlights
──────────
• FastAPI + Chart.js UI with real-time SSE updates
• System health monitoring with key metrics
• Agent status tracking with detailed agent information
• Entropy visualization with trend analysis
• Rollback management interface
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Union, Tuple

from fastapi import FastAPI, Request, APIRouter, HTTPException, Body
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Import alerts system
from monitoring.alerts import (
    AlertConfig, 
    AlertSeverity, 
    AlertCondition, 
    default_manager as alert_manager
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dashboard")

# ---------------------------------------------------------------------------—
# Data Models
# ---------------------------------------------------------------------------—
class MetricType(str, Enum):
    """Types of metrics tracked in the system."""
    ENTROPY = "entropy"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    AGENT = "agent"
    PLANNER = "planner"
    ROLLBACK = "rollback"

class AgentStatus(str, Enum):
    """Status of agents in the system."""
    ACTIVE = "active"
    IDLE = "idle"
    ERROR = "error"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

class SystemHealth(str, Enum):
    """Overall system health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class MetricsBus:
    """
    Enhanced metrics bus that stores metrics, provides filtered access,
    and supports real-time broadcasting via SSE.
    """

    def __init__(self, history_size: int = 1000) -> None:
        """Initialize metrics bus with configurable history size."""
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self._listeners: List[asyncio.Queue] = []
        self._agent_status: Dict[str, Dict[str, Any]] = {}
        self._start_time = time.time()
        
        # Initialize system status
        self._system_status = {
            "health": SystemHealth.UNKNOWN,
            "active_agents": 0,
            "total_agents": 0,
            "current_entropy": None,
            "memory_usage_mb": 0.0,
            "cpu_usage_percent": 0.0,
            "uptime_seconds": 0,
            "warnings": [],
            "errors": []
        }
        
        # Initialize entropy metrics
        self._entropy_metrics = {
            "bits": 0.0,
            "reduction_rate": 0.0,
            "estimated_completion": None,
            "initial_entropy": 0.0,
            "progress_percent": 0.0,
            "last_updated": time.time()
        }
        
        # Start background system status updater
        self._update_system_status_task = asyncio.create_task(self._update_system_status())
    
    def send(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metric_type: Union[MetricType, str] = MetricType.SYSTEM,
    ) -> None:
        """
        Send a metric to all listeners.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
            metric_type: Type of metric
        """
        # Convert string type to enum if needed
        if isinstance(metric_type, str):
            metric_type = getattr(MetricType, metric_type.upper(), MetricType.SYSTEM)
            
        rec = {
            "ts": time.time(),
            "name": name,
            "value": value,
            "type": metric_type.value if isinstance(metric_type, Enum) else metric_type,
            "tags": tags or {},
        }
        self._history.append(rec)
        
        # Update derived metrics
        if name.endswith("entropy_bits"):
            self._update_entropy_metrics(value)
        
        # Update agent status if applicable
        if rec["type"] == MetricType.AGENT.value and tags and "agent_id" in tags:
            self._update_agent_status(tags["agent_id"], tags.get("agent_type", "unknown"), value, name)
            
        # Check if metric triggers any alerts
        from monitoring.alerts import evaluate_metric
        alerts = evaluate_metric(name, value, tags)
        if alerts:
            # If any alerts were triggered, also send an alert metric
            for alert in alerts:
                self.send(
                    name=f"alert.triggered.{alert.config.id}",
                    value=1.0,
                    tags={
                        "alert_id": alert.config.id,
                        "metric_name": name,
                        "metric_value": str(value),
                        "severity": alert.config.severity
                    },
                    metric_type="SYSTEM"
                )
        
        # Broadcast to listeners
        for q in self._listeners:
            q.put_nowait(rec)
    
    def update_agent_status(
        self,
        agent_id: str,
        agent_type: str,
        status: Union[AgentStatus, str],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update status of an agent."""
        # Convert string status to enum if needed
        if isinstance(status, str):
            status = getattr(AgentStatus, status.upper(), AgentStatus.UNKNOWN)
            
        status_value = status.value if isinstance(status, Enum) else status
        
        self._agent_status[agent_id] = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "status": status_value,
            "metrics": metrics or {},
            "last_updated": time.time()
        }
        
        # Also send as a metric
        status_numeric = {
            AgentStatus.ACTIVE.value: 1.0,
            AgentStatus.IDLE.value: 0.5,
            AgentStatus.ERROR.value: -1.0,
            AgentStatus.STOPPED.value: 0.0,
            AgentStatus.UNKNOWN.value: -0.5
        }.get(status_value, 0.0)
        
        self.send(
            name=f"agent.status.{agent_id}",
            value=status_numeric,
            tags={"agent_id": agent_id, "agent_type": agent_type},
            metric_type=MetricType.AGENT
        )
    
    def update_rollback_status(self, bug_id: str, status: str, trigger: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """Update rollback status."""
        status_value = {
            "success": 1.0,
            "pending": 0.5,
            "failed": 0.0,
            "cancelled": 0.25
        }.get(status.lower(), -1.0)
        
        self.send(
            name=f"rollback.status.{bug_id}",
            value=status_value,
            tags={"bug_id": bug_id, "status": status, "trigger": trigger or "unknown"},
            metric_type=MetricType.ROLLBACK
        )
    
    # Implementation details
    # ---------------------
    def _update_entropy_metrics(self, current_bits: float) -> None:
        """Update entropy metrics with new value."""
        now = time.time()
        
        # Initialize if this is the first entropy value
        if self._entropy_metrics["initial_entropy"] == 0.0:
            self._entropy_metrics["initial_entropy"] = current_bits
            self._entropy_metrics["bits"] = current_bits
            self._entropy_metrics["last_updated"] = now
            return
        
        # Calculate reduction rate (bits per hour)
        old_bits = self._entropy_metrics["bits"]
        last_updated = self._entropy_metrics.get("last_updated", now)
        time_delta = (now - last_updated) / 3600
        
        if time_delta > 0:
            reduction_rate = (old_bits - current_bits) / time_delta
            # Use exponential moving average for smoothing
            alpha = 0.3  # Smoothing factor
            self._entropy_metrics["reduction_rate"] = (alpha * reduction_rate + 
                                                     (1 - alpha) * self._entropy_metrics["reduction_rate"])
        
        # Update entropy and calculate progress
        self._entropy_metrics["bits"] = current_bits
        initial = self._entropy_metrics["initial_entropy"]
        if initial > 0:
            self._entropy_metrics["progress_percent"] = ((initial - current_bits) / initial) * 100
        
        # Estimate completion time
        if self._entropy_metrics["reduction_rate"] > 0:
            hours_left = current_bits / self._entropy_metrics["reduction_rate"]
            self._entropy_metrics["estimated_completion"] = now + (hours_left * 3600)
        else:
            self._entropy_metrics["estimated_completion"] = None
        
        # Store last update time
        self._entropy_metrics["last_updated"] = now
    
    def _update_agent_status(self, agent_id: str, agent_type: str, value: float, metric_name: str) -> None:
        """Update agent status based on metrics."""
        if agent_id not in self._agent_status:
            self._agent_status[agent_id] = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "status": AgentStatus.UNKNOWN.value,
                "metrics": {},
                "last_updated": time.time()
            }
        
        # Update the metric
        self._agent_status[agent_id]["metrics"][metric_name] = value
        self._agent_status[agent_id]["last_updated"] = time.time()
    
    async def _update_system_status(self) -> None:
        """Background task to update system status."""
        while True:
            try:
                # Update uptime
                self._system_status["uptime_seconds"] = int(time.time() - self._start_time)
                
                # Count active agents
                now = time.time()
                active_count = 0
                total_count = len(self._agent_status)
                
                for agent_data in self._agent_status.values():
                    if agent_data["status"] == AgentStatus.ACTIVE.value:
                        # Check if recently updated (within 60 seconds)
                        if now - agent_data["last_updated"] < 60:
                            active_count += 1
                
                self._system_status["active_agents"] = active_count
                self._system_status["total_agents"] = total_count
                
                # Update entropy
                self._system_status["current_entropy"] = (
                    self._entropy_metrics["bits"] if self._entropy_metrics["bits"] > 0 else None
                )
                
                # Determine health status
                if len(self._system_status["errors"]) > 0:
                    self._system_status["health"] = SystemHealth.CRITICAL.value
                elif len(self._system_status["warnings"]) > 0:
                    self._system_status["health"] = SystemHealth.WARNING.value
                elif active_count > 0:
                    self._system_status["health"] = SystemHealth.HEALTHY.value
                else:
                    self._system_status["health"] = SystemHealth.UNKNOWN.value
                
                # Update system metrics as a metric too
                health_value = {
                    SystemHealth.HEALTHY.value: 1.0,
                    SystemHealth.WARNING.value: 0.5,
                    SystemHealth.CRITICAL.value: 0.0,
                    SystemHealth.UNKNOWN.value: -1.0
                }.get(self._system_status["health"], -1.0)
                
                self.send(
                    name="system.health",
                    value=health_value,
                    tags={"active_agents": str(active_count), "total_agents": str(total_count)},
                    metric_type=MetricType.SYSTEM
                )
                
                # Sleep for a bit
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error updating system status: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    # API methods
    # ----------
    def get_filtered_history(
        self,
        metric_type: Optional[Union[MetricType, str]] = None,
        metric_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Get filtered history based on criteria."""
        # Convert string type to enum if needed
        if isinstance(metric_type, str):
            metric_type = getattr(MetricType, metric_type.upper(), None)
            
        metric_type_value = metric_type.value if isinstance(metric_type, Enum) else metric_type
        
        filtered = []
        for rec in self._history:
            # Apply filters
            if metric_type_value and rec.get("type") != metric_type_value:
                continue
                
            if metric_name and rec.get("name") != metric_name:
                continue
                
            if tags:
                rec_tags = rec.get("tags", {})
                if not all(rec_tags.get(k) == v for k, v in tags.items()):
                    continue
            
            if time_range:
                start, end = time_range
                if rec.get("ts", 0) < start or rec.get("ts", 0) > end:
                    continue
            
            filtered.append(rec)
        
        return filtered
    
    def get_agent_status(self, agent_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get status of one or all agents."""
        if agent_id:
            return self._agent_status.get(agent_id, {})
        return list(self._agent_status.values())
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return self._system_status
    
    def get_entropy_metrics(self) -> Dict[str, Any]:
        """Get entropy metrics."""
        return self._entropy_metrics
    
    # SSE helpers
    # -----------
    def new_queue(self) -> asyncio.Queue:
        """Create a new queue for a listener."""
        q: asyncio.Queue = asyncio.Queue(maxsize=0)
        self._listeners.append(q)
        return q
    
    def history_json(self, limit: int = 100) -> str:
        """Get JSON representation of recent history."""
        recent = list(self._history)[-limit:] if self._history else []
        return json.dumps(recent)


# ---------------------------------------------------------------------------—
# FastAPI app setup
# ---------------------------------------------------------------------------—
app = FastAPI(title="FixWurx Dashboard")

# Set up templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir.resolve()))

# Set up static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Create API router
api_router = APIRouter(prefix="/api")


@app.on_event("startup")
async def _init_state() -> None:
    """Initialize application state."""
    if not hasattr(app.state, "metric_bus"):
        app.state.metric_bus = MetricsBus()
        logger.info("Created metrics bus for standalone demo mode")


# ---------------------------------------------------------------------------—
# Main page routes
# ---------------------------------------------------------------------------—
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main dashboard page with metrics visualization."""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "title": "FixWurx Dashboard"}
    )


@app.get("/agents", response_class=HTMLResponse)
async def agents_page(request: Request):
    """Agent status monitoring page."""
    return templates.TemplateResponse(
        "agents.html",
        {"request": request, "title": "Agent Status"}
    )


@app.get("/entropy", response_class=HTMLResponse)
async def entropy_page(request: Request):
    """Entropy visualization page."""
    return templates.TemplateResponse(
        "entropy.html",
        {"request": request, "title": "Entropy Tracking"}
    )


@app.get("/rollbacks", response_class=HTMLResponse)
async def rollbacks_page(request: Request):
    """Rollback management page."""
    return templates.TemplateResponse(
        "rollbacks.html",
        {"request": request, "title": "Rollback Management"}
    )


@app.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request):
    """Alert management page."""
    return templates.TemplateResponse(
        "alerts.html",
        {"request": request, "title": "Alert Management"}
    )


@app.get("/error-logs", response_class=HTMLResponse)
async def error_logs_page(request: Request):
    """Error logs page."""
    return templates.TemplateResponse(
        "error_logs.html",
        {"request": request, "title": "Error Logs"}
    )


@app.get("/error-visualization", response_class=HTMLResponse)
async def error_visualization_page(request: Request):
    """Error visualization and analysis dashboard."""
    return templates.TemplateResponse(
        "error_visualization.html",
        {"request": request, "title": "Error Log Analysis"}
    )


# ---------------------------------------------------------------------------—
# SSE endpoint
# ---------------------------------------------------------------------------—
@app.get("/events")
async def sse_events(request: Request):
    """Server-Sent Events endpoint streaming metric JSON frames."""
    bus: MetricsBus = app.state.metric_bus
    queue = bus.new_queue()

    async def event_stream():
        # Send history first
        yield f"data: {bus.history_json()}\n\n"
        
        # Then stream events as they arrive
        while True:
            if await request.is_disconnected():
                break
                
            item = await queue.get()
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------—
# API endpoints
# ---------------------------------------------------------------------------—
@api_router.get("/metrics")
async def get_metrics(
    metric_type: Optional[str] = None,
    metric_name: Optional[str] = None,
    hours: Optional[int] = None,
):
    """Get metrics with optional filtering."""
    bus: MetricsBus = app.state.metric_bus
    
    # Set up time range if hours specified
    time_range = None
    if hours:
        end = time.time()
        start = end - (hours * 3600)
        time_range = (start, end)
    
    metrics = bus.get_filtered_history(
        metric_type=metric_type,
        metric_name=metric_name,
        time_range=time_range
    )
    
    return metrics


@api_router.get("/agents")
async def get_agents():
    """Get status of all agents."""
    bus: MetricsBus = app.state.metric_bus
    return bus.get_agent_status()


@api_router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get status of a specific agent."""
    bus: MetricsBus = app.state.metric_bus
    agent = bus.get_agent_status(agent_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
    return agent


@api_router.get("/system")
async def get_system():
    """Get overall system status."""
    bus: MetricsBus = app.state.metric_bus
    return bus.get_system_status()


@api_router.get("/entropy")
async def get_entropy():
    """Get entropy metrics."""
    bus: MetricsBus = app.state.metric_bus
    return bus.get_entropy_metrics()


@api_router.get("/entropy-narrative", response_class=PlainTextResponse)
async def entropy_narrative():
    """Get a narrative description of entropy status."""
    bus: MetricsBus = app.state.metric_bus
    metrics = bus.get_entropy_metrics()
    
    if metrics["bits"] <= 0:
        return "No entropy data available yet."
    
    bits = metrics["bits"]
    progress = metrics["progress_percent"]
    reduction_rate = metrics["reduction_rate"]
    
    # Format estimated completion time if available
    completion_str = "unknown"
    if metrics["estimated_completion"]:
        completion_time = datetime.fromtimestamp(metrics["estimated_completion"])
        now = datetime.now()
        if completion_time > now:
            time_left = completion_time - now
            hours = time_left.total_seconds() / 3600
            if hours < 1:
                minutes = int(time_left.total_seconds() / 60)
                completion_str = f"~{minutes} minutes"
            elif hours < 24:
                completion_str = f"~{int(hours)} hours"
            else:
                days = int(hours / 24)
                completion_str = f"~{days} days"
        else:
            completion_str = "imminent"
    
    # Determine confidence level based on progress
    confidence = "low"
    if progress >= 75:
        confidence = "very high"
    elif progress >= 50:
        confidence = "high"
    elif progress >= 25:
        confidence = "medium"
    
    return (
        f"### Entropy Analysis\n\n"
        f"Current entropy: **{bits:.2f} bits** ({progress:.1f}% complete)\n\n"
        f"Reduction rate: {reduction_rate:.2f} bits/hour\n"
        f"Estimated completion: {completion_str}\n\n"
        f"Candidate space: ~{2**bits:.0f} possibilities\n"
        f"Confidence level: **{confidence}**\n\n"
        f"{'System converging rapidly' if reduction_rate > 1.0 else 'System converging steadily'}"
    )


# ---------------------------------------------------------------------------—
# Alert API endpoints
# ---------------------------------------------------------------------------—
@api_router.get("/alerts/active")
async def get_active_alerts():
    """Get all active alerts."""
    return alert_manager.get_active_alerts()


@api_router.get("/alerts/configs")
async def get_alert_configs():
    """Get all alert configurations."""
    return alert_manager.get_configs()


@api_router.get("/alerts/history")
async def get_alert_history(limit: int = 100):
    """Get alert history."""
    return alert_manager.get_alert_history(limit)


@api_router.post("/alerts/configs")
async def create_alert_config(config: Dict[str, Any] = Body(...)):
    """Create a new alert configuration."""
    # Generate a unique ID if not provided
    if "id" not in config:
        config["id"] = f"alert-{hash(str(time.time()))}"
    
    alert_config = AlertConfig.from_dict(config)
    alert_manager.add_config(alert_config)
    return {"status": "success", "id": alert_config.id}


@api_router.put("/alerts/configs")
async def update_alert_config(config: Dict[str, Any] = Body(...)):
    """Update an existing alert configuration."""
    if "id" not in config:
        raise HTTPException(status_code=400, detail="Alert ID is required for updates")
    
    alert_config = AlertConfig.from_dict(config)
    alert_manager.add_config(alert_config)
    return {"status": "success"}


@api_router.delete("/alerts/configs/{config_id}")
async def delete_alert_config(config_id: str):
    """Delete an alert configuration."""
    success = alert_manager.remove_config(config_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert configuration {config_id} not found")
    return {"status": "success"}


@api_router.post("/alerts/configs/{config_id}/toggle")
async def toggle_alert_config(config_id: str, data: Dict[str, Any] = Body(...)):
    """Enable or disable an alert configuration."""
    configs = alert_manager.get_configs()
    config_dict = next((c for c in configs if c["id"] == config_id), None)
    
    if not config_dict:
        raise HTTPException(status_code=404, detail=f"Alert configuration {config_id} not found")
    
    # Update enabled status
    config_dict["enabled"] = data.get("enabled", True)
    alert_config = AlertConfig.from_dict(config_dict)
    alert_manager.add_config(alert_config)
    
    return {"status": "success"}


@api_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, data: Dict[str, Any] = Body(...)):
    """Acknowledge an alert."""
    user = data.get("user", "system")
    success = alert_manager.acknowledge_alert(alert_id, user)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    
    return {"status": "success"}


@api_router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert."""
    success = alert_manager.resolve_alert(alert_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    
    return {"status": "success"}


# ---------------------------------------------------------------------------—
# Error Logs API endpoints
# ---------------------------------------------------------------------------—
@api_router.get("/error-logs")
async def get_error_logs(
    min_severity: Optional[str] = "WARNING",
    component: Optional[str] = None,
    hours: Optional[int] = 24,
    limit: Optional[int] = 1000
):
    """Get error logs with filtering."""
    try:
        # Get system monitor from app state
        bus: MetricsBus = app.state.metric_bus
        
        # Check if system_monitor attribute exists
        if not hasattr(bus, "system_monitor"):
            # For standalone mode, we need to create a connection to the system monitor
            from system_monitor import SystemMonitor
            import importlib
            
            # Try to import monitoring.error_log
            try:
                error_log_module = importlib.import_module("monitoring.error_log")
                # Check if there's a global instance
                if hasattr(error_log_module, "global_error_log"):
                    # Return logs from the global instance
                    logs = error_log_module.global_error_log.get_entries_for_dashboard(
                        hours=hours,
                        min_severity=min_severity,
                        limit=limit
                    )
                    
                    # Apply component filter if needed
                    if component:
                        logs = [log for log in logs if log.get("component") == component]
                        
                    return logs
            except ImportError:
                pass
            
            # If we can't find logs, return empty list
            logger.warning("Error log system not found, returning empty log list")
            return []
        else:
            # Use the system monitor attached to the bus
            logs = bus.system_monitor.get_errors_for_dashboard(
                hours=hours,
                min_severity=min_severity
            )
            
            # Apply component filter if needed
            if component:
                logs = [log for log in logs if log.get("component") == component]
                
            # Apply limit
            if limit and len(logs) > limit:
                logs = logs[:limit]
                
            return logs
    except Exception as e:
        logger.error(f"Error retrieving error logs: {e}")
        return []


@api_router.get("/error-stats")
async def get_error_stats():
    """Get statistics about error logs."""
    try:
        # Get system monitor from app state
        bus: MetricsBus = app.state.metric_bus
        
        # Check if system_monitor attribute exists
        if not hasattr(bus, "system_monitor"):
            # For standalone mode, we need to create a connection to the system monitor
            import importlib
            
            # Try to import monitoring.error_log
            try:
                error_log_module = importlib.import_module("monitoring.error_log")
                # Check if there's a global instance
                if hasattr(error_log_module, "global_error_log"):
                    # Return stats from the global instance
                    return error_log_module.global_error_log.get_stats()
            except ImportError:
                pass
            
            # If we can't find stats, return empty stats
            logger.warning("Error log system not found, returning empty stats")
            return {
                "total_entries": 0,
                "by_severity": {
                    "DEBUG": 0,
                    "INFO": 0,
                    "WARNING": 0,
                    "ERROR": 0,
                    "CRITICAL": 0
                },
                "by_component": {}
            }
        else:
            # Use the system monitor attached to the bus
            return bus.system_monitor.get_error_stats()
    except Exception as e:
        logger.error(f"Error retrieving error stats: {e}")
        return {
            "total_entries": 0,
            "by_severity": {},
            "by_component": {},
            "error": str(e)
        }


# Register API router
app.include_router(api_router)


# ---------------------------------------------------------------------------—
# CSS and Template setup (on import)
# ---------------------------------------------------------------------------—
# Create CSS file
static_dir.mkdir(parents=True, exist_ok=True)
(static_dir / "styles.css").write_text(
    """
/* Base styles */
:root {
    --primary: #3498db;
    --secondary: #2ecc71;
    --warning: #f39c12;
    --danger: #e74c3c;
    --light: #ecf0f1;
    --dark: #2c3e50;
    --bg-color: #f8f9fa;
    --text-color: #333;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    background-color: var(--bg-color);
    color: var(--text-color);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Header and navigation */
header {
    background-color: var(--dark);
    color: white;
    padding: 1rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.navbar-brand {
    font-size: 1.5rem;
    font-weight: bold;
    color: white;
    text-decoration: none;
}

.navbar-nav {
    display: flex;
    list-style: none;
    margin: 0;
    padding: 0;
}

.nav-item {
    margin-left: 20px;
}

.nav-link {
    color: rgba(255,255,255,0.8);
    text-decoration: none;
    padding: 0.5rem 0;
    transition: color 0.3s;
}

.nav-link:hover {
    color: white;
}

.nav-link.active {
    color: white;
    border-bottom: 2px solid var(--primary);
}

/* Dashboard layout */
.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    grid-gap: 20px;
    margin-top: 20px;
}

.card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    padding: 1.5rem;
    transition: transform 0.3s, box-shadow 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #eee;
}

.card-title {
    font-size: 1.25rem;
    margin: 0;
    color: var(--dark);
}

.card-body {
    min-height: 200px;
}

/* Charts */
.chart-container {
    position: relative;
    height: 250px;
    width: 100%;
}

/* Status indicators */
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 5px;
}

.status-healthy {
    background-color: var(--secondary);
}

.status-warning {
    background-color: var(--warning);
}

.status-critical {
    background-color: var(--danger);
}

.status-unknown {
    background-color: #95a5a6;
}

/* Tables */
.table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1rem;
}

.table th,
.table td {
    padding: 0.75rem;
    border-bottom: 1px solid #eee;
    text-align: left;
}

.table th {
    background-color: #f5f5f5;
    font-weight: 600;
}

.table tr:hover {
    background-color: #f9f9f9;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-primary { background-color: var(--primary); color: white; }
.badge-success { background-color: var(--secondary); color: white; }
.badge-warning { background-color: var(--warning); color: white; }
.badge-danger { background-color: var(--danger); color: white; }

/* Responsive adjustments */
@media (max-width: 768px) {
    .dashboard-grid {
        grid-template-columns: 1fr;
    }
    
    .navbar {
        flex-direction: column;
    }
    
    .navbar-nav {
        margin-top: 1rem;
    }
    
    .nav-item {
        margin-left: 0;
        margin-right: 20px;
    }
}

/* Entropy page specifics */
.entropy-dashboard {
    display: grid;
    grid-template-columns: 2fr 1fr;
    grid-gap: 20px;
}

/* Form elements */
.form-group {
    margin-bottom: 1rem;
}

.form-control {
    display: block;
    width: 100%;
    padding: 0.5rem;
    font-size: 1rem;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.form-actions {
    margin-top: 1.5rem;
}

.btn {
    padding: 0.5rem 1rem;
    background-color: var(--primary);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.btn:hover {
    background-color: #2980b9;
}

/* Rollback page specifics */
.toggle-switch {
    display: flex;
    align-items: center;
}

.toggle-switch input {
    height: 0;
    width: 0;
    visibility: hidden;
    position: absolute;
}

.toggle-switch label {
    cursor: pointer;
    width: 50px;
    height: 25px;
    background: #ccc;
    display: block;
    border-radius: 25px;
    position: relative;
}

.toggle-switch label:after {
    content: '';
    position: absolute;
    top: 3px;
    left: 3px;
    width: 19px;
    height: 19px;
    background: #fff;
    border-radius: 19px;
    transition: 0.3s;
}

.toggle-switch input:checked + label {
    background: var(--secondary);
}

.toggle-switch input:checked + label:after {
    left: calc(100% - 3px);
    transform: translateX(-100%);
}
""",
    encoding="utf-8"
)


# ---------------------------------------------------------------------------—
# Main function for direct execution
# ---------------------------------------------------------------------------—
def main():
    """Run the dashboard server directly."""
    import uvicorn
    
    print("Starting FixWurx Dashboard")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
