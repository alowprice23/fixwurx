#!/usr/bin/env python3
"""
service_health_monitor.py
──────────────────────────
Service health monitoring and auto-recovery system to ensure 99.9% uptime.

This component:
1. Monitors all critical FixWurx services
2. Detects failures or degraded performance
3. Automatically attempts recovery actions
4. Records uptime metrics and generates alerts
5. Provides a health status API endpoint
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import subprocess
import threading
import requests
import socket
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/service_health.log")
    ]
)
logger = logging.getLogger("service_health_monitor")

# Service definitions
SERVICES = {
    "api": {
        "name": "FixWurx API",
        "endpoint": "http://localhost:8000/health",
        "container": "fixwurx_main",
        "critical": True,
        "recovery_attempts": 3,
        "recovery_commands": [
            "docker restart fixwurx_main",
            "kubectl rollout restart deployment/fixwurx -n fixwurx"
        ],
        "timeout": 5,  # seconds
        "expect": {"status": "healthy"}
    },
    "dashboard": {
        "name": "FixWurx Dashboard",
        "endpoint": "http://localhost:8001/health",
        "container": "fixwurx_main",
        "critical": True,
        "recovery_attempts": 3,
        "recovery_commands": [
            "docker restart fixwurx_main",
            "kubectl rollout restart deployment/fixwurx -n fixwurx"
        ],
        "timeout": 5,
        "expect": {"status": "healthy"}
    },
    "db": {
        "name": "Database",
        "endpoint": "postgresql://fixwurx:password@localhost:5432/fixwurx",
        "container": "fixwurx_db",
        "critical": True,
        "recovery_attempts": 2,
        "recovery_commands": [
            "docker restart fixwurx_db",
            "kubectl rollout restart statefulset/postgres -n fixwurx"
        ],
        "timeout": 5,
        "expect": None,  # Special check for database connectivity
        "check_type": "postgres"
    },
    "prometheus": {
        "name": "Prometheus",
        "endpoint": "http://localhost:9090/-/healthy",
        "container": "fixwurx_prometheus",
        "critical": False,
        "recovery_attempts": 2,
        "recovery_commands": [
            "docker restart fixwurx_prometheus",
            "kubectl rollout restart deployment/prometheus -n monitoring"
        ],
        "timeout": 5,
        "expect": None  # Any 200 response is considered healthy
    },
    "grafana": {
        "name": "Grafana",
        "endpoint": "http://localhost:3000/api/health",
        "container": "fixwurx_grafana",
        "critical": False,
        "recovery_attempts": 2,
        "recovery_commands": [
            "docker restart fixwurx_grafana",
            "kubectl rollout restart deployment/grafana -n monitoring"
        ],
        "timeout": 5,
        "expect": {"database": "ok"}
    },
    "loadbalancer": {
        "name": "Load Balancer",
        "endpoint": "http://localhost:80/health",
        "container": "fixwurx_loadbalancer",
        "critical": True,
        "recovery_attempts": 3,
        "recovery_commands": [
            "docker restart fixwurx_loadbalancer",
            "kubectl rollout restart deployment/nginx-ingress -n ingress"
        ],
        "timeout": 5,
        "expect": None  # Any 200 response is considered healthy
    }
}

# Uptime tracking
uptime_data = {
    "start_time": time.time(),
    "services": {},
    "incidents": [],
    "current_month": {
        "total_seconds": 0,
        "downtime_seconds": 0,
        "uptime_percentage": 100.0
    }
}

for service_id in SERVICES:
    uptime_data["services"][service_id] = {
        "status": "unknown",
        "last_check": None,
        "last_success": None,
        "failures": 0,
        "recovery_attempts": 0,
        "current_month": {
            "total_seconds": 0,
            "downtime_seconds": 0,
            "uptime_percentage": 100.0
        }
    }

def check_http_service(service_id: str, service_config: Dict[str, Any]) -> Tuple[bool, str]:
    """Check the health of an HTTP service endpoint."""
    try:
        response = requests.get(service_config["endpoint"], timeout=service_config["timeout"])
        if response.status_code != 200:
            return False, f"HTTP status {response.status_code}"
        
        # If we expect specific response content
        if service_config["expect"]:
            try:
                data = response.json()
                for key, value in service_config["expect"].items():
                    if key not in data or data[key] != value:
                        return False, f"Response mismatch: expected {key}={value}"
                return True, "Healthy"
            except json.JSONDecodeError:
                return False, "Invalid JSON response"
            except Exception as e:
                return False, f"Response validation error: {str(e)}"
        
        # If we just needed a 200 response
        return True, "Healthy"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_postgres_service(service_id: str, service_config: Dict[str, Any]) -> Tuple[bool, str]:
    """Check the health of a PostgreSQL database."""
    try:
        # Extract connection details from the endpoint URL
        # This is a simplified example - in production, use a proper connection string parser
        if "postgresql://" in service_config["endpoint"]:
            conn_parts = service_config["endpoint"].replace("postgresql://", "").split("@")
            if len(conn_parts) == 2:
                auth, host_port = conn_parts
                user_pass = auth.split(":")
                host_db = host_port.split("/")
                
                if len(user_pass) == 2 and len(host_db) == 2:
                    host_port = host_db[0].split(":")
                    if len(host_port) == 2:
                        host, port = host_port
                        # Test connection using socket
                        with socket.create_connection((host, int(port)), timeout=service_config["timeout"]) as sock:
                            return True, "Database connection successful"
        
        return False, "Failed to parse database connection string"
    except socket.timeout:
        return False, "Database connection timeout"
    except ConnectionRefusedError:
        return False, "Database connection refused"
    except Exception as e:
        return False, f"Database error: {str(e)}"

def check_service(service_id: str) -> Tuple[bool, str]:
    """Check the health of a service based on its configuration."""
    service_config = SERVICES[service_id]
    
    # Determine which check function to use
    check_type = service_config.get("check_type", "http")
    if check_type == "postgres":
        return check_postgres_service(service_id, service_config)
    else:
        return check_http_service(service_id, service_config)

def execute_recovery(service_id: str, attempt: int) -> bool:
    """Execute recovery action for a failed service."""
    service_config = SERVICES[service_id]
    
    if attempt >= len(service_config["recovery_commands"]):
        # Use the last command if we've run out of options
        command = service_config["recovery_commands"][-1]
    else:
        command = service_config["recovery_commands"][attempt]
    
    logger.info(f"Attempting recovery for {service_config['name']}: {command}")
    
    try:
        # Execute the command
        process = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        if process.returncode == 0:
            logger.info(f"Recovery command succeeded: {process.stdout.strip()}")
            return True
        else:
            logger.error(f"Recovery command failed: {process.stderr.strip()}")
            return False
    except Exception as e:
        logger.error(f"Error executing recovery command: {str(e)}")
        return False

def update_uptime_metrics(service_id: str, is_healthy: bool) -> None:
    """Update uptime metrics for a service."""
    service_data = uptime_data["services"][service_id]
    now = time.time()
    
    # Initialize if this is the first check
    if service_data["last_check"] is None:
        service_data["last_check"] = now
        service_data["status"] = "healthy" if is_healthy else "unhealthy"
        if is_healthy:
            service_data["last_success"] = now
        return
    
    # Calculate elapsed time since last check
    elapsed = now - service_data["last_check"]
    service_data["last_check"] = now
    
    # Update monthly metrics
    service_data["current_month"]["total_seconds"] += elapsed
    
    # Update overall system metrics
    uptime_data["current_month"]["total_seconds"] += elapsed
    
    # If service was previously healthy but is now unhealthy, record the incident
    if service_data["status"] == "healthy" and not is_healthy:
        incident = {
            "service_id": service_id,
            "service_name": SERVICES[service_id]["name"],
            "start_time": now,
            "end_time": None,
            "resolved": False,
            "recovery_attempts": 0
        }
        uptime_data["incidents"].append(incident)
        
    # If service was previously unhealthy but is now healthy, resolve the incident
    elif service_data["status"] == "unhealthy" and is_healthy:
        # Find the unresolved incident
        for incident in reversed(uptime_data["incidents"]):
            if incident["service_id"] == service_id and not incident["resolved"]:
                incident["end_time"] = now
                incident["resolved"] = True
                incident_duration = now - incident["start_time"]
                service_data["current_month"]["downtime_seconds"] += incident_duration
                uptime_data["current_month"]["downtime_seconds"] += incident_duration
                break
    
    # If service remains unhealthy, add to downtime
    elif service_data["status"] == "unhealthy" and not is_healthy:
        service_data["current_month"]["downtime_seconds"] += elapsed
        uptime_data["current_month"]["downtime_seconds"] += elapsed
        # Update the last incident's recovery attempts
        for incident in reversed(uptime_data["incidents"]):
            if incident["service_id"] == service_id and not incident["resolved"]:
                incident["recovery_attempts"] = service_data["recovery_attempts"]
                break
    
    # Update current status
    service_data["status"] = "healthy" if is_healthy else "unhealthy"
    if is_healthy:
        service_data["last_success"] = now
    
    # Calculate uptime percentages
    if service_data["current_month"]["total_seconds"] > 0:
        uptime_seconds = service_data["current_month"]["total_seconds"] - service_data["current_month"]["downtime_seconds"]
        service_data["current_month"]["uptime_percentage"] = (uptime_seconds / service_data["current_month"]["total_seconds"]) * 100.0
    
    if uptime_data["current_month"]["total_seconds"] > 0:
        overall_uptime_seconds = uptime_data["current_month"]["total_seconds"] - uptime_data["current_month"]["downtime_seconds"]
        uptime_data["current_month"]["uptime_percentage"] = (overall_uptime_seconds / uptime_data["current_month"]["total_seconds"]) * 100.0

def monitor_service(service_id: str) -> None:
    """Continuously monitor a service and execute recovery if needed."""
    service_config = SERVICES[service_id]
    service_data = uptime_data["services"][service_id]
    
    while True:
        is_healthy, message = check_service(service_id)
        
        if is_healthy:
            logger.info(f"{service_config['name']} is healthy: {message}")
            service_data["failures"] = 0
            service_data["recovery_attempts"] = 0
        else:
            logger.warning(f"{service_config['name']} is unhealthy: {message}")
            service_data["failures"] += 1
            
            # Only attempt recovery after consecutive failures
            if service_data["failures"] >= 3:
                # Attempt recovery if we haven't exceeded the maximum attempts
                if service_data["recovery_attempts"] < service_config["recovery_attempts"]:
                    service_data["recovery_attempts"] += 1
                    success = execute_recovery(service_id, service_data["recovery_attempts"] - 1)
                    if success:
                        logger.info(f"Recovery attempt {service_data['recovery_attempts']} for {service_config['name']} succeeded")
                    else:
                        logger.error(f"Recovery attempt {service_data['recovery_attempts']} for {service_config['name']} failed")
                else:
                    logger.error(f"Maximum recovery attempts ({service_config['recovery_attempts']}) reached for {service_config['name']}")
        
        # Update metrics
        update_uptime_metrics(service_id, is_healthy)
        
        # Calculate sleep time based on health status
        sleep_time = 60 if is_healthy else 15  # Check more frequently when unhealthy
        time.sleep(sleep_time)

def save_uptime_data() -> None:
    """Save uptime data to a JSON file periodically."""
    while True:
        with open("data/uptime_metrics.json", "w") as f:
            json.dump(uptime_data, f, indent=2)
        time.sleep(300)  # Save every 5 minutes

def start_monitoring() -> None:
    """Start monitoring all services in separate threads."""
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    logger.info("Starting service health monitoring")
    
    # Start a thread for each service
    threads = []
    for service_id in SERVICES:
        thread = threading.Thread(
            target=monitor_service,
            args=(service_id,),
            name=f"monitor-{service_id}",
            daemon=True
        )
        thread.start()
        threads.append(thread)
        logger.info(f"Started monitoring thread for {SERVICES[service_id]['name']}")
    
    # Start metrics saving thread
    metrics_thread = threading.Thread(
        target=save_uptime_data,
        name="save-metrics",
        daemon=True
    )
    metrics_thread.start()
    threads.append(metrics_thread)
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down service health monitor")
        sys.exit(0)

def get_service_status() -> Dict[str, Any]:
    """Get the current status of all services."""
    result = {
        "timestamp": time.time(),
        "uptime_percentage": uptime_data["current_month"]["uptime_percentage"],
        "services": {}
    }
    
    for service_id, service_data in uptime_data["services"].items():
        result["services"][service_id] = {
            "name": SERVICES[service_id]["name"],
            "status": service_data["status"],
            "uptime_percentage": service_data["current_month"]["uptime_percentage"],
            "last_check": service_data["last_check"],
            "critical": SERVICES[service_id]["critical"]
        }
    
    return result

def start_api_server() -> None:
    """Start a simple API server for health status reporting."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health" or self.path == "/health/":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                
                status = get_service_status()
                self.wfile.write(json.dumps(status).encode())
            else:
                self.send_response(404)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Not found")
    
    server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
    logger.info("Starting health API server on port 8080")
    server.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FixWurx Service Health Monitor")
    parser.add_argument("--api-only", action="store_true", help="Run only the API server")
    parser.add_argument("--monitor-only", action="store_true", help="Run only the monitoring threads")
    args = parser.parse_args()
    
    if args.api_only:
        start_api_server()
    elif args.monitor_only:
        start_monitoring()
    else:
        # Start both monitoring and API server
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        start_monitoring()
