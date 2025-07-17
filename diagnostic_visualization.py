#!/usr/bin/env python3
"""
diagnostic_visualization.py
───────────────────────────
Visualization tools for the FixWurx shell diagnostics that leverage
LLM agents to create visualizations dynamically.
"""

import os
import tempfile
import webbrowser
from enum import Enum
import logging
from typing import Dict, Any, Optional, Union

# Internal imports
from shell_environment import register_command
from shell_diagnostics import get_health, get_metrics
from meta_agent import request_agent_task
from agent_shell_integration import get_agent_response

# Configure logging
logger = logging.getLogger("DiagnosticVisualization")
handler = logging.FileHandler("diagnostic_visualization.log")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class VisualizationType(Enum):
    """Types of visualizations supported."""
    SYSTEM_METRICS = "system_metrics"
    COMPONENT_HEALTH = "component_health"
    RESOURCE_USAGE = "resource_usage"
    ERROR_TRENDS = "error_trends"
    DASHBOARD = "dashboard"

class VisualizationFormat(Enum):
    """Output formats for visualizations."""
    HTML = "html"
    PNG = "png"
    JSON = "json"

def generate_visualization(
    visualization_type: Union[str, VisualizationType],
    output_format: Union[str, VisualizationFormat] = "html",
    data_points: int = 20,
    output_path: Optional[str] = None,
    open_browser: bool = True
) -> Dict[str, Any]:
    """Generate visualization using the agent system."""
    
    # Convert string to enum if needed
    if isinstance(visualization_type, str):
        try:
            visualization_type = VisualizationType(visualization_type)
        except ValueError:
            return {"success": False, "error": f"Unknown visualization type: {visualization_type}"}
    
    # Get visualization data
    try:
        if visualization_type == VisualizationType.COMPONENT_HEALTH:
            data = get_health()
        else:
            data = get_metrics(data_points)
    except Exception as e:
        return {"success": False, "error": f"Error retrieving data: {e}"}
    
    # Create temp file if no output path provided
    if not output_path:
        ext = ".html" if output_format == "html" else f".{output_format}"
        fd, output_path = tempfile.mkstemp(suffix=ext, prefix=f"{visualization_type.value}_")
        os.close(fd)
    
    # Request visualization from agent
    prompt = f"""
    Create a {output_format} visualization for {visualization_type.value} data.
    Save the visualization to {output_path}.
    Data: {data}
    """
    
    try:
        result = request_agent_task("visualization", prompt, timeout=30)
        
        if result.get("success", False):
            # Open browser for HTML visualizations if requested
            if output_format == "html" and open_browser:
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
                
            return {
                "success": True,
                "output_path": output_path,
                "format": output_format,
                "message": result.get("message", "Visualization generated successfully")
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error generating visualization")
            }
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        return {"success": False, "error": str(e)}

def visualize_command(args: str) -> int:
    """Handle the visualize command."""
    arg_parts = args.strip().split()
    
    if not arg_parts:
        print("Usage: visualize <type> [format] [datapoints] [output_path]")
        print("Available types:", ", ".join([t.value for t in VisualizationType]))
        print("Available formats:", ", ".join([f.value for f in VisualizationFormat]))
        return 1
    
    # Parse arguments
    vis_type = arg_parts[0]
    output_format = arg_parts[1] if len(arg_parts) > 1 else "html"
    data_points = int(arg_parts[2]) if len(arg_parts) > 2 and arg_parts[2].isdigit() else 20
    output_path = arg_parts[3] if len(arg_parts) > 3 else None
    
    print(f"Generating {vis_type} visualization...")
    result = generate_visualization(vis_type, output_format, data_points, output_path)
    
    if result.get("success", False):
        print(f"Visualization saved to: {result.get('output_path')}")
        return 0
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1

def dashboard_command(args: str) -> int:
    """Open a comprehensive diagnostic dashboard."""
    print("Opening diagnostic dashboard...")
    result = generate_visualization(VisualizationType.DASHBOARD, "html")
    
    if result.get("success", False):
        print(f"Dashboard opened in browser: {result.get('output_path')}")
        return 0
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1

# Register commands
register_command("visualize", visualize_command, "Generate visualizations of diagnostic data")
register_command("dashboard", dashboard_command, "Open diagnostic dashboard in browser")

logger.info("Diagnostic visualization system initialized")
