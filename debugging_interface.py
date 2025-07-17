#!/usr/bin/env python3
"""
Debugging Interface Module

This module provides advanced debugging interfaces as described in Section 11.3,
including live inspection, state dumping, step-by-step execution, conditional breakpoints,
watch expressions, and visualization tools.
"""

import os
import sys
import json
import time
import inspect
import logging
import traceback
import threading
from typing import Dict, List, Any, Optional, Union, Set, Callable
from pathlib import Path

logger = logging.getLogger("DebuggingInterface")

class DebuggingState:
    """Singleton class to store global debugging state."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DebuggingState, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the debugging state."""
        self.paused = False
        self.step_mode = False
        self.breakpoints = {}  # {component: {location: condition}}
        self.watches = {}  # {watch_id: (expression, last_value)}
        self.execution_stack = []
        self.execution_history = []
        self.current_execution = None
        self.variables = {}
        self.inspection_active = False
        self.inspection_target = None
        self.inspection_results = {}
        self.freeze_timestamp = None
        self.exec_listener = None
        self.watch_thread = None
        self.watch_active = False
        self.visualization_data = {}
        self.agents = {}  # {agent_id: agent_state}
    
    def set_exec_listener(self, listener: Callable):
        """Set execution listener callback."""
        self.exec_listener = listener
    
    def record_execution(self, component: str, action: str, context: Dict = None):
        """
        Record an execution step.
        
        Args:
            component: Component name
            action: Action name
            context: Execution context
        """
        step = {
            "timestamp": time.time(),
            "component": component,
            "action": action,
            "context": context or {}
        }
        
        self.execution_stack.append(step)
        self.execution_history.append(step)
        
        # Keep history limited to last 1000 steps
        if len(self.execution_history) > 1000:
            self.execution_history.pop(0)
        
        # Call execution listener if registered
        if self.exec_listener:
            try:
                self.exec_listener(step)
            except Exception as e:
                logger.error(f"Error in execution listener: {e}")
    
    def clear_history(self):
        """Clear execution history."""
        self.execution_history = []
    
    def add_breakpoint(self, component: str, location: str, condition: str = None):
        """
        Add a breakpoint.
        
        Args:
            component: Component name
            location: Breakpoint location (e.g., "method_name")
            condition: Condition expression (optional)
        """
        if component not in self.breakpoints:
            self.breakpoints[component] = {}
        
        self.breakpoints[component][location] = condition
    
    def remove_breakpoint(self, component: str, location: str):
        """
        Remove a breakpoint.
        
        Args:
            component: Component name
            location: Breakpoint location
        """
        if component in self.breakpoints and location in self.breakpoints[component]:
            del self.breakpoints[component][location]
    
    def check_breakpoint(self, component: str, location: str, context: Dict = None) -> bool:
        """
        Check if a breakpoint should pause execution.
        
        Args:
            component: Component name
            location: Breakpoint location
            context: Execution context
            
        Returns:
            True if execution should pause, False otherwise
        """
        if component not in self.breakpoints or location not in self.breakpoints[component]:
            return False
        
        condition = self.breakpoints[component][location]
        
        if not condition:
            return True
        
        # Evaluate condition
        try:
            context_vars = context or {}
            result = eval(condition, {"__builtins__": {}}, context_vars)
            return bool(result)
        except Exception as e:
            logger.error(f"Error evaluating breakpoint condition: {e}")
            return False
    
    def add_watch(self, expression: str) -> str:
        """
        Add a watch expression.
        
        Args:
            expression: Watch expression
            
        Returns:
            Watch ID
        """
        watch_id = f"watch_{len(self.watches) + 1}"
        self.watches[watch_id] = (expression, None)
        return watch_id
    
    def remove_watch(self, watch_id: str):
        """
        Remove a watch expression.
        
        Args:
            watch_id: Watch ID
        """
        if watch_id in self.watches:
            del self.watches[watch_id]
    
    def update_watches(self, context: Dict = None):
        """
        Update watch expression values.
        
        Args:
            context: Execution context
        """
        context_vars = context or {}
        
        for watch_id, (expression, _) in self.watches.items():
            try:
                value = eval(expression, {"__builtins__": {}}, context_vars)
                self.watches[watch_id] = (expression, value)
            except Exception as e:
                self.watches[watch_id] = (expression, f"Error: {e}")
    
    def get_watches(self) -> Dict:
        """
        Get current watch values.
        
        Returns:
            Dictionary of watch expressions and their values
        """
        return {watch_id: {"expression": expr, "value": value} 
                for watch_id, (expr, value) in self.watches.items()}
    
    def freeze_execution(self):
        """Freeze execution at current point."""
        self.paused = True
        self.freeze_timestamp = time.time()
    
    def resume_execution(self):
        """Resume execution."""
        self.paused = False
        self.step_mode = False
        self.freeze_timestamp = None
    
    def step_execution(self):
        """Enable step mode."""
        self.paused = False
        self.step_mode = True
    
    def get_execution_stack(self) -> List[Dict]:
        """
        Get the current execution stack.
        
        Returns:
            List of execution steps
        """
        return self.execution_stack
    
    def get_execution_history(self, limit: int = None) -> List[Dict]:
        """
        Get execution history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of execution steps
        """
        if limit:
            return self.execution_history[-limit:]
        return self.execution_history
    
    def start_inspection(self, target: str):
        """
        Start inspection of a target.
        
        Args:
            target: Inspection target (component name)
        """
        self.inspection_active = True
        self.inspection_target = target
        self.inspection_results = {}
    
    def stop_inspection(self):
        """Stop active inspection."""
        self.inspection_active = False
        self.inspection_target = None
    
    def record_inspection_result(self, key: str, value: Any):
        """
        Record an inspection result.
        
        Args:
            key: Result key
            value: Result value
        """
        self.inspection_results[key] = value
    
    def get_inspection_results(self) -> Dict:
        """
        Get inspection results.
        
        Returns:
            Dictionary of inspection results
        """
        return self.inspection_results
    
    def start_watch_thread(self):
        """Start the watch thread for continuous monitoring."""
        if self.watch_thread and self.watch_thread.is_alive():
            return
        
        self.watch_active = True
        self.watch_thread = threading.Thread(target=self._watch_thread_func)
        self.watch_thread.daemon = True
        self.watch_thread.start()
    
    def stop_watch_thread(self):
        """Stop the watch thread."""
        self.watch_active = False
        if self.watch_thread:
            self.watch_thread = None
    
    def _watch_thread_func(self):
        """Watch thread function."""
        while self.watch_active:
            # Update watches with current variables
            self.update_watches(self.variables)
            time.sleep(1)
    
    def register_agent(self, agent_id: str, agent_data: Dict):
        """
        Register an agent for debugging.
        
        Args:
            agent_id: Agent ID
            agent_data: Agent data
        """
        self.agents[agent_id] = agent_data
    
    def unregister_agent(self, agent_id: str):
        """
        Unregister an agent.
        
        Args:
            agent_id: Agent ID
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """
        Get agent data.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent data or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_agents(self) -> Dict:
        """
        Get all registered agents.
        
        Returns:
            Dictionary of agents
        """
        return self.agents
    
    def add_visualization_data(self, viz_type: str, data: Any):
        """
        Add visualization data.
        
        Args:
            viz_type: Visualization type
            data: Visualization data
        """
        if viz_type not in self.visualization_data:
            self.visualization_data[viz_type] = []
        
        self.visualization_data[viz_type].append({
            "timestamp": time.time(),
            "data": data
        })
        
        # Keep limited history
        if len(self.visualization_data[viz_type]) > 100:
            self.visualization_data[viz_type].pop(0)
    
    def get_visualization_data(self, viz_type: str = None) -> Dict:
        """
        Get visualization data.
        
        Args:
            viz_type: Visualization type (optional)
            
        Returns:
            Visualization data
        """
        if viz_type:
            return {viz_type: self.visualization_data.get(viz_type, [])}
        return self.visualization_data


class DebuggingInterface:
    """Main debugging interface."""
    
    def __init__(self):
        """Initialize the debugging interface."""
        self.state = DebuggingState()
        self.registry = None
    
    def set_registry(self, registry):
        """Set the component registry."""
        self.registry = registry
    
    def inspect_live(self, target: str = None) -> Dict:
        """
        Perform live inspection.
        
        Args:
            target: Inspection target (optional)
            
        Returns:
            Inspection results
        """
        result = {}
        
        # Get registry
        if not self.registry:
            return {"error": "Component registry not available"}
        
        if target:
            # Inspect specific component
            component = self.registry.get_component(target)
            if not component:
                return {"error": f"Component not found: {target}"}
            
            # Inspect component
            result[target] = self._inspect_object(component)
        else:
            # Inspect all components
            for name, component in self.registry.get_components().items():
                result[name] = self._inspect_object(component)
        
        return result
    
    def dump_state(self, include_history: bool = False) -> Dict:
        """
        Dump current system state.
        
        Args:
            include_history: Include execution history
            
        Returns:
            System state dump
        """
        result = {
            "timestamp": time.time(),
            "components": {},
            "breakpoints": self.state.breakpoints,
            "watches": self.state.get_watches(),
            "execution_stack": self.state.get_execution_stack(),
            "paused": self.state.paused,
            "step_mode": self.state.step_mode,
            "agents": self.state.get_agents()
        }
        
        if include_history:
            result["execution_history"] = self.state.get_execution_history()
        
        # Get component states
        if self.registry:
            for name, component in self.registry.get_components().items():
                # Get component status if available
                if hasattr(component, "get_status") and callable(component.get_status):
                    try:
                        result["components"][name] = component.get_status()
                    except Exception as e:
                        result["components"][name] = {"error": str(e)}
                else:
                    result["components"][name] = {"message": "No status method available"}
        
        return result
    
    def add_breakpoint(self, component: str, location: str, condition: str = None) -> bool:
        """
        Add a breakpoint.
        
        Args:
            component: Component name
            location: Breakpoint location
            condition: Condition expression (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.state.add_breakpoint(component, location, condition)
            return True
        except Exception as e:
            logger.error(f"Error adding breakpoint: {e}")
            return False
    
    def remove_breakpoint(self, component: str, location: str) -> bool:
        """
        Remove a breakpoint.
        
        Args:
            component: Component name
            location: Breakpoint location
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.state.remove_breakpoint(component, location)
            return True
        except Exception as e:
            logger.error(f"Error removing breakpoint: {e}")
            return False
    
    def list_breakpoints(self) -> Dict:
        """
        List all breakpoints.
        
        Returns:
            Dictionary of breakpoints
        """
        return self.state.breakpoints
    
    def add_watch(self, expression: str) -> str:
        """
        Add a watch expression.
        
        Args:
            expression: Watch expression
            
        Returns:
            Watch ID
        """
        return self.state.add_watch(expression)
    
    def remove_watch(self, watch_id: str) -> bool:
        """
        Remove a watch expression.
        
        Args:
            watch_id: Watch ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.state.remove_watch(watch_id)
            return True
        except Exception as e:
            logger.error(f"Error removing watch: {e}")
            return False
    
    def get_watches(self) -> Dict:
        """
        Get current watch values.
        
        Returns:
            Dictionary of watch expressions and their values
        """
        return self.state.get_watches()
    
    def pause_execution(self) -> bool:
        """
        Pause execution.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.state.freeze_execution()
            return True
        except Exception as e:
            logger.error(f"Error pausing execution: {e}")
            return False
    
    def resume_execution(self) -> bool:
        """
        Resume execution.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.state.resume_execution()
            return True
        except Exception as e:
            logger.error(f"Error resuming execution: {e}")
            return False
    
    def step_execution(self) -> bool:
        """
        Step execution.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.state.step_execution()
            return True
        except Exception as e:
            logger.error(f"Error stepping execution: {e}")
            return False
    
    def get_execution_stack(self) -> List[Dict]:
        """
        Get the current execution stack.
        
        Returns:
            List of execution steps
        """
        return self.state.get_execution_stack()
    
    def get_execution_history(self, limit: int = None) -> List[Dict]:
        """
        Get execution history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of execution steps
        """
        return self.state.get_execution_history(limit)
    
    def clear_history(self) -> bool:
        """
        Clear execution history.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.state.clear_history()
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False
    
    def visualize_execution_flow(self, limit: int = None) -> str:
        """
        Visualize execution flow.
        
        Args:
            limit: Maximum number of history items to include
            
        Returns:
            ASCII visualization of execution flow
        """
        history = self.state.get_execution_history(limit)
        if not history:
            return "No execution history available"
        
        result = []
        result.append("Execution Flow")
        result.append("=" * 14)
        result.append("")
        
        # Group by component
        components = {}
        for step in history:
            component = step["component"]
            if component not in components:
                components[component] = []
            components[component].append(step)
        
        # Create timeline
        timeline = []
        start_time = history[0]["timestamp"]
        
        for component, steps in components.items():
            component_line = f"{component:15} |"
            
            for i, step in enumerate(history):
                if step["component"] == component:
                    component_line += "O"
                else:
                    component_line += " "
            
            timeline.append(component_line)
        
        # Add time markers
        time_line = "               |"
        last_time = start_time
        
        for step in history:
            time_diff = step["timestamp"] - last_time
            if time_diff > 1.0:
                time_line += "+"
            else:
                time_line += "-"
            last_time = step["timestamp"]
        
        timeline.append(time_line)
        
        # Add labels for important events
        label_lines = []
        line_length = len(timeline[0])
        
        for i, step in enumerate(history):
            if "action" in step and step["action"] in ["start", "complete", "error"]:
                pos = 16 + i
                if pos < line_length:
                    label = f"{step['action']} ({step['component']})"
                    label_line = " " * pos + "| " + label
                    label_lines.append(label_line)
        
        # Combine all lines
        result.extend(timeline)
        if label_lines:
            result.append("")
            result.extend(label_lines)
        
        return "\n".join(result)
    
    def visualize_state_transitions(self, component: str = None, limit: int = 20) -> str:
        """
        Visualize state transitions.
        
        Args:
            component: Component name (optional)
            limit: Maximum number of transitions to include
            
        Returns:
            ASCII visualization of state transitions
        """
        history = self.state.get_execution_history()
        if not history:
            return "No execution history available"
        
        # Filter by component
        if component:
            history = [step for step in history if step["component"] == component]
        
        if not history:
            return f"No execution history available for component: {component}"
        
        # Extract state transitions
        transitions = []
        
        for step in history:
            if "context" in step and "state_change" in step["context"]:
                transitions.append({
                    "timestamp": step["timestamp"],
                    "component": step["component"],
                    "from_state": step["context"].get("from_state", "unknown"),
                    "to_state": step["context"].get("to_state", "unknown"),
                    "trigger": step["action"]
                })
        
        if not transitions:
            return "No state transitions found in execution history"
        
        # Limit the number of transitions
        transitions = transitions[-limit:]
        
        # Create visualization
        result = []
        result.append("State Transitions")
        result.append("=" * 16)
        result.append("")
        
        for i, transition in enumerate(transitions):
            comp = transition["component"]
            from_state = transition["from_state"]
            to_state = transition["to_state"]
            trigger = transition["trigger"]
            
            result.append(f"{comp}: {from_state} --> {to_state}  [{trigger}]")
            
            # Add arrow if not the last transition
            if i < len(transitions) - 1:
                result.append("         |")
                result.append("         V")
        
        return "\n".join(result)
    
    def visualize_agent_interactions(self, limit: int = None) -> str:
        """
        Visualize agent interactions.
        
        Args:
            limit: Maximum number of interactions to include
            
        Returns:
            ASCII visualization of agent interactions
        """
        history = self.state.get_execution_history(limit)
        if not history:
            return "No execution history available"
        
        # Filter agent interactions
        interactions = []
        
        for step in history:
            if ("context" in step and "interaction" in step["context"] and
                    "source" in step["context"] and "target" in step["context"]):
                interactions.append({
                    "timestamp": step["timestamp"],
                    "source": step["context"]["source"],
                    "target": step["context"]["target"],
                    "type": step["context"]["interaction"],
                    "message": step["context"].get("message", "")
                })
        
        if not interactions:
            return "No agent interactions found in execution history"
        
        # Create visualization
        result = []
        result.append("Agent Interactions")
        result.append("=" * 18)
        result.append("")
        
        # Get unique agents
        agents = set()
        for interaction in interactions:
            agents.add(interaction["source"])
            agents.add(interaction["target"])
        
        agents = sorted(list(agents))
        agent_positions = {agent: i * 20 for i, agent in enumerate(agents)}
        
        # Create header
        header = " " * 10
        for agent in agents:
            pos = agent_positions[agent]
            header += " " * (pos - len(header)) + agent
        
        result.append(header)
        result.append("-" * len(header))
        
        # Add interactions
        for interaction in interactions:
            source = interaction["source"]
            target = interaction["target"]
            message = interaction["message"]
            
            source_pos = agent_positions[source] + len(source) // 2
            target_pos = agent_positions[target] + len(target) // 2
            
            # Create arrow
            if source_pos < target_pos:
                line = " " * source_pos + "-" * (target_pos - source_pos - 1) + ">"
            else:
                line = " " * target_pos + "<" + "-" * (source_pos - target_pos - 1)
            
            # Add interaction type
            interaction_type = interaction["type"]
            if len(interaction_type) < abs(target_pos - source_pos) - 2:
                type_pos = min(source_pos, target_pos) + abs(target_pos - source_pos) // 2 - len(interaction_type) // 2
                line = line[:type_pos] + interaction_type + line[type_pos + len(interaction_type):]
            
            result.append(line)
            
            # Add message if it fits
            if message and len(message) < len(line):
                message_pos = (source_pos + target_pos) // 2 - len(message) // 2
                message_line = " " * message_pos + message
                result.append(message_line)
        
        return "\n".join(result)
    
    def export_visualization(self, visualization: str, output_path: str, format: str = "text") -> bool:
        """
        Export visualization to a file.
        
        Args:
            visualization: Visualization string
            output_path: Output file path
            format: Output format (text, html, svg)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Export based on format
            if format == "text":
                with open(output_path, "w") as f:
                    f.write(visualization)
            elif format == "html":
                html = self._convert_to_html(visualization)
                with open(output_path, "w") as f:
                    f.write(html)
            elif format == "svg":
                svg = self._convert_to_svg(visualization)
                with open(output_path, "w") as f:
                    f.write(svg)
            else:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error exporting visualization: {e}")
            return False
    
    def _inspect_object(self, obj: Any) -> Dict:
        """
        Inspect an object.
        
        Args:
            obj: Object to inspect
            
        Returns:
            Inspection results
        """
        result = {
            "type": type(obj).__name__,
            "id": id(obj),
            "attributes": {},
            "methods": {},
            "doc": inspect.getdoc(obj) or "No documentation"
        }
        
        # Get attributes and methods
        for name, value in inspect.getmembers(obj):
            # Skip private attributes and methods
            if name.startswith("_"):
                continue
            
            # Skip built-in attributes and methods
            if name in dir(type):
                continue
            
            if inspect.ismethod(value) or inspect.isfunction(value):
                # Method
                sig = inspect.signature(value)
                result["methods"][name] = {
                    "signature": str(sig),
                    "doc": inspect.getdoc(value) or "No documentation"
                }
            else:
                # Attribute
                try:
                    # Try to get a simple representation of the value
                    if isinstance(value, (str, int, float, bool, type(None))):
                        attr_value = value
                    elif isinstance(value, (list, tuple)):
                        attr_value = f"{type(value).__name__}[{len(value)}]"
                    elif isinstance(value, dict):
                        attr_value = f"dict[{len(value)}]"
                    else:
                        attr_value = type(value).__name__
                    
                    result["attributes"][name] = attr_value
                except Exception as e:
                    result["attributes"][name] = f"Error: {e}"
        
        return result
    
    def _convert_to_html(self, visualization: str) -> str:
        """Convert ASCII visualization to HTML."""
        # Replace special characters
        html = visualization.replace("<", "&lt;").replace(">", "&gt;")
        
        # Convert newlines to <br>
        html = html.replace("\n", "<br>\n")
        
        # Create HTML document
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Debugging Visualization</title>
    <style>
        body {{ font-family: monospace; white-space: pre; }}
    </style>
</head>
<body>
{html}
</body>
</html>
"""
    
    def _convert_to_svg(self, visualization: str) -> str:
        """Convert ASCII visualization to SVG."""
        lines = visualization.split("\n")
        height = len(lines) * 20  # 20px per line
        max_width = max(len(line) for line in lines) * 8  # 8px per character
        
        # Create SVG header
        svg = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{max_width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
"""
        
        # Add lines as text elements
        for i, line in enumerate(lines):
            y = (i + 1) * 20
            svg += f'  <text x="0" y="{y}" font-family="monospace">{line}</text>\n'
        
        # Close SVG
        svg += "</svg>"
        
        return svg


# Command handlers for debugging interface

def debug_command(args: str) -> int:
    """
    Main debug command handler.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    cmd_args = shlex.split(args)
    
    # No arguments - show help
    if not cmd_args:
        return debug_help_command("")
    
    command = cmd_args[0].lower()
    remaining_args = ' '.join(cmd_args[1:])
    
    # Dispatch to appropriate command
    if command == "help":
        return debug_help_command(remaining_args)
    elif command == "inspect":
        return debug_inspect_command(remaining_args)
    elif command == "dump":
        return debug_dump_command(remaining_args)
    elif command == "break":
        return debug_breakpoint_command(remaining_args)
    elif command == "watch":
        return debug_watch_command(remaining_args)
    elif command == "step":
        return debug_step_command(remaining_args)
    elif command == "pause":
        return debug_pause_command(remaining_args)
    elif command == "resume":
        return debug_resume_command(remaining_args)
    elif command == "history":
        return debug_history_command(remaining_args)
    elif command == "visualize":
        return debug_visualize_command(remaining_args)
    else:
        print(f"Unknown debug command: {command}")
        print("Type 'debug help' for a list of commands")
        return 1

def debug_help_command(args: str) -> int:
    """
    Display debug help.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    print("\nDebugging Interface Commands:")
    print("=============================")
    print("debug help                   - Show this help")
    print("debug inspect [component]    - Inspect component(s)")
    print("debug dump [--history]       - Dump system state")
    print("debug break add <comp> <loc> - Add breakpoint")
    print("debug break list             - List breakpoints")
    print("debug break remove <id>      - Remove breakpoint")
    print("debug watch add <expr>       - Add watch expression")
    print("debug watch list             - List watches")
    print("debug watch remove <id>      - Remove watch")
    print("debug step                   - Step execution")
    print("debug pause                  - Pause execution")
    print("debug resume                 - Resume execution")
    print("debug history [--limit <n>]  - Show execution history")
    print("debug visualize <type>       - Visualize execution")
    print("\nVisualization Types:")
    print("  flow       - Execution flow")
    print("  states     - State transitions")
    print
