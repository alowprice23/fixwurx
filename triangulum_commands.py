#!/usr/bin/env python3
"""
Triangulum Commands Module

This module registers command handlers for the Triangulum system within the shell environment,
enabling resource management, system monitoring, and orchestration capabilities.
"""

import os
import sys
import json
import time
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger("TriangulumCommands")

def register_triangulum_commands(registry):
    """
    Register Triangulum command handlers with the component registry.
    
    Args:
        registry: Component registry instance
    """
    # Try to import Triangulum components
    try:
        from triangulum_resource_manager import create_triangulum_resource_manager
        
        # Create and register resource manager
        resource_manager = create_triangulum_resource_manager()
        registry.register_component("triangulum", resource_manager)
        
        # Register command handlers
        registry.register_command_handler("run", run_command, "triangulum")
        registry.register_command_handler("status", status_command, "triangulum")
        registry.register_command_handler("queue", queue_command, "triangulum")
        registry.register_command_handler("rollback", rollback_command, "triangulum")
        registry.register_command_handler("dashboard", dashboard_command, "triangulum")
        
        registry.register_command_handler("triangulum:run", run_command, "triangulum")
        registry.register_command_handler("triangulum:status", status_command, "triangulum")
        registry.register_command_handler("triangulum:queue", queue_command, "triangulum")
        registry.register_command_handler("triangulum:rollback", rollback_command, "triangulum")
        registry.register_command_handler("triangulum:dashboard", dashboard_command, "triangulum")
        registry.register_command_handler("triangulum:plan", plan_command, "triangulum")
        registry.register_command_handler("triangulum:agents", agents_command, "triangulum")
        registry.register_command_handler("triangulum:entropy", entropy_command, "triangulum")
        
        # Register event handlers
        registry.register_event_handler("system_start", system_start_handler, "triangulum")
        registry.register_event_handler("system_stop", system_stop_handler, "triangulum")
        registry.register_event_handler("resource_allocation", resource_allocation_handler, "triangulum")
        
        logger.info("Triangulum commands registered")
    except ImportError:
        logger.warning("Failed to import Triangulum components")
    except Exception as e:
        logger.error(f"Error registering Triangulum commands: {e}")

def run_command(args: str) -> int:
    """
    Run the Triangulum system.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the Triangulum system")
    parser.add_argument("--config", default="system_config.yaml", help="Path to configuration file")
    parser.add_argument("--tick-ms", type=int, help="Override tick rate in milliseconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    config_file = cmd_args.config
    tick_ms = cmd_args.tick_ms
    verbose = cmd_args.verbose
    
    # Check if config file exists
    if not Path(config_file).exists():
        print(f"Error: Config file '{config_file}' not found")
        return 1
    
    # Use our helper script to start Triangulum properly
    cmd = [
        sys.executable,
        "start_triangulum.py"
    ]
    
    if verbose:
        print(f"Executing: {' '.join(cmd)}")
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.PIPE,
            text=True
        )
        
        # For verbose mode, we'll stay with the process and show output
        if verbose:
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\nShutdown requested. Terminating...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("Forcefully killing process...")
                    process.kill()
        else:
            # For non-verbose mode, capture and display the output
            stdout, stderr = process.communicate()
            if stdout:
                print(stdout.strip())
            if stderr:
                print(stderr.strip())
        
        return 0
    except Exception as e:
        print(f"Error running Triangulum: {e}")
        return 1

def status_command(args: str) -> int:
    """
    Display Triangulum system status.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Display Triangulum system status")
    parser.add_argument("--lines", "-n", type=int, default=20, help="Number of log lines to show")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    lines = cmd_args.lines
    follow = cmd_args.follow
    
    # Get resource manager
    registry = sys.modules.get("__main__").registry
    resource_manager = registry.get_component("triangulum")
    
    if not resource_manager:
        print("Error: Triangulum resource manager not available")
        return 1
    
    # Helper function to tail metrics
    def tail_metrics(n=20):
        """Read the tail of stderr log produced by SystemMonitor."""
        logf = Path(".triangulum/runtime.log")
        if not logf.exists():
            return ["No runtime.log yet – is Triangulum running with StdoutBus?"]

        try:
            log_lines = logf.read_text(encoding="utf-8").splitlines()[-n:]
            return log_lines
        except Exception as e:
            return [f"Error reading log: {e}"]
    
    # Check if Triangulum is running
    if not resource_manager.is_running():
        print("Triangulum system is not running")
        return 0
    
    # Get system status
    status = resource_manager.get_status()
    
    # Print status header
    print("\nTriangulum System Status:")
    print("=" * 60)
    
    # Print general status
    print(f"Running: Yes")
    print(f"Process ID: {status.get('process_id', 'Unknown')}")
    print(f"Started: {status.get('start_time', 'Unknown')}")
    print(f"Uptime: {status.get('uptime', 'Unknown')}")
    
    # Print resource usage
    resources = status.get("resources", {})
    if resources:
        print("\nResource Usage:")
        for key, value in resources.items():
            print(f"  {key}: {value}")
    
    # Print latest metrics
    if follow:
        try:
            # Live monitoring mode
            print("\nLive monitoring mode (press Ctrl+C to exit)")
            last_mtime = 0
            while True:
                logf = Path(".triangulum/runtime.log")
                if logf.exists():
                    mtime = logf.stat().st_mtime
                    if mtime > last_mtime:
                        print("\nLatest Metrics:")
                        print("-" * 60)
                        for line in tail_metrics(lines):
                            print(line)
                        last_mtime = mtime
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    else:
        # Regular mode
        print("\nLatest Metrics:")
        print("-" * 60)
        for line in tail_metrics(lines):
            print(line)
    
    return 0

def queue_command(args: str) -> int:
    """
    List items in the queue.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    import sqlite3
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="List queue items")
    parser.add_argument("--filter", help="Filter by status (e.g., PENDING)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    filter_status = cmd_args.filter
    verbose = cmd_args.verbose
    
    # Path to the review database
    review_db = Path(".triangulum/reviews.sqlite")
    
    # Check if the database exists
    if not review_db.exists():
        print("Review database not found - no queue yet")
        return 0
    
    # Helper function to format age
    def fmt_age(ts):
        secs = int(time.time() - ts)
        return "{:02}:{:02}:{:02}".format(secs // 3600, (secs % 3600) // 60, secs % 60)
    
    # Fetch queue items from the database
    try:
        conn = sqlite3.connect(review_db)
        cur = conn.cursor()
        
        if filter_status:
            cur.execute(
                "SELECT id, bug_id, status, created_at FROM reviews WHERE status=? ORDER BY id DESC",
                (filter_status.upper(),)
            )
        else:
            cur.execute("SELECT id, bug_id, status, created_at FROM reviews ORDER BY id DESC")
        
        rows = cur.fetchall()
        
        if not rows:
            print("No queue items found")
            return 0
        
        # Print queue items
        print("id bug_id     status      age")
        print("─" * 50)
        
        for rid, bug, st, ts in rows:
            print(f"{rid:<3} {bug:<9} {st:<10} {fmt_age(ts)}")
        
        # Print summary if verbose
        if verbose and rows:
            print(f"\nTotal items: {len(rows)}")
            
            # Count by status
            status_counts = {}
            for _, _, status, _ in rows:
                status_counts[status] = status_counts.get(status, 0) + 1
            
            print("\nStatus counts:")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")
        
        return 0
    except Exception as e:
        print(f"Error querying database: {e}")
        return 1
    finally:
        if 'conn' in locals():
            conn.close()

def rollback_command(args: str) -> int:
    """
    Execute a rollback operation.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Execute a rollback operation")
    parser.add_argument("review_id", type=int, help="Review ID to roll back")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    review_id = cmd_args.review_id
    
    # Check if rollback manager is available
    try:
        from rollback_manager import rollback_patch
    except ImportError:
        print("Rollback functionality not available")
        return 1
    
    # Execute rollback
    try:
        print(f"Rolling back review {review_id}...")
        rollback_patch(review_id)
        print(f"✓ Rollback of review {review_id} finished")
        return 0
    except Exception as e:
        print(f"✗ Rollback failed: {e}")
        return 1

def dashboard_command(args: str) -> int:
    """
    Start the metrics dashboard server.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Start the metrics dashboard server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the dashboard on")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    port = cmd_args.port
    
    # Check if uvicorn is available
    try:
        import uvicorn
    except ImportError:
        print("✗ Uvicorn not found. Install with:")
        print("  pip install uvicorn")
        return 1
    
    # Check if dashboard module is available
    try:
        import monitoring.dashboard as dashboard
    except ImportError:
        print("✗ Dashboard module not found")
        return 1
    
    # Start the dashboard
    print(f"Starting dashboard on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    
    try:
        uvicorn.run(dashboard.app, host="0.0.0.0", port=port)
        return 0
    except KeyboardInterrupt:
        print("\nDashboard stopped")
        return 0
    except Exception as e:
        print(f"✗ Failed to start dashboard: {e}")
        return 1

def plan_command(args: str) -> int:
    """
    Manage execution plans.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Manage execution plans")
    parser.add_argument("action", nargs="?", choices=["list", "view", "create", "execute", "cancel"], default="list", 
                        help="Action to perform")
    parser.add_argument("plan_id", nargs="?", help="Plan ID for view/execute/cancel actions")
    parser.add_argument("--file", help="Path to plan file for create action")
    parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                       help="Output format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    action = cmd_args.action
    plan_id = cmd_args.plan_id
    plan_file = cmd_args.file
    output_format = cmd_args.format
    
    # Get resource manager
    registry = sys.modules.get("__main__").registry
    resource_manager = registry.get_component("triangulum")
    
    if not resource_manager:
        print("Error: Triangulum resource manager not available")
        return 1
    
    try:
        # Check if plan storage is available
        try:
            from plan_storage import PlanStorage
            plan_storage = PlanStorage()
        except ImportError:
            print("Error: Plan storage not available")
            return 1
        
        # List plans
        if action == "list":
            plans = plan_storage.get_all_plans()
            
            if output_format == "json":
                print(json.dumps(plans, indent=2, default=str))
            elif output_format == "yaml":
                import yaml
                print(yaml.dump(plans, default_flow_style=False))
            else:  # text format
                print("\nExecution Plans:")
                
                if not plans:
                    print("No plans found")
                    return 0
                
                for i, plan in enumerate(plans, 1):
                    plan_id = plan.get("id", "Unknown")
                    status = plan.get("status", "Unknown")
                    created_at = plan.get("created_at", "Unknown")
                    description = plan.get("description", "No description")
                    
                    print(f"  {i}. {plan_id} - {status}")
                    print(f"     Created: {created_at}")
                    print(f"     Description: {description}")
                    print()
        
        # View plan details
        elif action == "view" and plan_id:
            plan = plan_storage.get_plan(plan_id)
            
            if not plan:
                print(f"Plan '{plan_id}' not found")
                return 1
            
            if output_format == "json":
                print(json.dumps(plan, indent=2, default=str))
            elif output_format == "yaml":
                import yaml
                print(yaml.dump(plan, default_flow_style=False))
            else:  # text format
                print(f"\nPlan: {plan_id}")
                print("=" * 60)
                
                # Print plan details
                print(f"Status: {plan.get('status', 'Unknown')}")
                print(f"Created: {plan.get('created_at', 'Unknown')}")
                print(f"Description: {plan.get('description', 'No description')}")
                
                # Print phases
                phases = plan.get("phases", [])
                if phases:
                    print("\nPhases:")
                    for i, phase in enumerate(phases, 1):
                        phase_id = phase.get("id", "Unknown")
                        phase_type = phase.get("type", "Unknown")
                        status = phase.get("status", "Unknown")
                        
                        print(f"  {i}. {phase_id} - {phase_type} - {status}")
                
                # Print progress
                progress = plan.get("progress", {})
                if progress:
                    print("\nProgress:")
                    for key, value in progress.items():
                        print(f"  {key}: {value}")
        
        # Create new plan
        elif action == "create" and plan_file:
            if not Path(plan_file).exists():
                print(f"Error: Plan file '{plan_file}' not found")
                return 1
            
            # Load plan from file
            try:
                with open(plan_file, 'r') as f:
                    if plan_file.endswith('.json'):
                        import json
                        plan_data = json.load(f)
                    elif plan_file.endswith('.yaml') or plan_file.endswith('.yml'):
                        import yaml
                        plan_data = yaml.safe_load(f)
                    else:
                        print("Error: Plan file must be JSON or YAML")
                        return 1
            except Exception as e:
                print(f"Error loading plan file: {e}")
                return 1
            
            # Create the plan
            plan_id = plan_storage.create_plan(plan_data)
            print(f"Plan '{plan_id}' created successfully")
        
        # Execute plan
        elif action == "execute" and plan_id:
            # Check if Triangulum is running
            if not resource_manager.is_running():
                print("Error: Triangulum system is not running")
                return 1
            
            # Execute the plan
            result = resource_manager.execute_plan(plan_id)
            
            if result.get("success", False):
                print(f"Plan '{plan_id}' execution started")
                print(f"Execution ID: {result.get('execution_id', 'Unknown')}")
            else:
                print(f"Error executing plan: {result.get('error', 'Unknown error')}")
                return 1
        
        # Cancel plan
        elif action == "cancel" and plan_id:
            # Check if Triangulum is running
            if not resource_manager.is_running():
                print("Error: Triangulum system is not running")
                return 1
            
            # Cancel the plan
            result = resource_manager.cancel_plan(plan_id)
            
            if result.get("success", False):
                print(f"Plan '{plan_id}' cancelled successfully")
            else:
                print(f"Error cancelling plan: {result.get('error', 'Unknown error')}")
                return 1
        
        else:
            print("Error: Invalid action or missing required arguments")
            print("Usage examples:")
            print("  plan list")
            print("  plan view <plan_id>")
            print("  plan create --file <plan_file>")
            print("  plan execute <plan_id>")
            print("  plan cancel <plan_id>")
            return 1
        
        return 0
    except Exception as e:
        print(f"Error managing plans: {e}")
        return 1

def agents_command(args: str) -> int:
    """
    Manage and monitor agents.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Manage and monitor agents")
    parser.add_argument("action", nargs="?", choices=["list", "status", "stop"], default="list", 
                        help="Action to perform")
    parser.add_argument("agent_id", nargs="?", help="Agent ID for status/stop actions")
    parser.add_argument("--type", help="Filter by agent type")
    parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                       help="Output format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    action = cmd_args.action
    agent_id = cmd_args.agent_id
    agent_type = cmd_args.type
    output_format = cmd_args.format
    
    # Get resource manager
    registry = sys.modules.get("__main__").registry
    resource_manager = registry.get_component("triangulum")
    
    if not resource_manager:
        print("Error: Triangulum resource manager not available")
        return 1
    
    try:
        # List agents
        if action == "list":
            # Check if Triangulum is running
            if not resource_manager.is_running():
                print("Error: Triangulum system is not running")
                return 1
            
            # Get agents
            agents = resource_manager.list_agents(agent_type=agent_type)
            
            if output_format == "json":
                print(json.dumps(agents, indent=2, default=str))
            elif output_format == "yaml":
                import yaml
                print(yaml.dump(agents, default_flow_style=False))
            else:  # text format
                print("\nActive Agents:")
                
                if not agents:
                    print("No agents found")
                    return 0
                
                print("id agent_type      status     last_active   plan_id")
                print("─" * 60)
                
                for agent in agents:
                    agent_id = agent.get("id", "Unknown")
                    agent_type = agent.get("type", "Unknown")
                    status = agent.get("status", "Unknown")
                    last_active = agent.get("last_active", "Unknown")
                    plan_id = agent.get("plan_id", "N/A")
                    
                    print(f"{agent_id:<3} {agent_type:<14} {status:<10} {last_active:<12} {plan_id}")
        
        # Show agent status
        elif action == "status" and agent_id:
            # Check if Triangulum is running
            if not resource_manager.is_running():
                print("Error: Triangulum system is not running")
                return 1
            
            # Get agent status
            agent = resource_manager.get_agent_status(agent_id)
            
            if not agent:
                print(f"Agent '{agent_id}' not found")
                return 1
            
            if output_format == "json":
                print(json.dumps(agent, indent=2, default=str))
            elif output_format == "yaml":
                import yaml
                print(yaml.dump(agent, default_flow_style=False))
            else:  # text format
                print(f"\nAgent: {agent_id}")
                print("=" * 60)
                
                # Print agent details
                print(f"Type: {agent.get('type', 'Unknown')}")
                print(f"Status: {agent.get('status', 'Unknown')}")
                print(f"Last Active: {agent.get('last_active', 'Unknown')}")
                
                # Print plan association
                plan_id = agent.get("plan_id")
                if plan_id:
                    print(f"Associated Plan: {plan_id}")
                
                # Print memory usage
                memory = agent.get("memory_mb", 0)
                print(f"Memory Usage: {memory:.1f} MB")
                
                # Print capabilities
                capabilities = agent.get("capabilities", [])
                if capabilities:
                    print("\nCapabilities:")
                    for capability in capabilities:
                        print(f"  • {capability}")
                
                # Print current task
                task = agent.get("current_task")
                if task:
                    print("\nCurrent Task:")
                    print(f"  {task}")
                
                # Print metrics
                metrics = agent.get("metrics", {})
                if metrics:
                    print("\nMetrics:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
        
        # Stop agent
        elif action == "stop" and agent_id:
            # Check if Triangulum is running
            if not resource_manager.is_running():
                print("Error: Triangulum system is not running")
                return 1
            
            # Stop the agent
            result = resource_manager.stop_agent(agent_id)
            
            if result.get("success", False):
                print(f"Agent '{agent_id}' stopped successfully")
            else:
                print(f"Error stopping agent: {result.get('error', 'Unknown error')}")
                return 1
        
        else:
            print("Error: Invalid action or missing required arguments")
            print("Usage examples:")
            print("  agents list [--type <agent_type>]")
            print("  agents status <agent_id>")
            print("  agents stop <agent_id>")
            return 1
        
        return 0
    except Exception as e:
        print(f"Error managing agents: {e}")
        return 1

def entropy_command(args: str) -> int:
    """
    Display and manage system entropy.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Display and manage system entropy")
    parser.add_argument("action", nargs="?", choices=["view", "analyze", "reduce"], default="view", 
                        help="Action to perform")
    parser.add_argument("--component", help="Filter by component")
    parser.add_argument("--threshold", type=float, default=0.7, help="Entropy threshold")
    parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                       help="Output format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    action = cmd_args.action
    component = cmd_args.component
    threshold = cmd_args.threshold
    output_format = cmd_args.format
    
    # Get resource manager
    registry = sys.modules.get("__main__").registry
    resource_manager = registry.get_component("triangulum")
    
    if not resource_manager:
        print("Error: Triangulum resource manager not available")
        return 1
    
    try:
        # Check if entropy calculator is available
        try:
            # This is a placeholder - in a real implementation, we would import the actual entropy calculator
            from scope_filter import EntropyCalculator
            entropy_calculator = EntropyCalculator()
        except ImportError:
            print("Error: Entropy calculator not available")
            return 1
        
        # View entropy
        if action == "view":
            # Get entropy data
            entropy_data = entropy_calculator.get_entropy(component=component)
            
            if output_format == "json":
                print(json.dumps(entropy_data, indent=2, default=str))
            elif output_format == "yaml":
                import yaml
                print(yaml.dump(entropy_data, default_flow_style=False))
            else:  # text format
                print("\nSystem Entropy:")
                
                # Print overall entropy
                overall = entropy_data.get("overall", 0)
                print(f"Overall Entropy: {overall:.4f}")
                
                # Print component entropy
                components = entropy_data.get("components", {})
                if components:
                    print("\nComponent Entropy:")
                    for comp_name, comp_entropy in sorted(components.items(), key=lambda x: x[1], reverse=True):
                        status = "HIGH" if comp_entropy > threshold else "NORMAL"
                        print(f"  {comp_name}: {comp_entropy:.4f} - {status}")
                
                # Print entropy trend
                trend = entropy_data.get("trend", {})
                if trend:
                    print("\nEntropy Trend:")
                    for key, value in trend.items():
                        print(f"  {key}: {value}")
        
        # Analyze entropy
        elif action == "analyze":
            # Get entropy analysis
            analysis = entropy_calculator.analyze_entropy(component=component, threshold=threshold)
            
            if output_format == "json":
                print(json.dumps(analysis, indent=2, default=str))
            elif output_format == "yaml":
                import yaml
                print(yaml.dump(analysis, default_flow_style=False))
            else:  # text format
                print("\nEntropy Analysis:")
                
                # Print analysis summary
                print(f"Overall Entropy: {analysis.get('overall', 0):.4f}")
                print(f"Threshold: {threshold}")
                
                # Print high entropy components
                high_entropy = analysis.get("high_entropy_components", [])
                if high_entropy:
                    print("\nHigh Entropy Components:")
                    for comp in high_entropy:
                        print(f"  • {comp.get('name', 'Unknown')}: {comp.get('entropy', 0):.4f}")
                        print(f"    Reason: {comp.get('reason', 'Unknown')}")
                
                # Print recommendations
                recommendations = analysis.get("recommendations", [])
                if recommendations:
                    print("\nRecommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"  {i}. {rec}")
        
        # Reduce entropy
        elif action == "reduce":
            # Check if Triangulum is running
            if not resource_manager.is_running():
                print("Error: Triangulum system is not running")
                return 1
            
            print(f"Reducing system entropy...")
            if component:
                print(f"Focusing on component: {component}")
            
            # Run entropy reduction
            result = entropy_calculator.reduce_entropy(component=component, threshold=threshold)
            
            if output_format == "json":
                print(json.dumps(result, indent=2, default=str))
            elif output_format == "yaml":
                import yaml
                print(yaml.dump(result, default_flow_style=False))
            else:  # text format
                print("\nEntropy Reduction Results:")
                
                # Print summary
                initial = result.get("initial_entropy", 0)
                final = result.get("final_entropy", 0)
                print(f"Initial Entropy: {initial:.4f}")
                print(f"Final Entropy: {final:.4f}")
                print(f"Reduction: {initial - final:.4f} ({(1 - final/initial)*100:.1f}%)")
                
                # Print actions taken
                actions = result.get("actions", [])
                if actions:
                    print("\nActions Taken:")
                    for i, action in enumerate(actions, 1):
                        print(f"  {i}. {action.get('description', 'Unknown')}")
                        print(f"     Component: {action.get('component', 'Unknown')}")
                        print(f"     Entropy Reduction: {action.get('reduction', 0):.4f}")
                
                # Print remaining issues
                remaining_issues = result.get("remaining_issues", [])
                if remaining_issues:
                    print("\nRemaining Issues:")
                    for i, issue in enumerate(remaining_issues, 1):
                        print(f"  {i}. {issue}")
        
        else:
            print("Error: Invalid action")
            print("Usage examples:")
            print("  entropy view [--component <component>]")
            print("  entropy analyze [--component <component>] [--threshold <threshold>]")
            print("  entropy reduce [--component <component>] [--threshold <threshold>]")
            return 1
        
        return 0
    except Exception as e:
        print(f"Error managing entropy: {e}")
        return 1

def system_start_handler(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle system start events.
    
    Args:
        event_data: Event data
        
    Returns:
        Result data
    """
    try:
        logger.info("System start event received")
        
        # Get resource manager
        registry = sys.modules.get("__main__").registry
        resource_manager = registry.get_component("triangulum")
        
        if not resource_manager:
            return {"success": False, "error": "Triangulum resource manager not available"}
        
        # Extract event data
        config_file = event_data.get("config_file", "system_config.yaml")
        process_id = event_data.get("process_id")
        
        # Record system start
        if process_id:
            resource_manager.set_process_id(process_id)
        
        # Set start time
        resource_manager.set_start_time(datetime.datetime.now())
        
        # Log start
        logger.info(f"Triangulum system started with config: {config_file}")
        
        return {
            "success": True,
            "config_file": config_file,
            "start_time": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error handling system start event: {e}")
        return {"success": False, "error": str(e)}

def system_stop_handler(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle system stop events.
    
    Args:
        event_data: Event data
        
    Returns:
        Result data
    """
    try:
        logger.info("System stop event received")
        
        # Get resource manager
        registry = sys.modules.get("__main__").registry
        resource_manager = registry.get_component("triangulum")
        
        if not resource_manager:
            return {"success": False, "error": "Triangulum resource manager not available"}
        
        # Extract event data
        reason = event_data.get("reason", "Unknown")
        clean = event_data.get("clean", True)
        
        # Record system stop
        resource_manager.clear_process_id()
        
        # Log stop
        logger.info(f"Triangulum system stopped. Reason: {reason}, Clean: {clean}")
        
        return {
            "success": True,
            "reason": reason,
            "clean": clean,
            "stop_time": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error handling system stop event: {e}")
        return {"success": False, "error": str(e)}

def resource_allocation_handler(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle resource allocation events.
    
    Args:
        event_data: Event data
        
    Returns:
        Result data
    """
    try:
        logger.info("Resource allocation event received")
        
        # Get resource manager
        registry = sys.modules.get("__main__").registry
        resource_manager = registry.get_component("triangulum")
        
        if not resource_manager:
            return {"success": False, "error": "Triangulum resource manager not available"}
        
        # Extract event data
        resource_type = event_data.get("resource_type", "Unknown")
        resource_id = event_data.get("resource_id", "Unknown")
        allocation = event_data.get("allocation", {})
        
        # Process allocation
        if resource_type == "memory":
            logger.info(f"Memory allocation for {resource_id}: {allocation.get('size_mb', 0)} MB")
        elif resource_type == "cpu":
            logger.info(f"CPU allocation for {resource_id}: {allocation.get('cores', 0)} cores")
        elif resource_type == "gpu":
            logger.info(f"GPU allocation for {resource_id}: {allocation.get('memory_mb', 0)} MB")
        else:
            logger.info(f"Resource allocation for {resource_type}/{resource_id}")
        
        # Record allocation
        resource_manager.record_allocation(resource_type, resource_id, allocation)
        
        return {
            "success": True,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "allocation": allocation
        }
    except Exception as e:
        logger.error(f"Error handling resource allocation event: {e}")
        return {"success": False, "error": str(e)}
