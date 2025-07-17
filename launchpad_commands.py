#!/usr/bin/env python3
"""
Launchpad Commands

This module provides command handlers for the Launchpad Agent, which serves
as the bootstrap and entry point for the entire FixWurx agent system.
"""

import os
import sys
import json
import shlex
import logging
import argparse
from typing import Dict, List, Any, Optional

logger = logging.getLogger("LaunchpadCommands")

def register_launchpad_commands(registry):
    """
    Register Launchpad command handlers with the component registry.
    
    Args:
        registry: Component registry instance
    """
    # Register command handlers
    registry.register_command_handler("launchpad:restart", launchpad_restart_command, "launchpad")
    registry.register_command_handler("launchpad:status", launchpad_status_command, "launchpad")
    registry.register_command_handler("launchpad:metrics", launchpad_metrics_command, "launchpad")
    
    # Register aliases
    registry.register_alias("lp:restart", "launchpad:restart")
    registry.register_alias("lp:status", "launchpad:status")
    registry.register_alias("lp:metrics", "launchpad:metrics")
    
    logger.info("Launchpad commands registered")

def launchpad_restart_command(args: str) -> int:
    """
    Restart a component through the Launchpad Agent's intelligent restart system.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Restart a component with LLM-guided optimization")
    parser.add_argument("component", nargs="?", help="Component to restart (e.g., agent_system, triangulum)")
    parser.add_argument("--force", "-f", action="store_true", help="Force restart even if the component is in use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    component = cmd_args.component
    force = cmd_args.force
    verbose = cmd_args.verbose
    
    if not component:
        print("Error: Component name required")
        print("Available components: agent_system, coordinator, handoff, auditor")
        return 1
    
    print(f"Restarting component: {component}{' (forced)' if force else ''}")
    
    try:
        # Get the LaunchpadAgent instance
        from agents.core.launchpad.agent import get_instance as get_launchpad_agent
        
        launchpad_agent = get_launchpad_agent()
        if not launchpad_agent:
            print("Error: LaunchpadAgent not available")
            return 1
        
        # Check if the agent is initialized
        if not hasattr(launchpad_agent, 'initialized') or not launchpad_agent.initialized:
            print("Error: LaunchpadAgent not initialized")
            return 1
        
        # Get LLM-guided recommendations if verbose mode is enabled
        if verbose:
            print("Requesting LLM-guided optimization recommendations...")
            recommendations = launchpad_agent._get_restart_recommendations(component)
            
            if recommendations:
                print("\nRestart Recommendations:")
                
                # Show pre-restart actions
                pre_actions = recommendations.get("pre_restart_actions", [])
                if pre_actions:
                    print("\nPre-restart actions:")
                    for i, action in enumerate(pre_actions, 1):
                        print(f"  {i}. {action}")
                
                # Show optimization suggestions
                suggestions = recommendations.get("optimization_suggestions", [])
                if suggestions:
                    print("\nOptimization suggestions:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"  {i}. {suggestion}")
        
        # Perform the restart
        print(f"Executing intelligent restart of {component}...")
        success = launchpad_agent.restart_component(component)
        
        # Show results
        if success:
            print(f"Component {component} restarted successfully")
            
            # Show post-restart actions in verbose mode
            if verbose and recommendations:
                post_actions = recommendations.get("post_restart_actions", [])
                if post_actions:
                    print("\nRecommended post-restart actions:")
                    for i, action in enumerate(post_actions, 1):
                        print(f"  {i}. {action}")
            
            return 0
        else:
            print(f"Failed to restart component {component}")
            return 1
    except ImportError as e:
        print(f"Error importing LaunchpadAgent: {e}")
        return 1
    except Exception as e:
        print(f"Error restarting component: {e}")
        return 1

def launchpad_status_command(args: str) -> int:
    """
    Get the status of the Launchpad Agent and its components.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Get the status of the Launchpad Agent")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--output", "-o", help="Output file for status information")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    output_format = cmd_args.format
    output_file = cmd_args.output
    
    try:
        # Get the LaunchpadAgent instance
        from agents.core.launchpad.agent import get_instance as get_launchpad_agent
        
        launchpad_agent = get_launchpad_agent()
        if not launchpad_agent:
            print("Error: LaunchpadAgent not available")
            return 1
        
        # Get status
        status = launchpad_agent.get_status()
        
        # Display status
        if output_format == "json":
            status_json = json.dumps(status, indent=2, default=str)
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(status_json)
                print(f"Status saved to {output_file}")
            else:
                print(status_json)
        else:  # text format
            print("\nLaunchpad Agent Status:")
            print("-" * 60)
            
            # Display initialization status
            initialized = status.get("initialized", False)
            print(f"Initialized: {initialized}")
            
            # Display uptime
            uptime = status.get("uptime", 0)
            uptime_str = f"{uptime:.2f} seconds"
            if uptime > 60:
                uptime_str = f"{uptime / 60:.2f} minutes"
            if uptime > 3600:
                uptime_str = f"{uptime / 3600:.2f} hours"
            print(f"Uptime: {uptime_str}")
            
            # Display LLM status
            llm_status = status.get("llm_status", "inactive")
            print(f"LLM Status: {llm_status}")
            
            # Display agent system status if available
            agent_system = status.get("agent_system")
            if agent_system:
                print("\nAgent System:")
                for key, value in agent_system.items():
                    if isinstance(value, dict):
                        continue  # Skip complex objects
                    print(f"  {key}: {value}")
            
            # Display LLM metrics if available
            llm_metrics = status.get("llm_metrics")
            if llm_metrics:
                print("\nLLM Metrics:")
                for key, value in llm_metrics.items():
                    print(f"  {key}: {value}")
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    f.write("Launchpad Agent Status\n")
                    f.write("-" * 60 + "\n\n")
                    
                    f.write(f"Initialized: {initialized}\n")
                    f.write(f"Uptime: {uptime_str}\n")
                    f.write(f"LLM Status: {llm_status}\n")
                    
                    if agent_system:
                        f.write("\nAgent System:\n")
                        for key, value in agent_system.items():
                            if isinstance(value, dict):
                                continue  # Skip complex objects
                            f.write(f"  {key}: {value}\n")
                    
                    if llm_metrics:
                        f.write("\nLLM Metrics:\n")
                        for key, value in llm_metrics.items():
                            f.write(f"  {key}: {value}\n")
                
                print(f"Status saved to {output_file}")
        
        return 0
    except ImportError as e:
        print(f"Error importing LaunchpadAgent: {e}")
        return 1
    except Exception as e:
        print(f"Error getting Launchpad Agent status: {e}")
        return 1

def launchpad_metrics_command(args: str) -> int:
    """
    Get the metrics of the Launchpad Agent's LLM usage.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Get the metrics of the Launchpad Agent's LLM usage")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--output", "-o", help="Output file for metrics information")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    output_format = cmd_args.format
    output_file = cmd_args.output
    
    try:
        # Get the LaunchpadAgent instance
        from agents.core.launchpad.agent import get_instance as get_launchpad_agent
        
        launchpad_agent = get_launchpad_agent()
        if not launchpad_agent:
            print("Error: LaunchpadAgent not available")
            return 1
        
        # Get status for metrics
        status = launchpad_agent.get_status()
        
        # Extract LLM metrics
        llm_metrics = status.get("llm_metrics", {})
        
        # Display metrics
        if output_format == "json":
            metrics_json = json.dumps(llm_metrics, indent=2)
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(metrics_json)
                print(f"Metrics saved to {output_file}")
            else:
                print(metrics_json)
        else:  # text format
            print("\nLaunchpad Agent LLM Metrics:")
            print("-" * 60)
            
            if not llm_metrics:
                print("No LLM metrics available")
            else:
                for key, value in llm_metrics.items():
                    print(f"{key}: {value}")
            
            # Calculate success rate if possible
            total_calls = llm_metrics.get("llm_calls", 0)
            successful_calls = llm_metrics.get("llm_successful_calls", 0)
            
            if total_calls > 0:
                success_rate = (successful_calls / total_calls) * 100
                print(f"\nLLM Success Rate: {success_rate:.2f}%")
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    f.write("Launchpad Agent LLM Metrics\n")
                    f.write("-" * 60 + "\n\n")
                    
                    if not llm_metrics:
                        f.write("No LLM metrics available\n")
                    else:
                        for key, value in llm_metrics.items():
                            f.write(f"{key}: {value}\n")
                    
                    if total_calls > 0:
                        f.write(f"\nLLM Success Rate: {success_rate:.2f}%\n")
                
                print(f"Metrics saved to {output_file}")
        
        return 0
    except ImportError as e:
        print(f"Error importing LaunchpadAgent: {e}")
        return 1
    except Exception as e:
        print(f"Error getting Launchpad Agent metrics: {e}")
        return 1
