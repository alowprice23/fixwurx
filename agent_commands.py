#!/usr/bin/env python3
"""
agent_commands.py
────────────────
Command handlers for agent system integration with the shell environment.

This module provides command handlers for interacting with the agent system,
including the Planner Agent (root) and its specialized agents (Observer, Analyst, Verifier).

Key features:
- Agent initialization and management
- Bug tracking and solution path generation
- Family tree management and visualization
- Metrics reporting and analysis
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import agent modules
try:
    # Try to import the real modules
    from agents.core.planner_agent import PlannerAgent, BugState
    from agent_memory import AgentMemory
    from data_structures import PlannerPath, FamilyTree
    from meta_agent import MetaAgent
    import specialized_agents
except ImportError:
    # Fall back to mock classes for testing
    from mock_agent_classes import PlannerAgent, AgentMemory, BugState, PlannerPath, FamilyTree, MetaAgent
    import mock_agent_classes as specialized_agents
    logging.warning("Using mock agent classes for testing")

# Configure logging
logger = logging.getLogger("AgentCommands")

# Global variables to maintain state across command calls
_planner: Optional[PlannerAgent] = None
_memory: Optional[AgentMemory] = None
_meta: Optional[MetaAgent] = None
_config: Dict[str, Any] = {}
_bugs: Dict[str, BugState] = {}
_initialized = False

def initialize(system_config: Dict[str, Any]) -> bool:
    """
    Initialize the agent system.
    
    Args:
        system_config: System configuration dictionary
        
    Returns:
        True if initialization was successful, False otherwise
    """
    global _planner, _memory, _meta, _config, _initialized
    
    try:
        # Create directory structure if it doesn't exist
        agent_dir = Path(".triangulum")
        agent_dir.mkdir(exist_ok=True)
        
        # Set up paths
        family_tree_path = agent_dir / "family_tree.json"
        memory_path = agent_dir / "agent_memory.json"
        kv_path = agent_dir / "kv_store.json"
        compressed_path = agent_dir / "compressed_store.json"
        meta_storage_path = agent_dir / "meta"
        
        # Update config with paths
        agent_config = system_config.get("agent_system", {})
        agent_config.update({
            "planner": {
                "enabled": True,
                "family-tree-path": str(family_tree_path),
                "solutions-per-bug": agent_config.get("solutions-per-bug", 3),
                "max-path-depth": agent_config.get("max-path-depth", 5),
                "fallback-threshold": agent_config.get("fallback-threshold", 0.3),
                "learning-rate": agent_config.get("learning-rate", 0.1),
                "pattern-threshold": agent_config.get("pattern-threshold", 0.7),
                "history-limit": agent_config.get("history-limit", 1000)
            },
            "meta": {
                "enabled": True,
                "meta_storage_path": str(meta_storage_path),
                "coordination_threshold": agent_config.get("coordination_threshold", 0.7),
                "oversight_interval": agent_config.get("oversight_interval", 5),
                "conflict_detection_sensitivity": agent_config.get("conflict_detection_sensitivity", 0.5),
                "max_activity_history": agent_config.get("max_activity_history", 100)
            }
        })
        
        # Initialize agent memory
        _memory = AgentMemory(
            mem_path=memory_path,
            kv_path=kv_path,
            compressed_path=compressed_path,
            family_tree_path=family_tree_path
        )
        
        # Initialize planner agent
        _planner = PlannerAgent(agent_config, _memory)
        
        # Initialize meta agent
        _meta = MetaAgent(agent_config.get("meta", {}))
        
        # Store config
        _config = agent_config
        
        # Register specialized agents if they don't exist in the family tree
        tree = _planner.get_family_relationships()
        relationships = tree.get("relationships", {})
        
        if "observer" not in relationships:
            _planner.register_agent("observer", "observer")
            # Register with meta agent
            _meta.register_agent("observer", "observer")
        
        if "analyst" not in relationships:
            _planner.register_agent("analyst", "analyst")
            # Register with meta agent
            _meta.register_agent("analyst", "analyst")
        
        if "verifier" not in relationships:
            _planner.register_agent("verifier", "verifier")
            # Register with meta agent
            _meta.register_agent("verifier", "verifier")
        
        # Register planner agent with meta agent
        _meta.register_agent("planner", "planner", {"role": "root"})
        
        # Start meta agent oversight
        _meta.start_oversight()
        
        _initialized = True
        logger.info("Agent system initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing agent system: {e}")
        return False

def register() -> Dict[str, Any]:
    """
    Register command handlers with the shell environment.
    
    Returns:
        Dictionary of command handlers
    """
    return {
        "agent": agent_command,
        "plan": plan_command,
        "observe": observe_command,
        "analyze": analyze_command,
        "verify": verify_command,
        "bug": bug_command,
        "meta": meta_command
    }

def agent_command(args: str) -> int:
    """
    Main entry point for agent system commands.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized:
        print("Error: Agent system not initialized. Run 'agent init' first.")
        return 1
    
    # Parse arguments
    parts = args.strip().split()
    if not parts:
        print_agent_help()
        return 0
    
    cmd = parts[0].lower()
    cmd_args = ' '.join(parts[1:])
    
    # Execute appropriate command
    if cmd == "init":
        # Already initialized
        print("Agent system already initialized.")
        return 0
    elif cmd == "status":
        return agent_status_command(cmd_args)
    elif cmd == "tree":
        return agent_tree_command(cmd_args)
    elif cmd == "metrics":
        return agent_metrics_command(cmd_args)
    elif cmd == "network":
        return agent_network_command(cmd_args)
    elif cmd == "help":
        print_agent_help()
        return 0
    else:
        print(f"Unknown agent command: {cmd}")
        print_agent_help()
        return 1

def agent_status_command(args: str) -> int:
    """
    Display status of the agent system.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _planner:
        print("Error: Agent system not initialized.")
        return 1
    
    metrics = _planner.get_metrics()
    
    print("\nAgent System Status:")
    print("=" * 50)
    print(f"Enabled: {metrics['enabled']}")
    print(f"Active Bugs: {metrics['active_bugs']}")
    print(f"Active Paths: {metrics['active_paths']}")
    print(f"Family Tree Size: {metrics['family_tree_size']}")
    print(f"Paths Generated: {metrics['paths_generated']}")
    print(f"Successful Fixes: {metrics['successful_fixes']}")
    print(f"Failed Fixes: {metrics['failed_fixes']}")
    
    # List active bugs
    if _bugs:
        print("\nActive Bugs:")
        for bug_id, bug in _bugs.items():
            print(f"  - {bug_id}: {bug.title or 'Untitled'} ({bug.metadata.get('status', 'unknown')})")
    
    return 0

def agent_tree_command(args: str) -> int:
    """
    Display the agent family tree.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _planner:
        print("Error: Agent system not initialized.")
        return 1
    
    tree = _planner.get_family_relationships()
    relationships = tree.get("relationships", {})
    
    print("\nAgent Family Tree:")
    print("=" * 50)
    
    def print_agent(agent_id: str, level: int = 0):
        """Print agent details with indentation."""
        agent_data = relationships.get(agent_id, {})
        agent_type = agent_data.get("metadata", {}).get("type", "unknown")
        print(f"{'  ' * level}- {agent_id} ({agent_type})")
        
        for child_id in agent_data.get("children", []):
            print_agent(child_id, level + 1)
    
    # Start with planner (root)
    if "planner" in relationships:
        print_agent("planner")
    
    return 0

def agent_metrics_command(args: str) -> int:
    """
    Display detailed metrics for the agent system.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _planner:
        print("Error: Agent system not initialized.")
        return 1
    
    metrics = _planner.get_metrics()
    
    print("\nAgent System Metrics:")
    print("=" * 50)
    
    # Print all metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Print neural weights if verbose
    if args.lower() == "verbose":
        print("\nNeural Weights:")
        for agent_type, weight in _planner.neural_weights.items():
            print(f"  {agent_type}: {weight:.2f}")
    
    return 0

def plan_command(args: str) -> int:
    """
    Execute planner agent commands.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized or not _planner:
        print("Error: Agent system not initialized. Run 'agent init' first.")
        return 1
    
    # Parse arguments
    parts = args.strip().split()
    if not parts:
        print_plan_help()
        return 0
    
    cmd = parts[0].lower()
    cmd_args = ' '.join(parts[1:])
    
    # Execute appropriate command
    if cmd == "generate":
        return plan_generate_command(cmd_args)
    elif cmd == "select":
        return plan_select_command(cmd_args)
    elif cmd == "help":
        print_plan_help()
        return 0
    else:
        print(f"Unknown plan command: {cmd}")
        print_plan_help()
        return 1

def plan_generate_command(args: str) -> int:
    """
    Generate solution paths for a bug.
    
    Args:
        args: Command arguments (bug_id)
        
    Returns:
        Exit code
    """
    if not args:
        print("Error: Bug ID required.")
        return 1
    
    bug_id = args.strip()
    
    if bug_id not in _bugs:
        print(f"Error: Bug {bug_id} not found.")
        return 1
    
    bug = _bugs[bug_id]
    
    # Generate paths
    paths = _planner.generate_solution_paths(bug)
    
    print(f"\nGenerated {len(paths)} solution paths for bug {bug_id}:")
    print("=" * 50)
    
    for i, path in enumerate(paths):
        print(f"\nPath {i+1} ({path.path_id}):")
        print(f"  Priority: {path.metadata.get('priority', 'unknown'):.2f}")
        print(f"  Entropy: {path.metadata.get('entropy', 'unknown'):.2f}")
        print(f"  Estimated Time: {path.metadata.get('estimated_time', 'unknown')} ticks")
        
        print("\n  Actions:")
        for j, action in enumerate(path.actions):
            print(f"    {j+1}. {action.get('type')} - {action.get('description')}")
        
        if path.fallbacks:
            print("\n  Fallbacks:")
            for j, fallback in enumerate(path.fallbacks):
                print(f"    {j+1}. {fallback.get('type')} - {fallback.get('description')}")
    
    return 0

def plan_select_command(args: str) -> int:
    """
    Select the best solution path for a bug.
    
    Args:
        args: Command arguments (bug_id)
        
    Returns:
        Exit code
    """
    if not args:
        print("Error: Bug ID required.")
        return 1
    
    bug_id = args.strip()
    
    if bug_id not in _bugs:
        print(f"Error: Bug {bug_id} not found.")
        return 1
    
    # Select best path
    best_path = _planner.select_best_path(bug_id)
    
    if not best_path:
        print(f"Error: No paths available for bug {bug_id}.")
        return 1
    
    print(f"\nSelected best path for bug {bug_id}:")
    print("=" * 50)
    print(f"Path ID: {best_path.path_id}")
    print(f"Priority: {best_path.metadata.get('priority', 'unknown'):.2f}")
    print(f"Entropy: {best_path.metadata.get('entropy', 'unknown'):.2f}")
    print(f"Estimated Time: {best_path.metadata.get('estimated_time', 'unknown')} ticks")
    
    print("\nActions:")
    for i, action in enumerate(best_path.actions):
        print(f"  {i+1}. {action.get('type')} - {action.get('description')}")
    
    return 0

def observe_command(args: str) -> int:
    """
    Execute observer agent commands.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized:
        print("Error: Agent system not initialized. Run 'agent init' first.")
        return 1
    
    # Create observer agent
    observer = specialized_agents.ObserverAgent()
    
    # Parse arguments
    parts = args.strip().split()
    if not parts:
        print("Error: Observer command requires arguments.")
        return 1
    
    cmd = parts[0].lower()
    cmd_args = ' '.join(parts[1:])
    
    # Execute appropriate command
    if cmd == "analyze":
        # Check if bug ID is provided
        if not cmd_args:
            print("Error: Bug ID required for analysis.")
            return 1
        
        bug_id = cmd_args.strip()
        
        # Get or create bug
        if bug_id in _bugs:
            bug = _bugs[bug_id]
        else:
            bug = BugState(
                bug_id=bug_id,
                title=f"Bug {bug_id}",
                description="Created by observer analyze command"
            )
            # Add metadata manually
            bug.metadata = {
                "created_at": time.time(),
                "created_by": "observer_command",
                "status": "new"
            }
            _bugs[bug_id] = bug
        
        print(f"\nAnalyzing bug {bug_id}...")
        print("=" * 50)
        print("Observer agent activated.")
        print("This would trigger the observer agent to analyze the bug in a real implementation.")
        print("For now, this is a placeholder for the observer functionality.")
        
        # Update bug status
        bug.metadata["status"] = "analyzed"
        
        # Add phase to the bug - handle both BugState implementations
        try:
            # Try to use the add_phase method if it exists
            bug.add_phase("ANALYZE", {
                "agent": "observer",
                "timestamp": time.time(),
                "result": "Analyzed by observer agent command"
            })
        except AttributeError:
            # Fall back to manually updating phase_history if add_phase doesn't exist
            if not hasattr(bug, "phase_history"):
                bug.phase_history = []
                
            bug.phase_history.append({
                "name": "ANALYZE",
                "timestamp": time.time(),
                "details": {
                    "agent": "observer",
                    "result": "Analyzed by observer agent command"
                }
            })
            
            # Also update in metadata for compatibility
            if "phase_history" not in bug.metadata:
                bug.metadata["phase_history"] = []
                
            bug.metadata["phase_history"] = bug.phase_history
        
        print(f"\nBug {bug_id} analyzed successfully.")
        return 0
    else:
        print(f"Unknown observer command: {cmd}")
        return 1

def analyze_command(args: str) -> int:
    """
    Execute analyst agent commands.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized:
        print("Error: Agent system not initialized. Run 'agent init' first.")
        return 1
    
    # Create analyst agent
    analyst = specialized_agents.AnalystAgent()
    
    # Parse arguments
    parts = args.strip().split()
    if not parts:
        print("Error: Analyst command requires arguments.")
        return 1
    
    cmd = parts[0].lower()
    cmd_args = ' '.join(parts[1:])
    
    # Execute appropriate command
    if cmd == "patch":
        # Check if bug ID is provided
        if not cmd_args:
            print("Error: Bug ID required for patch generation.")
            return 1
        
        bug_id = cmd_args.strip()
        
        # Check if bug exists
        if bug_id not in _bugs:
            print(f"Error: Bug {bug_id} not found.")
            return 1
        
        bug = _bugs[bug_id]
        
        print(f"\nGenerating patch for bug {bug_id}...")
        print("=" * 50)
        print("Analyst agent activated.")
        print("This would trigger the analyst agent to generate a patch in a real implementation.")
        print("For now, this is a placeholder for the analyst functionality.")
        
        # Update bug status
        bug.metadata["status"] = "patched"
        
        # Add phase to the bug - handle both BugState implementations
        try:
            # Try to use the add_phase method if it exists
            bug.add_phase("PATCH", {
                "agent": "analyst",
                "timestamp": time.time(),
                "result": "Patched by analyst agent command"
            })
        except AttributeError:
            # Fall back to manually updating phase_history if add_phase doesn't exist
            if not hasattr(bug, "phase_history"):
                bug.phase_history = []
                
            bug.phase_history.append({
                "name": "PATCH",
                "timestamp": time.time(),
                "details": {
                    "agent": "analyst",
                    "result": "Patched by analyst agent command"
                }
            })
            
            # Also update in metadata for compatibility
            if "phase_history" not in bug.metadata:
                bug.metadata["phase_history"] = []
                
            bug.metadata["phase_history"] = bug.phase_history
        
        print(f"\nPatch generated for bug {bug_id} successfully.")
        return 0
    else:
        print(f"Unknown analyst command: {cmd}")
        return 1

def verify_command(args: str) -> int:
    """
    Execute verifier agent commands.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized:
        print("Error: Agent system not initialized. Run 'agent init' first.")
        return 1
    
    # Create verifier agent
    verifier = specialized_agents.VerifierAgent()
    
    # Parse arguments
    parts = args.strip().split()
    if not parts:
        print("Error: Verifier command requires arguments.")
        return 1
    
    cmd = parts[0].lower()
    cmd_args = ' '.join(parts[1:])
    
    # Execute appropriate command
    if cmd == "test":
        # Check if bug ID is provided
        if not cmd_args:
            print("Error: Bug ID required for verification.")
            return 1
        
        bug_id = cmd_args.strip()
        
        # Check if bug exists
        if bug_id not in _bugs:
            print(f"Error: Bug {bug_id} not found.")
            return 1
        
        bug = _bugs[bug_id]
        
        print(f"\nVerifying fix for bug {bug_id}...")
        print("=" * 50)
        print("Verifier agent activated.")
        print("This would trigger the verifier agent to test the fix in a real implementation.")
        print("For now, this is a placeholder for the verifier functionality.")
        
        # Update bug status
        bug.metadata["status"] = "verified"
        
        # Add phase to the bug - handle both BugState implementations
        try:
            # Try to use the add_phase method if it exists
            bug.add_phase("VERIFY", {
                "agent": "verifier",
                "timestamp": time.time(),
                "result": "Verified by verifier agent command"
            })
        except AttributeError:
            # Fall back to manually updating phase_history if add_phase doesn't exist
            if not hasattr(bug, "phase_history"):
                bug.phase_history = []
                
            bug.phase_history.append({
                "name": "VERIFY",
                "timestamp": time.time(),
                "details": {
                    "agent": "verifier",
                    "result": "Verified by verifier agent command"
                }
            })
            
            # Also update in metadata for compatibility
            if "phase_history" not in bug.metadata:
                bug.metadata["phase_history"] = []
                
            bug.metadata["phase_history"] = bug.phase_history
        
        print(f"\nFix for bug {bug_id} verified successfully.")
        return 0
    else:
        print(f"Unknown verifier command: {cmd}")
        return 1

def bug_command(args: str) -> int:
    """
    Manage bugs in the system.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized:
        print("Error: Agent system not initialized. Run 'agent init' first.")
        return 1
    
    # Parse arguments
    parts = args.strip().split()
    if not parts:
        print_bug_help()
        return 0
    
    cmd = parts[0].lower()
    cmd_args = ' '.join(parts[1:])
    
    # Execute appropriate command
    if cmd == "create":
        return bug_create_command(cmd_args)
    elif cmd == "list":
        return bug_list_command(cmd_args)
    elif cmd == "show":
        return bug_show_command(cmd_args)
    elif cmd == "update":
        return bug_update_command(cmd_args)
    elif cmd == "help":
        print_bug_help()
        return 0
    else:
        print(f"Unknown bug command: {cmd}")
        print_bug_help()
        return 1

def bug_create_command(args: str) -> int:
    """
    Create a new bug.
    
    Args:
        args: Command arguments (bug_id [title])
        
    Returns:
        Exit code
    """
    # Parse arguments
    parts = args.strip().split(maxsplit=1)
    if not parts:
        print("Error: Bug ID required.")
        return 1
    
    bug_id = parts[0]
    title = parts[1] if len(parts) > 1 else f"Bug {bug_id}"
    
    # Check if bug already exists
    if bug_id in _bugs:
        print(f"Error: Bug {bug_id} already exists.")
        return 1
    
    # Create bug
    bug = BugState(
        bug_id=bug_id,
        title=title,
        severity="medium",
        description="Created by bug create command"
    )
    
    # Add metadata manually since it's not in the constructor
    bug.metadata = {
        "created_at": time.time(),
        "created_by": "bug_command",
        "status": "new"
    }
    
    # Add to bugs dictionary
    _bugs[bug_id] = bug
    
    print(f"Bug {bug_id} created: {title}")
    return 0

def bug_list_command(args: str) -> int:
    """
    List all bugs.
    
    Args:
        args: Command arguments (optional filter)
        
    Returns:
        Exit code
    """
    if not _bugs:
        print("No bugs found.")
        return 0
    
    # Parse filter
    status_filter = None
    if args:
        status_filter = args.strip().lower()
    
    print("\nBugs:")
    print("=" * 50)
    print(f"{'ID':<15} {'Status':<12} {'Title'}")
    print("-" * 50)
    
    for bug_id, bug in _bugs.items():
        bug_status = bug.metadata.get('status', 'unknown')
        if status_filter and bug_status.lower() != status_filter:
            continue
        
        print(f"{bug_id:<15} {bug_status:<12} {bug.title or 'Untitled'}")
    
    return 0

def bug_show_command(args: str) -> int:
    """
    Show details of a specific bug.
    
    Args:
        args: Command arguments (bug_id)
        
    Returns:
        Exit code
    """
    if not args:
        print("Error: Bug ID required.")
        return 1
    
    bug_id = args.strip()
    
    if bug_id not in _bugs:
        print(f"Error: Bug {bug_id} not found.")
        return 1
    
    bug = _bugs[bug_id]
    
    print(f"\nBug {bug_id}:")
    print("=" * 50)
    print(f"Title: {bug.title or 'Untitled'}")
    print(f"Status: {bug.metadata.get('status', 'unknown')}")
    print(f"Severity: {bug.severity}")
    
    if bug.description:
        print(f"\nDescription:")
        print(bug.description)
    
    if bug.tags:
        print(f"\nTags: {', '.join(bug.tags)}")
    
    if bug.phase_history:
        print("\nPhase History:")
        for i, phase in enumerate(bug.phase_history):
            print(f"  {i+1}. {phase.get('name')} at {time.ctime(phase.get('timestamp', 0))}")
    
    if bug.planner_solutions:
        print("\nPlanner Solutions:")
        for i, solution in enumerate(bug.planner_solutions):
            path_id = solution.get("path_id", "unknown")
            timestamp = solution.get("timestamp", 0)
            print(f"  {i+1}. {path_id} at {time.ctime(timestamp)}")
    
    return 0

def bug_update_command(args: str) -> int:
    """
    Update a bug's properties.
    
    Args:
        args: Command arguments (bug_id property=value)
        
    Returns:
        Exit code
    """
    parts = args.strip().split(maxsplit=1)
    if len(parts) < 2:
        print("Error: Bug ID and property=value required.")
        return 1
    
    bug_id = parts[0]
    prop_val = parts[1]
    
    if bug_id not in _bugs:
        print(f"Error: Bug {bug_id} not found.")
        return 1
    
    # Parse property=value
    if "=" not in prop_val:
        print("Error: Property must be in format property=value.")
        return 1
    
    prop, value = prop_val.split("=", 1)
    prop = prop.strip().lower()
    value = value.strip()
    
    bug = _bugs[bug_id]
    
    # Update property
    if prop == "title":
        bug.title = value
    elif prop == "description":
        bug.description = value
    elif prop == "severity":
        if value not in ["critical", "high", "medium", "low"]:
            print(f"Error: Invalid severity: {value}")
            print("Valid values: critical, high, medium, low")
            return 1
        bug.severity = value
    elif prop == "status":
        bug.metadata["status"] = value
    elif prop == "tag":
        bug.add_tag(value)
    else:
        print(f"Error: Unknown property: {prop}")
        print("Valid properties: title, description, severity, status, tag")
        return 1
    
    print(f"Bug {bug_id} updated: {prop}={value}")
    return 0

# Help text functions
def meta_command(args: str) -> int:
    """
    Execute Meta Agent commands.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized or not _meta:
        print("Error: Agent system not initialized. Run 'agent init' first.")
        return 1
    
    # Parse arguments
    parts = args.strip().split()
    if not parts:
        print_meta_help()
        return 0
    
    cmd = parts[0].lower()
    cmd_args = ' '.join(parts[1:])
    
    # Execute appropriate command
    if cmd == "status":
        return meta_status_command(cmd_args)
    elif cmd == "agents":
        return meta_agents_command(cmd_args)
    elif cmd == "network":
        return meta_network_command(cmd_args)
    elif cmd == "conflicts":
        return meta_conflicts_command(cmd_args)
    elif cmd == "insights":
        return meta_insights_command(cmd_args)
    elif cmd == "coordinate":
        return meta_coordinate_command(cmd_args)
    elif cmd == "help":
        print_meta_help()
        return 0
    else:
        print(f"Unknown meta command: {cmd}")
        print_meta_help()
        return 1

def agent_network_command(args: str) -> int:
    """
    Display agent network visualization.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized or not _meta:
        print("Error: Agent system not initialized.")
        return 1
    
    print("\nAgent Network Visualization:")
    print("=" * 50)
    
    # Get agent network from meta agent
    network = _meta.get_agent_network()
    
    # Display network information
    print(f"Network status as of {time.ctime(network['timestamp'])}")
    print(f"Nodes: {len(network['nodes'])}")
    print(f"Edges: {len(network['edges'])}")
    
    # Display nodes
    print("\nAgents:")
    for node in network['nodes']:
        print(f"  - {node['id']} ({node['type']}) - {node['status']}")
    
    # Display edges
    if network['edges']:
        print("\nConnections:")
        for edge in network['edges']:
            print(f"  - {edge['source']} <--> {edge['target']} (strength: {edge['strength']:.2f})")
    
    # Generate visualization file
    if args.lower() == "visualize":
        output_path = _meta.visualize_agent_network()
        print(f"\nVisualization saved to: {output_path}")
    
    return 0

def meta_status_command(args: str) -> int:
    """
    Display Meta Agent status.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized or not _meta:
        print("Error: Agent system not initialized.")
        return 1
    
    print("\nMeta Agent Status:")
    print("=" * 50)
    
    # Get metrics
    metrics = _meta.get_metrics()
    
    # Display metrics
    print(f"Enabled: {_meta.enabled}")
    print(f"Agent Count: {metrics['agent_count']}")
    print(f"Coordination Events: {metrics['coordination_events']}")
    print(f"Conflict Resolutions: {metrics['conflict_resolutions']}")
    print(f"Optimizations: {metrics['optimizations']}")
    print(f"Oversight Cycles: {metrics['oversight_cycles']}")
    print(f"Meta Insights Generated: {metrics['meta_insights_generated']}")
    
    return 0

def meta_agents_command(args: str) -> int:
    """
    Display agents registered with the Meta Agent.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized or not _meta:
        print("Error: Agent system not initialized.")
        return 1
    
    print("\nRegistered Agents:")
    print("=" * 50)
    
    # Display agents
    for agent_id, state in _meta.agent_states.items():
        print(f"Agent: {agent_id}")
        print(f"  Type: {state['type']}")
        print(f"  Status: {state['status']}")
        print(f"  Registered: {time.ctime(state['registered_at'])}")
        print(f"  Last Activity: {time.ctime(state['last_activity'])}")
        if state.get('metadata'):
            print(f"  Metadata: {state['metadata']}")
        print()
    
    if not _meta.agent_states:
        print("No agents registered")
    
    return 0

def meta_network_command(args: str) -> int:
    """
    Display agent network.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized or not _meta:
        print("Error: Agent system not initialized.")
        return 1
    
    print("\nAgent Network:")
    print("=" * 50)
    
    # Get agent network
    network = _meta.get_agent_network()
    
    # Display network nodes
    print("Nodes:")
    for node in network['nodes']:
        print(f"  - {node['id']} ({node['type']})")
    
    # Display network edges
    print("\nEdges:")
    for edge in network['edges']:
        print(f"  - {edge['source']} <--> {edge['target']} (strength: {edge['strength']:.2f})")
    
    if not network['edges']:
        print("  No significant connections")
    
    # Generate visualization
    if args.lower() == "visualize":
        output_path = _meta.visualize_agent_network()
        print(f"\nVisualization saved to: {output_path}")
    
    return 0

def meta_conflicts_command(args: str) -> int:
    """
    Display agent conflicts.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized or not _meta:
        print("Error: Agent system not initialized.")
        return 1
    
    print("\nAgent Conflicts:")
    print("=" * 50)
    
    # Display conflicts
    for conflict_id, conflict in _meta.agent_conflicts.items():
        print(f"Conflict: {conflict_id}")
        print(f"  Agents: {', '.join(conflict['agents'])}")
        print(f"  Resource: {conflict['resource']}")
        print(f"  Timestamp: {time.ctime(conflict['timestamp'])}")
        print(f"  Severity: {conflict['severity']:.2f}")
        print(f"  Resolved: {conflict['resolved']}")
        if conflict.get('resolved') and conflict.get('resolved_at'):
            print(f"  Resolved At: {time.ctime(conflict['resolved_at'])}")
            print(f"  Resolution: {conflict.get('resolution_strategy', 'unknown')}")
        print()
    
    if not _meta.agent_conflicts:
        print("No conflicts detected")
    
    return 0

def meta_insights_command(args: str) -> int:
    """
    Generate and display Meta Agent insights.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not _initialized or not _meta:
        print("Error: Agent system not initialized.")
        return 1
    
    # Generate insight if requested
    if args.lower() == "generate":
        insight = _meta.generate_insight()
        
        print("\nGenerated Insight:")
        print("=" * 50)
        print(f"ID: {insight['id']}")
        print(f"Type: {insight['type']}")
        print(f"Timestamp: {time.ctime(insight['timestamp'])}")
        print(f"Confidence: {insight['confidence']:.2f}")
        print(f"Description: {insight['description']}")
        
        print("\nDetails:")
        for key, value in insight['details'].items():
            print(f"  {key}: {value}")
        
        print("\nRecommendations:")
        for rec in insight['recommendations']:
            print(f"  - {rec['action']} (confidence: {rec['confidence']:.2f}, impact: {rec['expected_impact']:.2f})")
        
        return 0
    
    # If no argument provided, generate a new insight
    insight = _meta.generate_insight()
    
    print("\nNew Meta Agent Insight:")
    print("=" * 50)
    print(f"Type: {insight['type']}")
    print(f"Confidence: {insight['confidence']:.2f}")
    print(f"Description: {insight['description']}")
    
    print("\nRecommendations:")
    for rec in insight['recommendations']:
        print(f"  - {rec['action']} (confidence: {rec['confidence']:.2f}, impact: {rec['expected_impact']:.2f})")
    
    return 0

def meta_coordinate_command(args: str) -> int:
    """
    Coordinate activities between agents.
    
    Args:
        args: Command arguments (agent1,agent2,... task_id task_type)
        
    Returns:
        Exit code
    """
    if not _initialized or not _meta:
        print("Error: Agent system not initialized.")
        return 1
    
    # Parse arguments
    parts = args.strip().split()
    if len(parts) < 3:
        print("Error: Not enough arguments.")
        print("Usage: meta coordinate agent1,agent2,... task_id task_type")
        return 1
    
    # Get agent IDs
    agent_ids = parts[0].split(',')
    task_id = parts[1]
    task_type = parts[2]
    
    # Coordinate agents
    result = _meta.coordinate_agents(agent_ids, task_id, task_type)
    
    if not result.get('success'):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    print(f"\nCreated coordination plan for task {task_id}:")
    print("=" * 50)
    print(f"Agents: {result['agent_count']}")
    print(f"Steps: {result['step_count']}")
    print(f"\nCoordination plan saved to: .triangulum/meta/coordination_{task_id}.json")
    
    return 0

def print_agent_help():
    """Print help text for agent command."""
    print("\nAgent System Commands:")
    print("  agent init               - Initialize the agent system")
    print("  agent status             - Show agent system status")
    print("  agent tree               - Show agent family tree")
    print("  agent network [visualize]- Show agent network")
    print("  agent metrics [verbose]  - Show agent system metrics")
    print("  agent help               - Show this help text")

def print_meta_help():
    """Print help text for meta command."""
    print("\nMeta Agent Commands:")
    print("  meta status              - Show meta agent status")
    print("  meta agents              - List registered agents")
    print("  meta network [visualize] - Show agent network")
    print("  meta conflicts           - Show agent conflicts")
    print("  meta insights [generate] - Generate meta insights")
    print("  meta coordinate <agents> <task_id> <task_type> - Coordinate agents")
    print("  meta help                - Show this help text")

def print_plan_help():
    """Print help text for plan command."""
    print("\nPlanner Agent Commands:")
    print("  plan generate <bug_id>   - Generate solution paths for a bug")
    print("  plan select <bug_id>     - Select the best solution path for a bug")
    print("  plan help                - Show this help text")

def print_bug_help():
    """Print help text for bug command."""
    print("\nBug Management Commands:")
    print("  bug create <bug_id> [title]     - Create a new bug")
    print("  bug list [status]               - List all bugs, optionally filtered by status")
    print("  bug show <bug_id>               - Show details of a specific bug")
    print("  bug update <bug_id> prop=value  - Update a bug's properties")
    print("  bug help                        - Show this help text")
