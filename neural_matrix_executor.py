#!/usr/bin/env python3
"""
neural_matrix_executor.py
─────────────────────────
Demonstration script for neural matrix functionality.

This script serves as a bridge between all neural matrix components,
demonstrating how the neural connections work in practice. It can be
used to:

1. Initialize the neural matrix
2. Execute neural pattern matching 
3. Demonstrate agent coordination
4. Validate neural matrix integrity
5. Show learning capabilities

Usage:
    python neural_matrix_executor.py [command] [options]

Commands:
    init            Initialize neural matrix
    validate        Validate neural matrix integrity
    patterns        List or search neural patterns
    execute         Execute a neural-guided solution path
    learn           Demonstrate neural learning
    stats           Show neural matrix statistics
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import neural matrix components
try:
    import neural_matrix_init
    import triangulation_engine
    import hub
    import verification_engine
    import planner_agent
    import scope_filter
    import specialized_agents
    from agent_memory import AgentMemory
    from data_structures import BugState
except ImportError as e:
    print(f"Error importing neural matrix components: {e}")
    print("Make sure you're running this script from the FixWurx root directory.")
    sys.exit(1)


class NeuralMatrixExecutor:
    """
    Neural Matrix Executor for demonstrating and testing neural matrix functionality.
    """
    
    def __init__(self):
        """Initialize the neural matrix executor."""
        self.neural_matrix_dir = Path(".triangulum/neural_matrix")
        self.engine = None
        self.agent_memory = None
        self.verifier = None
        self.scope_filter = None
        
        # Check if neural matrix directory exists
        if not self.neural_matrix_dir.exists():
            print("Neural matrix not initialized. Run 'python neural_matrix_executor.py init' first.")
        
    def initialize(self, force: bool = False) -> None:
        """
        Initialize the neural matrix.
        
        Args:
            force: Whether to force reinitialization
        """
        print("Initializing neural matrix...")
        
        # Call the neural_matrix_init script
        neural_matrix_init.create_directory_structure(self.neural_matrix_dir, force)
        neural_matrix_init.initialize_neural_weights(self.neural_matrix_dir, force)
        neural_matrix_init.initialize_pattern_database(self.neural_matrix_dir, force)
        neural_matrix_init.initialize_family_tree_connections(self.neural_matrix_dir, force)
        neural_matrix_init.create_test_data(self.neural_matrix_dir, force)
        neural_matrix_init.create_readme(self.neural_matrix_dir)
        
        # Initialize database tables
        db_path = Path(".triangulum/reviews.sqlite")
        neural_matrix_init.initialize_database_tables(db_path, force)
        
        print("Neural matrix initialization complete!")
        print(f"Base directory: {self.neural_matrix_dir.resolve()}")
        print(f"Database: {db_path.resolve()}")
        
    def validate(self, detailed: bool = False) -> None:
        """
        Validate the neural matrix integrity.
        
        Args:
            detailed: Whether to show detailed validation results
        """
        print("Validating neural matrix integrity...")
        
        # Access the verification engine through its global verifier instance
        verifier = verification_engine._verifier
        
        # Run neural validation
        results = verifier.neural_validate(detailed=detailed)
        
        # Display results
        print(f"Validation result: {'PASSED' if results['valid'] else 'FAILED'}")
        print(f"Connections checked: {results['metrics']['connections_checked']}")
        print(f"Weights checked: {results['metrics']['weights_checked']}")
        print(f"Patterns checked: {results['metrics']['patterns_checked']}")
        
        # Show errors and warnings
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  - {error}")
                
        if results['warnings']:
            print("\nWarnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")
                
        # Show details if requested
        if detailed and results['details']:
            print("\nDetails:")
            print(json.dumps(results['details'], indent=2))
    
    def list_patterns(self, search_term: Optional[str] = None) -> None:
        """
        List or search neural patterns.
        
        Args:
            search_term: Optional search term to filter patterns
        """
        print("Neural patterns:")
        
        patterns_file = self.neural_matrix_dir / "patterns" / "starter_patterns.json"
        if not patterns_file.exists():
            print("No patterns found. Initialize neural matrix first.")
            return
            
        try:
            with open(patterns_file, 'r') as f:
                patterns = json.load(f)
                
            # Filter patterns if search term provided
            if search_term:
                filtered_patterns = []
                for pattern in patterns:
                    # Search in pattern_id, bug_type, and tags
                    if (search_term.lower() in pattern.get("pattern_id", "").lower() or
                        search_term.lower() in pattern.get("bug_type", "").lower() or
                        any(search_term.lower() in tag.lower() for tag in pattern.get("tags", []))):
                        filtered_patterns.append(pattern)
                patterns = filtered_patterns
                
            if not patterns:
                print(f"No patterns found matching '{search_term}'")
                return
                
            for i, pattern in enumerate(patterns, 1):
                print(f"\nPattern {i}: {pattern.get('pattern_id')}")
                print(f"  Bug Type: {pattern.get('bug_type')}")
                print(f"  Tags: {', '.join(pattern.get('tags', []))}")
                print(f"  Success Rate: {pattern.get('success_rate', 0.0):.2f}")
                print(f"  Sample Count: {pattern.get('sample_count', 0)}")
                
                # Show features
                features = pattern.get("features", [])
                if features:
                    print("  Features:")
                    for feature in features:
                        if isinstance(feature, dict):
                            name = feature.get("name", "unknown")
                            weight = feature.get("weight", 1.0)
                            print(f"    - {name} (weight: {weight:.2f})")
        
        except Exception as e:
            print(f"Error reading patterns: {e}")
    
    def setup_engine(self) -> None:
        """Set up the triangulation engine with neural components."""
        if self.engine is not None:
            return
            
        # Create mock config
        config = {
            "planner": {
                "enabled": True,
                "family-tree-path": ".triangulum/family_tree.json",
                "solutions-per-bug": 3,
                "max-path-depth": 5,
                "fallback-threshold": 0.3,
                "learning-rate": 0.1,
                "pattern-threshold": 0.7,
                "history-limit": 1000
            },
            "neural_matrix": {
                "enabled": True,
                "hub_url": "http://localhost:8001",
                "update_frequency": 60
            }
        }
        
        # Create agent memory
        self.agent_memory = AgentMemory()
        
        # Create a simplified engine for demonstration
        # This avoids the FamilyTree.load_from_file() issue
        # Instead of calling TriangulationEngine directly
        class SimplifiedEngine:
            def __init__(self, config):
                self.config = config
                self.planner = None
                
            def get_planner(self):
                return None
                
            def add_bug(self, bug):
                print(f"Added bug: {bug.bug_id}")
        
        self.engine = SimplifiedEngine(config)
        
        # Create scope filter
        self.scope_filter = scope_filter.ScopeFilter(
            languages=["python", "javascript"],
            max_entropy_bits=12
        )
        
        print("Triangulation engine initialized with neural components.")
    
    def execute_solution_path(self, bug_description: str) -> None:
        """
        Execute a neural-guided solution path.
        
        Args:
            bug_description: Description of the bug to solve
        """
        # Set up engine if not already done
        self.setup_engine()
        
        print(f"Executing neural-guided solution path for bug: {bug_description}")
        
        # Create a bug state
        bug_id = f"bug-{int(time.time())}"
        bug = BugState(
            bug_id=bug_id,
            title=f"Bug {bug_id}",
            description=bug_description,
            severity="medium",
            status="new"
        )
        
        # Add tags based on description
        for keyword, tag in [
            ("memory", "memory"),
            ("leak", "memory-leak"),
            ("null", "null-pointer"),
            ("pointer", "pointer"),
            ("race", "race-condition"),
            ("thread", "threading"),
            ("performance", "performance"),
            ("security", "security"),
            ("crash", "crash")
        ]:
            if keyword in bug_description.lower():
                bug.add_tag(tag)
                
        # Add bug to engine
        self.engine.add_bug(bug)
        
        # Get planner agent
        planner = self.engine.get_planner()
        if not planner:
            print("Error: Planner agent not available.")
            return
            
        # Generate solution paths
        print("Generating solution paths...")
        paths = planner.generate_solution_paths(bug)
        
        if not paths:
            print("No solution paths generated.")
            return
            
        print(f"Generated {len(paths)} solution paths.")
        
        # Select best path
        best_path = planner.select_best_path(bug_id)
        if not best_path:
            print("Error: Could not select best path.")
            return
            
        print(f"Selected best path: {best_path.path_id}")
        print(f"Path priority: {best_path.metadata.get('priority', 0.0):.2f}")
        print(f"Estimated time: {best_path.metadata.get('estimated_time', 0)} ticks")
        
        # Show actions
        print("\nActions:")
        for i, action in enumerate(best_path.actions, 1):
            action_type = action.get("type", "unknown")
            agent = action.get("agent", "unknown")
            description = action.get("description", "No description")
            print(f"  {i}. [{agent}] {action_type}: {description}")
            
        # Show fallbacks if any
        if best_path.fallbacks:
            print("\nFallback strategies:")
            for i, fallback in enumerate(best_path.fallbacks, 1):
                fallback_type = fallback.get("type", "unknown")
                description = fallback.get("description", "No description")
                actions = fallback.get("actions", [])
                print(f"  {i}. {fallback_type}: {description} ({len(actions)} actions)")
        
        # Simulate execution
        print("\nSimulating execution...")
        for i, action in enumerate(best_path.actions, 1):
            action_type = action.get("type", "unknown")
            agent = action.get("agent", "unknown")
            print(f"  Executing action {i}/{len(best_path.actions)}: [{agent}] {action_type}...")
            time.sleep(0.5)  # Simulate execution time
            
        # Record successful result
        print("\nRecording successful result...")
        planner.record_path_result(best_path.path_id, True, {
            "execution_time": len(best_path.actions) * 0.5,
            "actions_completed": len(best_path.actions)
        })
        
        # Apply neural learning
        print("\nApplying neural learning...")
        planner.neural_learning(bug, {
            "path_id": best_path.path_id,
            "success": True,
            "metrics": {
                "execution_time": len(best_path.actions) * 0.5,
                "actions_completed": len(best_path.actions)
            },
            "actions": best_path.actions
        })
        
        print("\nSolution path executed successfully!")
    
    def demonstrate_learning(self) -> None:
        """Demonstrate neural learning capabilities."""
        # Set up engine if not already done
        self.setup_engine()
        
        print("Demonstrating neural learning capabilities...")
        
        # Create a planner agent
        config = {
            "planner": {
                "enabled": True,
                "solutions-per-bug": 3,
                "learning-rate": 0.1,
                "pattern-threshold": 0.7
            }
        }
        planner = planner_agent.PlannerAgent(config, self.agent_memory)
        
        # Create a bug
        bug = BugState(
            bug_id="demo-bug-1",
            title="Memory leak in rendering function",
            description="The application leaks memory when rendering complex scenes",
            severity="high",
            status="new"
        )
        bug.add_tag("memory-leak")
        bug.add_tag("rendering")
        
        # Show initial neural weights
        print("\nInitial neural weights:")
        for agent_type, weight in planner.neural_weights.items():
            print(f"  {agent_type}: {weight:.2f}")
            
        # Simulate a successful solution
        print("\nSimulating successful solution...")
        
        # Create simulated actions
        actions = [
            {"type": "analyze", "agent": "observer", "description": "Analyze memory usage"},
            {"type": "patch", "agent": "analyst", "description": "Fix memory allocation"},
            {"type": "verify", "agent": "verifier", "description": "Verify fix"}
        ]
        
        # Apply neural learning
        planner.neural_learning(bug, {
            "path_id": "demo-path-1",
            "success": True,
            "metrics": {"execution_time": 1.5},
            "actions": actions
        })
        
        # Show updated weights
        print("\nNeural weights after successful solution:")
        for agent_type, weight in planner.neural_weights.items():
            print(f"  {agent_type}: {weight:.2f}")
            
        # Simulate a failed solution
        print("\nSimulating failed solution...")
        
        # Create simulated actions
        actions = [
            {"type": "analyze", "agent": "observer", "description": "Quick analysis"},
            {"type": "patch", "agent": "analyst", "description": "Attempt fix"}
        ]
        
        # Apply neural learning
        planner.neural_learning(bug, {
            "path_id": "demo-path-2",
            "success": False,
            "metrics": {"execution_time": 0.8},
            "actions": actions
        })
        
        # Show final weights
        print("\nNeural weights after failed solution:")
        for agent_type, weight in planner.neural_weights.items():
            print(f"  {agent_type}: {weight:.2f}")
            
        print("\nNeural learning demonstration complete!")
    
    def show_stats(self) -> None:
        """Show neural matrix statistics."""
        print("Neural matrix statistics:")
        
        # Count patterns
        patterns_file = self.neural_matrix_dir / "patterns" / "starter_patterns.json"
        pattern_count = 0
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    patterns = json.load(f)
                pattern_count = len(patterns)
            except Exception:
                pass
                
        # Check weights
        weights_file = self.neural_matrix_dir / "weights" / "default_weights.json"
        weight_count = 0
        if weights_file.exists():
            try:
                with open(weights_file, 'r') as f:
                    weights = json.load(f)
                weight_count = (
                    len(weights.get("agent_weights", {})) +
                    len(weights.get("feature_weights", {})) +
                    len(weights.get("learning_parameters", {}))
                )
            except Exception:
                pass
                
        # Check family tree
        family_tree_path = Path(".triangulum/family_tree.json")
        agent_count = 0
        connection_count = 0
        if family_tree_path.exists():
            try:
                with open(family_tree_path, 'r') as f:
                    tree_data = json.load(f)
                agent_count = len(tree_data.get("relationships", {}))
                neural_connections = tree_data.get("neural_connections", {})
                connection_count = sum(len(connections) for connections in neural_connections.values())
            except Exception:
                pass
                
        # Display statistics
        print(f"  Neural patterns: {pattern_count}")
        print(f"  Neural weights: {weight_count}")
        print(f"  Agents in family tree: {agent_count}")
        print(f"  Neural connections: {connection_count}")
        
        # Show directory structure
        print("\nNeural matrix directory structure:")
        dirs = [
            "patterns",
            "weights",
            "history",
            "connections",
            "test_data"
        ]
        for dir_name in dirs:
            dir_path = self.neural_matrix_dir / dir_name
            if dir_path.exists():
                print(f"  {dir_name}/")
                files = list(dir_path.glob("*"))
                for file in files[:5]:  # Show up to 5 files
                    print(f"    {file.name}")
                if len(files) > 5:
                    print(f"    ... ({len(files) - 5} more files)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Neural Matrix Executor")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize neural matrix")
    init_parser.add_argument("--force", action="store_true", help="Force reinitialization")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate neural matrix integrity")
    validate_parser.add_argument("--detailed", action="store_true", help="Show detailed validation results")
    
    # Patterns command
    patterns_parser = subparsers.add_parser("patterns", help="List or search neural patterns")
    patterns_parser.add_argument("--search", help="Search term for filtering patterns")
    
    # Execute command
    execute_parser = subparsers.add_parser("execute", help="Execute a neural-guided solution path")
    execute_parser.add_argument("--bug", required=True, help="Bug description")
    
    # Learn command
    subparsers.add_parser("learn", help="Demonstrate neural learning")
    
    # Stats command
    subparsers.add_parser("stats", help="Show neural matrix statistics")
    
    args = parser.parse_args()
    
    executor = NeuralMatrixExecutor()
    
    if args.command == "init":
        executor.initialize(args.force)
    elif args.command == "validate":
        executor.validate(args.detailed)
    elif args.command == "patterns":
        executor.list_patterns(args.search)
    elif args.command == "execute":
        executor.execute_solution_path(args.bug)
    elif args.command == "learn":
        executor.demonstrate_learning()
    elif args.command == "stats":
        executor.show_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
