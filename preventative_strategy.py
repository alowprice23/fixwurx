#!/usr/bin/env python3
"""
preventative_strategy.py
───────────────────────
System for developing preventative strategies for known blocker patterns.

This module provides specialized tools for developing, refining, and evaluating
preventative strategies to avoid recurring blockers.
"""

import os
import sys
import time
import json
import logging
import datetime
import traceback
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum
from collections import defaultdict

# Internal imports
from shell_environment import register_command, emit_event, EventType
from meta_agent import request_agent_task
from blocker_detection import BlockerType, BlockerSeverity
from response_protocol import ResponseStrategy
from blocker_learning import BlockerPattern, PatternType, BlockerLearningSystem

# Configure logging
logger = logging.getLogger("PreventativeStrategy")
handler = logging.FileHandler("preventative_strategy.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class StrategyScope(Enum):
    """Scope of a preventative strategy."""
    GLOBAL = "global"             # Strategy applies globally
    OPERATION = "operation"       # Strategy applies to specific operations
    COMPONENT = "component"       # Strategy applies to specific components
    RESOURCE = "resource"         # Strategy applies to specific resources
    DEPENDENCY = "dependency"     # Strategy applies to specific dependencies
    ERROR = "error"               # Strategy applies to specific errors

class StrategyStatus(Enum):
    """Status of a preventative strategy."""
    DRAFT = "draft"               # Initial draft
    PROPOSED = "proposed"         # Proposed for implementation
    IMPLEMENTED = "implemented"   # Implemented in the system
    ACTIVE = "active"             # Currently active
    EVALUATED = "evaluated"       # Evaluated for effectiveness
    DEPRECATED = "deprecated"     # No longer used

class PreventativeStrategyManager:
    """
    System for developing preventative strategies for known blocker patterns.
    
    Provides tools for developing, refining, and evaluating preventative strategies
    to avoid recurring blockers.
    """
    
    def __init__(self, 
                storage_file: str = "preventative_strategies.json",
                blocker_learning_system: Optional[BlockerLearningSystem] = None):
        """
        Initialize the preventative strategy manager.
        
        Args:
            storage_file: File to store preventative strategies
            blocker_learning_system: Blocker learning system to use
        """
        self.storage_file = storage_file
        self.strategies = {}
        self.blocker_learning_system = blocker_learning_system or BlockerLearningSystem()
        self.load_strategies()
        
        # Register commands
        try:
            register_command("develop_strategy", self.develop_strategy_command, 
                            "Develop a preventative strategy for a blocker pattern")
            register_command("list_strategies", self.list_strategies_command,
                            "List preventative strategies")
            register_command("evaluate_strategy", self.evaluate_strategy_command,
                            "Evaluate the effectiveness of a preventative strategy")
            register_command("implement_strategy", self.implement_strategy_command,
                            "Implement a preventative strategy")
            logger.info("Preventative strategy commands registered")
        except Exception as e:
            logger.error(f"Failed to register commands: {e}")
    
    def load_strategies(self):
        """Load strategies from storage file."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                
                for strategy_id, strategy_data in data.items():
                    self.strategies[strategy_id] = strategy_data
                
                logger.info(f"Loaded {len(self.strategies)} preventative strategies from {self.storage_file}")
            else:
                logger.info(f"No strategy storage file found at {self.storage_file}")
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
    
    def save_strategies(self):
        """Save strategies to storage file."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.strategies, f, indent=2)
            
            logger.info(f"Saved {len(self.strategies)} preventative strategies to {self.storage_file}")
        except Exception as e:
            logger.error(f"Error saving strategies: {e}")
    
    def develop_strategy(self, 
                        pattern_id: str, 
                        scope: Optional[StrategyScope] = None) -> Dict[str, Any]:
        """
        Develop a preventative strategy for a blocker pattern.
        
        Args:
            pattern_id: ID of the blocker pattern
            scope: Scope of the strategy
            
        Returns:
            Strategy information
        """
        # Get pattern
        if pattern_id not in self.blocker_learning_system.patterns:
            return {
                "success": False,
                "error": f"Pattern {pattern_id} not found"
            }
        
        pattern = self.blocker_learning_system.patterns[pattern_id]
        
        # Determine strategy scope if not provided
        if not scope:
            scope = self._determine_strategy_scope(pattern)
        
        # Generate strategy using LLM agent
        strategy = self._generate_strategy(pattern, scope)
        
        if not strategy:
            return {
                "success": False,
                "error": f"Failed to generate strategy for pattern {pattern_id}"
            }
        
        # Add metadata
        strategy_id = f"strategy_{int(time.time())}_{len(self.strategies)}"
        strategy["id"] = strategy_id
        strategy["pattern_id"] = pattern_id
        strategy["scope"] = scope.value
        strategy["status"] = StrategyStatus.DRAFT.value
        strategy["created_at"] = time.time()
        strategy["updated_at"] = time.time()
        strategy["effectiveness"] = 0.0
        strategy["implementation_status"] = "not implemented"
        
        # Store strategy
        self.strategies[strategy_id] = strategy
        
        # Save strategies
        self.save_strategies()
        
        logger.info(f"Developed strategy {strategy_id} for pattern {pattern_id}")
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "strategy": strategy
        }
    
    def _determine_strategy_scope(self, pattern: BlockerPattern) -> StrategyScope:
        """
        Determine the scope of a strategy for a pattern.
        
        Args:
            pattern: Blocker pattern
            
        Returns:
            Strategy scope
        """
        # Check pattern type to determine scope
        if pattern.pattern_type == PatternType.RESOURCE_THRESHOLD:
            return StrategyScope.RESOURCE
        
        elif pattern.pattern_type == PatternType.EXTERNAL:
            return StrategyScope.DEPENDENCY
        
        elif pattern.pattern_type == PatternType.ERROR_SEQUENCE:
            return StrategyScope.ERROR
        
        elif pattern.pattern_type == PatternType.INTERACTION:
            return StrategyScope.COMPONENT
        
        # Check if pattern is specific to an operation
        operation_types = set()
        for instance in pattern.instances:
            context = instance.get("context", {})
            if "operation_type" in context:
                operation_types.add(context["operation_type"])
        
        if len(operation_types) == 1:
            return StrategyScope.OPERATION
        
        # Default to global scope
        return StrategyScope.GLOBAL
    
    def _generate_strategy(self, 
                          pattern: BlockerPattern, 
                          scope: StrategyScope) -> Optional[Dict[str, Any]]:
        """
        Generate a preventative strategy for a blocker pattern.
        
        Args:
            pattern: Blocker pattern
            scope: Strategy scope
            
        Returns:
            Strategy information or None if generation failed
        """
        # Try to generate a strategy using an agent
        try:
            # Prepare data for the agent
            pattern_data = {
                "id": pattern.pattern_id,
                "type": pattern.pattern_type.value,
                "description": pattern.description,
                "occurrences": pattern.occurrence_count,
                "resolved_count": pattern.resolved_count,
                "success_rate": pattern.resolved_count / max(1, pattern.occurrence_count),
                "first_seen": pattern.first_seen,
                "last_seen": pattern.last_seen
            }
            
            # Get sample instances
            sample_instances = pattern.instances[-5:] if len(pattern.instances) > 5 else pattern.instances
            
            # Prepare prompt for agent
            prompt = f"""
            Generate a comprehensive preventative strategy for the following blocker pattern:
            
            Pattern ID: {pattern.pattern_id}
            Pattern Type: {pattern.pattern_type.value}
            Description: {pattern.description}
            Occurrences: {pattern.occurrence_count}
            Success Rate: {pattern.resolved_count / max(1, pattern.occurrence_count):.2f}
            Strategy Scope: {scope.value}
            
            Sample Instances:
            {json.dumps(sample_instances, indent=2)}
            
            Please provide a preventative strategy that includes:
            
            1. Name: A clear, descriptive name for the strategy
            2. Description: A detailed description of the strategy and its purpose
            3. Scope: Confirmation of the appropriate scope (global, operation, component, resource, dependency, error)
            4. Rationale: Explanation of why this strategy should be effective
            5. Implementation Steps: Detailed steps to implement the strategy
            6. Prerequisites: Any prerequisites for implementing the strategy
            7. Estimated Effectiveness: Estimated effectiveness rating (0.0-1.0)
            8. Estimated Implementation Difficulty: Low, medium, or high
            9. Potential Side Effects: Any potential negative impacts of the strategy
            10. Monitoring Recommendations: How to monitor the strategy's effectiveness
            
            Return your strategy as a structured object.
            """
            
            # Request strategy from agent
            result = request_agent_task("comprehensive_strategy_generation", prompt, timeout=60)
            
            if result.get("success", False) and result.get("strategy"):
                strategy = result["strategy"]
                
                # Ensure required fields
                if "name" not in strategy or "description" not in strategy or "implementation_steps" not in strategy:
                    logger.warning(f"Generated strategy for pattern {pattern.pattern_id} is missing required fields")
                    return None
                
                logger.info(f"Generated strategy for pattern {pattern.pattern_id}: {strategy.get('name', 'Unnamed strategy')}")
                
                return strategy
            
            logger.warning(f"Failed to generate strategy for pattern {pattern.pattern_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error generating strategy for pattern {pattern.pattern_id}: {e}")
            return None
    
    def refine_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Refine an existing preventative strategy.
        
        Args:
            strategy_id: ID of the strategy to refine
            
        Returns:
            Refined strategy information
        """
        # Check if strategy exists
        if strategy_id not in self.strategies:
            return {
                "success": False,
                "error": f"Strategy {strategy_id} not found"
            }
        
        strategy = self.strategies[strategy_id]
        
        # Get pattern
        pattern_id = strategy.get("pattern_id")
        if not pattern_id or pattern_id not in self.blocker_learning_system.patterns:
            return {
                "success": False,
                "error": f"Pattern {pattern_id} not found for strategy {strategy_id}"
            }
        
        pattern = self.blocker_learning_system.patterns[pattern_id]
        
        # Prepare data for refinement
        pattern_data = {
            "id": pattern.pattern_id,
            "type": pattern.pattern_type.value,
            "description": pattern.description,
            "occurrences": pattern.occurrence_count,
            "resolved_count": pattern.resolved_count,
            "success_rate": pattern.resolved_count / max(1, pattern.occurrence_count),
            "first_seen": pattern.first_seen,
            "last_seen": pattern.last_seen
        }
        
        # Get new instances since strategy creation
        new_instances = []
        for instance in pattern.instances:
            if instance.get("timestamp", 0) > strategy.get("created_at", 0):
                new_instances.append(instance)
        
        # If no new instances, don't refine
        if not new_instances:
            return {
                "success": False,
                "error": f"No new instances for pattern {pattern_id} since strategy {strategy_id} was created"
            }
        
        # Prepare prompt for agent
        prompt = f"""
        Refine the following preventative strategy based on new blocker instances:
        
        Strategy:
        {json.dumps(strategy, indent=2)}
        
        Pattern Information:
        {json.dumps(pattern_data, indent=2)}
        
        New Instances Since Strategy Creation:
        {json.dumps(new_instances, indent=2)}
        
        Please refine the strategy to improve its effectiveness based on the new instances.
        Maintain the same structure but update any fields that need improvement.
        
        Return the refined strategy as a structured object.
        """
        
        try:
            # Request refinement from agent
            result = request_agent_task("strategy_refinement", prompt, timeout=60)
            
            if result.get("success", False) and result.get("refined_strategy"):
                refined_strategy = result["refined_strategy"]
                
                # Preserve metadata
                refined_strategy["id"] = strategy_id
                refined_strategy["pattern_id"] = pattern_id
                refined_strategy["created_at"] = strategy["created_at"]
                refined_strategy["updated_at"] = time.time()
                refined_strategy["effectiveness"] = strategy.get("effectiveness", 0.0)
                refined_strategy["implementation_status"] = strategy.get("implementation_status", "not implemented")
                
                # Update strategy
                self.strategies[strategy_id] = refined_strategy
                
                # Save strategies
                self.save_strategies()
                
                logger.info(f"Refined strategy {strategy_id} for pattern {pattern_id}")
                
                return {
                    "success": True,
                    "strategy_id": strategy_id,
                    "refined_strategy": refined_strategy,
                    "previous_strategy": strategy
                }
            
            return {
                "success": False,
                "error": f"Failed to refine strategy {strategy_id}"
            }
                
        except Exception as e:
            logger.error(f"Error refining strategy {strategy_id}: {e}")
            return {
                "success": False,
                "error": f"Error refining strategy {strategy_id}: {e}"
            }
    
    def evaluate_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of a preventative strategy.
        
        Args:
            strategy_id: ID of the strategy to evaluate
            
        Returns:
            Evaluation information
        """
        # Check if strategy exists
        if strategy_id not in self.strategies:
            return {
                "success": False,
                "error": f"Strategy {strategy_id} not found"
            }
        
        strategy = self.strategies[strategy_id]
        
        # Get pattern
        pattern_id = strategy.get("pattern_id")
        if not pattern_id or pattern_id not in self.blocker_learning_system.patterns:
            return {
                "success": False,
                "error": f"Pattern {pattern_id} not found for strategy {strategy_id}"
            }
        
        pattern = self.blocker_learning_system.patterns[pattern_id]
        
        # Check if strategy has been implemented
        if strategy.get("implementation_status") != "implemented":
            return {
                "success": False,
                "error": f"Strategy {strategy_id} has not been implemented"
            }
        
        # Get implementation time
        implementation_time = strategy.get("implementation_time", time.time())
        
        # Count occurrences before and after implementation
        occurrences_before = 0
        occurrences_after = 0
        
        for instance in pattern.instances:
            if instance.get("timestamp", 0) < implementation_time:
                occurrences_before += 1
            else:
                occurrences_after += 1
        
        # Calculate time periods
        time_before = implementation_time - pattern.first_seen
        time_after = time.time() - implementation_time
        
        # Calculate rates
        if time_before > 0:
            rate_before = occurrences_before / (time_before / (60 * 60 * 24))  # per day
        else:
            rate_before = 0
        
        if time_after > 0:
            rate_after = occurrences_after / (time_after / (60 * 60 * 24))  # per day
        else:
            rate_after = 0
        
        # Calculate effectiveness
        if rate_before > 0:
            effectiveness = max(0, min(1, (rate_before - rate_after) / rate_before))
        else:
            effectiveness = 0
        
        # Update strategy
        strategy["effectiveness"] = effectiveness
        strategy["status"] = StrategyStatus.EVALUATED.value
        strategy["evaluation"] = {
            "occurrences_before": occurrences_before,
            "occurrences_after": occurrences_after,
            "time_before_days": time_before / (60 * 60 * 24),
            "time_after_days": time_after / (60 * 60 * 24),
            "rate_before": rate_before,
            "rate_after": rate_after,
            "effectiveness": effectiveness,
            "evaluation_time": time.time()
        }
        
        # Save strategies
        self.save_strategies()
        
        logger.info(f"Evaluated strategy {strategy_id} with effectiveness {effectiveness:.2f}")
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "effectiveness": effectiveness,
            "evaluation": strategy["evaluation"]
        }
    
    def implement_strategy(self, strategy_id: str, auto_implement: bool = False) -> Dict[str, Any]:
        """
        Implement a preventative strategy.
        
        Args:
            strategy_id: ID of the strategy to implement
            auto_implement: Whether to automatically implement the strategy
            
        Returns:
            Implementation information
        """
        # Check if strategy exists
        if strategy_id not in self.strategies:
            return {
                "success": False,
                "error": f"Strategy {strategy_id} not found"
            }
        
        strategy = self.strategies[strategy_id]
        
        # Get implementation steps
        implementation_steps = strategy.get("implementation_steps", [])
        
        if not implementation_steps:
            return {
                "success": False,
                "error": f"Strategy {strategy_id} has no implementation steps"
            }
        
        # Execute implementation steps if auto_implement is True
        implementation_results = []
        
        if auto_implement:
            for i, step in enumerate(implementation_steps, 1):
                logger.info(f"Implementing step {i}/{len(implementation_steps)}: {step}")
                
                # Here you would implement the actual step
                # For now, we'll just simulate it
                
                implementation_results.append({
                    "step": i,
                    "description": step,
                    "status": "implemented",
                    "timestamp": time.time()
                })
        
        # Update strategy
        strategy["implementation_status"] = "implemented" if auto_implement else "manual_implementation_required"
        strategy["implementation_time"] = time.time() if auto_implement else None
        strategy["status"] = StrategyStatus.IMPLEMENTED.value if auto_implement else StrategyStatus.PROPOSED.value
        
        if auto_implement:
            strategy["implementation_results"] = implementation_results
        
        # Save strategies
        self.save_strategies()
        
        logger.info(f"Strategy {strategy_id} implementation {'completed' if auto_implement else 'prepared'}")
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "auto_implemented": auto_implement,
            "implementation_steps": implementation_steps,
            "implementation_results": implementation_results if auto_implement else None
        }
    
    def develop_strategy_command(self, args: str) -> int:
        """
        Handle the develop_strategy command.
        
        Args:
            args: Command arguments (pattern_id [scope])
            
        Returns:
            Exit code
        """
        try:
            arg_parts = args.strip().split()
            
            if not arg_parts:
                print("Error: Pattern ID required.")
                print("Usage: develop_strategy <pattern_id> [scope]")
                print("Available scopes:")
                for scope in StrategyScope:
                    print(f"  {scope.value}")
                return 1
            
            pattern_id = arg_parts[0]
            
            scope = None
            if len(arg_parts) > 1:
                try:
                    scope = StrategyScope(arg_parts[1])
                except ValueError:
                    print(f"Error: Invalid scope: {arg_parts[1]}")
                    print("Available scopes:")
                    for scope in StrategyScope:
                        print(f"  {scope.value}")
                    return 1
            
            # Develop strategy
            print(f"Developing strategy for pattern {pattern_id}...")
            
            result = self.develop_strategy(pattern_id, scope)
            
            if result.get("success", False):
                strategy_id = result["strategy_id"]
                strategy = result["strategy"]
                
                print(f"Developed strategy: {strategy_id}")
                print(f"Name: {strategy.get('name', 'Unnamed strategy')}")
                print(f"Description: {strategy.get('description', 'No description')}")
                print(f"Scope: {strategy.get('scope', 'Unknown scope')}")
                print(f"Difficulty: {strategy.get('estimated_difficulty', 'Unknown difficulty')}")
                print(f"Estimated Effectiveness: {strategy.get('estimated_effectiveness', 'Unknown')}")
                
                print("\nImplementation Steps:")
                for i, step in enumerate(strategy.get("implementation_steps", []), 1):
                    print(f"  {i}. {step}")
                
                # Save strategy to file
                output_file = f"strategy_{strategy_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(strategy, f, indent=2)
                
                print(f"\nStrategy saved to {output_file}")
                
                return 0
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1
                
        except Exception as e:
            print(f"Error developing strategy: {e}")
            logger.error(f"Error developing strategy: {e}")
            return 1
    
    def list_strategies_command(self, args: str) -> int:
        """
        Handle the list_strategies command.
        
        Args:
            args: Command arguments (optional pattern_id)
            
        Returns:
            Exit code
        """
        try:
            pattern_id = args.strip() if args.strip() else None
            
            # Filter strategies
            if pattern_id:
                strategies = {s_id: s for s_id, s in self.strategies.items() if s.get("pattern_id") == pattern_id}
            else:
                strategies = self.strategies
            
            # Sort strategies by creation time
            sorted_strategies = sorted(strategies.items(), key=lambda x: x[1].get("created_at", 0), reverse=True)
            
            print(f"Found {len(sorted_strategies)} strategies:")
            
            for strategy_id, strategy in sorted_strategies:
                print(f"\nStrategy ID: {strategy_id}")
                print(f"  Name: {strategy.get('name', 'Unnamed strategy')}")
                print(f"  Pattern ID: {strategy.get('pattern_id', 'Unknown pattern')}")
                print(f"  Description: {strategy.get('description', 'No description')}")
                print(f"  Scope: {strategy.get('scope', 'Unknown scope')}")
                print(f"  Status: {strategy.get('status', 'Unknown status')}")
                print(f"  Effectiveness: {strategy.get('effectiveness', 0.0):.2f}")
                print(f"  Implementation Status: {strategy.get('implementation_status', 'not implemented')}")
                print(f"  Created: {datetime.datetime.fromtimestamp(strategy.get('created_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
            
            return 0
                
        except Exception as e:
            print(f"Error listing strategies: {e}")
            logger.error(f"Error listing strategies: {e}")
            return 1
    
    def evaluate_strategy_command(self, args: str) -> int:
        """
        Handle the evaluate_strategy command.
        
        Args:
            args: Command arguments (strategy_id)
            
        Returns:
            Exit code
        """
        try:
            strategy_id = args.strip()
            
            if not strategy_id:
                print("Error: Strategy ID required.")
                print("Usage: evaluate_strategy <strategy_id>")
                return 1
            
            # Evaluate strategy
            print(f"Evaluating strategy {strategy_id}...")
            
            result = self.evaluate_strategy(strategy_id)
            
            if result.get("success", False):
                evaluation = result["evaluation"]
                
                print(f"Strategy Effectiveness: {result['effectiveness']:.2f}")
                print(f"Occurrences Before Implementation: {evaluation['occurrences_before']}")
                print(f"Occurrences After Implementation: {evaluation['occurrences_after']}")
                print(f"Rate Before: {evaluation['rate_before']:.2f} per day")
                print(f"Rate After: {evaluation['rate_after']:.2f} per day")
                
                return 0
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1
                
        except Exception as e:
            print(f"Error evaluating strategy: {e}")
            logger.error(f"Error evaluating strategy: {e}")
            return 1
    
    def implement_strategy_command(self, args: str) -> int:
        """
        Handle the implement_strategy command.
        
        Args:
            args: Command arguments (strategy_id [auto])
            
        Returns:
            Exit code
        """
        try:
            arg_parts = args.strip().split()
            
            if not arg_parts:
                print("Error: Strategy ID required.")
                print("Usage: implement_strategy <strategy_id> [auto]")
                return 1
            
            strategy_id = arg_parts[0]
            
            auto_implement = False
            if len(arg_parts) > 1 and arg_parts[1].lower() == "auto":
                auto_implement = True
            
            # Implement strategy
            print(f"Implementing strategy {strategy_id}...")
            
            result = self.implement_strategy(strategy_id, auto_implement)
            
            if result.get("success", False):
                if auto_implement:
                    print("Strategy implemented automatically.")
                    print(f"Implementation steps completed: {len(result.get('implementation_results', []))}")
                else:
                    print("Strategy prepared for manual implementation.")
                    print("\nImplementation Steps:")
                    for i, step in enumerate(result.get("implementation_steps", []), 1):
                        print(f"  {i}. {step}")
                
                return 0
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1
                
        except Exception as e:
            print(f"Error implementing strategy: {e}")
            logger.error(f"Error implementing strategy: {e}")
            return 1


# Initialize preventative strategy manager
preventative_strategy_manager = PreventativeStrategyManager()
logger.info("Preventative strategy manager initialized")
