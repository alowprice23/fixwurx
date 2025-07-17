#!/usr/bin/env python3
"""
Solution Paths Module

This module provides multiple solution strategies for fixing bugs, with primary and fallback paths,
prioritization mechanisms, complexity-aware selection, and neural matrix integration.
"""

import os
import sys
import json
import logging
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("solution_paths.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SolutionPaths")

# Solution Strategy Types
STRATEGY_TYPES = [
    "SYNTAX_FIX",         # Fix syntax errors
    "LOGIC_FIX",          # Fix logical errors
    "VALIDATION_FIX",     # Add input validation
    "ERROR_HANDLING_FIX", # Improve error handling
    "PERFORMANCE_FIX",    # Optimize performance
    "SECURITY_FIX",       # Fix security issues
    "REFACTOR",           # Clean up code
    "BEST_PRACTICE_FIX",  # Apply best practices
    "TEST_FIX"            # Fix test failures
]

class SolutionStrategy:
    """
    A solution strategy for fixing a specific type of bug.
    """
    
    def __init__(self, strategy_type: str, name: str, description: str, 
                 complexity: int, success_probability: float, 
                 handler: Optional[Callable] = None):
        """
        Initialize a solution strategy.
        
        Args:
            strategy_type: Type of strategy (from STRATEGY_TYPES)
            name: Name of the strategy
            description: Description of the strategy
            complexity: Complexity score (1-10, where 1 is simplest)
            success_probability: Probability of success (0.0-1.0)
            handler: Function to execute the strategy
        """
        self.strategy_type = strategy_type
        self.name = name
        self.description = description
        self.complexity = complexity
        self.success_probability = success_probability
        self.handler = handler
        self.execution_history = []
        
        # Neural network weights for this strategy
        # These will be adjusted based on learning
        self.weights = {
            "bug_type_weights": {},      # Weights for different bug types
            "language_weights": {},      # Weights for different languages
            "file_size_weight": 1.0,     # Weight for file size factor
            "complexity_weight": 1.0,    # Weight for complexity factor
            "history_weight": 1.0        # Weight for historical success
        }
        
        logger.info(f"Initialized solution strategy: {name} ({strategy_type})")
    
    def execute(self, bug_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the solution strategy.
        
        Args:
            bug_data: Data about the bug to fix
            context: Additional context for fixing the bug
            
        Returns:
            Result of the execution
        """
        logger.info(f"Executing solution strategy: {self.name}")
        
        start_time = context.get("start_time", 0)
        result = {
            "success": False,
            "strategy": self.name,
            "strategy_type": self.strategy_type,
            "bug_id": bug_data.get("id", "unknown"),
            "fixes_applied": [],
            "execution_time": 0,
            "error": None
        }
        
        try:
            if self.handler:
                fix_result = self.handler(bug_data, context)
                result.update(fix_result)
            else:
                # Mock implementation for testing
                result["success"] = random.random() < self.success_probability
                result["fixes_applied"] = ["Mock fix"]
                result["execution_time"] = random.uniform(0.1, 2.0)
        except Exception as e:
            logger.error(f"Error executing strategy {self.name}: {e}")
            result["error"] = str(e)
        
        # Record execution for learning
        self.record_execution(bug_data, context, result)
        
        return result
    
    def record_execution(self, bug_data: Dict[str, Any], context: Dict[str, Any], 
                        result: Dict[str, Any]) -> None:
        """
        Record the execution of this strategy for learning.
        
        Args:
            bug_data: Data about the bug
            context: Context of the execution
            result: Result of the execution
        """
        execution_record = {
            "bug_id": bug_data.get("id", "unknown"),
            "bug_type": bug_data.get("type", "unknown"),
            "language": bug_data.get("language", "unknown"),
            "file_size": bug_data.get("file_size", 0),
            "complexity": bug_data.get("complexity", 0),
            "success": result.get("success", False),
            "execution_time": result.get("execution_time", 0),
            "fixes_applied": result.get("fixes_applied", []),
            "timestamp": context.get("timestamp", 0)
        }
        
        self.execution_history.append(execution_record)
        
        # Limit history size
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def calculate_score(self, bug_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Calculate a score for this strategy based on the bug data and context.
        
        Args:
            bug_data: Data about the bug
            context: Additional context
            
        Returns:
            Score for this strategy (higher is better)
        """
        bug_type = bug_data.get("type", "unknown")
        language = bug_data.get("language", "unknown")
        file_size = bug_data.get("file_size", 0)
        bug_complexity = bug_data.get("complexity", 0)
        
        # Base score is the success probability
        score = self.success_probability
        
        # Adjust for bug type
        bug_type_weight = self.weights["bug_type_weights"].get(bug_type, 1.0)
        score *= bug_type_weight
        
        # Adjust for language
        language_weight = self.weights["language_weights"].get(language, 1.0)
        score *= language_weight
        
        # Adjust for file size
        file_size_factor = 1.0 if file_size < 1000 else 0.9
        score *= file_size_factor * self.weights["file_size_weight"]
        
        # Adjust for complexity
        # Lower scores for complex bugs with simple strategies and vice versa
        complexity_match = 1.0 - abs(self.complexity - bug_complexity) / 10.0
        score *= complexity_match * self.weights["complexity_weight"]
        
        # Adjust for historical performance
        if self.execution_history:
            # Calculate success rate for similar bugs
            similar_bugs = [
                record for record in self.execution_history
                if record["bug_type"] == bug_type and record["language"] == language
            ]
            
            if similar_bugs:
                success_rate = sum(1 for record in similar_bugs if record["success"]) / len(similar_bugs)
                score *= (0.5 + 0.5 * success_rate) * self.weights["history_weight"]
        
        return score

class SolutionPath:
    """
    A solution path consisting of primary and fallback strategies.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a solution path.
        
        Args:
            name: Name of the solution path
            description: Description of the solution path
        """
        self.name = name
        self.description = description
        self.primary_strategies = []
        self.fallback_strategies = []
        self.execution_history = []
        
        logger.info(f"Initialized solution path: {name}")
    
    def add_primary_strategy(self, strategy: SolutionStrategy) -> None:
        """
        Add a primary strategy to the solution path.
        
        Args:
            strategy: Strategy to add
        """
        self.primary_strategies.append(strategy)
    
    def add_fallback_strategy(self, strategy: SolutionStrategy) -> None:
        """
        Add a fallback strategy to the solution path.
        
        Args:
            strategy: Strategy to add
        """
        self.fallback_strategies.append(strategy)
    
    def execute(self, bug_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the solution path.
        
        Args:
            bug_data: Data about the bug to fix
            context: Additional context for fixing the bug
            
        Returns:
            Result of the execution
        """
        logger.info(f"Executing solution path: {self.name}")
        
        result = {
            "success": False,
            "path": self.name,
            "bug_id": bug_data.get("id", "unknown"),
            "strategy_results": [],
            "execution_time": 0,
            "error": None
        }
        
        start_time = context.get("start_time", 0)
        
        # Try primary strategies first
        for strategy in self.primary_strategies:
            strategy_result = strategy.execute(bug_data, context)
            result["strategy_results"].append(strategy_result)
            
            if strategy_result.get("success", False):
                result["success"] = True
                result["primary_strategy_succeeded"] = True
                break
        
        # If primary strategies failed, try fallbacks
        if not result["success"] and self.fallback_strategies:
            for strategy in self.fallback_strategies:
                strategy_result = strategy.execute(bug_data, context)
                result["strategy_results"].append(strategy_result)
                
                if strategy_result.get("success", False):
                    result["success"] = True
                    result["fallback_strategy_succeeded"] = True
                    break
        
        # Record execution for learning
        self.record_execution(bug_data, context, result)
        
        return result
    
    def record_execution(self, bug_data: Dict[str, Any], context: Dict[str, Any], 
                         result: Dict[str, Any]) -> None:
        """
        Record the execution of this path for learning.
        
        Args:
            bug_data: Data about the bug
            context: Context of the execution
            result: Result of the execution
        """
        execution_record = {
            "bug_id": bug_data.get("id", "unknown"),
            "bug_type": bug_data.get("type", "unknown"),
            "language": bug_data.get("language", "unknown"),
            "success": result.get("success", False),
            "primary_succeeded": result.get("primary_strategy_succeeded", False),
            "fallback_succeeded": result.get("fallback_strategy_succeeded", False),
            "strategies_attempted": len(result.get("strategy_results", [])),
            "timestamp": context.get("timestamp", 0)
        }
        
        self.execution_history.append(execution_record)
        
        # Limit history size
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def calculate_score(self, bug_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Calculate a score for this path based on the bug data and context.
        
        Args:
            bug_data: Data about the bug
            context: Additional context
            
        Returns:
            Score for this path (higher is better)
        """
        # Calculate scores for all strategies
        primary_scores = [
            strategy.calculate_score(bug_data, context)
            for strategy in self.primary_strategies
        ]
        
        fallback_scores = [
            strategy.calculate_score(bug_data, context)
            for strategy in self.fallback_strategies
        ]
        
        # Base score is the average of primary strategy scores
        if primary_scores:
            score = sum(primary_scores) / len(primary_scores)
        else:
            score = 0.0
        
        # Boost score if we have fallbacks
        if fallback_scores:
            fallback_boost = 0.2 * (sum(fallback_scores) / len(fallback_scores))
            score += fallback_boost
        
        # Adjust for historical performance
        if self.execution_history:
            # Calculate success rate for similar bugs
            similar_bugs = [
                record for record in self.execution_history
                if record["bug_type"] == bug_data.get("type", "unknown") and 
                record["language"] == bug_data.get("language", "unknown")
            ]
            
            if similar_bugs:
                success_rate = sum(1 for record in similar_bugs if record["success"]) / len(similar_bugs)
                score *= (0.5 + 0.5 * success_rate)
        
        return score

class SolutionPathGenerator:
    """
    Generates solution paths for a given bug.
    
    This class creates new solution paths based on bug characteristics and available strategies.
    """
    
    def __init__(self):
        """
        Initialize the solution path generator.
        """
        self.strategy_templates = {}
        self.neural_matrix = None  # Will be set externally
        
        logger.info("Initialized solution path generator")
    
    def register_strategy_template(self, strategy_type: str, template: Dict[str, Any]) -> None:
        """
        Register a strategy template.
        
        Args:
            strategy_type: Type of strategy
            template: Template for the strategy
        """
        self.strategy_templates[strategy_type] = template
        logger.info(f"Registered strategy template: {strategy_type}")
    
    def generate_paths(self, bug_data: Dict[str, Any], context: Dict[str, Any], 
                       count: int = 3) -> List[SolutionPath]:
        """
        Generate solution paths for a given bug.
        
        Args:
            bug_data: Data about the bug
            context: Additional context
            count: Number of paths to generate
            
        Returns:
            List of generated solution paths
        """
        logger.info(f"Generating {count} solution paths for bug {bug_data.get('id', 'unknown')}")
        
        # Extract bug attributes for path generation
        bug_type = bug_data.get("type", "unknown")
        language = bug_data.get("language", "unknown")
        complexity = bug_data.get("complexity", 5)
        
        # Generate paths
        paths = []
        for i in range(count):
            # Create a path with a unique name
            path_name = f"{bug_type}_path_{i}"
            path = SolutionPath(
                name=path_name,
                description=f"Auto-generated path for {bug_type} bug in {language}"
            )
            
            # Add strategies based on bug characteristics
            self._add_strategies_to_path(path, bug_data, context)
            
            paths.append(path)
        
        return paths
    
    def _add_strategies_to_path(self, path: SolutionPath, bug_data: Dict[str, Any], 
                               context: Dict[str, Any]) -> None:
        """
        Add appropriate strategies to a path based on bug characteristics.
        
        Args:
            path: Path to add strategies to
            bug_data: Data about the bug
            context: Additional context
        """
        # This would be more sophisticated in a real implementation,
        # leveraging neural matrix insights and strategy templates.
        # For now, we'll use a simple rule-based approach.
        
        bug_type = bug_data.get("type", "unknown")
        complexity = bug_data.get("complexity", 5)
        
        # Add primary strategies
        if bug_type == "syntax_error":
            # Add syntax fix strategies
            if complexity <= 3:
                path.add_primary_strategy(
                    SolutionStrategy("SYNTAX_FIX", "Simple Syntax Fix", 
                                    "Fix simple syntax errors", 1, 0.95)
                )
            else:
                path.add_primary_strategy(
                    SolutionStrategy("SYNTAX_FIX", "Complex Syntax Fix", 
                                    "Fix complex syntax errors", 5, 0.75)
                )
        elif bug_type == "logic_error":
            # Add logic fix strategies
            path.add_primary_strategy(
                SolutionStrategy("LOGIC_FIX", "Conditional Logic Fix", 
                                "Fix issues with conditionals", 3, 0.85)
            )
            path.add_primary_strategy(
                SolutionStrategy("LOGIC_FIX", "Data Flow Fix", 
                                "Fix issues with data flow", 4, 0.8)
            )
        elif bug_type == "performance_issue":
            # Add performance strategies
            path.add_primary_strategy(
                SolutionStrategy("PERFORMANCE_FIX", "Algorithm Optimization", 
                                "Optimize algorithms", 6, 0.7)
            )
        else:
            # Generic approach for unknown bug types
            path.add_primary_strategy(
                SolutionStrategy("BEST_PRACTICE_FIX", "Code Improvement", 
                                "Improve code quality", 3, 0.8)
            )
        
        # Add fallback strategies
        path.add_fallback_strategy(
            SolutionStrategy("REFACTOR", "Simple Refactoring", 
                            "Refactor code for clarity", 2, 0.9)
        )
        
        # Add more fallbacks for complex bugs
        if complexity > 5:
            path.add_fallback_strategy(
                SolutionStrategy("ERROR_HANDLING_FIX", "Robust Error Handling", 
                                "Add comprehensive error handling", 4, 0.8)
            )

class SolutionPathSelector:
    """
    Selects the best solution path for a given bug.
    """
    
    def __init__(self):
        """
        Initialize the solution path selector.
        """
        self.solution_paths = []
        self.neural_matrix = None  # Will be set externally
        
        logger.info("Initialized solution path selector")
    
    def register_path(self, path: SolutionPath) -> None:
        """
        Register a solution path.
        
        Args:
            path: Solution path to register
        """
        self.solution_paths.append(path)
        logger.info(f"Registered solution path: {path.name}")
    
    def select_path(self, bug_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[SolutionPath]:
        """
        Select the best solution path for a given bug.
        
        Args:
            bug_data: Data about the bug
            context: Additional context
            
        Returns:
            Selected solution path, or None if no paths are available
        """
        if not self.solution_paths:
            logger.warning("No solution paths available for selection")
            return None
        
        # Calculate scores for all paths
        path_scores = []
        for path in self.solution_paths:
            score = path.calculate_score(bug_data, context)
            
            # Apply neural guidance if available
            if self.neural_matrix:
                neural_score = self.get_neural_score(path, bug_data, context)
                score = 0.7 * score + 0.3 * neural_score
            
            path_scores.append((path, score))
        
        # Sort by score (highest first)
        path_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Log scores for debugging
        for path, score in path_scores:
            logger.debug(f"Path {path.name}: score {score:.4f}")
        
        # Return the path with the highest score
        selected_path = path_scores[0][0]
        logger.info(f"Selected solution path: {selected_path.name} (score: {path_scores[0][1]:.4f})")
        
        return selected_path
    
    def get_neural_score(self, path: SolutionPath, bug_data: Dict[str, Any], 
                         context: Dict[str, Any]) -> float:
        """
        Get a score from the neural matrix.
        
        Args:
            path: Solution path to score
            bug_data: Data about the bug
            context: Additional context
            
        Returns:
            Neural score (0.0-1.0)
        """
        try:
            # This is a mock implementation - in practice, this would call the actual neural matrix
            if self.neural_matrix:
                # Create feature vector for neural scoring
                features = {
                    "path_name": path.name,
                    "bug_type": bug_data.get("type", "unknown"),
                    "language": bug_data.get("language", "unknown"),
                    "file_size": bug_data.get("file_size", 0),
                    "complexity": bug_data.get("complexity", 0),
                    "num_primary_strategies": len(path.primary_strategies),
                    "num_fallback_strategies": len(path.fallback_strategies)
                }
                
                # In a real implementation, this would pass the features to the neural matrix
                # and get a score back. For now, we'll just return a random score.
                return random.random()
            else:
                return 0.5  # Default score if no neural matrix is available
        except Exception as e:
            logger.error(f"Error getting neural score: {e}")
            return 0.5  # Default score on error

class SolutionManager:
    """
    Manages solution strategies and paths.
    """
    
    def __init__(self):
        """
        Initialize the solution manager.
        """
        self.strategies = {}
        self.paths = {}
        self.selector = SolutionPathSelector()
        
        logger.info("Initialized solution manager")
        
        # Create default strategies and paths
        self._create_default_strategies()
        self._create_default_paths()
    
    def _create_default_strategies(self) -> None:
        """
        Create default solution strategies.
        """
        # Syntax error strategies
        self.register_strategy(SolutionStrategy(
            "SYNTAX_FIX", "Simple Syntax Fix", 
            "Fix simple syntax errors like missing parentheses, brackets, etc.",
            1, 0.95
        ))
        
        self.register_strategy(SolutionStrategy(
            "SYNTAX_FIX", "Complex Syntax Fix", 
            "Fix complex syntax errors that may span multiple lines or blocks",
            4, 0.8
        ))
        
        # Logic error strategies
        self.register_strategy(SolutionStrategy(
            "LOGIC_FIX", "Conditional Logic Fix", 
            "Fix issues with if/else statements, boolean expressions, etc.",
            3, 0.85
        ))
        
        self.register_strategy(SolutionStrategy(
            "LOGIC_FIX", "Loop Logic Fix", 
            "Fix issues with loops, including infinite loops, off-by-one errors, etc.",
            4, 0.8
        ))
        
        self.register_strategy(SolutionStrategy(
            "LOGIC_FIX", "Data Processing Fix", 
            "Fix issues with data processing, transformations, etc.",
            5, 0.75
        ))
        
        # Validation strategies
        self.register_strategy(SolutionStrategy(
            "VALIDATION_FIX", "Input Parameter Validation", 
            "Add validation for function/method parameters",
            2, 0.9
        ))
        
        self.register_strategy(SolutionStrategy(
            "VALIDATION_FIX", "User Input Validation", 
            "Add validation for user input data",
            3, 0.85
        ))
        
        self.register_strategy(SolutionStrategy(
            "VALIDATION_FIX", "Complex Data Validation", 
            "Add validation for complex data structures or objects",
            5, 0.75
        ))
        
        # Error handling strategies
        self.register_strategy(SolutionStrategy(
            "ERROR_HANDLING_FIX", "Basic Exception Handling", 
            "Add basic try/catch or try/except blocks",
            2, 0.9
        ))
        
        self.register_strategy(SolutionStrategy(
            "ERROR_HANDLING_FIX", "Advanced Exception Handling", 
            "Add detailed error handling with specific exception types and recovery logic",
            5, 0.75
        ))
        
        # Performance strategies
        self.register_strategy(SolutionStrategy(
            "PERFORMANCE_FIX", "Algorithm Optimization", 
            "Optimize inefficient algorithms or data structures",
            7, 0.65
        ))
        
        self.register_strategy(SolutionStrategy(
            "PERFORMANCE_FIX", "Resource Usage Optimization", 
            "Optimize resource usage (memory, CPU, I/O, etc.)",
            6, 0.7
        ))
        
        # Security strategies
        self.register_strategy(SolutionStrategy(
            "SECURITY_FIX", "Input Sanitization", 
            "Add sanitization for user input to prevent injection attacks",
            3, 0.85
        ))
        
        self.register_strategy(SolutionStrategy(
            "SECURITY_FIX", "Authentication Fix", 
            "Fix issues with authentication or session management",
            6, 0.7
        ))
        
        self.register_strategy(SolutionStrategy(
            "SECURITY_FIX", "Encryption Fix", 
            "Fix issues with encryption, hashing, or secure communications",
            8, 0.6
        ))
        
        # Refactoring strategies
        self.register_strategy(SolutionStrategy(
            "REFACTOR", "Simple Refactoring", 
            "Rename variables, extract methods, etc.",
            2, 0.9
        ))
        
        self.register_strategy(SolutionStrategy(
            "REFACTOR", "Complex Refactoring", 
            "Restructure code organization, design patterns, etc.",
            7, 0.65
        ))
        
        # Best practices strategies
        self.register_strategy(SolutionStrategy(
            "BEST_PRACTICE_FIX", "Code Style Fix", 
            "Fix code style issues like inconsistent naming, formatting, etc.",
            1, 0.95
        ))
        
        self.register_strategy(SolutionStrategy(
            "BEST_PRACTICE_FIX", "Documentation Fix", 
            "Add or improve code documentation",
            2, 0.9
        ))
        
        # Test fixing strategies
        self.register_strategy(SolutionStrategy(
            "TEST_FIX", "Test Case Fix", 
            "Fix failing test cases",
            3, 0.85
        ))
        
        self.register_strategy(SolutionStrategy(
            "TEST_FIX", "Test Coverage Fix", 
            "Add test cases to improve coverage",
            4, 0.8
        ))
    
    def _create_default_paths(self) -> None:
        """
        Create default solution paths.
        """
        # Syntax error path
        syntax_path = SolutionPath(
            "Syntax Error Path", 
            "Path for fixing syntax errors"
        )
        syntax_path.add_primary_strategy(self.strategies["SYNTAX_FIX"]["Simple Syntax Fix"])
        syntax_path.add_fallback_strategy(self.strategies["SYNTAX_FIX"]["Complex Syntax Fix"])
        self.register_path(syntax_path)
        
        # Logic error path
        logic_path = SolutionPath(
            "Logic Error Path", 
            "Path for fixing logical errors"
        )
        logic_path.add_primary_strategy(self.strategies["LOGIC_FIX"]["Conditional Logic Fix"])
        logic_path.add_primary_strategy(self.strategies["LOGIC_FIX"]["Loop Logic Fix"])
        logic_path.add_fallback_strategy(self.strategies["LOGIC_FIX"]["Data Processing Fix"])
        self.register_path(logic_path)
        
        # Validation path
        validation_path = SolutionPath(
            "Validation Path", 
            "Path for adding input validation"
        )
        validation_path.add_primary_strategy(self.strategies["VALIDATION_FIX"]["Input Parameter Validation"])
        validation_path.add_fallback_strategy(self.strategies["VALIDATION_FIX"]["User Input Validation"])
        validation_path.add_fallback_strategy(self.strategies["VALIDATION_FIX"]["Complex Data Validation"])
        self.register_path(validation_path)
        
        # Error handling path
        error_handling_path = SolutionPath(
            "Error Handling Path", 
            "Path for improving error handling"
        )
        error_handling_path.add_primary_strategy(self.strategies["ERROR_HANDLING_FIX"]["Basic Exception Handling"])
        error_handling_path.add_fallback_strategy(self.strategies["ERROR_HANDLING_FIX"]["Advanced Exception Handling"])
        self.register_path(error_handling_path)
        
        # Performance path
        performance_path = SolutionPath(
            "Performance Path", 
            "Path for optimizing performance"
        )
        performance_path.add_primary_strategy(self.strategies["PERFORMANCE_FIX"]["Algorithm Optimization"])
        performance_path.add_fallback_strategy(self.strategies["PERFORMANCE_FIX"]["Resource Usage Optimization"])
        self.register_path(performance_path)
        
        # Security path
        security_path = SolutionPath(
            "Security Path", 
            "Path for fixing security issues"
        )
        security_path.add_primary_strategy(self.strategies["SECURITY_FIX"]["Input Sanitization"])
        security_path.add_fallback_strategy(self.strategies["SECURITY_FIX"]["Authentication Fix"])
        security_path.add_fallback_strategy(self.strategies["SECURITY_FIX"]["Encryption Fix"])
        self.register_path(security_path)
        
        # Refactoring path
        refactoring_path = SolutionPath(
            "Refactoring Path", 
            "Path for refactoring code"
        )
        refactoring_path.add_primary_strategy(self.strategies["REFACTOR"]["Simple Refactoring"])
        refactoring_path.add_fallback_strategy(self.strategies["REFACTOR"]["Complex Refactoring"])
        self.register_path(refactoring_path)
        
        # Best practices path
        best_practices_path = SolutionPath(
            "Best Practices Path", 
            "Path for applying best practices"
        )
        best_practices_path.add_primary_strategy(self.strategies["BEST_PRACTICE_FIX"]["Code Style Fix"])
        best_practices_path.add_fallback_strategy(self.strategies["BEST_PRACTICE_FIX"]["Documentation Fix"])
        self.register_path(best_practices_path)
        
        # Test fixing path
        test_path = SolutionPath(
            "Test Fixing Path", 
            "Path for fixing tests"
        )
        test_path.add_primary_strategy(self.strategies["TEST_FIX"]["Test Case Fix"])
        test_path.add_fallback_strategy(self.strategies["TEST_FIX"]["Test Coverage Fix"])
        self.register_path(test_path)
    
    def register_strategy(self, strategy: SolutionStrategy) -> None:
        """
        Register a solution strategy.
        
        Args:
            strategy: Strategy to register
        """
        if strategy.strategy_type not in self.strategies:
            self.strategies[strategy.strategy_type] = {}
        
        self.strategies[strategy.strategy_type][strategy.name] = strategy
        logger.info(f"Registered strategy: {strategy.name} ({strategy.strategy_type})")
    
    def register_path(self, path: SolutionPath) -> None:
        """
        Register a solution path.
        
        Args:
            path: Path to register
        """
        self.paths[path.name] = path
        self.selector.register_path(path)
        logger.info(f"Registered path: {path.name}")
    
    def get_strategy(self, strategy_type: str, name: str) -> Optional[SolutionStrategy]:
        """
        Get a strategy by type and name.
        
        Args:
            strategy_type: Type of strategy
            name: Name of strategy
            
        Returns:
            Strategy object, or None if not found
        """
        if strategy_type in self.strategies and name in self.strategies[strategy_type]:
            return self.strategies[strategy_type][name]
        
        return None
    
    def get_path(self, name: str) -> Optional[SolutionPath]:
        """
        Get a path by name.
        
        Args:
            name: Name of path
            
        Returns:
            Path object, or None if not found
        """
        return self.paths.get(name)
    
    def select_path(self, bug_data: Dict[str, Any], context: Dict[str, Any]) -> Optional[SolutionPath]:
        """
        Select the best solution path for a given bug.
        
        Args:
            bug_data: Data about the bug
            context: Additional context
            
        Returns:
            Selected solution path, or None if no paths are available
        """
        return self.selector.select_path(bug_data, context)
    
    def set_neural_matrix(self, neural_matrix: Any) -> None:
        """
        Set the neural matrix for neural-guided path selection.
        
        Args:
            neural_matrix: Neural matrix object
        """
        self.selector.neural_matrix = neural_matrix
        logger.info("Set neural matrix for solution path selector")
    
    def execute_solution(self, bug_data: Dict[str, Any], context: Dict[str, Any], 
                         path_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a solution for a bug.
        
        Args:
            bug_data: Data about the bug
            context: Additional context
            path_name: Name of path to use, or None to select automatically
            
        Returns:
            Result of the execution
        """
        logger.info(f"Executing solution for bug {bug_data.get('id', 'unknown')}")
        
        # Get the path to use
        if path_name:
            path = self.get_path(path_name)
            if not path:
                logger.error(f"Path not found: {path_name}")
                return {
                    "success": False,
                    "error": f"Path not found: {path_name}"
                }
        else:
            path = self.select_path(bug_data, context)
            if not path:
                logger.error("No suitable path found")
                return {
                    "success": False,
                    "error": "No suitable path found"
                }
        
        # Execute the path
        result = path.execute(bug_data, context)
        
        return result

# API Functions

def create_solution_manager() -> SolutionManager:
    """
    Create and initialize a solution manager.
    
    Returns:
        Initialized solution manager
    """
    manager = SolutionManager()
    return manager

def select_solution_path(bug_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Select a solution path for a bug.
    
    Args:
        bug_data: Data about the bug
        context: Additional context
        
    Returns:
        Result with selected path
    """
    if context is None:
        context = {}
    
    manager = create_solution_manager()
    
    path = manager.select_path(bug_data, context)
    
    if path:
        return {
            "success": True,
            "path": path.name,
            "description": path.description,
            "primary_strategies": [s.name for s in path.primary_strategies],
            "fallback_strategies": [s.name for s in path.fallback_strategies]
        }
    else:
        return {
            "success": False,
            "error": "No suitable path found"
        }

def execute_solution_path(bug_data: Dict[str, Any], path_name: str = None, 
                         context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute a solution path for a bug.
    
    Args:
        bug_data: Data about the bug
        path_name: Name of path to use, or None to select automatically
        context: Additional context
        
    Returns:
        Result of the execution
    """
    if context is None:
        context = {}
    
    # Add timestamp
    context["timestamp"] = context.get("timestamp", 0) or int(time.time())
    
    manager = create_solution_manager()
    
    return manager.execute_solution(bug_data, context, path_name)


if __name__ == "__main__":
    import argparse
    import time
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Solution Paths Tool")
    parser.add_argument("--bug-file", help="JSON file with bug data")
    parser.add_argument("--path", help="Solution path to use")
    parser.add_argument("--list-paths", action="store_true", help="List available solution paths")
    parser.add_argument("--list-strategies", action="store_true", help="List available solution strategies")
    
    args = parser.parse_args()
    
    # Create solution manager
    manager = create_solution_manager()
    
    # List paths
    if args.list_paths:
        print("\nAvailable Solution Paths:")
        for name, path in manager.paths.items():
            print(f"\n{name}: {path.description}")
            print("  Primary strategies:")
            for strategy in path.primary_strategies:
                print(f"    - {strategy.name}: {strategy.description}")
            
            print("  Fallback strategies:")
            for strategy in path.fallback_strategies:
                print(f"    - {strategy.name}: {strategy.description}")
    
    # List strategies
    elif args.list_strategies:
        print("\nAvailable Solution Strategies:")
        for strategy_type, strategies in manager.strategies.items():
            print(f"\n{strategy_type}:")
            for name, strategy in strategies.items():
                print(f"  {name}: {strategy.description}")
                print(f"    Complexity: {strategy.complexity}, Success probability: {strategy.success_probability:.2f}")
    
    # Execute solution
    elif args.bug_file:
        try:
            with open(args.bug_file, 'r') as f:
                bug_data = json.load(f)
            
            context = {
                "timestamp": int(time.time()),
                "command_line": True
            }
            
            result = manager.execute_solution(bug_data, context, args.path)
            
            print("\nExecution Result:")
            print(f"Success: {result.get('success', False)}")
            
            if result.get("success", False):
                print(f"Path: {result.get('path', 'Unknown')}")
                
                if result.get("primary_strategy_succeeded", False):
                    print("Primary strategy succeeded")
                elif result.get("fallback_strategy_succeeded", False):
                    print("Fallback strategy succeeded")
                
                print("\nStrategy Results:")
                for i, strategy_result in enumerate(result.get("strategy_results", []), 1):
                    print(f"\n  Strategy {i}: {strategy_result.get('strategy', 'Unknown')}")
                    print(f"    Success: {strategy_result.get('success', False)}")
                    print(f"    Fixes applied: {', '.join(strategy_result.get('fixes_applied', []))}")
                    
                    if not strategy_result.get("success", False):
                        print(f"    Error: {strategy_result.get('error', 'Unknown')}")
            else:
                print(f"Error: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        parser.print_help()
