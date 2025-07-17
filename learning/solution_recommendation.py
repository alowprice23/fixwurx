#!/usr/bin/env python3
"""
Solution Recommendation Module

This module provides solution recommendation capabilities, suggesting fixes
for bugs based on historical data and pattern recognition.
"""

import os
import sys
import json
import logging
import time
import re
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pathlib import Path
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("solution_recommendation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SolutionRecommendation")

class Solution:
    """
    Represents a solution to a bug.
    """
    
    def __init__(self, solution_id: str, solution_type: str, pattern_id: str = None,
               description: str = None, code: str = None, confidence: float = 0.0,
               application_count: int = 0, success_count: int = 0):
        """
        Initialize solution.
        
        Args:
            solution_id: Unique solution ID
            solution_type: Type of solution
            pattern_id: ID of associated bug pattern
            description: Description of the solution
            code: Solution code or patch
            confidence: Confidence score (0.0 to 1.0)
            application_count: Number of times this solution was applied
            success_count: Number of successful applications
        """
        self.solution_id = solution_id
        self.solution_type = solution_type
        self.pattern_id = pattern_id
        self.description = description or ""
        self.code = code or ""
        self.confidence = confidence
        self.application_count = application_count
        self.success_count = success_count
        self.creation_time = time.time()
        self.last_updated = time.time()
        self.related_solutions = []
        self.examples = []
        
        logger.debug(f"Created solution: {solution_id} ({solution_type})")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "solution_id": self.solution_id,
            "solution_type": self.solution_type,
            "pattern_id": self.pattern_id,
            "description": self.description,
            "code": self.code,
            "confidence": self.confidence,
            "application_count": self.application_count,
            "success_count": self.success_count,
            "creation_time": self.creation_time,
            "last_updated": self.last_updated,
            "related_solutions": self.related_solutions,
            "examples": self.examples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Solution':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Solution
        """
        solution = cls(
            solution_id=data.get("solution_id", ""),
            solution_type=data.get("solution_type", ""),
            pattern_id=data.get("pattern_id"),
            description=data.get("description", ""),
            code=data.get("code", ""),
            confidence=data.get("confidence", 0.0),
            application_count=data.get("application_count", 0),
            success_count=data.get("success_count", 0)
        )
        
        # Restore timestamps and related solutions
        solution.creation_time = data.get("creation_time", time.time())
        solution.last_updated = data.get("last_updated", time.time())
        solution.related_solutions = data.get("related_solutions", [])
        solution.examples = data.get("examples", [])
        
        return solution
    
    def record_application(self, success: bool, context: Dict[str, Any] = None) -> None:
        """
        Record an application of this solution.
        
        Args:
            success: Whether the application was successful
            context: Application context
        """
        self.application_count += 1
        
        if success:
            self.success_count += 1
        
        self.last_updated = time.time()
        
        # Update confidence
        if self.application_count > 0:
            self.confidence = self.success_count / self.application_count
        
        # Add to examples
        if context:
            example = {
                "timestamp": time.time(),
                "success": success,
                "context": context
            }
            
            self.examples.append(example)
            
            # Keep only the latest 10 examples to avoid excessive memory usage
            if len(self.examples) > 10:
                self.examples = self.examples[-10:]
        
        logger.debug(f"Recorded application for solution {self.solution_id} (success: {success})")
    
    def relate_to_solution(self, solution_id: str) -> None:
        """
        Relate this solution to another solution.
        
        Args:
            solution_id: ID of related solution
        """
        if solution_id not in self.related_solutions:
            self.related_solutions.append(solution_id)
            self.last_updated = time.time()
            
            logger.debug(f"Related solution {self.solution_id} to {solution_id}")

class SolutionRecommender:
    """
    Recommends solutions to bugs.
    """
    
    def __init__(self, solutions_file: str = None, min_confidence: float = 0.6):
        """
        Initialize solution recommender.
        
        Args:
            solutions_file: Path to solutions database file
            min_confidence: Minimum confidence threshold for recommendations
        """
        self.solutions_file = solutions_file or "bug_solutions.json"
        self.solutions = {}
        self.solution_types = set()
        self.pattern_solutions = {}
        self.min_confidence = min_confidence
        
        # Load solutions if file exists
        if os.path.exists(self.solutions_file):
            self._load_solutions()
        
        logger.info("Solution recommender initialized")
    
    def _load_solutions(self) -> None:
        """Load solutions from database file."""
        try:
            with open(self.solutions_file, "r") as f:
                data = json.load(f)
            
            # Load solutions
            solutions_data = data.get("solutions", {})
            
            for solution_id, solution_data in solutions_data.items():
                self.solutions[solution_id] = Solution.from_dict(solution_data)
            
            # Extract solution types
            self.solution_types = set(solution.solution_type for solution in self.solutions.values())
            
            # Build pattern -> solutions mapping
            self.pattern_solutions = {}
            for solution in self.solutions.values():
                if solution.pattern_id:
                    if solution.pattern_id not in self.pattern_solutions:
                        self.pattern_solutions[solution.pattern_id] = []
                    
                    self.pattern_solutions[solution.pattern_id].append(solution.solution_id)
            
            # Load configuration
            self.min_confidence = data.get("min_confidence", 0.6)
            
            logger.info(f"Loaded {len(self.solutions)} solutions from {self.solutions_file}")
        except Exception as e:
            logger.error(f"Error loading solutions from {self.solutions_file}: {e}")
            self.solutions = {}
            self.solution_types = set()
            self.pattern_solutions = {}
    
    def save_solutions(self) -> None:
        """Save solutions to database file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.solutions_file)), exist_ok=True)
            
            data = {
                "solutions": {solution_id: solution.to_dict() for solution_id, solution in self.solutions.items()},
                "min_confidence": self.min_confidence,
                "last_updated": time.time()
            }
            
            with open(self.solutions_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.solutions)} solutions to {self.solutions_file}")
        except Exception as e:
            logger.error(f"Error saving solutions to {self.solutions_file}: {e}")
    
    def add_solution(self, solution: Solution) -> None:
        """
        Add a solution.
        
        Args:
            solution: Solution to add
        """
        self.solutions[solution.solution_id] = solution
        
        # Update solution types
        self.solution_types.add(solution.solution_type)
        
        # Update pattern -> solutions mapping
        if solution.pattern_id:
            if solution.pattern_id not in self.pattern_solutions:
                self.pattern_solutions[solution.pattern_id] = []
            
            if solution.solution_id not in self.pattern_solutions[solution.pattern_id]:
                self.pattern_solutions[solution.pattern_id].append(solution.solution_id)
        
        # Save solutions
        self.save_solutions()
        
        logger.info(f"Added solution: {solution.solution_id}")
    
    def create_solution(self, solution_type: str, pattern_id: str = None,
                      description: str = None, code: str = None) -> Solution:
        """
        Create a new solution.
        
        Args:
            solution_type: Type of solution
            pattern_id: ID of associated bug pattern
            description: Description of the solution
            code: Solution code or patch
            
        Returns:
            Created solution
        """
        # Generate solution ID
        solution_id = f"solution_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create solution
        solution = Solution(
            solution_id=solution_id,
            solution_type=solution_type,
            pattern_id=pattern_id,
            description=description,
            code=code,
            confidence=0.5  # Initial confidence
        )
        
        # Add to solutions
        self.add_solution(solution)
        
        return solution
    
    def get_solution(self, solution_id: str) -> Optional[Solution]:
        """
        Get a solution by ID.
        
        Args:
            solution_id: Solution ID
            
        Returns:
            Solution, or None if not found
        """
        return self.solutions.get(solution_id)
    
    def get_solutions_by_type(self, solution_type: str) -> List[Solution]:
        """
        Get solutions by type.
        
        Args:
            solution_type: Solution type
            
        Returns:
            List of matching solutions
        """
        return [s for s in self.solutions.values() if s.solution_type == solution_type]
    
    def get_solutions_for_pattern(self, pattern_id: str) -> List[Solution]:
        """
        Get solutions for a pattern.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            List of solutions for the pattern
        """
        solution_ids = self.pattern_solutions.get(pattern_id, [])
        return [self.solutions[sid] for sid in solution_ids if sid in self.solutions]
    
    def recommend_solutions(self, pattern_id: str, context: Dict[str, Any] = None,
                          max_solutions: int = 3) -> List[Solution]:
        """
        Recommend solutions for a bug pattern.
        
        Args:
            pattern_id: Pattern ID
            context: Bug context
            max_solutions: Maximum number of solutions to recommend
            
        Returns:
            List of recommended solutions
        """
        # Get solutions for pattern
        solutions = self.get_solutions_for_pattern(pattern_id)
        
        if not solutions:
            logger.warning(f"No solutions found for pattern: {pattern_id}")
            return []
        
        # Filter by confidence
        solutions = [s for s in solutions if s.confidence >= self.min_confidence]
        
        if not solutions:
            logger.warning(f"No solutions with confidence >= {self.min_confidence} for pattern: {pattern_id}")
            return []
        
        # Sort by confidence (descending)
        solutions.sort(key=lambda s: s.confidence, reverse=True)
        
        # Return top solutions
        return solutions[:max_solutions]
    
    def recommend_solution_types(self, error_type: str) -> List[str]:
        """
        Recommend solution types for an error type.
        
        Args:
            error_type: Type of error
            
        Returns:
            List of recommended solution types
        """
        # Map error types to solution types
        solution_type_map = {
            "type_error": ["type_check", "type_cast", "parameter_validation"],
            "attribute_error": ["null_check", "object_validation", "property_initialization"],
            "index_error": ["bounds_check", "length_validation", "default_value"],
            "key_error": ["key_check", "default_dict", "exception_handling"],
            "name_error": ["import_fix", "variable_declaration", "scope_correction"],
            "syntax_error": ["syntax_correction", "code_reformatting", "parser_error_fix"],
            "import_error": ["dependency_installation", "import_path_correction", "module_rename"],
            "io_error": ["file_handling", "exception_handling", "resource_management"],
            "permission_error": ["permission_request", "privilege_escalation", "alternative_resource"],
            "value_error": ["input_validation", "format_correction", "range_check"],
            "zero_division_error": ["division_check", "conditional_execution", "default_value"],
            "memory_error": ["memory_optimization", "resource_limiting", "garbage_collection"],
            "recursion_error": ["base_case_fix", "iteration_conversion", "depth_limiting"],
            "timeout_error": ["timeout_increase", "performance_optimization", "async_conversion"],
            "runtime_error": ["exception_handling", "error_logging", "defensive_programming"],
            "assertion_error": ["condition_correction", "test_update", "assumption_validation"]
        }
        
        # Get recommended solution types
        recommended_types = solution_type_map.get(error_type, ["exception_handling", "defensive_programming"])
        
        return recommended_types
    
    def generate_solution_template(self, solution_type: str, error_message: str = None,
                                 buggy_code: str = None) -> str:
        """
        Generate a solution template based on solution type.
        
        Args:
            solution_type: Type of solution
            error_message: Error message
            buggy_code: Buggy code
            
        Returns:
            Solution template
        """
        # Template for null check
        if solution_type == "null_check":
            return """
# Add null/None check before accessing attribute
if object_name is not None:
    result = object_name.attribute_name
else:
    # Handle the None case
    result = default_value
"""
        
        # Template for bounds check
        elif solution_type == "bounds_check":
            return """
# Add bounds check before accessing index
if 0 <= index < len(array_name):
    result = array_name[index]
else:
    # Handle the out-of-bounds case
    result = default_value
"""
        
        # Template for key check
        elif solution_type == "key_check":
            return """
# Add key check before accessing dictionary
if key_name in dict_name:
    result = dict_name[key_name]
else:
    # Handle the missing key case
    result = default_value
"""
        
        # Template for type check
        elif solution_type == "type_check":
            return """
# Add type check before operation
if isinstance(variable_name, expected_type):
    result = operation_with_variable
else:
    # Handle the unexpected type case
    result = default_value
"""
        
        # Template for exception handling
        elif solution_type == "exception_handling":
            return """
try:
    # Risky operation
    result = risky_operation()
except ExceptionType as e:
    # Handle the exception
    logger.error(f"Error: {e}")
    result = default_value
"""
        
        # Template for input validation
        elif solution_type == "input_validation":
            return """
# Validate input before processing
if is_valid_input(input_value):
    result = process_input(input_value)
else:
    # Handle invalid input
    raise ValueError(f"Invalid input: {input_value}")
"""
        
        # Template for division check
        elif solution_type == "division_check":
            return """
# Check for zero before division
if divisor != 0:
    result = dividend / divisor
else:
    # Handle division by zero
    result = default_value
"""
        
        # Template for performance optimization
        elif solution_type == "performance_optimization":
            return """
# Optimize performance by caching results
if result not in cache:
    cache[key] = expensive_computation()
return cache[key]
"""
        
        # Template for resource management
        elif solution_type == "resource_management":
            return """
# Use context manager for proper resource handling
with open(file_path, 'r') as file:
    data = file.read()
# File is automatically closed after the block
"""
        
        # Template for defensive programming
        elif solution_type == "defensive_programming":
            return """
# Add defensive checks
assert precondition, "Precondition failed"
result = operation()
assert postcondition, "Postcondition failed"
return result
"""
        
        # Default template
        else:
            return """
# TODO: Implement solution
# 1. Identify the root cause
# 2. Apply the appropriate fix
# 3. Add validation to prevent similar issues
"""
    
    def learn_from_fix(self, pattern_id: str, buggy_code: str, fixed_code: str,
                     error_message: str = None, solution_type: str = None,
                     context: Dict[str, Any] = None) -> Solution:
        """
        Learn from a successful bug fix.
        
        Args:
            pattern_id: Pattern ID
            buggy_code: Original buggy code
            fixed_code: Fixed code
            error_message: Error message
            solution_type: Type of solution
            context: Additional context
            
        Returns:
            Created or updated solution
        """
        # If solution type is not provided, try to infer it
        if solution_type is None:
            if error_message:
                error_type = self._infer_error_type(error_message)
                solution_types = self.recommend_solution_types(error_type)
                solution_type = solution_types[0] if solution_types else "unknown"
            else:
                solution_type = "unknown"
        
        # Generate diff between buggy and fixed code
        diff = self._generate_diff(buggy_code, fixed_code)
        
        # Check if similar solutions exist
        existing_solutions = self.get_solutions_for_pattern(pattern_id)
        best_match = None
        best_similarity = 0.0
        
        for solution in existing_solutions:
            if solution.solution_type == solution_type:
                similarity = self._calculate_similarity(diff, solution.code)
                
                if similarity > best_similarity and similarity >= 0.8:
                    best_similarity = similarity
                    best_match = solution
        
        if best_match:
            # Update existing solution
            best_match.record_application(True, context)
            self.save_solutions()
            
            logger.info(f"Updated existing solution: {best_match.solution_id} (similarity: {best_similarity:.2f})")
            return best_match
        
        # Create new solution
        description = f"Fix for {solution_type} issue"
        
        if error_message:
            description += f": {error_message}"
        
        new_solution = self.create_solution(
            solution_type=solution_type,
            pattern_id=pattern_id,
            description=description,
            code=diff
        )
        
        # Record successful application
        new_solution.record_application(True, context)
        self.save_solutions()
        
        logger.info(f"Created new solution: {new_solution.solution_id}")
        
        return new_solution
    
    def _infer_error_type(self, error_message: str) -> str:
        """
        Infer error type from error message.
        
        Args:
            error_message: Error message
            
        Returns:
            Inferred error type
        """
        if "TypeError" in error_message:
            return "type_error"
        elif "AttributeError" in error_message:
            return "attribute_error"
        elif "IndexError" in error_message:
            return "index_error"
        elif "KeyError" in error_message:
            return "key_error"
        elif "NameError" in error_message:
            return "name_error"
        elif "SyntaxError" in error_message:
            return "syntax_error"
        elif "ImportError" in error_message or "ModuleNotFoundError" in error_message:
            return "import_error"
        elif "IOError" in error_message or "FileNotFoundError" in error_message:
            return "io_error"
        elif "PermissionError" in error_message:
            return "permission_error"
        elif "ValueError" in error_message:
            return "value_error"
        elif "ZeroDivisionError" in error_message:
            return "zero_division_error"
        elif "MemoryError" in error_message:
            return "memory_error"
        elif "RecursionError" in error_message:
            return "recursion_error"
        elif "TimeoutError" in error_message:
            return "timeout_error"
        elif "RuntimeError" in error_message:
            return "runtime_error"
        elif "AssertionError" in error_message:
            return "assertion_error"
        else:
            return "unknown"
    
    def _generate_diff(self, buggy_code: str, fixed_code: str) -> str:
        """
        Generate a diff between buggy and fixed code.
        
        Args:
            buggy_code: Original buggy code
            fixed_code: Fixed code
            
        Returns:
            Diff string
        """
        # Simple line-by-line diff
        buggy_lines = buggy_code.splitlines()
        fixed_lines = fixed_code.splitlines()
        
        diff_lines = []
        diff_lines.append("```diff")
        
        # Use a simple diff algorithm
        for line in buggy_lines:
            if line not in fixed_lines:
                diff_lines.append(f"- {line}")
        
        for line in fixed_lines:
            if line not in buggy_lines:
                diff_lines.append(f"+ {line}")
        
        diff_lines.append("```")
        
        return "\n".join(diff_lines)
    
    def _calculate_similarity(self, code1: str, code2: str) -> float:
        """
        Calculate similarity between two code snippets.
        
        Args:
            code1: First code snippet
            code2: Second code snippet
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple Jaccard similarity on code lines
        lines1 = set(code1.splitlines())
        lines2 = set(code2.splitlines())
        
        if not lines1 and not lines2:
            return 1.0
        
        if not lines1 or not lines2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(lines1.intersection(lines2))
        union = len(lines1.union(lines2))
        
        return intersection / union
    
    def delete_solution(self, solution_id: str) -> bool:
        """
        Delete a solution.
        
        Args:
            solution_id: Solution ID
            
        Returns:
            Whether the solution was deleted
        """
        if solution_id in self.solutions:
            solution = self.solutions.pop(solution_id)
            
            # Update pattern -> solutions mapping
            if solution.pattern_id and solution.pattern_id in self.pattern_solutions:
                if solution_id in self.pattern_solutions[solution.pattern_id]:
                    self.pattern_solutions[solution.pattern_id].remove(solution_id)
            
            # Update solution types
            self.solution_types = set(s.solution_type for s in self.solutions.values())
            
            # Save solutions
            self.save_solutions()
            
            logger.info(f"Deleted solution: {solution_id}")
            return True
        
        return False
    
    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """
        Generate a report of solution statistics.
        
        Args:
            output_file: Output file path
            
        Returns:
            Report data
        """
        # Calculate solution type statistics
        solution_type_stats = {}
        
        for solution_type in self.solution_types:
            solutions = self.get_solutions_by_type(solution_type)
            
            total_applications = sum(s.application_count for s in solutions)
            total_successes = sum(s.success_count for s in solutions)
            success_rate = total_successes / total_applications if total_applications > 0 else 0.0
            
            solution_type_stats[solution_type] = {
                "count": len(solutions),
                "total_applications": total_applications,
                "total_successes": total_successes,
                "success_rate": success_rate
            }
        
        # Calculate overall statistics
        total_solutions = len(self.solutions)
        total_applications = sum(s.application_count for s in self.solutions.values())
        total_successes = sum(s.success_count for s in self.solutions.values())
        overall_success_rate = total_successes / total_applications if total_applications > 0 else 0.0
        
        # Create report
        report = {
            "timestamp": time.time(),
            "total_solutions": total_solutions,
            "total_applications": total_applications,
            "total_successes": total_successes,
            "overall_success_rate": overall_success_rate,
            "solution_type_stats": solution_type_stats
        }
        
        # Save report to file if specified
        if output_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Saved solution report to {output_file}")
            except Exception as e:
                logger.error(f"Error saving solution report to {output_file}: {e}")
        
        return report
    
    def plot_success_rates(self, output_file: str = None) -> str:
        """
        Plot success rates by solution type.
        
        Args:
            output_file: Output file path
            
        Returns:
            Output file path
        """
        # Calculate success rates
        solution_types = []
        success_rates = []
        
        for solution_type in sorted(self.solution_types):
            solutions = self.get_solutions_by_type(solution_type)
            
            total_applications = sum(s.application_count for s in solutions)
            total_successes = sum(s.success_count for s in solutions)
            success_rate = total_successes / total_applications if total_applications > 0 else 0.0
            
            solution_types.append(solution_type)
            success_rates.append(success_rate)
        
        if not solution_types:
            logger.warning("No solution types to plot")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot success rates
        plt.bar(solution_types, success_rates, color="green", alpha=0.7)
        
        # Add labels and title
        plt.xlabel("Solution Type")
        plt.ylabel("Success Rate")
        plt.title("Solution Success Rates by Type")
        plt.grid(True, axis="y")
        
        # Add percentage labels
        for i, rate in enumerate(success_rates):
            plt.text(i, rate + 0.02, f"{rate:.1%}", ha="center")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        
        # Set y-axis limits
        plt.ylim(0, 1.1)
        
        # Tight layout
        plt.tight_layout()
        
        # Create output file path if not provided
        if output_file is None:
            os.makedirs("solution_plots", exist_ok=True)
            output_file = f"solution_plots/success_rates_{int(time.time())}.png"
        
        # Save figure
        plt.savefig(output_file)
        plt.close()
        
        return output_file

class SolutionGenerator:
    """
    Generates solutions for bugs.
    """
    
    def __init__(self, recommender: SolutionRecommender = None):
        """
        Initialize solution generator.
        
        Args:
            recommender: Solution recommender
        """
        self.recommender = recommender or SolutionRecommender()
        
        logger.info("Solution generator initialized")
    
    def generate_solution(self, pattern_id: str, buggy_code: str, error_message: str = None,
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a solution for a bug.
        
        Args:
            pattern_id: Pattern ID
            buggy_code: Buggy code
            error_message: Error message
            context: Bug context
            
        Returns:
            Generated solution
        """
        # Get recommended solutions
        recommended_solutions = self.recommender.recommend_solutions(pattern_id, context)
        
        if recommended_solutions:
            # Use highest confidence solution
            solution = recommended_solutions[0]
            
            # Apply solution pattern to buggy code
            fixed_code = self._apply_solution(buggy_code, solution.code)
            
            # Record application (assume success for now)
            solution.record_application(True, context)
            self.recommender.save_solutions()
            
            return {
                "solution_id": solution.solution_id,
                "solution_type": solution.solution_type,
                "description": solution.description,
                "fixed_code": fixed_code,
                "confidence": solution.confidence
            }
        
        # No recommended solutions, try to generate a new one
        if error_message:
            error_type = self.recommender._infer_error_type(error_message)
            solution_types = self.recommender.recommend_solution_types(error_type)
            
            if solution_types:
                solution_type = solution_types[0]
                
                # Generate solution template
                template = self.recommender.generate_solution_template(
                    solution_type=solution_type,
                    error_message=error_message,
                    buggy_code=buggy_code
                )
                
                # Create a new solution
                new_solution = self.recommender.create_solution(
                    solution_type=solution_type,
                    pattern_i
