#!/usr/bin/env python3
"""
blocker_learning.py
──────────────────
Learning system for recording and analyzing blocker patterns.

This module provides tools to record patterns of blockers, analyze them to
develop preventative strategies, and create a system improvement feedback loop.
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

# Configure logging
logger = logging.getLogger("BlockerLearning")
handler = logging.FileHandler("blocker_learning.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class PatternType(Enum):
    """Types of blocker patterns."""
    RECURRING_TYPE = "recurring_type"            # Same blocker type recurring
    ERROR_SEQUENCE = "error_sequence"            # Specific sequence of errors
    CONTEXT_DEPENDENT = "context_dependent"      # Blocker dependent on context
    RESOURCE_THRESHOLD = "resource_threshold"    # Resource usage threshold
    TIME_DEPENDENT = "time_dependent"            # Time-dependent blocker
    INTERACTION = "interaction"                  # Interaction between components
    EXTERNAL = "external"                        # External system dependency

class BlockerPattern:
    """
    Represents a pattern of blockers that can be identified and learned from.
    """
    
    def __init__(self, pattern_id: str, pattern_type: PatternType, description: str):
        """
        Initialize a blocker pattern.
        
        Args:
            pattern_id: Unique ID for the pattern
            pattern_type: Type of pattern
            description: Description of the pattern
        """
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.description = description
        self.instances = []
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.occurrence_count = 0
        self.resolved_count = 0
        self.preventative_strategies = []
    
    def add_instance(self, blocker: Dict[str, Any], context: Dict[str, Any], resolution: Optional[Dict[str, Any]] = None):
        """
        Add an instance of the pattern.
        
        Args:
            blocker: Blocker information
            context: Context in which the blocker occurred
            resolution: How the blocker was resolved (if available)
        """
        self.instances.append({
            "blocker": blocker,
            "context": context,
            "resolution": resolution,
            "timestamp": time.time()
        })
        self.last_seen = time.time()
        self.occurrence_count += 1
        
        if resolution and resolution.get("success", False):
            self.resolved_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the pattern to a dictionary.
        
        Returns:
            Dictionary representation of the pattern
        """
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "description": self.description,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "occurrence_count": self.occurrence_count,
            "resolved_count": self.resolved_count,
            "success_rate": self.resolved_count / max(1, self.occurrence_count),
            "instances": self.instances,
            "preventative_strategies": self.preventative_strategies
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BlockerPattern':
        """
        Create a pattern from a dictionary.
        
        Args:
            data: Dictionary representation of the pattern
            
        Returns:
            BlockerPattern instance
        """
        pattern = cls(
            data["pattern_id"],
            PatternType(data["pattern_type"]),
            data["description"]
        )
        pattern.first_seen = data.get("first_seen", time.time())
        pattern.last_seen = data.get("last_seen", time.time())
        pattern.occurrence_count = data.get("occurrence_count", 0)
        pattern.resolved_count = data.get("resolved_count", 0)
        pattern.instances = data.get("instances", [])
        pattern.preventative_strategies = data.get("preventative_strategies", [])
        
        return pattern

class BlockerLearningSystem:
    """
    System for learning from blockers.
    
    Records patterns of blockers, analyzes them to develop preventative strategies,
    and creates a system improvement feedback loop.
    """
    
    def __init__(self, storage_file: str = "blocker_patterns.json"):
        """
        Initialize the blocker learning system.
        
        Args:
            storage_file: File to store blocker patterns
        """
        self.storage_file = storage_file
        self.patterns = {}
        self.load_patterns()
        
        # Register commands
        try:
            register_command("record_blocker", self.record_blocker_command, 
                            "Record a blocker pattern")
            register_command("list_patterns", self.list_patterns_command,
                            "List recorded blocker patterns")
            register_command("analyze_patterns", self.analyze_patterns_command,
                            "Analyze blocker patterns and develop strategies")
            logger.info("Blocker learning commands registered")
        except Exception as e:
            logger.error(f"Failed to register commands: {e}")
    
    def load_patterns(self):
        """Load patterns from storage file."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                
                for pattern_data in data:
                    pattern = BlockerPattern.from_dict(pattern_data)
                    self.patterns[pattern.pattern_id] = pattern
                
                logger.info(f"Loaded {len(self.patterns)} blocker patterns from {self.storage_file}")
            else:
                logger.info(f"No pattern storage file found at {self.storage_file}")
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
    
    def save_patterns(self):
        """Save patterns to storage file."""
        try:
            patterns_data = [pattern.to_dict() for pattern in self.patterns.values()]
            
            with open(self.storage_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            logger.info(f"Saved {len(self.patterns)} blocker patterns to {self.storage_file}")
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def record_blocker(self, 
                      blocker: Dict[str, Any], 
                      context: Dict[str, Any], 
                      resolution: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a blocker and identify or create a pattern.
        
        Args:
            blocker: Blocker information
            context: Context in which the blocker occurred
            resolution: How the blocker was resolved (if available)
            
        Returns:
            ID of the pattern
        """
        # Try to match to existing pattern
        pattern_id = self._match_pattern(blocker, context)
        
        # If no match, create new pattern
        if not pattern_id:
            pattern_id = self._create_pattern(blocker, context)
        
        # Add instance to pattern
        self.patterns[pattern_id].add_instance(blocker, context, resolution)
        
        # Save patterns
        self.save_patterns()
        
        # Return pattern ID
        return pattern_id
    
    def _match_pattern(self, blocker: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """
        Match a blocker to an existing pattern.
        
        Args:
            blocker: Blocker information
            context: Context in which the blocker occurred
            
        Returns:
            Pattern ID if matched, None otherwise
        """
        # Extract key features for matching
        blocker_type = blocker.get("type", "unknown")
        blocker_message = blocker.get("message", "")
        operation_type = context.get("operation_type", "unknown")
        
        # Look for matches in existing patterns
        for pattern_id, pattern in self.patterns.items():
            # Check for recurring type patterns
            if pattern.pattern_type == PatternType.RECURRING_TYPE:
                # Check if this is the same type of blocker in a similar context
                for instance in pattern.instances:
                    inst_blocker = instance.get("blocker", {})
                    inst_context = instance.get("context", {})
                    
                    if (inst_blocker.get("type") == blocker_type and
                        inst_context.get("operation_type") == operation_type):
                        return pattern_id
            
            # Check for context-dependent patterns
            elif pattern.pattern_type == PatternType.CONTEXT_DEPENDENT:
                # Check for matching context features
                context_matches = 0
                required_matches = 3  # Number of context features that must match
                
                for instance in pattern.instances:
                    inst_context = instance.get("context", {})
                    
                    # Count matching context features
                    for key, value in inst_context.items():
                        if key in context and context[key] == value:
                            context_matches += 1
                    
                    if context_matches >= required_matches:
                        return pattern_id
            
            # Check for error sequence patterns
            elif pattern.pattern_type == PatternType.ERROR_SEQUENCE:
                # Check if this is part of a sequence of errors
                if "errors" in context and len(context["errors"]) > 1:
                    for instance in pattern.instances:
                        inst_context = instance.get("context", {})
                        
                        if "errors" in inst_context and len(inst_context["errors"]) > 1:
                            # Check if error sequences are similar
                            if self._compare_error_sequences(context["errors"], inst_context["errors"]):
                                return pattern_id
        
        # No matching pattern found
        return None
    
    def _compare_error_sequences(self, seq1: List[str], seq2: List[str]) -> bool:
        """
        Compare two error sequences to see if they are similar.
        
        Args:
            seq1: First sequence of errors
            seq2: Second sequence of errors
            
        Returns:
            True if sequences are similar, False otherwise
        """
        # If sequences are identical, they are similar
        if seq1 == seq2:
            return True
        
        # If sequences are very different in length, they are not similar
        if abs(len(seq1) - len(seq2)) > min(len(seq1), len(seq2)) / 2:
            return False
        
        # Count matching errors
        matches = 0
        for error1 in seq1:
            for error2 in seq2:
                if self._similar_errors(error1, error2):
                    matches += 1
                    break
        
        # If more than half of errors match, sequences are similar
        return matches >= min(len(seq1), len(seq2)) / 2
    
    def _similar_errors(self, error1: str, error2: str) -> bool:
        """
        Check if two errors are similar.
        
        Args:
            error1: First error message
            error2: Second error message
            
        Returns:
            True if errors are similar, False otherwise
        """
        # Remove variable parts of errors (e.g., specific values, paths)
        # and compare the remaining structure
        
        # For now, use a simple approach: check if one is a substring of the other
        # A more sophisticated approach would use similarity metrics like Levenshtein distance
        return error1 in error2 or error2 in error1
    
    def _create_pattern(self, blocker: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Create a new pattern for a blocker.
        
        Args:
            blocker: Blocker information
            context: Context in which the blocker occurred
            
        Returns:
            ID of the new pattern
        """
        # Generate pattern ID
        pattern_id = f"pattern_{int(time.time())}_{len(self.patterns)}"
        
        # Determine pattern type
        pattern_type = self._determine_pattern_type(blocker, context)
        
        # Generate description
        description = self._generate_pattern_description(blocker, context, pattern_type)
        
        # Create pattern
        pattern = BlockerPattern(pattern_id, pattern_type, description)
        
        # Add to patterns
        self.patterns[pattern_id] = pattern
        
        logger.info(f"Created new blocker pattern: {pattern_id} ({pattern_type.value})")
        
        return pattern_id
    
    def _determine_pattern_type(self, blocker: Dict[str, Any], context: Dict[str, Any]) -> PatternType:
        """
        Determine the type of pattern for a blocker.
        
        Args:
            blocker: Blocker information
            context: Context in which the blocker occurred
            
        Returns:
            Pattern type
        """
        # Check for resource threshold
        if (blocker.get("type") == BlockerType.RESOURCE.value or
            "memory_usage" in context or "cpu_usage" in context or "disk_usage" in context):
            return PatternType.RESOURCE_THRESHOLD
        
        # Check for error sequence
        if "errors" in context and len(context["errors"]) > 1:
            return PatternType.ERROR_SEQUENCE
        
        # Check for time-dependent
        if "elapsed_time" in context and context["elapsed_time"] > 30:
            return PatternType.TIME_DEPENDENT
        
        # Check for external
        if blocker.get("type") == BlockerType.EXTERNAL.value:
            return PatternType.EXTERNAL
        
        # Check for interaction
        if "component_interactions" in context:
            return PatternType.INTERACTION
        
        # Check for context-dependent
        if len(context) > 3:
            return PatternType.CONTEXT_DEPENDENT
        
        # Default to recurring type
        return PatternType.RECURRING_TYPE
    
    def _generate_pattern_description(self, 
                                     blocker: Dict[str, Any], 
                                     context: Dict[str, Any], 
                                     pattern_type: PatternType) -> str:
        """
        Generate a description for a pattern.
        
        Args:
            blocker: Blocker information
            context: Context in which the blocker occurred
            pattern_type: Type of pattern
            
        Returns:
            Description of the pattern
        """
        blocker_type = blocker.get("type", "unknown")
        blocker_severity = blocker.get("severity", "unknown")
        blocker_message = blocker.get("message", "Unknown blocker")
        operation_type = context.get("operation_type", "unknown operation")
        
        if pattern_type == PatternType.RECURRING_TYPE:
            return f"Recurring {blocker_type} blocker during {operation_type}: {blocker_message}"
        
        elif pattern_type == PatternType.RESOURCE_THRESHOLD:
            resource_type = "memory" if "memory_usage" in context else "CPU" if "cpu_usage" in context else "disk" if "disk_usage" in context else "resource"
            threshold = context.get("memory_usage", context.get("cpu_usage", context.get("disk_usage", "high")))
            return f"{resource_type.capitalize()} usage threshold ({threshold}) exceeded during {operation_type}"
        
        elif pattern_type == PatternType.ERROR_SEQUENCE:
            return f"Sequence of errors during {operation_type} leading to {blocker_type} blocker"
        
        elif pattern_type == PatternType.TIME_DEPENDENT:
            elapsed_time = context.get("elapsed_time", "long")
            return f"Time-dependent {blocker_type} blocker after {elapsed_time} seconds of {operation_type}"
        
        elif pattern_type == PatternType.EXTERNAL:
            return f"External dependency blocker during {operation_type}: {blocker_message}"
        
        elif pattern_type == PatternType.INTERACTION:
            return f"Component interaction blocker during {operation_type}: {blocker_message}"
        
        elif pattern_type == PatternType.CONTEXT_DEPENDENT:
            return f"Context-dependent {blocker_type} blocker during {operation_type}"
        
        else:
            return f"{blocker_type} blocker during {operation_type}: {blocker_message}"
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze blocker patterns and develop preventative strategies.
        
        Returns:
            Analysis results
        """
        if not self.patterns:
            return {
                "success": False,
                "error": "No patterns to analyze"
            }
        
        # Prepare for analysis
        pattern_stats = {}
        blocker_type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        context_features = defaultdict(int)
        recurring_patterns = []
        frequent_patterns = []
        severe_patterns = []
        
        # Collect statistics
        for pattern_id, pattern in self.patterns.items():
            pattern_stats[pattern_id] = {
                "type": pattern.pattern_type.value,
                "description": pattern.description,
                "occurrences": pattern.occurrence_count,
                "resolutions": pattern.resolved_count,
                "success_rate": pattern.resolved_count / max(1, pattern.occurrence_count),
                "first_seen": pattern.first_seen,
                "last_seen": pattern.last_seen,
                "age_days": (time.time() - pattern.first_seen) / (60 * 60 * 24)
            }
            
            # Track frequent patterns
            if pattern.occurrence_count >= 3:
                frequent_patterns.append(pattern_id)
            
            # Track recurring patterns (seen recently)
            if time.time() - pattern.last_seen < 7 * 24 * 60 * 60:  # Within last 7 days
                recurring_patterns.append(pattern_id)
            
            # Analyze instances
            for instance in pattern.instances:
                blocker = instance.get("blocker", {})
                context = instance.get("context", {})
                
                # Track blocker types
                blocker_type = blocker.get("type", "unknown")
                blocker_type_counts[blocker_type] += 1
                
                # Track severities
                severity = blocker.get("severity", "unknown")
                severity_counts[severity] += 1
                
                # Track high severity patterns
                if severity in ["high", "fatal"] and pattern_id not in severe_patterns:
                    severe_patterns.append(pattern_id)
                
                # Track context features
                for key, value in context.items():
                    if isinstance(value, (str, int, float, bool)):
                        context_features[key] += 1
        
        # Generate strategies for top patterns
        priority_patterns = list(set(frequent_patterns + recurring_patterns + severe_patterns))
        strategies = {}
        
        for pattern_id in priority_patterns:
            pattern = self.patterns[pattern_id]
            strategy = self._generate_strategy(pattern)
            
            if strategy:
                pattern.preventative_strategies.append(strategy)
                strategies[pattern_id] = strategy
        
        # Save patterns with new strategies
        self.save_patterns()
        
        # Prepare analysis results
        results = {
            "success": True,
            "pattern_count": len(self.patterns),
            "pattern_stats": pattern_stats,
            "blocker_type_distribution": dict(blocker_type_counts),
            "severity_distribution": dict(severity_counts),
            "common_context_features": dict(sorted(context_features.items(), key=lambda x: x[1], reverse=True)[:10]),
            "priority_patterns": priority_patterns,
            "strategies_generated": len(strategies),
            "strategies": strategies
        }
        
        return results
    
    def _generate_strategy(self, pattern: BlockerPattern) -> Optional[Dict[str, Any]]:
        """
        Generate a preventative strategy for a pattern.
        
        Args:
            pattern: Blocker pattern
            
        Returns:
            Strategy information or None if no strategy could be generated
        """
        # Try to generate a strategy using an agent
        try:
            # Prepare data for the agent
            pattern_data = pattern.to_dict()
            
            # Limit the number of instances to avoid overwhelming the agent
            if len(pattern_data["instances"]) > 5:
                pattern_data["instances"] = pattern_data["instances"][-5:]
            
            # Prepare prompt for agent
            prompt = f"""
            Generate a preventative strategy for the following blocker pattern:
            
            Pattern Type: {pattern.pattern_type.value}
            Description: {pattern.description}
            Occurrences: {pattern.occurrence_count}
            Success Rate: {pattern.resolved_count / max(1, pattern.occurrence_count):.2f}
            
            Sample Instances:
            {json.dumps(pattern_data["instances"], indent=2)}
            
            Please provide:
            1. A name for the strategy
            2. A detailed description of the strategy
            3. Specific steps to implement the strategy
            4. Expected effectiveness
            5. Implementation difficulty (low, medium, high)
            
            Return your strategy as a structured object.
            """
            
            # Request strategy from agent
            result = request_agent_task("strategy_generation", prompt, timeout=30)
            
            if result.get("success", False) and result.get("strategy"):
                strategy = result["strategy"]
                strategy["timestamp"] = time.time()
                
                logger.info(f"Generated strategy for pattern {pattern.pattern_id}: {strategy.get('name', 'Unnamed strategy')}")
                
                return strategy
            
            logger.warning(f"Failed to generate strategy for pattern {pattern.pattern_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error generating strategy for pattern {pattern.pattern_id}: {e}")
            return None
    
    def apply_learning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply learned strategies to prevent blockers.
        
        Args:
            context: Execution context
            
        Returns:
            Modified context with preventative measures
        """
        operation_type = context.get("operation_type", "unknown")
        
        # Create a copy of the context to modify
        new_context = context.copy()
        
        # Track applied strategies
        applied_strategies = []
        
        # Look for applicable strategies
        for pattern_id, pattern in self.patterns.items():
            for strategy in pattern.preventative_strategies:
                # Check if strategy is applicable to this context
                if self._is_strategy_applicable(pattern, strategy, context):
                    # Apply the strategy
                    strategy_result = self._apply_strategy(strategy, new_context)
                    
                    if strategy_result.get("success", False):
                        applied_strategies.append({
                            "pattern_id": pattern_id,
                            "strategy_name": strategy.get("name", "Unnamed strategy"),
                            "result": strategy_result
                        })
        
        # Add metadata about applied strategies
        if applied_strategies:
            new_context["applied_strategies"] = applied_strategies
            logger.info(f"Applied {len(applied_strategies)} preventative strategies to {operation_type} operation")
        
        return new_context
    
    def _is_strategy_applicable(self, 
                               pattern: BlockerPattern, 
                               strategy: Dict[str, Any], 
                               context: Dict[str, Any]) -> bool:
        """
        Check if a strategy is applicable to a context.
        
        Args:
            pattern: Blocker pattern
            strategy: Strategy to check
            context: Execution context
            
        Returns:
            True if the strategy is applicable, False otherwise
        """
        operation_type = context.get("operation_type", "unknown")
        
        # Check if this pattern has ever occurred in this type of operation
        operation_match = False
        for instance in pattern.instances:
            inst_context = instance.get("context", {})
            if inst_context.get("operation_type") == operation_type:
                operation_match = True
                break
        
        if not operation_match:
            return False
        
        # Check for specific indicators that make this strategy applicable
        if pattern.pattern_type == PatternType.RESOURCE_THRESHOLD:
            # Check if resource usage is approaching threshold
            memory_usage = context.get("memory_usage")
            cpu_usage = context.get("cpu_usage")
            disk_usage = context.get("disk_usage")
            
            if (memory_usage and memory_usage > 70) or (cpu_usage and cpu_usage > 70) or (disk_usage and disk_usage > 70):
                return True
        
        elif pattern.pattern_type == PatternType.TIME_DEPENDENT:
            # Check if operation has been running for a while
            elapsed_time = context.get("elapsed_time")
            
            if elapsed_time and elapsed_time > 15:  # 15 seconds is getting close to potential timeout
                return True
        
        elif pattern.pattern_type == PatternType.ERROR_SEQUENCE:
            # Check if we're starting to see a similar error sequence
            if "errors" in context and len(context["errors"]) > 0:
                for instance in pattern.instances:
                    inst_context = instance.get("context", {})
                    if "errors" in inst_context and len(inst_context["errors"]) > 0:
                        # Check if the first error is similar
                        if self._similar_errors(context["errors"][0], inst_context["errors"][0]):
                            return True
        
        # For other pattern types, apply if we have a high occurrence count
        return pattern.occurrence_count >= 3
    
    def _apply_strategy(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a strategy to a context.
        
        Args:
            strategy: Strategy to apply
            context: Execution context
            
        Returns:
            Result of applying the strategy
        """
        strategy_name = strategy.get("name", "Unnamed strategy")
        steps = strategy.get("steps", [])
        
        logger.info(f"Applying strategy: {strategy_name}")
        
        # Apply each step of the strategy
        for i, step in enumerate(steps, 1):
            logger.info(f"Applying step {i}/{len(steps)}: {step}")
            
            # Here we would implement the actual step application
            # For now, we'll just simulate it by modifying the context
            
            # Example modifications based on step descriptions
            if "resource" in step.lower():
                # Resource-related step (e.g., reducing memory usage)
                if "reduce" in step.lower() and "memory" in step.lower():
                    context["memory_optimization"] = True
                elif "limit" in step.lower() and "cpu" in step.lower():
                    context["cpu_limit"] = True
                elif "disk" in step.lower():
                    context["disk_optimization"] = True
            
            elif "timeout" in step.lower() or "time" in step.lower():
                # Timeout-related step
                if "extend" in step.lower() or "increase" in step.lower():
                    context["extended_timeout"] = True
                elif "break" in step.lower() and "smaller" in step.lower():
                    context["chunked_operation"] = True
            
            elif "dependency" in step.lower():
                # Dependency-related step
                if "check" in step.lower():
                    context["dependency_check"] = True
                elif "update" in step.lower() or "upgrade" in step.lower():
                    context["dependency_update"] = True
            
            elif "permission" in step.lower() or "access" in step.lower():
                # Permission-related step
                if "elevate" in step.lower() or "increase" in step.lower():
                    context["elevated_permissions"] = True
                elif "verify" in step.lower() or "check" in step.lower():
                    context["permission_verification"] = True
            
            elif "retry" in step.lower():
                # Retry-related step
                if "delay" in step.lower() or "wait" in step.lower():
                    context["delayed_retry"] = True
                elif "exponential" in step.lower():
                    context["exponential_backoff"] = True
        
        return {
            "success": True,
            "strategy_name": strategy_name,
            "applied_steps": len(steps),
            "context_modifications": [k for k, v in context.items() if isinstance(v, bool) and v is True]
        }
    
    def record_blocker_command(self, args: str) -> int:
        """
        Handle the record_blocker command.
        
        Args:
            args: Command arguments (blocker_file context_file [resolution_file])
            
        Returns:
            Exit code
        """
        try:
            arg_parts = args.strip().split()
            
            if len(arg_parts) < 2:
                print("Error: Insufficient arguments.")
                print("Usage: record_blocker <blocker_file> <context_file> [resolution_file]")
                return 1
            
            blocker_file = arg_parts[0]
            context_file = arg_parts[1]
            resolution_file = arg_parts[2] if len(arg_parts) > 2 else None
            
            # Load blocker
            try:
                with open(blocker_file, 'r') as f:
                    blocker = json.load(f)
            except Exception as e:
                print(f"Error loading blocker file: {e}")
                return 1
            
            # Load context
            try:
                with open(context_file, 'r') as f:
                    context = json.load(f)
            except Exception as e:
                print(f"Error loading context file: {e}")
                return 1
            
            # Load resolution (if provided)
            resolution = None
            if resolution_file:
                try:
                    with open(resolution_file, 'r') as f:
                        resolution = json.load(f)
                except Exception as e:
                    print(f"Error loading resolution file: {e}")
                    return 1
            
            # Record blocker
            pattern_id = self.record_blocker(blocker, context, resolution)
            
            print(f"Blocker recorded as pattern {pattern_id}")
            print(f"Pattern type: {self.patterns[pattern_id].pattern_type.value}")
            print(f"Description: {self.patterns[pattern_id].description}")
            print(f"Occurrence count: {self.patterns[pattern_id].occurrence_count}")
            
            return 0
                
        except Exception as e:
            print(f"Error recording blocker: {e}")
            logger.error(f"Error recording blocker: {e}")
            return 1
    
    def list_patterns_command(self, args: str) -> int:
        """
        Handle the list_patterns command.
        
        Args:
            args: Command arguments (optional pattern type filter)
            
        Returns:
            Exit code
        """
        try:
            pattern_type_filter = args.strip() if args.strip() else None
            
            if pattern_type_filter:
                try:
                    pattern_type_filter = PatternType(pattern_type_filter).value
                except ValueError:
                    print(f"Error: Invalid pattern type filter: {pattern_type_filter}")
                    print("Valid pattern types:")
                    for pattern_type in PatternType:
                        print(f"  {pattern_type.value}")
                    return 1
            
            # Filter patterns
            if pattern_type_filter:
                patterns = [p for p in self.patterns.values() if p.pattern_type.value == pattern_type_filter]
            else:
                patterns = list(self.patterns.values())
            
            # Sort patterns by occurrence count
            patterns.sort(key=lambda p: p.occurrence_count, reverse=True)
            
            print(f"Found {len(patterns)} patterns:")
            
            for i, pattern in enumerate(patterns, 1):
                print(f"\n{i}. Pattern ID: {pattern.pattern_id}")
                print(f"   Type: {pattern.pattern_type.value}")
                print(f"   Description: {pattern.description}")
                print(f"   Occurrences: {pattern.occurrence_count}")
                print(f"   Success Rate: {pattern.resolved_count / max(1, pattern.occurrence_count):.2f}")
                print(f"   First Seen: {datetime.datetime.fromtimestamp(pattern.first_seen).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Last Seen: {datetime.datetime.fromtimestamp(pattern.last_seen).strftime('%Y-%m-%d %H:%M:%S')}")
                
                if pattern.preventative_strategies:
                    print(f"   Strategies: {len(pattern.preventative_strategies)}")
                    for j, strategy in enumerate(pattern.preventative_strategies, 1):
                        print(f"     {j}. {strategy.get('name', 'Unnamed strategy')}")
            
            return 0
                
        except Exception as e:
            print(f"Error listing patterns: {e}")
            logger.error(f"Error listing patterns: {e}")
            return 1
    
    def analyze_patterns_command(self, args: str) -> int:
        """
        Handle the analyze_patterns command.
        
        Args:
            args: Command arguments (none required)
            
        Returns:
            Exit code
        """
        try:
            print("Analyzing blocker patterns...")
            
            results = self.analyze_patterns()
            
            if not results.get("success", False):
                print(f"Error: {results.get('error', 'Unknown error')}")
                return 1
            
            print(f"Analyzed {results['pattern_count']} patterns")
            
            print("\nBlocker Type Distribution:")
            for blocker_type, count in results["blocker_type_distribution"].items():
                print(f"  {blocker_type}: {count}")
            
            print("\nSeverity Distribution:")
            for severity, count in results["severity_distribution"].items():
                print(f"  {severity}: {count}")
            
            print("\nCommon Context Features:")
            for feature, count in results["common_context_features"].items():
                print(f"  {feature}: {count}")
            
            print(f"\nGenerated {results['strategies_generated']} preventative strategies")
            
            if results['strategies']:
                print("\nNew Strategies:")
                for pattern_id, strategy in results['strategies'].items():
                    print(f"  Pattern {pattern_id}:")
                    print(f"    Strategy: {strategy.get('name', 'Unnamed strategy')}")
                    print(f"    Difficulty: {strategy.get('difficulty', 'Unknown')}")
                    print(f"    Effectiveness: {strategy.get('effectiveness', 'Unknown')}")
            
            # Save analysis to file
            output_file = f"pattern_analysis_{int(time.time())}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nDetailed analysis saved to {output_file}")
            
            return 0
                
        except Exception as e:
            print(f"Error analyzing patterns: {e}")
            logger.error(f"Error analyzing patterns: {e}")
            return 1


# Initialize blocker learning system
blocker_learning_system = BlockerLearningSystem()
logger.info("Blocker learning system initialized")
