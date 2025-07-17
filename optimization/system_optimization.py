#!/usr/bin/env python3
"""
System Optimization Module

This module provides system optimization capabilities, including parameter tuning,
token optimization, entropy-based scheduling, MTTR optimization, and neural weight adjustment.
"""

import os
import sys
import json
import logging
import time
import threading
import queue
import random
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("system_optimization.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SystemOptimization")

class ParameterTuner:
    """
    Tunes system parameters for optimal performance.
    """
    
    def __init__(self, config_file: str = None, optimization_metric: str = "response_time"):
        """
        Initialize parameter tuner.
        
        Args:
            config_file: Path to configuration file with parameter definitions
            optimization_metric: Metric to optimize (e.g., "response_time", "throughput", "memory_usage")
        """
        self.config_file = config_file or "optimization_parameters.json"
        self.optimization_metric = optimization_metric
        self.parameters = {}
        self.parameter_history = {}
        self.current_values = {}
        self.best_values = {}
        self.best_score = float('inf') if optimization_metric in ["response_time", "memory_usage", "error_rate"] else 0
        self.minimize = optimization_metric in ["response_time", "memory_usage", "error_rate"]
        self.exploration_rate = 0.3
        self.learning_rate = 0.1
        self.iterations = 0
        
        # Load parameters if config file exists
        if os.path.exists(self.config_file):
            self._load_parameters()
        
        logger.info(f"Parameter tuner initialized with metric: {optimization_metric}")
    
    def _load_parameters(self) -> None:
        """Load parameters from configuration file."""
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            
            # Load parameters
            self.parameters = config.get("parameters", {})
            
            # Initialize current values
            for param_name, param_config in self.parameters.items():
                if "current_value" in param_config:
                    self.current_values[param_name] = param_config["current_value"]
                else:
                    # Default to middle of range
                    min_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 1)
                    default_val = param_config.get("default", (min_val + max_val) / 2)
                    self.current_values[param_name] = default_val
                
                # Initialize parameter history
                self.parameter_history[param_name] = []
            
            # Copy current values to best values
            self.best_values = self.current_values.copy()
            
            logger.info(f"Loaded {len(self.parameters)} parameters from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading parameters from {self.config_file}: {e}")
            self.parameters = {}
            self.current_values = {}
            self.best_values = {}
    
    def save_parameters(self) -> None:
        """Save parameters to configuration file."""
        try:
            # Update parameters with current values
            for param_name, current_value in self.current_values.items():
                if param_name in self.parameters:
                    self.parameters[param_name]["current_value"] = current_value
            
            # Create config
            config = {
                "parameters": self.parameters,
                "best_values": self.best_values,
                "best_score": self.best_score,
                "iterations": self.iterations,
                "last_updated": time.time()
            }
            
            # Save to file
            os.makedirs(os.path.dirname(os.path.abspath(self.config_file)), exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved parameters to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving parameters to {self.config_file}: {e}")
    
    def add_parameter(self, param_name: str, param_config: Dict[str, Any]) -> None:
        """
        Add a parameter to tune.
        
        Args:
            param_name: Parameter name
            param_config: Parameter configuration
                {
                    "min": 0,
                    "max": 100,
                    "default": 50,
                    "step": 5,
                    "type": "int",
                    "description": "Parameter description"
                }
        """
        # Add parameter
        self.parameters[param_name] = param_config
        
        # Initialize current value
        if "current_value" in param_config:
            self.current_values[param_name] = param_config["current_value"]
        else:
            # Default to middle of range
            min_val = param_config.get("min", 0)
            max_val = param_config.get("max", 1)
            default_val = param_config.get("default", (min_val + max_val) / 2)
            self.current_values[param_name] = default_val
        
        # Initialize parameter history
        self.parameter_history[param_name] = []
        
        # Update best values if not set
        if param_name not in self.best_values:
            self.best_values[param_name] = self.current_values[param_name]
        
        logger.info(f"Added parameter: {param_name}")
    
    def remove_parameter(self, param_name: str) -> bool:
        """
        Remove a parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Whether the parameter was removed
        """
        if param_name in self.parameters:
            del self.parameters[param_name]
            
            if param_name in self.current_values:
                del self.current_values[param_name]
            
            if param_name in self.best_values:
                del self.best_values[param_name]
            
            if param_name in self.parameter_history:
                del self.parameter_history[param_name]
            
            logger.info(f"Removed parameter: {param_name}")
            return True
        
        return False
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """
        Suggest parameter values for the next iteration.
        
        Returns:
            Dictionary of parameter values
        """
        suggested_values = {}
        
        for param_name, param_config in self.parameters.items():
            # Decide whether to explore or exploit
            if random.random() < self.exploration_rate:
                # Explore: choose a random value within the range
                min_val = param_config.get("min", 0)
                max_val = param_config.get("max", 1)
                param_type = param_config.get("type", "float")
                
                if param_type == "int":
                    suggested_values[param_name] = random.randint(min_val, max_val)
                elif param_type == "float":
                    suggested_values[param_name] = random.uniform(min_val, max_val)
                elif param_type == "bool":
                    suggested_values[param_name] = random.choice([True, False])
                elif param_type == "choice":
                    choices = param_config.get("choices", [])
                    if choices:
                        suggested_values[param_name] = random.choice(choices)
                    else:
                        suggested_values[param_name] = self.current_values.get(param_name)
                else:
                    suggested_values[param_name] = self.current_values.get(param_name)
            else:
                # Exploit: use the best known value with some noise
                best_value = self.best_values.get(param_name, self.current_values.get(param_name))
                param_type = param_config.get("type", "float")
                
                if param_type == "int":
                    min_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 1)
                    step = param_config.get("step", 1)
                    
                    # Add some noise
                    noise = random.randint(-2, 2) * step
                    suggested_value = best_value + noise
                    
                    # Ensure value is within range
                    suggested_value = max(min_val, min(max_val, suggested_value))
                    suggested_values[param_name] = suggested_value
                elif param_type == "float":
                    min_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 1)
                    
                    # Add some noise
                    noise = random.uniform(-0.1, 0.1) * (max_val - min_val)
                    suggested_value = best_value + noise
                    
                    # Ensure value is within range
                    suggested_value = max(min_val, min(max_val, suggested_value))
                    suggested_values[param_name] = suggested_value
                elif param_type == "bool":
                    # Occasionally flip the value
                    if random.random() < 0.1:
                        suggested_values[param_name] = not best_value
                    else:
                        suggested_values[param_name] = best_value
                elif param_type == "choice":
                    choices = param_config.get("choices", [])
                    if choices and random.random() < 0.1:
                        # Occasionally try a different choice
                        other_choices = [c for c in choices if c != best_value]
                        if other_choices:
                            suggested_values[param_name] = random.choice(other_choices)
                        else:
                            suggested_values[param_name] = best_value
                    else:
                        suggested_values[param_name] = best_value
                else:
                    suggested_values[param_name] = best_value
        
        return suggested_values
    
    def update_with_result(self, parameters: Dict[str, Any], score: float) -> None:
        """
        Update tuner with result of parameter values.
        
        Args:
            parameters: Parameter values used
            score: Result score (lower is better for "response_time", "memory_usage", "error_rate"; higher is better for others)
        """
        # Record parameter values and score
        for param_name, param_value in parameters.items():
            if param_name in self.parameters:
                self.parameter_history[param_name].append((param_value, score))
                self.current_values[param_name] = param_value
        
        # Update best values if score is better
        if (self.minimize and score < self.best_score) or (not self.minimize and score > self.best_score):
            self.best_score = score
            self.best_values = parameters.copy()
            logger.info(f"New best score: {score}")
        
        # Increment iterations
        self.iterations += 1
        
        # Adjust exploration rate
        self.exploration_rate = max(0.1, self.exploration_rate * 0.99)
        
        # Save parameters
        self.save_parameters()
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get the best parameter values found.
        
        Returns:
            Dictionary of parameter values
        """
        return self.best_values
    
    def get_parameter_history(self, param_name: str) -> List[Tuple[Any, float]]:
        """
        Get history of parameter values and scores.
        
        Args:
            param_name: Parameter name
            
        Returns:
            List of (value, score) tuples
        """
        return self.parameter_history.get(param_name, [])
    
    def analyze_parameter_impact(self, param_name: str) -> Dict[str, Any]:
        """
        Analyze the impact of a parameter on the optimization metric.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Analysis results
        """
        if param_name not in self.parameter_history or not self.parameter_history[param_name]:
            return {
                "parameter": param_name,
                "impact": None,
                "correlation": None,
                "recommendation": None
            }
        
        # Get parameter history
        history = self.parameter_history[param_name]
        
        # Extract values and scores
        values = [h[0] for h in history]
        scores = [h[1] for h in history]
        
        # Check parameter type
        param_type = self.parameters[param_name].get("type", "float")
        
        if param_type in ["int", "float"]:
            # Calculate correlation
            try:
                correlation = np.corrcoef(values, scores)[0, 1]
            except:
                correlation = None
            
            # Determine impact
            if correlation is not None:
                if abs(correlation) < 0.1:
                    impact = "Low"
                elif abs(correlation) < 0.3:
                    impact = "Medium"
                else:
                    impact = "High"
            else:
                impact = "Unknown"
            
            # Generate recommendation
            if correlation is not None:
                if (self.minimize and correlation > 0.1) or (not self.minimize and correlation < -0.1):
                    recommendation = "Decrease"
                elif (self.minimize and correlation < -0.1) or (not self.minimize and correlation > 0.1):
                    recommendation = "Increase"
                else:
                    recommendation = "No change"
            else:
                recommendation = "Gather more data"
        elif param_type == "bool":
            # Calculate average score for each value
            true_scores = [score for value, score in history if value]
            false_scores = [score for value, score in history if not value]
            
            if true_scores and false_scores:
                true_avg = sum(true_scores) / len(true_scores)
                false_avg = sum(false_scores) / len(false_scores)
                
                # Calculate impact
                score_diff = abs(true_avg - false_avg)
                score_range = max(scores) - min(scores)
                
                if score_range == 0:
                    impact = "None"
                elif score_diff / score_range < 0.1:
                    impact = "Low"
                elif score_diff / score_range < 0.3:
                    impact = "Medium"
                else:
                    impact = "High"
                
                # Determine correlation (positive correlation means True is better)
                correlation = true_avg - false_avg
                
                # Generate recommendation
                if (self.minimize and correlation > 0) or (not self.minimize and correlation < 0):
                    recommendation = "False"
                elif (self.minimize and correlation < 0) or (not self.minimize and correlation > 0):
                    recommendation = "True"
                else:
                    recommendation = "No change"
            else:
                impact = "Unknown"
                correlation = None
                recommendation = "Gather more data"
        elif param_type == "choice":
            # Calculate average score for each choice
            choice_scores = {}
            
            for value, score in history:
                if value not in choice_scores:
                    choice_scores[value] = []
                
                choice_scores[value].append(score)
            
            # Calculate average score for each choice
            choice_avgs = {}
            
            for choice, scores_list in choice_scores.items():
                choice_avgs[choice] = sum(scores_list) / len(scores_list)
            
            # Determine best choice
            if self.minimize:
                best_choice = min(choice_avgs.items(), key=lambda x: x[1])[0]
            else:
                best_choice = max(choice_avgs.items(), key=lambda x: x[1])[0]
            
            # Calculate impact
            if choice_avgs:
                score_range = max(choice_avgs.values()) - min(choice_avgs.values())
                score_avg = sum(choice_avgs.values()) / len(choice_avgs)
                
                if score_avg == 0:
                    impact = "Unknown"
                elif score_range / score_avg < 0.1:
                    impact = "Low"
                elif score_range / score_avg < 0.3:
                    impact = "Medium"
                else:
                    impact = "High"
            else:
                impact = "Unknown"
            
            correlation = None
            recommendation = best_choice
        else:
            impact = "Unknown"
            correlation = None
            recommendation = "Unknown"
        
        return {
            "parameter": param_name,
            "impact": impact,
            "correlation": correlation,
            "recommendation": recommendation,
            "best_value": self.best_values.get(param_name)
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a report of parameter tuning results.
        
        Returns:
            Report data
        """
        # Analyze all parameters
        parameter_analyses = {}
        
        for param_name in self.parameters:
            parameter_analyses[param_name] = self.analyze_parameter_impact(param_name)
        
        # Generate report
        report = {
            "optimization_metric": self.optimization_metric,
            "iterations": self.iterations,
            "best_score": self.best_score,
            "best_values": self.best_values,
            "current_values": self.current_values,
            "parameter_analyses": parameter_analyses,
            "timestamp": time.time()
        }
        
        return report

class TokenOptimizer:
    """
    Optimizes token usage for LLM interactions.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize token optimizer.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or "token_optimizer.json"
        self.config = {}
        self.token_usage = {}
        self.prompt_templates = {}
        self.compression_strategies = {}
        
        # Load configuration if file exists
        if os.path.exists(self.config_file):
            self._load_config()
        else:
            # Default configuration
            self.config = {
                "max_tokens": 8192,
                "target_tokens": 4096,
                "token_cost": 0.002,  # Cost per 1K tokens
                "compression_enabled": True,
                "context_window_management": "sliding",
                "priority_content": ["code", "errors", "recent_history"]
            }
        
        logger.info("Token optimizer initialized")
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
            
            logger.info(f"Loaded token optimizer configuration from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading token optimizer configuration: {e}")
            
            # Default configuration
            self.config = {
                "max_tokens": 8192,
                "target_tokens": 4096,
                "token_cost": 0.002,  # Cost per 1K tokens
                "compression_enabled": True,
                "context_window_management": "sliding",
                "priority_content": ["code", "errors", "recent_history"]
            }
    
    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.config_file)), exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved token optimizer configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving token optimizer configuration: {e}")
    
    def register_prompt_template(self, template_name: str, template: str, estimated_tokens: int = None) -> None:
        """
        Register a prompt template.
        
        Args:
            template_name: Template name
            template: Prompt template string
            estimated_tokens: Estimated token count for template
        """
        token_count = estimated_tokens or self._estimate_token_count(template)
        
        self.prompt_templates[template_name] = {
            "template": template,
            "token_count": token_count,
            "usage_count": 0,
            "avg_total_tokens": token_count
        }
        
        logger.info(f"Registered prompt template: {template_name} ({token_count} tokens)")
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for a text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Simple estimation: 1 token ~= 4 characters
        return len(text) // 4 + 1
    
    def compress_content(self, content: str, target_tokens: int = None) -> str:
        """
        Compress content to reduce token usage.
        
        Args:
            content: Content to compress
            target_tokens: Target token count
            
        Returns:
            Compressed content
        """
        if not self.config.get("compression_enabled", True):
            return content
        
        target = target_tokens or self.config.get("target_tokens", 4096)
        estimated_tokens = self._estimate_token_count(content)
        
        if estimated_tokens <= target:
            return content
        
        # Apply compression strategies
        compressed = content
        
        # Strategy 1: Remove unnecessary whitespace
        compressed = " ".join(compressed.split())
        
        # Strategy 2: Truncate if still too long
        if self._estimate_token_count(compressed) > target:
            # Calculate target length in characters
            target_chars = target * 4
            
            # Keep the beginning and end, remove the middle
            if len(compressed) > target_chars:
                keep_chars = target_chars // 2
                compressed = compressed[:keep_chars] + "\n...[content truncated]...\n" + compressed[-keep_chars:]
        
        logger.debug(f"Compressed content from ~{estimated_tokens} to ~{self._estimate_token_count(compressed)} tokens")
        
        return compressed
    
    def optimize_prompt(self, template_name: str, variables: Dict[str, str]) -> str:
        """
        Optimize a prompt using a template and variables.
        
        Args:
            template_name: Template name
            variables: Template variables
            
        Returns:
            Optimized prompt
        """
        if template_name not in self.prompt_templates:
            logger.warning(f"Unknown prompt template: {template_name}")
            
            # Combine variables into a prompt
            prompt = "\n".join([f"{k}: {v}" for k, v in variables.items()])
            
            # Compress if needed
            return self.compress_content(prompt)
        
        template = self.prompt_templates[template_name]["template"]
        
        # Compress large variable values if needed
        max_tokens = self.config.get("max_tokens", 8192)
        template_tokens = self.prompt_templates[template_name]["token_count"]
        remaining_tokens = max_tokens - template_tokens
        
        # Estimate tokens for each variable
        var_tokens = {}
        for var_name, var_value in variables.items():
            var_tokens[var_name] = self._estimate_token_count(var_value)
        
        # Calculate total tokens
        total_tokens = template_tokens + sum(var_tokens.values())
        
        # Compress variables if total exceeds max tokens
        if total_tokens > max_tokens:
            # Sort variables by token count (descending)
            sorted_vars = sorted(var_tokens.items(), key=lambda x: x[1], reverse=True)
            
            # Compress variables until total is below max
            for var_name, token_count in sorted_vars:
                if total_tokens <= max_tokens:
                    break
                
                if token_count > 100:  # Only compress large variables
                    # Calculate target tokens for this variable
                    target_var_tokens = token_count - (total_tokens - max_tokens)
                    target_var_tokens = max(50, target_var_tokens)  # Ensure minimum size
                    
                    # Compress variable
                    variables[var_name] = self.compress_content(variables[var_name], target_var_tokens)
                    
                    # Recalculate token count
                    var_tokens[var_name] = self._estimate_token_count(variables[var_name])
                    total_tokens = template_tokens + sum(var_tokens.values())
        
        # Format template with variables
        try:
            prompt = template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in template: {e}")
            
            # Fallback: combine variables into a prompt
            prompt = "\n".join([f"{k}: {v}" for k, v in variables.items()])
        
        # Update usage statistics
        self.prompt_templates[template_name]["usage_count"] += 1
        
        # Update average total tokens
        template_data = self.prompt_templates[template_name]
        avg_tokens = template_data["avg_total_tokens"]
        usage_count = template_data["usage_count"]
        new_avg = (avg_tokens * (usage_count - 1) + total_tokens) / usage_count
        template_data["avg_total_tokens"] = new_avg
        
        return prompt
    
    def record_token_usage(self, context: str, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Record token usage for analytics.
        
        Args:
            context: Usage context (e.g., "bug_detection", "patch_generation")
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        if context not in self.token_usage:
            self.token_usage[context] = {
                "count": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "avg_prompt_tokens": 0,
                "avg_completion_tokens": 0,
                "estimated_cost": 0
            }
        
        usage = self.token_usage[context]
        usage["count"] += 1
        usage["total_prompt_tokens"] += prompt_tokens
        usage["total_completion_tokens"] += completion_tokens
        usage["avg_prompt_tokens"] = usage["total_prompt_tokens"] / usage["count"]
        usage["avg_completion_tokens"] = usage["total_completion_tokens"] / usage["count"]
        
        # Calculate estimated cost
        token_cost = self.config.get("token_cost", 0.002)  # Cost per 1K tokens
        prompt_cost = (prompt_tokens / 1000) * token_cost
        completion_cost = (completion_tokens / 1000) * token_cost
        usage["estimated_cost"] += prompt_cost + completion_cost
        
        logger.debug(f"Recorded token usage for {context}: {prompt_tokens} prompt, {completion_tokens} completion")
    
    def get_token_usage_stats(self) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Returns:
            Token usage statistics
        """
        # Calculate total usage
        total_stats = {
            "count": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "avg_prompt_tokens": 0,
            "avg_completion_tokens": 0,
            "estimated_cost": 0
        }
        
        for context, stats in self.token_usage.items():
            total_stats["count"] += stats["count"]
            total_stats["total_prompt_tokens"] += stats["total_prompt_tokens"]
            total_stats["total_completion_tokens"] += stats["total_completion_tokens"]
            total_stats["estimated_cost"] += stats["estimated_cost"]
        
        if total_stats["count"] > 0:
            total_stats["avg_prompt_tokens"] = total_stats["total_prompt_tokens"] / total_stats["count"]
            total_stats["avg_completion_tokens"] = total_stats["total_completion_tokens"] / total_stats["count"]
        
        # Return combined stats
        return {
            "by_context": self.token_usage,
            "total": total_stats
        }
    
    def optimize_context_window(self, context_parts: List[Dict[str, str]], max_tokens: int = None) -> List[Dict[str, str]]:
        """
        Optimize context window content.
        
        Args:
            context_parts: List of context parts with 'content' and 'type' keys
            max_tokens: Maximum tokens (defaults to config)
            
        Returns:
            Optimized context parts
        """
        max_tokens = max_tokens or self.config.get("max_tokens", 8192)
        
        # Estimate tokens for each part
        for part in context_parts:
            part["tokens"] = self._estimate_token_count(part.get("content", ""))
        
        # Calculate total tokens
        total_tokens = sum(part["tokens"] for part in context_parts)
        
        # If total is within limit, return unchanged
        if total_tokens <= max_tokens:
            return context_parts
        
        # Get context window management strategy
        strategy = self.config.get("context_window_management", "sliding")
        
        if strategy == "priority":
            # Priority-based retention
            # Sort parts by priority (based on type)
            priority_types = self.config.get("priority_content", ["code", "errors", "recent_history"])
            
            # Assign priority to each part (lower number = higher priority)
            for part in context_parts:
                part_type = part.get("type", "")
                try:
                    part["priority"] = priority_types.index(part_type)
                except ValueError:
                    part["priority"] = len(priority_types)
            
            # Sort by priority
            sorted_parts = sorted(context_parts, key=lambda x: x["priority"])
            
            # Keep parts until we hit the token limit
            optimized_parts = []
            current_tokens = 0
            
            for part in sorted_parts:
                if current_tokens + part["tokens"] <= max_tokens:
                    optimized_parts.append(part)
                    current_tokens += part["tokens"]
                else:
                    # Try to compress this part
                    target_tokens = max_tokens - current_tokens
                    if target_tokens > 100:  # Only compress if we can keep a meaningful amount
                        compressed_content = self.compress_content(part.get("content", ""), target_tokens)
                        compressed_tokens = self._estimate_token_count(compressed_content)
                        
                        part
