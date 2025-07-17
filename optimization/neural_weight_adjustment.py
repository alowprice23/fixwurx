#!/usr/bin/env python3
"""
Neural Weight Adjustment Module

This module provides neural weight adjustment capabilities, enabling dynamic
optimization of neural network weights based on performance feedback.
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
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("neural_weight_adjustment.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NeuralWeightAdjustment")

class NeuralWeight:
    """
    Represents a neural weight with adjustment capabilities.
    """
    
    def __init__(self, name: str, value: float = 0.5, min_value: float = 0.0, 
                max_value: float = 1.0, learning_rate: float = 0.01):
        """
        Initialize neural weight.
        
        Args:
            name: Weight name
            value: Initial weight value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            learning_rate: Learning rate for adjustments
        """
        self.name = name
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.learning_rate = learning_rate
        self.history = [(time.time(), value)]
        self.adjustment_history = []
        
        logger.debug(f"Created neural weight: {name} (value: {value:.4f})")
    
    def adjust(self, gradient: float) -> float:
        """
        Adjust weight value based on gradient.
        
        Args:
            gradient: Gradient direction and magnitude
            
        Returns:
            New weight value
        """
        # Calculate adjustment
        adjustment = self.learning_rate * gradient
        
        # Apply adjustment
        old_value = self.value
        self.value = max(self.min_value, min(self.max_value, self.value + adjustment))
        
        # Record adjustment
        timestamp = time.time()
        self.history.append((timestamp, self.value))
        self.adjustment_history.append((timestamp, gradient, adjustment, old_value, self.value))
        
        logger.debug(f"Adjusted neural weight: {self.name} ({old_value:.4f} -> {self.value:.4f}, gradient: {gradient:.4f})")
        
        return self.value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "learning_rate": self.learning_rate,
            "history": self.history[-10:],  # Save only the last 10 history points
            "last_updated": self.history[-1][0] if self.history else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralWeight':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Neural weight
        """
        weight = cls(
            name=data.get("name", "unknown"),
            value=data.get("value", 0.5),
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 1.0),
            learning_rate=data.get("learning_rate", 0.01)
        )
        
        # Restore history
        weight.history = data.get("history", [(time.time(), weight.value)])
        
        return weight

class WeightGroup:
    """
    Represents a group of related neural weights.
    """
    
    def __init__(self, name: str, weights: Dict[str, NeuralWeight] = None):
        """
        Initialize weight group.
        
        Args:
            name: Group name
            weights: Dictionary of neural weights
        """
        self.name = name
        self.weights = weights or {}
        
        logger.debug(f"Created weight group: {name} with {len(self.weights)} weights")
    
    def add_weight(self, weight: NeuralWeight) -> None:
        """
        Add a neural weight to the group.
        
        Args:
            weight: Neural weight to add
        """
        self.weights[weight.name] = weight
        logger.debug(f"Added weight {weight.name} to group {self.name}")
    
    def remove_weight(self, weight_name: str) -> bool:
        """
        Remove a neural weight from the group.
        
        Args:
            weight_name: Name of weight to remove
            
        Returns:
            Whether the weight was removed
        """
        if weight_name in self.weights:
            del self.weights[weight_name]
            logger.debug(f"Removed weight {weight_name} from group {self.name}")
            return True
        
        return False
    
    def get_weight(self, weight_name: str) -> Optional[NeuralWeight]:
        """
        Get a neural weight by name.
        
        Args:
            weight_name: Name of weight to get
            
        Returns:
            Neural weight, or None if not found
        """
        return self.weights.get(weight_name)
    
    def get_weight_value(self, weight_name: str) -> Optional[float]:
        """
        Get a neural weight value by name.
        
        Args:
            weight_name: Name of weight to get
            
        Returns:
            Neural weight value, or None if not found
        """
        weight = self.get_weight(weight_name)
        return weight.value if weight else None
    
    def adjust_weight(self, weight_name: str, gradient: float) -> Optional[float]:
        """
        Adjust a neural weight by name.
        
        Args:
            weight_name: Name of weight to adjust
            gradient: Gradient direction and magnitude
            
        Returns:
            New weight value, or None if weight not found
        """
        weight = self.get_weight(weight_name)
        
        if weight:
            return weight.adjust(gradient)
        
        return None
    
    def normalize_weights(self) -> None:
        """Normalize weights so they sum to 1.0."""
        total = sum(weight.value for weight in self.weights.values())
        
        if total > 0:
            for weight in self.weights.values():
                weight.value = weight.value / total
                weight.history.append((time.time(), weight.value))
            
            logger.debug(f"Normalized weights in group {self.name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "weights": {name: weight.to_dict() for name, weight in self.weights.items()},
            "total_weights": len(self.weights)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WeightGroup':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Weight group
        """
        group = cls(name=data.get("name", "unknown"))
        
        # Load weights
        weights_data = data.get("weights", {})
        
        for weight_name, weight_data in weights_data.items():
            group.weights[weight_name] = NeuralWeight.from_dict(weight_data)
        
        return group

class NeuralWeightAdjuster:
    """
    Adjusts neural weights based on performance feedback.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize neural weight adjuster.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or "neural_weights.json"
        self.weight_groups = {}
        self.performance_history = []
        self.max_history = 1000
        self.adjustment_callbacks = []
        
        # Load weights if file exists
        if os.path.exists(self.config_file):
            self._load_weights()
        
        logger.info("Neural weight adjuster initialized")
    
    def _load_weights(self) -> None:
        """Load weights from configuration file."""
        try:
            with open(self.config_file, "r") as f:
                data = json.load(f)
            
            # Load weight groups
            groups_data = data.get("weight_groups", {})
            
            for group_name, group_data in groups_data.items():
                self.weight_groups[group_name] = WeightGroup.from_dict(group_data)
            
            # Load performance history
            self.performance_history = data.get("performance_history", [])
            
            logger.info(f"Loaded {len(self.weight_groups)} weight groups from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading weights from {self.config_file}: {e}")
            self.weight_groups = {}
            self.performance_history = []
    
    def save_weights(self) -> None:
        """Save weights to configuration file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.config_file)), exist_ok=True)
            
            data = {
                "weight_groups": {name: group.to_dict() for name, group in self.weight_groups.items()},
                "performance_history": self.performance_history[-100:],  # Save only the last 100 performance points
                "last_updated": time.time()
            }
            
            with open(self.config_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.weight_groups)} weight groups to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving weights to {self.config_file}: {e}")
    
    def add_weight_group(self, group: WeightGroup) -> None:
        """
        Add a weight group.
        
        Args:
            group: Weight group to add
        """
        self.weight_groups[group.name] = group
        logger.info(f"Added weight group: {group.name}")
    
    def remove_weight_group(self, group_name: str) -> bool:
        """
        Remove a weight group.
        
        Args:
            group_name: Name of group to remove
            
        Returns:
            Whether the group was removed
        """
        if group_name in self.weight_groups:
            del self.weight_groups[group_name]
            logger.info(f"Removed weight group: {group_name}")
            return True
        
        return False
    
    def get_weight_group(self, group_name: str) -> Optional[WeightGroup]:
        """
        Get a weight group by name.
        
        Args:
            group_name: Name of group to get
            
        Returns:
            Weight group, or None if not found
        """
        return self.weight_groups.get(group_name)
    
    def create_weight(self, group_name: str, weight_name: str, value: float = 0.5,
                    min_value: float = 0.0, max_value: float = 1.0,
                    learning_rate: float = 0.01) -> Optional[NeuralWeight]:
        """
        Create a new neural weight in a group.
        
        Args:
            group_name: Name of group
            weight_name: Name of weight
            value: Initial weight value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            learning_rate: Learning rate for adjustments
            
        Returns:
            Created neural weight, or None if group not found
        """
        group = self.get_weight_group(group_name)
        
        if group is None:
            # Create group if it doesn't exist
            group = WeightGroup(group_name)
            self.add_weight_group(group)
        
        # Create weight
        weight = NeuralWeight(
            name=weight_name,
            value=value,
            min_value=min_value,
            max_value=max_value,
            learning_rate=learning_rate
        )
        
        # Add to group
        group.add_weight(weight)
        
        # Save weights
        self.save_weights()
        
        return weight
    
    def get_weight(self, group_name: str, weight_name: str) -> Optional[NeuralWeight]:
        """
        Get a neural weight from a group.
        
        Args:
            group_name: Name of group
            weight_name: Name of weight
            
        Returns:
            Neural weight, or None if not found
        """
        group = self.get_weight_group(group_name)
        
        if group:
            return group.get_weight(weight_name)
        
        return None
    
    def get_weight_value(self, group_name: str, weight_name: str) -> Optional[float]:
        """
        Get a neural weight value from a group.
        
        Args:
            group_name: Name of group
            weight_name: Name of weight
            
        Returns:
            Neural weight value, or None if not found
        """
        weight = self.get_weight(group_name, weight_name)
        
        if weight:
            return weight.value
        
        return None
    
    def adjust_weight(self, group_name: str, weight_name: str, gradient: float) -> Optional[float]:
        """
        Adjust a neural weight.
        
        Args:
            group_name: Name of group
            weight_name: Name of weight
            gradient: Gradient direction and magnitude
            
        Returns:
            New weight value, or None if weight not found
        """
        group = self.get_weight_group(group_name)
        
        if group:
            result = group.adjust_weight(weight_name, gradient)
            
            if result is not None:
                # Notify callbacks
                for callback in self.adjustment_callbacks:
                    try:
                        callback(group_name, weight_name, gradient, result)
                    except Exception as e:
                        logger.error(f"Error in adjustment callback: {e}")
                
                # Save weights
                self.save_weights()
            
            return result
        
        return None
    
    def add_adjustment_callback(self, callback: Callable[[str, str, float, float], None]) -> None:
        """
        Add a callback for weight adjustments.
        
        Args:
            callback: Callback function that receives group name, weight name, gradient, and new value
        """
        if callback not in self.adjustment_callbacks:
            self.adjustment_callbacks.append(callback)
            logger.debug(f"Added adjustment callback: {callback.__name__}")
    
    def remove_adjustment_callback(self, callback: Callable[[str, str, float, float], None]) -> bool:
        """
        Remove a callback.
        
        Args:
            callback: Callback function
            
        Returns:
            Whether the callback was removed
        """
        if callback in self.adjustment_callbacks:
            self.adjustment_callbacks.remove(callback)
            logger.debug(f"Removed adjustment callback: {callback.__name__}")
            return True
        
        return False
    
    def record_performance(self, performance_score: float, context: Dict[str, Any] = None) -> None:
        """
        Record a performance score for later analysis.
        
        Args:
            performance_score: Performance score (higher is better)
            context: Additional context information
        """
        timestamp = time.time()
        
        # Add to performance history
        self.performance_history.append({
            "timestamp": timestamp,
            "score": performance_score,
            "context": context or {}
        })
        
        # Trim history if needed
        if len(self.performance_history) > self.max_history:
            self.performance_history = self.performance_history[-self.max_history:]
        
        logger.debug(f"Recorded performance score: {performance_score:.4f}")
    
    def get_performance_trend(self, window_size: int = 10) -> Optional[float]:
        """
        Get performance trend over recent history.
        
        Args:
            window_size: Number of recent performance points to consider
            
        Returns:
            Trend coefficient (positive means improving, negative means declining), or None if not enough data
        """
        if len(self.performance_history) < window_size:
            return None
        
        # Get recent performance points
        recent = self.performance_history[-window_size:]
        
        # Extract timestamps and scores
        x = np.array([point["timestamp"] for point in recent])
        y = np.array([point["score"] for point in recent])
        
        # Normalize x for numerical stability
        x_mean = np.mean(x)
        x = x - x_mean
        
        # Calculate trend (slope of linear regression)
        try:
            trend = np.sum(x * y) / np.sum(x * x)
            return trend
        except:
            return None
    
    def auto_adjust_weights(self, group_name: str, performance_delta: float, 
                          learning_rate_scale: float = 1.0) -> Dict[str, float]:
        """
        Automatically adjust weights in a group based on performance change.
        
        Args:
            group_name: Name of group to adjust
            performance_delta: Change in performance since last adjustment
            learning_rate_scale: Scale factor for learning rate
            
        Returns:
            Dictionary mapping weight names to new values
        """
        group = self.get_weight_group(group_name)
        
        if group is None or not group.weights:
            return {}
        
        results = {}
        
        # Adjust each weight based on performance delta
        for weight_name, weight in group.weights.items():
            # Scale gradient by performance delta and learning rate scale
            gradient = performance_delta * learning_rate_scale
            
            # Adjust weight
            new_value = group.adjust_weight(weight_name, gradient)
            results[weight_name] = new_value
            
            # Notify callbacks
            for callback in self.adjustment_callbacks:
                try:
                    callback(group_name, weight_name, gradient, new_value)
                except Exception as e:
                    logger.error(f"Error in adjustment callback: {e}")
        
        # Normalize weights if needed
        group.normalize_weights()
        
        # Update results with normalized values
        for weight_name, weight in group.weights.items():
            results[weight_name] = weight.value
        
        # Save weights
        self.save_weights()
        
        return results
    
    def optimize_weights(self, group_name: str, objective_function: Callable[[Dict[str, float]], float],
                       iterations: int = 10, exploration_rate: float = 0.1) -> Dict[str, float]:
        """
        Optimize weights in a group using a black-box optimization approach.
        
        Args:
            group_name: Name of group to optimize
            objective_function: Function that evaluates weight values and returns a score (higher is better)
            iterations: Number of optimization iterations
            exploration_rate: Rate of exploration vs. exploitation
            
        Returns:
            Dictionary mapping weight names to optimized values
        """
        group = self.get_weight_group(group_name)
        
        if group is None or not group.weights:
            return {}
        
        # Track best weights and score
        best_weights = {name: weight.value for name, weight in group.weights.items()}
        best_score = objective_function(best_weights)
        
        # Record initial performance
        self.record_performance(best_score, {
            "group": group_name,
            "iteration": 0,
            "weights": best_weights.copy()
        })
        
        # Optimization loop
        for i in range(iterations):
            # Decide whether to explore or exploit
            if random.random() < exploration_rate:
                # Exploration: try random weight values
                test_weights = {}
                
                for name, weight in group.weights.items():
                    # Generate random value within allowed range
                    test_weights[name] = random.uniform(weight.min_value, weight.max_value)
                
                # Normalize if needed
                total = sum(test_weights.values())
                if total > 0:
                    test_weights = {name: value / total for name, value in test_weights.items()}
            else:
                # Exploitation: adjust best weights slightly
                test_weights = {}
                
                for name, value in best_weights.items():
                    weight = group.get_weight(name)
                    
                    if weight:
                        # Add some noise
                        noise = random.uniform(-0.1, 0.1) * (weight.max_value - weight.min_value)
                        test_value = value + noise
                        
                        # Ensure value is within range
                        test_weights[name] = max(weight.min_value, min(weight.max_value, test_value))
                
                # Normalize if needed
                total = sum(test_weights.values())
                if total > 0:
                    test_weights = {name: value / total for name, value in test_weights.items()}
            
            # Evaluate test weights
            test_score = objective_function(test_weights)
            
            # Record performance
            self.record_performance(test_score, {
                "group": group_name,
                "iteration": i + 1,
                "weights": test_weights.copy()
            })
            
            # Update best weights if better
            if test_score > best_score:
                best_weights = test_weights
                best_score = test_score
                
                logger.debug(f"Found better weights in iteration {i + 1}: score = {best_score:.4f}")
        
        # Apply best weights to group
        for name, value in best_weights.items():
            weight = group.get_weight(name)
            
            if weight:
                weight.value = value
                weight.history.append((time.time(), value))
        
        # Save weights
        self.save_weights()
        
        return best_weights
    
    def plot_weight_history(self, group_name: str, weight_name: str = None, 
                          output_file: str = None) -> str:
        """
        Plot weight history over time.
        
        Args:
            group_name: Name of group
            weight_name: Name of weight, or None to plot all weights in group
            output_file: Output file path
            
        Returns:
            Output file path
        """
        group = self.get_weight_group(group_name)
        
        if group is None:
            logger.warning(f"Weight group not found: {group_name}")
            return None
        
        # Determine which weights to plot
        if weight_name:
            weight = group.get_weight(weight_name)
            
            if weight is None:
                logger.warning(f"Weight not found: {weight_name}")
                return None
            
            weights = {weight_name: weight}
        else:
            weights = group.weights
        
        if not weights:
            logger.warning(f"No weights to plot in group: {group_name}")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot each weight's history
        for name, weight in weights.items():
            # Extract timestamps and values
            timestamps = [h[0] for h in weight.history]
            values = [h[1] for h in weight.history]
            
            # Convert timestamps to datetime
            datetimes = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
            
            # Plot weight history
            plt.plot(datetimes, values, label=name)
        
        # Add labels and title
        plt.xlabel("Time")
        plt.ylabel("Weight Value")
        
        title = f"Weight History for Group: {group_name}"
        if weight_name:
            title += f" - Weight: {weight_name}"
        
        plt.title(title)
        plt.grid(True)
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Create output file path if not provided
        if output_file is None:
            os.makedirs("weight_plots", exist_ok=True)
            output_file = f"weight_plots/weights_{group_name}_{int(time.time())}.png"
        
        # Save figure
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def plot_performance_history(self, window: int = None, output_file: str = None) -> str:
        """
        Plot performance history over time.
        
        Args:
            window: Number of recent performance points to plot, or None for all
            output_file: Output file path
            
        Returns:
            Output file path
        """
        if not self.performance_history:
            logger.warning("No performance history to plot")
            return None
        
        # Filter performance history
        history = self.performance_history
        
        if window is not None:
            history = history[-window:]
        
        if not history:
            logger.warning("No performance history to plot after filtering")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Extract timestamps and scores
        timestamps = [point["timestamp"] for point in history]
        scores = [point["score"] for point in history]
        
        # Convert timestamps to datetime
        datetimes = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Plot performance history
        plt.plot(datetimes, scores, "g-", marker="o")
        
        # Add trendline
        try:
            z = np.polyfit(timestamps, scores, 1)
            p = np.poly1d(z)
            plt.plot(datetimes, p(timestamps), "r--", label=f"Trend: {'Improving' if z[0] > 0 else 'Declining'}")
        except:
            pass
        
        # Add labels and title
        plt.xlabel("Time")
        plt.ylabel("Performance Score")
        plt.title("Performance History")
        plt.grid(True)
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Create output file path if not provided
        if output_file is None:
            os.makedirs("performance_plots", exist_ok=True)
            output_file = f"performance_plots/performance_{int(time.time())}.png"
        
        # Save figure
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """
        Generate a report of neural weights and performance.
        
        Args:
            output_file: Output file path
            
        Returns:
            Report data
        """
        # Create weight summary
        weight_summary = {}
        
        for group_name, group in self.weight_groups.items():
            weight_summary[group_name] = {
                "total_weights": len(group.weights),
                "weights": {name: weight.value for name, weight in group.weights.items()}
            }
        
        # Calculate performance statistics
        performance_stats = {}
        
        if self.performance_history:
            scores = [point["score"] for point in self.performance_history]
            
            performance_stats = {
                "count": len(scores),
                "min": min(scores),
                "max": max(scores),
                "mean": sum(scores) / len(scores),
                "latest": scores[-1]
            }
            
            # Calculate trend
            trend = self.get_performance_trend()
            
            if trend is not None:
                performance_stats["trend"] = trend
                performance_stats["trend_direction"] = "improving" if trend > 0 else "declining"
        
        # Create report
        report = {
            "timestamp": time.time(),
            "weight_groups": len(self.weight_groups),
            "total_weights": sum(len(group.weights) for group in self.weight_groups.values()),
            "weight_summary": weight_summary,
            "performance_stats": performance_stats
        }
        
        # Save report to file if specified
        if output_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Saved neural weight report to {output_file}")
            except Exception as e:
                logger.error(f"Error saving neural weight report to {output_file}: {e}")
        
        return report

# Example usage
def example_usage():
    # Create adjuster
    adjuster = NeuralWeightAdjuster()
    
    # Create a weight group for solution path selection
    adjuster.create_weight("solution_paths", "path_a", value=0.3)
    adjuster.create_weight("solution_paths", "path_b", value=0.4)
    adjuster.create_weight("solution_paths", "path_c", value=0.3)
    
    # Create a weight group for code analysis
    adjuster.create_weight("code_analysis", "syntax", value=0.5)
    adjuster.create_weight("code_analysis", "performance", value=0.3)
    adjuster.create_weight("code_analysis", "security", value=0.2)
    
    # Define an objective function for optimization
    def objective_function(weights):
        # In a real application, this would evaluate the weights in some way
        # For this example, we'll just return a simple function of the weights
        return weights["path_a"] * 0.7 + weights["path_b"] * 0.3 + weights["path_c"] * 0.5
    
    # Optimize weights
    optimized = adjuster.optimize_weights("solution_paths", objective_function, iterations=5)
    print("Optimized weights:", optimized)
    
    # Plot weight history
    adjuster.plot_weight_history("solution_paths")
    
    # Generate report
    report = adjuster.generate_report()
    print("Report:", report)

if __name__ == "__main__":
    example_usage()
