#!/usr/bin/env python3
"""
Weight Adjustment Module

This module provides neural weight adjustment capabilities for the learning system,
enabling dynamic optimization of the system's performance.
"""

import os
import sys
import json
import logging
import time
import random
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pathlib import Path
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("weight_adjustment.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("WeightAdjustment")

class WeightAdjuster:
    """
    Adjusts neural weights based on performance feedback.
    """
    
    def __init__(self, weights_file: str = None):
        """
        Initialize weight adjuster.
        
        Args:
            weights_file: Path to weights database file
        """
        self.weights_file = weights_file or "neural_weights.json"
        self.weights = {}
        self.weight_groups = {}
        self.performance_history = []
        self.max_history = 1000
        self.adjustment_callbacks = []
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.previous_adjustments = {}
        
        # Load weights if file exists
        if os.path.exists(self.weights_file):
            self._load_weights()
        
        logger.info("Weight adjuster initialized")
    
    def _load_weights(self) -> None:
        """Load weights from database file."""
        try:
            with open(self.weights_file, "r") as f:
                data = json.load(f)
            
            # Load weights
            self.weights = data.get("weights", {})
            
            # Load weight groups
            self.weight_groups = data.get("weight_groups", {})
            
            # Load performance history
            self.performance_history = data.get("performance_history", [])
            
            # Load configuration
            self.learning_rate = data.get("learning_rate", 0.01)
            self.momentum = data.get("momentum", 0.9)
            
            logger.info(f"Loaded {len(self.weights)} weights from {self.weights_file}")
        except Exception as e:
            logger.error(f"Error loading weights from {self.weights_file}: {e}")
            self.weights = {}
            self.weight_groups = {}
            self.performance_history = []
    
    def save_weights(self) -> None:
        """Save weights to database file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.weights_file)), exist_ok=True)
            
            data = {
                "weights": self.weights,
                "weight_groups": self.weight_groups,
                "performance_history": self.performance_history[-100:],  # Save only the last 100 performance points
                "learning_rate": self.learning_rate,
                "momentum": self.momentum,
                "last_updated": time.time()
            }
            
            with open(self.weights_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.weights)} weights to {self.weights_file}")
        except Exception as e:
            logger.error(f"Error saving weights to {self.weights_file}: {e}")
    
    def create_weight(self, weight_id: str, value: float = 0.5, group: str = None,
                    min_value: float = 0.0, max_value: float = 1.0,
                    description: str = None) -> Dict[str, Any]:
        """
        Create a new weight.
        
        Args:
            weight_id: Weight ID
            value: Initial weight value
            group: Weight group
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            description: Weight description
            
        Returns:
            Created weight
        """
        # Create weight
        weight = {
            "value": value,
            "min_value": min_value,
            "max_value": max_value,
            "description": description or f"Weight {weight_id}",
            "created_at": time.time(),
            "updated_at": time.time(),
            "history": [(time.time(), value)]
        }
        
        # Add to weights
        self.weights[weight_id] = weight
        
        # Add to group if specified
        if group:
            if group not in self.weight_groups:
                self.weight_groups[group] = []
            
            if weight_id not in self.weight_groups[group]:
                self.weight_groups[group].append(weight_id)
        
        # Initialize previous adjustment
        self.previous_adjustments[weight_id] = 0.0
        
        # Save weights
        self.save_weights()
        
        logger.info(f"Created weight: {weight_id} = {value}")
        
        return weight
    
    def get_weight(self, weight_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a weight by ID.
        
        Args:
            weight_id: Weight ID
            
        Returns:
            Weight, or None if not found
        """
        return self.weights.get(weight_id)
    
    def get_weight_value(self, weight_id: str) -> Optional[float]:
        """
        Get a weight value by ID.
        
        Args:
            weight_id: Weight ID
            
        Returns:
            Weight value, or None if not found
        """
        weight = self.get_weight(weight_id)
        return weight["value"] if weight else None
    
    def set_weight(self, weight_id: str, value: float) -> Optional[float]:
        """
        Set a weight value.
        
        Args:
            weight_id: Weight ID
            value: New weight value
            
        Returns:
            New weight value, or None if weight not found
        """
        weight = self.get_weight(weight_id)
        
        if weight:
            # Ensure value is within bounds
            value = max(weight["min_value"], min(weight["max_value"], value))
            
            # Update weight
            weight["value"] = value
            weight["updated_at"] = time.time()
            weight["history"].append((time.time(), value))
            
            # Trim history if needed
            if len(weight["history"]) > 100:
                weight["history"] = weight["history"][-100:]
            
            # Save weights
            self.save_weights()
            
            logger.debug(f"Set weight: {weight_id} = {value}")
            
            return value
        
        return None
    
    def adjust_weight(self, weight_id: str, adjustment: float) -> Optional[float]:
        """
        Adjust a weight by adding an adjustment value.
        
        Args:
            weight_id: Weight ID
            adjustment: Adjustment value
            
        Returns:
            New weight value, or None if weight not found
        """
        weight = self.get_weight(weight_id)
        
        if weight:
            # Apply momentum
            momentum_adjustment = adjustment + self.momentum * self.previous_adjustments.get(weight_id, 0.0)
            
            # Apply learning rate
            scaled_adjustment = self.learning_rate * momentum_adjustment
            
            # Calculate new value
            new_value = weight["value"] + scaled_adjustment
            
            # Ensure value is within bounds
            new_value = max(weight["min_value"], min(weight["max_value"], new_value))
            
            # Update weight
            weight["value"] = new_value
            weight["updated_at"] = time.time()
            weight["history"].append((time.time(), new_value))
            
            # Trim history if needed
            if len(weight["history"]) > 100:
                weight["history"] = weight["history"][-100:]
            
            # Save weights
            self.save_weights()
            
            # Store adjustment for momentum
            self.previous_adjustments[weight_id] = scaled_adjustment
            
            # Notify callbacks
            for callback in self.adjustment_callbacks:
                try:
                    callback(weight_id, adjustment, new_value)
                except Exception as e:
                    logger.error(f"Error in adjustment callback: {e}")
            
            logger.debug(f"Adjusted weight: {weight_id} += {scaled_adjustment} -> {new_value}")
            
            return new_value
        
        return None
    
    def adjust_group(self, group: str, adjustments: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust weights in a group.
        
        Args:
            group: Weight group
            adjustments: Dictionary mapping weight IDs to adjustment values
            
        Returns:
            Dictionary mapping weight IDs to new values
        """
        results = {}
        
        if group in self.weight_groups:
            for weight_id in self.weight_groups[group]:
                if weight_id in adjustments:
                    new_value = self.adjust_weight(weight_id, adjustments[weight_id])
                    
                    if new_value is not None:
                        results[weight_id] = new_value
        
        return results
    
    def normalize_group(self, group: str) -> Dict[str, float]:
        """
        Normalize weights in a group so they sum to 1.0.
        
        Args:
            group: Weight group
            
        Returns:
            Dictionary mapping weight IDs to new values
        """
        results = {}
        
        if group in self.weight_groups:
            # Get weight values
            weight_values = {}
            
            for weight_id in self.weight_groups[group]:
                weight = self.get_weight(weight_id)
                
                if weight:
                    weight_values[weight_id] = weight["value"]
            
            if weight_values:
                # Calculate sum
                total = sum(weight_values.values())
                
                if total > 0:
                    # Normalize weights
                    for weight_id, value in weight_values.items():
                        normalized_value = value / total
                        new_value = self.set_weight(weight_id, normalized_value)
                        
                        if new_value is not None:
                            results[weight_id] = new_value
        
        return results
    
    def record_performance(self, performance_score: float, context: Dict[str, Any] = None) -> None:
        """
        Record a performance score for later analysis.
        
        Args:
            performance_score: Performance score (higher is better)
            context: Additional context
        """
        timestamp = time.time()
        
        # Create performance record
        record = {
            "timestamp": timestamp,
            "score": performance_score,
            "context": context or {}
        }
        
        # Add to performance history
        self.performance_history.append(record)
        
        # Trim history if needed
        if len(self.performance_history) > self.max_history:
            self.performance_history = self.performance_history[-self.max_history:]
        
        # Save weights
        self.save_weights()
        
        logger.debug(f"Recorded performance score: {performance_score}")
    
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
        x = x - np.mean(x)
        
        # Calculate trend (slope of linear regression)
        if np.sum(x * x) == 0:
            return 0.0
        
        trend = np.sum(x * y) / np.sum(x * x)
        return trend
    
    def optimize_weights(self, weight_ids: List[str], objective_function: Callable[[Dict[str, float]], float],
                       iterations: int = 10, exploration_rate: float = 0.1) -> Dict[str, float]:
        """
        Optimize weights using a black-box optimization approach.
        
        Args:
            weight_ids: List of weight IDs to optimize
            objective_function: Function that takes a dictionary of weight values and returns a score (higher is better)
            iterations: Number of optimization iterations
            exploration_rate: Rate of exploration vs. exploitation
            
        Returns:
            Dictionary mapping weight IDs to optimized values
        """
        # Get current weight values
        current_values = {}
        
        for weight_id in weight_ids:
            weight = self.get_weight(weight_id)
            
            if weight:
                current_values[weight_id] = weight["value"]
        
        if not current_values:
            return {}
        
        # Track best weights and score
        best_values = current_values.copy()
        best_score = objective_function(best_values)
        
        # Record initial performance
        self.record_performance(best_score, {
            "action": "optimize_weights",
            "weights": best_values.copy(),
            "iteration": 0
        })
        
        # Optimization loop
        for i in range(iterations):
            # Decide whether to explore or exploit
            if random.random() < exploration_rate:
                # Exploration: try random weight values
                test_values = {}
                
                for weight_id in weight_ids:
                    weight = self.get_weight(weight_id)
                    
                    if weight:
                        # Generate random value within allowed range
                        test_values[weight_id] = random.uniform(weight["min_value"], weight["max_value"])
            else:
                # Exploitation: adjust best weights slightly
                test_values = {}
                
                for weight_id in weight_ids:
                    weight = self.get_weight(weight_id)
                    
                    if weight and weight_id in best_values:
                        # Add some noise
                        noise = random.uniform(-0.1, 0.1) * (weight["max_value"] - weight["min_value"])
                        test_value = best_values[weight_id] + noise
                        
                        # Ensure value is within range
                        test_values[weight_id] = max(weight["min_value"], min(weight["max_value"], test_value))
            
            # Evaluate test weights
            test_score = objective_function(test_values)
            
            # Record performance
            self.record_performance(test_score, {
                "action": "optimize_weights",
                "weights": test_values.copy(),
                "iteration": i + 1
            })
            
            # Update best weights if better
            if test_score > best_score:
                best_values = test_values.copy()
                best_score = test_score
                
                logger.debug(f"Found better weights in iteration {i + 1}: score = {best_score}")
        
        # Apply best weights
        for weight_id, value in best_values.items():
            self.set_weight(weight_id, value)
        
        return best_values
    
    def add_adjustment_callback(self, callback: Callable[[str, float, float], None]) -> None:
        """
        Add a callback for weight adjustments.
        
        Args:
            callback: Callback function that receives weight ID, adjustment, and new value
        """
        if callback not in self.adjustment_callbacks:
            self.adjustment_callbacks.append(callback)
            logger.debug(f"Added adjustment callback: {callback.__name__}")
    
    def remove_adjustment_callback(self, callback: Callable[[str, float, float], None]) -> bool:
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
    
    def get_weight_groups(self) -> Dict[str, List[str]]:
        """
        Get all weight groups.
        
        Returns:
            Dictionary mapping group names to lists of weight IDs
        """
        return self.weight_groups
    
    def get_group_weights(self, group: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all weights in a group.
        
        Args:
            group: Weight group
            
        Returns:
            Dictionary mapping weight IDs to weights
        """
        result = {}
        
        if group in self.weight_groups:
            for weight_id in self.weight_groups[group]:
                weight = self.get_weight(weight_id)
                
                if weight:
                    result[weight_id] = weight
        
        return result
    
    def create_group(self, group: str, weights: Dict[str, float] = None) -> List[str]:
        """
        Create a new weight group.
        
        Args:
            group: Group name
            weights: Dictionary mapping weight IDs to initial values
            
        Returns:
            List of weight IDs in the group
        """
        if group not in self.weight_groups:
            self.weight_groups[group] = []
        
        # Add weights if specified
        if weights:
            for weight_id, value in weights.items():
                if weight_id not in self.weights:
                    self.create_weight(weight_id, value, group)
                elif weight_id not in self.weight_groups[group]:
                    self.weight_groups[group].append(weight_id)
        
        # Save weights
        self.save_weights()
        
        logger.info(f"Created weight group: {group}")
        
        return self.weight_groups[group]
    
    def delete_group(self, group: str, delete_weights: bool = False) -> bool:
        """
        Delete a weight group.
        
        Args:
            group: Group name
            delete_weights: Whether to delete the weights in the group
            
        Returns:
            Whether the group was deleted
        """
        if group in self.weight_groups:
            if delete_weights:
                for weight_id in self.weight_groups[group]:
                    if weight_id in self.weights:
                        del self.weights[weight_id]
                        
                        if weight_id in self.previous_adjustments:
                            del self.previous_adjustments[weight_id]
            
            del self.weight_groups[group]
            
            # Save weights
            self.save_weights()
            
            logger.info(f"Deleted weight group: {group}")
            
            return True
        
        return False
    
    def delete_weight(self, weight_id: str) -> bool:
        """
        Delete a weight.
        
        Args:
            weight_id: Weight ID
            
        Returns:
            Whether the weight was deleted
        """
        if weight_id in self.weights:
            del self.weights[weight_id]
            
            # Remove from groups
            for group, weight_ids in self.weight_groups.items():
                if weight_id in weight_ids:
                    self.weight_groups[group].remove(weight_id)
            
            # Remove from previous adjustments
            if weight_id in self.previous_adjustments:
                del self.previous_adjustments[weight_id]
            
            # Save weights
            self.save_weights()
            
            logger.info(f"Deleted weight: {weight_id}")
            
            return True
        
        return False
    
    def plot_weight_history(self, weight_id: str, output_file: str = None) -> str:
        """
        Plot weight history over time.
        
        Args:
            weight_id: Weight ID
            output_file: Output file path
            
        Returns:
            Output file path
        """
        weight = self.get_weight(weight_id)
        
        if not weight or not weight["history"]:
            logger.warning(f"No history for weight: {weight_id}")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Extract timestamps and values
        timestamps = [h[0] for h in weight["history"]]
        values = [h[1] for h in weight["history"]]
        
        # Convert timestamps to datetime
        datetimes = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Plot weight history
        plt.plot(datetimes, values, "b-")
        plt.scatter(datetimes, values, color="b", alpha=0.5)
        
        # Add trendline
        if len(timestamps) > 1:
            z = np.polyfit(timestamps, values, 1)
            p = np.poly1d(z)
            
            plt.plot(
                datetimes,
                p(timestamps),
                "r--",
                label=f"Trend: {'Increasing' if z[0] > 0 else 'Decreasing'}"
            )
        
        # Add labels and title
        plt.xlabel("Time")
        plt.ylabel("Weight Value")
        plt.title(f"Weight History: {weight_id}")
        plt.grid(True)
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Create output file path if not provided
        if output_file is None:
            os.makedirs("weight_plots", exist_ok=True)
            output_file = f"weight_plots/weight_{weight_id}_{int(time.time())}.png"
        
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
        plt.plot(datetimes, scores, "g-")
        plt.scatter(datetimes, scores, color="g", alpha=0.5)
        
        # Add trendline
        if len(timestamps) > 1:
            z = np.polyfit(timestamps, scores, 1)
            p = np.poly1d(z)
            
            plt.plot(
                datetimes,
                p(timestamps),
                "r--",
                label=f"Trend: {'Improving' if z[0] > 0 else 'Declining'}"
            )
        
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
        Generate a report of weights and performance.
        
        Args:
            output_file: Output file path
            
        Returns:
            Report data
        """
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
        
        # Create weight statistics
        weight_stats = {}
        
        for weight_id, weight in self.weights.items():
            weight_stats[weight_id] = {
                "value": weight["value"],
                "min_value": weight["min_value"],
                "max_value": weight["max_value"],
                "description": weight["description"],
                "updated_at": weight["updated_at"]
            }
        
        # Create report
        report = {
            "timestamp": time.time(),
            "weights": weight_stats,
            "weight_groups": self.weight_groups,
            "performance_stats": performance_stats,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum
        }
        
        # Save report to file if specified
        if output_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Saved weight report to {output_file}")
            except Exception as e:
                logger.error(f"Error saving weight report to {output_file}: {e}")
        
        return report

# Example usage
def example_usage():
    # Create weight adjuster
    adjuster = WeightAdjuster()
    
    # Create weights
    adjuster.create_weight("detection_threshold", 0.7, "detection", 0.1, 1.0, "Threshold for bug detection")
    adjuster.create_weight("fix_confidence", 0.8, "fixing", 0.2, 1.0, "Confidence threshold for applying fixes")
    adjuster.create_weight("learning_rate", 0.01, "learning", 0.001, 0.1, "Learning rate for neural updates")
    
    # Create a weight group
    adjuster.create_group("solution_scoring", {
        "solution_a_weight": 0.3,
        "solution_b_weight": 0.5,
        "solution_c_weight": 0.2
    })
    
    # Normalize group
    adjuster.normalize_group("solution_scoring")
    
    # Define an objective function for optimization
    def objective_function(weights):
        # In a real application, this would evaluate the weights in some way
        # For this example, we'll just return a simple function of the weights
        return weights["solution_a_weight"] * 0.7 + weights["solution_b_weight"] * 0.3 + weights["solution_c_weight"] * 0.5
    
    # Optimize weights
    group_weights = adjuster.get_group_weights("solution_scoring")
    weight_ids = list(group_weights.keys())
    
    optimized = adjuster.optimize_weights(weight_ids, objective_function, iterations=5)
    print("Optimized weights:", optimized)
    
    # Plot weight history
    adjuster.plot_weight_history("solution_a_weight")
    
    # Generate report
    report = adjuster.generate_report()
    print("Report:", report)

if __name__ == "__main__":
    example_usage()
