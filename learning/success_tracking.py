#!/usr/bin/env python3
"""
Success Rate Tracking Module

This module provides capabilities for tracking success rates of repair attempts
and solutions, enabling continuous improvement of the system.
"""

import os
import sys
import json
import logging
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("success_tracking.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SuccessTracking")

class SuccessTracker:
    """
    Tracks success rates of repair attempts and solutions.
    """
    
    def __init__(self, tracking_file: str = None):
        """
        Initialize success tracker.
        
        Args:
            tracking_file: Path to tracking database file
        """
        self.tracking_file = tracking_file or "success_tracking.json"
        self.tracking_data = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "categories": {},
            "history": [],
            "patterns": {},
            "solutions": {}
        }
        
        # Load tracking data if file exists
        if os.path.exists(self.tracking_file):
            self._load_tracking_data()
        
        logger.info("Success tracker initialized")
    
    def _load_tracking_data(self) -> None:
        """Load tracking data from database file."""
        try:
            with open(self.tracking_file, "r") as f:
                self.tracking_data = json.load(f)
            
            logger.info(f"Loaded tracking data from {self.tracking_file}")
        except Exception as e:
            logger.error(f"Error loading tracking data from {self.tracking_file}: {e}")
            # Initialize with empty data
            self.tracking_data = {
                "total_attempts": 0,
                "successful_attempts": 0,
                "categories": {},
                "history": [],
                "patterns": {},
                "solutions": {}
            }
    
    def save_tracking_data(self) -> None:
        """Save tracking data to database file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.tracking_file)), exist_ok=True)
            
            with open(self.tracking_file, "w") as f:
                json.dump(self.tracking_data, f, indent=2)
            
            logger.info(f"Saved tracking data to {self.tracking_file}")
        except Exception as e:
            logger.error(f"Error saving tracking data to {self.tracking_file}: {e}")
    
    def record_attempt(self, success: bool, category: str = None, 
                     pattern_id: str = None, solution_id: str = None,
                     context: Dict[str, Any] = None) -> None:
        """
        Record a repair attempt.
        
        Args:
            success: Whether the attempt was successful
            category: Repair category
            pattern_id: Bug pattern ID
            solution_id: Solution ID
            context: Additional context
        """
        # Update overall statistics
        self.tracking_data["total_attempts"] += 1
        
        if success:
            self.tracking_data["successful_attempts"] += 1
        
        # Update category statistics
        if category:
            if category not in self.tracking_data["categories"]:
                self.tracking_data["categories"][category] = {
                    "total_attempts": 0,
                    "successful_attempts": 0
                }
            
            self.tracking_data["categories"][category]["total_attempts"] += 1
            
            if success:
                self.tracking_data["categories"][category]["successful_attempts"] += 1
        
        # Update pattern statistics
        if pattern_id:
            if pattern_id not in self.tracking_data["patterns"]:
                self.tracking_data["patterns"][pattern_id] = {
                    "total_attempts": 0,
                    "successful_attempts": 0,
                    "solutions": {}
                }
            
            self.tracking_data["patterns"][pattern_id]["total_attempts"] += 1
            
            if success:
                self.tracking_data["patterns"][pattern_id]["successful_attempts"] += 1
            
            # Update solution statistics for this pattern
            if solution_id:
                if solution_id not in self.tracking_data["patterns"][pattern_id]["solutions"]:
                    self.tracking_data["patterns"][pattern_id]["solutions"][solution_id] = {
                        "total_attempts": 0,
                        "successful_attempts": 0
                    }
                
                self.tracking_data["patterns"][pattern_id]["solutions"][solution_id]["total_attempts"] += 1
                
                if success:
                    self.tracking_data["patterns"][pattern_id]["solutions"][solution_id]["successful_attempts"] += 1
        
        # Update solution statistics
        if solution_id:
            if solution_id not in self.tracking_data["solutions"]:
                self.tracking_data["solutions"][solution_id] = {
                    "total_attempts": 0,
                    "successful_attempts": 0,
                    "patterns": {}
                }
            
            self.tracking_data["solutions"][solution_id]["total_attempts"] += 1
            
            if success:
                self.tracking_data["solutions"][solution_id]["successful_attempts"] += 1
            
            # Update pattern statistics for this solution
            if pattern_id:
                if pattern_id not in self.tracking_data["solutions"][solution_id]["patterns"]:
                    self.tracking_data["solutions"][solution_id]["patterns"][pattern_id] = {
                        "total_attempts": 0,
                        "successful_attempts": 0
                    }
                
                self.tracking_data["solutions"][solution_id]["patterns"][pattern_id]["total_attempts"] += 1
                
                if success:
                    self.tracking_data["solutions"][solution_id]["patterns"][pattern_id]["successful_attempts"] += 1
        
        # Add to history
        history_entry = {
            "timestamp": time.time(),
            "success": success,
            "category": category,
            "pattern_id": pattern_id,
            "solution_id": solution_id,
            "context": context or {}
        }
        
        self.tracking_data["history"].append(history_entry)
        
        # Trim history if needed (keep last 1000 entries)
        if len(self.tracking_data["history"]) > 1000:
            self.tracking_data["history"] = self.tracking_data["history"][-1000:]
        
        # Save tracking data
        self.save_tracking_data()
        
        logger.debug(f"Recorded {'successful' if success else 'failed'} attempt in category {category}")
    
    def get_overall_success_rate(self) -> Optional[float]:
        """
        Get overall success rate.
        
        Returns:
            Success rate (0.0 to 1.0), or None if no attempts
        """
        if self.tracking_data["total_attempts"] > 0:
            return self.tracking_data["successful_attempts"] / self.tracking_data["total_attempts"]
        
        return None
    
    def get_category_success_rate(self, category: str) -> Optional[float]:
        """
        Get success rate for a category.
        
        Args:
            category: Category name
            
        Returns:
            Success rate (0.0 to 1.0), or None if no attempts
        """
        if category in self.tracking_data["categories"]:
            category_data = self.tracking_data["categories"][category]
            
            if category_data["total_attempts"] > 0:
                return category_data["successful_attempts"] / category_data["total_attempts"]
        
        return None
    
    def get_pattern_success_rate(self, pattern_id: str) -> Optional[float]:
        """
        Get success rate for a pattern.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Success rate (0.0 to 1.0), or None if no attempts
        """
        if pattern_id in self.tracking_data["patterns"]:
            pattern_data = self.tracking_data["patterns"][pattern_id]
            
            if pattern_data["total_attempts"] > 0:
                return pattern_data["successful_attempts"] / pattern_data["total_attempts"]
        
        return None
    
    def get_solution_success_rate(self, solution_id: str) -> Optional[float]:
        """
        Get success rate for a solution.
        
        Args:
            solution_id: Solution ID
            
        Returns:
            Success rate (0.0 to 1.0), or None if no attempts
        """
        if solution_id in self.tracking_data["solutions"]:
            solution_data = self.tracking_data["solutions"][solution_id]
            
            if solution_data["total_attempts"] > 0:
                return solution_data["successful_attempts"] / solution_data["total_attempts"]
        
        return None
    
    def get_best_solution_for_pattern(self, pattern_id: str, min_attempts: int = 5) -> Optional[str]:
        """
        Get the best solution for a pattern.
        
        Args:
            pattern_id: Pattern ID
            min_attempts: Minimum number of attempts required
            
        Returns:
            Solution ID, or None if no solutions meet the criteria
        """
        if pattern_id in self.tracking_data["patterns"]:
            pattern_data = self.tracking_data["patterns"][pattern_id]
            solutions = pattern_data["solutions"]
            
            best_solution_id = None
            best_success_rate = -1.0
            
            for solution_id, solution_data in solutions.items():
                if solution_data["total_attempts"] >= min_attempts:
                    success_rate = solution_data["successful_attempts"] / solution_data["total_attempts"]
                    
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_solution_id = solution_id
            
            return best_solution_id
        
        return None
    
    def get_category_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all categories.
        
        Returns:
            Dictionary mapping categories to statistics
        """
        result = {}
        
        for category, data in self.tracking_data["categories"].items():
            total = data["total_attempts"]
            successful = data["successful_attempts"]
            
            result[category] = {
                "total_attempts": total,
                "successful_attempts": successful,
                "success_rate": successful / total if total > 0 else None
            }
        
        return result
    
    def get_pattern_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all patterns.
        
        Returns:
            Dictionary mapping patterns to statistics
        """
        result = {}
        
        for pattern_id, data in self.tracking_data["patterns"].items():
            total = data["total_attempts"]
            successful = data["successful_attempts"]
            
            result[pattern_id] = {
                "total_attempts": total,
                "successful_attempts": successful,
                "success_rate": successful / total if total > 0 else None,
                "solutions_count": len(data["solutions"])
            }
        
        return result
    
    def get_solution_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all solutions.
        
        Returns:
            Dictionary mapping solutions to statistics
        """
        result = {}
        
        for solution_id, data in self.tracking_data["solutions"].items():
            total = data["total_attempts"]
            successful = data["successful_attempts"]
            
            result[solution_id] = {
                "total_attempts": total,
                "successful_attempts": successful,
                "success_rate": successful / total if total > 0 else None,
                "patterns_count": len(data["patterns"])
            }
        
        return result
    
    def get_success_trend(self, days: int = 30, interval_hours: int = 24) -> Dict[str, List[Any]]:
        """
        Get success rate trend over time.
        
        Args:
            days: Number of days to include
            interval_hours: Interval size in hours
            
        Returns:
            Dictionary with timestamps and success rates
        """
        # Calculate cutoff time
        cutoff = time.time() - (days * 24 * 60 * 60)
        
        # Filter history
        history = [entry for entry in self.tracking_data["history"] if entry["timestamp"] >= cutoff]
        
        if not history:
            return {"timestamps": [], "success_rates": []}
        
        # Sort history by timestamp
        history.sort(key=lambda entry: entry["timestamp"])
        
        # Calculate interval size in seconds
        interval = interval_hours * 60 * 60
        
        # Create intervals
        intervals = []
        current_time = history[0]["timestamp"]
        end_time = history[-1]["timestamp"] + interval
        
        while current_time <= end_time:
            intervals.append((current_time, current_time + interval))
            current_time += interval
        
        # Calculate success rate for each interval
        timestamps = []
        success_rates = []
        
        for start, end in intervals:
            interval_history = [entry for entry in history if start <= entry["timestamp"] < end]
            
            if interval_history:
                total = len(interval_history)
                successful = len([entry for entry in interval_history if entry["success"]])
                
                timestamps.append(start)
                success_rates.append(successful / total)
        
        return {"timestamps": timestamps, "success_rates": success_rates}
    
    def plot_success_trend(self, days: int = 30, interval_hours: int = 24, output_file: str = None) -> str:
        """
        Plot success rate trend over time.
        
        Args:
            days: Number of days to include
            interval_hours: Interval size in hours
            output_file: Output file path
            
        Returns:
            Output file path
        """
        # Get success trend
        trend = self.get_success_trend(days, interval_hours)
        
        if not trend["timestamps"]:
            logger.warning("No data to plot success trend")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot success rate trend
        timestamps = [datetime.datetime.fromtimestamp(ts) for ts in trend["timestamps"]]
        success_rates = trend["success_rates"]
        
        plt.plot(timestamps, success_rates, "g-", marker="o")
        
        # Add labels and title
        plt.xlabel("Time")
        plt.ylabel("Success Rate")
        plt.title(f"Repair Success Rate Trend (Last {days} Days)")
        plt.grid(True)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Create output file path if not provided
        if output_file is None:
            os.makedirs("success_plots", exist_ok=True)
            output_file = f"success_plots/success_trend_{int(time.time())}.png"
        
        # Save figure
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def plot_category_success_rates(self, output_file: str = None) -> str:
        """
        Plot success rates by category.
        
        Args:
            output_file: Output file path
            
        Returns:
            Output file path
        """
        # Get category statistics
        category_stats = self.get_category_statistics()
        
        if not category_stats:
            logger.warning("No categories to plot success rates")
            return None
        
        # Extract categories and success rates
        categories = []
        success_rates = []
        
        for category, stats in category_stats.items():
            if stats["success_rate"] is not None:
                categories.append(category)
                success_rates.append(stats["success_rate"])
        
        if not categories:
            logger.warning("No categories with success rates to plot")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot success rates
        plt.bar(categories, success_rates, color="green", alpha=0.7)
        
        # Add labels and title
        plt.xlabel("Category")
        plt.ylabel("Success Rate")
        plt.title("Repair Success Rates by Category")
        plt.grid(True, axis="y")
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        
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
            os.makedirs("success_plots", exist_ok=True)
            output_file = f"success_plots/category_success_rates_{int(time.time())}.png"
        
        # Save figure
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """
        Generate a success tracking report.
        
        Args:
            output_file: Output file path
            
        Returns:
            Report data
        """
        # Get overall statistics
        overall_success_rate = self.get_overall_success_rate()
        
        # Get category statistics
        category_stats = self.get_category_statistics()
        
        # Get pattern statistics
        pattern_stats = self.get_pattern_statistics()
        
        # Get solution statistics
        solution_stats = self.get_solution_statistics()
        
        # Get success trend
        success_trend = self.get_success_trend()
        
        # Create report
        report = {
            "timestamp": time.time(),
            "total_attempts": self.tracking_data["total_attempts"],
            "successful_attempts": self.tracking_data["successful_attempts"],
            "overall_success_rate": overall_success_rate,
            "categories": category_stats,
            "patterns": pattern_stats,
            "solutions": solution_stats,
            "success_trend": success_trend
        }
        
        # Save report to file if specified
        if output_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Saved success tracking report to {output_file}")
            except Exception as e:
                logger.error(f"Error saving success tracking report to {output_file}: {e}")
        
        return report
    
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get suggestions for improvement based on success tracking data.
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Check overall success rate
        overall_success_rate = self.get_overall_success_rate()
        
        if overall_success_rate is not None:
            if overall_success_rate < 0.5:
                suggestions.append({
                    "type": "overall",
                    "priority": "high",
                    "message": f"Overall success rate is low ({overall_success_rate:.1%}). Consider improving general repair strategies."
                })
            elif overall_success_rate < 0.7:
                suggestions.append({
                    "type": "overall",
                    "priority": "medium",
                    "message": f"Overall success rate is moderate ({overall_success_rate:.1%}). Look for specific areas to improve."
                })
        
        # Check category success rates
        category_stats = self.get_category_statistics()
        
        for category, stats in category_stats.items():
            if stats["total_attempts"] >= 10:
                if stats["success_rate"] < 0.4:
                    suggestions.append({
                        "type": "category",
                        "category": category,
                        "priority": "high",
                        "message": f"Success rate for category '{category}' is very low ({stats['success_rate']:.1%}). Needs immediate attention."
                    })
                elif stats["success_rate"] < 0.6:
                    suggestions.append({
                        "type": "category",
                        "category": category,
                        "priority": "medium",
                        "message": f"Success rate for category '{category}' is below average ({stats['success_rate']:.1%}). Consider targeted improvements."
                    })
        
        # Check pattern success rates
        pattern_stats = self.get_pattern_statistics()
        
        for pattern_id, stats in pattern_stats.items():
            if stats["total_attempts"] >= 5:
                if stats["success_rate"] < 0.3:
                    suggestions.append({
                        "type": "pattern",
                        "pattern_id": pattern_id,
                        "priority": "high",
                        "message": f"Success rate for pattern '{pattern_id}' is very low ({stats['success_rate']:.1%}). Needs new solution approaches."
                    })
        
        # Check solution success rates
        solution_stats = self.get_solution_statistics()
        
        for solution_id, stats in solution_stats.items():
            if stats["total_attempts"] >= 10:
                if stats["success_rate"] < 0.2:
                    suggestions.append({
                        "type": "solution",
                        "solution_id": solution_id,
                        "priority": "high",
                        "message": f"Solution '{solution_id}' has a very low success rate ({stats['success_rate']:.1%}). Consider replacing or significantly improving it."
                    })
        
        # Check success trend
        trend = self.get_success_trend(days=30, interval_hours=24 * 7)  # Weekly intervals
        
        if len(trend["timestamps"]) >= 2:
            first_rate = trend["success_rates"][0]
            last_rate = trend["success_rates"][-1]
            
            if last_rate < first_rate * 0.8:
                suggestions.append({
                    "type": "trend",
                    "priority": "high",
                    "message": f"Success rate has declined significantly over the past month (from {first_rate:.1%} to {last_rate:.1%}). Investigate recent changes."
                })
            elif last_rate > first_rate * 1.2:
                suggestions.append({
                    "type": "trend",
                    "priority": "low",
                    "message": f"Success rate has improved over the past month (from {first_rate:.1%} to {last_rate:.1%}). Continue current improvement strategies."
                })
        
        return suggestions

class CategoryTracker:
    """
    Tracks success rates by category.
    """
    
    def __init__(self, category: str, tracker: SuccessTracker):
        """
        Initialize category tracker.
        
        Args:
            category: Category name
            tracker: Success tracker
        """
        self.category = category
        self.tracker = tracker
    
    def record_attempt(self, success: bool, pattern_id: str = None, solution_id: str = None,
                     context: Dict[str, Any] = None) -> None:
        """
        Record a repair attempt in this category.
        
        Args:
            success: Whether the attempt was successful
            pattern_id: Bug pattern ID
            solution_id: Solution ID
            context: Additional context
        """
        self.tracker.record_attempt(success, self.category, pattern_id, solution_id, context)
    
    def get_success_rate(self) -> Optional[float]:
        """
        Get success rate for this category.
        
        Returns:
            Success rate (0.0 to 1.0), or None if no attempts
        """
        return self.tracker.get_category_success_rate(self.category)

# Example usage
def example_usage():
    # Create success tracker
    tracker = SuccessTracker()
    
    # Record some attempts
    tracker.record_attempt(True, "syntax_error", "pattern_1", "solution_1")
    tracker.record_attempt(False, "syntax_error", "pattern_1", "solution_2")
    tracker.record_attempt(True, "logic_error", "pattern_2", "solution_3")
    tracker.record_attempt(True, "logic_error", "pattern_2", "solution_3")
    tracker.record_attempt(False, "logic_error", "pattern_3", "solution_4")
    
    # Get success rates
    overall_rate = tracker.get_overall_success_rate()
    syntax_rate = tracker.get_category_success_rate("syntax_error")
    logic_rate = tracker.get_category_success_rate("logic_error")
    
    print(f"Overall success rate: {overall_rate:.1%}")
    print(f"Syntax error success rate: {syntax_rate:.1%}")
    print(f"Logic error success rate: {logic_rate:.1%}")
    
    # Generate report
    report = tracker.generate_report()
    print("Report:", report)
    
    # Get improvement suggestions
    suggestions = tracker.get_improvement_suggestions()
    print("Improvement suggestions:", suggestions)
    
    # Create a category tracker
    syntax_tracker = CategoryTracker("syntax_error", tracker)
    syntax_tracker.record_attempt(True, "pattern_1", "solution_5")
    
    syntax_rate = syntax_tracker.get_success_rate()
    print(f"Updated syntax error success rate: {syntax_rate:.1%}")

if __name__ == "__main__":
    example_usage()
