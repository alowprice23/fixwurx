#!/usr/bin/env python3
"""
MTTR (Mean Time To Repair) Optimizer Module

This module provides MTTR optimization capabilities, analyzing repair times
and suggesting improvements to reduce the time required to fix issues.
"""

import os
import sys
import json
import logging
import time
import datetime
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mttr_optimization.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MTTROptimizer")

class RepairRecord:
    """
    Represents a single repair/fix record.
    """
    
    def __init__(self, repair_id: str, issue_type: str, start_time: float,
                end_time: float = None, success: bool = None, 
                solution_path: str = None, metadata: Dict[str, Any] = None):
        """
        Initialize repair record.
        
        Args:
            repair_id: Unique repair ID
            issue_type: Type of issue being repaired
            start_time: Repair start time (Unix timestamp)
            end_time: Repair end time (Unix timestamp)
            success: Whether the repair was successful
            solution_path: Solution path used
            metadata: Additional metadata
        """
        self.repair_id = repair_id
        self.issue_type = issue_type
        self.start_time = start_time
        self.end_time = end_time
        self.success = success
        self.solution_path = solution_path
        self.metadata = metadata or {}
        
        # Calculate duration if start and end times are available
        self.duration = None
        if self.start_time is not None and self.end_time is not None:
            self.duration = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "repair_id": self.repair_id,
            "issue_type": self.issue_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "success": self.success,
            "solution_path": self.solution_path,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RepairRecord':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Repair record
        """
        return cls(
            repair_id=data.get("repair_id"),
            issue_type=data.get("issue_type"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            success=data.get("success"),
            solution_path=data.get("solution_path"),
            metadata=data.get("metadata", {})
        )

class MTTROptimizer:
    """
    Optimizes Mean Time To Repair (MTTR).
    """
    
    def __init__(self, db_file: str = None):
        """
        Initialize MTTR optimizer.
        
        Args:
            db_file: Database file path
        """
        self.db_file = db_file or "mttr_data.json"
        self.repair_records = []
        self.issue_types = set()
        self.solution_paths = set()
        
        # Load data if file exists
        if os.path.exists(self.db_file):
            self._load_data()
        
        logger.info("MTTR optimizer initialized")
    
    def _load_data(self) -> None:
        """Load data from database file."""
        try:
            with open(self.db_file, "r") as f:
                data = json.load(f)
            
            # Load repair records
            self.repair_records = [RepairRecord.from_dict(record) for record in data.get("repair_records", [])]
            
            # Extract issue types and solution paths
            self.issue_types = set(record.issue_type for record in self.repair_records if record.issue_type)
            self.solution_paths = set(record.solution_path for record in self.repair_records if record.solution_path)
            
            logger.info(f"Loaded {len(self.repair_records)} repair records from {self.db_file}")
        except Exception as e:
            logger.error(f"Error loading data from {self.db_file}: {e}")
            self.repair_records = []
            self.issue_types = set()
            self.solution_paths = set()
    
    def save_data(self) -> None:
        """Save data to database file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_file)), exist_ok=True)
            
            data = {
                "repair_records": [record.to_dict() for record in self.repair_records],
                "last_updated": time.time()
            }
            
            with open(self.db_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.repair_records)} repair records to {self.db_file}")
        except Exception as e:
            logger.error(f"Error saving data to {self.db_file}: {e}")
    
    def add_repair_record(self, record: RepairRecord) -> None:
        """
        Add a repair record.
        
        Args:
            record: Repair record
        """
        self.repair_records.append(record)
        
        # Update issue types and solution paths
        if record.issue_type:
            self.issue_types.add(record.issue_type)
        if record.solution_path:
            self.solution_paths.add(record.solution_path)
        
        # Save data
        self.save_data()
        
        logger.info(f"Added repair record: {record.repair_id}")
    
    def start_repair(self, issue_type: str, metadata: Dict[str, Any] = None) -> str:
        """
        Start a new repair and return the repair ID.
        
        Args:
            issue_type: Type of issue being repaired
            metadata: Additional metadata
            
        Returns:
            Repair ID
        """
        import uuid
        
        # Generate a unique repair ID
        repair_id = str(uuid.uuid4())
        
        # Create a new repair record
        record = RepairRecord(
            repair_id=repair_id,
            issue_type=issue_type,
            start_time=time.time(),
            metadata=metadata
        )
        
        # Add record
        self.add_repair_record(record)
        
        return repair_id
    
    def end_repair(self, repair_id: str, success: bool, solution_path: str = None, 
                 metadata_updates: Dict[str, Any] = None) -> None:
        """
        End a repair.
        
        Args:
            repair_id: Repair ID
            success: Whether the repair was successful
            solution_path: Solution path used
            metadata_updates: Metadata updates
        """
        # Find the repair record
        for record in self.repair_records:
            if record.repair_id == repair_id:
                # Update record
                record.end_time = time.time()
                record.success = success
                record.solution_path = solution_path
                
                # Update duration
                if record.start_time is not None:
                    record.duration = record.end_time - record.start_time
                
                # Update metadata
                if metadata_updates:
                    record.metadata.update(metadata_updates)
                
                # Save data
                self.save_data()
                
                logger.info(f"Ended repair: {repair_id} (success: {success}, duration: {record.duration:.2f}s)")
                return
        
        logger.warning(f"Repair record not found: {repair_id}")
    
    def get_mttr(self, issue_type: str = None, solution_path: str = None,
                time_period: int = None, success_only: bool = True) -> Optional[float]:
        """
        Get Mean Time To Repair (MTTR).
        
        Args:
            issue_type: Filter by issue type
            solution_path: Filter by solution path
            time_period: Time period in seconds to consider
            success_only: Whether to consider only successful repairs
            
        Returns:
            MTTR in seconds, or None if no matching repairs
        """
        # Filter repair records
        records = self.repair_records
        
        # Filter by issue type
        if issue_type:
            records = [r for r in records if r.issue_type == issue_type]
        
        # Filter by solution path
        if solution_path:
            records = [r for r in records if r.solution_path == solution_path]
        
        # Filter by time period
        if time_period:
            cutoff = time.time() - time_period
            records = [r for r in records if r.start_time >= cutoff]
        
        # Filter by success
        if success_only:
            records = [r for r in records if r.success]
        
        # Filter out records without duration
        records = [r for r in records if r.duration is not None]
        
        if not records:
            return None
        
        # Calculate MTTR
        durations = [r.duration for r in records]
        mttr = statistics.mean(durations) if durations else None
        
        return mttr
    
    def get_mttr_by_issue_type(self, time_period: int = None, 
                             success_only: bool = True) -> Dict[str, float]:
        """
        Get MTTR by issue type.
        
        Args:
            time_period: Time period in seconds to consider
            success_only: Whether to consider only successful repairs
            
        Returns:
            Dictionary mapping issue types to MTTR
        """
        result = {}
        
        for issue_type in self.issue_types:
            mttr = self.get_mttr(issue_type=issue_type, time_period=time_period, success_only=success_only)
            if mttr is not None:
                result[issue_type] = mttr
        
        return result
    
    def get_mttr_by_solution_path(self, issue_type: str = None, time_period: int = None,
                                success_only: bool = True) -> Dict[str, float]:
        """
        Get MTTR by solution path.
        
        Args:
            issue_type: Filter by issue type
            time_period: Time period in seconds to consider
            success_only: Whether to consider only successful repairs
            
        Returns:
            Dictionary mapping solution paths to MTTR
        """
        result = {}
        
        for solution_path in self.solution_paths:
            mttr = self.get_mttr(issue_type=issue_type, solution_path=solution_path, 
                               time_period=time_period, success_only=success_only)
            if mttr is not None:
                result[solution_path] = mttr
        
        return result
    
    def get_success_rate(self, issue_type: str = None, solution_path: str = None,
                       time_period: int = None) -> Optional[float]:
        """
        Get repair success rate.
        
        Args:
            issue_type: Filter by issue type
            solution_path: Filter by solution path
            time_period: Time period in seconds to consider
            
        Returns:
            Success rate (0.0 to 1.0), or None if no matching repairs
        """
        # Filter repair records
        records = self.repair_records
        
        # Filter by issue type
        if issue_type:
            records = [r for r in records if r.issue_type == issue_type]
        
        # Filter by solution path
        if solution_path:
            records = [r for r in records if r.solution_path == solution_path]
        
        # Filter by time period
        if time_period:
            cutoff = time.time() - time_period
            records = [r for r in records if r.start_time >= cutoff]
        
        # Filter out records without success flag
        records = [r for r in records if r.success is not None]
        
        if not records:
            return None
        
        # Calculate success rate
        successes = len([r for r in records if r.success])
        success_rate = successes / len(records)
        
        return success_rate
    
    def get_success_rate_by_issue_type(self, time_period: int = None) -> Dict[str, float]:
        """
        Get success rate by issue type.
        
        Args:
            time_period: Time period in seconds to consider
            
        Returns:
            Dictionary mapping issue types to success rate
        """
        result = {}
        
        for issue_type in self.issue_types:
            success_rate = self.get_success_rate(issue_type=issue_type, time_period=time_period)
            if success_rate is not None:
                result[issue_type] = success_rate
        
        return result
    
    def get_success_rate_by_solution_path(self, issue_type: str = None, 
                                        time_period: int = None) -> Dict[str, float]:
        """
        Get success rate by solution path.
        
        Args:
            issue_type: Filter by issue type
            time_period: Time period in seconds to consider
            
        Returns:
            Dictionary mapping solution paths to success rate
        """
        result = {}
        
        for solution_path in self.solution_paths:
            success_rate = self.get_success_rate(issue_type=issue_type, solution_path=solution_path, 
                                               time_period=time_period)
            if success_rate is not None:
                result[solution_path] = success_rate
        
        return result
    
    def get_mttr_trend(self, issue_type: str = None, solution_path: str = None,
                     time_bins: int = 10, success_only: bool = True) -> Dict[str, List[Any]]:
        """
        Get MTTR trend over time.
        
        Args:
            issue_type: Filter by issue type
            solution_path: Filter by solution path
            time_bins: Number of time bins
            success_only: Whether to consider only successful repairs
            
        Returns:
            Dictionary with timestamps and MTTRs
        """
        # Filter repair records
        records = self.repair_records
        
        # Filter by issue type
        if issue_type:
            records = [r for r in records if r.issue_type == issue_type]
        
        # Filter by solution path
        if solution_path:
            records = [r for r in records if r.solution_path == solution_path]
        
        # Filter by success
        if success_only:
            records = [r for r in records if r.success]
        
        # Filter out records without duration
        records = [r for r in records if r.duration is not None]
        
        if not records:
            return {"timestamps": [], "mttrs": []}
        
        # Sort records by start time
        records.sort(key=lambda r: r.start_time)
        
        # Create time bins
        min_time = records[0].start_time
        max_time = records[-1].start_time
        
        if min_time == max_time:
            return {
                "timestamps": [min_time],
                "mttrs": [statistics.mean([r.duration for r in records])]
            }
        
        bin_width = (max_time - min_time) / time_bins
        bins = []
        
        for i in range(time_bins):
            bin_start = min_time + i * bin_width
            bin_end = min_time + (i + 1) * bin_width
            
            # Find records in this bin
            bin_records = [r for r in records if bin_start <= r.start_time < bin_end]
            
            if bin_records:
                bin_mttr = statistics.mean([r.duration for r in bin_records])
                bins.append((bin_start, bin_mttr))
        
        # Return trend data
        return {
            "timestamps": [b[0] for b in bins],
            "mttrs": [b[1] for b in bins]
        }
    
    def get_repair_count_trend(self, issue_type: str = None, solution_path: str = None,
                             time_bins: int = 10, success_only: bool = None) -> Dict[str, List[Any]]:
        """
        Get repair count trend over time.
        
        Args:
            issue_type: Filter by issue type
            solution_path: Filter by solution path
            time_bins: Number of time bins
            success_only: Whether to consider only successful repairs
            
        Returns:
            Dictionary with timestamps and counts
        """
        # Filter repair records
        records = self.repair_records
        
        # Filter by issue type
        if issue_type:
            records = [r for r in records if r.issue_type == issue_type]
        
        # Filter by solution path
        if solution_path:
            records = [r for r in records if r.solution_path == solution_path]
        
        # Filter by success
        if success_only is not None:
            records = [r for r in records if r.success == success_only]
        
        if not records:
            return {"timestamps": [], "counts": []}
        
        # Sort records by start time
        records.sort(key=lambda r: r.start_time)
        
        # Create time bins
        min_time = records[0].start_time
        max_time = records[-1].start_time
        
        if min_time == max_time:
            return {
                "timestamps": [min_time],
                "counts": [len(records)]
            }
        
        bin_width = (max_time - min_time) / time_bins
        bins = []
        
        for i in range(time_bins):
            bin_start = min_time + i * bin_width
            bin_end = min_time + (i + 1) * bin_width
            
            # Find records in this bin
            bin_records = [r for r in records if bin_start <= r.start_time < bin_end]
            
            bins.append((bin_start, len(bin_records)))
        
        # Return trend data
        return {
            "timestamps": [b[0] for b in bins],
            "counts": [b[1] for b in bins]
        }
    
    def recommend_solution_path(self, issue_type: str, time_period: int = None) -> Optional[str]:
        """
        Recommend the best solution path for an issue type based on MTTR and success rate.
        
        Args:
            issue_type: Issue type
            time_period: Time period in seconds to consider
            
        Returns:
            Recommended solution path, or None if no data
        """
        # Get MTTR by solution path
        mttr_by_path = self.get_mttr_by_solution_path(issue_type=issue_type, time_period=time_period)
        
        # Get success rate by solution path
        success_rate_by_path = self.get_success_rate_by_solution_path(issue_type=issue_type, time_period=time_period)
        
        if not mttr_by_path or not success_rate_by_path:
            return None
        
        # Calculate score for each solution path (lower is better)
        # Score = MTTR * (1 / success_rate)
        scores = {}
        
        for path in mttr_by_path:
            if path in success_rate_by_path and success_rate_by_path[path] > 0:
                scores[path] = mttr_by_path[path] * (1 / success_rate_by_path[path])
        
        if not scores:
            return None
        
        # Return solution path with lowest score
        return min(scores.items(), key=lambda x: x[1])[0]
    
    def get_mttr_factors(self, time_period: int = None) -> Dict[str, float]:
        """
        Get factors that affect MTTR based on metadata analysis.
        
        Args:
            time_period: Time period in seconds to consider
            
        Returns:
            Dictionary mapping factors to correlation coefficients
        """
        # Filter repair records
        records = self.repair_records
        
        # Filter by time period
        if time_period:
            cutoff = time.time() - time_period
            records = [r for r in records if r.start_time >= cutoff]
        
        # Filter out records without duration
        records = [r for r in records if r.duration is not None]
        
        if not records:
            return {}
        
        # Extract metadata factors that appear in multiple records
        factors = {}
        
        for record in records:
            for key, value in record.metadata.items():
                if isinstance(value, (int, float)):
                    if key not in factors:
                        factors[key] = []
                    
                    factors[key].append((value, record.duration))
        
        # Filter out factors that don't have enough data points
        factors = {k: v for k, v in factors.items() if len(v) >= 5}
        
        if not factors:
            return {}
        
        # Calculate correlation coefficients
        correlations = {}
        
        for factor, data_points in factors.items():
            x = [d[0] for d in data_points]
            y = [d[1] for d in data_points]
            
            try:
                correlation = np.corrcoef(x, y)[0, 1]
                correlations[factor] = correlation
            except:
                pass
        
        return correlations
    
    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """
        Generate an MTTR optimization report.
        
        Args:
            output_file: Output file path
            
        Returns:
            Report data
        """
        # Calculate overall metrics
        overall_mttr = self.get_mttr()
        overall_success_rate = self.get_success_rate()
        
        # Calculate metrics by issue type
        mttr_by_issue_type = self.get_mttr_by_issue_type()
        success_rate_by_issue_type = self.get_success_rate_by_issue_type()
        
        # Calculate MTTR trend
        mttr_trend = self.get_mttr_trend()
        
        # Calculate repair count trend
        repair_count_trend = self.get_repair_count_trend()
        
        # Get MTTR factors
        mttr_factors = self.get_mttr_factors()
        
        # Generate recommendations
        recommendations = []
        
        # Recommendation 1: Issue types with high MTTR
        if mttr_by_issue_type:
            high_mttr_issues = sorted(mttr_by_issue_type.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for issue_type, mttr in high_mttr_issues:
                recommendations.append({
                    "type": "high_mttr",
                    "issue_type": issue_type,
                    "mttr": mttr,
                    "message": f"High MTTR for issue type '{issue_type}': {mttr:.2f} seconds"
                })
        
        # Recommendation 2: Issue types with low success rate
        if success_rate_by_issue_type:
            low_success_issues = sorted(success_rate_by_issue_type.items(), key=lambda x: x[1])[:3]
            
            for issue_type, success_rate in low_success_issues:
                recommendations.append({
                    "type": "low_success_rate",
                    "issue_type": issue_type,
                    "success_rate": success_rate,
                    "message": f"Low success rate for issue type '{issue_type}': {success_rate:.2%}"
                })
        
        # Recommendation 3: Best solution paths
        for issue_type in self.issue_types:
            recommended_path = self.recommend_solution_path(issue_type)
            
            if recommended_path:
                recommendations.append({
                    "type": "solution_path",
                    "issue_type": issue_type,
                    "solution_path": recommended_path,
                    "message": f"Recommended solution path for issue type '{issue_type}': {recommended_path}"
                })
        
        # Create report
        report = {
            "timestamp": time.time(),
            "repair_record_count": len(self.repair_records),
            "overall_mttr": overall_mttr,
            "overall_success_rate": overall_success_rate,
            "mttr_by_issue_type": mttr_by_issue_type,
            "success_rate_by_issue_type": success_rate_by_issue_type,
            "mttr_trend": mttr_trend,
            "repair_count_trend": repair_count_trend,
            "mttr_factors": mttr_factors,
            "recommendations": recommendations
        }
        
        # Save report to file if specified
        if output_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Saved MTTR report to {output_file}")
            except Exception as e:
                logger.error(f"Error saving MTTR report to {output_file}: {e}")
        
        return report
    
    def plot_mttr_trend(self, issue_type: str = None, solution_path: str = None,
                      output_file: str = None) -> str:
        """
        Plot MTTR trend over time.
        
        Args:
            issue_type: Filter by issue type
            solution_path: Filter by solution path
            output_file: Output file path
            
        Returns:
            Output file path
        """
        # Get MTTR trend
        trend_data = self.get_mttr_trend(issue_type=issue_type, solution_path=solution_path)
        
        if not trend_data["timestamps"]:
            logger.warning("No data to plot MTTR trend")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot MTTR trend
        timestamps = [datetime.datetime.fromtimestamp(ts) for ts in trend_data["timestamps"]]
        mttrs = trend_data["mttrs"]
        
        plt.plot(timestamps, mttrs, "b-", marker="o")
        
        # Add labels and title
        plt.xlabel("Time")
        plt.ylabel("MTTR (seconds)")
        
        title = "MTTR Trend"
        if issue_type:
            title += f" for Issue Type: {issue_type}"
        if solution_path:
            title += f" with Solution Path: {solution_path}"
        
        plt.title(title)
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Create output file path if not provided
        if output_file is None:
            os.makedirs("mttr_plots", exist_ok=True)
            output_file = f"mttr_plots/mttr_trend_{int(time.time())}.png"
        
        # Save figure
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def plot_success_rate_by_issue_type(self, output_file: str = None) -> str:
        """
        Plot success rate by issue type.
        
        Args:
            output_file: Output file path
            
        Returns:
            Output file path
        """
        # Get success rate by issue type
        success_rates = self.get_success_rate_by_issue_type()
        
        if not success_rates:
            logger.warning("No data to plot success rate by issue type")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot success rate by issue type
        issue_types = list(success_rates.keys())
        rates = [success_rates[it] for it in issue_types]
        
        plt.bar(issue_types, rates, color="green", alpha=0.7)
        
        # Add labels and title
        plt.xlabel("Issue Type")
        plt.ylabel("Success Rate")
        plt.title("Repair Success Rate by Issue Type")
        plt.grid(True, axis="y")
        
        # Add percentage labels
        for i, rate in enumerate(rates):
            plt.text(i, rate + 0.02, f"{rate:.1%}", ha="center")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        
        # Set y-axis limits
        plt.ylim(0, 1.1)
        
        # Tight layout
        plt.tight_layout()
        
        # Create output file path if not provided
        if output_file is None:
            os.makedirs("mttr_plots", exist_ok=True)
            output_file = f"mttr_plots/success_rate_by_issue_type_{int(time.time())}.png"
        
        # Save figure
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def plot_mttr_by_issue_type(self, output_file: str = None) -> str:
        """
        Plot MTTR by issue type.
        
        Args:
            output_file: Output file path
            
        Returns:
            Output file path
        """
        # Get MTTR by issue type
        mttrs = self.get_mttr_by_issue_type()
        
        if not mttrs:
            logger.warning("No data to plot MTTR by issue type")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot MTTR by issue type
        issue_types = list(mttrs.keys())
        mttr_values = [mttrs[it] for it in issue_types]
        
        plt.bar(issue_types, mttr_values, color="blue", alpha=0.7)
        
        # Add labels and title
        plt.xlabel("Issue Type")
        plt.ylabel("MTTR (seconds)")
        plt.title("Mean Time To Repair by Issue Type")
        plt.grid(True, axis="y")
        
        # Add value labels
        for i, mttr in enumerate(mttr_values):
            plt.text(i, mttr + max(mttr_values) * 0.02, f"{mttr:.1f}s", ha="center")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        
        #
