"""
FixWurx Time-Series Database

This module implements a time-series database for the Auditor agent to track 
metrics over time. It provides functionality for storing, querying, and 
analyzing time-series data such as energy metrics, proof metrics, and
performance benchmarks.

See docs/auditor_agent_specification.md for full specification.
"""

import os
import json
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [TimeSeriesDB] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('time_series_database')


class TimePoint:
    """
    Represents a single point in a time-series.
    
    Each time point has a timestamp and a set of values.
    """
    
    def __init__(self, timestamp: datetime.datetime, values: Dict[str, float]):
        """
        Initialize a time point.
        
        Args:
            timestamp: The timestamp of the point
            values: Dictionary of metric values
        """
        self.timestamp = timestamp
        self.values = values
    
    def to_dict(self) -> Dict:
        """Convert time point to dictionary representation"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "values": self.values
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TimePoint':
        """Create time point from dictionary representation"""
        try:
            timestamp = datetime.datetime.fromisoformat(data["timestamp"])
            return cls(
                timestamp=timestamp,
                values=data["values"]
            )
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to create TimePoint from dict: {e}")
            # Return a default time point if parsing fails
            return cls(
                timestamp=datetime.datetime.now(),
                values={}
            )


class TimeSeries:
    """
    Represents a time-series of metric values.
    
    Each time-series has a name, description, unit, and a list of time points.
    """
    
    def __init__(self, name: str, description: str = "", unit: str = "", points: List[TimePoint] = None):
        """
        Initialize a time-series.
        
        Args:
            name: The name of the time-series
            description: Description of the time-series
            unit: Unit of measurement
            points: List of time points
        """
        self.name = name
        self.description = description
        self.unit = unit
        self.points = points or []
    
    def add_point(self, point: TimePoint) -> None:
        """Add a time point to the time-series"""
        self.points.append(point)
        
        # Sort points by timestamp
        self.points.sort(key=lambda p: p.timestamp)
    
    def get_points(self, start_time: datetime.datetime = None, end_time: datetime.datetime = None) -> List[TimePoint]:
        """
        Get time points within a specified time range.
        
        Args:
            start_time: Start time of the range (inclusive)
            end_time: End time of the range (inclusive)
            
        Returns:
            List of time points within the range
        """
        if start_time is None and end_time is None:
            return self.points
        
        result = []
        for point in self.points:
            if start_time is not None and point.timestamp < start_time:
                continue
            if end_time is not None and point.timestamp > end_time:
                continue
            result.append(point)
        
        return result
    
    def get_values(self, metric_name: str, start_time: datetime.datetime = None, end_time: datetime.datetime = None) -> List[Tuple[datetime.datetime, float]]:
        """
        Get values for a specific metric within a time range.
        
        Args:
            metric_name: Name of the metric
            start_time: Start time of the range (inclusive)
            end_time: End time of the range (inclusive)
            
        Returns:
            List of (timestamp, value) tuples
        """
        points = self.get_points(start_time, end_time)
        result = []
        
        for point in points:
            if metric_name in point.values:
                result.append((point.timestamp, point.values[metric_name]))
        
        return result
    
    def get_latest_value(self, metric_name: str) -> Optional[Tuple[datetime.datetime, float]]:
        """
        Get the latest value for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Tuple of (timestamp, value) or None if no values exist
        """
        if not self.points:
            return None
        
        # Get the latest point
        latest_point = max(self.points, key=lambda p: p.timestamp)
        
        if metric_name in latest_point.values:
            return (latest_point.timestamp, latest_point.values[metric_name])
        
        return None
    
    def to_dict(self) -> Dict:
        """Convert time-series to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "points": [point.to_dict() for point in self.points]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TimeSeries':
        """Create time-series from dictionary representation"""
        try:
            points = [TimePoint.from_dict(point_data) for point_data in data.get("points", [])]
            return cls(
                name=data["name"],
                description=data.get("description", ""),
                unit=data.get("unit", ""),
                points=points
            )
        except KeyError as e:
            logger.error(f"Failed to create TimeSeries from dict: {e}")
            # Return a default time-series if parsing fails
            return cls(
                name="unknown",
                description="",
                unit="",
                points=[]
            )
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert time-series to pandas DataFrame.
        
        Returns:
            DataFrame with timestamp as index and metrics as columns
        """
        if not self.points:
            return pd.DataFrame()
        
        # Extract all metric names
        metric_names = set()
        for point in self.points:
            metric_names.update(point.values.keys())
        
        # Create data dict
        data = {
            "timestamp": [point.timestamp for point in self.points]
        }
        
        for metric_name in metric_names:
            data[metric_name] = [point.values.get(metric_name, np.nan) for point in self.points]
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        return df


class TimeSeriesDatabase:
    """
    Time-series database implementation for the Auditor agent.
    
    Provides functionality for storing, querying, and analyzing time-series
    data such as energy metrics, proof metrics, and performance benchmarks.
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize the time-series database.
        
        Args:
            storage_path: Path to the storage directory
        """
        self.storage_path = storage_path
        self.time_series = {}  # Dict of name -> TimeSeries
        self._load_time_series()
    
    def create_time_series(self, name: str, description: str = "", unit: str = "") -> TimeSeries:
        """
        Create a new time-series.
        
        Args:
            name: The name of the time-series
            description: Description of the time-series
            unit: Unit of measurement
            
        Returns:
            The created time-series
        """
        if name in self.time_series:
            logger.warning(f"Time-series {name} already exists")
            return self.time_series[name]
        
        time_series = TimeSeries(name, description, unit)
        self.time_series[name] = time_series
        
        logger.info(f"Created time-series {name}")
        
        # Save time-series
        self._save_time_series(time_series)
        
        return time_series
    
    def get_time_series(self, name: str) -> Optional[TimeSeries]:
        """
        Get a time-series by name.
        
        Args:
            name: The name of the time-series
            
        Returns:
            The time-series, or None if it doesn't exist
        """
        if name not in self.time_series:
            logger.warning(f"Time-series {name} does not exist")
            return None
        
        return self.time_series[name]
    
    def add_point(self, series_name: str, timestamp: datetime.datetime, values: Dict[str, float]) -> bool:
        """
        Add a data point to a time-series.
        
        Args:
            series_name: The name of the time-series
            timestamp: The timestamp of the point
            values: Dictionary of metric values
            
        Returns:
            True if the point was added, False if the time-series doesn't exist
        """
        time_series = self.get_time_series(series_name)
        if time_series is None:
            return False
        
        point = TimePoint(timestamp, values)
        time_series.add_point(point)
        
        logger.info(f"Added point to {series_name} at {timestamp}")
        
        # Save time-series
        self._save_time_series(time_series)
        
        return True
    
    def get_metric_values(self, series_name: str, metric_name: str, 
                         start_time: datetime.datetime = None,
                         end_time: datetime.datetime = None) -> List[Tuple[datetime.datetime, float]]:
        """
        Get values for a specific metric within a time range.
        
        Args:
            series_name: The name of the time-series
            metric_name: Name of the metric
            start_time: Start time of the range (inclusive)
            end_time: End time of the range (inclusive)
            
        Returns:
            List of (timestamp, value) tuples
        """
        time_series = self.get_time_series(series_name)
        if time_series is None:
            return []
        
        return time_series.get_values(metric_name, start_time, end_time)
    
    def get_latest_metric_value(self, series_name: str, metric_name: str) -> Optional[Tuple[datetime.datetime, float]]:
        """
        Get the latest value for a specific metric.
        
        Args:
            series_name: The name of the time-series
            metric_name: Name of the metric
            
        Returns:
            Tuple of (timestamp, value) or None if no values exist
        """
        time_series = self.get_time_series(series_name)
        if time_series is None:
            return None
        
        return time_series.get_latest_value(metric_name)
    
    def get_trend_analysis(self, series_name: str, metric_name: str, 
                          window_size: int = 10) -> Dict:
        """
        Perform trend analysis on a metric.
        
        Args:
            series_name: The name of the time-series
            metric_name: Name of the metric
            window_size: Size of the moving average window
            
        Returns:
            Dictionary with trend analysis results
        """
        time_series = self.get_time_series(series_name)
        if time_series is None:
            return {
                "trend": "UNKNOWN",
                "slope": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        
        # Get values for the metric
        values = time_series.get_values(metric_name)
        if not values:
            return {
                "trend": "UNKNOWN",
                "slope": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        
        # Extract timestamps and values
        timestamps = [t.timestamp() for t, v in values]
        metric_values = [v for t, v in values]
        
        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        metric_values = np.array(metric_values)
        
        # Calculate statistics
        mean = np.mean(metric_values)
        std = np.std(metric_values)
        min_val = np.min(metric_values)
        max_val = np.max(metric_values)
        
        # Calculate trend using linear regression
        if len(timestamps) >= 2:
            slope, intercept = np.polyfit(timestamps, metric_values, 1)
            
            # Determine trend direction
            if slope > 0.01:
                trend = "INCREASING"
            elif slope < -0.01:
                trend = "DECREASING"
            else:
                trend = "STABLE"
        else:
            slope = 0.0
            trend = "UNKNOWN"
        
        # Calculate moving average if enough data points
        moving_avg = None
        if len(metric_values) >= window_size:
            moving_avg = []
            for i in range(len(metric_values) - window_size + 1):
                window = metric_values[i:i+window_size]
                moving_avg.append(np.mean(window))
        
        return {
            "trend": trend,
            "slope": slope,
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "moving_avg": moving_avg
        }
    
    def export_to_csv(self, series_name: str, file_path: str) -> bool:
        """
        Export a time-series to CSV file.
        
        Args:
            series_name: The name of the time-series
            file_path: Path to the output CSV file
            
        Returns:
            True if export was successful, False otherwise
        """
        time_series = self.get_time_series(series_name)
        if time_series is None:
            return False
        
        try:
            # Convert to DataFrame
            df = time_series.to_dataframe()
            
            # Export to CSV
            df.to_csv(file_path)
            
            logger.info(f"Exported time-series {series_name} to {file_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to export time-series {series_name} to CSV: {e}")
            return False
    
    def list_time_series(self) -> List[str]:
        """
        Get a list of all time-series names.
        
        Returns:
            List of time-series names
        """
        return list(self.time_series.keys())
    
    def delete_time_series(self, name: str) -> bool:
        """
        Delete a time-series.
        
        Args:
            name: The name of the time-series
            
        Returns:
            True if the time-series was deleted, False if it doesn't exist
        """
        if name not in self.time_series:
            logger.warning(f"Time-series {name} does not exist")
            return False
        
        # Remove from memory
        del self.time_series[name]
        
        # Remove from storage
        try:
            file_path = os.path.join(self.storage_path, f"{name}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            logger.info(f"Deleted time-series {name}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete time-series file for {name}: {e}")
            return False
    
    def _load_time_series(self) -> None:
        """Load all time-series from storage"""
        try:
            # Ensure storage path exists
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Find all time-series files
            files = [f for f in os.listdir(self.storage_path) if f.endswith(".json")]
            
            for file in files:
                try:
                    file_path = os.path.join(self.storage_path, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    time_series = TimeSeries.from_dict(data)
                    self.time_series[time_series.name] = time_series
                except Exception as e:
                    logger.error(f"Failed to load time-series from {file}: {e}")
            
            logger.info(f"Loaded {len(self.time_series)} time-series")
        except Exception as e:
            logger.error(f"Failed to load time-series: {e}")
    
    def _save_time_series(self, time_series: TimeSeries) -> None:
        """
        Save a time-series to storage.
        
        Args:
            time_series: The time-series to save
        """
        try:
            # Ensure storage path exists
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Convert to dict
            data = time_series.to_dict()
            
            # Save to file
            file_path = os.path.join(self.storage_path, f"{time_series.name}.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved time-series {time_series.name}")
        except Exception as e:
            logger.error(f"Failed to save time-series {time_series.name}: {e}")


# Example usage
if __name__ == "__main__":
    # Create time-series database
    db = TimeSeriesDatabase("time_series_data")
    
    # Create time-series
    energy_metrics = db.create_time_series(
        name="energy_metrics",
        description="Energy metrics from the auditor",
        unit="energy units"
    )
    
    # Add data points
    now = datetime.datetime.now()
    db.add_point(
        series_name="energy_metrics",
        timestamp=now - datetime.timedelta(hours=2),
        values={"E": 0.5, "delta_E": 0.05, "lambda": 0.8}
    )
    db.add_point(
        series_name="energy_metrics",
        timestamp=now - datetime.timedelta(hours=1),
        values={"E": 0.3, "delta_E": 0.03, "lambda": 0.7}
    )
    db.add_point(
        series_name="energy_metrics",
        timestamp=now,
        values={"E": 0.1, "delta_E": 0.01, "lambda": 0.6}
    )
    
    # Get latest value
    latest_e = db.get_latest_metric_value("energy_metrics", "E")
    print(f"Latest E value: {latest_e[1]} at {latest_e[0]}")
    
    # Get trend analysis
    trend = db.get_trend_analysis("energy_metrics", "E")
    print(f"E trend: {trend['trend']} (slope: {trend['slope']})")
    
    # Export to CSV
    db.export_to_csv("energy_metrics", "energy_metrics.csv")
