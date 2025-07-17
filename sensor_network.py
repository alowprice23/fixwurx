#!/usr/bin/env python3
"""
sensor_network.py
────────────────
Comprehensive sensor network for the FixWurx platform.

This module provides a framework for collecting, processing, and analyzing
data from various sensors across the system, enabling real-time monitoring,
alerts, and insights into system behavior.
"""

import os
import sys
import time
import threading
import logging
import json
import uuid
import re
import statistics
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set, Type
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import queue
from abc import ABC, abstractmethod
import sqlite3
import pickle
import hashlib
import io
from collections import deque, defaultdict

# Internal imports
from shell_environment import register_event_handler, emit_event, EventType
from process_monitoring import get_system_resource_usage
from progress_indicators import IndicatorStyle, create_progress, update_progress, complete_progress

# Configure logging
logger = logging.getLogger("SensorNetwork")

# Constants
DEFAULT_SAMPLING_INTERVAL = 5.0  # seconds
DEFAULT_STORAGE_RETENTION = 86400  # 24 hours in seconds
DEFAULT_AGGREGATION_WINDOW = 60  # 1 minute in seconds
DEFAULT_DB_PATH = "~/.fixwurx/sensor_data.db"
DEFAULT_CACHE_SIZE = 1000  # Number of data points to cache in memory
DEFAULT_MAX_BATCH_SIZE = 100  # Maximum number of data points to write in a batch
DEFAULT_STORAGE_FLUSH_INTERVAL = 30.0  # seconds
DEFAULT_ALERT_CHECK_INTERVAL = 10.0  # seconds
DEFAULT_EVENT_DEBOUNCE_INTERVAL = 60.0  # seconds
DEFAULT_EVENT_COOLDOWN = 300.0  # seconds

class SensorType(Enum):
    """Types of sensors in the system."""
    SYSTEM = auto()  # System-level metrics (CPU, memory, disk, network)
    PROCESS = auto()  # Process-level metrics
    APPLICATION = auto()  # Application-specific metrics
    NETWORK = auto()  # Network-specific metrics
    CUSTOM = auto()  # Custom/user-defined metrics
    COMPOSITE = auto()  # Derived from multiple other sensors
    EVENT = auto()  # Event-based metrics
    EXTERNAL = auto()  # External data sources
    PERFORMANCE = auto()  # Performance-specific metrics
    SECURITY = auto()  # Security-related metrics
    AUDIT = auto()  # Audit-related metrics
    HEALTH = auto()  # Health checks

class DataType(Enum):
    """Data types for sensor values."""
    NUMERIC = auto()  # Numeric value (int, float)
    BOOLEAN = auto()  # Boolean value (True, False)
    STRING = auto()  # String value
    TIMESTAMP = auto()  # Timestamp value
    ENUM = auto()  # Enumeration value
    DICT = auto()  # Dictionary/object value
    LIST = auto()  # List/array value
    BINARY = auto()  # Binary data
    COMPLEX = auto()  # Complex data structure

class AggregationType(Enum):
    """Aggregation types for sensor data."""
    SUM = auto()  # Sum of values
    AVG = auto()  # Average of values
    MIN = auto()  # Minimum value
    MAX = auto()  # Maximum value
    COUNT = auto()  # Count of values
    LAST = auto()  # Last value
    FIRST = auto()  # First value
    PERCENTILE = auto()  # Percentile value
    STDDEV = auto()  # Standard deviation
    RATE = auto()  # Rate of change
    DELTA = auto()  # Delta between values
    CUSTOM = auto()  # Custom aggregation function

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = auto()  # Informational alert
    WARNING = auto()  # Warning alert
    ERROR = auto()  # Error alert
    CRITICAL = auto()  # Critical alert

class AlertState(Enum):
    """Alert states."""
    INACTIVE = auto()  # Alert is inactive
    ACTIVE = auto()  # Alert is active
    ACKNOWLEDGED = auto()  # Alert is acknowledged
    RESOLVED = auto()  # Alert is resolved

@dataclass
class SensorMetadata:
    """Metadata for a sensor."""
    id: str
    name: str
    description: str
    type: SensorType
    data_type: DataType
    unit: Optional[str] = None
    aggregation_type: AggregationType = AggregationType.LAST
    aggregation_window: int = DEFAULT_AGGREGATION_WINDOW
    sampling_interval: float = DEFAULT_SAMPLING_INTERVAL
    retention_period: int = DEFAULT_STORAGE_RETENTION
    tags: Dict[str, str] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.name,
            "data_type": self.data_type.name,
            "unit": self.unit,
            "aggregation_type": self.aggregation_type.name,
            "aggregation_window": self.aggregation_window,
            "sampling_interval": self.sampling_interval,
            "retention_period": self.retention_period,
            "tags": self.tags,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorMetadata':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            type=SensorType[data["type"]],
            data_type=DataType[data["data_type"]],
            unit=data.get("unit"),
            aggregation_type=AggregationType[data.get("aggregation_type", "LAST")],
            aggregation_window=data.get("aggregation_window", DEFAULT_AGGREGATION_WINDOW),
            sampling_interval=data.get("sampling_interval", DEFAULT_SAMPLING_INTERVAL),
            retention_period=data.get("retention_period", DEFAULT_STORAGE_RETENTION),
            tags=data.get("tags", {}),
            properties=data.get("properties", {})
        )

@dataclass
class SensorData:
    """Data point from a sensor."""
    sensor_id: str
    timestamp: float
    value: Any
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp,
            "value": self.value,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorData':
        """Create from dictionary."""
        return cls(
            sensor_id=data["sensor_id"],
            timestamp=data["timestamp"],
            value=data["value"],
            tags=data.get("tags", {})
        )

@dataclass
class AlertRule:
    """Alert rule for a sensor."""
    id: str
    name: str
    description: str
    sensor_id: str
    condition: str  # Python expression string
    severity: AlertSeverity
    notification_channels: List[str]
    enabled: bool = True
    cooldown_period: float = DEFAULT_EVENT_COOLDOWN
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "sensor_id": self.sensor_id,
            "condition": self.condition,
            "severity": self.severity.name,
            "notification_channels": self.notification_channels,
            "enabled": self.enabled,
            "cooldown_period": self.cooldown_period,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRule':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            sensor_id=data["sensor_id"],
            condition=data["condition"],
            severity=AlertSeverity[data["severity"]],
            notification_channels=data["notification_channels"],
            enabled=data.get("enabled", True),
            cooldown_period=data.get("cooldown_period", DEFAULT_EVENT_COOLDOWN),
            properties=data.get("properties", {})
        )

@dataclass
class Alert:
    """Alert generated from a sensor."""
    id: str
    rule_id: str
    sensor_id: str
    timestamp: float
    severity: AlertSeverity
    state: AlertState
    message: str
    value: Any
    properties: Dict[str, Any] = field(default_factory=dict)
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp,
            "severity": self.severity.name,
            "state": self.state.name,
            "message": self.message,
            "value": self.value,
            "properties": self.properties,
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            rule_id=data["rule_id"],
            sensor_id=data["sensor_id"],
            timestamp=data["timestamp"],
            severity=AlertSeverity[data["severity"]],
            state=AlertState[data["state"]],
            message=data["message"],
            value=data["value"],
            properties=data.get("properties", {}),
            acknowledged_at=data.get("acknowledged_at"),
            resolved_at=data.get("resolved_at")
        )

class Sensor(ABC):
    """
    Base sensor class.
    
    This is the abstract base class for all sensors in the system.
    """
    
    def __init__(self, metadata: SensorMetadata):
        """
        Initialize the sensor.
        
        Args:
            metadata: Sensor metadata.
        """
        self.metadata = metadata
        self.last_sample_time = 0.0
        self.enabled = True
    
    @abstractmethod
    def sample(self) -> Any:
        """
        Sample the sensor.
        
        Returns:
            Sensor value.
        """
        pass
    
    def collect(self) -> Optional[SensorData]:
        """
        Collect data from the sensor.
        
        Returns:
            Sensor data or None if not ready.
        """
        if not self.enabled:
            return None
        
        now = time.time()
        if now - self.last_sample_time < self.metadata.sampling_interval:
            return None
        
        try:
            value = self.sample()
            self.last_sample_time = now
            
            return SensorData(
                sensor_id=self.metadata.id,
                timestamp=now,
                value=value
            )
        except Exception as e:
            logger.error(f"Error sampling sensor {self.metadata.id}: {e}")
            return None
    
    def enable(self) -> None:
        """Enable the sensor."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the sensor."""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if the sensor is enabled."""
        return self.enabled
    
    def set_sampling_interval(self, interval: float) -> None:
        """
        Set the sampling interval.
        
        Args:
            interval: Sampling interval in seconds.
        """
        self.metadata.sampling_interval = max(0.1, interval)
    
    def get_sampling_interval(self) -> float:
        """
        Get the sampling interval.
        
        Returns:
            Sampling interval in seconds.
        """
        return self.metadata.sampling_interval

class SystemCpuSensor(Sensor):
    """System CPU usage sensor."""
    
    def __init__(self):
        """Initialize the sensor."""
        metadata = SensorMetadata(
            id="system.cpu.usage",
            name="System CPU Usage",
            description="System-wide CPU usage percentage",
            type=SensorType.SYSTEM,
            data_type=DataType.NUMERIC,
            unit="percent",
            aggregation_type=AggregationType.AVG,
            sampling_interval=1.0
        )
        super().__init__(metadata)
    
    def sample(self) -> float:
        """
        Sample the sensor.
        
        Returns:
            CPU usage percentage.
        """
        return get_system_resource_usage()["cpu"]["percent"]

class SystemMemorySensor(Sensor):
    """System memory usage sensor."""
    
    def __init__(self):
        """Initialize the sensor."""
        metadata = SensorMetadata(
            id="system.memory.usage",
            name="System Memory Usage",
            description="System-wide memory usage percentage",
            type=SensorType.SYSTEM,
            data_type=DataType.NUMERIC,
            unit="percent",
            aggregation_type=AggregationType.AVG,
            sampling_interval=1.0
        )
        super().__init__(metadata)
    
    def sample(self) -> float:
        """
        Sample the sensor.
        
        Returns:
            Memory usage percentage.
        """
        return get_system_resource_usage()["memory"]["percent"]

class SystemDiskSensor(Sensor):
    """System disk usage sensor."""
    
    def __init__(self):
        """Initialize the sensor."""
        metadata = SensorMetadata(
            id="system.disk.usage",
            name="System Disk Usage",
            description="System-wide disk usage percentage",
            type=SensorType.SYSTEM,
            data_type=DataType.NUMERIC,
            unit="percent",
            aggregation_type=AggregationType.AVG,
            sampling_interval=5.0
        )
        super().__init__(metadata)
    
    def sample(self) -> float:
        """
        Sample the sensor.
        
        Returns:
            Disk usage percentage.
        """
        return get_system_resource_usage()["disk"]["percent"]

class ProcessCountSensor(Sensor):
    """Process count sensor."""
    
    def __init__(self):
        """Initialize the sensor."""
        metadata = SensorMetadata(
            id="system.process.count",
            name="System Process Count",
            description="Number of processes in the system",
            type=SensorType.SYSTEM,
            data_type=DataType.NUMERIC,
            unit="count",
            aggregation_type=AggregationType.LAST,
            sampling_interval=5.0
        )
        super().__init__(metadata)
    
    def sample(self) -> int:
        """
        Sample the sensor.
        
        Returns:
            Process count.
        """
        return get_system_resource_usage()["processes"]["count"]

class NetworkThroughputSensor(Sensor):
    """Network throughput sensor."""
    
    def __init__(self):
        """Initialize the sensor."""
        metadata = SensorMetadata(
            id="system.network.throughput",
            name="Network Throughput",
            description="Network throughput (bytes/sec)",
            type=SensorType.NETWORK,
            data_type=DataType.DICT,
            unit="bytes/sec",
            aggregation_type=AggregationType.LAST,
            sampling_interval=1.0
        )
        super().__init__(metadata)
        self.last_bytes_sent = 0
        self.last_bytes_recv = 0
        self.last_sample_timestamp = None
    
    def sample(self) -> Dict[str, float]:
        """
        Sample the sensor.
        
        Returns:
            Network throughput in bytes/sec.
        """
        network_data = get_system_resource_usage()["network"]
        now = time.time()
        
        result = {
            "sent": 0.0,
            "received": 0.0,
            "total": 0.0
        }
        
        if self.last_sample_timestamp is not None:
            time_diff = now - self.last_sample_timestamp
            if time_diff > 0:
                bytes_sent_diff = network_data["bytes_sent"] - self.last_bytes_sent
                bytes_recv_diff = network_data["bytes_recv"] - self.last_bytes_recv
                
                sent_rate = bytes_sent_diff / time_diff
                recv_rate = bytes_recv_diff / time_diff
                
                result["sent"] = sent_rate
                result["received"] = recv_rate
                result["total"] = sent_rate + recv_rate
        
        self.last_bytes_sent = network_data["bytes_sent"]
        self.last_bytes_recv = network_data["bytes_recv"]
        self.last_sample_timestamp = now
        
        return result

class CustomSensor(Sensor):
    """Custom sensor with user-provided sampling function."""
    
    def __init__(self, metadata: SensorMetadata, sample_func: Callable[[], Any]):
        """
        Initialize the sensor.
        
        Args:
            metadata: Sensor metadata.
            sample_func: Function to call for sampling the sensor.
        """
        super().__init__(metadata)
        self._sample_func = sample_func
    
    def sample(self) -> Any:
        """
        Sample the sensor.
        
        Returns:
            Sensor value.
        """
        return self._sample_func()

class CompositeSensor(Sensor):
    """Composite sensor derived from multiple other sensors."""
    
    def __init__(self, metadata: SensorMetadata, sensors: List[str], 
                derive_func: Callable[[Dict[str, Any]], Any]):
        """
        Initialize the sensor.
        
        Args:
            metadata: Sensor metadata.
            sensors: List of sensor IDs to derive from.
            derive_func: Function to derive value from source sensors.
        """
        super().__init__(metadata)
        self._source_sensors = sensors
        self._derive_func = derive_func
        self._sensor_network = None  # Will be set by the sensor network
    
    def set_sensor_network(self, sensor_network: 'SensorNetwork') -> None:
        """
        Set the sensor network.
        
        Args:
            sensor_network: Sensor network.
        """
        self._sensor_network = sensor_network
    
    def sample(self) -> Any:
        """
        Sample the sensor.
        
        Returns:
            Derived sensor value.
        """
        if self._sensor_network is None:
            raise RuntimeError("Sensor network not set for composite sensor")
        
        # Get latest values from source sensors
        values = {}
        for sensor_id in self._source_sensors:
            latest = self._sensor_network.get_latest_value(sensor_id)
            if latest is not None:
                values[sensor_id] = latest
        
        # Derive value from source sensors
        return self._derive_func(values)

class ExternalSensor(Sensor):
    """External sensor with manually set value."""
    
    def __init__(self, metadata: SensorMetadata):
        """
        Initialize the sensor.
        
        Args:
            metadata: Sensor metadata.
        """
        super().__init__(metadata)
        self._value = None
        self._last_update = 0.0
    
    def set_value(self, value: Any) -> None:
        """
        Set the sensor value.
        
        Args:
            value: Sensor value.
        """
        self._value = value
        self._last_update = time.time()
    
    def sample(self) -> Any:
        """
        Sample the sensor.
        
        Returns:
            Sensor value.
        """
        if self._value is None:
            raise ValueError("Sensor value not set")
        
        return self._value

class SensorDataStorage:
    """
    Storage for sensor data.
    
    This class provides persistent storage for sensor data using SQLite.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the storage.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        
        # Expand ~ to user's home directory
        self._db_path = os.path.expanduser(db_path)
        
        # Create directory if it doesn't exist
        db_dir = os.path.dirname(self._db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        # Initialize database
        self._init_db()
        
        # Create write queue
        self._write_queue = queue.Queue()
        
        # Create cache
        self._cache = {}
        
        # Create lock
        self._lock = threading.RLock()
        
        # Start write thread
        self._stop_write = threading.Event()
        self._write_thread = threading.Thread(
            target=self._write_loop,
            daemon=True,
            name="SensorDataWriteThread"
        )
        self._write_thread.start()
        
        logger.info("Sensor data storage initialized")
    
    def _init_db(self) -> None:
        """Initialize the database."""
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_metadata (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sensor_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    value_type TEXT NOT NULL,
                    value BLOB NOT NULL,
                    tags TEXT,
                    FOREIGN KEY (sensor_id) REFERENCES sensor_metadata (id)
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_sensor_data_sensor_id
                ON sensor_data (sensor_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp
                ON sensor_data (timestamp)
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    sensor_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    data TEXT NOT NULL,
                    FOREIGN KEY (rule_id) REFERENCES alert_rules (id),
                    FOREIGN KEY (sensor_id) REFERENCES sensor_metadata (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.debug("Database schema initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _write_loop(self) -> None:
        """Write loop for sensor data."""
        while not self._stop_write.is_set():
            try:
                batch = []
                batch_size = 0
                
                # Get items from queue with timeout
                try:
                    while batch_size < DEFAULT_MAX_BATCH_SIZE:
                        # Get item with timeout
                        item = self._write_queue.get(timeout=DEFAULT_STORAGE_FLUSH_INTERVAL)
                        batch.append(item)
                        batch_size += 1
                        self._write_queue.task_done()
                except queue.Empty:
                    pass
                
                # Write batch to database
                if batch:
                    self._write_batch(batch)
            except Exception as e:
                logger.error(f"Error in sensor data write loop: {e}")
            
            # Sleep for a short time
            time.sleep(0.1)
    
    def _write_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        Write a batch of sensor data to the database.
        
        Args:
            batch: Batch of sensor data.
        """
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            for item in batch:
                if item["type"] == "data":
                    data = item["data"]
                    
                    # Serialize value
                    value_type = type(data.value).__name__
                    value_blob = pickle.dumps(data.value)
                    
                    # Serialize tags
                    tags_json = json.dumps(data.tags) if data.tags else None
                    
                    # Insert data
                    cursor.execute(
                        "INSERT INTO sensor_data (sensor_id, timestamp, value_type, value, tags) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (data.sensor_id, data.timestamp, value_type, value_blob, tags_json)
                    )
                elif item["type"] == "metadata":
                    metadata = item["metadata"]
                    
                    # Serialize metadata
                    metadata_json = json.dumps(metadata.to_dict())
                    
                    # Insert or update metadata
                    cursor.execute(
                        "INSERT OR REPLACE INTO sensor_metadata (id, data) VALUES (?, ?)",
                        (metadata.id, metadata_json)
                    )
                elif item["type"] == "alert_rule":
                    rule = item["rule"]
                    
                    # Serialize rule
                    rule_json = json.dumps(rule.to_dict())
                    
                    # Insert or update rule
                    cursor.execute(
                        "INSERT OR REPLACE INTO alert_rules (id, data) VALUES (?, ?)",
                        (rule.id, rule_json)
                    )
                elif item["type"] == "alert":
                    alert = item["alert"]
                    
                    # Serialize alert
                    alert_json = json.dumps(alert.to_dict())
                    
                    # Insert or update alert
                    cursor.execute(
                        "INSERT OR REPLACE INTO alerts (id, rule_id, sensor_id, timestamp, data) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (alert.id, alert.rule_id, alert.sensor_id, alert.timestamp, alert_json)
                    )
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Wrote {len(batch)} items to database")
        except Exception as e:
            logger.error(f"Error writing batch to database: {e}")
    
    def stop(self) -> None:
        """Stop the storage."""
        self._stop_write.set()
        self._write_thread.join(timeout=5.0)
        
        # Flush remaining items
        try:
            batch = []
            while not self._write_queue.empty():
                batch.append(self._write_queue.get_nowait())
                self._write_queue.task_done()
            
            if batch:
                self._write_batch(batch)
        except Exception as e:
            logger.error(f"Error flushing remaining items: {e}")
    
    def store_data(self, data: SensorData) -> None:
        """
        Store sensor data.
        
        Args:
            data: Sensor data.
        """
        # Add to cache
        with self._lock:
            if data.sensor_id not in self._cache:
                self._cache[data.sensor_id] = deque(maxlen=DEFAULT_CACHE_SIZE)
            
            self._cache[data.sensor_id].append(data)
        
        # Add to write queue
        self._write_queue.put({
            "type": "data",
            "data": data
        })
    
    def store_metadata(self, metadata: SensorMetadata) -> None:
        """
        Store sensor metadata.
        
        Args:
            metadata: Sensor metadata.
        """
        # Add to write queue
        self._write_queue.put({
            "type": "metadata",
            "metadata": metadata
        })
    
    def store_alert_rule(self, rule: AlertRule) -> None:
        """
        Store alert rule.
        
        Args:
            rule: Alert rule.
        """
        # Add to write queue
        self._write_queue.put({
            "type": "alert_rule",
            "rule": rule
        })
    
    def store_alert(self, alert: Alert) -> None:
        """
        Store alert.
        
        Args:
            alert: Alert.
        """
        # Add to write queue
        self._write_queue.put({
            "type": "alert",
            "alert": alert
        })
    
    def get_latest_value(self, sensor_id: str) -> Optional[Any]:
        """
        Get the latest value for a sensor.
        
        Args:
            sensor_id: Sensor ID.
            
        Returns:
            Latest value or None if not found.
        """
        with self._lock:
            if sensor_id in self._cache and self._cache[sensor_id]:
                return self._cache[sensor_id][-1].value
        
        # Try to get from database
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT value FROM sensor_data "
                "WHERE sensor_id = ? "
                "ORDER BY timestamp DESC "
                "LIMIT 1",
                (sensor_id,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return pickle.loads(row[0])
        except Exception as e:
            logger.error(f"Error getting latest value from database: {e}")
        
        return None
    
    def get_range_values(self, sensor_id: str, start_time: float,
                        end_time: float) -> List[SensorData]:
        """
        Get values for a sensor in a time range.
        
        Args:
            sensor_id: Sensor ID.
            start_time: Start time.
            end_time: End time.
            
        Returns:
            List of sensor data.
        """
        result = []
        
        # Check cache first
        with self._lock:
            if sensor_id in self._cache:
                for data in self._cache[sensor_id]:
                    if start_time <= data.timestamp <= end_time:
                        result.append(data)
        
        # Get from database
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT sensor_id, timestamp, value_type, value, tags "
                "FROM sensor_data "
                "WHERE sensor_id = ? AND timestamp >= ? AND timestamp <= ? "
                "ORDER BY timestamp",
                (sensor_id, start_time, end_time)
            )
            
            for row in cursor.fetchall():
                sensor_id, timestamp, value_type, value_blob, tags_json = row
                
                # Deserialize value
                value = pickle.loads(value_blob)
                
                # Deserialize tags
                tags = json.loads(tags_json) if tags_json else {}
                
                # Create sensor data
                data = SensorData(
                    sensor_id=sensor_id,
                    timestamp=timestamp,
                    value=value,
                    tags=tags
                )
                
                result.append(data)
            
            conn.close()
        except Exception as e:
            logger.error(f"Error getting range values from database: {e}")
        
        return result
    
    def get_aggregated_values(self, sensor_id: str, start_time: float,
                             end_time: float, window: int,
                             agg_type: AggregationType) -> List[Tuple[float, Any]]:
        """
        Get aggregated values for a sensor in a time range.
        
        Args:
            sensor_id: Sensor ID.
            start_time: Start time.
            end_time: End time.
            window: Aggregation window in seconds.
            agg_type: Aggregation type.
            
        Returns:
            List of (timestamp, value) tuples.
        """
        # Get values in range
        values = self.get_range_values(sensor_id, start_time, end_time)
        
        # Group values by time window
        windows = {}
        for data in values:
            window_start = int(data.timestamp / window) * window
            if window_start not in windows:
                windows[window_start] = []
            windows[window_start].append(data.value)
        
        # Aggregate values
        result = []
        for window_start, window_values in sorted(windows.items()):
            if not window_values:
                continue
            
            if agg_type == AggregationType.SUM:
                try:
                    agg_value = sum(window_values)
                except (TypeError, ValueError):
                    agg_value = None
            elif agg_type == AggregationType.AVG:
                try:
                    agg_value = sum(window_values) / len(window_values)
                except (TypeError, ValueError):
                    agg_value = None
            elif agg_type == AggregationType.MIN:
                try:
                    agg_value = min(window_values)
                except (TypeError, ValueError):
                    agg_value = None
            elif agg_type == AggregationType.MAX:
                try:
                    agg_value = max(window_values)
                except (TypeError, ValueError):
                    agg_value = None
            elif agg_type == AggregationType.COUNT:
                agg_value = len(window_values)
            elif agg_type == AggregationType.LAST:
                agg_value = window_values[-1]
            elif agg_type == AggregationType.FIRST:
                agg_value = window_values[0]
            elif agg_type == AggregationType.PERCENTILE:
                try:
                    agg_value = sorted(window_values)[int(len(window_values) * 0.95)]
                except (TypeError, ValueError, IndexError):
                    agg_value = None
            elif agg_type == AggregationType.STDDEV:
                try:
                    agg_value = statistics.stdev(window_values)
                except (TypeError, ValueError, statistics.StatisticsError):
                    agg_value = None
            else:
                agg_value = None
            
            result.append((window_start, agg_value))
        
        return result
    
    def cleanup_old_data(self, retention_period: Optional[int] = None) -> int:
        """
        Clean up old sensor data.
        
        Args:
            retention_period: Retention period in seconds.
            
        Returns:
            Number of rows deleted.
        """
        if retention_period is None:
            retention_period = DEFAULT_STORAGE_RETENTION
        
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = time.time() - retention_period
            
            # Delete old data
            cursor.execute(
                "DELETE FROM sensor_data WHERE timestamp < ?",
                (cutoff_time,)
            )
            
            rows_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Deleted {rows_deleted} old sensor data rows")
            
            return rows_deleted
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    def get_metadata(self, sensor_id: str) -> Optional[SensorMetadata]:
        """
        Get metadata for a sensor.
        
        Args:
            sensor_id: Sensor ID.
            
        Returns:
            Sensor metadata or None if not found.
        """
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT data FROM sensor_metadata WHERE id = ?",
                (sensor_id,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                metadata_dict = json.loads(row[0])
                return SensorMetadata.from_dict(metadata_dict)
        except Exception as e:
            logger.error(f"Error getting metadata from database: {e}")
        
        return None
    
    def get_all_metadata(self) -> Dict[str, SensorMetadata]:
        """
        Get metadata for all sensors.
        
        Returns:
            Dictionary of sensor IDs to metadata.
        """
        result = {}
        
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, data FROM sensor_metadata")
            
            for row in cursor.fetchall():
                sensor_id, data = row
                metadata_dict = json.loads(data)
                result[sensor_id] = SensorMetadata.from_dict(metadata_dict)
            
            conn.close()
        except Exception as e:
            logger.error(f"Error getting all metadata from database: {e}")
        
        return result
    
    def get_alert_rules(self, sensor_id: Optional[str] = None) -> List[AlertRule]:
        """
        Get alert rules for a sensor.
        
        Args:
            sensor_id: Sensor ID or None for all rules.
            
        Returns:
            List of alert rules.
        """
        result = []
        
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            if sensor_id:
                cursor.execute(
                    "SELECT data FROM alert_rules WHERE data LIKE ?",
                    (f'%"sensor_id": "{sensor_id}"%',)
                )
            else:
                cursor.execute("SELECT data FROM alert_rules")
            
            for row in cursor.fetchall():
                rule_dict = json.loads(row[0])
                result.append(AlertRule.from_dict(rule_dict))
            
            conn.close()
        except Exception as e:
            logger.error(f"Error getting alert rules from database: {e}")
        
        return result
    
    def get_alerts(self, rule_id: Optional[str] = None,
                 sensor_id: Optional[str] = None,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None,
                 limit: int = 100) -> List[Alert]:
        """
        Get alerts.
        
        Args:
            rule_id: Rule ID or None for all rules.
            sensor_id: Sensor ID or None for all sensors.
            start_time: Start time or None for all time.
            end_time: End time or None for all time.
            limit: Maximum number of alerts to return.
            
        Returns:
            List of alerts.
        """
        result = []
        
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            query = "SELECT data FROM alerts"
            params = []
            
            where_clauses = []
            if rule_id:
                where_clauses.append("rule_id = ?")
                params.append(rule_id)
            if sensor_id:
                where_clauses.append("sensor_id = ?")
                params.append(sensor_id)
            if start_time:
                where_clauses.append("timestamp >= ?")
                params.append(start_time)
            if end_time:
                where_clauses.append("timestamp <= ?")
                params.append(end_time)
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                alert_dict = json.loads(row[0])
                result.append(Alert.from_dict(alert_dict))
            
            conn.close()
        except Exception as e:
            logger.error(f"Error getting alerts from database: {e}")
        
        return result


class AlertManager:
    """
    Alert manager for the sensor network.
    
    This class manages alert rules, checks for rule violations,
    and generates alerts when thresholds are exceeded.
    """
    
    def __init__(self, storage: SensorDataStorage):
        """
        Initialize the alert manager.
        
        Args:
            storage: Sensor data storage.
        """
        self._storage = storage
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._rule_last_triggered: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stop_check = threading.Event()
        self._check_thread = None
        
        # Load alert rules from storage
        self._load_rules()
        
        logger.info("Alert manager initialized")
    
    def _load_rules(self) -> None:
        """Load alert rules from storage."""
        rules = self._storage.get_alert_rules()
        
        with self._lock:
            self._rules = {rule.id: rule for rule in rules}
    
    def start(self) -> None:
        """Start the alert manager."""
        if self._check_thread is None or not self._check_thread.is_alive():
            self._stop_check.clear()
            self._check_thread = threading.Thread(
                target=self._check_loop,
                daemon=True,
                name="AlertCheckThread"
            )
            self._check_thread.start()
    
    def stop(self) -> None:
        """Stop the alert manager."""
        if self._check_thread and self._check_thread.is_alive():
            self._stop_check.set()
            self._check_thread.join(timeout=5.0)
    
    def _check_loop(self) -> None:
        """Check for alert rule violations periodically."""
        while not self._stop_check.is_set():
            try:
                self._check_rules()
            except Exception as e:
                logger.error(f"Error checking alert rules: {e}")
            
            # Sleep before next check
            time.sleep(DEFAULT_ALERT_CHECK_INTERVAL)
    
    def _check_rules(self) -> None:
        """Check all alert rules."""
        with self._lock:
            for rule_id, rule in self._rules.items():
                if not rule.enabled:
                    continue
                
                # Check if rule is in cooldown
                now = time.time()
                if rule_id in self._rule_last_triggered:
                    last_triggered = self._rule_last_triggered[rule_id]
                    if now - last_triggered < rule.cooldown_period:
                        continue
                
                try:
                    self._check_rule(rule)
                except Exception as e:
                    logger.error(f"Error checking rule {rule.id}: {e}")
    
    def _check_rule(self, rule: AlertRule) -> None:
        """
        Check a single alert rule.
        
        Args:
            rule: Alert rule to check.
        """
        # Get latest value for sensor
        latest_value = self._storage.get_latest_value(rule.sensor_id)
        if latest_value is None:
            return
        
        # Create evaluation context
        context = {
            "value": latest_value,
            "sensor_id": rule.sensor_id,
            "rule_id": rule.id,
            "now": time.time()
        }
        
        # Evaluate condition
        try:
            condition_met = eval(rule.condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.error(f"Error evaluating rule condition: {e}")
            return
        
        # If condition is met, trigger alert
        if condition_met:
            self._trigger_alert(rule, latest_value)
        else:
            # If condition is not met, resolve any active alerts for this rule
            self._resolve_alert(rule)
    
    def _trigger_alert(self, rule: AlertRule, value: Any) -> None:
        """
        Trigger an alert for a rule.
        
        Args:
            rule: Alert rule.
            value: Sensor value that triggered the alert.
        """
        now = time.time()
        
        # Update last triggered time
        self._rule_last_triggered[rule.id] = now
        
        # Check if alert already exists
        if rule.id in self._active_alerts:
            # Alert already active, nothing to do
            return
        
        # Create alert
        alert_id = f"alert_{int(now * 1000)}_{rule.id}"
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            sensor_id=rule.sensor_id,
            timestamp=now,
            severity=rule.severity,
            state=AlertState.ACTIVE,
            message=rule.description,
            value=value,
            properties=rule.properties
        )
        
        # Store alert
        self._active_alerts[rule.id] = alert
        self._storage.store_alert(alert)
        
        # Emit alert event
        event_data = {
            "alert_id": alert.id,
            "rule_id": rule.id,
            "sensor_id": rule.sensor_id,
            "severity": rule.severity.name,
            "message": rule.description,
            "value": value,
            "timestamp": now
        }
        
        self._emit_alert_event(event_data)
    
    def _resolve_alert(self, rule: AlertRule) -> None:
        """
        Resolve an active alert for a rule.
        
        Args:
            rule: Alert rule.
        """
        if rule.id not in self._active_alerts:
            return
        
        # Get alert
        alert = self._active_alerts[rule.id]
        
        # Update alert
        alert.state = AlertState.RESOLVED
        alert.resolved_at = time.time()
        
        # Store updated alert
        self._storage.store_alert(alert)
        
        # Remove from active alerts
        del self._active_alerts[rule.id]
        
        # Emit alert resolved event
        event_data = {
            "alert_id": alert.id,
            "rule_id": rule.id,
            "sensor_id": rule.sensor_id,
            "severity": alert.severity.name,
            "message": f"Alert resolved: {alert.message}",
            "timestamp": alert.resolved_at
        }
        
        self._emit_alert_resolved_event(event_data)
    
    def _emit_alert_event(self, event_data: Dict[str, Any]) -> None:
        """
        Emit an alert event.
        
        Args:
            event_data: Event data.
        """
        try:
            emit_event(EventType.SENSOR_ALERT, event_data)
        except Exception as e:
            logger.error(f"Error emitting alert event: {e}")
    
    def _emit_alert_resolved_event(self, event_data: Dict[str, Any]) -> None:
        """
        Emit an alert resolved event.
        
        Args:
            event_data: Event data.
        """
        try:
            emit_event(EventType.SENSOR_ALERT_RESOLVED, event_data)
        except Exception as e:
            logger.error(f"Error emitting alert resolved event: {e}")
    
    def add_rule(self, rule: AlertRule) -> None:
        """
        Add an alert rule.
        
        Args:
            rule: Alert rule.
        """
        with self._lock:
            self._rules[rule.id] = rule
            self._storage.store_alert_rule(rule)
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            rule_id: Rule ID.
            
        Returns:
            True if rule was removed, False otherwise.
        """
        with self._lock:
            if rule_id not in self._rules:
                return False
            
            del self._rules[rule_id]
            
            # Remove any active alerts for this rule
            if rule_id in self._active_alerts:
                del self._active_alerts[rule_id]
            
            return True
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """
        Get an alert rule.
        
        Args:
            rule_id: Rule ID.
            
        Returns:
            Alert rule or None if not found.
        """
        with self._lock:
            return self._rules.get(rule_id)
    
    def get_rules(self, sensor_id: Optional[str] = None) -> List[AlertRule]:
        """
        Get alert rules.
        
        Args:
            sensor_id: Sensor ID or None for all rules.
            
        Returns:
            List of alert rules.
        """
        with self._lock:
            if sensor_id:
                return [rule for rule in self._rules.values() if rule.sensor_id == sensor_id]
            else:
                return list(self._rules.values())
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID.
            
        Returns:
            True if alert was acknowledged, False otherwise.
        """
        # Get alert from storage
        alerts = self._storage.get_alerts(limit=1)
        if not alerts:
            return False
        
        alert = alerts[0]
        
        # Update alert state
        alert.state = AlertState.ACKNOWLEDGED
        alert.acknowledged_at = time.time()
        
        # Store updated alert
        self._storage.store_alert(alert)
        
        return True
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get active alerts.
        
        Returns:
            List of active alerts.
        """
        with self._lock:
            return list(self._active_alerts.values())
    
    def get_alerts(self, rule_id: Optional[str] = None,
                  sensor_id: Optional[str] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: int = 100) -> List[Alert]:
        """
        Get alerts.
        
        Args:
            rule_id: Rule ID or None for all rules.
            sensor_id: Sensor ID or None for all sensors.
            start_time: Start time or None for all time.
            end_time: End time or None for all time.
            limit: Maximum number of alerts to return.
            
        Returns:
            List of alerts.
        """
        return self._storage.get_alerts(rule_id, sensor_id, start_time, end_time, limit)


class SensorNetwork:
    """
    Sensor network for the FixWurx platform.
    
    This class manages sensors, collects data, and provides access to
    sensor data and alerts.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the sensor network.
        
        Args:
            db_path: Path to the sensor data database.
        """
        # Initialize storage
        self._storage = SensorDataStorage(db_path)
        
        # Initialize alert manager
        self._alert_manager = AlertManager(self._storage)
        
        # Initialize sensors
        self._sensors: Dict[str, Sensor] = {}
        self._lock = threading.RLock()
        
        # Start collector thread
        self._stop_collector = threading.Event()
        self._collector_thread = None
        
        # Initialize dashboard
        self._dashboard = None
        
        logger.info("Sensor network initialized")
    
    def start(self) -> None:
        """Start the sensor network."""
        # Start alert manager
        self._alert_manager.start()
        
        # Start collector thread
        if self._collector_thread is None or not self._collector_thread.is_alive():
            self._stop_collector.clear()
            self._collector_thread = threading.Thread(
                target=self._collector_loop,
                daemon=True,
                name="SensorCollectorThread"
            )
            self._collector_thread.start()
        
        logger.info("Sensor network started")
    
    def stop(self) -> None:
        """Stop the sensor network."""
        # Stop collector thread
        if self._collector_thread and self._collector_thread.is_alive():
            self._stop_collector.set()
            self._collector_thread.join(timeout=5.0)
        
        # Stop alert manager
        self._alert_manager.stop()
        
        # Stop storage
        self._storage.stop()
        
        logger.info("Sensor network stopped")
    
    def _collector_loop(self) -> None:
        """Collect data from sensors periodically."""
        while not self._stop_collector.is_set():
            try:
                self._collect_sensor_data()
            except Exception as e:
                logger.error(f"Error collecting sensor data: {e}")
            
            # Sleep for a short time
            time.sleep(0.1)
    
    def _collect_sensor_data(self) -> None:
        """Collect data from all sensors."""
        with self._lock:
            for sensor_id, sensor in self._sensors.items():
                try:
                    # Collect data from sensor
                    data = sensor.collect()
                    if data:
                        # Store data
                        self._storage.store_data(data)
                except Exception as e:
                    logger.error(f"Error collecting data from sensor {sensor_id}: {e}")
    
    def add_sensor(self, sensor: Sensor) -> None:
        """
        Add a sensor to the network.
        
        Args:
            sensor: Sensor to add.
        """
        with self._lock:
            sensor_id = sensor.metadata.id
            self._sensors[sensor_id] = sensor
            
            # Store sensor metadata
            self._storage.store_metadata(sensor.metadata)
            
            # Set sensor network for composite sensors
            if isinstance(sensor, CompositeSensor):
                sensor.set_sensor_network(self)
        
        logger.info(f"Added sensor: {sensor.metadata.name} ({sensor.metadata.id})")
    
    def remove_sensor(self, sensor_id: str) -> bool:
        """
        Remove a sensor from the network.
        
        Args:
            sensor_id: Sensor ID.
            
        Returns:
            True if sensor was removed, False otherwise.
        """
        with self._lock:
            if sensor_id not in self._sensors:
                return False
            
            del self._sensors[sensor_id]
            return True
    
    def get_sensor(self, sensor_id: str) -> Optional[Sensor]:
        """
        Get a sensor by ID.
        
        Args:
            sensor_id: Sensor ID.
            
        Returns:
            Sensor or None if not found.
        """
        with self._lock:
            return self._sensors.get(sensor_id)
    
    def get_sensors(self, sensor_type: Optional[SensorType] = None) -> List[Sensor]:
        """
        Get sensors.
        
        Args:
            sensor_type: Sensor type or None for all sensors.
            
        Returns:
            List of sensors.
        """
        with self._lock:
            if sensor_type:
                return [s for s in self._sensors.values() if s.metadata.type == sensor_type]
            else:
                return list(self._sensors.values())
    
    def get_sensor_metadata(self, sensor_id: str) -> Optional[SensorMetadata]:
        """
        Get metadata for a sensor.
        
        Args:
            sensor_id: Sensor ID.
            
        Returns:
            Sensor metadata or None if not found.
        """
        with self._lock:
            if sensor_id in self._sensors:
                return self._sensors[sensor_id].metadata
        
        # Try to get from storage
        return self._storage.get_metadata(sensor_id)
    
    def get_all_sensor_metadata(self) -> Dict[str, SensorMetadata]:
        """
        Get metadata for all sensors.
        
        Returns:
            Dictionary of sensor IDs to metadata.
        """
        return self._storage.get_all_metadata()
    
    def get_latest_value(self, sensor_id: str) -> Optional[Any]:
        """
        Get the latest value for a sensor.
        
        Args:
            sensor_id: Sensor ID.
            
        Returns:
            Latest value or None if not found.
        """
        return self._storage.get_latest_value(sensor_id)
    
    def get_range_values(self, sensor_id: str, start_time: float,
                        end_time: float) -> List[SensorData]:
        """
        Get values for a sensor in a time range.
        
        Args:
            sensor_id: Sensor ID.
            start_time: Start time.
            end_time: End time.
            
        Returns:
            List of sensor data.
        """
        return self._storage.get_range_values(sensor_id, start_time, end_time)
    
    def get_aggregated_values(self, sensor_id: str, start_time: float,
                             end_time: float, window: int,
                             agg_type: AggregationType) -> List[Tuple[float, Any]]:
        """
        Get aggregated values for a sensor in a time range.
        
        Args:
            sensor_id: Sensor ID.
            start_time: Start time.
            end_time: End time.
            window: Aggregation window in seconds.
            agg_type: Aggregation type.
            
        Returns:
            List of (timestamp, value) tuples.
        """
        return self._storage.get_aggregated_values(
            sensor_id, start_time, end_time, window, agg_type
        )
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """
        Add an alert rule.
        
        Args:
            rule: Alert rule.
        """
        self._alert_manager.add_rule(rule)
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            rule_id: Rule ID.
            
        Returns:
            True if rule was removed, False otherwise.
        """
        return self._alert_manager.remove_rule(rule_id)
    
    def get_alert_rule(self, rule_id: str) -> Optional[AlertRule]:
        """
        Get an alert rule.
        
        Args:
            rule_id: Rule ID.
            
        Returns:
            Alert rule or None if not found.
        """
        return self._alert_manager.get_rule(rule_id)
    
    def get_alert_rules(self, sensor_id: Optional[str] = None) -> List[AlertRule]:
        """
        Get alert rules.
        
        Args:
            sensor_id: Sensor ID or None for all rules.
            
        Returns:
            List of alert rules.
        """
        return self._alert_manager.get_rules(sensor_id)
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get active alerts.
        
        Returns:
            List of active alerts.
        """
        return self._alert_manager.get_active_alerts()
    
    def get_alerts(self, rule_id: Optional[str] = None,
                  sensor_id: Optional[str] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: int = 100) -> List[Alert]:
        """
        Get alerts.
        
        Args:
            rule_id: Rule ID or None for all rules.
            sensor_id: Sensor ID or None for all sensors.
            start_time: Start time or None for all time.
            end_time: End time or None for all time.
            limit: Maximum number of alerts to return.
            
        Returns:
            List of alerts.
        """
        return self._alert_manager.get_alerts(
            rule_id, sensor_id, start_time, end_time, limit
        )
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID.
            
        Returns:
            True if alert was acknowledged, False otherwise.
        """
        return self._alert_manager.acknowledge_alert(alert_id)
    
    def cleanup_old_data(self, retention_period: Optional[int] = None) -> int:
        """
        Clean up old sensor data.
        
        Args:
            retention_period: Retention period in seconds.
            
        Returns:
            Number of rows deleted.
        """
        return self._storage.cleanup_old_data(retention_period)


# Create system sensors
def create_system_sensors() -> List[Sensor]:
    """
    Create a set of system sensors.
    
    Returns:
        List of system sensors.
    """
    return [
        SystemCpuSensor(),
        SystemMemorySensor(),
        SystemDiskSensor(),
        ProcessCountSensor(),
        NetworkThroughputSensor()
    ]

# Create singleton sensor network
_sensor_network = None

def get_sensor_network(db_path: Optional[str] = None) -> SensorNetwork:
    """
    Get the singleton sensor network instance.
    
    Args:
        db_path: Path to the sensor data database.
        
    Returns:
        Sensor network instance.
    """
    global _sensor_network
    
    if _sensor_network is None:
        _sensor_network = SensorNetwork(db_path)
        
        # Add system sensors
        for sensor in create_system_sensors():
            _sensor_network.add_sensor(sensor)
    
    return _sensor_network

def add_sensor(sensor: Sensor) -> None:
    """
    Add a sensor to the network.
    
    Args:
        sensor: Sensor to add.
    """
    get_sensor_network().add_sensor(sensor)

def remove_sensor(sensor_id: str) -> bool:
    """
    Remove a sensor from the network.
    
    Args:
        sensor_id: Sensor ID.
        
    Returns:
        True if sensor was removed, False otherwise.
    """
    return get_sensor_network().remove_sensor(sensor_id)

def get_sensor(sensor_id: str) -> Optional[Sensor]:
    """
    Get a sensor by ID.
    
    Args:
        sensor_id: Sensor ID.
        
    Returns:
        Sensor or None if not found.
    """
    return get_sensor_network().get_sensor(sensor_id)

def get_sensors(sensor_type: Optional[SensorType] = None) -> List[Sensor]:
    """
    Get sensors.
    
    Args:
        sensor_type: Sensor type or None for all sensors.
        
    Returns:
        List of sensors.
    """
    return get_sensor_network().get_sensors(sensor_type)

def get_latest_value(sensor_id: str) -> Optional[Any]:
    """
    Get the latest value for a sensor.
    
    Args:
        sensor_id: Sensor ID.
        
    Returns:
        Latest value or None if not found.
    """
    return get_sensor_network().get_latest_value(sensor_id)

def add_alert_rule(rule: AlertRule) -> None:
    """
    Add an alert rule.
    
    Args:
        rule: Alert rule.
    """
    get_sensor_network().add_alert_rule(rule)

def get_active_alerts() -> List[Alert]:
    """
    Get active alerts.
    
    Returns:
        List of active alerts.
    """
    return get_sensor_network().get_active_alerts()

def acknowledge_alert(alert_id: str) -> bool:
    """
    Acknowledge an alert.
    
    Args:
        alert_id: Alert ID.
        
    Returns:
        True if alert was acknowledged, False otherwise.
    """
    return get_sensor_network().acknowledge_alert(alert_id)

def start_sensor_network() -> None:
    """Start the sensor network."""
    get_sensor_network().start()

def stop_sensor_network() -> None:
    """Stop the sensor network."""
    if _sensor_network is not None:
        _sensor_network.stop()

# Initialize sensor network if not in a test environment
if not any(arg.endswith('test.py') for arg in sys.argv):
    # Start sensor network in a separate thread
    threading.Thread(
        target=start_sensor_network,
        daemon=True,
        name="SensorNetworkStartThread"
    ).start()
