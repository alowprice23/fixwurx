#!/usr/bin/env python3
"""
Persistent Auditing Module

This module provides persistent auditing capabilities for the auditor agent,
enabling long-term logging, event tracking, and historical analysis of system activities.
"""

import os
import sys
import json
import logging
import time
import threading
import queue
import uuid
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("persistent_auditing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PersistentAuditing")

class AuditEvent:
    """
    Represents a single audit event.
    """
    
    def __init__(self, event_type: str, source: str, data: Dict[str, Any] = None, 
                timestamp: float = None, event_id: str = None):
        """
        Initialize audit event.
        
        Args:
            event_type: Type of event
            source: Source of event
            data: Additional event data
            timestamp: Event timestamp
            event_id: Unique event ID
        """
        self.event_type = event_type
        self.source = source
        self.data = data or {}
        self.timestamp = timestamp or time.time()
        self.event_id = event_id or str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Audit event
        """
        return cls(
            event_type=data.get("event_type"),
            source=data.get("source"),
            data=data.get("data", {}),
            timestamp=data.get("timestamp"),
            event_id=data.get("event_id")
        )

class AuditStorage:
    """
    Base class for audit storage implementations.
    """
    
    def store_event(self, event: AuditEvent) -> bool:
        """
        Store an audit event.
        
        Args:
            event: Audit event to store
            
        Returns:
            Whether the event was stored
        """
        raise NotImplementedError("Subclass must implement store_event")
    
    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """
        Get an audit event by ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            Audit event, or None if not found
        """
        raise NotImplementedError("Subclass must implement get_event")
    
    def query_events(self, query: Dict[str, Any] = None, 
                    start_time: float = None, end_time: float = None,
                    limit: int = 100, offset: int = 0) -> List[AuditEvent]:
        """
        Query audit events.
        
        Args:
            query: Query criteria
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of events to return
            offset: Offset for pagination
            
        Returns:
            List of audit events
        """
        raise NotImplementedError("Subclass must implement query_events")
    
    def get_event_count(self, query: Dict[str, Any] = None,
                       start_time: float = None, end_time: float = None) -> int:
        """
        Get count of audit events matching criteria.
        
        Args:
            query: Query criteria
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Event count
        """
        raise NotImplementedError("Subclass must implement get_event_count")
    
    def clear_events(self, older_than: float = None) -> int:
        """
        Clear audit events.
        
        Args:
            older_than: Clear events older than this timestamp
            
        Returns:
            Number of events cleared
        """
        raise NotImplementedError("Subclass must implement clear_events")

class SqliteAuditStorage(AuditStorage):
    """
    SQLite implementation of audit storage.
    """
    
    def __init__(self, db_file: str = None):
        """
        Initialize SQLite audit storage.
        
        Args:
            db_file: Database file path
        """
        self.db_file = os.path.abspath(db_file or "audit.db")
        self.conn = None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        logger.info(f"SQLite audit storage initialized at {self.db_file}")
    
    def _init_db(self) -> None:
        """Initialize database."""
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        
        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL;")
        
        # Create tables
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            source TEXT NOT NULL,
            data TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """)
        
        # Create indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events (event_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON audit_events (source)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events (timestamp)")
        
        self.conn.commit()
    
    def store_event(self, event: AuditEvent) -> bool:
        """
        Store an audit event.
        
        Args:
            event: Audit event to store
            
        Returns:
            Whether the event was stored
        """
        try:
            self.conn.execute(
                "INSERT INTO audit_events (event_id, event_type, source, data, timestamp) VALUES (?, ?, ?, ?, ?)",
                (event.event_id, event.event_type, event.source, json.dumps(event.data), event.timestamp)
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error storing audit event: {e}")
            return False
    
    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """
        Get an audit event by ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            Audit event, or None if not found
        """
        try:
            cursor = self.conn.execute(
                "SELECT event_id, event_type, source, data, timestamp FROM audit_events WHERE event_id = ?",
                (event_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return AuditEvent(
                event_id=row[0],
                event_type=row[1],
                source=row[2],
                data=json.loads(row[3]),
                timestamp=row[4]
            )
        except Exception as e:
            logger.error(f"Error getting audit event: {e}")
            return None
    
    def query_events(self, query: Dict[str, Any] = None, 
                    start_time: float = None, end_time: float = None,
                    limit: int = 100, offset: int = 0) -> List[AuditEvent]:
        """
        Query audit events.
        
        Args:
            query: Query criteria
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of events to return
            offset: Offset for pagination
            
        Returns:
            List of audit events
        """
        try:
            # Build query
            sql = "SELECT event_id, event_type, source, data, timestamp FROM audit_events WHERE 1=1"
            params = []
            
            # Add time range
            if start_time is not None:
                sql += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time is not None:
                sql += " AND timestamp <= ?"
                params.append(end_time)
            
            # Add query criteria
            if query:
                if "event_type" in query:
                    sql += " AND event_type = ?"
                    params.append(query["event_type"])
                
                if "source" in query:
                    sql += " AND source = ?"
                    params.append(query["source"])
            
            # Add order, limit, and offset
            sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.append(limit)
            params.append(offset)
            
            # Execute query
            cursor = self.conn.execute(sql, params)
            rows = cursor.fetchall()
            
            # Convert to AuditEvent objects
            events = []
            for row in rows:
                events.append(AuditEvent(
                    event_id=row[0],
                    event_type=row[1],
                    source=row[2],
                    data=json.loads(row[3]),
                    timestamp=row[4]
                ))
            
            return events
        except Exception as e:
            logger.error(f"Error querying audit events: {e}")
            return []
    
    def get_event_count(self, query: Dict[str, Any] = None,
                       start_time: float = None, end_time: float = None) -> int:
        """
        Get count of audit events matching criteria.
        
        Args:
            query: Query criteria
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Event count
        """
        try:
            # Build query
            sql = "SELECT COUNT(*) FROM audit_events WHERE 1=1"
            params = []
            
            # Add time range
            if start_time is not None:
                sql += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time is not None:
                sql += " AND timestamp <= ?"
                params.append(end_time)
            
            # Add query criteria
            if query:
                if "event_type" in query:
                    sql += " AND event_type = ?"
                    params.append(query["event_type"])
                
                if "source" in query:
                    sql += " AND source = ?"
                    params.append(query["source"])
            
            # Execute query
            cursor = self.conn.execute(sql, params)
            row = cursor.fetchone()
            
            return row[0] if row else 0
        except Exception as e:
            logger.error(f"Error getting audit event count: {e}")
            return 0
    
    def clear_events(self, older_than: float = None) -> int:
        """
        Clear audit events.
        
        Args:
            older_than: Clear events older than this timestamp
            
        Returns:
            Number of events cleared
        """
        try:
            if older_than is None:
                # Clear all events
                cursor = self.conn.execute("DELETE FROM audit_events")
            else:
                # Clear events older than timestamp
                cursor = self.conn.execute(
                    "DELETE FROM audit_events WHERE timestamp < ?",
                    (older_than,)
                )
            
            self.conn.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Error clearing audit events: {e}")
            return 0

class MemoryAuditStorage(AuditStorage):
    """
    In-memory implementation of audit storage.
    """
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize memory audit storage.
        
        Args:
            max_events: Maximum number of events to store
        """
        self.events = {}
        self.max_events = max_events
        self.lock = threading.Lock()
        
        logger.info(f"Memory audit storage initialized with max_events={max_events}")
    
    def store_event(self, event: AuditEvent) -> bool:
        """
        Store an audit event.
        
        Args:
            event: Audit event to store
            
        Returns:
            Whether the event was stored
        """
        with self.lock:
            self.events[event.event_id] = event
            
            # Trim events if needed
            if len(self.events) > self.max_events:
                # Sort by timestamp and remove oldest
                event_ids = sorted(self.events.keys(), key=lambda eid: self.events[eid].timestamp)
                to_remove = event_ids[:len(self.events) - self.max_events]
                
                for eid in to_remove:
                    del self.events[eid]
            
            return True
    
    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """
        Get an audit event by ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            Audit event, or None if not found
        """
        with self.lock:
            return self.events.get(event_id)
    
    def query_events(self, query: Dict[str, Any] = None, 
                    start_time: float = None, end_time: float = None,
                    limit: int = 100, offset: int = 0) -> List[AuditEvent]:
        """
        Query audit events.
        
        Args:
            query: Query criteria
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of events to return
            offset: Offset for pagination
            
        Returns:
            List of audit events
        """
        with self.lock:
            # Filter events
            filtered_events = self.events.values()
            
            if start_time is not None:
                filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
            
            if end_time is not None:
                filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
            
            if query:
                if "event_type" in query:
                    filtered_events = [e for e in filtered_events if e.event_type == query["event_type"]]
                
                if "source" in query:
                    filtered_events = [e for e in filtered_events if e.source == query["source"]]
            
            # Sort by timestamp (newest first)
            filtered_events = sorted(filtered_events, key=lambda e: e.timestamp, reverse=True)
            
            # Apply pagination
            paginated_events = filtered_events[offset:offset + limit]
            
            return paginated_events
    
    def get_event_count(self, query: Dict[str, Any] = None,
                       start_time: float = None, end_time: float = None) -> int:
        """
        Get count of audit events matching criteria.
        
        Args:
            query: Query criteria
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Event count
        """
        with self.lock:
            # Filter events
            filtered_events = self.events.values()
            
            if start_time is not None:
                filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
            
            if end_time is not None:
                filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
            
            if query:
                if "event_type" in query:
                    filtered_events = [e for e in filtered_events if e.event_type == query["event_type"]]
                
                if "source" in query:
                    filtered_events = [e for e in filtered_events if e.source == query["source"]]
            
            return len(filtered_events)
    
    def clear_events(self, older_than: float = None) -> int:
        """
        Clear audit events.
        
        Args:
            older_than: Clear events older than this timestamp
            
        Returns:
            Number of events cleared
        """
        with self.lock:
            if older_than is None:
                # Clear all events
                count = len(self.events)
                self.events = {}
                return count
            else:
                # Clear events older than timestamp
                to_delete = [eid for eid, event in self.events.items() if event.timestamp < older_than]
                
                for eid in to_delete:
                    del self.events[eid]
                
                return len(to_delete)

class AuditManager:
    """
    Manages audit events and storage.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize audit manager.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.storage_type = self.config.get("storage_type", "sqlite")
        self.storage = self._create_storage()
        self.retention_days = self.config.get("retention_days", 90)
        self.cleanup_interval = self.config.get("cleanup_interval", 86400)  # 1 day
        self.cleanup_thread = None
        self.stop_event = threading.Event()
        self.batch_size = self.config.get("batch_size", 100)
        self.batch_interval = self.config.get("batch_interval", 5)  # seconds
        self.batch_queue = queue.Queue()
        self.batch_thread = None
        
        # Start cleanup thread if enabled
        if self.config.get("auto_cleanup", True):
            self._start_cleanup_thread()
        
        # Start batch thread if enabled
        if self.config.get("batch_mode", True):
            self._start_batch_thread()
        
        logger.info(f"Audit manager initialized with storage_type={self.storage_type}")
    
    def _create_storage(self) -> AuditStorage:
        """
        Create audit storage.
        
        Returns:
            Audit storage
        """
        if self.storage_type == "sqlite":
            db_file = self.config.get("db_file", "audit.db")
            return SqliteAuditStorage(db_file)
        elif self.storage_type == "memory":
            max_events = self.config.get("max_events", 10000)
            return MemoryAuditStorage(max_events)
        else:
            raise ValueError(f"Invalid storage type: {self.storage_type}")
    
    def _start_cleanup_thread(self) -> None:
        """Start cleanup thread."""
        self.stop_event.clear()
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
        
        logger.info("Cleanup thread started")
    
    def _cleanup_loop(self) -> None:
        """Cleanup loop for old audit events."""
        while not self.stop_event.is_set():
            try:
                # Calculate cutoff timestamp
                cutoff = time.time() - (self.retention_days * 86400)
                
                # Clear old events
                cleared = self.storage.clear_events(older_than=cutoff)
                
                if cleared > 0:
                    logger.info(f"Cleared {cleared} audit events older than {datetime.datetime.fromtimestamp(cutoff)}")
                
                # Sleep until next cleanup
                self.stop_event.wait(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                self.stop_event.wait(60)  # Wait a bit before retrying
    
    def _start_batch_thread(self) -> None:
        """Start batch thread."""
        self.stop_event.clear()
        self.batch_thread = threading.Thread(target=self._batch_loop)
        self.batch_thread.daemon = True
        self.batch_thread.start()
        
        logger.info("Batch thread started")
    
    def _batch_loop(self) -> None:
        """Batch loop for storing audit events."""
        while not self.stop_event.is_set():
            try:
                batch = []
                
                # Try to get events from queue
                try:
                    # Get first event with timeout
                    event = self.batch_queue.get(timeout=self.batch_interval)
                    batch.append(event)
                    
                    # Get more events without timeout
                    for _ in range(self.batch_size - 1):
                        try:
                            event = self.batch_queue.get_nowait()
                            batch.append(event)
                        except queue.Empty:
                            break
                except queue.Empty:
                    continue
                
                # Store batch
                if batch:
                    for event in batch:
                        self.storage.store_event(event)
                        self.batch_queue.task_done()
                    
                    logger.debug(f"Stored batch of {len(batch)} audit events")
            except Exception as e:
                logger.error(f"Error in batch loop: {e}")
                self.stop_event.wait(5)  # Wait a bit before retrying
    
    def log_event(self, event_type: str, source: str, data: Dict[str, Any] = None) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            source: Source of event
            data: Additional event data
            
        Returns:
            Created audit event
        """
        event = AuditEvent(event_type, source, data)
        
        # If batch mode is enabled, add to queue
        if self.config.get("batch_mode", True) and self.batch_thread and self.batch_thread.is_alive():
            self.batch_queue.put(event)
        else:
            # Otherwise, store directly
            self.storage.store_event(event)
        
        return event
    
    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """
        Get an audit event by ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            Audit event, or None if not found
        """
        return self.storage.get_event(event_id)
    
    def query_events(self, query: Dict[str, Any] = None, 
                    start_time: float = None, end_time: float = None,
                    limit: int = 100, offset: int = 0) -> List[AuditEvent]:
        """
        Query audit events.
        
        Args:
            query: Query criteria
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of events to return
            offset: Offset for pagination
            
        Returns:
            List of audit events
        """
        return self.storage.query_events(query, start_time, end_time, limit, offset)
    
    def get_event_count(self, query: Dict[str, Any] = None,
                       start_time: float = None, end_time: float = None) -> int:
        """
        Get count of audit events matching criteria.
        
        Args:
            query: Query criteria
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Event count
        """
        return self.storage.get_event_count(query, start_time, end_time)
    
    def clear_events(self, older_than: float = None) -> int:
        """
        Clear audit events.
        
        Args:
            older_than: Clear events older than this timestamp
            
        Returns:
            Number of events cleared
        """
        return self.storage.clear_events(older_than)
    
    def shutdown(self) -> None:
        """Shut down audit manager."""
        # Stop threads
        self.stop_event.set()
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
            if self.cleanup_thread.is_alive():
                logger.warning("Cleanup thread did not terminate gracefully")
        
        if self.batch_thread:
            self.batch_thread.join(timeout=5)
            if self.batch_thread.is_alive():
                logger.warning("Batch thread did not terminate gracefully")
            
            # Process remaining events in queue
            try:
                while True:
                    event = self.batch_queue.get_nowait()
                    self.storage.store_event(event)
                    self.batch_queue.task_done()
            except queue.Empty:
                pass
        
        logger.info("Audit manager shut down")

class AuditorRegistry:
    """
    Registry for auditor hooks and callbacks.
    """
    
    def __init__(self):
        """Initialize auditor registry."""
        self.hooks = {}
        self.callbacks = {}
        
        logger.info("Auditor registry initialized")
    
    def register_hook(self, hook_name: str, hook_func: Callable) -> None:
        """
        Register an auditor hook.
        
        Args:
            hook_name: Hook name
            hook_func: Hook function
        """
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        
        self.hooks[hook_name].append(hook_func)
        logger.debug(f"Registered hook: {hook_name}")
    
    def unregister_hook(self, hook_name: str, hook_func: Callable) -> bool:
        """
        Unregister an auditor hook.
        
        Args:
            hook_name: Hook name
            hook_func: Hook function
            
        Returns:
            Whether the hook was unregistered
        """
        if hook_name not in self.hooks:
            return False
        
        if hook_func in self.hooks[hook_name]:
            self.hooks[hook_name].remove(hook_func)
            logger.debug(f"Unregistered hook: {hook_name}")
            return True
        
        return False
    
    def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Trigger an auditor hook.
        
        Args:
            hook_name: Hook name
            *args: Hook arguments
            **kwargs: Hook keyword arguments
            
        Returns:
            List of hook results
        """
        if hook_name not in self.hooks:
            return []
        
        results = []
        
        for hook_func in self.hooks[hook_name]:
            try:
                result = hook_func(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in hook {hook_name}: {e}")
                results.append(None)
        
        return results
    
    def register_callback(self, event_type: str, callback_func: Callable) -> None:
        """
        Register an event callback.
        
        Args:
            event_type: Event type
            callback_func: Callback function
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        self.callbacks[event_type].append(callback_func)
        logger.debug(f"Registered callback for event type: {event_type}")
    
    def unregister_callback(self, event_type: str, callback_func: Callable) -> bool:
        """
        Unregister an event callback.
        
        Args:
            event_type: Event type
            callback_func: Callback function
            
        Returns:
            Whether the callback was unregistered
        """
        if event_type not in self.callbacks:
            return False
        
        if callback_func in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback_func)
            logger.debug(f"Unregistered callback for event type: {event_type}")
            return True
        
        return False
    
    def trigger_callbacks(self, event: AuditEvent) -> List[Any]:
        """
        Trigger callbacks for an event.
        
        Args:
            event: Audit event
            
        Returns:
            List of callback results
        """
        if event.event_type not in self.callbacks:
            return []
        
        results = []
        
        for callback_func in self.callbacks[event.event_type]:
            try:
                result = callback_func(event)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in callback for event type {event.event_type}: {e}")
                results.append(None)
        
        return results

# Global registry and manager instances
auditor_registry = AuditorRegistry()
audit_manager = None

def initialize_audit_manager(config: Dict[str, Any] = None) -> AuditManager:
    """
    Initialize the global audit manager.
    
    Args:
        config: Configuration options
        
    Returns:
        Audit manager
    """
    global audit_manager
    
    if audit_manager is None:
        audit_manager = AuditManager(config)
    
    return audit_manager

def get_audit_manager() -> Optional[AuditManager]:
    """
    Get the global audit manager.
    
    Returns:
        Audit manager, or None if not initialized
    """
    return audit_manager

def log_event(event_type: str, source: str, data: Dict[str, Any] = None) -> Optional[AuditEvent]:
    """
    Log an audit event.
    
    Args:
        event_type: Type of event
        source: Source of event
        data: Additional event data
        
    Returns:
        Created audit event, or None if audit manager is not initialized
    """
    if audit_manager is None:
        logger.warning("Audit manager not initialized")
        return None
    
    event = audit_manager.log_event(event_type, source, data)
    
    # Trigger callbacks
    auditor_registry.trigger_callbacks(event)
    
    return event

def register_hook(hook_name: str, hook_func: Callable) -> None:
    """
    Register an auditor hook.
    
    Args:
        hook_name: Hook name
        hook_func: Hook function
    """
    auditor_registry.register_hook(hook_name, hook_func)

def unregister_hook(hook_name: str, hook_func: Callable) -> bool:
    """
    Unregister an auditor hook.
    
    Args:
        hook_name: Hook name
        hook_func: Hook function
        
    Returns:
        Whether the hook was unregistered
    """
    return auditor_registry.unregister_hook(hook_name, hook_func)

def trigger_hook(hook_name: str, *args, **kwargs) -> List[Any]:
    """
    Trigger an auditor hook.
    
    Args:
        hook
