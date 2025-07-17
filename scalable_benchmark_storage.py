"""
FixWurx Auditor - Scalable Benchmark Storage

This module implements a tiered storage system for benchmark data with
memory caching, SQLite databases, and compressed archives for efficient
storage and retrieval.
"""

import logging
import time
import json
import os
import sqlite3
import shutil
from typing import Dict, List, Any, Optional, Tuple, Iterator
from datetime import datetime, timedelta
import threading
import gzip

logger = logging.getLogger('scalable_benchmark_storage')

class ScalableBenchmarkStorage:
    """
    Scalable storage system for benchmark data with tiered architecture.
    
    This system implements a tiered storage approach:
    1. In-memory cache for most recent/active data
    2. SQLite databases for medium-term storage
    3. Compressed JSON archives for long-term storage
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ScalableBenchmarkStorage.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Base storage directories
        self.base_dir = self.config.get("base_dir", "auditor_data/benchmarks")
        self.active_dir = os.path.join(self.base_dir, "active")
        self.archive_dir = os.path.join(self.base_dir, "archive")
        
        # Create directories
        os.makedirs(self.active_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        
        # Memory cache settings
        self.cache_size = self.config.get("cache_size", 1000)  # Max entries in memory
        self.cache_ttl = self.config.get("cache_ttl", 3600)    # Time to live (seconds)
        
        # In-memory cache (project -> session -> timestamp -> data)
        self.cache: Dict[str, Dict[str, Dict[float, Dict[str, Any]]]] = {}
        self.cache_timestamps: Dict[str, Dict[str, float]] = {}  # Last access time
        
        # Cache lock for thread safety
        self.cache_lock = threading.RLock()
        
        # Database connections (project -> session -> connection)
        self.db_connections: Dict[str, Dict[str, sqlite3.Connection]] = {}
        self.db_lock = threading.RLock()  # Lock for database operations
        
        # Archiving settings
        self.archive_threshold = self.config.get("archive_threshold", 30)  # Days
        self.compression_level = self.config.get("compression_level", 9)   # 1-9 (9 is max)
        
        # Start background cleanup thread
        self.cleanup_interval = self.config.get("cleanup_interval", 3600)  # Hourly
        self.cleanup_stop = threading.Event()
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_routine,
            daemon=True,
            name="BenchmarkStorageCleanup"
        )
        self.cleanup_thread.start()
        
        # Statistics
        self.stats = {
            "store_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "db_operations": 0,
            "archive_operations": 0
        }
        
        logger.info(f"Initialized ScalableBenchmarkStorage in {self.base_dir}")
    
    def store_benchmark(self, project: str, session: str, 
                       benchmark: Dict[str, Any]) -> bool:
        """
        Store a benchmark data point.
        
        Args:
            project: Project identifier
            session: Session identifier
            benchmark: Benchmark data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure benchmark has a timestamp
            if "timestamp" not in benchmark:
                benchmark["timestamp"] = time.time()
            
            timestamp = benchmark["timestamp"]
            
            # Store in memory cache
            with self.cache_lock:
                if project not in self.cache:
                    self.cache[project] = {}
                    self.cache_timestamps[project] = {}
                
                if session not in self.cache[project]:
                    self.cache[project][session] = {}
                    self.cache_timestamps[project][session] = time.time()
                else:
                    self.cache_timestamps[project][session] = time.time()
                
                # Add to cache
                self.cache[project][session][timestamp] = benchmark
                
                # Check cache size and evict if necessary
                self._check_cache_size(project, session)
            
            # Store in database (async for better performance)
            self.stats["store_operations"] += 1
            threading.Thread(
                target=self._store_in_db,
                args=(project, session, benchmark),
                daemon=True
            ).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing benchmark: {str(e)}")
            return False
    
    def get_benchmarks(self, project: str, session: str, 
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get benchmark data for a session.
        
        Args:
            project: Project identifier
            session: Session identifier
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of benchmarks to return
            
        Returns:
            List of benchmark data points
        """
        results = []
        
        try:
            # Check memory cache first
            with self.cache_lock:
                if project in self.cache and session in self.cache[project]:
                    # Update last access time
                    self.cache_timestamps[project][session] = time.time()
                    
                    # Get data from cache
                    cache_data = self.cache[project][session]
                    
                    # Apply time filters
                    filtered_data = {}
                    for ts, data in cache_data.items():
                        if ((start_time is None or ts >= start_time) and 
                            (end_time is None or ts <= end_time)):
                            filtered_data[ts] = data
                    
                    # Sort by timestamp
                    sorted_timestamps = sorted(filtered_data.keys())
                    
                    # Add to results (up to limit)
                    for ts in sorted_timestamps[:limit]:
                        results.append(filtered_data[ts])
                    
                    self.stats["cache_hits"] += 1
            
            # If we have enough results, return them
            if len(results) >= limit:
                return results[:limit]
            
            # Otherwise, query database
            self.stats["cache_misses"] += 1
            remaining = limit - len(results)
            db_results = self._query_db(
                project, session, start_time, end_time, remaining
            )
            
            # Combine results
            results.extend(db_results)
            
            # If we still need more, check archives
            if len(results) < limit and (
                start_time is None or 
                start_time < time.time() - (self.archive_threshold * 86400)
            ):
                remaining = limit - len(results)
                archive_results = self._query_archives(
                    project, session, start_time, end_time, remaining
                )
                results.extend(archive_results)
            
            # Sort by timestamp and return
            return sorted(results, key=lambda x: x["timestamp"])[:limit]
            
        except Exception as e:
            logger.error(f"Error getting benchmarks: {str(e)}")
            return []
    
    def get_sessions(self, project: str) -> List[Dict[str, Any]]:
        """
        Get a list of all sessions for a project.
        
        Args:
            project: Project identifier
            
        Returns:
            List of session information dictionaries
        """
        sessions = []
        
        try:
            # Get active sessions from database
            db_path = os.path.join(self.active_dir, f"{project}_sessions.db")
            if os.path.exists(db_path):
                with self.db_lock:
                    conn = self._get_sessions_db(project)
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        "SELECT session_id, start_time, end_time, metadata FROM sessions"
                    )
                    
                    for row in cursor.fetchall():
                        session_id, start_time, end_time, metadata_json = row
                        metadata = json.loads(metadata_json) if metadata_json else {}
                        
                        sessions.append({
                            "session_id": session_id,
                            "start_time": start_time,
                            "end_time": end_time,
                            "metadata": metadata,
                            "status": "active"
                        })
            
            # Get archived sessions
            archive_dir = os.path.join(self.archive_dir, project)
            if os.path.exists(archive_dir):
                for filename in os.listdir(archive_dir):
                    if filename.endswith("_meta.json"):
                        session_id = filename[:-10]  # Remove _meta.json
                        
                        # Read metadata
                        with open(os.path.join(archive_dir, filename), 'r') as f:
                            metadata = json.load(f)
                        
                        sessions.append({
                            "session_id": session_id,
                            "start_time": metadata.get("start_time", 0),
                            "end_time": metadata.get("end_time", 0),
                            "metadata": metadata.get("metadata", {}),
                            "status": "archived"
                        })
            
            # Sort by start time (descending)
            return sorted(sessions, key=lambda x: x["start_time"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting sessions: {str(e)}")
            return []
    
    def create_session(self, project: str, session: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new session.
        
        Args:
            project: Project identifier
            session: Session identifier
            metadata: Optional metadata for the session
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get sessions database
            with self.db_lock:
                conn = self._get_sessions_db(project)
                cursor = conn.cursor()
                
                # Check if session already exists
                cursor.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?",
                    (session,)
                )
                
                if cursor.fetchone():
                    # Update existing session
                    cursor.execute(
                        "UPDATE sessions SET metadata = ? WHERE session_id = ?",
                        (json.dumps(metadata or {}), session)
                    )
                else:
                    # Create new session
                    cursor.execute(
                        "INSERT INTO sessions (session_id, start_time, metadata) VALUES (?, ?, ?)",
                        (session, time.time(), json.dumps(metadata or {}))
                    )
                
                conn.commit()
                
                # Create session database
                self._get_session_db(project, session)
            
            logger.info(f"Created session {session} for project {project}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            return False
    
    def end_session(self, project: str, session: str,
                   metadata_updates: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark a session as ended.
        
        Args:
            project: Project identifier
            session: Session identifier
            metadata_updates: Optional metadata updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get sessions database
            with self.db_lock:
                conn = self._get_sessions_db(project)
                cursor = conn.cursor()
                
                # Update session end time
                if metadata_updates:
                    cursor.execute(
                        "SELECT metadata FROM sessions WHERE session_id = ?",
                        (session,)
                    )
                    row = cursor.fetchone()
                    if row:
                        metadata_json = row[0]
                        metadata = json.loads(metadata_json) if metadata_json else {}
                        metadata.update(metadata_updates)
                        
                        cursor.execute(
                            "UPDATE sessions SET end_time = ?, metadata = ? WHERE session_id = ?",
                            (time.time(), json.dumps(metadata), session)
                        )
                else:
                    cursor.execute(
                        "UPDATE sessions SET end_time = ? WHERE session_id = ?",
                        (time.time(), session)
                    )
                
                conn.commit()
            
            # Flush cache to database
            self._flush_session_cache(project, session)
            
            logger.info(f"Ended session {session} for project {project}")
            return True
            
        except Exception as e:
            logger.error(f"Error ending session: {str(e)}")
            return False
    
    def delete_session(self, project: str, session: str) -> bool:
        """
        Delete a session and all its data.
        
        Args:
            project: Project identifier
            session: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from memory cache
            with self.cache_lock:
                if project in self.cache and session in self.cache[project]:
                    del self.cache[project][session]
                    del self.cache_timestamps[project][session]
            
            # Delete from sessions database
            with self.db_lock:
                conn = self._get_sessions_db(project)
                cursor = conn.cursor()
                
                cursor.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session,)
                )
                
                conn.commit()
                
                # Delete session database
                db_path = os.path.join(self.active_dir, f"{project}_{session}.db")
                if os.path.exists(db_path):
                    # Close connection if open
                    if (project in self.db_connections and 
                        session in self.db_connections[project]):
                        self.db_connections[project][session].close()
                        del self.db_connections[project][session]
                    
                    # Delete file
                    os.remove(db_path)
            
            # Delete archive if exists
            archive_path = os.path.join(self.archive_dir, project, f"{session}.gz")
            if os.path.exists(archive_path):
                os.remove(archive_path)
                
            meta_path = os.path.join(self.archive_dir, project, f"{session}_meta.json")
            if os.path.exists(meta_path):
                os.remove(meta_path)
            
            logger.info(f"Deleted session {session} for project {project}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = self.stats.copy()
        
        # Count projects and sessions
        stats["projects"] = 0
        stats["active_sessions"] = 0
        stats["archived_sessions"] = 0
        
        # Count active sessions
        for project_file in os.listdir(self.active_dir):
            if project_file.endswith("_sessions.db"):
                stats["projects"] += 1
                
                # Count sessions in this project
                project = project_file[:-12]  # Remove _sessions.db
                
                try:
                    with self.db_lock:
                        conn = self._get_sessions_db(project)
                        cursor = conn.cursor()
                        
                        cursor.execute("SELECT COUNT(*) FROM sessions")
                        count = cursor.fetchone()[0]
                        stats["active_sessions"] += count
                except Exception as e:
                    logger.error(f"Error counting sessions for {project}: {str(e)}")
        
        # Count archived sessions
        for project_dir in os.listdir(self.archive_dir):
            project_path = os.path.join(self.archive_dir, project_dir)
            if os.path.isdir(project_path):
                # Count meta files
                meta_files = [f for f in os.listdir(project_path) if f.endswith("_meta.json")]
                stats["archived_sessions"] += len(meta_files)
        
        # Cache stats
        stats["cache_entries"] = 0
        with self.cache_lock:
            for project in self.cache:
                for session in self.cache[project]:
                    stats["cache_entries"] += len(self.cache[project][session])
        
        # Database connections
        stats["db_connections"] = 0
        with self.db_lock:
            for project in self.db_connections:
                stats["db_connections"] += len(self.db_connections[project])
        
        return stats
    
    def _get_sessions_db(self, project: str) -> sqlite3.Connection:
        """Get connection to the sessions database for a project."""
        # Create database if it doesn't exist
        db_path = os.path.join(self.active_dir, f"{project}_sessions.db")
        conn = sqlite3.connect(db_path)
        
        # Create table if it doesn't exist
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL,
                metadata TEXT
            )
        """)
        conn.commit()
        
        return conn
    
    def _get_session_db(self, project: str, session: str) -> sqlite3.Connection:
        """Get connection to the database for a specific session."""
        # Check if we already have a connection
        with self.db_lock:
            if (project in self.db_connections and 
                session in self.db_connections[project]):
                return self.db_connections[project][session]
            
            # Create database if it doesn't exist
            db_path = os.path.join(self.active_dir, f"{project}_{session}.db")
            conn = sqlite3.connect(db_path)
            
            # Create table if it doesn't exist
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmarks (
                    timestamp REAL PRIMARY KEY,
                    data TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON benchmarks(timestamp)")
            conn.commit()
            
            # Store connection
            if project not in self.db_connections:
                self.db_connections[project] = {}
            self.db_connections[project][session] = conn
            
            return conn
    
    def _store_in_db(self, project: str, session: str, benchmark: Dict[str, Any]):
        """Store a benchmark in the database."""
        try:
            self.stats["db_operations"] += 1
            with self.db_lock:
                conn = self._get_session_db(project, session)
                cursor = conn.cursor()
                
                timestamp = benchmark["timestamp"]
                data_json = json.dumps(benchmark)
                
                # Insert or replace
                cursor.execute(
                    "INSERT OR REPLACE INTO benchmarks (timestamp, data) VALUES (?, ?)",
                    (timestamp, data_json)
                )
                
                conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing in database: {str(e)}")
    
    def _query_db(self, project: str, session: str, 
                 start_time: Optional[float], end_time: Optional[float],
                 limit: int) -> List[Dict[str, Any]]:
        """Query the database for benchmarks."""
        results = []
        
        try:
            # Get database connection
            with self.db_lock:
                try:
                    conn = self._get_session_db(project, session)
                    cursor = conn.cursor()
                    
                    # Build query
                    query = "SELECT data FROM benchmarks"
                    params = []
                    
                    if start_time is not None or end_time is not None:
                        query += " WHERE"
                        
                        if start_time is not None:
                            query += " timestamp >= ?"
                            params.append(start_time)
                            
                            if end_time is not None:
                                query += " AND"
                        
                        if end_time is not None:
                            query += " timestamp <= ?"
                            params.append(end_time)
                    
                    query += " ORDER BY timestamp DESC LIMIT ?"
                    params.append(limit)
                    
                    # Execute query
                    cursor.execute(query, params)
                    
                    # Process results
                    for row in cursor.fetchall():
                        data_json = row[0]
                        benchmark = json.loads(data_json)
                        results.append(benchmark)
                    
                except sqlite3.OperationalError:
                    # Database doesn't exist yet
                    pass
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}")
            return []
    
    def _query_archives(self, project: str, session: str,
                       start_time: Optional[float], end_time: Optional[float],
                       limit: int) -> List[Dict[str, Any]]:
        """Query archived data for benchmarks."""
        results = []
        
        try:
            self.stats["archive_operations"] += 1
            
            # Check if archive exists
            archive_path = os.path.join(self.archive_dir, project, f"{session}.gz")
            if not os.path.exists(archive_path):
                return []
            
            # Read archive
            with gzip.open(archive_path, 'rt') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    benchmark = json.loads(line)
                    timestamp = benchmark.get("timestamp", 0)
                    
                    # Apply time filters
                    if ((start_time is None or timestamp >= start_time) and 
                        (end_time is None or timestamp <= end_time)):
                        results.append(benchmark)
                    
                    # Check limit
                    if len(results) >= limit:
                        break
            
            # Sort by timestamp (descending)
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error querying archives: {str(e)}")
            return []
    
    def _check_cache_size(self, project: str, session: str):
        """Check if cache is too large and evict if necessary."""
        session_cache = self.cache[project][session]
        
        # Check if we're over the limit
        if len(session_cache) > self.cache_size:
            # Sort by timestamp
            sorted_timestamps = sorted(session_cache.keys())
            
            # Keep only the newest entries
            to_keep = sorted_timestamps[-self.cache_size:]
            
            # Create new cache with only the entries to keep
            new_cache = {ts: session_cache[ts] for ts in to_keep}
            
            # Write evicted entries to database
            evicted = {ts: session_cache[ts] for ts in sorted_timestamps[:-self.cache_size]}
            for ts, data in evicted.items():
                self._store_in_db(project, session, data)
            
            # Update cache
            self.cache[project][session] = new_cache
    
    def _flush_session_cache(self, project: str, session: str):
        """Flush session cache to database."""
        with self.cache_lock:
            if project in self.cache and session in self.cache[project]:
                # Store all entries in database
                for ts, data in self.cache[project][session].items():
                    self._store_in_db(project, session, data)
                
                # Clear cache
                self.cache[project][session] = {}
                
                logger.debug(f"Flushed cache for {project}/{session}")
    
    def _cleanup_routine(self):
        """Background routine for cleanup and archiving."""
        while not self.cleanup_stop.wait(self.cleanup_interval):
            try:
                # Clean up cache
                self._cleanup_cache()
                
                # Archive old sessions
                self._archive_old_sessions()
                
                # Clean up database connections
                self._cleanup_db_connections()
                
                logger.info("Completed storage cleanup routine")
                
            except Exception as e:
                logger.error(f"Error in cleanup routine: {str(e)}")
    
    def _cleanup_cache(self):
        """Clean up old entries from cache."""
        with self.cache_lock:
            current_time = time.time()
            projects_to_remove = []
            
            for project in self.cache:
                sessions_to_remove = []
                
                for session in self.cache[project]:
                    # Check if session is old
                    last_access = self.cache_timestamps[project][session]
                    if current_time - last_access > self.cache_ttl:
                        # Flush to database
                        self._flush_session_cache(project, session)
                        sessions_to_remove.append(session)
                
                # Remove old sessions
                for session in sessions_to_remove:
                    del self.cache[project][session]
                    del self.cache_timestamps[project][session]
                
                # Check if project is empty
                if not self.cache[project]:
                    projects_to_remove.append(project)
            
            # Remove empty projects
            for project in projects_to_remove:
                del self.cache[project]
                del self.cache_timestamps[project]
            
            if projects_to_remove or sum(len(sessions_to_remove) for _, sessions_to_remove in enumerate(self.cache.values())):
                logger.debug("Cleaned up cache")
    
    def _archive_old_sessions(self):
        """Archive old sessions."""
        try:
            cutoff_time = time.time() - (self.archive_threshold * 86400)
            
            # Check each project
            for project_file in os.listdir(self.active_dir):
                if project_file.endswith("_sessions.db"):
                    project = project_file[:-12]  # Remove _sessions.db
                    
                    # Get sessions database
                    with self.db_lock:
                        db_path = os.path.join(self.active_dir, project_file)
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        
                        # Find old sessions
                        cursor.execute(
                            "SELECT session_id, start_time, end_time, metadata FROM sessions "
                            "WHERE end_time IS NOT NULL AND end_time < ?",
                            (cutoff_time,)
                        )
                        
                        for row in cursor.fetchall():
                            session_id, start_time, end_time, metadata_json = row
                            
                            # Archive session
                            self._archive_session(project, session_id, {
                                "start_time": start_time,
                                "end_time": end_time,
                                "metadata": json.loads(metadata_json) if metadata_json else {}
                            })
                            
                            # Delete from sessions database
                            cursor.execute(
                                "DELETE FROM sessions WHERE session_id = ?",
                                (session_id,)
                            )
                        
                        conn.commit()
                        
                        # Vacuum database
                        conn.execute("VACUUM")
                        conn.commit()
                        conn.close()
            
        except Exception as e:
            logger.error(f"Error archiving old sessions: {str(e)}")
    
    def _archive_session(self, project: str, session: str, metadata: Dict[str, Any]):
        """Archive a single session."""
        try:
            # Create project directory in archive
            project_dir = os.path.join(self.archive_dir, project)
            os.makedirs(project_dir, exist_ok=True)
            
            # Get session database
            db_path = os.path.join(self.active_dir, f"{project}_{session}.db")
            if not os.path.exists(db_path):
                return
            
            # Close connection if open
            with self.db_lock:
                if (project in self.db_connections and 
                    session in self.db_connections[project]):
                    self.db_connections[project][session].close()
                    del self.db_connections[project][session]
                
                # Open database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get all benchmarks
                cursor.execute("SELECT data FROM benchmarks ORDER BY timestamp")
                
                # Create archive file
                archive_path = os.path.join(project_dir, f"{session}.gz")
                with gzip.open(archive_path, 'wt', compresslevel=self.compression_level) as f:
                    for row in cursor.fetchall():
                        data_json = row[0]
                        f.write(data_json + "\n")
                
                # Create metadata file
                meta_path = os.path.join(project_dir, f"{session}_meta.json")
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Close and delete database
                conn.close()
                os.remove(db_path)
            
            logger.info(f"Archived session {project}/{session}")
            
        except Exception as e:
            logger.error(f"Error archiving session {project}/{session}: {str(e)}")
    
    def _cleanup_db_connections(self):
        """Clean up old database connections."""
        try:
            with self.db_lock:
                projects_to_remove = []
                
                for project in self.db_connections:
                    sessions_to_remove = []
                    
                    for session in self.db_connections[project]:
                        # Check if database file still exists
                        db_path = os.path.join(self.active_dir, f"{project}_{session}.db")
                        if not os.path.exists(db_path):
                            # Close connection
                            self.db_connections[project][session].close()
                            sessions_to_remove.append(session)
                    
                    # Remove closed connections
                    for session in sessions_to_remove:
                        del self.db_connections[project][session]
                    
                    # Check if project is empty
                    if not self.db_connections[project]:
                        projects_to_remove.append(project)
                
                # Remove empty projects
                for project in projects_to_remove:
                    del self.db_connections[project]
                
                if projects_to_remove or sum(len(sessions_to_remove) for _, sessions_to_remove in enumerate(self.db_connections.values())):
                    logger.debug("Cleaned up database connections")
            
        except Exception as e:
            logger.error(f"Error cleaning up database connections: {str(e)}")
    
    def close(self):
        """Close storage and clean up resources."""
        # Stop cleanup thread
        self.cleanup_stop.set()
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        # Flush all caches
        with self.cache_lock:
            for project in list(self.cache.keys()):
                for session in list(self.cache[project].keys()):
                    self._flush_session_cache(project, session)
        
        # Close database connections
        with self.db_lock:
            for project in list(self.db_connections.keys()):
                for session in list(self.db_connections[project].keys()):
                    try:
                        self.db_connections[project][session].close()
                    except Exception as e:
                        logger.error(f"Error closing database connection for {project}/{session}: {str(e)}")
                self.db_connections[project].clear()
            self.db_connections.clear()
        
        logger.info("Closed benchmark storage system")

# Factory function
def create_scalable_benchmark_storage(config: Optional[Dict[str, Any]] = None) -> ScalableBenchmarkStorage:
    """
    Create and initialize a ScalableBenchmarkStorage.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized ScalableBenchmarkStorage
    """
    return ScalableBenchmarkStorage(config)
