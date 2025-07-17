#!/usr/bin/env python3
"""
Storage Manager Module

This module provides storage capabilities for FixWurx, including:
- Compressed storage for fixes and plans
- Rotating buffer for error logs
- Version control with rollback
- Neural pattern storage
- Cross-session knowledge persistence
"""

import os
import sys
import json
import logging
import time
import gzip
import shutil
import hashlib
import pickle
import uuid
import io
import threading
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, BinaryIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("storage_manager.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("StorageManager")

class CompressedStorage:
    """
    Compressed storage for fixes and plans.
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize compressed storage.
        
        Args:
            storage_dir: Directory for storing compressed files
        """
        self.storage_dir = os.path.abspath(storage_dir)
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Index of stored items
        self.index_file = os.path.join(self.storage_dir, "index.json")
        self.index = self._load_index()
        
        logger.info(f"Compressed storage initialized at {self.storage_dir}")
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load index from file.
        
        Returns:
            Index dictionary
        """
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading index: {e}")
        
        return {}
    
    def _save_index(self) -> None:
        """
        Save index to file.
        """
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def store(self, data: Union[str, bytes, Dict[str, Any]], item_type: str, 
              metadata: Dict[str, Any] = None) -> str:
        """
        Store data in compressed format.
        
        Args:
            data: Data to store (string, bytes, or dictionary)
            item_type: Type of item (e.g., "fix", "plan")
            metadata: Additional metadata
            
        Returns:
            Item ID
        """
        # Generate unique ID
        item_id = str(uuid.uuid4())
        
        # Convert data to bytes if necessary
        if isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Compress data
        compressed_data = gzip.compress(data_bytes)
        
        # Generate hash for integrity check
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        
        # Prepare metadata
        metadata = metadata or {}
        metadata.update({
            "item_type": item_type,
            "timestamp": time.time(),
            "size": len(data_bytes),
            "compressed_size": len(compressed_data),
            "hash": data_hash
        })
        
        # Save compressed data
        file_path = os.path.join(self.storage_dir, f"{item_id}.gz")
        try:
            with open(file_path, 'wb') as f:
                f.write(compressed_data)
            
            # Update index
            self.index[item_id] = metadata
            self._save_index()
            
            logger.info(f"Stored {item_type} item {item_id} ({len(data_bytes)} bytes, {len(compressed_data)} compressed)")
            return item_id
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return None
    
    def retrieve(self, item_id: str) -> Tuple[Optional[Union[str, bytes, Dict[str, Any]]], Dict[str, Any]]:
        """
        Retrieve data from compressed storage.
        
        Args:
            item_id: Item ID
            
        Returns:
            Tuple of (data, metadata)
        """
        # Check if item exists
        if item_id not in self.index:
            logger.warning(f"Item {item_id} not found in index")
            return None, {}
        
        # Get metadata
        metadata = self.index[item_id]
        
        # Get file path
        file_path = os.path.join(self.storage_dir, f"{item_id}.gz")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File for item {item_id} not found at {file_path}")
            return None, metadata
        
        try:
            # Read and decompress data
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
            
            data_bytes = gzip.decompress(compressed_data)
            
            # Verify hash
            data_hash = hashlib.sha256(data_bytes).hexdigest()
            if data_hash != metadata.get("hash"):
                logger.warning(f"Hash mismatch for item {item_id}")
            
            # Convert data based on item type
            item_type = metadata.get("item_type", "")
            
            if item_type in ["fix", "plan", "config"]:
                # Assume JSON for these types
                try:
                    return json.loads(data_bytes.decode('utf-8')), metadata
                except:
                    # If not valid JSON, return as string
                    return data_bytes.decode('utf-8'), metadata
            else:
                # Return as bytes for other types
                return data_bytes, metadata
        
        except Exception as e:
            logger.error(f"Error retrieving data for item {item_id}: {e}")
            return None, metadata
    
    def delete(self, item_id: str) -> bool:
        """
        Delete item from storage.
        
        Args:
            item_id: Item ID
            
        Returns:
            Whether the item was deleted
        """
        # Check if item exists
        if item_id not in self.index:
            logger.warning(f"Item {item_id} not found in index")
            return False
        
        # Get file path
        file_path = os.path.join(self.storage_dir, f"{item_id}.gz")
        
        # Delete file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error deleting file for item {item_id}: {e}")
                return False
        
        # Remove from index
        del self.index[item_id]
        self._save_index()
        
        logger.info(f"Deleted item {item_id}")
        return True
    
    def list_items(self, item_type: str = None) -> List[Dict[str, Any]]:
        """
        List items in storage.
        
        Args:
            item_type: Filter by item type (or None for all)
            
        Returns:
            List of items with metadata
        """
        items = []
        
        for item_id, metadata in self.index.items():
            if item_type is None or metadata.get("item_type") == item_type:
                items.append({
                    "id": item_id,
                    **metadata
                })
        
        return items

class RotatingBuffer:
    """
    Rotating buffer for error logs and other data.
    """
    
    def __init__(self, buffer_file: str, max_entries: int = 1000, 
                 compression: bool = True):
        """
        Initialize rotating buffer.
        
        Args:
            buffer_file: File to store buffer
            max_entries: Maximum number of entries
            compression: Whether to compress the buffer
        """
        self.buffer_file = os.path.abspath(buffer_file)
        self.max_entries = max_entries
        self.compression = compression
        self.lock = threading.Lock()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.buffer_file), exist_ok=True)
        
        # Load buffer
        self.buffer = self._load_buffer()
        
        logger.info(f"Rotating buffer initialized at {self.buffer_file} with {len(self.buffer)} entries")
    
    def _load_buffer(self) -> List[Dict[str, Any]]:
        """
        Load buffer from file.
        
        Returns:
            Buffer as a list of entries
        """
        if os.path.exists(self.buffer_file):
            try:
                if self.compression and self.buffer_file.endswith('.gz'):
                    with gzip.open(self.buffer_file, 'rt', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    with open(self.buffer_file, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.error(f"Error loading buffer: {e}")
        
        return []
    
    def _save_buffer(self) -> None:
        """
        Save buffer to file.
        """
        try:
            if self.compression and self.buffer_file.endswith('.gz'):
                with gzip.open(self.buffer_file, 'wt', encoding='utf-8') as f:
                    json.dump(self.buffer, f)
            else:
                with open(self.buffer_file, 'w') as f:
                    json.dump(self.buffer, f)
        except Exception as e:
            logger.error(f"Error saving buffer: {e}")
    
    def add_entry(self, entry: Dict[str, Any]) -> None:
        """
        Add entry to buffer.
        
        Args:
            entry: Entry to add
        """
        with self.lock:
            # Add timestamp if not present
            if "timestamp" not in entry:
                entry["timestamp"] = time.time()
            
            # Add entry to buffer
            self.buffer.append(entry)
            
            # Trim buffer if too large
            if len(self.buffer) > self.max_entries:
                self.buffer = self.buffer[-self.max_entries:]
            
            # Save buffer
            self._save_buffer()
    
    def get_entries(self, count: int = None, filter_func: callable = None) -> List[Dict[str, Any]]:
        """
        Get entries from buffer.
        
        Args:
            count: Maximum number of entries to return (or None for all)
            filter_func: Function to filter entries
            
        Returns:
            List of entries
        """
        with self.lock:
            # Apply filter if provided
            if filter_func:
                entries = [e for e in self.buffer if filter_func(e)]
            else:
                entries = self.buffer.copy()
            
            # Sort by timestamp (newest first)
            entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
            
            # Limit to count if provided
            if count is not None:
                entries = entries[:count]
            
            return entries
    
    def clear(self) -> None:
        """
        Clear buffer.
        """
        with self.lock:
            self.buffer = []
            self._save_buffer()
            logger.info("Buffer cleared")

class VersionControl:
    """
    Version control with rollback capability.
    """
    
    def __init__(self, repo_dir: str):
        """
        Initialize version control.
        
        Args:
            repo_dir: Directory for storing version history
        """
        self.repo_dir = os.path.abspath(repo_dir)
        
        # Create repo directory if it doesn't exist
        os.makedirs(self.repo_dir, exist_ok=True)
        
        # Index of versions
        self.index_file = os.path.join(self.repo_dir, "version_index.json")
        self.index = self._load_index()
        
        logger.info(f"Version control initialized at {self.repo_dir}")
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load index from file.
        
        Returns:
            Index dictionary
        """
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading version index: {e}")
        
        return {}
    
    def _save_index(self) -> None:
        """
        Save index to file.
        """
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving version index: {e}")
    
    def _get_file_versions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get versions for a file.
        
        Args:
            file_path: File path
            
        Returns:
            List of versions
        """
        file_key = os.path.normpath(file_path)
        
        if file_key not in self.index:
            return []
        
        # Get versions
        versions = self.index[file_key]
        
        # Sort by version number
        return sorted(versions.values(), key=lambda v: v["version"])
    
    def commit(self, file_path: str, data: Union[str, bytes], 
               message: str = "", tags: List[str] = None) -> str:
        """
        Commit a new version.
        
        Args:
            file_path: File path
            data: File content
            message: Commit message
            tags: Tags for this version
            
        Returns:
            Version ID
        """
        file_key = os.path.normpath(file_path)
        
        # Initialize file entry if not present
        if file_key not in self.index:
            self.index[file_key] = {}
        
        # Get latest version number
        versions = self._get_file_versions(file_path)
        version_num = len(versions) + 1
        
        # Generate version ID
        version_id = str(uuid.uuid4())
        
        # Convert data to bytes if necessary
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Generate hash
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        
        # Save version data
        version_dir = os.path.join(self.repo_dir, file_key.replace('/', '_').replace('\\', '_'))
        os.makedirs(version_dir, exist_ok=True)
        
        version_file = os.path.join(version_dir, f"{version_id}.gz")
        
        try:
            # Compress and save data
            with gzip.open(version_file, 'wb') as f:
                f.write(data_bytes)
            
            # Create version metadata
            version_info = {
                "id": version_id,
                "version": version_num,
                "timestamp": time.time(),
                "message": message,
                "tags": tags or [],
                "hash": data_hash,
                "file": file_key,
                "size": len(data_bytes)
            }
            
            # Update index
            self.index[file_key][version_id] = version_info
            self._save_index()
            
            logger.info(f"Committed version {version_num} for {file_key} ({version_id})")
            return version_id
        
        except Exception as e:
            logger.error(f"Error committing version for {file_key}: {e}")
            if os.path.exists(version_file):
                os.remove(version_file)
            return None
    
    def get_version(self, file_path: str, version_id: str = None, 
                   version_num: int = None) -> Tuple[Optional[Union[str, bytes]], Dict[str, Any]]:
        """
        Get a specific version.
        
        Args:
            file_path: File path
            version_id: Version ID (or None for latest)
            version_num: Version number (ignored if version_id is provided)
            
        Returns:
            Tuple of (data, version_info)
        """
        file_key = os.path.normpath(file_path)
        
        # Check if file exists in index
        if file_key not in self.index:
            logger.warning(f"File {file_key} not found in version index")
            return None, {}
        
        # Get versions
        versions = self.index[file_key]
        
        # Get target version
        version_info = None
        
        if version_id:
            # Get by ID
            if version_id in versions:
                version_info = versions[version_id]
            else:
                logger.warning(f"Version {version_id} not found for {file_key}")
                return None, {}
        elif version_num:
            # Get by number
            for v in versions.values():
                if v["version"] == version_num:
                    version_info = v
                    break
            
            if not version_info:
                logger.warning(f"Version {version_num} not found for {file_key}")
                return None, {}
        else:
            # Get latest
            all_versions = self._get_file_versions(file_path)
            if all_versions:
                version_info = all_versions[-1]
            else:
                logger.warning(f"No versions found for {file_key}")
                return None, {}
        
        # Get version data
        version_id = version_info["id"]
        version_dir = os.path.join(self.repo_dir, file_key.replace('/', '_').replace('\\', '_'))
        version_file = os.path.join(version_dir, f"{version_id}.gz")
        
        if not os.path.exists(version_file):
            logger.warning(f"Version file not found: {version_file}")
            return None, version_info
        
        try:
            # Read and decompress data
            with gzip.open(version_file, 'rb') as f:
                data = f.read()
            
            # Verify hash
            data_hash = hashlib.sha256(data).hexdigest()
            if data_hash != version_info.get("hash"):
                logger.warning(f"Hash mismatch for version {version_id}")
            
            # Try to decode as text
            try:
                return data.decode('utf-8'), version_info
            except UnicodeDecodeError:
                # Return as bytes if not valid UTF-8
                return data, version_info
        
        except Exception as e:
            logger.error(f"Error reading version {version_id} for {file_key}: {e}")
            return None, version_info
    
    def list_versions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        List versions for a file.
        
        Args:
            file_path: File path
            
        Returns:
            List of version info
        """
        return self._get_file_versions(file_path)
    
    def rollback(self, file_path: str, version_id: str = None, 
                version_num: int = None) -> Tuple[bool, Optional[Union[str, bytes]]]:
        """
        Rollback to a specific version.
        
        Args:
            file_path: File path
            version_id: Version ID (or None for latest)
            version_num: Version number (ignored if version_id is provided)
            
        Returns:
            Tuple of (success, data)
        """
        # Get version
        data, version_info = self.get_version(file_path, version_id, version_num)
        
        if data is None:
            return False, None
        
        logger.info(f"Rolled back {file_path} to version {version_info.get('version')} ({version_info.get('id')})")
        return True, data

class NeuralPatternStorage:
    """
    Storage for neural patterns and weights.
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize neural pattern storage.
        
        Args:
            storage_dir: Directory for storing patterns
        """
        self.storage_dir = os.path.abspath(storage_dir)
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Index of patterns
        self.index_file = os.path.join(self.storage_dir, "pattern_index.json")
        self.index = self._load_index()
        
        logger.info(f"Neural pattern storage initialized at {self.storage_dir}")
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load index from file.
        
        Returns:
            Index dictionary
        """
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading pattern index: {e}")
        
        return {}
    
    def _save_index(self) -> None:
        """
        Save index to file.
        """
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving pattern index: {e}")
    
    def store_pattern(self, pattern_type: str, pattern_data: Dict[str, Any], 
                     metadata: Dict[str, Any] = None) -> str:
        """
        Store a neural pattern.
        
        Args:
            pattern_type: Type of pattern (e.g., "bug", "solution")
            pattern_data: Pattern data
            metadata: Additional metadata
            
        Returns:
            Pattern ID
        """
        # Generate unique ID
        pattern_id = str(uuid.uuid4())
        
        # Prepare metadata
        metadata = metadata or {}
        metadata.update({
            "pattern_type": pattern_type,
            "timestamp": time.time()
        })
        
        # Save pattern
        pattern_file = os.path.join(self.storage_dir, f"{pattern_id}.pkl.gz")
        try:
            # Serialize and compress pattern data
            with gzip.open(pattern_file, 'wb') as f:
                pickle.dump(pattern_data, f)
            
            # Update index
            self.index[pattern_id] = metadata
            self._save_index()
            
            logger.info(f"Stored {pattern_type} pattern {pattern_id}")
            return pattern_id
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
            if os.path.exists(pattern_file):
                os.remove(pattern_file)
            return None
    
    def retrieve_pattern(self, pattern_id: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve a neural pattern.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Tuple of (pattern_data, metadata)
        """
        # Check if pattern exists
        if pattern_id not in self.index:
            logger.warning(f"Pattern {pattern_id} not found in index")
            return None, {}
        
        # Get metadata
        metadata = self.index[pattern_id]
        
        # Get file path
        pattern_file = os.path.join(self.storage_dir, f"{pattern_id}.pkl.gz")
        
        # Check if file exists
        if not os.path.exists(pattern_file):
            logger.warning(f"File for pattern {pattern_id} not found at {pattern_file}")
            return None, metadata
        
        try:
            # Read and decompress pattern data
            with gzip.open(pattern_file, 'rb') as f:
                pattern_data = pickle.load(f)
            
            return pattern_data, metadata
        
        except Exception as e:
            logger.error(f"Error retrieving pattern {pattern_id}: {e}")
            return None, metadata
    
    def update_pattern(self, pattern_id: str, pattern_data: Dict[str, Any], 
                      metadata_updates: Dict[str, Any] = None) -> bool:
        """
        Update a neural pattern.
        
        Args:
            pattern_id: Pattern ID
            pattern_data: Updated pattern data
            metadata_updates: Updates to metadata
            
        Returns:
            Whether the pattern was updated
        """
        # Check if pattern exists
        if pattern_id not in self.index:
            logger.warning(f"Pattern {pattern_id} not found in index")
            return False
        
        # Get metadata
        metadata = self.index[pattern_id]
        
        # Update metadata
        if metadata_updates:
            metadata.update(metadata_updates)
        
        # Set update timestamp
        metadata["updated"] = time.time()
        
        # Get file path
        pattern_file = os.path.join(self.storage_dir, f"{pattern_id}.pkl.gz")
        
        try:
            # Serialize and compress pattern data
            with gzip.open(pattern_file, 'wb') as f:
                pickle.dump(pattern_data, f)
            
            # Update index
            self.index[pattern_id] = metadata
            self._save_index()
            
            logger.info(f"Updated pattern {pattern_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error updating pattern {pattern_id}: {e}")
            return False
    
    def list_patterns(self, pattern_type: str = None) -> List[Dict[str, Any]]:
        """
        List patterns.
        
        Args:
            pattern_type: Filter by pattern type (or None for all)
            
        Returns:
            List of patterns with metadata
        """
        patterns = []
        
        for pattern_id, metadata in self.index.items():
            if pattern_type is None or metadata.get("pattern_type") == pattern_type:
                patterns.append({
                    "id": pattern_id,
                    **metadata
                })
        
        return patterns
    
    def delete_pattern(self, pattern_id: str) -> bool:
        """
        Delete a pattern.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Whether the pattern was deleted
        """
        # Check if pattern exists
        if pattern_id not in self.index:
            logger.warning(f"Pattern {pattern_id} not found in index")
            return False
        
        # Get file path
        pattern_file = os.path.join(self.storage_dir, f"{pattern_id}.pkl.gz")
        
        # Delete file
        if os.path.exists(pattern_file):
            try:
                os.remove(pattern_file)
            except Exception as e:
                logger.error(f"Error deleting file for pattern {pattern_id}: {e}")
                return False
        
        # Remove from index
        del self.index[pattern_id]
        self._save_index()
        
        logger.info(f"Deleted pattern {pattern_id}")
        return True

class PersistenceManager:
    """
    Cross-session knowledge persistence.
    """
    
    def __init__(self, db_file: str):
        """
        Initialize persistence manager.
        
        Args:
            db_file: Database file
        """
        self.db_file = os.path.abspath(db_file)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
        
        # Load database
        self.db = self._load_db()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Persistence manager initialized at {self.db_file}")
    
    def _load_db(self) -> Dict[str, Dict[str, Any]]:
        """
        Load database from file.
        
        Returns:
            Database dictionary
        """
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading persistence database: {e}")
        
        return {
            "metadata": {
                "created": time.time(),
                "version": "1.0"
            },
            "collections": {}
        }
    
    def _save_db(self) -> None:
        """
        Save database to file.
        """
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.db, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving persistence database: {e}")
    
    def get_collection(self, collection: str) -> Dict[str, Any]:
        """
        Get a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Collection data
        """
        with self.lock:
            collections = self.db.get("collections", {})
            if collection not in collections:
                collections[collection] = {}
                self.db["collections"] = collections
                self._save_db()
            
            return collections[collection].copy()
    
    def store_item(self, collection: str, key: str, value: Any) -> bool:
        """
        Store an item in a collection.
        
        Args:
            collection: Collection name
            key: Item key
            value: Item value
            
        Returns:
            Whether the item was stored
        """
        with self.lock:
            try:
                # Get or create collection
                collections = self.db.get("collections", {})
                if collection not in collections:
                    collections[collection] = {}
                
                # Store item
                collections[collection][key] = value
                
                # Update database
                self.db["collections"] = collections
                self.db["metadata"]["updated"] = time.time()
                self._save_db()
                
                logger.info(f"Stored item {key} in collection {collection}")
                return True
            except Exception as e:
                logger.error(f"Error storing item {key} in collection {collection}: {e}")
                return False
    
    def retrieve_item(self, collection: str, key: str, default: Any = None) -> Any:
        """
        Retrieve an item from a collection.
        
        Args:
            collection: Collection name
            key: Item key
            default: Default value if item not found
            
        Returns:
            Item value, or default if not found
        """
        with self.lock:
            try:
                # Get collection
                collections = self.db.get("collections", {})
                if collection not in collections:
                    return default
                
                # Get item
                return collections[collection].get(key, default)
            except Exception as e:
                logger.error(f"Error retrieving item {key} from collection {collection}: {e}")
                return default
    
    def delete_item(self, collection: str, key: str) -> bool:
        """
        Delete an item from a collection.
        
        Args:
            collection: Collection name
            key: Item key
            
        Returns:
            Whether the item was deleted
        """
        with self.lock:
            try:
                # Get collection
                collections = self.db.get("collections", {})
                if collection not in collections:
                    return False
                
                # Delete item
                if key in collections[collection]:
                    del collections[collection][key]
                    
                    # Update database
                    self.db["collections"] = collections
                    self.db["metadata"]["updated"] = time.time()
                    self._save_db()
                    
                    logger.info(f"Deleted item {key} from collection {collection}")
                    return True
                else:
                    return False
            except Exception as e:
                logger.error(f"Error deleting item {key} from collection {collection}: {e}")
                return False
    
    def clear_collection(self, collection: str) -> bool:
        """
        Clear a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Whether the collection was cleared
        """
        with self.lock:
            try:
                # Get collection
                collections = self.db.get("collections", {})
                if collection not in collections:
                    return False
                
                # Clear collection
                collections[collection] = {}
                
                # Update database
                self.db["collections"] = collections
                self.db["metadata"]["updated"] = time.time()
                self._save_db()
                
                logger.info(f"Cleared collection {collection}")
                return True
            except Exception as e:
                logger.error(f"Error clearing collection {collection}: {e}")
                return False
    
    def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Returns:
            List of collection names
        """
        with self.lock:
            return list(self.db.get("collections", {}).keys())

class StorageManager:
    """
    Main storage manager class that combines all storage components.
    """
    
    def __init__(self, base_dir: str = None, config: Dict[str, Any] = None):
        """
        Initialize storage manager.
        
        Args:
            base_dir: Base directory for all storage components
            config: Configuration options
        """
        self.config = config or {}
        
        # Set base directory
        self.base_dir = os.path.abspath(base_dir or "storage")
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize components
        self.compressed_storage = CompressedStorage(
            os.path.join(self.base_dir, "compressed")
        )
        
        self.error_log_buffer = RotatingBuffer(
            os.path.join(self.base_dir, "logs", "error_log.json.gz"),
            max_entries=self.config.get("error_log_buffer_size", 10000),
            compression=True
        )
        
        self.version_control = VersionControl(
            os.path.join(self.base_dir, "versions")
        )
        
        self.neural_storage = NeuralPatternStorage(
            os.path.join(self.base_dir, "neural_patterns")
        )
        
        self.persistence = PersistenceManager(
            os.path.join(self.base_dir, "persistence", "db.json")
        )
        
        logger.info(f"Storage manager initialized at {self.base_dir}")
    
    def store_fix(self, fix_data: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """
        Store a fix.
        
        Args:
            fix_data: Fix data
            metadata: Additional metadata
            
        Returns:
            Fix ID
        """
        return self.compressed_storage.store(fix_data, "fix", metadata)
    
    def store_plan(self, plan_data: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """
        Store a plan.
        
        Args:
            plan_data: Plan data
            metadata: Additional metadata
            
        Returns:
            Plan ID
        """
        return self.compressed_storage.store(plan_data, "plan", metadata)
    
    def log_error(self, error_data: Dict[str, Any]) -> None:
        """
        Log an error.
        
        Args:
            error_data: Error data
        """
        self.error_log_buffer.add_entry(error_data)
    
    def get_recent_errors(self, count: int = 10, filter_func: callable = None) -> List[Dict[str, Any]]:
        """
        Get recent errors.
        
        Args:
            count: Maximum number of errors to return
            filter_func: Function to filter errors
            
        Returns:
            List of errors
        """
        return self.error_log_buffer.get_entries(count, filter_func)
    
    def commit_file_version(self, file_path: str, content: Union[str, bytes], 
                           message: str = "", tags: List[str] = None) -> str:
        """
        Commit a new version of a file.
        
        Args:
            file_path: File path
            content: File content
            message: Commit message
            tags: Tags for this version
            
        Returns:
            Version ID
        """
        return self.version_control.commit(file_path, content, message, tags)
    
    def rollback_file(self, file_path: str, version_id: str = None, 
                     version_num: int = None) -> Tuple[bool, Optional[Union[str, bytes]]]:
        """
        Rollback a file to a specific version.
        
        Args:
            file_path: File path
            version_id: Version ID (or None for latest)
            version_num: Version number (ignored if version_id is provided)
            
        Returns:
            Tuple of (success, content)
        """
        return self.version_control.rollback(file_path, version_id, version_num)
    
    def store_neural_pattern(self, pattern_type: str, pattern_data: Dict[str, Any], 
                            metadata: Dict[str, Any] = None) -> str:
        """
        Store a neural pattern.
        
        Args:
            pattern_type: Pattern type
            pattern_data: Pattern data
            metadata: Additional metadata
            
        Returns:
            Pattern ID
        """
        return self.neural_storage.store_pattern(pattern_type, pattern_data, metadata)
    
    def persist_data(self, collection: str, key: str, value: Any) -> bool:
        """
        Persist data across sessions.
        
        Args:
            collection: Collection name
            key: Data key
            value: Data value
            
        Returns:
            Whether the data was persisted
        """
        return self.persistence.store_item(collection, key, value)
    
    def retrieve_persisted_data(self, collection: str, key: str, default: Any = None) -> Any:
        """
        Retrieve persisted data.
        
        Args:
            collection: Collection name
            key: Data key
            default: Default value if key not found
            
        Returns:
            Persisted data value
        """
        return self.persistence.retrieve_item(collection, key, default)


# API Functions

def create_storage_manager(base_dir: str = None, config: Dict[str, Any] = None) -> StorageManager:
    """
    Create a storage manager.
    
    Args:
        base_dir: Base directory for all storage components
        config: Configuration options
        
    Returns:
        Storage manager
    """
    return StorageManager(base_dir, config)

def store_fix(fix_data: Dict[str, Any], metadata: Dict[str, Any] = None, 
             storage_dir: str = "storage/compressed") -> str:
    """
    Store a fix.
    
    Args:
        fix_data: Fix data
        metadata: Additional metadata
        storage_dir: Storage directory
        
    Returns:
        Fix ID
    """
    storage = CompressedStorage(storage_dir)
    return storage.store(fix_data, "fix", metadata)

def store_plan(plan_data: Dict[str, Any], metadata: Dict[str, Any] = None, 
              storage_dir: str = "storage/compressed") -> str:
    """
    Store a plan.
    
    Args:
        plan_data: Plan data
        metadata: Additional metadata
        storage_dir: Storage directory
        
    Returns:
        Plan ID
    """
    storage = CompressedStorage(storage_dir)
    return storage.store(plan_data, "plan", metadata)

def commit_file_version(file_path: str, content: Union[str, bytes], 
                       message: str = "", tags: List[str] = None,
                       repo_dir: str = "storage/versions") -> str:
    """
    Commit a new version of a file.
    
    Args:
        file_path: File path
        content: File content
        message: Commit message
        tags: Tags for this version
        repo_dir: Repository directory
        
    Returns:
        Version ID
    """
    vc = VersionControl(repo_dir)
    return vc.commit(file_path, content, message, tags)

def rollback_file(file_path: str, version_id: str = None, version_num: int = None,
                 repo_dir: str = "storage/versions") -> Tuple[bool, Optional[Union[str, bytes]]]:
    """
    Rollback a file to a specific version.
    
    Args:
        file_path: File path
        version_id: Version ID (or None for latest)
        version_num: Version number (ignored if version_id is provided)
        repo_dir: Repository directory
        
    Returns:
        Tuple of (success, content)
    """
    vc = VersionControl(repo_dir)
    return vc.rollback(file_path, version_id, version_num)

def log_error(error_data: Dict[str, Any], buffer_file: str = "storage/logs/error_log.json.gz",
             max_entries: int = 10000) -> None:
    """
    Log an error.
    
    Args:
        error_data: Error data
        buffer_file: Buffer file
        max_entries: Maximum number of entries
    """
    buffer = RotatingBuffer(buffer_file, max_entries, True)
    buffer.add_entry(error_data)

def get_recent_errors(count: int = 10, filter_func: callable = None,
                     buffer_file: str = "storage/logs/error_log.json.gz") -> List[Dict[str, Any]]:
    """
    Get recent errors.
    
    Args:
        count: Maximum number of errors to return
        filter_func: Function to filter errors
        buffer_file: Buffer file
        
    Returns:
        List of errors
    """
    buffer = RotatingBuffer(buffer_file, 10000, True)
    return buffer.get_entries(count, filter_func)

def store_neural_pattern(pattern_type: str, pattern_data: Dict[str, Any], 
                        metadata: Dict[str, Any] = None,
                        storage_dir: str = "storage/neural_patterns") -> str:
    """
    Store a neural pattern.
    
    Args:
        pattern_type: Pattern type
        pattern_data: Pattern data
        metadata: Additional metadata
        storage_dir: Storage directory
        
    Returns:
        Pattern ID
    """
    storage = NeuralPatternStorage(storage_dir)
    return storage.store_pattern(pattern_type, pattern_data, metadata)

def persist_data(collection: str, key: str, value: Any,
               db_file: str = "storage/persistence/db.json") -> bool:
    """
    Persist data across sessions.
    
    Args:
        collection: Collection name
        key: Data key
        value: Data value
        db_file: Database file
        
    Returns:
        Whether the data was persisted
    """
    manager = PersistenceManager(db_file)
    return manager.store_item(collection, key, value)

def retrieve_persisted_data(collection: str, key: str, default: Any = None,
                           db_file: str = "storage/persistence/db.json") -> Any:
    """
    Retrieve persisted data.
    
    Args:
        collection: Collection name
        key: Data key
        default: Default value if key not found
        db_file: Database file
        
    Returns:
        Persisted data value
    """
    manager = PersistenceManager(db_file)
    return manager.retrieve_item(collection, key, default)


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Storage Manager")
    parser.add_argument("--base-dir", help="Base directory", default="storage")
    parser.add_argument("--store-fix", help="Store a fix (JSON file)")
    parser.add_argument("--store-plan", help="Store a plan (JSON file)")
    parser.add_argument("--commit", help="Commit a file")
    parser.add_argument("--message", help="Commit message", default="")
    parser.add_argument("--rollback", help="Rollback a file")
    parser.add_argument("--version", help="Version ID or number")
    parser.add_argument("--log-error", help="Log an error (JSON file)")
    parser.add_argument("--get-errors", action="store_true", help="Get recent errors")
    parser.add_argument("--count", type=int, default=10, help="Number of items to return")
    
    args = parser.parse_args()
    
    # Create storage manager
    manager = create_storage_manager(args.base_dir)
    
    # Store a fix
    if args.store_fix:
        try:
            with open(args.store_fix, 'r') as f:
                fix_data = json.load(f)
            
            fix_id = manager.store_fix(fix_data)
            print(f"Fix stored with ID: {fix_id}")
        except Exception as e:
            print(f"Error storing fix: {e}")
    
    # Store a plan
    elif args.store_plan:
        try:
            with open(args.store_plan, 'r') as f:
                plan_data = json.load(f)
            
            plan_id = manager.store_plan(plan_data)
            print(f"Plan stored with ID: {plan_id}")
        except Exception as e:
            print(f"Error storing plan: {e}")
    
    # Commit a file
    elif args.commit:
        try:
            with open(args.commit, 'r') as f:
                content = f.read()
            
            version_id = manager.commit_file_version(args.commit, content, args.message)
            print(f"File committed with version ID: {version_id}")
        except Exception as e:
            print(f"Error committing file: {e}")
    
    # Rollback a file
    elif args.rollback:
        try:
            version = None
            if args.version:
                try:
                    version = int(args.version)
                except ValueError:
                    version = args.version
            
            success, content = manager.rollback_file(args.rollback, version_id=version if isinstance(version, str) else None,
                                                  version_num=version if isinstance(version, int) else None)
            
            if success:
                print(f"File rolled back successfully")
                if isinstance(content, str):
                    print("\nContent:")
                    print(content[:1000] + "..." if len(content) > 1000 else content)
                else:
                    print(f"Binary content ({len(content)} bytes)")
            else:
                print("Rollback failed")
        except Exception as e:
            print(f"Error rolling back file: {e}")
    
    # Log an error
    elif args.log_error:
        try:
            with open(args.log_error, 'r') as f:
                error_data = json.load(f)
            
            manager.log_error(error_data)
            print("Error logged")
        except Exception as e:
            print(f"Error logging error: {e}")
    
    # Get recent errors
    elif args.get_errors:
        try:
            errors = manager.get_recent_errors(args.count)
            print(f"Recent errors ({len(errors)}):")
            for i, error in enumerate(errors):
                print(f"\n{i+1}. {error.get('message', 'Unknown error')}")
                print(f"   Time: {time.ctime(error.get('timestamp', 0))}")
                print(f"   Type: {error.get('type', 'Unknown')}")
                if 'details' in error:
                    print(f"   Details: {error['details']}")
        except Exception as e:
            print(f"Error getting errors: {e}")
    
    # Print help
    else:
        parser.print_help()
