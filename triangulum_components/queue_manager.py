#!/usr/bin/env python3
"""
Queue Manager Component for Triangulum Integration

This module provides the QueueManager class for managing priority queues.
"""

import logging
import threading
import queue
from typing import Dict, Any, Optional, List

# Configure logging if not already configured
logger = logging.getLogger("TriangulumIntegration")

class QueueManager:
    """
    Manages task queues for Triangulum integration.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize queue manager.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.queues = {}
        self.default_queue = "default"
        self.lock = threading.Lock()
        
        # Create default queue
        self.create_queue(self.default_queue)
        
        logger.info("Queue manager initialized")
    
    def create_queue(self, queue_name: str) -> bool:
        """
        Create a new queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Whether the queue was created
        """
        with self.lock:
            if queue_name in self.queues:
                logger.warning(f"Queue {queue_name} already exists")
                return False
            
            self.queues[queue_name] = queue.PriorityQueue()
            logger.info(f"Created queue: {queue_name}")
            return True
    
    def delete_queue(self, queue_name: str) -> bool:
        """
        Delete a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Whether the queue was deleted
        """
        with self.lock:
            if queue_name == self.default_queue:
                logger.warning("Cannot delete default queue")
                return False
            
            if queue_name not in self.queues:
                logger.warning(f"Queue {queue_name} does not exist")
                return False
            
            del self.queues[queue_name]
            logger.info(f"Deleted queue: {queue_name}")
            return True
    
    def get_queue_names(self) -> List[str]:
        """
        Get names of all queues.
        
        Returns:
            List of queue names
        """
        with self.lock:
            return list(self.queues.keys())
    
    def get_queue_size(self, queue_name: str = None) -> int:
        """
        Get size of a queue.
        
        Args:
            queue_name: Name of the queue (or None for default)
            
        Returns:
            Size of the queue
        """
        queue_name = queue_name or self.default_queue
        
        with self.lock:
            if queue_name not in self.queues:
                logger.warning(f"Queue {queue_name} does not exist")
                return 0
            
            return self.queues[queue_name].qsize()
    
    def enqueue(self, item: Any, queue_name: str = None, priority: int = 0) -> bool:
        """
        Add an item to a queue.
        
        Args:
            item: Item to add
            queue_name: Name of the queue (or None for default)
            priority: Priority of the item (lower is higher priority)
            
        Returns:
            Whether the item was added
        """
        queue_name = queue_name or self.default_queue
        
        with self.lock:
            if queue_name not in self.queues:
                logger.warning(f"Queue {queue_name} does not exist")
                return False
            
            self.queues[queue_name].put((priority, item))
            logger.debug(f"Enqueued item in {queue_name} queue with priority {priority}")
            return True
    
    def dequeue(self, queue_name: str = None) -> Optional[Any]:
        """
        Remove and return an item from a queue.
        
        Args:
            queue_name: Name of the queue (or None for default)
            
        Returns:
            Item from the queue, or None if empty
        """
        queue_name = queue_name or self.default_queue
        
        with self.lock:
            if queue_name not in self.queues:
                logger.warning(f"Queue {queue_name} does not exist")
                return None
            
            try:
                priority, item = self.queues[queue_name].get(block=False)
                logger.debug(f"Dequeued item from {queue_name} queue with priority {priority}")
                return item
            except queue.Empty:
                logger.debug(f"Queue {queue_name} is empty")
                return None
    
    def peek(self, queue_name: str = None) -> Optional[Any]:
        """
        Return an item from a queue without removing it.
        
        Args:
            queue_name: Name of the queue (or None for default)
            
        Returns:
            Item from the queue, or None if empty
        """
        queue_name = queue_name or self.default_queue
        
        with self.lock:
            if queue_name not in self.queues:
                logger.warning(f"Queue {queue_name} does not exist")
                return None
            
            q = self.queues[queue_name]
            
            # We can't peek directly, so we'll get the item and put it back
            try:
                priority, item = q.get(block=False)
                q.put((priority, item))
                logger.debug(f"Peeked item from {queue_name} queue with priority {priority}")
                return item
            except queue.Empty:
                logger.debug(f"Queue {queue_name} is empty")
                return None
    
    def clear_queue(self, queue_name: str = None) -> bool:
        """
        Clear a queue.
        
        Args:
            queue_name: Name of the queue (or None for default)
            
        Returns:
            Whether the queue was cleared
        """
        queue_name = queue_name or self.default_queue
        
        with self.lock:
            if queue_name not in self.queues:
                logger.warning(f"Queue {queue_name} does not exist")
                return False
            
            # Create a new queue to replace the old one
            self.queues[queue_name] = queue.PriorityQueue()
            logger.info(f"Cleared queue: {queue_name}")
            return True
