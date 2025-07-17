#!/usr/bin/env python3
"""
Metrics Bus for Triangulation Engine

This module implements the metrics collection and broadcasting system for the
Triangulation Engine and related components.
"""

import os
import sys
import json
import time
import logging
import threading
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(".triangulum/metrics.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("MetricBus")

class MetricBus:
    """
    Metrics collection and broadcasting system.
    
    The MetricBus collects metrics from various components and broadcasts them
    to registered subscribers. It provides real-time metrics monitoring and
    historical data analysis capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the MetricBus.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.metrics = {}  # source -> {metric_name -> metric_value}
        self.historical_metrics = {}  # source -> {metric_name -> [(timestamp, value), ...]}
        self.subscribers = {}  # subscription_id -> (callback, filters)
        self.metrics_queue = queue.Queue()
        
        # Maximum number of historical data points per metric
        self.max_history_size = self.config.get("max_history_size", 1000)
        
        # Historical data pruning interval
        self.prune_interval = self.config.get("prune_interval", 3600)  # 1 hour
        self.last_prune_time = time.time()
        
        # Start metrics processor thread
        self._shutdown = threading.Event()
        self._processor_thread = threading.Thread(target=self._metrics_processor)
        self._processor_thread.daemon = True
        self._processor_thread.start()
        
        logger.info("MetricBus initialized")
    
    def _metrics_processor(self) -> None:
        """Process metrics from the queue."""
        while not self._shutdown.is_set():
            try:
                # Get metric from queue
                try:
                    source, metric_name, metric_value, timestamp = self.metrics_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Update current metrics
                if source not in self.metrics:
                    self.metrics[source] = {}
                
                self.metrics[source][metric_name] = metric_value
                
                # Update historical metrics
                if source not in self.historical_metrics:
                    self.historical_metrics[source] = {}
                
                if metric_name not in self.historical_metrics[source]:
                    self.historical_metrics[source][metric_name] = []
                
                # Add to historical data
                self.historical_metrics[source][metric_name].append((timestamp, metric_value))
                
                # Trim historical data if needed
                if len(self.historical_metrics[source][metric_name]) > self.max_history_size:
                    # Keep the most recent data points
                    self.historical_metrics[source][metric_name] = \
                        self.historical_metrics[source][metric_name][-self.max_history_size:]
                
                # Notify subscribers
                self._notify_subscribers(source, metric_name, metric_value, timestamp)
                
                # Mark as done
                self.metrics_queue.task_done()
                
                # Prune historical data if needed
                if time.time() - self.last_prune_time > self.prune_interval:
                    self._prune_historical_data()
                    self.last_prune_time = time.time()
            
            except Exception as e:
                logger.error(f"Error processing metrics: {e}")
    
    def _notify_subscribers(self, source: str, metric_name: str, metric_value: Any, timestamp: float) -> None:
        """
        Notify subscribers about a metric update.
        
        Args:
            source: Source of the metric
            metric_name: Name of the metric
            metric_value: Value of the metric
            timestamp: Timestamp of the metric
        """
        for sub_id, (callback, filters) in self.subscribers.items():
            try:
                # Check if the metric matches the filters
                if filters.get("sources") and source not in filters["sources"]:
                    continue
                
                if filters.get("metrics") and metric_name not in filters["metrics"]:
                    continue
                
                # Call the callback with the metric update
                callback({
                    "source": source,
                    "metric": metric_name,
                    "value": metric_value,
                    "timestamp": timestamp
                })
            except Exception as e:
                logger.error(f"Error notifying subscriber {sub_id}: {e}")
    
    def _prune_historical_data(self) -> None:
        """Prune historical data to save memory."""
        try:
            # Prune historical data based on configuration
            max_age = self.config.get("max_metric_age", 86400)  # 1 day
            now = time.time()
            
            for source in self.historical_metrics:
                for metric_name in list(self.historical_metrics[source].keys()):
                    # Remove data points older than max_age
                    self.historical_metrics[source][metric_name] = [
                        (ts, val) for ts, val in self.historical_metrics[source][metric_name]
                        if now - ts <= max_age
                    ]
            
            logger.debug("Historical data pruned")
        except Exception as e:
            logger.error(f"Error pruning historical data: {e}")
    
    def publish_metric(self, source: str, metric_name: str, metric_value: Any) -> None:
        """
        Publish a metric to the bus.
        
        Args:
            source: Source of the metric
            metric_name: Name of the metric
            metric_value: Value of the metric
        """
        try:
            # Add to metrics queue
            timestamp = time.time()
            self.metrics_queue.put((source, metric_name, metric_value, timestamp))
        except Exception as e:
            logger.error(f"Error publishing metric: {e}")
    
    def subscribe(self, callback: Callable[[Dict[str, Any]], None], 
                  sources: Set[str] = None, metrics: Set[str] = None) -> str:
        """
        Subscribe to metric updates.
        
        Args:
            callback: Callback function to call with metric updates
            sources: Set of sources to subscribe to (None for all)
            metrics: Set of metrics to subscribe to (None for all)
            
        Returns:
            str: Subscription ID
        """
        try:
            # Create subscription ID
            sub_id = f"sub-{len(self.subscribers) + 1}"
            
            # Create filters
            filters = {}
            if sources:
                filters["sources"] = sources
            if metrics:
                filters["metrics"] = metrics
            
            # Add subscriber
            self.subscribers[sub_id] = (callback, filters)
            
            logger.info(f"Added subscriber {sub_id}")
            return sub_id
        except Exception as e:
            logger.error(f"Error adding subscriber: {e}")
            return ""
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from metric updates.
        
        Args:
            subscription_id: Subscription ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if subscription_id not in self.subscribers:
                return False
            
            # Remove subscriber
            del self.subscribers[subscription_id]
            
            logger.info(f"Removed subscriber {subscription_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing subscriber: {e}")
            return False
    
    def get_current_metrics(self, source: str = None, metric_name: str = None) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Args:
            source: Source of the metrics (None for all)
            metric_name: Name of the metric (None for all)
            
        Returns:
            Dict[str, Any]: Current metrics
        """
        try:
            if source and metric_name:
                # Get specific metric
                if source in self.metrics and metric_name in self.metrics[source]:
                    return {source: {metric_name: self.metrics[source][metric_name]}}
                return {}
            
            if source:
                # Get all metrics for a source
                if source in self.metrics:
                    return {source: self.metrics[source]}
                return {}
            
            if metric_name:
                # Get a specific metric for all sources
                result = {}
                for src, metrics in self.metrics.items():
                    if metric_name in metrics:
                        if src not in result:
                            result[src] = {}
                        result[src][metric_name] = metrics[metric_name]
                return result
            
            # Get all metrics
            return self.metrics
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {}
    
    def get_historical_metrics(self, source: str, metric_name: str, 
                              start_time: float = None, end_time: float = None,
                              max_points: int = None) -> List[Tuple[float, Any]]:
        """
        Get historical metrics.
        
        Args:
            source: Source of the metrics
            metric_name: Name of the metric
            start_time: Start time (None for all)
            end_time: End time (None for all)
            max_points: Maximum number of data points to return
            
        Returns:
            List[Tuple[float, Any]]: Historical metrics as (timestamp, value) pairs
        """
        try:
            # Check if historical data exists
            if source not in self.historical_metrics or metric_name not in self.historical_metrics[source]:
                return []
            
            # Get historical data
            historical_data = self.historical_metrics[source][metric_name]
            
            # Filter by time range
            if start_time is not None or end_time is not None:
                start_time = start_time or 0
                end_time = end_time or float('inf')
                
                historical_data = [
                    (ts, val) for ts, val in historical_data
                    if start_time <= ts <= end_time
                ]
            
            # Limit number of data points
            if max_points and len(historical_data) > max_points:
                # Simple downsampling: take every nth point
                n = len(historical_data) // max_points
                historical_data = historical_data[::n]
            
            return historical_data
        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return []
    
    def clear_metrics(self, source: str = None, metric_name: str = None) -> bool:
        """
        Clear metrics.
        
        Args:
            source: Source of the metrics (None for all)
            metric_name: Name of the metric (None for all)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if source and metric_name:
                # Clear specific metric
                if source in self.metrics and metric_name in self.metrics[source]:
                    del self.metrics[source][metric_name]
                
                if source in self.historical_metrics and metric_name in self.historical_metrics[source]:
                    del self.historical_metrics[source][metric_name]
            
            elif source:
                # Clear all metrics for a source
                if source in self.metrics:
                    del self.metrics[source]
                
                if source in self.historical_metrics:
                    del self.historical_metrics[source]
            
            elif metric_name:
                # Clear a specific metric for all sources
                for src in list(self.metrics.keys()):
                    if metric_name in self.metrics[src]:
                        del self.metrics[src][metric_name]
                
                for src in list(self.historical_metrics.keys()):
                    if metric_name in self.historical_metrics[src]:
                        del self.historical_metrics[src][metric_name]
            
            else:
                # Clear all metrics
                self.metrics = {}
                self.historical_metrics = {}
            
            return True
        except Exception as e:
            logger.error(f"Error clearing metrics: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the MetricBus."""
        try:
            # Signal processor thread to stop
            self._shutdown.set()
            
            # Wait for processor thread to finish
            self._processor_thread.join(timeout=5.0)
            
            logger.info("MetricBus shutdown")
        except Exception as e:
            logger.error(f"Error shutting down MetricBus: {e}")

# Singleton instance of the MetricBus
_metric_bus = None

def get_metric_bus(config: Dict[str, Any] = None) -> MetricBus:
    """
    Get the singleton instance of the MetricBus.
    
    Args:
        config: Configuration options (used only if MetricBus is not initialized)
        
    Returns:
        MetricBus: The MetricBus instance
    """
    global _metric_bus
    
    if _metric_bus is None:
        _metric_bus = MetricBus(config)
    
    return _metric_bus

# API Functions

def publish_metric(source: str, metric_name: str, metric_value: Any) -> None:
    """
    Publish a metric to the bus.
    
    Args:
        source: Source of the metric
        metric_name: Name of the metric
        metric_value: Value of the metric
    """
    metric_bus = get_metric_bus()
    metric_bus.publish_metric(source, metric_name, metric_value)

def subscribe_to_metrics(callback: Callable[[Dict[str, Any]], None], 
                         sources: Set[str] = None, metrics: Set[str] = None) -> str:
    """
    Subscribe to metric updates.
    
    Args:
        callback: Callback function to call with metric updates
        sources: Set of sources to subscribe to (None for all)
        metrics: Set of metrics to subscribe to (None for all)
        
    Returns:
        str: Subscription ID
    """
    metric_bus = get_metric_bus()
    return metric_bus.subscribe(callback, sources, metrics)

def unsubscribe_from_metrics(subscription_id: str) -> bool:
    """
    Unsubscribe from metric updates.
    
    Args:
        subscription_id: Subscription ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    metric_bus = get_metric_bus()
    return metric_bus.unsubscribe(subscription_id)

def get_metrics(source: str = None, metric_name: str = None) -> Dict[str, Any]:
    """
    Get current metrics.
    
    Args:
        source: Source of the metrics (None for all)
        metric_name: Name of the metric (None for all)
        
    Returns:
        Dict[str, Any]: Current metrics
    """
    metric_bus = get_metric_bus()
    return metric_bus.get_current_metrics(source, metric_name)

def get_historical_metrics(source: str, metric_name: str, 
                          start_time: float = None, end_time: float = None,
                          max_points: int = None) -> List[Tuple[float, Any]]:
    """
    Get historical metrics.
    
    Args:
        source: Source of the metrics
        metric_name: Name of the metric
        start_time: Start time (None for all)
        end_time: End time (None for all)
        max_points: Maximum number of data points to return
        
    Returns:
        List[Tuple[float, Any]]: Historical metrics as (timestamp, value) pairs
    """
    metric_bus = get_metric_bus()
    return metric_bus.get_historical_metrics(source, metric_name, start_time, end_time, max_points)

def clear_metrics(source: str = None, metric_name: str = None) -> bool:
    """
    Clear metrics.
    
    Args:
        source: Source of the metrics (None for all)
        metric_name: Name of the metric (None for all)
        
    Returns:
        bool: True if successful, False otherwise
    """
    metric_bus = get_metric_bus()
    return metric_bus.clear_metrics(source, metric_name)
