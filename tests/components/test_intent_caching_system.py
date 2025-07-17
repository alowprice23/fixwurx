import pytest
import time
from components.intent_caching_system import (
    LRUCache, 
    PredictiveClassifier, 
    PerformanceTracker, 
    IntentOptimizationSystem,
    create_context_hash
)

def test_lru_cache():
    """Test the LRU cache functionality."""
    # Create a cache with capacity of 3
    cache = LRUCache(capacity=3)
    
    # Test basic put and get
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None  # Key doesn't exist yet
    
    # Test capacity limit
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    cache.put("key4", "value4")  # This should evict key1
    
    assert cache.get("key1") is None  # Should be evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"
    
    # Test LRU behavior
    # Access key2, making key3 the least recently used
    cache.get("key2")
    cache.put("key5", "value5")  # This should evict key3
    
    assert cache.get("key3") is None  # Should be evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key4") == "value4"
    assert cache.get("key5") == "value5"
    
    # Test stats
    stats = cache.get_stats()
    assert stats["size"] == 3
    assert stats["capacity"] == 3
    assert stats["hits"] == 8  # We've had 8 successful gets
    assert stats["misses"] == 3  # We've had 3 misses (key2 initially and key1/key3 after eviction)
    
    # Test clear
    cache.clear()
    assert cache.get("key2") is None
    assert cache.get("key4") is None
    assert cache.get("key5") is None
    
    stats = cache.get_stats()
    assert stats["size"] == 0

def test_predictive_classifier():
    """Test the predictive intent classification."""
    classifier = PredictiveClassifier(history_size=5)
    
    # No predictions without history
    assert classifier.predict_next_intent() == []
    
    # Record some intents
    classifier.record_intent("file_access", {})
    classifier.record_intent("file_modification", {})
    classifier.record_intent("command_execution", {})
    classifier.record_intent("file_access", {})
    
    # Test prediction based on sequence
    predictions = classifier.predict_next_intent("file_access")
    assert len(predictions) == 1
    assert predictions[0][0] == "file_modification"  # After file_access comes file_modification in our history
    
    # Record more to create a different pattern
    classifier.record_intent("command_execution", {})
    classifier.record_intent("file_access", {})
    classifier.record_intent("command_execution", {})
    
    # Now the prediction should change
    predictions = classifier.predict_next_intent("file_access")
    assert len(predictions) > 0
    # The highest probability should be command_execution now
    assert predictions[0][0] == "command_execution"
    
    # Test stats
    stats = classifier.get_stats()
    assert stats["unique_intents"] == 3
    assert stats["total_intents_processed"] == 7
    assert stats["history_length"] == 5  # We limited to 5

def test_performance_tracker():
    """Test the performance tracking functionality."""
    tracker = PerformanceTracker(window_size=5)
    
    # Test classification time tracking
    tracker.record_classification_time(10.5)
    tracker.record_classification_time(15.2)
    tracker.record_classification_time(12.8)
    
    stats = tracker.get_classification_stats()
    assert abs(stats["avg"] - 12.83) < 0.1  # Average should be close to 12.83
    assert stats["min"] == 10.5
    assert stats["max"] == 15.2
    assert stats["count"] == 3
    
    # Test execution time tracking
    tracker.record_execution_time("file_access", 50.0)
    tracker.record_execution_time("file_access", 45.0)
    tracker.record_execution_time("command_execution", 100.0)
    tracker.record_execution_time("command_execution", 120.0)
    
    # Test stats for a specific intent type
    stats = tracker.get_execution_stats("file_access")
    assert abs(stats["avg"] - 47.5) < 0.1
    assert stats["min"] == 45.0
    assert stats["max"] == 50.0
    assert stats["count"] == 2
    
    # Test stats for all intent types
    all_stats = tracker.get_execution_stats()
    assert "file_access" in all_stats
    assert "command_execution" in all_stats
    assert abs(all_stats["command_execution"]["avg"] - 110.0) < 0.1
    
    # Test slow operation identification
    slow_ops = tracker.identify_slow_operations(threshold_ms=90.0)
    assert "command_execution" in slow_ops
    assert "file_access" not in slow_ops
    
    # Test window size limit
    for i in range(10):
        tracker.record_classification_time(i)
    
    stats = tracker.get_classification_stats()
    assert stats["count"] == 5  # Should be limited to window_size

def test_intent_optimization_system():
    """Test the integrated intent optimization system."""
    system = IntentOptimizationSystem(cache_capacity=10, history_size=20, window_size=10)
    
    # Test caching
    context_hash = create_context_hash({"history": ["previous message"]})
    
    # Initially, nothing in cache
    assert system.get_cached_intent("show me file.txt", context_hash) is None
    
    # Cache a result
    result = {"type": "file_access", "parameters": {"path": "file.txt"}}
    system.cache_intent_result("show me file.txt", context_hash, result)
    
    # Now we should get it from cache
    cached = system.get_cached_intent("show me file.txt", context_hash)
    assert cached == result
    
    # Test performance tracking
    system.track_classification_performance(15.5)
    system.track_execution_performance("file_access", 75.0)
    
    report = system.get_performance_report()
    assert "cache" in report
    assert "predictor" in report
    assert "classification" in report
    assert "execution" in report
    assert "slow_operations" in report
    assert "queue_size" in report
    
    # Test prediction after caching some intents
    system.cache_intent_result("edit file.txt", create_context_hash({}), {"type": "file_modification"})
    system.cache_intent_result("run ls", create_context_hash({}), {"type": "command_execution"})
    
    predictions = system.get_intent_predictions("file_access")
    assert len(predictions) > 0
    
    # Test priority queue
    operation1 = {"type": "operation1"}
    operation2 = {"type": "operation2"}
    operation3 = {"type": "operation3"}
    
    system.enqueue_operation(operation1, priority="normal")
    system.enqueue_operation(operation2, priority="high")
    system.enqueue_operation(operation3, priority="low")
    
    # We should get the highest priority (lowest number) first
    next_op = system.get_next_operation()
    assert next_op["type"] == "operation2"  # High priority
    
    next_op = system.get_next_operation()
    assert next_op["type"] == "operation1"  # Normal priority
    
    next_op = system.get_next_operation()
    assert next_op["type"] == "operation3"  # Low priority
    
    # Queue should be empty now
    assert system.get_next_operation() is None

def test_context_hash():
    """Test the context hash creation."""
    # Empty context
    hash1 = create_context_hash({})
    
    # Context with history
    context2 = {
        "history": [
            {"role": "user", "content": "Show me file.txt"},
            {"role": "system", "content": "Here's file.txt"}
        ]
    }
    hash2 = create_context_hash(context2)
    
    # Different contexts should have different hashes
    assert hash1 != hash2
    
    # Same contexts should have same hash
    hash3 = create_context_hash(context2)
    assert hash2 == hash3
    
    # Only relevant parts of context should affect hash
    context4 = {
        "history": [
            {"role": "user", "content": "Show me file.txt"},
            {"role": "system", "content": "Here's file.txt"}
        ],
        "irrelevant_key": "This shouldn't affect the hash"
    }
    hash4 = create_context_hash(context4)
    assert hash2 == hash4  # Should ignore irrelevant_key
