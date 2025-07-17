import pytest
import time
from unittest.mock import MagicMock, patch

from components.intent_classification_system import Intent
from components.intent_classification_integration import OptimizedIntentClassifier

class MockRegistry:
    def get_component(self, name):
        return None

@pytest.fixture
def optimized_classifier():
    return OptimizedIntentClassifier(MockRegistry())

def test_cached_classification(optimized_classifier):
    """Test that the integration properly caches intent classifications."""
    # Mock the base classifier to track calls
    base_classify = optimized_classifier.intent_classifier.classify_intent
    optimized_classifier.intent_classifier.classify_intent = MagicMock()
    
    # Set up the mock to return a specific intent
    mock_intent = Intent("file_access")
    mock_intent.parameters = {"path": "test.txt"}
    mock_intent.execution_path = "direct"
    mock_intent.required_agents = ["analyst"]
    optimized_classifier.intent_classifier.classify_intent.return_value = mock_intent
    
    # First call should use the base classifier
    context = {"history": [{"role": "user", "content": "previous message"}]}
    result1 = optimized_classifier.classify_intent("show me test.txt", context)
    
    # Verify base classifier was called
    optimized_classifier.intent_classifier.classify_intent.assert_called_once()
    
    # Reset the mock to verify second call
    optimized_classifier.intent_classifier.classify_intent.reset_mock()
    
    # Second call with same query and context should use cache
    result2 = optimized_classifier.classify_intent("show me test.txt", context)
    
    # Verify base classifier was NOT called (cache hit)
    optimized_classifier.intent_classifier.classify_intent.assert_not_called()
    
    # Results should be equivalent
    assert result1.type == result2.type
    assert result1.parameters == result2.parameters
    assert result1.execution_path == result2.execution_path
    assert result1.required_agents == result2.required_agents
    
    # Restore original method
    optimized_classifier.intent_classifier.classify_intent = base_classify

def test_performance_tracking(optimized_classifier):
    """Test that performance is tracked properly."""
    # Mock time.time to control timing
    with patch('time.time') as mock_time:
        # Set up time sequence: start=100, end=100.2 (200ms)
        mock_time.side_effect = [100.0, 100.2]
        
        # Mock the optimization system's tracking method
        optimized_classifier.optimization_system.track_classification_performance = MagicMock()
        
        # Mock the base classifier
        base_classify = optimized_classifier.intent_classifier.classify_intent
        mock_intent = Intent("command_execution")
        optimized_classifier.intent_classifier.classify_intent = MagicMock(return_value=mock_intent)
        
        # Perform classification
        optimized_classifier.classify_intent("run ls", {})
        
        # Verify performance was tracked with ~200ms
        args, kwargs = optimized_classifier.optimization_system.track_classification_performance.call_args
        duration_ms = args[0]
        assert abs(duration_ms - 200.0) < 0.1
        
        # Restore original method
        optimized_classifier.intent_classifier.classify_intent = base_classify

def test_predict_next_intents(optimized_classifier):
    """Test the prediction of next intents."""
    # Mock the optimization system's prediction method
    optimized_classifier.optimization_system.get_intent_predictions = MagicMock()
    optimized_classifier.optimization_system.get_intent_predictions.return_value = [
        ("file_access", 0.7),
        ("command_execution", 0.2),
        ("file_modification", 0.05)  # Below threshold
    ]
    
    # Get predictions
    predictions = optimized_classifier.predict_next_intents("previous_intent")
    
    # Verify predictions were filtered and formatted
    assert len(predictions) == 2  # Only 2 above threshold
    assert predictions[0]["intent_type"] == "file_access"
    assert predictions[0]["probability"] == 0.7
    assert "suggested_query" in predictions[0]
    
    assert predictions[1]["intent_type"] == "command_execution"
    assert predictions[1]["probability"] == 0.2
    assert "suggested_query" in predictions[1]

def test_agent_failure_handling(optimized_classifier):
    """Test that agent failure handling is properly delegated."""
    # Create a test intent
    intent = Intent("system_debugging")
    intent.required_agents = ["auditor", "analyst", "verifier"]
    
    # Mock the base classifier's handle_agent_failure method
    optimized_classifier.intent_classifier.handle_agent_failure = MagicMock()
    optimized_classifier.intent_classifier.handle_agent_failure.return_value = (
        ["auditor", "observer", "verifier"],  # Updated agents
        {"analyst": "observer"}  # Fallback mapping
    )
    
    # Call the integrated handler
    failed_agents = ["analyst"]
    updated_agents, fallbacks = optimized_classifier.handle_agent_failure(intent, failed_agents)
    
    # Verify delegation
    optimized_classifier.intent_classifier.handle_agent_failure.assert_called_once_with(intent, failed_agents)
    
    # Verify results
    assert updated_agents == ["auditor", "observer", "verifier"]
    assert fallbacks == {"analyst": "observer"}

def test_performance_report(optimized_classifier):
    """Test that performance report includes all required information."""
    # Mock the optimization system's report method
    mock_report = {
        "cache": {"hit_rate": 0.75, "size": 100, "capacity": 1000},
        "predictor": {"unique_intents": 10, "total_intents_processed": 500},
        "classification": {"avg": 50.0, "min": 10.0, "max": 200.0},
        "execution": {"file_access": {"avg": 100.0}},
        "slow_operations": ["command_execution"],
        "queue_size": 5
    }
    optimized_classifier.optimization_system.get_performance_report = MagicMock(return_value=mock_report)
    
    # Get report
    report = optimized_classifier.get_performance_report()
    
    # Verify all expected sections are present
    assert "cache" in report
    assert "predictor" in report
    assert "classification" in report
    assert "execution" in report
    assert "slow_operations" in report
    assert "queue_size" in report

def test_operation_prioritization(optimized_classifier):
    """Test that operations can be prioritized properly."""
    # Mock the optimization system's queue methods
    optimized_classifier.optimization_system.enqueue_operation = MagicMock()
    optimized_classifier.optimization_system.get_next_operation = MagicMock()
    
    # Create test operations
    op1 = {"id": 1, "type": "file_op"}
    op2 = {"id": 2, "type": "command_op"}
    
    # Enqueue with different priorities
    optimized_classifier.prioritize_operation(op1, "high")
    optimized_classifier.prioritize_operation(op2, "normal")
    
    # Verify correct priorities were used
    args1, kwargs1 = optimized_classifier.optimization_system.enqueue_operation.call_args_list[0]
    args2, kwargs2 = optimized_classifier.optimization_system.enqueue_operation.call_args_list[1]
    
    assert args1[0] == op1
    assert args1[1] == "high"
    assert args2[0] == op2
    assert args2[1] == "normal"
    
    # Test getting next operation
    optimized_classifier.get_next_prioritized_operation()
    optimized_classifier.optimization_system.get_next_operation.assert_called_once()

def test_resource_optimization_recommendations(optimized_classifier):
    """Test that resource optimization provides useful recommendations."""
    # Mock the performance report to trigger recommendations
    mock_report = {
        "cache": {"hit_rate": 0.3, "total_requests": 200},
        "slow_operations": ["heavy_operation", "another_slow_one"]
    }
    optimized_classifier.get_performance_report = MagicMock(return_value=mock_report)
    
    # Get optimization recommendations
    recommendations = optimized_classifier.optimize_resource_usage()
    
    # Verify recommendations
    assert "recommendations" in recommendations
    assert len(recommendations["recommendations"]) >= 2  # Should have at least 2 recommendations
    
    # One about low cache hit rate
    assert any("Low cache hit rate" in rec for rec in recommendations["recommendations"])
    
    # One about slow operations
    assert any("Slow operations detected" in rec for rec in recommendations["recommendations"])
