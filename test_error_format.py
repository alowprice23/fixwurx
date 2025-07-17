"""
Test script to verify the standardized error format implementation.
This script creates test errors with the extended fields and verifies
that they are properly stored, contextualized, and displayed.
"""

import sys
import os
import logging
import datetime
import json
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [TestErrorFormat] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('test_error_format')

# Import required components
from sensor_registry import ErrorReport, create_sensor_registry
from component_sensors import ObligationLedgerSensor
from llm_integrations import LLMManager, LLMResponse
from llm_sensor_integration import create_llm_integration

# Mock LLM Manager for testing
class MockLLMManager(LLMManager):
    """Mock LLM Manager that returns predefined responses."""
    
    def __init__(self):
        """Initialize the mock LLM manager."""
        logger.info("Initialized MockLLMManager")
    
    def chat(self, role: str, content: str, task_type: str = None, 
            complexity: str = None) -> LLMResponse:
        """
        Return a mock response based on the content.
        
        Args:
            role: Role for the message
            content: Message content
            task_type: Type of task
            complexity: Complexity level
            
        Returns:
            Mock LLM response
        """
        # Generate explanation if requested
        if "Explain" in content:
            return LLMResponse(text="This is a mock explanation of the error. The error occurred due to test conditions.")
        
        # Generate recommendations if requested
        if "recommendations" in content.lower():
            return LLMResponse(text="1. First recommendation: Do something\n2. Second recommendation: Do something else\n3. Third recommendation: Yet another action")
        
        # Generate pattern analysis if requested
        if "analyzing error patterns" in content:
            return LLMResponse(text="IDENTIFIED PATTERNS:\n1. Pattern one\n2. Pattern two\n\nROOT CAUSE HYPOTHESES:\n1. Hypothesis one\n2. Hypothesis two")
        
        # Generate diagnosis if requested
        if "diagnostic" in content:
            return LLMResponse(text="This is a mock diagnosis of the issue.")
        
        # Generate state explanation if requested
        if "explaining" in content:
            return LLMResponse(text="This is a mock explanation of the internal state.")
        
        # Generate correction suggestions if requested
        if "corrections" in content:
            return LLMResponse(text="CORRECTION 1: Fix Something\nCOMPLEXITY: LOW\nSTEPS:\n1. Do this\n2. Do that\nEXPECTED IMPACT: Better performance")
        
        # Default response
        return LLMResponse(text="This is a mock LLM response.")


def test_create_error_with_extended_fields():
    """Test creating an error report with extended fields."""
    logger.info("Testing creating an error report with extended fields")
    
    # Create an error report
    report = ErrorReport(
        sensor_id="test_sensor",
        component_name="TestComponent",
        error_type="TEST_ERROR",
        severity="MEDIUM",
        details={"message": "Test error message"},
        context={"source_file": "test_file.py", "source_function": "test_function", "line_number": 42}
    )
    
    # Set extended fields
    report.root_cause = {
        "cause_type": "test_cause",
        "confidence": 0.9,
        "details": {
            "description": "Test root cause description",
            "source_file": "test_file.py",
            "source_function": "test_function",
            "line_number": 42
        },
        "potential_causes": [
            {
                "cause_type": "potential_cause",
                "confidence": 0.5,
                "details": {
                    "description": "Potential cause description"
                }
            }
        ]
    }
    
    report.impact = {
        "severity": "MEDIUM",
        "scope": "single_component",
        "affected_components": ["TestComponent"],
        "affected_functionality": ["test_functionality"],
        "user_impact": "Test user impact",
        "system_impact": "Test system impact"
    }
    
    report.related_errors = ["ERR-20250713000000-test123", "ERR-20250713000001-test456"]
    report.recommendations = ["First recommendation", "Second recommendation"]
    
    # Verify fields were set correctly
    assert report.root_cause["cause_type"] == "test_cause", "Root cause type not set correctly"
    assert report.impact["scope"] == "single_component", "Impact scope not set correctly"
    assert len(report.related_errors) == 2, "Related errors not set correctly"
    assert len(report.recommendations) == 2, "Recommendations not set correctly"
    
    # Convert to dictionary and verify all fields are included
    report_dict = report.to_dict()
    assert "root_cause" in report_dict, "Root cause not included in dictionary"
    assert "impact" in report_dict, "Impact not included in dictionary"
    assert "related_errors" in report_dict, "Related errors not included in dictionary"
    assert "recommendations" in report_dict, "Recommendations not included in dictionary"
    
    logger.info("Error report with extended fields created and verified successfully")
    return report


def test_error_contextualization():
    """Test contextualizing an error report with LLM."""
    logger.info("Testing error contextualization with LLM")
    
    # Create mock LLM manager
    llm_manager = MockLLMManager()
    
    # Create registry and sensor manager
    registry, sensor_manager = create_sensor_registry()
    
    # Create LLM integration components
    llm_components = create_llm_integration(registry, sensor_manager, llm_manager)
    
    # Get error contextualizer
    error_contextualizer = llm_components["error_contextualizer"]
    
    # Create basic error report
    report = ErrorReport(
        sensor_id="test_sensor",
        component_name="TestComponent",
        error_type="TEST_ERROR",
        severity="MEDIUM",
        details={"message": "Test error message"},
        context={"source_file": "test_file.py", "source_function": "test_function", "line_number": 42}
    )
    
    # Contextualize the error
    enhanced = error_contextualizer.contextualize(report)
    
    # Verify extended fields were populated
    assert report.root_cause is not None, "Root cause not populated"
    assert report.impact is not None, "Impact not populated"
    assert hasattr(report, "recommendations"), "Recommendations not populated"
    
    # Verify dictionary contains extended fields
    assert "root_cause" in enhanced, "Root cause not in enhanced dictionary"
    assert "impact" in enhanced, "Impact not in enhanced dictionary"
    assert "recommendations" in enhanced, "Recommendations not in enhanced dictionary"
    
    logger.info("Error contextualization verified successfully")
    return report


def test_yaml_serialization():
    """Test YAML serialization of error reports with extended fields."""
    logger.info("Testing YAML serialization of error reports")
    
    # Create an error report with extended fields
    report = test_create_error_with_extended_fields()
    
    # Serialize to YAML
    yaml_str = yaml.dump(report.to_dict())
    
    # Deserialize from YAML
    data = yaml.safe_load(yaml_str)
    
    # Create a new report from the data
    new_report = ErrorReport.from_dict(data)
    
    # Verify fields were preserved
    assert new_report.root_cause["cause_type"] == report.root_cause["cause_type"], "Root cause type not preserved"
    assert new_report.impact["scope"] == report.impact["scope"], "Impact scope not preserved"
    assert len(new_report.related_errors) == len(report.related_errors), "Related errors not preserved"
    assert len(new_report.recommendations) == len(report.recommendations), "Recommendations not preserved"
    
    logger.info("YAML serialization verified successfully")


def run_tests():
    """Run all tests."""
    logger.info("Starting tests")
    
    try:
        # Test creating an error report with extended fields
        test_create_error_with_extended_fields()
        
        # Test error contextualization
        test_error_contextualization()
        
        # Test YAML serialization
        test_yaml_serialization()
        
        logger.info("All tests passed successfully")
        return True
    
    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
