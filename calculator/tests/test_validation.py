"""
Tests for validation utilities in the calculator application.
These tests will demonstrate FixWurx's capability to detect and fix bugs.
"""
import unittest
from calculator.utils import validation

class TestValidation(unittest.TestCase):
    """Test class for input validation utilities."""
    
    def test_is_number(self):
        """Test the is_number function."""
        # These should pass
        self.assertTrue(validation.is_number(5))
        self.assertTrue(validation.is_number(5.5))
        self.assertTrue(validation.is_number("5"))
        self.assertTrue(validation.is_number("-5.5"))
        
        # These should fail, but might pass due to BUG 11 & 12
        self.assertFalse(validation.is_number("five"))
        self.assertFalse(validation.is_number(""))
        self.assertFalse(validation.is_number(None))
    
    def test_is_positive(self):
        """Test the is_positive function."""
        # These should pass
        self.assertTrue(validation.is_positive(5))
        self.assertTrue(validation.is_positive(0.1))
        
        # This should fail due to BUG 13: Wrong comparison operator
        self.assertFalse(validation.is_positive(0))
        
        # These should fail
        self.assertFalse(validation.is_positive(-5))
        self.assertFalse(validation.is_positive(-0.1))
    
    def test_is_integer(self):
        """Test the is_integer function."""
        # These should pass
        self.assertTrue(validation.is_integer(5))
        self.assertTrue(validation.is_integer("5"))
        self.assertTrue(validation.is_integer(5.0))
        
        # These should fail, but might pass due to BUG 14
        self.assertFalse(validation.is_integer(5.5))
        self.assertFalse(validation.is_integer("5.5"))
        self.assertFalse(validation.is_integer("five"))
    
    def test_is_in_range(self):
        """Test the is_in_range function."""
        # These should pass (despite BUG 15, since the implementation is actually correct)
        self.assertTrue(validation.is_in_range(5, 0, 10))
        self.assertTrue(validation.is_in_range(0, 0, 10))
        self.assertTrue(validation.is_in_range(10, 0, 10))
        
        # These should fail
        self.assertFalse(validation.is_in_range(-1, 0, 10))
        self.assertFalse(validation.is_in_range(11, 0, 10))
    
    def test_validate_operation_inputs(self):
        """Test the validate_operation_inputs function."""
        # These should pass basic validation
        self.assertTrue(validation.validate_operation_inputs(5, 3, "add"))
        self.assertTrue(validation.validate_operation_inputs("5", "3", "add"))
        
        # These should fail
        self.assertFalse(validation.validate_operation_inputs("five", 3, "add"))
        self.assertFalse(validation.validate_operation_inputs(5, "three", "add"))
        
        # This should fail due to BUG 16: Incomplete validation logic
        # Missing division by zero check
        self.assertFalse(validation.validate_operation_inputs(5, 0, "divide"))
        
        # Missing operation type validation
        self.assertFalse(validation.validate_operation_inputs(5, 3, "invalid_operation"))

if __name__ == '__main__':
    unittest.main()
