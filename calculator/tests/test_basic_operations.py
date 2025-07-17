"""
Tests for basic operations in the calculator application.
These tests will demonstrate FixWurx's capability to detect and fix bugs.
"""
import unittest
from calculator.operations import basic_operations

class TestBasicOperations(unittest.TestCase):
    """Test class for basic arithmetic operations."""
    
    def test_add(self):
        """Test the add function."""
        self.assertEqual(basic_operations.add(5, 3), 8)
        self.assertEqual(basic_operations.add(-1, 1), 0)
        self.assertEqual(basic_operations.add(0, 0), 0)
    
    def test_subtract(self):
        """Test the subtract function."""
        # This will fail due to BUG 1: Incorrect order of operands
        self.assertEqual(basic_operations.subtract(5, 3), 2)
        self.assertEqual(basic_operations.subtract(1, 1), 0)
        self.assertEqual(basic_operations.subtract(0, 5), -5)
    
    def test_multiply(self):
        """Test the multiply function."""
        # This will fail due to BUG 2: Using addition instead of multiplication
        self.assertEqual(basic_operations.multiply(5, 3), 15)
        self.assertEqual(basic_operations.multiply(1, 0), 0)
        self.assertEqual(basic_operations.multiply(-2, 3), -6)
    
    def test_divide(self):
        """Test the divide function."""
        self.assertEqual(basic_operations.divide(6, 3), 2)
        self.assertEqual(basic_operations.divide(5, 2), 2.5)
        
        # This will fail due to BUG 3: No zero division check
        with self.assertRaises(ZeroDivisionError):
            basic_operations.divide(5, 0)
    
    def test_power(self):
        """Test the power function."""
        # This will fail due to BUG 4: Incorrect implementation
        self.assertEqual(basic_operations.power(2, 3), 8)
        self.assertEqual(basic_operations.power(5, 0), 1)
        self.assertEqual(basic_operations.power(0, 5), 0)
    
    def test_modulus(self):
        """Test the modulus function."""
        self.assertEqual(basic_operations.modulus(5, 3), 2)
        self.assertEqual(basic_operations.modulus(10, 5), 0)
        
        # This will fail due to missing zero check (similar to divide)
        with self.assertRaises(ZeroDivisionError):
            basic_operations.modulus(5, 0)

if __name__ == '__main__':
    unittest.main()
