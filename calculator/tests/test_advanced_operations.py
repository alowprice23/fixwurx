"""
Tests for advanced operations in the calculator application.
These tests will demonstrate FixWurx's capability to detect and fix bugs.
"""
import unittest
import math
from calculator.operations import advanced_operations

class TestAdvancedOperations(unittest.TestCase):
    """Test class for advanced mathematical operations."""
    
    def test_square_root(self):
        """Test the square_root function."""
        self.assertEqual(advanced_operations.square_root(9), 3)
        self.assertEqual(advanced_operations.square_root(2), math.sqrt(2))
        
        # This will fail due to BUG 6: No negative number check
        with self.assertRaises(ValueError):
            advanced_operations.square_root(-1)
    
    def test_factorial(self):
        """Test the factorial function."""
        self.assertEqual(advanced_operations.factorial(5), 120)
        self.assertEqual(advanced_operations.factorial(1), 1)
        
        # This will fail due to BUG 7: Incorrect base case
        self.assertEqual(advanced_operations.factorial(0), 1)
    
    def test_logarithm(self):
        """Test the logarithm function."""
        # This will fail due to BUG 8: Wrong math function
        self.assertAlmostEqual(advanced_operations.logarithm(100), 2)
        self.assertAlmostEqual(advanced_operations.logarithm(100, 2), math.log(100, 2))
        
        # Should handle invalid inputs
        with self.assertRaises(ValueError):
            advanced_operations.logarithm(0)
    
    def test_sine(self):
        """Test the sine function."""
        # This will fail due to BUG 9: Incorrect function (using cosine)
        self.assertAlmostEqual(advanced_operations.sine(30), 0.5)
        self.assertAlmostEqual(advanced_operations.sine(90), 1)
        self.assertAlmostEqual(advanced_operations.sine(0), 0)
    
    def test_random_operation(self):
        """Test the random_operation function."""
        # This will fail due to BUG 5 & 10: Missing import and undefined 'random' module
        min_val, max_val = 1, 10
        for _ in range(100):  # Test multiple times due to randomness
            result = advanced_operations.random_operation(min_val, max_val)
            self.assertTrue(min_val <= result <= max_val)

if __name__ == '__main__':
    unittest.main()
