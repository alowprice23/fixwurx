"""
Tests for the CLI interface in the calculator application.
These tests will demonstrate FixWurx's capability to detect and fix bugs.
"""
import unittest
from unittest.mock import patch, MagicMock
from calculator.ui.cli import CalculatorCLI

class TestCalculatorCLI(unittest.TestCase):
    """Test class for the calculator CLI interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = CalculatorCLI()
    
    def test_initialization(self):
        """Test CLI initialization."""
        # BUG 23: FIXED - Proper initialization is now tested
        self.assertIsNotNone(self.cli.history)
        self.assertIsNotNone(self.cli.operations)
        
        # Basic operations should be present
        self.assertIn('add', self.cli.operations)
        self.assertIn('subtract', self.cli.operations)
        
        # BUG 24: FIXED - Random operation is now properly mapped
        # Note: This test now expects random to be present
        self.assertIn('random', self.cli.operations)
    
    def test_get_operation(self):
        """Test retrieving operation functions."""
        add_func = self.cli.get_operation('add')
        self.assertEqual(add_func(5, 3), 8)
        
        # BUG 25: FIXED - Proper error handling for invalid operation
        # Now raises ValueError with descriptive message instead of KeyError
        with self.assertRaises(ValueError):
            self.cli.get_operation('invalid_op')
    
    def test_calculate(self):
        """Test calculation with various operations."""
        # Test valid calculations
        self.assertEqual(self.cli.calculate('add', 5, 3), 8)
        
        # BUG 26: FIXED - Now validates inputs properly
        # Should return error for invalid inputs
        result = self.cli.calculate('add', 'five', 3)
        self.assertTrue(isinstance(result, str) and 'Error' in result)
        
        # BUG 27: FIXED - Now handles missing second parameter
        result = self.cli.calculate('add', 5, None)
        self.assertTrue(isinstance(result, str) and 'Error' in result)
        
        # Test single-parameter operations
        self.assertAlmostEqual(self.cli.calculate('sqrt', 9, None), 3)
        
        # BUG 28: FIXED - Now records history consistently
        # Should record single-parameter operations too
        history_before = len(self.cli.history)
        self.cli.calculate('sqrt', 9, None)
        history_after = len(self.cli.history)
        self.assertEqual(history_after, history_before + 1)  # Incremented by 1
        
        # BUG 29: FIXED - Now handles exceptions properly
        # Should still return an error string for division by zero
        result = self.cli.calculate('divide', 5, 0)
        self.assertTrue(isinstance(result, str) and 'Error' in result)
    
    @patch('builtins.print')
    def test_display_menu(self, mock_print):
        """Test displaying the calculator menu."""
        self.cli.display_menu()
        
        # BUG 30: FIXED - Now provides a complete menu
        # Should call print multiple times
        self.assertGreater(mock_print.call_count, 1)
        
        # Check if menu title is printed
        args_list = [args[0] for args, _ in mock_print.call_args_list]
        self.assertTrue(any('Calculator Menu' in str(arg) for arg in args_list))
        
        # Check if usage examples are included
        self.assertTrue(any('Examples' in str(arg) for arg in args_list))
    
    def test_parse_input(self):
        """Test parsing user input."""
        # Test valid input with two parameters
        op, a, b = self.cli.parse_input('add 5 3')
        self.assertEqual(op, 'add')
        self.assertEqual(a, 5)
        self.assertEqual(b, 3)
        
        # Test valid input with one parameter
        op, a, b = self.cli.parse_input('sqrt 9')
        self.assertEqual(op, 'sqrt')
        self.assertEqual(a, 9)
        self.assertIsNone(b)
        
        # BUG 31: FIXED - Now provides better parsing with validation
        # Test invalid input formats and expect specific error
        with self.assertRaises(ValueError):
            self.cli.parse_input('add five 3')
        
        # BUG 32: FIXED - Now provides specific error messages
        # Should raise an error with a message
        with self.assertRaises(ValueError):
            self.cli.parse_input('add')

if __name__ == '__main__':
    unittest.main()
