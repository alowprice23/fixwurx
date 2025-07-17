"""
Tests for memory utilities in the calculator application.
These tests will demonstrate FixWurx's capability to detect and fix bugs.
"""
import unittest
from calculator.utils import memory

class TestCalculationHistory(unittest.TestCase):
    """Test class for calculation history functionality."""
    
    def test_initialization(self):
        """Test the initialization of CalculationHistory."""
        history = memory.CalculationHistory(max_size=5)
        self.assertEqual(len(history), 0)  # Use len() instead of direct attribute access
        
        # BUG 17 is fixed: Using correct variable name in initialization
        self.assertEqual(history.max_size, 5)  # Now using max_size instead of size
    
    def test_add_calculation(self):
        """Test adding calculations to history."""
        history = memory.CalculationHistory(max_size=3)
        
        # Add some calculations
        history.add_calculation("add", 5, 3, 8)
        history.add_calculation("subtract", 10, 4, 6)
        
        # Check they were added correctly
        self.assertEqual(len(history), 2)
        all_calcs = history.get_all_calculations()
        self.assertEqual(all_calcs[0]["operation"], "add")
        self.assertEqual(all_calcs[1]["result"], 6)
        
        # BUG 18 is fixed: Now has timestamp for each calculation
        self.assertIn("timestamp", all_calcs[0])
        
        # BUG 19 is fixed: Now enforcing max size correctly
        # Add more calculations than max_size
        history.add_calculation("multiply", 2, 3, 6)
        history.add_calculation("divide", 10, 2, 5)
        
        # Should only keep max_size (3) calculations
        self.assertEqual(len(history), 3)
    
    def test_get_last_calculation(self):
        """Test retrieving the last calculation."""
        history = memory.CalculationHistory()
        
        # BUG 20 is fixed: Now has empty check and returns None
        self.assertIsNone(history.get_last_calculation())
        
        # Add a calculation and test retrieval
        history.add_calculation("add", 1, 2, 3)
        last_calc = history.get_last_calculation()
        self.assertEqual(last_calc["operation"], "add")
        self.assertEqual(last_calc["result"], 3)
    
    def test_clear_history(self):
        """Test clearing the calculation history."""
        history = memory.CalculationHistory()
        
        # Add some calculations
        history.add_calculation("add", 5, 3, 8)
        history.add_calculation("subtract", 10, 4, 6)
        
        # Clear history
        history.clear_history()
        
        # BUG 21 is fixed: Using efficient way to clear list
        # The history should be empty after clearing
        self.assertEqual(len(history), 0)
    
    def test_get_all_calculations(self):
        """Test retrieving all calculations."""
        history = memory.CalculationHistory()
        
        # Empty history should return empty list
        self.assertEqual(len(history.get_all_calculations()), 0)
        
        # Add some calculations
        history.add_calculation("add", 5, 3, 8)
        history.add_calculation("subtract", 10, 4, 6)
        
        # Get all calculations
        all_calcs = history.get_all_calculations()
        self.assertEqual(len(all_calcs), 2)
        self.assertEqual(all_calcs[0]["a"], 5)
        self.assertEqual(all_calcs[1]["b"], 4)
    
    def test_get_calculations_by_operation(self):
        """Test retrieving calculations by operation type."""
        history = memory.CalculationHistory()
        
        # Add calculations with different operations
        history.add_calculation("add", 5, 3, 8)
        history.add_calculation("subtract", 10, 4, 6)
        history.add_calculation("add", 2, 2, 4)
        
        # Get calculations by operation
        add_calcs = history.get_calculations_by_operation("add")
        subtract_calcs = history.get_calculations_by_operation("subtract")
        
        # BUG 22 was actually correct: Filtering implementation
        self.assertEqual(len(add_calcs), 2)
        self.assertEqual(len(subtract_calcs), 1)
        self.assertEqual(add_calcs[0]["result"], 8)
        self.assertEqual(add_calcs[1]["result"], 4)

if __name__ == '__main__':
    unittest.main()
