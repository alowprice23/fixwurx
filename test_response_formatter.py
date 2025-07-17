#!/usr/bin/env python3
"""
Test Response Formatter

This script tests the response formatter implementation to ensure it correctly
formats different types of content with various verbosity levels.
"""

import os
import sys
import unittest
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_response_formatter.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("TestResponseFormatter")

# Import response formatter
from response_formatter import get_instance, ResponseFormatter

class TestResponseFormatter(unittest.TestCase):
    """
    Test suite for the Response Formatter implementation.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create formatter instances with different verbosity levels
        self.concise_formatter = get_instance(verbosity="concise")
        self.normal_formatter = get_instance(verbosity="normal")
        self.verbose_formatter = get_instance(verbosity="verbose")
        
        # Ensure colors are disabled for predictable testing
        self.concise_formatter.use_colors = False
        self.normal_formatter.use_colors = False
        self.verbose_formatter.use_colors = False
    
    def test_text_formatting(self):
        """
        Test formatting of plain text.
        """
        test_text = "This is a simple test message that should be properly formatted."
        
        # Test with different verbosity levels
        concise_result = self.concise_formatter.format_text(test_text)
        normal_result = self.normal_formatter.format_text(test_text)
        verbose_result = self.verbose_formatter.format_text(test_text)
        
        # All should be the same for simple text
        self.assertEqual(concise_result, test_text)
        self.assertEqual(normal_result, test_text)
        self.assertEqual(verbose_result, test_text)
    
    def test_code_block_formatting(self):
        """
        Test formatting of code blocks.
        """
        test_code = "def hello_world():\n    print('Hello, world!')\n\nhello_world()"
        
        # Test with different languages
        python_result = self.normal_formatter.format_code_block(test_code, "python")
        js_result = self.normal_formatter.format_code_block(test_code, "javascript")
        
        # Check that the code is included in the result
        self.assertIn(test_code, python_result)
        self.assertIn(test_code, js_result)
        
        # Check that language identifiers are included
        self.assertIn("```python", python_result)
        self.assertIn("```javascript", js_result)
    
    def test_error_formatting(self):
        """
        Test formatting of error messages.
        """
        test_message = "File not found"
        test_details = "The file 'missing.txt' could not be found."
        
        # Test with different verbosity levels
        concise_result = self.concise_formatter.format_error(test_message, test_details)
        normal_result = self.normal_formatter.format_error(test_message, test_details)
        verbose_result = self.verbose_formatter.format_error(test_message, test_details)
        
        # Check that message is included in all
        self.assertIn(test_message, concise_result)
        self.assertIn(test_message, normal_result)
        self.assertIn(test_message, verbose_result)
        
        # Check that details are only included in normal and verbose
        self.assertNotIn(test_details, concise_result)
        self.assertIn(test_details, normal_result)
        self.assertIn(test_details, verbose_result)
    
    def test_command_output_formatting(self):
        """
        Test formatting of command output.
        """
        test_command = "ls -la"
        test_output = "total 12\nline1\nline2\nline3\nline4\nline5\nline6\nline7"
        
        # Test with different verbosity levels
        concise_result = self.concise_formatter.format_command_output(test_output, test_command)
        normal_result = self.normal_formatter.format_command_output(test_output, test_command)
        verbose_result = self.verbose_formatter.format_command_output(test_output, test_command)
        
        # Check that command is included in all
        self.assertIn(test_command, concise_result)
        self.assertIn(test_command, normal_result)
        self.assertIn(test_command, verbose_result)
        
        # Check that concise output is truncated for long outputs
        self.assertIn("...", concise_result)
        self.assertNotIn("line4", concise_result)
        
        # Check that normal and verbose include full output
        self.assertIn("line4", normal_result)
        self.assertIn("line4", verbose_result)
    
    def test_table_formatting(self):
        """
        Test formatting of tables.
        """
        test_headers = ["Name", "Age", "Occupation"]
        test_rows = [
            ["Alice", "32", "Engineer"],
            ["Bob", "28", "Designer"],
            ["Charlie", "45", "Manager"]
        ]
        
        # Test table formatting
        result = self.normal_formatter.format_table(test_headers, test_rows)
        
        # Check that all data is included
        for header in test_headers:
            self.assertIn(header, result)
        
        for row in test_rows:
            for cell in row:
                self.assertIn(cell, result)
    
    def test_list_formatting(self):
        """
        Test formatting of lists.
        """
        test_items = ["First item", "Second item", "Third item"]
        
        # Test ordered and unordered lists
        ordered_result = self.normal_formatter.format_list(test_items, ordered=True)
        unordered_result = self.normal_formatter.format_list(test_items, ordered=False)
        
        # Check that all items are included
        for item in test_items:
            self.assertIn(item, ordered_result)
            self.assertIn(item, unordered_result)
        
        # Check for ordered list numbering
        self.assertIn("1.", ordered_result)
        self.assertIn("2.", ordered_result)
        self.assertIn("3.", ordered_result)
        
        # Check for unordered list bullets
        for i in range(len(test_items)):
            self.assertIn("- ", unordered_result)
    
    def test_complete_response_formatting(self):
        """
        Test formatting of a complete response.
        """
        test_response = {
            "message": "Here's the result of your request:",
            "code_blocks": [
                {
                    "language": "python",
                    "code": "def hello_world():\n    print('Hello, world!')\n\nhello_world()"
                }
            ],
            "tables": [
                {
                    "headers": ["Name", "Value"],
                    "rows": [["foo", "42"], ["bar", "99"]],
                    "title": "Configuration"
                }
            ],
            "command_outputs": [
                {
                    "command": "ls -la",
                    "output": "total 12\ndrwxr-xr-x 2 user user 4096 Jul 15 14:30 .\ndrwxr-xr-x 4 user user 4096 Jul 15 14:29 ..",
                    "exit_code": 0
                }
            ]
        }
        
        # Format the complete response
        result = self.normal_formatter.format_response(test_response)
        
        # Check that all sections are included
        self.assertIn(test_response["message"], result)
        self.assertIn("hello_world", result)
        self.assertIn("Configuration", result)
        self.assertIn("ls -la", result)
    
    def test_integration_with_conversational_interface(self):
        """
        Test integration with the conversational interface.
        """
        try:
            # Import conversational interface
            from conversational_interface import ConversationalInterface
            
            # Create a mock registry
            class MockRegistry:
                def __init__(self):
                    self.components = {}
                
                def register_component(self, name, component):
                    self.components[name] = component
                
                def get_component(self, name):
                    return self.components.get(name)
            
            # Initialize the interface
            registry = MockRegistry()
            interface = ConversationalInterface(registry)
            
            # Verify formatter is initialized
            self.assertIsNotNone(interface.formatter)
            self.assertEqual(interface.formatter.verbosity, "normal")
            
            # Test verbosity setting
            interface.set_verbosity("verbose")
            self.assertEqual(interface.formatter.verbosity, "verbose")
            
            # Test response formatting
            test_response = "This is a test response."
            interface.display_response(test_response)
            
            # Test code block formatting
            code_response = "Here's some code:\n```python\ndef test():\n    return True\n```"
            interface.display_response(code_response)
            
            # Test error formatting
            error_response = "Error: Something went wrong"
            interface.display_response(error_response)
            
            self.assertTrue(True)  # If we get here without errors, the test passes
            
        except ImportError:
            self.skipTest("ConversationalInterface not available")
        except Exception as e:
            self.fail(f"Error testing integration: {e}")

def run_tests():
    """
    Run the test suite.
    """
    logger.info("Starting Response Formatter Tests")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("Response Formatter Tests completed")

if __name__ == "__main__":
    run_tests()
