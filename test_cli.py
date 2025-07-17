#!/usr/bin/env python3
"""
test_cli.py
-----------
Automated tests for the cli.py module.

Tests various commands and functionality of the FixWurx CLI to ensure
proper operation even in the absence of required dependencies.
"""

import unittest
import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the cli module
sys.path.append('.')
import cli

class TestCLI(unittest.TestCase):
    """Test class for cli.py functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temp directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.triangulum_dir = Path(self.test_dir) / '.triangulum'
        self.triangulum_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock log file
        self.log_file = self.triangulum_dir / 'runtime.log'
        with open(self.log_file, 'w') as f:
            f.write("tick: 1000 agents: 5 entropy_bits: 3.14\n")
            f.write("tick: 1001 agents: 5 entropy_bits: 3.12\n")
            f.write("tick: 1002 agents: 5 entropy_bits: 3.10\n")
        
        # Create mock review db
        self.reviews_db = self.triangulum_dir / 'reviews.sqlite'
        
        # Save original values
        self.original_cwd = os.getcwd()
        self.original_review_db = cli.REVIEW_DB
        
        # Set to test values
        os.chdir(self.test_dir)
        cli.REVIEW_DB = self.reviews_db
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original values
        os.chdir(self.original_cwd)
        cli.REVIEW_DB = self.original_review_db
        
        # Remove temp directory
        shutil.rmtree(self.test_dir)
    
    def test_banner(self):
        """Test the _banner function."""
        with patch('builtins.print') as mock_print:
            cli._banner("TEST BANNER")
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            self.assertIn("TEST BANNER", call_args)
    
    def test_fmt_age(self):
        """Test the _fmt_age function."""
        # Test with current time (age 0)
        current_time = int(cli.time.time())
        self.assertEqual(cli._fmt_age(current_time), "00:00:00")
        
        # Test with time 1 hour ago
        hour_ago = current_time - 3600
        self.assertEqual(cli._fmt_age(hour_ago), "01:00:00")
        
        # Test with time 1 day, 2 hours, 3 minutes and 4 seconds ago
        time_ago = current_time - (86400 + 7200 + 180 + 4)
        self.assertEqual(cli._fmt_age(time_ago), "26:03:04")
    
    def test_tail_metrics(self):
        """Test the _tail_metrics function."""
        # Test with existing log file
        lines = cli._tail_metrics(2)
        self.assertEqual(len(lines), 2)
        self.assertIn("entropy_bits: 3.12", lines[0])
        self.assertIn("entropy_bits: 3.10", lines[1])
        
        # Test with non-existent log file
        os.remove(self.log_file)
        lines = cli._tail_metrics()
        self.assertEqual(len(lines), 1)
        self.assertIn("no runtime.log", lines[0])
    
    @patch('cli.subprocess.run')
    def test_cmd_run(self, mock_run):
        """Test the cmd_run function."""
        # Create mock args
        args = MagicMock()
        args.config = "config.yaml"
        args.tick_ms = 100
        args.verbose = True
        
        # Call the function
        with patch('builtins.print') as mock_print:
            cli.cmd_run(args)
            
            # Check that it called subprocess.run with the correct args
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            self.assertEqual(call_args[0], sys.executable)
            self.assertEqual(call_args[1], "main.py")
            self.assertEqual(call_args[2], "--config")
            self.assertEqual(call_args[3], "config.yaml")
            self.assertEqual(call_args[4], "--tick-ms")
            self.assertEqual(call_args[5], "100")
            
            # Check that it printed the command
            mock_print.assert_called_once()
    
    def test_cmd_status(self):
        """Test the cmd_status function."""
        # Create mock args
        args = MagicMock()
        args.lines = 2
        args.follow = False
        
        # Call the function
        with patch('builtins.print') as mock_print:
            cli.cmd_status(args)
            
            # Check that it printed the banner and lines
            calls = mock_print.call_args_list
            self.assertGreaterEqual(len(calls), 3)  # Banner + 2 lines
            self.assertIn("LATEST METRICS", calls[0][0][0])
    
    def test_parser_setup(self):
        """Test the argument parser setup."""
        parser = cli.setup_parser()
        
        # Test the main parser
        self.assertIsNotNone(parser)
        
        # Test a few subparsers
        args = parser.parse_args(['run', '--config', 'test.yaml'])
        self.assertEqual(args.command, 'run')
        self.assertEqual(args.config, 'test.yaml')
        
        args = parser.parse_args(['status', '--lines', '10'])
        self.assertEqual(args.command, 'status')
        self.assertEqual(args.lines, 10)
        
        args = parser.parse_args(['entropy', '--verbose'])
        self.assertEqual(args.command, 'entropy')
        self.assertTrue(args.verbose)
    
    @patch('cli._check_auth', return_value='test_user')
    @patch('cli.cmd_status')
    def test_main_dispatch(self, mock_cmd_status, mock_check_auth):
        """Test the main function dispatching to commands."""
        # Test dispatch to status command
        with patch('sys.argv', ['cli.py', 'status']):
            with patch('cli.setup_parser', return_value=cli.setup_parser()):
                cli.main()
                mock_cmd_status.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling in the main function."""
        # Test unknown command
        with patch('sys.argv', ['cli.py', 'unknown_command']):
            with patch('cli.setup_parser', return_value=cli.setup_parser()):
                with patch('builtins.print') as mock_print:
                    with patch('sys.exit') as mock_exit:
                        cli.main()
                        mock_print.assert_called()
                        # Verify that sys.exit was called at least once with error code 1 or 2
                        # (The actual number of calls may vary based on argparse behavior)
                        self.assertTrue(mock_exit.called)
                        exit_codes = [call[0][0] for call in mock_exit.call_args_list]
                        self.assertTrue(1 in exit_codes or 2 in exit_codes)

if __name__ == '__main__':
    unittest.main()
