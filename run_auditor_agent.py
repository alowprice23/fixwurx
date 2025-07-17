#!/usr/bin/env python3
"""
FixWurx Auditor Agent Runner

This script provides a command-line interface for running the FixWurx Auditor Agent.
It initializes the agent, starts monitoring, and provides ways to interact with it.

Usage:
    python run_auditor_agent.py --config auditor_config.yaml [--verbose] [--no-autonomous]
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
import datetime
from typing import Dict, Any, Optional

# Force UTF-8 encoding for stdout/stderr on Windows
if sys.platform == 'win32':
    import codecs
    # Use utf-8 encoding for console output
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'backslashreplace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'backslashreplace')

# Import Auditor Agent
from auditor_agent import AuditorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [AuditorAgent] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('run_auditor_agent')


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run the FixWurx Auditor Agent')
    
    parser.add_argument(
        '--config',
        type=str,
        default='auditor_config.yaml',
        help='Path to configuration file (default: auditor_config.yaml)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-autonomous',
        action='store_true',
        help='Disable autonomous mode'
    )
    
    parser.add_argument(
        '--run-audit',
        action='store_true',
        help='Run an initial audit'
    )
    
    parser.add_argument(
        '--monitor-time',
        type=int,
        default=0,
        help='Time to run monitoring in seconds (0 = indefinitely)'
    )
    
    parser.add_argument(
        '--trigger-event',
        type=str,
        help='Trigger an event (code_change, build_failure, performance_alert, audit_request)'
    )
    
    parser.add_argument(
        '--event-data',
        type=str,
        help='JSON string with event data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output file for agent status'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If configuration file has invalid format
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in configuration file: {e}")
        raise


def setup_logging(config: Dict[str, Any], verbose: bool) -> None:
    """
    Set up logging based on configuration and command line arguments.
    
    Args:
        config: Configuration dictionary
        verbose: Whether to enable verbose output
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_file = log_config.get('file')
    
    # Override log level if verbose flag is set
    if verbose:
        log_level = 'DEBUG'
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Add file handler if specified
    if log_file:
        from logging.handlers import RotatingFileHandler
        
        max_size = log_config.get('max_size_mb', 10) * 1024 * 1024
        backup_count = log_config.get('backup_count', 5)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        
        file_format = log_config.get('format', '%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")


def format_output(data: Dict[str, Any]) -> str:
    """
    Format agent status output.
    
    Args:
        data: Agent status data
        
    Returns:
        Formatted output string
    """
    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "agent_status": data
    }
    
    return json.dumps(output, indent=2, default=str)


def write_output(output: str, output_path: Optional[str]) -> None:
    """
    Write output to file or stdout.
    
    Args:
        output: Output string
        output_path: Path to output file, or None for stdout
    """
    if output_path:
        try:
            with open(output_path, 'w') as f:
                f.write(output)
            logger.info(f"Wrote output to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write output to {output_path}: {e}")
            print(output)
    else:
        print(output)


def get_event_data(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Get event data from command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Event data dictionary
    """
    if args.event_data:
        try:
            return json.loads(args.event_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in event data: {e}")
            return {}
    
    # Default event data based on event type
    if args.trigger_event == "code_change":
        return {
            "files": ["example.py", "test_example.py"]
        }
    elif args.trigger_event == "build_failure":
        return {
            "build_id": "build-123",
            "components": ["component1", "component2"],
            "dependencies": [
                {
                    "component": "component1",
                    "status": "mismatch",
                    "expected_version": "1.0.0",
                    "actual_version": "0.9.0"
                }
            ]
        }
    elif args.trigger_event == "performance_alert":
        return {
            "components": ["component1"],
            "metrics": {
                "response_time": 2.5,
                "threshold": 1.0
            }
        }
    elif args.trigger_event == "audit_request":
        return {
            "force_full": True
        }
    
    return {}


def main() -> int:
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set up logging
        setup_logging(config, args.verbose)
        
        # Initialize agent
        logger.info("Initializing Auditor Agent...")
        agent = AuditorAgent(args.config, autonomous_mode=not args.no_autonomous)
        
        # Start agent
        logger.info("Starting Auditor Agent...")
        agent.start()
        
        try:
            # Run initial audit if requested
            if args.run_audit:
                logger.info("Running initial audit...")
                audit_result = agent.run_audit()
                logger.info(f"Audit result: {audit_result.get('audit_stamp', {}).get('status')}")
            
            # Trigger event if specified
            if args.trigger_event:
                logger.info(f"Triggering event: {args.trigger_event}")
                event_data = get_event_data(args)
                agent.trigger_event(args.trigger_event, event_data)
            
            # Run monitoring for specified time
            if args.monitor_time > 0:
                logger.info(f"Running monitoring for {args.monitor_time} seconds...")
                time.sleep(args.monitor_time)
            elif args.monitor_time == 0 and not args.run_audit and not args.trigger_event:
                # Run indefinitely if no other actions were specified
                logger.info("Running monitoring indefinitely. Press Ctrl+C to stop...")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Monitoring stopped by user")
            
            # Get agent status
            status = agent.get_status()
            
            # Format and write output
            output = format_output(status)
            write_output(output, args.output)
            
            return 0
            
        finally:
            # Stop agent
            logger.info("Stopping Auditor Agent...")
            agent.stop()
        
    except Exception as e:
        logger.error(f"Auditor Agent failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
