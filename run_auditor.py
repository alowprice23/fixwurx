#!/usr/bin/env python3
"""
FixWurx Auditor Runner

This script provides a command-line interface for running the FixWurx Auditor Agent.
It loads the configuration, initializes the auditor, runs the audit process,
and outputs the results.

Usage:
    python run_auditor.py --config auditor_config.yaml [--verbose] [--output OUTPUT_FILE]
"""

import os
import sys
import argparse
import logging
import yaml
import json
import datetime
from typing import Dict, Any, Optional

# Import auditor components
from auditor import Auditor
from graph_database import GraphDatabase
from time_series_database import TimeSeriesDatabase
from document_store import DocumentStore
from benchmarking_system import BenchmarkingSystem, BenchmarkConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('run_auditor')


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
        '--output',
        type=str,
        help='Path to output file for audit results'
    )
    
    parser.add_argument(
        '--run-benchmarks',
        action='store_true',
        help='Run benchmarks before audit'
    )
    
    parser.add_argument(
        '--check',
        choices=['completeness', 'correctness', 'meta-awareness', 'all'],
        default='all',
        help='Specify which check(s) to run (default: all)'
    )
    
    parser.add_argument(
        '--report-format',
        choices=['yaml', 'json', 'text'],
        default='yaml',
        help='Format for the audit report (default: yaml)'
    )
    
    parser.add_argument(
        '--delta-rules',
        type=str,
        help='Override path to delta rules file'
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


def run_benchmarks(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run benchmarks before the audit.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with benchmark results
    """
    benchmark_config = config.get('benchmarking', {})
    if not benchmark_config.get('enabled', True):
        logger.info("Benchmarking is disabled in configuration")
        return {}
    
    benchmark_path = benchmark_config.get('path', 'auditor_data/benchmarks')
    
    logger.info("Running benchmarks...")
    
    # Initialize benchmarking system
    benchmarking = BenchmarkingSystem(benchmark_path)
    
    # Define benchmarks to run
    benchmarks = [
        BenchmarkConfig(
            name="obligation_closure",
            target="auditor",
            benchmark_type="PERFORMANCE",
            command="python -c \"from auditor import ObligationLedger; import time; start=time.time(); ledger=ObligationLedger(); ledger.load_delta_rules('delta_rules.json'); ledger.compute_delta_closure({'authenticate_user', 'store_data'}); print(time.time()-start)\"",
            iterations=3
        ),
        BenchmarkConfig(
            name="repository_scan",
            target="auditor",
            benchmark_type="PERFORMANCE",
            command="python -c \"from auditor import RepoModules; import time; start=time.time(); repo=RepoModules('.'); repo.scan_repository(); print(time.time()-start)\"",
            iterations=3
        )
    ]
    
    # Run benchmarks
    results = {}
    for config in benchmarks:
        try:
            result = benchmarking.run_benchmark(config)
            results[config.name] = {
                "mean": result.statistics.get("execution_time", {}).get("mean", 0),
                "min": result.statistics.get("execution_time", {}).get("min", 0),
                "max": result.statistics.get("execution_time", {}).get("max", 0)
            }
            logger.info(f"Benchmark {config.name}: mean={results[config.name]['mean']:.3f}s")
        except Exception as e:
            logger.error(f"Failed to run benchmark {config.name}: {e}")
    
    return results


def format_output(data: Dict[str, Any], format_type: str) -> str:
    """
    Format output data based on specified format.
    
    Args:
        data: Data to format
        format_type: Format type (yaml, json, text)
        
    Returns:
        Formatted output string
    """
    if format_type == 'yaml':
        return yaml.dump(data, default_flow_style=False)
    elif format_type == 'json':
        return json.dumps(data, indent=2)
    else:  # text
        lines = []
        
        # Add stamp status
        stamp = data.get('audit_stamp', {})
        status = stamp.get('status', 'UNKNOWN')
        lines.append(f"Audit-Stamp: {status}")
        
        # Add timestamp
        timestamp = stamp.get('timestamp', '')
        if timestamp:
            lines.append(f"Timestamp: {timestamp}")
        
        # Add reason and details for FAIL status
        if status == 'FAIL':
            reason = stamp.get('reason', '')
            if reason:
                lines.append(f"Reason: {reason}")
            
            details = stamp.get('details', {})
            if details:
                lines.append("Details:")
                for key, value in details.items():
                    lines.append(f"  {key}: {value}")
        
        # Add benchmarks if present
        benchmarks = data.get('benchmarks', {})
        if benchmarks:
            lines.append("\nBenchmarks:")
            for name, metrics in benchmarks.items():
                lines.append(f"  {name}:")
                for metric, value in metrics.items():
                    lines.append(f"    {metric}: {value:.3f}s")
        
        return '\n'.join(lines)


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


def main() -> int:
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set up logging
        setup_logging(config, args.verbose)
        
        # Override delta rules file if specified
        if args.delta_rules:
            config['delta_rules_file'] = args.delta_rules
        
        # Run benchmarks if requested
        benchmark_results = {}
        if args.run_benchmarks:
            benchmark_results = run_benchmarks(config)
        
        # Initialize auditor
        logger.info("Initializing auditor...")
        auditor = Auditor(config)
        
        # Run audit
        logger.info("Running audit...")
        
        if args.check == 'completeness':
            # Run only completeness check
            result = auditor.check_completeness()
            if not result.get('success', False):
                audit_result = auditor._fail("MISSING_OBLIGATION", result.get('details'))
            else:
                audit_result = auditor._pass_audit()
        elif args.check == 'correctness':
            # Run only correctness check
            result = auditor.check_correctness()
            if not result.get('success', False):
                audit_result = auditor._fail(result.get('reason'), result.get('details'))
            else:
                audit_result = auditor._pass_audit()
        elif args.check == 'meta-awareness':
            # Run only meta-awareness check
            result = auditor.check_meta_awareness()
            if not result.get('success', False):
                audit_result = auditor._fail(result.get('reason'), result.get('details'))
            else:
                audit_result = auditor._pass_audit()
        else:
            # Run full audit
            audit_result = auditor.run_audit()
        
        # Add benchmarks to result
        if benchmark_results:
            audit_result['benchmarks'] = benchmark_results
        
        # Format output
        output = format_output(audit_result, args.report_format)
        
        # Write output
        write_output(output, args.output)
        
        # Return exit code based on audit result
        return 0 if audit_result.get('audit_stamp', {}).get('status') == 'PASS' else 1
        
    except Exception as e:
        logger.error(f"Audit failed: {e}", exc_info=True)
        
        # Create failure audit stamp
        audit_result = {
            "audit_stamp": {
                "status": "ERROR",
                "reason": "AUDITOR_EXECUTION_FAILED",
                "details": {"error": str(e)},
                "timestamp": datetime.datetime.now().isoformat()
            }
        }
        
        # Format and write output
        output = format_output(audit_result, args.report_format)
        write_output(output, args.output)
        
        return 2


if __name__ == "__main__":
    sys.exit(main())
