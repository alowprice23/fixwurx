#!/usr/bin/env python3
"""
FixWurx Auditor Components Test Script

This script tests the core components of the FixWurx Auditor Agent:
1. System Auditor - Comprehensive system auditing
2. Error Analysis - Advanced error detection and analysis
3. Functionality Verification - Behavioral and compliance testing

Usage:
    python test_auditor_components.py [--component COMPONENT] [--verbose]
"""

import os
import sys
import argparse
import logging
import yaml
import json
import datetime
import time
from typing import Dict, List, Any, Optional

# Force UTF-8 encoding for stdout/stderr on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'backslashreplace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'backslashreplace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('auditor_test')

# Mock database implementations for testing
class MockDocumentStore:
    """Mock implementation of document store for testing"""
    
    def __init__(self):
        self.collections = {}
        logger.info("Initialized mock document store")
    
    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections
    
    def create_collection(self, collection_name: str) -> None:
        if not self.collection_exists(collection_name):
            self.collections[collection_name] = {}
            logger.info(f"Created collection: {collection_name}")
    
    def create_document(self, collection_name: str, doc_id: str = None, fields: Dict[str, Any] = None) -> str:
        if not self.collection_exists(collection_name):
            self.create_collection(collection_name)
        
        if doc_id is None:
            doc_id = f"doc-{len(self.collections[collection_name])}"
        
        self.collections[collection_name][doc_id] = fields or {}
        logger.info(f"Created document in {collection_name}: {doc_id}")
        return doc_id
    
    def get_document(self, collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        if not self.collection_exists(collection_name) or doc_id not in self.collections[collection_name]:
            return None
        return self.collections[collection_name][doc_id]
    
    def update_document(self, collection_name: str, doc_id: str, updates: Dict[str, Any]) -> bool:
        if not self.collection_exists(collection_name) or doc_id not in self.collections[collection_name]:
            return False
        
        self.collections[collection_name][doc_id].update(updates)
        logger.info(f"Updated document in {collection_name}: {doc_id}")
        return True
    
    def query_documents(self, collection_name: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.collection_exists(collection_name):
            return []
        
        # Simple query implementation for testing
        results = []
        for doc_id, doc in self.collections[collection_name].items():
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            
            if match:
                doc_copy = doc.copy()
                doc_copy["doc_id"] = doc_id
                results.append(doc_copy)
        
        return results


class MockGraphDatabase:
    """Mock implementation of graph database for testing"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        logger.info("Initialized mock graph database")
    
    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any] = None) -> None:
        self.nodes[node_id] = {
            "type": node_type,
            "properties": properties or {}
        }
        logger.info(f"Added node: {node_id} ({node_type})")
    
    def add_edge(self, from_id: str, to_id: str, edge_type: str, properties: Dict[str, Any] = None) -> None:
        edge_id = f"{from_id}:{to_id}:{edge_type}"
        self.edges[edge_id] = {
            "from": from_id,
            "to": to_id,
            "type": edge_type,
            "properties": properties or {}
        }
        logger.info(f"Added edge: {from_id} -[{edge_type}]-> {to_id}")
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.nodes.get(node_id)
    
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        if node_id not in self.nodes:
            return False
        
        self.nodes[node_id]["properties"].update(properties)
        logger.info(f"Updated node: {node_id}")
        return True
    
    def get_edges(self, from_id: str = None, to_id: str = None, edge_type: str = None) -> List[str]:
        results = []
        
        for edge_id, edge in self.edges.items():
            if from_id and edge["from"] != from_id:
                continue
            if to_id and edge["to"] != to_id:
                continue
            if edge_type and edge["type"] != edge_type:
                continue
            
            # Return the target node ID if filtering by from_id and edge_type
            if from_id and edge_type:
                results.append(edge["to"])
            else:
                results.append(edge_id)
        
        return results


class MockTimeSeriesDatabase:
    """Mock implementation of time series database for testing"""
    
    def __init__(self):
        self.series = {}
        logger.info("Initialized mock time series database")
    
    def add_point(self, series_name: str, timestamp: datetime.datetime, values: Dict[str, Any]) -> None:
        if series_name not in self.series:
            self.series[series_name] = []
        
        self.series[series_name].append({
            "timestamp": timestamp,
            "values": values
        })
        logger.info(f"Added point to series {series_name} at {timestamp}")
    
    def get_points(self, series_name: str, start_time: datetime.datetime = None, 
                  end_time: datetime.datetime = None, limit: int = None) -> List[Dict[str, Any]]:
        if series_name not in self.series:
            return []
        
        results = []
        for point in self.series[series_name]:
            if start_time and point["timestamp"] < start_time:
                continue
            if end_time and point["timestamp"] > end_time:
                continue
            
            results.append(point)
        
        if limit is not None and len(results) > limit:
            results = results[:limit]
        
        return results
    
    def get_series(self) -> List[str]:
        return list(self.series.keys())


# Test configuration
DEFAULT_CONFIG = {
    "system_auditor": {
        "component_thresholds": {
            "function_complexity_threshold": 10,
            "min_test_coverage": 0.7,
            "min_comment_ratio": 0.1,
            "max_memory_usage": 200,
            "max_cpu_usage": 50
        },
        "system_thresholds": {
            "max_api_response_time": 0.3,
            "max_database_read_time": 0.5,
            "max_database_write_time": 1.0,
            "security_compliance_level": "high"
        }
    },
    "error_analysis": {
        "sample_errors": [
            {
                "component": "database",
                "message": "Connection timeout after 30s",
                "severity": "high",
                "status": "open",
                "stack_trace": 'File "database.py", line 45, in connect\n  raise TimeoutError("Connection timeout after 30s")',
                "timestamp": "2025-07-12T22:15:30"
            },
            {
                "component": "api",
                "message": "Invalid parameter: user_id is required",
                "severity": "medium",
                "status": "open",
                "stack_trace": 'File "api.py", line 120, in get_user\n  raise ValueError("Invalid parameter: user_id is required")',
                "timestamp": "2025-07-12T23:05:12"
            }
        ]
    },
    "functionality_verification": {
        "tests_directory": "tests",
        "sample_test_suite": {
            "suite_id": "test-suite-1",
            "name": "Core Functionality Tests",
            "description": "Tests for core system functionality",
            "component": "core",
            "tags": ["core", "critical"],
            "test_cases": [
                {
                    "test_id": "test-1",
                    "name": "User Authentication",
                    "description": "Test user authentication flow",
                    "component": "auth",
                    "category": "behavioral",
                    "inputs": {"username": "test_user", "password": "password123"},
                    "expected_outputs": {"success": True, "user_id": "12345"}
                },
                {
                    "test_id": "test-2",
                    "name": "Data Storage",
                    "description": "Test data storage functionality",
                    "component": "database",
                    "category": "behavioral",
                    "inputs": {"key": "test_key", "value": "test_value"},
                    "expected_outputs": {"success": True}
                }
            ]
        }
    }
}


def create_test_config() -> None:
    """Create a test configuration file"""
    config_path = "auditor_test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f)
    logger.info(f"Created test configuration: {config_path}")
    return config_path


def create_test_directory_structure() -> None:
    """Create directory structure for testing"""
    # Create tests directory
    os.makedirs("tests", exist_ok=True)
    
    # Create sample test suite file
    with open("tests/sample_suite.yaml", 'w') as f:
        yaml.dump(DEFAULT_CONFIG["functionality_verification"]["sample_test_suite"], f)
    
    # Create sample source files for testing
    os.makedirs("src", exist_ok=True)
    with open("src/sample.py", 'w') as f:
        f.write("""
# Sample file for testing
def test_function():
    \"\"\"This is a test function\"\"\"
    return True

class TestClass:
    def method1(self):
        \"\"\"Test method\"\"\"
        return "test"
""")
    
    logger.info("Created test directory structure")


def test_system_auditor() -> Dict[str, Any]:
    """Test the System Auditor component"""
    logger.info("Testing System Auditor...")
    
    try:
        # Import the module
        from system_auditor import SystemAuditor
        
        # Create mock databases
        document_store = MockDocumentStore()
        time_series_db = MockTimeSeriesDatabase()
        graph_db = MockGraphDatabase()
        
        # Create test config
        config_path = create_test_config()
        
        # Initialize the system auditor
        auditor = SystemAuditor(config_path, document_store, time_series_db, graph_db)
        
        # Test component auditing
        component_results = auditor.run_component_audits()
        logger.info(f"Component audit results: {component_results['summary']['passed']} passed, "
                   f"{component_results['summary']['failed']} failed")
        
        # Test system auditing
        system_results = auditor.run_system_audits()
        logger.info(f"System audit results: {system_results['summary']['passed']} passed, "
                  f"{system_results['summary']['failed']} failed")
        
        # Test meta auditing
        meta_results = auditor.run_meta_audits()
        logger.info(f"Meta audit results: {meta_results['summary']['passed']} passed, "
                  f"{meta_results['summary']['failed']} failed")
        
        # Test comprehensive audit
        comprehensive_results = auditor.run_comprehensive_audit()
        logger.info(f"Comprehensive audit completed")
        
        return {
            "status": "success",
            "results": {
                "component_audit": component_results,
                "system_audit": system_results,
                "meta_audit": meta_results,
                "comprehensive_audit": comprehensive_results
            }
        }
        
    except Exception as e:
        logger.error(f"Error testing System Auditor: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


def test_error_analysis() -> Dict[str, Any]:
    """Test the Error Analysis component"""
    logger.info("Testing Error Analysis...")
    
    try:
        # Import the modules
        from advanced_error_analysis import ErrorRepository, ErrorAnalyzer
        
        # Create mock databases
        document_store = MockDocumentStore()
        time_series_db = MockTimeSeriesDatabase()
        graph_db = MockGraphDatabase()
        
        # Initialize the error repository
        repository = ErrorRepository(document_store, graph_db)
        
        # Store sample errors
        error_ids = []
        for error_data in DEFAULT_CONFIG["error_analysis"]["sample_errors"]:
            error_id = repository.store_error(error_data)
            error_ids.append(error_id)
            logger.info(f"Stored sample error: {error_id}")
        
        # Initialize the error analyzer
        analyzer = ErrorAnalyzer(repository, document_store, graph_db, time_series_db)
        
        # Test error analysis
        analysis_results = {}
        for error_id in error_ids:
            analysis = analyzer.analyze_error(error_id)
            analysis_results[error_id] = analysis
            logger.info(f"Analyzed error {error_id}: {analysis.get('root_cause', {}).get('cause_type', 'unknown')}")
        
        # Test trend analysis
        trends = analyzer.identify_error_trends()
        logger.info(f"Identified error trends: {trends['summary']['total_errors']} total errors")
        
        # Test pattern analysis
        patterns = analyzer.identify_error_patterns()
        logger.info(f"Identified {len(patterns)} error patterns")
        
        # Test report generation
        report = analyzer.generate_error_report()
        logger.info("Generated error report")
        
        return {
            "status": "success",
            "results": {
                "error_ids": error_ids,
                "analyses": analysis_results,
                "trends": trends,
                "patterns": patterns,
                "report": report
            }
        }
        
    except Exception as e:
        logger.error(f"Error testing Error Analysis: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


def test_functionality_verification() -> Dict[str, Any]:
    """Test the Functionality Verification component"""
    logger.info("Testing Functionality Verification...")
    
    try:
        # Import the module
        from functionality_verification import FunctionalityVerifier, TestSuite, TestCase
        
        # Create mock databases
        document_store = MockDocumentStore()
        time_series_db = MockTimeSeriesDatabase()
        graph_db = MockGraphDatabase()
        
        # Create test directory structure
        create_test_directory_structure()
        
        # Create test config
        config_path = create_test_config()
        
        # Initialize the functionality verifier
        verifier = FunctionalityVerifier(config_path, document_store, time_series_db, graph_db)
        
        # Create a test suite programmatically
        suite = TestSuite(
            suite_id="test-suite-2",
            name="API Tests",
            description="Tests for API functionality",
            component="api",
            tags=["api", "integration"]
        )
        
        # Add test cases
        suite.add_test_case(TestCase(
            test_id="api-test-1",
            name="Get User API",
            description="Test get user API endpoint",
            component="api",
            category="behavioral",
            inputs={"user_id": "12345"},
            expected_outputs={"success": True, "username": "test_user"}
        ))
        
        suite.add_test_case(TestCase(
            test_id="api-test-2",
            name="Create User API",
            description="Test create user API endpoint",
            component="api",
            category="behavioral",
            inputs={"username": "new_user", "email": "new@example.com"},
            expected_outputs={"success": True, "user_id": "67890"}
        ))
        
        # Add suite to verifier
        verifier.test_suites[suite.suite_id] = suite
        
        # Test behavioral tests
        behavioral_results = verifier.run_behavioral_tests()
        logger.info(f"Behavioral tests: {behavioral_results['passed']} passed, {behavioral_results['failed']} failed")
        
        # Test quality assurance tests
        qa_results = verifier.run_quality_assurance_tests()
        logger.info(f"QA tests: {qa_results['passed']} passed, {qa_results['failed']} failed")
        
        # Test compliance tests
        compliance_results = verifier.run_compliance_tests()
        logger.info(f"Compliance tests: {compliance_results['passed']} passed, {compliance_results['failed']} failed")
        
        # Test all tests
        all_results = verifier.run_all_tests()
        logger.info(f"All tests: {all_results['passed']} passed, {all_results['failed']} failed")
        
        # Test coverage analysis
        coverage = verifier.analyze_test_coverage()
        logger.info(f"Test coverage: {coverage['total_test_cases']} test cases")
        
        # Test verification report
        report = verifier.generate_verification_report()
        logger.info("Generated verification report")
        
        return {
            "status": "success",
            "results": {
                "behavioral_tests": behavioral_results,
                "qa_tests": qa_results,
                "compliance_tests": compliance_results,
                "all_tests": all_results,
                "coverage": coverage,
                "report": report
            }
        }
        
    except Exception as e:
        logger.error(f"Error testing Functionality Verification: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


def run_all_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run all component tests"""
    # Set log level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    start_time = time.time()
    
    # Run tests for each component
    system_auditor_results = test_system_auditor()
    error_analysis_results = test_error_analysis()
    functionality_verification_results = test_functionality_verification()
    
    # Compile results
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "runtime": time.time() - start_time,
        "components": {
            "system_auditor": system_auditor_results,
            "error_analysis": error_analysis_results,
            "functionality_verification": functionality_verification_results
        },
        "summary": {
            "total_components": 3,
            "successful_components": sum(1 for r in [system_auditor_results, error_analysis_results, functionality_verification_results] 
                                       if r["status"] == "success"),
            "failed_components": sum(1 for r in [system_auditor_results, error_analysis_results, functionality_verification_results] 
                                    if r["status"] == "error")
        }
    }
    
    return results


def display_results(results: Dict[str, Any]) -> None:
    """Display test results in a readable format"""
    print("\n" + "="*80)
    print(f"FixWurx Auditor Components Test Results")
    print("="*80)
    
    print(f"\nTest completed at: {results['timestamp']}")
    print(f"Total runtime: {results['runtime']:.2f} seconds")
    print(f"Components tested: {results['summary']['total_components']}")
    print(f"Successful components: {results['summary']['successful_components']}")
    print(f"Failed components: {results['summary']['failed_components']}")
    
    # Display component results
    for component_name, component_results in results["components"].items():
        print(f"\n{'-'*40}")
        print(f"Component: {component_name}")
        print(f"Status: {component_results['status']}")
        
        if component_results["status"] == "error":
            print(f"Error: {component_results['error']}")
        else:
            # Display specific results for each component
            if component_name == "system_auditor":
                audit_results = component_results["results"]["comprehensive_audit"]
                summary = audit_results.get("summary", {})
                print(f"Total checks: {summary.get('total_checks', 0)}")
                print(f"Passed: {summary.get('passed', 0)}")
                print(f"Failed: {summary.get('failed', 0)}")
                
            elif component_name == "error_analysis":
                trends = component_results["results"]["trends"]["summary"]
                print(f"Total errors analyzed: {trends['total_errors']}")
                print(f"Open errors: {trends['open_errors']}")
                print(f"Resolved errors: {trends['resolved_errors']}")
                print(f"Error patterns identified: {len(component_results['results']['patterns'])}")
                
            elif component_name == "functionality_verification":
                all_tests = component_results["results"]["all_tests"]
                print(f"Total tests: {all_tests['total_tests']}")
                print(f"Passed: {all_tests['passed']}")
                print(f"Failed: {all_tests['failed']}")
                print(f"Pass rate: {all_tests.get('pass_rate', 0):.2%}")
    
    print("\n" + "="*80)
    status = "PASSED" if results['summary']['failed_components'] == 0 else "FAILED"
    print(f"Overall test status: {status}")
    print("="*80 + "\n")


def main() -> int:
    """Main function"""
    parser = argparse.ArgumentParser(description="Test FixWurx Auditor components")
    
    parser.add_argument(
        '--component',
        choices=['system_auditor', 'error_analysis', 'functionality_verification', 'all'],
        default='all',
        help='Component to test (default: all)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output file for test results'
    )
    
    args = parser.parse_args()
    
    try:
        if args.component == 'all':
            results = run_all_tests(args.verbose)
        elif args.component == 'system_auditor':
            results = {
                "timestamp": datetime.datetime.now().isoformat(),
                "runtime": 0,
                "components": {
                    "system_auditor": test_system_auditor()
                },
                "summary": {
                    "total_components": 1,
                    "successful_components": 1 if test_system_auditor()["status"] == "success" else 0,
                    "failed_components": 1 if test_system_auditor()["status"] == "error" else 0
                }
            }
        elif args.component == 'error_analysis':
            results = {
                "timestamp": datetime.datetime.now().isoformat(),
                "runtime": 0,
                "components": {
                    "error_analysis": test_error_analysis()
                },
                "summary": {
                    "total_components": 1,
                    "successful_components": 1 if test_error_analysis()["status"] == "success" else 0,
                    "failed_components": 1 if test_error_analysis()["status"] == "error" else 0
                }
            }
        elif args.component == 'functionality_verification':
            results = {
                "timestamp": datetime.datetime.now().isoformat(),
                "runtime": 0,
                "components": {
                    "functionality_verification": test_functionality_verification()
                },
                "summary": {
                    "total_components": 1,
                    "successful_components": 1 if test_functionality_verification()["status"] == "success" else 0,
                    "failed_components": 1 if test_functionality_verification()["status"] == "error" else 0
                }
            }
        
        # Display results
        display_results(results)
        
        # Write results to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Wrote test results to {args.output}")
        
        # Return exit code based on test results
        return 0 if results["summary"]["failed_components"] == 0 else 1
        
    except Exception as e:
        logger.error(f"Error running tests: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
