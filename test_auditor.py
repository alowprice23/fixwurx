#!/usr/bin/env python3
"""
FixWurx Auditor Agent Tests

This script contains tests for the FixWurx Auditor Agent components.
"""

import os
import shutil
import unittest
import tempfile
import datetime
import json
import yaml
from typing import Dict, Any, Set

# Import auditor components
from auditor import (
    Auditor, ObligationLedger, RepoModules, EnergyCalculator,
    ProofMetrics, MetaAwareness, ErrorReporting
)
from graph_database import GraphDatabase, Node, Edge
from time_series_database import TimeSeriesDatabase
from document_store import DocumentStore
from benchmarking_system import BenchmarkingSystem, BenchmarkConfig


class TestObligationLedger(unittest.TestCase):
    """Tests for the ObligationLedger class"""

    def setUp(self):
        self.ledger = ObligationLedger()
        
        # Create a temporary rules file
        self.temp_dir = tempfile.mkdtemp()
        self.rules_file = os.path.join(self.temp_dir, "test_rules.json")
        with open(self.rules_file, 'w') as f:
            json.dump([
                {
                    "pattern": "authenticate_user",
                    "transforms_to": ["validate_credentials", "manage_sessions"]
                },
                {
                    "pattern": "store_data",
                    "transforms_to": ["validate_data", "persist_data"]
                }
            ], f)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_load_delta_rules(self):
        """Test loading delta rules from a file"""
        self.ledger.load_delta_rules(self.rules_file)
        self.assertEqual(len(self.ledger.delta_rules), 2)
    
    def test_compute_delta_closure(self):
        """Test computing delta closure"""
        self.ledger.load_delta_rules(self.rules_file)
        initial_goals = {"authenticate_user", "store_data"}
        closure = self.ledger.compute_delta_closure(initial_goals)
        
        # Check that closure contains all expected obligations
        expected_obligations = {
            "authenticate_user", "store_data",
            "validate_credentials", "manage_sessions",
            "validate_data", "persist_data"
        }
        self.assertEqual(closure, expected_obligations)
    
    def test_rule_application(self):
        """Test rule application logic"""
        rule = {
            "pattern": "authenticate_user",
            "transforms_to": ["validate_credentials", "manage_sessions"]
        }
        
        # Rule should apply to matching obligation
        self.assertTrue(self.ledger._rule_applies(rule, "authenticate_user"))
        
        # Rule should not apply to non-matching obligation
        self.assertFalse(self.ledger._rule_applies(rule, "authorize_user"))
        
        # Applying rule should return expected transformations
        result = self.ledger._apply_rule(rule, "authenticate_user")
        self.assertEqual(result, {"validate_credentials", "manage_sessions"})


class TestGraphDatabase(unittest.TestCase):
    """Tests for the GraphDatabase class"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db = GraphDatabase(self.temp_dir)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_add_node(self):
        """Test adding a node to the graph"""
        node = Node("test1", "test", {"name": "Test Node"})
        result = self.db.add_node(node)
        self.assertTrue(result)
        
        # Verify node was added
        retrieved = self.db.get_node("test1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, "test1")
        self.assertEqual(retrieved.properties["name"], "Test Node")
    
    def test_add_edge(self):
        """Test adding an edge to the graph"""
        # Add nodes first
        self.db.add_node(Node("node1", "component", {"name": "Component 1"}))
        self.db.add_node(Node("node2", "component", {"name": "Component 2"}))
        
        # Add edge
        edge = Edge("node1", "node2", "depends_on")
        result = self.db.add_edge(edge)
        self.assertTrue(result)
        
        # Verify edge was added
        retrieved = self.db.get_edge("node1", "node2")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.type, "depends_on")
    
    def test_find_path(self):
        """Test finding a path in the graph"""
        # Create a simple graph
        self.db.add_node(Node("A", "component", {}))
        self.db.add_node(Node("B", "component", {}))
        self.db.add_node(Node("C", "component", {}))
        
        self.db.add_edge(Edge("A", "B", "connects_to"))
        self.db.add_edge(Edge("B", "C", "connects_to"))
        
        # Find path from A to C
        path = self.db.find_path("A", "C")
        
        # Should be 3 nodes in the path: A -> B -> C
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0][0].id, "A")
        self.assertEqual(path[1][0].id, "B")
        self.assertEqual(path[2][0].id, "C")


class TestTimeSeriesDatabase(unittest.TestCase):
    """Tests for the TimeSeriesDatabase class"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db = TimeSeriesDatabase(self.temp_dir)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_time_series(self):
        """Test creating a time series"""
        ts = self.db.create_time_series("test_series", "Test Series", "seconds")
        self.assertIsNotNone(ts)
        self.assertEqual(ts.name, "test_series")
        
        # Verify time series was added to database
        retrieved = self.db.get_time_series("test_series")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test_series")
    
    def test_add_point(self):
        """Test adding a data point"""
        self.db.create_time_series("test_series")
        
        # Add a data point
        now = datetime.datetime.now()
        result = self.db.add_point("test_series", now, {"value": 42.0})
        self.assertTrue(result)
        
        # Verify point was added
        ts = self.db.get_time_series("test_series")
        self.assertEqual(len(ts.points), 1)
        self.assertEqual(ts.points[0].values["value"], 42.0)
    
    def test_get_latest_metric_value(self):
        """Test getting the latest metric value"""
        self.db.create_time_series("test_series")
        
        # Add data points
        now = datetime.datetime.now()
        self.db.add_point("test_series", now - datetime.timedelta(hours=1), {"value": 10.0})
        self.db.add_point("test_series", now, {"value": 20.0})
        
        # Get latest value
        latest = self.db.get_latest_metric_value("test_series", "value")
        self.assertIsNotNone(latest)
        self.assertEqual(latest[1], 20.0)  # Latest value should be 20.0


class TestDocumentStore(unittest.TestCase):
    """Tests for the DocumentStore class"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = DocumentStore(self.temp_dir)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_collection(self):
        """Test creating a collection"""
        collection = self.store.create_collection("test_collection")
        self.assertIsNotNone(collection)
        self.assertEqual(collection.name, "test_collection")
        
        # Verify collection was added
        collections = self.store.list_collections()
        self.assertIn("test_collection", collections)
    
    def test_create_document(self):
        """Test creating a document"""
        self.store.create_collection("test_collection")
        
        # Create a document
        doc = self.store.create_document(
            collection_name="test_collection",
            doc_type="test",
            fields={"name": "Test Document", "value": 42}
        )
        
        self.assertIsNotNone(doc)
        self.assertEqual(doc.type, "test")
        self.assertEqual(doc.fields["name"], "Test Document")
        
        # Verify document was added
        result = self.store.find_documents("test_collection", {"name": "Test Document"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].fields["value"], 42)


class TestBenchmarkingSystem(unittest.TestCase):
    """Tests for the BenchmarkingSystem class"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.benchmarking = BenchmarkingSystem(self.temp_dir)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_run_benchmark(self):
        """Test running a simple benchmark"""
        # Create a benchmark configuration
        config = BenchmarkConfig(
            name="test_benchmark",
            target="test",
            benchmark_type="PERFORMANCE",
            command="echo 'test'",
            iterations=2
        )
        
        # Run the benchmark
        result = self.benchmarking.run_benchmark(config)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.config.name, "test_benchmark")
        self.assertTrue(result.success)
        self.assertIn("execution_time", result.statistics)


class TestAuditor(unittest.TestCase):
    """Tests for the Auditor class"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a basic configuration
        self.config = {
            "repo_path": ".",
            "data_path": self.temp_dir,
            "delta_rules_file": os.path.join(self.temp_dir, "delta_rules.json"),
            "thresholds": {
                "energy_delta": 1e-7,
                "lambda": 0.9,
                "bug_probability": 1.1e-4,
                "drift": 0.02
            }
        }
        
        # Create delta rules
        with open(self.config["delta_rules_file"], 'w') as f:
            json.dump([
                {
                    "pattern": "authenticate_user",
                    "transforms_to": ["validate_credentials", "manage_sessions"]
                }
            ], f)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initializing the Auditor"""
        auditor = Auditor(self.config)
        self.assertIsNotNone(auditor)
        self.assertEqual(auditor.config, self.config)
    
    def test_pass_audit(self):
        """Test generating a PASS audit stamp"""
        auditor = Auditor(self.config)
        result = auditor._pass_audit()
        
        self.assertIn("audit_stamp", result)
        self.assertEqual(result["audit_stamp"]["status"], "PASS")
    
    def test_fail_audit(self):
        """Test generating a FAIL audit stamp"""
        auditor = Auditor(self.config)
        result = auditor._fail("TEST_REASON", {"test": "details"})
        
        self.assertIn("audit_stamp", result)
        self.assertEqual(result["audit_stamp"]["status"], "FAIL")
        self.assertEqual(result["audit_stamp"]["reason"], "TEST_REASON")
        self.assertEqual(result["audit_stamp"]["details"]["test"], "details")


if __name__ == "__main__":
    # Create test directory structure
    os.makedirs("auditor_data", exist_ok=True)
    
    # Run tests
    unittest.main()
