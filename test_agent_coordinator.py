#!/usr/bin/env python3
"""
test_agent_coordinator.py
────────────────────────
Test suite for the enhanced AgentCoordinator module.

This test verifies:
1. Enhanced agent handoff protocol
2. Path-based execution flow
3. Planner integration
4. Fallback strategy execution
5. Metrics collection and reporting
"""

import asyncio
import json
import unittest
import unittest.mock
from typing import Dict, List, Any, Optional
import tempfile
from pathlib import Path

from agent_coordinator import AgentCoordinator, HandoffStatus, _Artefacts
# Import only the types we need for type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Specialized_agents import ObserverAgent, AnalystAgent, VerifierAgent
    from planner_agent import PlannerAgent
from state_machine import Phase
from triangulation_engine import TriangulationEngine
from data_structures import BugState, PlannerPath


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name: str = "mock"):
        self.name = name
        self.prompts = []
        self.responses = {}
        
        # Default responses
        self.responses["INITIALIZATION"] = "INITIALIZED"
        self.responses["Initialize the family tree"] = json.dumps({"status": "success"})
    
    async def ask(self, prompt: str) -> str:
        """Mock ask method."""
        self.prompts.append(prompt)
        
        # Return predefined response if available
        for key, response in self.responses.items():
            if key in prompt:
                return response
        
        # Default response with JSON status
        if "HANDOFF:" in prompt:
            return json.dumps({"status": "SUCCESS", "message": "Handoff processed"})
        
        return "OK"


class TestAgentCoordinator(unittest.TestCase):
    """Test case for the AgentCoordinator class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock agents
        self.mock_observer = MockAgent("observer")
        self.mock_analyst = MockAgent("analyst")
        self.mock_verifier = MockAgent("verifier")
        self.mock_planner = MockAgent("planner")
        
        # Create a config
        self.config = {
            "use_enhanced_handoff": True,
            "collect_metrics": True,
            "max_retry_attempts": 2
        }
        
        # Create a mock coordinator that doesn't initialize real agents
        self.coordinator = AgentCoordinator(self.config)
        
        # Replace the init method to avoid initialization errors
        def mock_init(obj, config=None):
            obj.config = config or {}
            obj._observer = self.mock_observer
            obj._analyst = self.mock_analyst
            obj._verifier = self.mock_verifier
            obj._planner = self.mock_planner
            obj._art = _Artefacts()
            obj._last_phase = None
            obj._family_tree_updated = False
            obj._agents_initialized = False
            obj.use_enhanced_handoff = obj.config.get("use_enhanced_handoff", True)
            obj.collect_metrics = obj.config.get("collect_metrics", True)
            obj.max_retry_attempts = obj.config.get("max_retry_attempts", 3)
            obj.retry_count = 0
            obj._agent_registry = {
                "observer": obj._observer,
                "analyst": obj._analyst,
                "verifier": obj._verifier,
                "planner": obj._planner
            }
            return obj
            
        # Apply the mock init
        self.coordinator.__init__ = mock_init
        self.coordinator.__init__(self.coordinator, self.config)
        
        # Update agent registry
        self.coordinator._agent_registry = {
            "observer": self.mock_observer,
            "analyst": self.mock_analyst,
            "verifier": self.mock_verifier,
            "planner": self.mock_planner
        }
        
        # Create a test bug
        self.bug = BugState(
            bug_id="test-bug-123",
            title="Test Bug",
            description="This is a test bug",
            severity="high",
            status="new"
        )
        self.bug.phase = Phase.REPRO
        
        # Create an engine with the bug
        self.engine = TriangulationEngine()
        self.engine.bugs = [self.bug]
        
        # Set up observer response
        observer_report = {
            "summary": "Test bug report",
            "repro_steps": ["Step 1", "Step 2"],
            "evidence": ["Error message"],
            "root_cause": "Logic error",
            "complexity": "medium",
            "status": "SUCCESS"
        }
        self.mock_observer.responses["analyze bug"] = json.dumps(observer_report)
        
        # Set up analyst response
        patch_response = """
        {
            "status": "SUCCESS",
            "message": "Patch generated successfully"
        }
        
        diff --git a/file.py b/file.py
        index 123456..789012 100644
        --- a/file.py
        +++ b/file.py
        @@ -10,7 +10,7 @@
             def test_function():
        -        return None
        +        return "fixed"
        """
        self.mock_analyst.responses["Generate patch"] = patch_response
        
        # Set up verifier response
        verify_response = json.dumps({
            "attempt": "first",
            "status": "PASS",
            "details": "Verification successful"
        })
        self.mock_verifier.responses["Verify fix"] = verify_response
        
        # Set up planner responses
        solution_paths = [
            {
                "path_id": "test-path-1",
                "bug_id": "test-bug-123",
                "actions": [
                    {"type": "analyze", "agent": "observer", "description": "Analyze bug"},
                    {"type": "patch", "agent": "analyst", "description": "Generate patch"},
                    {"type": "verify", "agent": "verifier", "description": "Verify fix"}
                ],
                "fallbacks": []
            },
            {
                "path_id": "test-path-2",
                "bug_id": "test-bug-123",
                "actions": [
                    {"type": "analyze", "agent": "observer", "description": "Analyze bug"},
                    {"type": "patch", "agent": "analyst", "description": "Generate minimal patch"},
                    {"type": "verify", "agent": "verifier", "description": "Verify fix"}
                ],
                "fallbacks": []
            }
        ]
        self.mock_planner.responses["Generate solution paths"] = json.dumps({
            "status": "SUCCESS",
            "solution_paths": solution_paths
        })
        self.mock_planner.responses["Select best path"] = json.dumps({
            "status": "SUCCESS",
            "selected_path_id": "test-path-1"
        })
    
    # Helper methods
    async def run_coordinator_tick(self):
        """Run the coordinator tick."""
        await self.coordinator.coordinate_tick(self.engine)
    
    async def complete_full_cycle(self):
        """Run a full cycle through all phases."""
        # Repro phase
        self.bug.phase = Phase.REPRO
        await self.run_coordinator_tick()
        
        # Patch phase
        self.bug.phase = Phase.PATCH
        await self.run_coordinator_tick()
        
        # Verify phase
        self.bug.phase = Phase.VERIFY
        await self.run_coordinator_tick()
    
    # Tests
    async def test_agent_initialization(self):
        """Test agent initialization."""
        # Run a tick to initialize agents
        await self.run_coordinator_tick()
        
        # Check agents were initialized
        self.assertTrue(self.coordinator._agents_initialized)
        self.assertEqual(len(self.mock_observer.prompts), 1)
        self.assertEqual(len(self.mock_analyst.prompts), 1)
        self.assertEqual(len(self.mock_verifier.prompts), 1)
        self.assertEqual(len(self.mock_planner.prompts), 1)
        
        # Check initialization prompts
        for prompt in self.mock_observer.prompts:
            self.assertIn("INITIALIZATION", prompt)
    
    async def test_family_tree_setup(self):
        """Test family tree setup."""
        # Run a tick to setup the family tree
        await self.run_coordinator_tick()
        
        # Check family tree was updated
        self.assertTrue(self.coordinator._family_tree_updated)
        self.assertIn("Initialize the family tree", self.mock_planner.prompts[1])
    
    async def test_enhanced_handoff_protocol(self):
        """Test the enhanced agent handoff protocol."""
        # Create handoff data
        handoff_data = {
            "bug_id": "test-bug-123",
            "handoff_type": "test",
            "action": "Test handoff",
            "data": {"key": "value"}
        }
        
        # Execute handoff
        status, result = await self.coordinator._agent_handoff(
            "planner", "observer", handoff_data
        )
        
        # Check handoff was successful
        self.assertEqual(status, HandoffStatus.SUCCESS)
        self.assertIn("raw_response", result)
        self.assertIn("HANDOFF: PLANNER → OBSERVER", self.mock_observer.prompts[-1])
        
        # Check metrics were recorded
        self.assertIn("planner_to_observer", self.coordinator._art.handoff_counts)
    
    async def test_path_based_execution(self):
        """Test path-based execution flow."""
        # Setup mock responses for _run_planner_for_paths
        async def mock_run_planner(*args, **kwargs):
            self.coordinator._art.solution_paths = [
                {
                    "path_id": "test-path-1",
                    "bug_id": "test-bug-123",
                    "actions": [
                        {"type": "analyze", "agent": "observer", "description": "Analyze bug"},
                        {"type": "patch", "agent": "analyst", "description": "Generate patch"},
                        {"type": "verify", "agent": "verifier", "description": "Verify fix"}
                    ]
                }
            ]
            self.coordinator._art.current_path_id = "test-path-1"
            return True
        
        # Replace methods with mocks
        self.coordinator._run_planner_for_paths = mock_run_planner
        
        # Setup mock for execute_analysis_action
        async def mock_execute_analysis(*args, **kwargs):
            self.coordinator._art.observer_report = json.dumps({"test": "report"})
            return True
        
        self.coordinator._execute_analysis_action = mock_execute_analysis
        
        # Run coordinator in REPRO phase
        await self.run_coordinator_tick()
        
        # Check path was executed
        self.assertEqual(self.coordinator._art.current_path_id, "test-path-1")
        self.assertEqual(self.coordinator._art.current_action_index, 1)  # Advanced to next action
        self.assertIsNotNone(self.coordinator._art.observer_report)
    
    async def test_metrics_collection(self):
        """Test metrics collection and reporting."""
        # Setup mocks
        async def mock_run_planner(*args, **kwargs):
            self.coordinator._art.solution_paths = [
                {
                    "path_id": "test-path-1",
                    "bug_id": "test-bug-123",
                    "actions": [
                        {"type": "analyze", "agent": "observer", "description": "Analyze bug"}
                    ]
                }
            ]
            self.coordinator._art.current_path_id = "test-path-1"
            return True
        
        async def mock_execute_analysis(*args, **kwargs):
            self.coordinator._art.observer_report = json.dumps({"test": "report"})
            return True
        
        async def mock_report_metrics(*args, **kwargs):
            # Just record that this was called
            self.coordinator._art.add_to_context("metrics_reported", True)
            return True
        
        # Replace methods with mocks
        self.coordinator._run_planner_for_paths = mock_run_planner
        self.coordinator._execute_analysis_action = mock_execute_analysis
        self.coordinator._report_metrics = mock_report_metrics
        
        # Run coordinator in REPRO phase
        await self.run_coordinator_tick()
        
        # Advance to DONE phase to trigger metrics reporting
        self.bug.phase = Phase.DONE
        await self.run_coordinator_tick()
        
        # Check metrics were reported
        self.assertTrue(self.coordinator._art.get_from_context("metrics_reported", False))
        
        # Check metrics were collected
        metrics = self.coordinator._art.get_metrics()
        self.assertIn("total_duration", metrics)
        self.assertIn("action_times", metrics)
        self.assertIn("handoff_counts", metrics)
        self.assertIn("error_counts", metrics)
    
    async def test_fallback_strategy(self):
        """Test fallback strategy execution."""
        # Setup mocks
        async def mock_run_planner(*args, **kwargs):
            self.coordinator._art.solution_paths = [
                {
                    "path_id": "test-path-1",
                    "bug_id": "test-bug-123",
                    "actions": [
                        {"type": "analyze", "agent": "observer", "description": "Analyze bug"}
                    ],
                    "fallbacks": [
                        {
                            "path_id": "fallback-path-1",
                            "actions": [
                                {"type": "patch", "agent": "analyst", "description": "Generate conservative patch"}
                            ]
                        }
                    ]
                }
            ]
            self.coordinator._art.current_path_id = "test-path-1"
            return True
        
        async def mock_execute_analysis(*args, **kwargs):
            # Simulate analysis failure
            return False
        
        async def mock_try_fallback(*args, **kwargs):
            self.coordinator._art.fallbacks_used += 1
            self.coordinator._art.add_to_context("fallback_used", True)
            return True
        
        # Replace methods with mocks
        self.coordinator._run_planner_for_paths = mock_run_planner
        self.coordinator._execute_analysis_action = mock_execute_analysis
        self.coordinator._try_fallback_path = mock_try_fallback
        
        # Run with max retries to trigger fallback
        self.coordinator.retry_count = self.coordinator.max_retry_attempts
        
        # Run coordinator in REPRO phase
        await self.run_coordinator_tick()
        
        # Check fallback was used
        self.assertTrue(self.coordinator._art.get_from_context("fallback_used", False))
        self.assertEqual(self.coordinator._art.fallbacks_used, 1)


class TestAgentCoordinatorIntegration(unittest.TestCase):
    """Integration tests for the AgentCoordinator class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temp dir for any filesystem interactions
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create a config
        self.config = {
            "use_enhanced_handoff": True,
            "collect_metrics": True
        }
        
        # Create a mock coordinator that doesn't initialize real agents
        self.coordinator = AgentCoordinator(self.config)
        
        # Replace the init method to avoid initialization errors
        def mock_init(obj, config=None):
            obj.config = config or {}
            obj._observer = MockAgent("observer")
            obj._analyst = MockAgent("analyst")
            obj._verifier = MockAgent("verifier")
            obj._planner = MockAgent("planner")
            obj._art = _Artefacts()
            obj._last_phase = None
            obj._family_tree_updated = False
            obj._agents_initialized = False
            obj.use_enhanced_handoff = obj.config.get("use_enhanced_handoff", True)
            obj.collect_metrics = obj.config.get("collect_metrics", True)
            obj.max_retry_attempts = obj.config.get("max_retry_attempts", 3)
            obj.retry_count = 0
            obj._agent_registry = {
                "observer": obj._observer,
                "analyst": obj._analyst,
                "verifier": obj._verifier,
                "planner": obj._planner
            }
            return obj
            
        # Apply the mock init
        self.coordinator.__init__ = mock_init
        self.coordinator.__init__(self.coordinator, self.config)
        
        # Create a test bug
        self.bug = BugState(
            bug_id="integration-bug",
            title="Integration Test Bug",
            description="This is an integration test bug",
            severity="high",
            status="new"
        )
        self.bug.phase = Phase.REPRO
        
        # Create an engine with the bug
        self.engine = TriangulationEngine()
        self.engine.bugs = [self.bug]
        
        # Patch the agent ask methods to return mock responses
        def patch_agent(agent, responses):
            async def mock_ask(prompt):
                for key, response in responses.items():
                    if key in prompt:
                        return response
                return "OK"
            agent.ask = mock_ask
        
        # Observer responses
        observer_responses = {
            "INITIALIZATION": "INITIALIZED",
            "analyze bug": json.dumps({
                "summary": "Integration test bug",
                "repro_steps": ["Step 1", "Step 2"],
                "evidence": ["Error in file.py"],
                "root_cause": "Logic error in calculation",
                "complexity": "medium",
                "status": "SUCCESS"
            })
        }
        
        # Analyst responses
        analyst_responses = {
            "INITIALIZATION": "INITIALIZED",
            "Generate patch": """
            {
                "status": "SUCCESS",
                "message": "Patch generated successfully"
            }
            
            diff --git a/file.py b/file.py
            index 123456..789012 100644
            --- a/file.py
            +++ b/file.py
            @@ -10,7 +10,7 @@
                 def integration_test_function():
            -        return None
            +        return "integration_fixed"
            """
        }
        
        # Verifier responses
        verifier_responses = {
            "INITIALIZATION": "INITIALIZED",
            "Verify fix": json.dumps({
                "attempt": "first",
                "status": "PASS",
                "details": "Integration verification successful"
            })
        }
        
        # Planner responses
        planner_responses = {
            "INITIALIZATION": "INITIALIZED",
            "Initialize the family tree": json.dumps({"status": "success"}),
            "Generate solution paths": json.dumps({
                "status": "SUCCESS",
                "solution_paths": [
                    {
                        "path_id": "integration-path-1",
                        "bug_id": "integration-bug",
                        "actions": [
                            {"type": "analyze", "agent": "observer", "description": "Analyze bug"},
                            {"type": "patch", "agent": "analyst", "description": "Generate patch"},
                            {"type": "verify", "agent": "verifier", "description": "Verify fix"}
                        ],
                        "fallbacks": []
                    }
                ]
            }),
            "Select best path": json.dumps({
                "status": "SUCCESS",
                "selected_path_id": "integration-path-1"
            }),
            "progress": json.dumps({
                "status": "SUCCESS",
                "acknowledged": True
            })
        }
        
        # Patch all agents
        patch_agent(self.coordinator._observer, observer_responses)
        patch_agent(self.coordinator._analyst, analyst_responses)
        patch_agent(self.coordinator._verifier, verifier_responses)
        patch_agent(self.coordinator._planner, planner_responses)
    
    def tearDown(self):
        """Clean up after test."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    async def test_full_integration_cycle(self):
        """Test a full integration cycle through all phases."""
        # Add necessary methods that might not be fully implemented yet
        async def run_planner_for_paths(bug):
            """Generate solution paths for a bug."""
            # Call the planner to generate paths
            prompt = f"Generate solution paths for bug {bug.id}"
            response = await self.coordinator._planner.ask(prompt)
            
            try:
                # Parse the response
                data = json.loads(response)
                if data.get("status") == "SUCCESS" and "solution_paths" in data:
                    # Store the paths
                    self.coordinator._art.solution_paths = data["solution_paths"]
                    return True
            except json.JSONDecodeError:
                pass
            
            return False
        
        async def select_solution_path(bug):
            """Select the best solution path for a bug."""
            if not self.coordinator._art.solution_paths:
                return False
            
            # Call the planner to select the best path
            prompt = f"Select best path for bug {bug.id}"
            response = await self.coordinator._planner.ask(prompt)
            
            try:
                # Parse the response
                data = json.loads(response)
                if data.get("status") == "SUCCESS" and "selected_path_id" in data:
                    # Store the selected path ID
                    self.coordinator._art.current_path_id = data["selected_path_id"]
                    return True
            except json.JSONDecodeError:
                pass
            
            # Default to the first path if planner doesn't select one
            self.coordinator._art.current_path_id = self.coordinator._art.solution_paths[0]["path_id"]
            return True
        
        async def notify_planner_of_progress(bug, action_type, success, details):
            """Notify the planner of progress."""
            prompt = f"Progress update for bug {bug.id}: Action {action_type} was {'successful' if success else 'unsuccessful'}"
            await self.coordinator._planner.ask(prompt)
            return True
        
        async def try_fallback_path(bug):
            """Try a fallback path if available."""
            self.coordinator._art.add_to_context("fallback_attempted", True)
            return True
        
        async def report_metrics(bug):
            """Report metrics to the planner."""
            metrics = self.coordinator._art.get_metrics()
            prompt = f"Metrics for bug {bug.id}: {json.dumps(metrics)}"
            await self.coordinator._planner.ask(prompt)
            self.coordinator._art.add_to_context("metrics_reported", True)
            return True
        
        # Add the methods to the coordinator
        self.coordinator._run_planner_for_paths = run_planner_for_paths
        self.coordinator._select_solution_path = select_solution_path
        self.coordinator._notify_planner_of_progress = notify_planner_of_progress
        self.coordinator._try_fallback_path = try_fallback_path
        self.coordinator._report_metrics = report_metrics
        
        # Run through all phases
        # REPRO phase
        await self.coordinator.coordinate_tick(self.engine)
        
        # Check initial state
        self.assertTrue(self.coordinator._agents_initialized)
        self.assertTrue(self.coordinator._family_tree_updated)
        self.assertGreater(len(self.coordinator._art.solution_paths), 0)
        self.assertIsNotNone(self.coordinator._art.current_path_id)
        
        # Manually progress through phases for testing
        # PATCH phase
        self.bug.phase = Phase.PATCH
        await self.coordinator.coordinate_tick(self.engine)
        
        # VERIFY phase
        self.bug.phase = Phase.VERIFY
        await self.coordinator.coordinate_tick(self.engine)
        
        # DONE phase
        self.bug.phase = Phase.DONE
        await self.coordinator.coordinate_tick(self.engine)
        
        # Check final state
        self.assertTrue(self.coordinator._art.get_from_context("metrics_reported", False))
        
        # Check metrics
        metrics = self.coordinator._art.get_metrics()
        self.assertIn("total_duration", metrics)
        self.assertIn("solution_paths_count", metrics)


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


async def run_tests():
    """Run all tests."""
    # Create test loader
    loader = unittest.TestLoader()
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestAgentCoordinator))
    suite.addTest(loader.loadTestsFromTestCase(TestAgentCoordinatorIntegration))
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    print_header("AGENT COORDINATOR TEST SUITE")
    result = runner.run(suite)
    print_header("TEST SUITE COMPLETE")
    
    return result


if __name__ == "__main__":
    # Run tests
    asyncio.run(run_tests())
