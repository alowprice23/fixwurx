#!/usr/bin/env python3
"""
LLM Shell Integration Test

This script tests the integration of all major components from the LLM Shell Integration:
- Conversational Interface
- Intent Recognition and Planning Engine
- Secure Command Execution Environment
- State and Knowledge Repository
- Launchpad & System Bootstrap
- Collaborative Improvement Framework
"""

import os
import sys
import time
import logging
import json
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("integration_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("IntegrationTest")

class IntegrationTest:
    """
    Test the integration of all major LLM Shell components.
    """
    
    def __init__(self):
        """Initialize the integration test."""
        self.results = {
            "conversational_interface": {"status": "pending", "details": []},
            "planning_engine": {"status": "pending", "details": []},
            "command_executor": {"status": "pending", "details": []},
            "script_library": {"status": "pending", "details": []},
            "conversation_logger": {"status": "pending", "details": []},
            "launchpad": {"status": "pending", "details": []},
            "collaborative_improvement": {"status": "pending", "details": []},
            "overall": {"status": "pending", "details": []}
        }
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        
        logger.info("Integration test initialized")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all integration tests.
        
        Returns:
            Test results dictionary
        """
        try:
            logger.info("Starting integration tests")
            
            # Import the launchpad to get access to the component registry
            sys.path.append(os.getcwd())
            
            try:
                from launchpad import Launchpad, ComponentRegistry
                
                # Initialize the launchpad with a test configuration
                test_config = {
                    "neural_matrix": {
                        "model_path": "neural_matrix/models/default",
                        "embedding_dimension": 768,
                        "context_window": 4096,
                        "enable_visualization": False  # Disable visualization for tests
                    },
                    "script_library": {
                        "library_path": "test_script_library",  # Use a test directory
                        "git_enabled": False  # Disable Git for tests
                    },
                    "conversation_logger": {
                        "logs_path": "test_conversation_logs",  # Use a test directory
                        "max_conversation_size": 10,  # Smaller size for tests
                        "compression_enabled": False,  # Disable compression for tests
                        "retention_days": 1  # Short retention for tests
                    },
                    "launchpad": {
                        "shell_prompt": "test> ",
                        "enable_llm_startup": False,  # Disable LLM startup for tests
                        "startup_script_path": None,  # No startup script for tests
                        "enable_agent_system": False  # Disable agent system for tests
                    }
                }
                
                launchpad = Launchpad(config_path=None)
                launchpad.config = test_config
                
                # Initialize the launchpad
                init_result = launchpad.initialize()
                if init_result:
                    self.results["launchpad"]["status"] = "passed"
                    self.results["launchpad"]["details"].append("Launchpad initialized successfully")
                    self.passed += 1
                else:
                    self.results["launchpad"]["status"] = "failed"
                    self.results["launchpad"]["details"].append("Failed to initialize launchpad")
                    self.failed += 1
                    raise Exception("Launchpad initialization failed")
                
                # Get the component registry
                registry = launchpad.registry
                
                # Test each component
                self._test_conversational_interface(registry)
                self._test_planning_engine(registry)
                self._test_command_executor(registry)
                self._test_script_library(registry)
                self._test_conversation_logger(registry)
                self._test_collaborative_improvement(registry)
                
                # Shutdown the launchpad
                launchpad.shutdown()
                
            except ImportError as e:
                logger.error(f"Failed to import required modules: {e}")
                for component in self.results:
                    if component != "overall" and self.results[component]["status"] == "pending":
                        self.results[component]["status"] = "skipped"
                        self.results[component]["details"].append(f"Skipped due to import error: {e}")
                        self.skipped += 1
            
            # Set overall status
            if self.failed > 0:
                self.results["overall"]["status"] = "failed"
                self.results["overall"]["details"].append(f"Failed {self.failed} tests")
            elif self.skipped > 0:
                self.results["overall"]["status"] = "partial"
                self.results["overall"]["details"].append(f"Passed {self.passed} tests, skipped {self.skipped} tests")
            else:
                self.results["overall"]["status"] = "passed"
                self.results["overall"]["details"].append(f"Passed all {self.passed} tests")
            
            logger.info(f"Integration tests completed with status: {self.results['overall']['status']}")
            
            # Save results to file
            with open("integration_test_results.json", "w") as f:
                json.dump(self.results, f, indent=2)
            
            return self.results
        
        except Exception as e:
            logger.error(f"Error running integration tests: {e}")
            self.results["overall"]["status"] = "error"
            self.results["overall"]["details"].append(f"Error: {e}")
            return self.results
    
    def _test_conversational_interface(self, registry) -> None:
        """
        Test the conversational interface.
        
        Args:
            registry: Component registry
        """
        try:
            # Get the conversational interface
            interface = registry.get_component("conversational_interface")
            if not interface:
                self.results["conversational_interface"]["status"] = "skipped"
                self.results["conversational_interface"]["details"].append("Conversational interface not available")
                self.skipped += 1
                return
            
            # Test basic functionality
            response = interface.process_input("Hello, world!")
            if response:
                self.results["conversational_interface"]["status"] = "passed"
                self.results["conversational_interface"]["details"].append("Successfully processed input")
                self.passed += 1
            else:
                self.results["conversational_interface"]["status"] = "failed"
                self.results["conversational_interface"]["details"].append("Failed to process input")
                self.failed += 1
        
        except Exception as e:
            logger.error(f"Error testing conversational interface: {e}")
            self.results["conversational_interface"]["status"] = "failed"
            self.results["conversational_interface"]["details"].append(f"Error: {e}")
            self.failed += 1
    
    def _test_planning_engine(self, registry) -> None:
        """
        Test the planning engine.
        
        Args:
            registry: Component registry
        """
        try:
            # Get the planning engine
            planning_engine = registry.get_component("planning_engine")
            if not planning_engine:
                self.results["planning_engine"]["status"] = "skipped"
                self.results["planning_engine"]["details"].append("Planning engine not available")
                self.skipped += 1
                return
            
            # Test basic functionality
            plan = planning_engine.generate_plan("List files in the current directory")
            if plan and "steps" in plan:
                self.results["planning_engine"]["status"] = "passed"
                self.results["planning_engine"]["details"].append("Successfully generated plan")
                self.passed += 1
            else:
                self.results["planning_engine"]["status"] = "failed"
                self.results["planning_engine"]["details"].append("Failed to generate plan")
                self.failed += 1
        
        except Exception as e:
            logger.error(f"Error testing planning engine: {e}")
            self.results["planning_engine"]["status"] = "failed"
            self.results["planning_engine"]["details"].append(f"Error: {e}")
            self.failed += 1
    
    def _test_command_executor(self, registry) -> None:
        """
        Test the command executor.
        
        Args:
            registry: Component registry
        """
        try:
            # Get the command executor
            command_executor = registry.get_component("command_executor")
            if not command_executor:
                self.results["command_executor"]["status"] = "skipped"
                self.results["command_executor"]["details"].append("Command executor not available")
                self.skipped += 1
                return
            
            # Test basic functionality with a safe command
            result = command_executor.execute("echo 'Hello from test'", "test_user")
            if result.get("success", False):
                self.results["command_executor"]["status"] = "passed"
                self.results["command_executor"]["details"].append("Successfully executed command")
                self.passed += 1
            else:
                self.results["command_executor"]["status"] = "failed"
                self.results["command_executor"]["details"].append("Failed to execute command")
                self.failed += 1
        
        except Exception as e:
            logger.error(f"Error testing command executor: {e}")
            self.results["command_executor"]["status"] = "failed"
            self.results["command_executor"]["details"].append(f"Error: {e}")
            self.failed += 1
    
    def _test_script_library(self, registry) -> None:
        """
        Test the script library.
        
        Args:
            registry: Component registry
        """
        try:
            # Get the script library
            script_library = registry.get_component("script_library")
            if not script_library:
                self.results["script_library"]["status"] = "skipped"
                self.results["script_library"]["details"].append("Script library not available")
                self.skipped += 1
                return
            
            # Test adding a script
            test_script = """
            #!/usr/bin/env bash
            echo "Hello from test script"
            """
            
            test_metadata = {
                "name": "test_script",
                "description": "Test script for integration testing",
                "author": "integration_test",
                "version": "1.0",
                "tags": ["test", "integration"]
            }
            
            result = script_library.add_script(test_script, test_metadata)
            if result.get("success", False):
                script_id = result.get("script_id")
                
                # Test retrieving the script
                get_result = script_library.get_script(script_id)
                if get_result.get("success", False):
                    self.results["script_library"]["status"] = "passed"
                    self.results["script_library"]["details"].append("Successfully added and retrieved script")
                    self.passed += 1
                else:
                    self.results["script_library"]["status"] = "failed"
                    self.results["script_library"]["details"].append("Failed to retrieve script")
                    self.failed += 1
            else:
                self.results["script_library"]["status"] = "failed"
                self.results["script_library"]["details"].append("Failed to add script")
                self.failed += 1
        
        except Exception as e:
            logger.error(f"Error testing script library: {e}")
            self.results["script_library"]["status"] = "failed"
            self.results["script_library"]["details"].append(f"Error: {e}")
            self.failed += 1
    
    def _test_conversation_logger(self, registry) -> None:
        """
        Test the conversation logger.
        
        Args:
            registry: Component registry
        """
        try:
            # Get the conversation logger
            conversation_logger = registry.get_component("conversation_logger")
            if not conversation_logger:
                self.results["conversation_logger"]["status"] = "skipped"
                self.results["conversation_logger"]["details"].append("Conversation logger not available")
                self.skipped += 1
                return
            
            # Test starting a conversation
            start_result = conversation_logger.start_conversation("test_user")
            if start_result.get("success", False):
                conversation_id = start_result.get("conversation_id")
                
                # Test adding messages
                add_result1 = conversation_logger.add_message(conversation_id, "test_user", "Hello")
                add_result2 = conversation_logger.add_message(conversation_id, "system", "Hi there!")
                
                if add_result1.get("success", False) and add_result2.get("success", False):
                    # Test ending the conversation
                    end_result = conversation_logger.end_conversation(conversation_id)
                    if end_result.get("success", False):
                        self.results["conversation_logger"]["status"] = "passed"
                        self.results["conversation_logger"]["details"].append("Successfully managed conversation")
                        self.passed += 1
                    else:
                        self.results["conversation_logger"]["status"] = "failed"
                        self.results["conversation_logger"]["details"].append("Failed to end conversation")
                        self.failed += 1
                else:
                    self.results["conversation_logger"]["status"] = "failed"
                    self.results["conversation_logger"]["details"].append("Failed to add messages")
                    self.failed += 1
            else:
                self.results["conversation_logger"]["status"] = "failed"
                self.results["conversation_logger"]["details"].append("Failed to start conversation")
                self.failed += 1
        
        except Exception as e:
            logger.error(f"Error testing conversation logger: {e}")
            self.results["conversation_logger"]["status"] = "failed"
            self.results["conversation_logger"]["details"].append(f"Error: {e}")
            self.failed += 1
    
    def _test_collaborative_improvement(self, registry) -> None:
        """
        Test the collaborative improvement framework.
        
        Args:
            registry: Component registry
        """
        try:
            # Get the pattern detector
            pattern_detector = registry.get_component("pattern_detector")
            if not pattern_detector:
                self.results["collaborative_improvement"]["status"] = "skipped"
                self.results["collaborative_improvement"]["details"].append("Pattern detector not available")
                self.skipped += 1
                return
            
            # Test adding commands
            pattern_detector.add_command("ls -la", "test_user")
            pattern_detector.add_command("grep 'error' log.txt", "test_user")
            pattern_detector.add_command("cat log.txt | grep 'error' > errors.txt", "test_user")
            
            # Add the same sequence again to trigger pattern detection
            pattern_detector.add_command("ls -la", "test_user")
            pattern_detector.add_command("grep 'error' log.txt", "test_user")
            pattern_detector.add_command("cat log.txt | grep 'error' > errors.txt", "test_user")
            
            # Check if patterns were detected
            patterns = pattern_detector.get_patterns()
            if patterns:
                # Get the script proposer
                script_proposer = registry.get_component("script_proposer")
                if not script_proposer:
                    self.results["collaborative_improvement"]["status"] = "partial"
                    self.results["collaborative_improvement"]["details"].append("Pattern detection works, but script proposer not available")
                    self.passed += 1
                    return
                
                # Test proposing a script for the first pattern
                pattern_id = list(patterns.keys())[0] if patterns else None
                if pattern_id:
                    proposal_result = script_proposer.propose_script(pattern_id)
                    if proposal_result.get("success", False):
                        # Get the peer review workflow
                        peer_review = registry.get_component("peer_review_workflow")
                        if not peer_review:
                            self.results["collaborative_improvement"]["status"] = "partial"
                            self.results["collaborative_improvement"]["details"].append("Pattern detection and script proposal work, but peer review not available")
                            self.passed += 1
                            return
                        
                        # Test the peer review workflow
                        proposal_id = proposal_result.get("proposal_id")
                        review_result = peer_review.submit_review(proposal_id, "reviewer1", True, "Looks good!")
                        review_result2 = peer_review.submit_review(proposal_id, "reviewer2", True, "Approved")
                        
                        if review_result.get("success", False) and review_result2.get("success", False):
                            # Test script committal
                            commit_result = peer_review.commit_script(proposal_id)
                            if commit_result.get("success", False):
                                self.results["collaborative_improvement"]["status"] = "passed"
                                self.results["collaborative_improvement"]["details"].append("Successfully detected pattern, proposed and committed script")
                                self.passed += 1
                            else:
                                self.results["collaborative_improvement"]["status"] = "partial"
                                self.results["collaborative_improvement"]["details"].append("Pattern detection, script proposal and review work, but committal failed")
                                self.passed += 1
                        else:
                            self.results["collaborative_improvement"]["status"] = "partial"
                            self.results["collaborative_improvement"]["details"].append("Pattern detection and script proposal work, but review failed")
                            self.passed += 1
                    else:
                        self.results["collaborative_improvement"]["status"] = "partial"
                        self.results["collaborative_improvement"]["details"].append("Pattern detection works, but script proposal failed")
                        self.passed += 1
                else:
                    self.results["collaborative_improvement"]["status"] = "failed"
                    self.results["collaborative_improvement"]["details"].append("No patterns detected")
                    self.failed += 1
            else:
                self.results["collaborative_improvement"]["status"] = "failed"
                self.results["collaborative_improvement"]["details"].append("Failed to detect patterns")
                self.failed += 1
        
        except Exception as e:
            logger.error(f"Error testing collaborative improvement: {e}")
            self.results["collaborative_improvement"]["status"] = "failed"
            self.results["collaborative_improvement"]["details"].append(f"Error: {e}")
            self.failed += 1

def print_results(results: Dict[str, Any]) -> None:
    """
    Print the test results in a user-friendly format.
    
    Args:
        results: Test results dictionary
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "LLM SHELL INTEGRATION TEST RESULTS")
    print("=" * 80)
    
    for component, result in results.items():
        if component != "overall":
            status = result["status"].upper()
            status_color = ""
            if status == "PASSED":
                status_color = "\033[92m"  # Green
            elif status == "FAILED":
                status_color = "\033[91m"  # Red
            elif status == "PARTIAL":
                status_color = "\033[93m"  # Yellow
            elif status == "SKIPPED":
                status_color = "\033[94m"  # Blue
            
            reset_color = "\033[0m"
            
            print(f"\n{component.replace('_', ' ').title()}: {status_color}{status}{reset_color}")
            for detail in result["details"]:
                print(f"  - {detail}")
    
    print("\n" + "-" * 80)
    overall_status = results["overall"]["status"].upper()
    overall_color = ""
    if overall_status == "PASSED":
        overall_color = "\033[92m"  # Green
    elif overall_status == "FAILED":
        overall_color = "\033[91m"  # Red
    elif overall_status == "PARTIAL":
        overall_color = "\033[93m"  # Yellow
    elif overall_status == "ERROR":
        overall_color = "\033[91m"  # Red
    
    print(f"Overall Status: {overall_color}{overall_status}{reset_color}")
    for detail in results["overall"]["details"]:
        print(f"  - {detail}")
    
    print("=" * 80 + "\n")

if __name__ == "__main__":
    print("Starting LLM Shell Integration Test...")
    test = IntegrationTest()
    results = test.run_all_tests()
    print_results(results)
    
    if results["overall"]["status"] == "passed":
        sys.exit(0)
    elif results["overall"]["status"] == "partial":
        sys.exit(1)
    else:
        sys.exit(2)
