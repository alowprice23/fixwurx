import unittest
import os
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components.launchpad import Launchpad
import components.launchpad as launchpad
import components.conversational_interface as conversational_interface
import components.command_executor as command_executor
import components.planning_engine as planning_engine
import components.llm_client_real as llm_client_real
import components.ui_commands as ui_commands
import components.decision_tree_integration as decision_tree_integration

def reset_singletons():
    launchpad._instance = None
    conversational_interface._instance = None
    command_executor._instance = None
    planning_engine._instance = None
    llm_client_real._instance = None
    decision_tree_integration._instance = None

class TestDecisionTreeIntegration(unittest.TestCase):

    def setUp(self):
        reset_singletons()
        self.launchpad = Launchpad()
        self.launchpad.initialize()
        self.planning_engine = self.launchpad.registry.get_component("planning_engine")
        # Create a dummy buggy file
        self.buggy_file_path = "sample_buggy.py"
        with open(self.buggy_file_path, "w") as f:
            f.write("""
def buggy_function():
    # This function has a bug
    a = 1
    b = 0
    return a / b

buggy_function()
""")

    def tearDown(self):
        self.launchpad.shutdown()
        if os.path.exists(self.buggy_file_path):
            os.remove(self.buggy_file_path)
        # Clean up any patch files
        for f in os.listdir("."):
            if f.endswith(".patch"):
                os.remove(f)


    def test_bug_fix_goal_triggers_decision_tree(self):
        goal = f"fix a bug in {self.buggy_file_path} (py)"
        plan = self.planning_engine.generate_plan(goal)
        self.assertTrue(plan.get("success"))
        self.assertIn("Fix applied", plan["steps"][0]["description"])

    def test_syntax_error_fix(self):
        with open(self.buggy_file_path, "w") as f:
            f.write("print 'hello world'")
        goal = f"fix a bug in {self.buggy_file_path} (py)"
        plan = self.planning_engine.generate_plan(goal)
        self.assertTrue(plan.get("success"))
        self.assertIn("Fix applied", plan["steps"][0]["description"])
        with open(self.buggy_file_path, "r") as f:
            content = f.read()
            self.assertIn("print('hello world')", content)

    def test_runtime_error_fix(self):
        with open(self.buggy_file_path, "w") as f:
            f.write("""
def buggy_function():
    a = {}
    return a["b"]

buggy_function()
""")
        goal = f"fix a bug in {self.buggy_file_path} (py)"
        plan = self.planning_engine.generate_plan(goal)
        self.assertTrue(plan.get("success"))
        self.assertIn("Fix applied", plan["steps"][0]["description"])
        with open(self.buggy_file_path, "r") as f:
            content = f.read()
            self.assertIn("a.get(\"b\")", content)


if __name__ == '__main__':
    # Set up a logger for the test to capture the output
    logging.basicConfig(level=logging.INFO)
    unittest.main()
