import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components.launchpad import Launchpad
from components.state_manager import State
import components.launchpad as launchpad
import components.conversational_interface as conversational_interface
import components.command_executor as command_executor
import components.planning_engine as planning_engine
import components.llm_client_real as llm_client_real
import components.ui_commands as ui_commands
import components.decision_tree_integration as decision_tree_integration
import components.state_manager as state_manager

def reset_singletons():
    launchpad._instance = None
    conversational_interface._instance = None
    command_executor._instance = None
    planning_engine._instance = None
    llm_client_real._instance = None
    decision_tree_integration._instance = None
    state_manager._instance = None

class TestStateManagement(unittest.TestCase):

    def setUp(self):
        reset_singletons()
        self.launchpad = Launchpad()
        self.launchpad.initialize()
        self.conversational_interface = self.launchpad.registry.get_component("conversational_interface")
        self.state_manager = self.launchpad.registry.get_component("state_manager")

    def tearDown(self):
        self.launchpad.shutdown()

    def test_greeting_sets_idle_state(self):
        self.conversational_interface.process_input("hi")
        self.assertEqual(self.state_manager.get_state(), State.IDLE)

    def test_bug_fix_sets_awaiting_feedback_state(self):
        # This will fail if the file doesn't exist, but we are just testing the state change
        self.conversational_interface.process_input("fix a bug in a file")
        self.assertEqual(self.state_manager.get_state(), State.AWAITING_FEEDBACK)

if __name__ == '__main__':
    unittest.main()
