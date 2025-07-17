import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components.launchpad import Launchpad
import components.launchpad as launchpad
import components.conversational_interface as conversational_interface
import components.command_executor as command_executor
import components.planning_engine as planning_engine
import components.llm_client_real as llm_client_real
import components.ui_commands as ui_commands

def reset_singletons():
    # This is a bit of a hack, but it's necessary for testing
    # to ensure that each test gets a fresh set of components.
    launchpad._instance = None
    conversational_interface._instance = None
    command_executor._instance = None
    planning_engine._instance = None
    llm_client_real._instance = None

class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        reset_singletons()
        # The launchpad will initialize all components, including the conversational interface
        self.launchpad = Launchpad()
        self.launchpad.initialize()
        self.conversational_interface = self.launchpad.registry.get_component("conversational_interface")

    def tearDown(self):
        self.launchpad.shutdown()
        if os.path.exists("test.txt"):
            os.remove("test.txt")
        if os.path.exists("test_dir"):
            os.rmdir("test_dir")


    def test_echo_command(self):
        response = self.conversational_interface.process_input("!echo 'hello world'")
        self.assertIn("hello world", response)

    def test_mkdir_command(self):
        response = self.conversational_interface.process_input("!mkdir test_dir")
        self.assertTrue(os.path.isdir("test_dir"))

    def test_ui_command(self):
        response = self.conversational_interface.process_input("!dashboard:update")
        self.assertIn("Dashboard updated successfully", response)

    def test_conversational_query(self):
        # This test will fail if the real LLM is not configured
        # It's more of an integration test
        response = self.conversational_interface.process_input("create a new file called test.txt")
        # We can't easily assert the response from the LLM, 
        # but we can check that it doesn't error out
        self.assertIsNotNone(response)

if __name__ == '__main__':
    unittest.main()
