import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components.planning_engine import PlanningEngine

class MockRegistry:
    def __init__(self):
        self.components = {}
    def register_component(self, name, component):
        self.components[name] = component
    def get_component(self, name):
        return self.components.get(name)

class TestPlanningEngine(unittest.TestCase):

    def setUp(self):
        self.registry = MockRegistry()
        self.engine = PlanningEngine(self.registry)
        # Create a dummy command lexicon file
        self.lexicon_path = "test_fixwurx_shell_commands.md"
        with open(self.lexicon_path, "w") as f:
            f.write("""
# FixWurx Shell Commands

## `test_command`
**Description:** A test command.
**Arguments:**
- `--arg1` - Description for arg1
- `--arg2` - Description for arg2
**Examples:**
- `test_command --arg1 value1`

## `another_command`
**Description:** Another test command.
**Examples:**
- `another_command`
""")
        self.engine.command_lexicon_path = self.lexicon_path


    def tearDown(self):
        if os.path.exists(self.lexicon_path):
            os.remove(self.lexicon_path)

    def test_parse_command_lexicon(self):
        self.engine._load_command_lexicon()
        lexicon = self.engine.command_lexicon
        
        self.assertIn("test_command", lexicon)
        self.assertEqual(lexicon["test_command"]["description"], "A test command.")
        self.assertEqual(len(lexicon["test_command"]["arguments"]), 2)
        self.assertEqual(lexicon["test_command"]["arguments"][0]["name"], "arg1")
        self.assertEqual(lexicon["test_command"]["arguments"][1]["description"], "Description for arg2")
        self.assertEqual(len(lexicon["test_command"]["examples"]), 1)
        self.assertEqual(lexicon["test_command"]["examples"][0], "test_command --arg1 value1")
        
        self.assertIn("another_command", lexicon)
        self.assertEqual(lexicon["another_command"]["description"], "Another test command.")
        self.assertEqual(len(lexicon["another_command"]["arguments"]), 0)
        self.assertEqual(len(lexicon["another_command"]["examples"]), 1)


if __name__ == '__main__':
    unittest.main()
