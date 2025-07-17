import pytest
import re
from components.intent_classification_system import IntentClassificationSystem, Intent

class MockRegistry:
    def get_component(self, name):
        return None # For now, we don't need other components for this test

@pytest.fixture
def intent_system():
    return IntentClassificationSystem(MockRegistry())

def test_load_patterns(intent_system):
    assert "agent_introspection" in intent_system.pattern_matchers
    assert "file_modification" in intent_system.pattern_matchers
    assert "command_execution" in intent_system.pattern_matchers
    assert "file_access" in intent_system.pattern_matchers
    assert "decision_tree" in intent_system.pattern_matchers
    assert "script_execution" in intent_system.pattern_matchers
    assert "rotate_credentials" in intent_system.pattern_matchers

def test_load_semantic_keywords(intent_system):
    assert "bug_fix" in intent_system.semantic_keywords
    assert "system_debugging" in intent_system.semantic_keywords
    assert "performance_optimization" in intent_system.semantic_keywords

def test_classify_intent_pattern_matching(intent_system):
    intent = intent_system.classify_intent("run the command `ls -l`", {})
    assert intent.type == "command_execution"
    assert intent.parameters["command"] == "ls"
    assert intent.parameters["args"] == ["-l"]
    assert intent.execution_path == "direct"

    intent = intent_system.classify_intent("change file my_file.txt in /path/to/file", {})
    assert intent.type == "file_modification"
    assert intent.parameters["path"] == "my_file.txt"
    assert intent.execution_path == "direct"

def test_classify_intent_semantic_understanding(intent_system):
    # Test for bug_fix intent
    intent = intent_system.classify_intent("I found a bug in the code", {})
    assert intent.type == "bug_fix"
    assert intent.execution_path == "agent_collaboration"

    # Test for performance_optimization intent
    intent = intent_system.classify_intent("optimize the system performance", {})
    assert intent.type == "performance_optimization"
    assert intent.execution_path == "agent_collaboration"

    # Test for security_audit intent
    intent = intent_system.classify_intent("perform a security audit", {})
    assert intent.type == "security_audit"
    assert intent.execution_path == "agent_collaboration"

def test_classify_intent_generic_fallback(intent_system):
    intent = intent_system.classify_intent("This is a random query", {})
    assert intent.type == "generic"
    assert intent.execution_path == "planning" # Default for generic intent

def test_determine_required_agents(intent_system):
    intent = intent_system.classify_intent("fix the bug", {})
    assert "analyst" in intent.required_agents
    assert "verifier" in intent.required_agents

    intent = intent_system.classify_intent("debug the system", {})
    assert "auditor" in intent.required_agents
    assert "analyst" in intent.required_agents
    assert "verifier" in intent.required_agents

    intent = intent_system.classify_intent("rotate my credentials", {})
    assert intent.required_agents == ["auditor"]

def test_determine_execution_path(intent_system):
    intent = intent_system.classify_intent("read file config.txt", {})
    assert intent.execution_path == "direct"

    intent = intent_system.classify_intent("run the decision tree", {})
    assert intent.execution_path == "decision_tree"

    intent = intent_system.classify_intent("how are you doing", {})
    assert intent.execution_path == "direct" # agent_introspection is direct

def test_classify_intent_with_context_awareness(intent_system):
    """
    Test context-aware intent classification where the system uses conversation
    history to understand follow-up queries.
    """
    # Simulate conversation history
    history = [
        {"role": "user", "content": "show me the file /etc/config.conf"},
        {"role": "system", "content": "File: /etc/config.conf\n\n[...content...]"},
        {"role": "user", "content": "now change it to use port 8080"}
    ]
    
    # The context should contain the conversation history
    context = {"history": history}
    
    # Classify the follow-up intent
    intent = intent_system.classify_intent("now change it to use port 8080", context)
    
    # The system should correctly identify this as a file modification intent
    assert intent.type == "file_modification"
    
    # It should also correctly identify the file path from the context
    assert intent.parameters.get("path") == "/etc/config.conf"
    
    # The execution path should be direct
    assert intent.execution_path == "direct"
    
    # Check that the context reference was correctly identified
    assert "file_references" in intent.context_references
    assert intent.context_references["file_references"][0] == "/etc/config.conf"
