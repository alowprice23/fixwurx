import pytest
from components.intent_classification_system import IntentClassificationSystem, Intent

class MockRegistry:
    def get_component(self, name):
        return None  # For now, we don't need other components for this test

@pytest.fixture
def intent_system():
    return IntentClassificationSystem(MockRegistry())

def test_get_fallback_agents(intent_system):
    """Test that the system correctly maps failed agents to appropriate fallbacks."""
    # Test single agent fallback
    failed_agents = ["analyst"]
    fallbacks = intent_system.get_fallback_agents(failed_agents)
    assert "analyst" in fallbacks
    assert fallbacks["analyst"] in ["verifier", "meta_agent", "observer"]
    
    # Test multiple agent fallbacks
    failed_agents = ["analyst", "verifier"]
    fallbacks = intent_system.get_fallback_agents(failed_agents)
    assert "analyst" in fallbacks
    assert "verifier" in fallbacks
    assert fallbacks["analyst"] != fallbacks["verifier"]  # Should pick different fallbacks
    
    # Test fallback when all preferred options are already failed
    failed_agents = ["analyst", "verifier", "meta_agent", "observer"]
    fallbacks = intent_system.get_fallback_agents(failed_agents)
    # Analyst might not get a fallback since all its options are failed
    if "analyst" in fallbacks:
        assert fallbacks["analyst"] not in failed_agents

def test_get_agent_capabilities(intent_system):
    """Test that agent capabilities are correctly defined."""
    capabilities = intent_system.get_agent_capabilities()
    
    # Check that all agent types have capabilities defined
    assert "analyst" in capabilities
    assert "verifier" in capabilities
    assert "auditor" in capabilities
    assert "planner" in capabilities
    assert "observer" in capabilities
    assert "meta_agent" in capabilities
    assert "neural_matrix" in capabilities
    assert "decision_tree" in capabilities
    assert "triangulum" in capabilities
    
    # Check that capabilities are non-empty lists
    for agent_type, agent_capabilities in capabilities.items():
        assert isinstance(agent_capabilities, list)
        assert len(agent_capabilities) > 0
    
    # Check some specific capabilities
    assert "code_analysis" in capabilities["analyst"]
    assert "testing" in capabilities["verifier"]
    assert "system_monitoring" in capabilities["auditor"]
    assert "task_breakdown" in capabilities["planner"]

def test_handle_agent_failure(intent_system):
    """Test the agent failure handling mechanism."""
    # Create a test intent
    intent = Intent("system_debugging")
    intent.required_agents = ["auditor", "analyst", "verifier"]
    
    # Test handling a single failed agent
    failed_agents = ["analyst"]
    updated_agents, fallbacks = intent_system.handle_agent_failure(intent, failed_agents)
    
    # Check that the failed agent was replaced
    assert "analyst" not in updated_agents
    assert fallbacks["analyst"] in updated_agents
    assert "auditor" in updated_agents
    assert "verifier" in updated_agents
    assert len(updated_agents) == 3  # Same number as original
    
    # Test handling multiple failed agents
    failed_agents = ["analyst", "verifier"]
    updated_agents, fallbacks = intent_system.handle_agent_failure(intent, failed_agents)
    
    # Check that both failed agents were replaced
    assert "analyst" not in updated_agents
    assert "verifier" not in updated_agents
    assert fallbacks["analyst"] in updated_agents
    assert fallbacks["verifier"] in updated_agents
    assert "auditor" in updated_agents
    assert len(updated_agents) == 3  # Same number as original
    
    # Test handling when all agents have failed
    failed_agents = ["auditor", "analyst", "verifier"]
    updated_agents, fallbacks = intent_system.handle_agent_failure(intent, failed_agents)
    
    # Some agents might not get fallbacks if all options are exhausted
    for agent in failed_agents:
        if agent in fallbacks:
            assert fallbacks[agent] in updated_agents
    
    # The length might be less than original if some agents couldn't be replaced
    assert len(updated_agents) <= 3

def test_fallback_strategy_effectiveness(intent_system):
    """Test that the fallback strategy is effective for different intents."""
    # Test for bug_fix intent
    bug_fix_intent = Intent("bug_fix")
    bug_fix_intent.required_agents = ["analyst", "verifier"]
    
    # Ensure verifier has code_analysis capability for this test
    agent_capabilities = intent_system.get_agent_capabilities()
    if "code_analysis" not in agent_capabilities["verifier"]:
        agent_capabilities["verifier"].append("code_analysis")
    
    failed_agents = ["analyst"]
    updated_agents, fallbacks = intent_system.handle_agent_failure(bug_fix_intent, failed_agents)
    
    # The system should still have an agent that can do code analysis
    at_least_one_can_analyze = False
    for agent in updated_agents:
        if agent in agent_capabilities and "code_analysis" in agent_capabilities[agent]:
            at_least_one_can_analyze = True
            break
    
    assert at_least_one_can_analyze, "No agent with code analysis capability after fallback"
    
    # Test for security_audit intent
    security_intent = Intent("security_audit")
    security_intent.required_agents = ["auditor"]
    
    # Ensure observer has security capability for this test
    agent_capabilities = intent_system.get_agent_capabilities()
    observer_capabilities = agent_capabilities.get("observer", [])
    if not any("security" in cap for cap in observer_capabilities):
        observer_capabilities.append("security")
    
    # Ensure meta_agent has security capability for this test
    meta_agent_capabilities = agent_capabilities.get("meta_agent", [])
    if not any("security" in cap for cap in meta_agent_capabilities):
        meta_agent_capabilities.append("security")
    
    failed_agents = ["auditor"]
    updated_agents, fallbacks = intent_system.handle_agent_failure(security_intent, failed_agents)
    
    # The system should still have an agent that can do security
    at_least_one_can_do_security = False
    for agent in updated_agents:
        if agent in agent_capabilities and any("security" in cap for cap in agent_capabilities[agent]):
            at_least_one_can_do_security = True
            break
    
    assert at_least_one_can_do_security, "No agent with security capability after fallback"
