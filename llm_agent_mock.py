#!/usr/bin/env python3
"""
LLM Agent Mock Implementation

This module provides mock implementations of agent classes that call the OpenAI API.
These mocks are used for testing the LLM integration without depending on the full agent system.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("llm_mock.log"), logging.StreamHandler()]
)
logger = logging.getLogger("LLMMockAgent")

class OpenAIClient:
    """Simple OpenAI client for agent testing."""
    
    # Class-level tracking for API calls
    api_called = False
    last_call_agent = None
    
    def __init__(self):
        """Initialize the OpenAI client."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            logger.error("OpenAI Python package not installed")
            raise ImportError("OpenAI Python package not installed")
    
    def generate_text(self, prompt: str, agent_type: str = "unknown", max_tokens: int = 200) -> str:
        """
        Generate text using the OpenAI API.
        
        Args:
            prompt: The prompt to send to the OpenAI API
            agent_type: Type of agent making the call
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Generated text
        """
        try:
            # Set the tracking flags
            OpenAIClient.api_called = True
            OpenAIClient.last_call_agent = agent_type
            
            print(f"✅ OpenAI API called by {agent_type.upper()} agent!")
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {e}"

class BaseMockAgent:
    """Base mock agent class."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.openai_client = OpenAIClient()
        self.agent_type = "base"
    
    def call_llm(self, prompt: str) -> str:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response
        """
        logger.info(f"{self.agent_type.upper()} agent calling LLM with prompt: {prompt[:50]}...")
        return self.openai_client.generate_text(prompt, self.agent_type)

class MetaAgentMock(BaseMockAgent):
    """Mock implementation of the Meta Agent."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Meta Agent mock."""
        super().__init__(config)
        self.agent_type = "meta"
    
    def coordinate_agents(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate agents to accomplish a task.
        
        Args:
            task: Task to accomplish
            
        Returns:
            Result of the task
        """
        prompt = f"As a Meta Agent, I need to coordinate other agents to accomplish this task: {task}. How should I approach this?"
        response = self.call_llm(prompt)
        
        return {
            "task": task,
            "plan": response,
            "status": "coordinated"
        }
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a task.
        
        Args:
            task: Task to process
            
        Returns:
            Result of processing the task
        """
        prompt = f"As a Meta Agent, I need to process this task: {task}. How should I handle it?"
        response = self.call_llm(prompt)
        
        return {
            "task": task,
            "result": response,
            "status": "processed"
        }

class PlannerAgentMock(BaseMockAgent):
    """Mock implementation of the Planner Agent."""
    
    def __init__(self, config: Dict[str, Any], agent_memory: Any = None):
        """Initialize the Planner Agent mock."""
        super().__init__(config)
        self.agent_type = "planner"
        self.agent_memory = agent_memory
    
    def generate_solution_paths(self, bug_id: str) -> List[Dict[str, Any]]:
        """
        Generate solution paths for a bug.
        
        Args:
            bug_id: ID of the bug
            
        Returns:
            List of solution paths
        """
        prompt = f"As a Planner Agent, I need to generate solution paths for bug {bug_id}. What are some possible solutions?"
        response = self.call_llm(prompt)
        
        # Create a simple solution path from the response
        return [{
            "path_id": f"{bug_id}-path-1",
            "bug_id": bug_id,
            "actions": [
                {"type": "analyze", "agent": "observer", "description": f"Analyze bug {bug_id}"},
                {"type": "patch", "agent": "analyst", "description": f"Patch bug {bug_id}"},
                {"type": "verify", "agent": "verifier", "description": f"Verify bug {bug_id}"}
            ],
            "reasoning": response
        }]
    
    def plan_solution(self, bug: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan a solution for a bug.
        
        Args:
            bug: Bug information
            
        Returns:
            Solution plan
        """
        prompt = f"As a Planner Agent, I need to plan a solution for this bug: {bug}. What steps should I take?"
        response = self.call_llm(prompt)
        
        return {
            "bug_id": bug.get("id", "unknown"),
            "plan": response,
            "status": "planned"
        }

class ObserverAgentMock(BaseMockAgent):
    """Mock implementation of the Observer Agent."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Observer Agent mock."""
        super().__init__(config)
        self.agent_type = "observer"
    
    def analyze_bug(self, bug_id: str) -> Dict[str, Any]:
        """
        Analyze a bug.
        
        Args:
            bug_id: ID of the bug
            
        Returns:
            Analysis results
        """
        prompt = f"As an Observer Agent, I need to analyze bug {bug_id}. What should I look for?"
        response = self.call_llm(prompt)
        
        return {
            "bug_id": bug_id,
            "analysis": response,
            "status": "analyzed"
        }
    
    def observe(self, bug: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observe a bug.
        
        Args:
            bug: Bug information
            
        Returns:
            Observation results
        """
        prompt = f"As an Observer Agent, I need to observe this bug: {bug}. What do I notice?"
        response = self.call_llm(prompt)
        
        return {
            "bug_id": bug.get("id", "unknown"),
            "observations": response,
            "status": "observed"
        }

class AnalystAgentMock(BaseMockAgent):
    """Mock implementation of the Analyst Agent."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Analyst Agent mock."""
        super().__init__(config)
        self.agent_type = "analyst"
    
    def generate_patch(self, bug_id: str) -> Dict[str, Any]:
        """
        Generate a patch for a bug.
        
        Args:
            bug_id: ID of the bug
            
        Returns:
            Generated patch
        """
        prompt = f"As an Analyst Agent, I need to generate a patch for bug {bug_id}. What code changes should I make?"
        response = self.call_llm(prompt)
        
        return {
            "bug_id": bug_id,
            "patch": response,
            "status": "generated"
        }
    
    def analyze(self, bug: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a bug.
        
        Args:
            bug: Bug information
            
        Returns:
            Analysis results
        """
        prompt = f"As an Analyst Agent, I need to analyze this bug: {bug}. What's the root cause?"
        response = self.call_llm(prompt)
        
        return {
            "bug_id": bug.get("id", "unknown"),
            "root_cause": response,
            "status": "analyzed"
        }

class VerifierAgentMock(BaseMockAgent):
    """Mock implementation of the Verifier Agent."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Verifier Agent mock."""
        super().__init__(config)
        self.agent_type = "verifier"
    
    def verify_patch(self, bug_id: str, patch_id: str) -> Dict[str, Any]:
        """
        Verify a patch.
        
        Args:
            bug_id: ID of the bug
            patch_id: ID of the patch
            
        Returns:
            Verification results
        """
        prompt = f"As a Verifier Agent, I need to verify patch {patch_id} for bug {bug_id}. How do I verify it works?"
        response = self.call_llm(prompt)
        
        return {
            "bug_id": bug_id,
            "patch_id": patch_id,
            "verification": response,
            "status": "verified"
        }
    
    def verify(self, bug: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a patch.
        
        Args:
            bug: Bug information
            patch: Patch information
            
        Returns:
            Verification results
        """
        prompt = f"As a Verifier Agent, I need to verify this patch: {patch} for this bug: {bug}. Does it fix the issue?"
        response = self.call_llm(prompt)
        
        return {
            "bug_id": bug.get("id", "unknown"),
            "patch_id": patch.get("id", "unknown"),
            "result": response,
            "status": "verified"
        }

class LaunchpadAgentMock(BaseMockAgent):
    """Mock implementation of the Launchpad Agent."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Launchpad Agent mock."""
        super().__init__(config)
        self.agent_type = "launchpad"
    
    def initialize_agents(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize agents for a task.
        
        Args:
            task: Task to initialize agents for
            
        Returns:
            Initialization results
        """
        prompt = f"As a Launchpad Agent, I need to initialize agents for this task: {task}. Which agents should I initialize?"
        response = self.call_llm(prompt)
        
        return {
            "task": task,
            "agents": response,
            "status": "initialized"
        }
    
    def launch(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Launch a task.
        
        Args:
            task: Task to launch
            
        Returns:
            Launch results
        """
        prompt = f"As a Launchpad Agent, I need to launch this task: {task}. How should I proceed?"
        response = self.call_llm(prompt)
        
        return {
            "task": task,
            "launch_plan": response,
            "status": "launched"
        }

class AuditorAgentMock(BaseMockAgent):
    """Mock implementation of the Auditor Agent."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Auditor Agent mock."""
        super().__init__(config)
        self.agent_type = "auditor"
    
    def audit_system_component(self, component: str) -> Dict[str, Any]:
        """
        Audit a system component.
        
        Args:
            component: Component to audit
            
        Returns:
            Audit results
        """
        prompt = f"As an Auditor Agent, I need to audit the system component '{component}'. What checks should I perform?"
        response = self.call_llm(prompt)
        
        return {
            "component": component,
            "audit": response,
            "status": "audited"
        }
    
    def audit(self, component: str) -> Dict[str, Any]:
        """
        Audit a component.
        
        Args:
            component: Component to audit
            
        Returns:
            Audit results
        """
        prompt = f"As an Auditor Agent, I need to audit '{component}'. What compliance checks should I perform?"
        response = self.call_llm(prompt)
        
        return {
            "component": component,
            "compliance_check": response,
            "status": "audited"
        }

# Map of agent types to their mock implementations
AGENT_MOCKS = {
    "meta": MetaAgentMock,
    "planner": PlannerAgentMock,
    "observer": ObserverAgentMock,
    "analyst": AnalystAgentMock,
    "verifier": VerifierAgentMock,
    "launchpad": LaunchpadAgentMock,
    "auditor": AuditorAgentMock
}

def get_mock_agent(agent_type: str, config: Dict[str, Any] = None) -> BaseMockAgent:
    """
    Get a mock agent of the specified type.
    
    Args:
        agent_type: Type of agent to get
        config: Configuration for the agent
        
    Returns:
        Mock agent
    """
    config = config or {}
    
    if agent_type not in AGENT_MOCKS:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Special case for planner agent which needs agent_memory
    if agent_type == "planner":
        class MockAgentMemory:
            def store(self, key, value):
                pass
            
            def retrieve(self, key):
                return None
        
        return AGENT_MOCKS[agent_type](config, MockAgentMemory())
    
    return AGENT_MOCKS[agent_type](config)

# Test all mock agents if run directly
if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        exit(1)
    
    # Test each agent type
    for agent_type in AGENT_MOCKS:
        print(f"Testing {agent_type.upper()} agent...")
        try:
            # Get mock agent
            agent = get_mock_agent(agent_type)
            
            # Call appropriate method based on agent type
            if agent_type == "meta":
                result = agent.coordinate_agents({"task": "test task"})
            elif agent_type == "planner":
                result = agent.generate_solution_paths("bug-001")
            elif agent_type == "observer":
                result = agent.analyze_bug("bug-001")
            elif agent_type == "analyst":
                result = agent.generate_patch("bug-001")
            elif agent_type == "verifier":
                result = agent.verify_patch("bug-001", "patch-001")
            elif agent_type == "launchpad":
                result = agent.initialize_agents({"task": "test task"})
            elif agent_type == "auditor":
                result = agent.audit_system_component("test component")
            
            # Print result
            print(f"SUCCESS: {agent_type.upper()} agent successfully called the OpenAI API")
            print(f"Result: {result}\n")
        except Exception as e:
            print(f"ERROR: Failed to test {agent_type.upper()} agent: {e}\n")
