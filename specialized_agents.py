"""
specialized_agents.py
───────────────────────
Implementation of specialized agent classes used by the agent coordinator system.

This module provides specialized agent implementations for different roles in the
bug fixing and verification process.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

# System prompts for agents
_OBSERVER_PROMPT = """
You are an Observer Agent specialized in analyzing and understanding software bugs.
Your responsibilities include:
1. Carefully analyzing bug reports and error logs
2. Identifying root causes of issues
3. Documenting reproducible steps
4. Classifying bugs by severity and type
5. Providing detailed analysis to the Analyst Agent

Focus on being thorough and accurate in your analysis. Provide evidence for your conclusions.
"""

_ANALYST_PROMPT = """
You are an Analyst Agent specialized in developing fixes for software bugs.
Your responsibilities include:
1. Reviewing the Observer's analysis
2. Designing solutions that address the root cause
3. Implementing fixes with minimal code changes
4. Testing the changes in isolation
5. Providing detailed patch information to the Verifier

Focus on creating robust, maintainable fixes that fully address the issue.
"""

_VERIFIER_PROMPT = """
You are a Verifier Agent specialized in testing and validating bug fixes.
Your responsibilities include:
1. Applying patches in test environments
2. Running comprehensive test suites
3. Verifying the fix resolves the original issue
4. Checking for regressions or new issues
5. Providing verification results to the Planner

Focus on thorough validation and ensuring quality standards are met.
"""

_PLANNER_PROMPT = """
You are a Planner Agent specialized in coordinating the bug resolution process.
Your responsibilities include:
1. Directing the overall bug resolution workflow
2. Prioritizing bugs based on severity and impact
3. Assigning tasks to specialized agents
4. Managing the state transitions between phases
5. Making strategic decisions when issues arise

Focus on efficient coordination and optimal resource allocation.
"""

class BaseAgent:
    """Base class for all specialized agents."""
    
    def __init__(self):
        """Initialize the base agent."""
        self.name = "base"
        self.context = {}
    
    async def ask(self, prompt: str) -> str:
        """
        Send a prompt to the agent and get a response.
        
        Args:
            prompt: The prompt to send to the agent
            
        Returns:
            Agent's response as a string
        """
        logger.info(f"{self.name.capitalize()} agent received prompt: {prompt[:50]}...")
        # In a real implementation, this would interact with an LLM or other agent system
        return f"INITIALIZED - Mock {self.name.capitalize()} agent response"
    
    def update_context(self, context: Dict[str, Any]) -> None:
        """
        Update the agent's context.
        
        Args:
            context: Context dictionary
        """
        self.context.update(context)


class ObserverAgent(BaseAgent):
    """Agent responsible for observing and analyzing bugs."""
    
    def __init__(self):
        """Initialize the observer agent."""
        super().__init__()
        self.name = "observer"
    
    async def ask(self, prompt: str) -> str:
        """
        Send a prompt to the observer agent and get a response.
        
        Args:
            prompt: The prompt to send to the agent
            
        Returns:
            Agent's response as a string
        """
        logger.info(f"Observer agent received prompt: {prompt[:50]}...")
        
        if "INITIALIZATION" in prompt:
            return "INITIALIZED - Observer ready to analyze bugs"
        
        # Mock a typical observer response
        return """
        {
            "status": "SUCCESS",
            "summary": "Mock bug analysis from Observer",
            "repro_steps": ["Step 1", "Step 2", "Step 3"],
            "evidence": ["Error log 1", "Error log 2"],
            "root_cause": "Mock root cause identification",
            "complexity": "medium"
        }
        """


class AnalystAgent(BaseAgent):
    """Agent responsible for analyzing bugs and generating patches."""
    
    def __init__(self):
        """Initialize the analyst agent."""
        super().__init__()
        self.name = "analyst"
    
    async def ask(self, prompt: str) -> str:
        """
        Send a prompt to the analyst agent and get a response.
        
        Args:
            prompt: The prompt to send to the agent
            
        Returns:
            Agent's response as a string
        """
        logger.info(f"Analyst agent received prompt: {prompt[:50]}...")
        
        if "INITIALIZATION" in prompt:
            return "INITIALIZED - Analyst ready to generate patches"
        
        # Mock a typical analyst response
        return """
        {
            "status": "SUCCESS",
            "description": "Mock patch implementation",
            "files_changed": 2,
            "lines_changed": 15,
            "confidence": 0.85
        }
        
        diff --git a/mock_file.py b/mock_file.py
        index abc123..def456 100644
        --- a/mock_file.py
        +++ b/mock_file.py
        @@ -10,7 +10,7 @@
             # Mock patch content
        -    bug_line = "This has a bug"
        +    fixed_line = "This is fixed now"
        """


class VerifierAgent(BaseAgent):
    """Agent responsible for verifying patches."""
    
    def __init__(self):
        """Initialize the verifier agent."""
        super().__init__()
        self.name = "verifier"
    
    async def ask(self, prompt: str) -> str:
        """
        Send a prompt to the verifier agent and get a response.
        
        Args:
            prompt: The prompt to send to the agent
            
        Returns:
            Agent's response as a string
        """
        logger.info(f"Verifier agent received prompt: {prompt[:50]}...")
        
        if "INITIALIZATION" in prompt:
            return "INITIALIZED - Verifier ready to test patches"
        
        # Mock a typical verifier response
        return """
        {
            "status": "SUCCESS",
            "verification_passed": true,
            "tests_run": 5,
            "tests_passed": 5,
            "edge_cases_checked": ["Case 1", "Case 2"],
            "performance_impact": "minimal"
        }
        """


class PlannerAgent(BaseAgent):
    """Agent responsible for planning and coordinating the bug fixing process."""
    
    def __init__(self):
        """Initialize the planner agent."""
        super().__init__()
        self.name = "planner"
    
    async def ask(self, prompt: str) -> str:
        """
        Send a prompt to the planner agent and get a response.
        
        Args:
            prompt: The prompt to send to the agent
            
        Returns:
            Agent's response as a string
        """
        logger.info(f"Planner agent received prompt: {prompt[:50]}...")
        
        if "INITIALIZATION" in prompt:
            return "INITIALIZED - Planner ready to coordinate"
        
        if "family tree" in prompt.lower():
            return """
            {
                "status": "success",
                "family_tree": {
                    "root": "planner",
                    "children": ["observer", "analyst", "verifier"]
                },
                "message": "Family tree initialized successfully"
            }
            """
        
        # Mock a typical planner response
        return """
        {
            "status": "SUCCESS",
            "plan": {
                "path_id": "mock-path-1",
                "actions": [
                    {
                        "type": "analyze",
                        "agent": "observer",
                        "description": "Analyze bug root cause",
                        "parameters": {"depth": "full"}
                    },
                    {
                        "type": "patch",
                        "agent": "analyst",
                        "description": "Generate patch",
                        "parameters": {"conservative": false}
                    },
                    {
                        "type": "verify",
                        "agent": "verifier",
                        "description": "Verify patch effectiveness",
                        "parameters": {"run_regression": true}
                    }
                ]
            }
        }
        """

# Export all the agent classes
__all__ = [
    'ObserverAgent',
    'AnalystAgent',
    'VerifierAgent',
    'PlannerAgent'
]
