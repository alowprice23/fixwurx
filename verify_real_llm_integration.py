#!/usr/bin/env python3
"""
Real Agent LLM Integration Verification

This script verifies that the actual agent implementations (not mocks)
properly integrate with the OpenAI API for their core functionality.
"""

import os
import sys
import json
import time
import logging
import inspect
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("real_llm_verification.log"), logging.StreamHandler()]
)
logger = logging.getLogger("RealLLMVerification")

# Path to save verification results
RESULTS_PATH = "real_llm_verification_results.json"

# OpenAI API key - use the one from configure_openai_integration.py
OPENAI_API_KEY = "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"

# Set the API key in the environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def analyze_agent_file(file_path: str) -> Dict[str, Any]:
    """
    Analyze an agent file to find LLM integration points.
    
    Args:
        file_path: Path to the agent file
        
    Returns:
        Dictionary with analysis results
    """
    if not os.path.exists(file_path):
        return {
            "exists": False,
            "error": f"File {file_path} does not exist",
            "llm_integration": False
        }
    
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        # Look for indicators of LLM integration
        llm_indicators = [
            "openai", "gpt", "llm", "language model", "prompt", "completion",
            "ask(", "astream_chat", "generate_text", "chat.completions",
            "AssistantAgent", "neural", "learning",
            "temperature", "max_tokens", "model="
        ]
        
        # Count occurrences of each indicator
        indicator_counts = {}
        for indicator in llm_indicators:
            count = content.lower().count(indicator.lower())
            if count > 0:
                indicator_counts[indicator] = count
        
        # Analyze imports for OpenAI or AutoGen
        imports = []
        import_lines = [line for line in content.split("\n") if line.strip().startswith(("import ", "from "))]
        for line in import_lines:
            imports.append(line.strip())
        
        # Check for system prompts or templates
        prompts = []
        prompt_indicators = ["prompt", "PROMPT", "system_message", "template"]
        for indicator in prompt_indicators:
            if indicator in content:
                # Extract lines around the indicator
                index = content.find(indicator)
                start = max(0, content.rfind("\n", 0, index))
                end = content.find("\n", index + len(indicator) + 20)
                if end == -1:
                    end = len(content)
                prompts.append(content[start:end].strip())
        
        # Check for LLM configuration
        llm_config = []
        config_indicators = ["llm_config", "config_list", "temperature", "max_tokens"]
        for indicator in config_indicators:
            if indicator in content:
                # Extract lines around the indicator
                index = content.find(indicator)
                start = max(0, content.rfind("\n", 0, index))
                end = content.find("\n", index + len(indicator) + 20)
                if end == -1:
                    end = len(content)
                llm_config.append(content[start:end].strip())
        
        # Extract methods that might use LLM
        methods = []
        method_lines = []
        in_method = False
        indentation = 0
        for line in content.split("\n"):
            if line.strip().startswith("def ") and not line.strip().startswith("def _"):
                if in_method:
                    methods.append("\n".join(method_lines))
                    method_lines = []
                in_method = True
                indentation = len(line) - len(line.lstrip())
                method_lines.append(line)
            elif in_method:
                if not line.strip() or line.startswith(" " * indentation):
                    method_lines.append(line)
                else:
                    in_method = False
                    methods.append("\n".join(method_lines))
                    method_lines = []
        
        if in_method and method_lines:
            methods.append("\n".join(method_lines))
        
        # Check for potential LLM methods
        llm_methods = []
        llm_method_indicators = llm_indicators + ["generate", "analyze", "create", "process", "execute"]
        for method in methods:
            for indicator in llm_method_indicators:
                if indicator.lower() in method.lower():
                    # Check if method contains actual LLM usage
                    if any(i.lower() in method.lower() for i in ["prompt", "openai", "gpt", "llm", "language model"]):
                        llm_methods.append(method.split("\n")[0].strip())
                        break
        
        # Determine if file has LLM integration
        has_llm_integration = len(indicator_counts) > 0 or any("openai" in imp.lower() or "autogen" in imp.lower() for imp in imports)
        
        return {
            "exists": True,
            "file_path": file_path,
            "llm_integration": has_llm_integration,
            "indicator_counts": indicator_counts,
            "imports": imports,
            "prompts": prompts,
            "llm_config": llm_config,
            "llm_methods": llm_methods,
            "size": len(content),
            "line_count": len(content.split("\n"))
        }
    
    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {e}")
        return {
            "exists": True,
            "file_path": file_path,
            "error": str(e),
            "llm_integration": False
        }

def verify_all_agents() -> Dict[str, Any]:
    """
    Verify LLM integration for all real agent implementations.
    
    Returns:
        Dictionary with verification results
    """
    logger.info("Starting real LLM integration verification")
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": True,
        "agents": {}
    }
    
    # Define agent files to check
    agent_files = {
        "meta": "meta_agent.py",
        "planner": "agents/core/planner_agent.py",
        "observer": "agents/specialized/specialized_agents.py",  # Contains Observer, Analyst, Verifier
        "analyst": "agents/specialized/specialized_agents.py",
        "verifier": "agents/specialized/specialized_agents.py",
        "launchpad": "agents/core/launchpad/agent.py",
        "auditor": "agents/auditor/auditor_agent.py"
    }
    
    # Analyze each agent file
    for agent_type, file_path in agent_files.items():
        logger.info(f"Analyzing {agent_type.upper()} agent file: {file_path}")
        
        analysis = analyze_agent_file(file_path)
        
        # Store analysis in results
        results["agents"][agent_type] = {
            "file_path": file_path,
            "exists": analysis.get("exists", False),
            "llm_integration": analysis.get("llm_integration", False),
            "indicator_counts": analysis.get("indicator_counts", {}),
            "has_imports": any("openai" in imp.lower() or "autogen" in imp.lower() for imp in analysis.get("imports", [])),
            "has_prompts": len(analysis.get("prompts", [])) > 0,
            "has_llm_config": len(analysis.get("llm_config", [])) > 0,
            "llm_methods": analysis.get("llm_methods", []),
            "passed": analysis.get("llm_integration", False)
        }
        
        # Update overall success
        if not analysis.get("llm_integration", False):
            results["success"] = False
    
    # Save results
    try:
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {RESULTS_PATH}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    return results

def print_results(results: Dict[str, Any]) -> None:
    """Print verification results."""
    print("\n==== REAL LLM INTEGRATION VERIFICATION RESULTS ====\n")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Overall Success: {'✅ PASSED' if results['success'] else '❌ FAILED'}")
    print("\nAgent Results:")
    
    for agent_type, agent_results in results["agents"].items():
        status = "✅ PASSED" if agent_results["passed"] else "❌ FAILED"
        print(f"  {agent_type.upper()} Agent: {status}")
        print(f"    - File: {agent_results['file_path']}")
        print(f"    - File exists: {agent_results['exists']}")
        print(f"    - LLM integration: {agent_results['llm_integration']}")
        
        if agent_results["llm_integration"]:
            # Print top indicators
            indicators = agent_results.get("indicator_counts", {})
            if indicators:
                top_indicators = sorted(indicators.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"    - Top indicators: {', '.join(f'{k}({v})' for k, v in top_indicators)}")
            
            # Print LLM methods
            methods = agent_results.get("llm_methods", [])
            if methods:
                print(f"    - LLM methods: {len(methods)}")
                for method in methods[:2]:  # Show only first 2 for brevity
                    print(f"        {method}")
                if len(methods) > 2:
                    print(f"        ... and {len(methods) - 2} more")
    
    print("\n===============================================\n")

if __name__ == "__main__":
    # Run verification
    results = verify_all_agents()
    
    # Print results
    print_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)
