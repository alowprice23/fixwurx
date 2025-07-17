#!/usr/bin/env python3
"""
Solution Planning Commands - Shell integration for the Solution Planning Flow.

This module provides command-line interface commands for integrating 
the solution planning flow with the shell system.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple

# Import solution planning flow
from solution_planning_flow import SolutionPlanningFlow

# Import bug detection for integration
try:
    from bug_detection_flow import BugDetectionFlow
except ImportError:
    BugDetectionFlow = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("solution_planning_commands.log"), logging.StreamHandler()]
)
logger = logging.getLogger("SolutionPlanningCommands")

# Create a singleton instance of SolutionPlanningFlow
_planning_flow = None

def get_planning_flow() -> SolutionPlanningFlow:
    """
    Get the singleton instance of SolutionPlanningFlow.
    
    Returns:
        SolutionPlanningFlow instance
    """
    global _planning_flow
    if _planning_flow is None:
        _planning_flow = SolutionPlanningFlow()
    return _planning_flow

def plan_solutions(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shell command to plan solutions for detected bugs.
    
    This command integrates bug detection and solution planning to generate 
    solution plans for detected bugs. It uses the LLM agent system to 
    generate intelligent solution strategies.
    
    Args:
        args: Arguments for the command
            - file: File or directory to analyze for bugs
            - options: Optional planning options
        context: Shell context
        
    Returns:
        Planning results
    """
    file_path = args.get("file", "")
    options = args.get("options", {})
    
    if not file_path:
        return {
            "success": False,
            "error": "File or directory path required",
            "usage": "plan_solutions --file=<path> [--options=<json>]"
        }
    
    logger.info(f"Planning solutions for {file_path}")
    start_time = time.time()
    
    # 1. Detect bugs using BugDetectionFlow if available
    bug_detection_results = None
    if BugDetectionFlow:
        try:
            logger.info(f"Detecting bugs in {file_path}")
            bug_detector = BugDetectionFlow()
            bug_detection_results = bug_detector.detect_bugs(file_path)
            logger.info(f"Bug detection completed with {bug_detection_results.get('bug_count', 0)} bugs found")
        except Exception as e:
            logger.error(f"Error detecting bugs: {str(e)}")
            return {
                "success": False,
                "error": f"Bug detection failed: {str(e)}",
                "elapsed_time": time.time() - start_time
            }
    else:
        # If bug detection flow is not available, check if detection results were provided
        detection_results_path = args.get("detection_results", "")
        if detection_results_path:
            try:
                with open(detection_results_path, 'r') as f:
                    bug_detection_results = json.load(f)
            except Exception as e:
                logger.error(f"Error loading detection results: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to load detection results: {str(e)}",
                    "elapsed_time": time.time() - start_time
                }
        else:
            return {
                "success": False,
                "error": "Bug detection flow not available and no detection results provided",
                "usage": "plan_solutions --file=<path> --detection_results=<path> [--options=<json>]",
                "elapsed_time": time.time() - start_time
            }
    
    # 2. Plan solutions using SolutionPlanningFlow
    if bug_detection_results:
        try:
            logger.info("Planning solutions using LLM agent system")
            planning_flow = get_planning_flow()
            planning_results = planning_flow.run_planning_flow(bug_detection_results, options)
            logger.info(f"Solution planning completed with {planning_results.get('plan_count', 0)} plans generated")
            
            # 3. Return combined results
            elapsed_time = time.time() - start_time
            return {
                "success": True,
                "file": file_path,
                "elapsed_time": elapsed_time,
                "detection_results": bug_detection_results,
                "planning_results": planning_results
            }
        except Exception as e:
            logger.error(f"Error planning solutions: {str(e)}")
            return {
                "success": False,
                "error": f"Solution planning failed: {str(e)}",
                "elapsed_time": time.time() - start_time
            }
    else:
        return {
            "success": False,
            "error": "No bug detection results available",
            "elapsed_time": time.time() - start_time
        }

def implement_solution(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shell command to implement a solution for a bug.
    
    This command implements a solution for a specific bug using the LLM agent system.
    
    Args:
        args: Arguments for the command
            - bug_id: ID of the bug to implement a solution for
        context: Shell context
        
    Returns:
        Implementation results
    """
    bug_id = args.get("bug_id", "")
    
    if not bug_id:
        return {
            "success": False,
            "error": "Bug ID required",
            "usage": "implement_solution --bug_id=<bug_id>"
        }
    
    logger.info(f"Implementing solution for bug {bug_id}")
    start_time = time.time()
    
    try:
        planning_flow = get_planning_flow()
        result = planning_flow.implement_solution(bug_id)
        
        elapsed_time = time.time() - start_time
        return {
            "success": result.get("success", False),
            "bug_id": bug_id,
            "elapsed_time": elapsed_time,
            "implementation_results": result
        }
    except Exception as e:
        logger.error(f"Error implementing solution: {str(e)}")
        return {
            "success": False,
            "error": f"Solution implementation failed: {str(e)}",
            "elapsed_time": time.time() - start_time
        }

def get_solution_plans(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shell command to get solution plans for detected bugs.
    
    This command retrieves solution plans that have been generated for detected bugs.
    
    Args:
        args: Arguments for the command
            - bug_id: Optional ID of a specific bug to get plans for
            - all: If true, get plans for all bugs
        context: Shell context
        
    Returns:
        Solution plans
    """
    bug_id = args.get("bug_id", "")
    get_all = args.get("all", False)
    
    if not bug_id and not get_all:
        return {
            "success": False,
            "error": "Bug ID or --all flag required",
            "usage": "get_solution_plans --bug_id=<bug_id> OR get_solution_plans --all=true"
        }
    
    logger.info(f"Getting solution plans for {'all bugs' if get_all else f'bug {bug_id}'}")
    
    try:
        planning_flow = get_planning_flow()
        
        if get_all:
            # Get all processed bugs
            return {
                "success": True,
                "plans": planning_flow.processed_bugs
            }
        else:
            # Get specific bug
            plan = planning_flow.get_solution_plan(bug_id)
            if plan:
                return {
                    "success": True,
                    "bug_id": bug_id,
                    "plan": plan
                }
            else:
                return {
                    "success": False,
                    "error": f"No solution plan found for bug {bug_id}"
                }
    except Exception as e:
        logger.error(f"Error getting solution plans: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to get solution plans: {str(e)}"
        }

# Shell command registration mapping
commands = {
    "plan_solutions": {
        "function": plan_solutions,
        "help": "Plan solutions for detected bugs using the LLM agent system",
        "args": {
            "file": {
                "help": "File or directory to analyze for bugs",
                "required": True
            },
            "detection_results": {
                "help": "Path to bug detection results JSON (optional)",
                "required": False
            },
            "options": {
                "help": "Planning options as JSON string",
                "required": False
            }
        }
    },
    "implement_solution": {
        "function": implement_solution,
        "help": "Implement a solution for a bug",
        "args": {
            "bug_id": {
                "help": "ID of the bug to implement a solution for",
                "required": True
            }
        }
    },
    "get_solution_plans": {
        "function": get_solution_plans,
        "help": "Get solution plans for detected bugs",
        "args": {
            "bug_id": {
                "help": "ID of a specific bug to get plans for",
                "required": False
            },
            "all": {
                "help": "If true, get plans for all bugs",
                "required": False
            }
        }
    }
}
