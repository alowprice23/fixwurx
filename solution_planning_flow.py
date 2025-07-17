#!/usr/bin/env python3
"""
solution_planning_flow.py
─────────────────────────
Implements the solution planning flow for the FixWurx system.

This module provides the core flow for planning solutions to detected bugs,
including strategy selection, path generation, resource allocation, and
planning optimization. It integrates with various components of the system
including the agent system, triangulation engine, and neural matrix.
"""

import os
import sys
import json
import logging
import time
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

# Import core components
from triangulation_engine import TriangulationEngine
from neural_matrix_core import NeuralMatrix
from meta_agent import MetaAgent
from resource_manager import ResourceManager
from plan_storage import PlanStorage
from solution_paths import SolutionPathGenerator

# Configure logging
logger = logging.getLogger("SolutionPlanningFlow")

class SolutionPlanningFlow:
    """
    Implements the solution planning flow for the FixWurx system.
    
    This class orchestrates the entire solution planning process, from analyzing
    detected bugs to generating and optimizing solution paths. It serves as the
    main entry point for the solution planning subsystem.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the solution planning flow.
        
        Args:
            config: Configuration for the solution planning flow.
        """
        self.config = config or {}
        self.triangulation_engine = TriangulationEngine()
        self.neural_matrix = NeuralMatrix()
        self.meta_agent = MetaAgent()
        self.resource_manager = ResourceManager()
        self.plan_storage = PlanStorage()
        self.path_generator = SolutionPathGenerator()
        
        # Initialize state
        self.current_plan_id = None
        self.current_context = {}
        self.generated_plans = []
        
        logger.info("Solution Planning Flow initialized")
    
    def start_planning(self, 
                      bug_report: Dict[str, Any], 
                      planning_options: Dict[str, Any] = None) -> str:
        """
        Start the solution planning process for detected bugs.
        
        Args:
            bug_report: Bug detection report.
            planning_options: Options for the planning process.
            
        Returns:
            Plan ID for the planning process.
        """
        planning_options = planning_options or {}
        
        # Generate a plan ID
        timestamp = int(time.time())
        plan_id = f"plan_{timestamp}_{str(uuid.uuid4())[:8]}"
        self.current_plan_id = plan_id
        
        # Initialize planning context
        self.current_context = {
            "plan_id": plan_id,
            "bug_report": bug_report,
            "start_time": timestamp,
            "options": planning_options,
            "status": "started",
            "bugs_analyzed": 0,
            "paths_generated": 0
        }
        
        logger.info(f"Starting solution planning {plan_id} for {len(bug_report.get('bugs', []))} bugs")
        
        # Trigger the planning flow
        self._execute_planning_flow(bug_report, planning_options)
        
        return plan_id
    
    def _execute_planning_flow(self, 
                              bug_report: Dict[str, Any], 
                              planning_options: Dict[str, Any]) -> None:
        """
        Execute the solution planning flow.
        
        Args:
            bug_report: Bug detection report.
            planning_options: Options for the planning process.
        """
        try:
            # Phase 1: Bug analysis
            logger.info("Phase 1: Bug analysis")
            analyzed_bugs = self._analyze_bugs(bug_report.get('bugs', []), planning_options)
            
            # Phase 2: Resource estimation
            logger.info("Phase 2: Resource estimation")
            resource_estimates = self._estimate_resources(analyzed_bugs)
            
            # Phase 3: Solution path generation
            logger.info("Phase 3: Solution path generation")
            solution_paths = self._generate_solution_paths(analyzed_bugs, resource_estimates)
            
            # Phase 4: Path optimization
            logger.info("Phase 4: Path optimization")
            optimized_paths = self._optimize_paths(solution_paths, planning_options)
            
            # Phase 5: Plan creation
            logger.info("Phase 5: Plan creation")
            plan = self._create_plan(optimized_paths, resource_estimates)
            
            # Update context
            self.current_context["status"] = "completed"
            self.current_context["end_time"] = int(time.time())
            self.current_context["bugs_analyzed"] = len(analyzed_bugs)
            self.current_context["paths_generated"] = len(optimized_paths)
            self.current_context["plan"] = plan
            
            # Store generated plan
            self.generated_plans.append(plan)
            
            # Store plan in plan storage
            self.plan_storage.store_plan(plan)
            
            logger.info(f"Solution planning {self.current_plan_id} completed with {len(optimized_paths)} solution paths")
            
        except Exception as e:
            logger.error(f"Error in solution planning flow: {e}")
            self.current_context["status"] = "failed"
            self.current_context["error"] = str(e)
            self.current_context["end_time"] = int(time.time())
            raise
    
    def _analyze_bugs(self, 
                     bugs: List[Dict[str, Any]], 
                     planning_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze bugs to prepare for solution planning.
        
        Args:
            bugs: List of detected bugs.
            planning_options: Planning options.
            
        Returns:
            List of analyzed bugs.
        """
        analyzed_bugs = []
        
        for bug in bugs:
            # Skip bugs that don't meet priority threshold
            if planning_options.get('priority_threshold'):
                if bug.get('priority', 0) < planning_options['priority_threshold']:
                    continue
            
            # Get detailed analysis from triangulation engine
            analysis = self.triangulation_engine.analyze_bug_for_planning(bug)
            
            # Add analysis to the bug
            bug["planning_analysis"] = analysis
            
            # Get neural insights for planning
            neural_insights = self.neural_matrix.get_planning_insights(bug)
            
            # Add neural insights to the bug
            bug["planning_neural_insights"] = neural_insights
            
            analyzed_bugs.append(bug)
        
        # Update context
        self.current_context["bugs_analyzed"] = len(analyzed_bugs)
        
        return analyzed_bugs
    
    def _estimate_resources(self, 
                           analyzed_bugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate resources needed for fixing bugs.
        
        Args:
            analyzed_bugs: List of analyzed bugs.
            
        Returns:
            Resource estimation data.
        """
        resources = {
            "total_estimated_time": 0,
            "cpu_resources": 0,
            "memory_resources": 0,
            "bugs": []
        }
        
        for bug in analyzed_bugs:
            analysis = bug.get("planning_analysis", {})
            complexity = analysis.get("complexity", 0.5)
            estimated_time = analysis.get("estimated_time", 60)  # Default 60 seconds
            
            # Adjust based on neural insights
            neural_insights = bug.get("planning_neural_insights", {})
            time_adjustment = neural_insights.get("time_adjustment_factor", 1.0)
            adjusted_time = estimated_time * time_adjustment
            
            # Calculate resource needs
            cpu_need = complexity * 2  # Simple heuristic
            memory_need = complexity * 1024  # MB
            
            # Add to total
            resources["total_estimated_time"] += adjusted_time
            resources["cpu_resources"] += cpu_need
            resources["memory_resources"] += memory_need
            
            # Store per-bug resources
            resources["bugs"].append({
                "bug_id": bug.get("id"),
                "estimated_time": adjusted_time,
                "cpu_need": cpu_need,
                "memory_need": memory_need
            })
        
        # Check resource availability
        available_resources = self.resource_manager.get_available_resources()
        resources["resource_availability"] = {
            "cpu_available": available_resources.get("cpu", 0),
            "memory_available": available_resources.get("memory", 0),
            "sufficient": (
                available_resources.get("cpu", 0) >= resources["cpu_resources"] and
                available_resources.get("memory", 0) >= resources["memory_resources"]
            )
        }
        
        return resources
    
    def _generate_solution_paths(self, 
                                analyzed_bugs: List[Dict[str, Any]], 
                                resource_estimates: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate solution paths for bugs.
        
        Args:
            analyzed_bugs: List of analyzed bugs.
            resource_estimates: Resource estimation data.
            
        Returns:
            List of solution paths.
        """
        solution_paths = []
        
        for bug in analyzed_bugs:
            # Generate multiple solution paths for each bug
            paths = self.path_generator.generate_paths(bug)
            
            # Add resource estimates to paths
            for path in paths:
                bug_resources = next(
                    (r for r in resource_estimates["bugs"] if r["bug_id"] == bug.get("id")),
                    {"estimated_time": 60, "cpu_need": 1, "memory_need": 512}
                )
                
                path["resources"] = {
                    "estimated_time": bug_resources["estimated_time"] / len(paths),
                    "cpu_need": bug_resources["cpu_need"] / len(paths),
                    "memory_need": bug_resources["memory_need"] / len(paths)
                }
            
            solution_paths.extend(paths)
        
        # Update context
        self.current_context["paths_generated"] = len(solution_paths)
        
        return solution_paths
    
    def _optimize_paths(self, 
                       solution_paths: List[Dict[str, Any]], 
                       planning_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Optimize solution paths.
        
        Args:
            solution_paths: List of solution paths.
            planning_options: Planning options.
            
        Returns:
            List of optimized solution paths.
        """
        # Sort paths by priority
        solution_paths.sort(key=lambda p: p.get("priority", 0), reverse=True)
        
        # Apply neural optimization
        optimized_paths = self.neural_matrix.optimize_solution_paths(solution_paths)
        
        # Apply any additional optimizations from options
        if planning_options.get("path_limit"):
            optimized_paths = optimized_paths[:planning_options["path_limit"]]
        
        # Group paths by bug
        bug_paths = {}
        for path in optimized_paths:
            bug_id = path.get("bug_id")
            if bug_id not in bug_paths:
                bug_paths[bug_id] = []
            bug_paths[bug_id].append(path)
        
        # Ensure each bug has at least one path
        final_paths = []
        for bug_id, paths in bug_paths.items():
            # Take at least one path per bug
            final_paths.append(paths[0])
            
            # Take additional paths if they meet threshold
            if planning_options.get("path_diversity_threshold"):
                threshold = planning_options["path_diversity_threshold"]
                for path in paths[1:]:
                    if path.get("diversity_score", 0) >= threshold:
                        final_paths.append(path)
        
        return final_paths
    
    def _create_plan(self, 
                    optimized_paths: List[Dict[str, Any]], 
                    resource_estimates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a plan from optimized solution paths.
        
        Args:
            optimized_paths: List of optimized solution paths.
            resource_estimates: Resource estimation data.
            
        Returns:
            Solution plan.
        """
        # Calculate plan metrics
        total_paths = len(optimized_paths)
        total_bugs = len(set(path.get("bug_id") for path in optimized_paths))
        total_time = sum(path.get("resources", {}).get("estimated_time", 0) for path in optimized_paths)
        
        # Create plan
        plan = {
            "plan_id": self.current_plan_id,
            "created_at": int(time.time()),
            "total_bugs": total_bugs,
            "total_paths": total_paths,
            "estimated_completion_time": total_time,
            "resource_estimates": resource_estimates,
            "paths": optimized_paths
        }
        
        # Add execution schedule
        plan["execution_schedule"] = self._create_execution_schedule(optimized_paths)
        
        # Add dependencies
        plan["dependencies"] = self._identify_dependencies(optimized_paths)
        
        # Notify the Meta Agent
        self.meta_agent.notify_plan_creation(plan)
        
        return plan
    
    def _create_execution_schedule(self, 
                                  paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create an execution schedule for solution paths.
        
        Args:
            paths: List of solution paths.
            
        Returns:
            Execution schedule.
        """
        # Sort paths by priority
        sorted_paths = sorted(paths, key=lambda p: p.get("priority", 0), reverse=True)
        
        # Create schedule
        schedule = []
        current_time = 0
        
        for path in sorted_paths:
            # Calculate start time based on dependencies
            dependencies = path.get("dependencies", [])
            dependency_end_times = [
                next(
                    (s["end_time"] for s in schedule if s["path_id"] == dep),
                    0
                )
                for dep in dependencies
            ]
            
            start_time = max(dependency_end_times) if dependency_end_times else current_time
            duration = path.get("resources", {}).get("estimated_time", 60)
            end_time = start_time + duration
            
            # Add to schedule
            schedule.append({
                "path_id": path.get("id"),
                "bug_id": path.get("bug_id"),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration
            })
            
            # Update current time
            current_time = max(current_time, end_time)
        
        return schedule
    
    def _identify_dependencies(self, 
                              paths: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Identify dependencies between solution paths.
        
        Args:
            paths: List of solution paths.
            
        Returns:
            Dependencies between paths.
        """
        dependencies = {}
        
        # Group paths by bug
        bug_paths = {}
        for path in paths:
            bug_id = path.get("bug_id")
            if bug_id not in bug_paths:
                bug_paths[bug_id] = []
            bug_paths[bug_id].append(path)
        
        # Initialize dependencies for each path
        for path in paths:
            dependencies[path.get("id")] = []
        
        # Identify inter-bug dependencies
        for path in paths:
            bug_deps = path.get("bug_dependencies", [])
            for bug_dep in bug_deps:
                # Find paths for this bug
                if bug_dep in bug_paths:
                    # Depend on the highest priority path
                    dep_paths = sorted(
                        bug_paths[bug_dep],
                        key=lambda p: p.get("priority", 0),
                        reverse=True
                    )
                    if dep_paths:
                        dependencies[path.get("id")].append(dep_paths[0].get("id"))
        
        return dependencies
    
    def get_plan_by_id(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a plan by its ID.
        
        Args:
            plan_id: Plan ID.
            
        Returns:
            Plan data or None if not found.
        """
        if plan_id == self.current_plan_id and self.current_context.get("plan"):
            return self.current_context["plan"]
        
        # Try to find in generated plans
        for plan in self.generated_plans:
            if plan.get("plan_id") == plan_id:
                return plan
        
        # Try to retrieve from storage
        return self.plan_storage.get_plan(plan_id)
    
    def get_planning_status(self) -> Dict[str, Any]:
        """
        Get the status of the current planning process.
        
        Returns:
            Planning status data.
        """
        return self.current_context
    
    def save_plan(self, 
                 output_path: str = None, 
                 format: str = "json") -> str:
        """
        Save the solution plan to a file.
        
        Args:
            output_path: Path to save the plan. If None, a default path is used.
            format: Plan format (json or html).
            
        Returns:
            Path to the saved plan.
        """
        if not self.current_context.get("plan"):
            raise ValueError("No plan available to save")
        
        # Create default output path if not provided
        if not output_path:
            timestamp = self.current_context["start_time"]
            filename = f"solution_plan_{self.current_plan_id}.{format}"
            output_path = os.path.join(os.getcwd(), filename)
        
        # Save plan in the specified format
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.current_context["plan"], f, indent=2)
        elif format == "html":
            # Generate HTML plan
            html_plan = self._generate_html_plan(self.current_context["plan"])
            
            with open(output_path, "w") as f:
                f.write(html_plan)
        else:
            raise ValueError(f"Unsupported plan format: {format}")
        
        logger.info(f"Solution plan saved to {output_path}")
        
        return output_path
    
    def _generate_html_plan(self, plan: Dict[str, Any]) -> str:
        """
        Generate an HTML plan from the solution planning data.
        
        Args:
            plan: Solution plan data.
            
        Returns:
            HTML plan as a string.
        """
        # Generate HTML plan
        # This is a simplified implementation
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Solution Plan: {plan['plan_id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .timeline {{ height: 30px; background-color: #e0e0e0; position: relative; margin-top: 20px; }}
                .path {{ position: absolute; height: 20px; top: 5px; background-color: #4285f4; border-radius: 3px; }}
                .priority-high {{ background-color: #ff5252; }}
                .priority-medium {{ background-color: #ffb74d; }}
                .priority-low {{ background-color: #4caf50; }}
            </style>
        </head>
        <body>
            <h1>Solution Plan</h1>
            
            <h2>Summary</h2>
            <table>
                <tr><th>Plan ID</th><td>{plan['plan_id']}</td></tr>
                <tr><th>Created At</th><td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(plan['created_at']))}</td></tr>
                <tr><th>Total Bugs</th><td>{plan['total_bugs']}</td></tr>
                <tr><th>Total Paths</th><td>{plan['total_paths']}</td></tr>
                <tr><th>Estimated Completion Time</th><td>{plan['estimated_completion_time']} seconds</td></tr>
            </table>
            
            <h2>Resource Estimates</h2>
            <table>
                <tr><th>Total Estimated Time</th><td>{plan['resource_estimates']['total_estimated_time']} seconds</td></tr>
                <tr><th>CPU Resources</th><td>{plan['resource_estimates']['cpu_resources']}</td></tr>
                <tr><th>Memory Resources</th><td>{plan['resource_estimates']['memory_resources']} MB</td></tr>
                <tr>
                    <th>Resource Availability</th>
                    <td>
                        CPU: {plan['resource_estimates']['resource_availability']['cpu_available']}<br>
                        Memory: {plan['resource_estimates']['resource_availability']['memory_available']} MB<br>
                        Sufficient: {'Yes' if plan['resource_estimates']['resource_availability']['sufficient'] else 'No'}
                    </td>
                </tr>
            </table>
            
            <h2>Execution Schedule</h2>
            <div class="timeline">
        """
        
        # Add execution schedule to timeline
        schedule = plan.get("execution_schedule", [])
        if schedule:
            total_time = max(s["end_time"] for s in schedule)
            
            for item in schedule:
                start_percent = (item["start_time"] / total_time) * 100
                width_percent = ((item["end_time"] - item["start_time"]) / total_time) * 100
                
                # Determine priority class
                path = next((p for p in plan["paths"] if p["id"] == item["path_id"]), {})
                priority = path.get("priority", 0.5)
                priority_class = "priority-high" if priority > 0.7 else "priority-medium" if priority > 0.4 else "priority-low"
                
                html += f"""
                <div class="path {priority_class}" style="left: {start_percent}%; width: {width_percent}%;" title="Path {item['path_id']} for Bug {item['bug_id']}"></div>
                """
        
        html += """
            </div>
            
            <h2>Solution Paths</h2>
        """
        
        # Add solution paths
        for i, path in enumerate(plan["paths"]):
            html += f"""
            <div class="path-details">
                <h3>Path #{i+1}: {path.get('title', 'Untitled Path')}</h3>
                <table>
                    <tr><th>ID</th><td>{path.get('id', 'N/A')}</td></tr>
                    <tr><th>Bug ID</th><td>{path.get('bug_id', 'N/A')}</td></tr>
                    <tr><th>Priority</th><td>{path.get('priority', 'N/A')}</td></tr>
                    <tr><th>Estimated Time</th><td>{path.get('resources', {}).get('estimated_time', 'N/A')} seconds</td></tr>
                    <tr><th>Description</th><td>{path.get('description', 'N/A')}</td></tr>
                </table>
                
                <h4>Steps</h4>
                <ol>
            """
            
            for step in path.get("steps", []):
                html += f"<li>{step.get('description', 'N/A')}</li>"
            
            html += """
                </ol>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html

# Main entry point
def plan_solutions(bug_report: Dict[str, Any], options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main function to plan solutions for detected bugs.
    
    Args:
        bug_report: Bug detection report.
        options: Planning options.
        
    Returns:
        Solution plan.
    """
    flow = SolutionPlanningFlow()
    plan_id = flow.start_planning(bug_report, options)
    
    # Wait for planning to complete
    while flow.get_planning_status()["status"] not in ["completed", "failed"]:
        time.sleep(0.1)
    
    # Get planning results
    planning_status = flow.get_planning_status()
    
    if planning_status["status"] == "failed":
        logger.error(f"Solution planning failed: {planning_status.get('error', 'Unknown error')}")
        raise RuntimeError(f"Solution planning failed: {planning_status.get('error', 'Unknown error')}")
    
    return planning_status["plan"]

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python solution_planning_flow.py <bug_report_json> [options_json]")
        sys.exit(1)
    
    bug_report_path = sys.argv[1]
    
    # Load bug report
    try:
        with open(bug_report_path, "r") as f:
            bug_report = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading bug report: {e}")
        sys.exit(1)
    
    # Parse options if provided
    options = {}
    if len(sys.argv) > 2:
        try:
            options = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            print("Error: options must be a valid JSON string")
            sys.exit(1)
    
    # Run solution planning
    try:
        plan = plan_solutions(bug_report, options)
        
        # Create output path
        output_path = options.get("output_path", f"solution_plan_{int(time.time())}.json")
        
        # Save plan
        with open(output_path, "w") as f:
            json.dump(plan, f, indent=2)
        
        print(f"Solution plan saved to {output_path}")
        
        # Print summary
        print("\nSummary:")
        print(f"Total bugs: {plan['total_bugs']}")
        print(f"Total paths: {plan['total_paths']}")
        print(f"Estimated completion time: {plan['estimated_completion_time']} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
