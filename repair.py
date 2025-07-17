"""
tooling/repair.py
─────────────────
Enhanced patch orchestration module that supports both traditional DAG-aware repairs
and planner-guided solution paths.

Traditional DAG-Aware Mode
─────────────────────────
1. **Dependency Graph Construction**  
   • Builds a directed graph G = (V,E) where each vertex is a source file and
     an edge *u → v* means  *v* `import`/`require`s *u*  (definition before use).

2. **Tarjan SCC Decomposition**  
   • Runs Tarjan's linear-time algorithm to obtain *strongly connected
     components* (SCCs).  Each SCC becomes an **atomic repair unit** because
     any file inside the component can ripple to all others.

3. **Ripple-Score Heuristic**  
   • For every SCC `Cᵢ` compute  

        ripple(Cᵢ) = |Cᵢ|  +  Σ_{(Cᵢ→Cⱼ)}  |Cⱼ|
     
     i.e. direct size plus size of *downstream* dependents.  Sorting SCCs
     descending by that score yields a patch order that minimises cascades.

Planner-Guided Mode
──────────────────
1. **Solution Path Fetching**
   • Retrieves optimal solution paths from the planner via the hub API.
   • Solution paths contain ordered sequences of actions and agent assignments.

2. **Action Execution**
   • Executes each action in the solution path using the assigned agent.
   • Provides feedback on action success/failure to improve future planning.

3. **Adaptive Patching**
   • Falls back to traditional DAG-aware patching if planner solutions fail.
   • Records outcomes to improve future solution paths.

Public API
──────────
Traditional mode:
    graph = build_dep_graph(file_paths)
    sccs  = tarjan_scc(graph)                   # list[list[node]]
    order = ripple_sort(graph, sccs)            # list[list[node]] largest first
    apply_in_order(order, patch_callback)       # user-provided patcher

Planner-guided mode:
    solution = get_solution_path(bug_id)        # Fetch optimal solution from hub
    success = apply_solution_path(solution)     # Execute solution actions
    record_outcome(solution, success)           # Provide feedback to planner

Implementation constraints
──────────────────────────
* **No external libraries** for core algorithms – pure std-lib.
* **RESTful communication** with planner hub for solution paths.
* **O(V + E)** overall complexity for DAG operations.
"""

from __future__ import annotations

import ast
import json
import re
import requests
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Hub API base URL - configurable
HUB_API_BASE = "http://localhost:8001"

# Type aliases
Graph = Dict[str, Set[str]]
SCC = List[str]  # type: Final


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Dependency Graph Construction
# ──────────────────────────────────────────────────────────────────────────────
def build_dep_graph(file_paths: List[str]) -> Graph:
    """
    Build a dependency graph from a list of Python source files.
    An edge u -> v means v depends on u.
    """
    graph: Graph = defaultdict(set)
    # Regex to find 'from .module import ...'
    import_re = re.compile(r"^\s*from\s+([.\w]+)\s+import\s+", re.MULTILINE)

    for path in file_paths:
        # Ensure node exists for every file
        if path not in graph:
            graph[path] = set()

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # AST-based import finding
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    # Simplistic resolver: assumes modules map to file names
                    # This would need to be much more robust for a real system
                    dep_path = f"{node.module.replace('.', '/')}.py"
                    if dep_path in file_paths:
                        graph[dep_path].add(path) # v depends on u
        except (SyntaxError, FileNotFoundError) as e:
            # Ignore files that can't be parsed; they might be data, etc.
            pass
            
    return graph


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Tarjan's SCC Algorithm
# ──────────────────────────────────────────────────────────────────────────────
def tarjan_scc(graph: Graph) -> List[SCC]:
    """
    Find strongly connected components in a graph using Tarjan's algorithm.
    """
    n = len(graph)
    nodes = list(graph.keys())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    visited = [False] * n
    stack = []
    on_stack = [False] * n
    ids = [-1] * n
    low = [-1] * n
    at = 0
    sccs = []

    def dfs(i):
        nonlocal at
        stack.append(i)
        on_stack[i] = True
        ids[i] = low[i] = at
        at += 1

        for neighbor_node in graph[nodes[i]]:
            neighbor_idx = node_to_idx.get(neighbor_node)
            if neighbor_idx is None: continue

            if ids[neighbor_idx] == -1:
                dfs(neighbor_idx)
            if on_stack[neighbor_idx]:
                low[i] = min(low[i], low[neighbor_idx])

        if ids[i] == low[i]:
            scc = []
            while stack:
                node_idx = stack.pop()
                on_stack[node_idx] = False
                low[node_idx] = ids[i]
                scc.append(nodes[node_idx])
                if node_idx == i: break
            sccs.append(scc)

    for i in range(n):
        if ids[i] == -1:
            dfs(i)
            
    return sccs


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Ripple-Score Heuristic
# ──────────────────────────────────────────────────────────────────────────────
def ripple_sort(graph: Graph, sccs: List[SCC]) -> List[SCC]:
    """
    Sort SCCs by "ripple score" to minimize downstream breaks.
    """
    scc_graph = _build_scc_graph(graph, sccs)
    
    memo = {}
    def get_downstream_size(scc_id: int) -> int:
        if scc_id in memo:
            return memo[scc_id]
        
        size = len(sccs[scc_id])
        for neighbor_id in scc_graph.get(scc_id, []):
            size += get_downstream_size(neighbor_id)
        
        memo[scc_id] = size
        return size

    scores = {i: get_downstream_size(i) for i in range(len(sccs))}
    
    sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    
    return [sccs[i] for i in sorted_indices]


def _build_scc_graph(graph: Graph, sccs: List[SCC]) -> Dict[int, Set[int]]:
    """Condense the file graph into an SCC graph."""
    node_to_scc_id = {}
    for i, scc in enumerate(sccs):
        for node in scc:
            node_to_scc_id[node] = i
            
    scc_graph: Dict[int, Set[int]] = defaultdict(set)
    for u, neighbors in graph.items():
        u_id = node_to_scc_id[u]
        for v in neighbors:
            v_id = node_to_scc_id.get(v)
            if v_id is not None and u_id != v_id:
                scc_graph[u_id].add(v_id)
                
    return scc_graph


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Traditional Orchestration
# ──────────────────────────────────────────────────────────────────────────────
def apply_in_order(
    sorted_sccs: List[SCC],
    patch_callback: Callable[[SCC], bool]
) -> bool:
    """
    Apply patches to SCCs in the given order.
    If any patch fails, stop and return False.
    """
    for scc in sorted_sccs:
        if not patch_callback(scc):
            print(f"Patch failed for SCC: {scc}. Aborting.")
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Planner Integration
# ──────────────────────────────────────────────────────────────────────────────
class SolutionAction:
    """Represents a single action in a planner-generated solution path."""
    
    def __init__(self, action_type: str, agent: str, params: Dict[str, Any]):
        self.type = action_type
        self.agent = agent
        self.params = params
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SolutionAction':
        """Create a SolutionAction from a dictionary."""
        return cls(
            action_type=data["type"],
            agent=data["agent"],
            params=data.get("params", {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        return {
            "type": self.type,
            "agent": self.agent,
            "params": self.params
        }
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return f"{self.type} (by {self.agent}): {self.params}"


class SolutionPath:
    """Represents a complete solution path from the planner."""
    
    def __init__(
        self,
        solution_id: str,
        bug_id: str,
        actions: List[SolutionAction],
        metadata: Dict[str, Any] = None
    ):
        self.solution_id = solution_id
        self.bug_id = bug_id
        self.actions = actions
        self.metadata = metadata or {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SolutionPath':
        """Create a SolutionPath from a dictionary."""
        return cls(
            solution_id=data["solution_id"],
            bug_id=data["bug_id"],
            actions=[SolutionAction.from_dict(a) for a in data.get("actions", [])],
            metadata=data.get("metadata", {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        return {
            "solution_id": self.solution_id,
            "bug_id": self.bug_id,
            "actions": [a.to_dict() for a in self.actions],
            "metadata": self.metadata
        }


def get_solution_path(bug_id: str) -> Optional[SolutionPath]:
    """
    Fetch the highest-priority solution path for a given bug from the hub API.
    
    Args:
        bug_id: The ID of the bug to fetch solutions for
        
    Returns:
        A SolutionPath if one is available, None otherwise
    """
    try:
        response = requests.get(f"{HUB_API_BASE}/planner/solutions/{bug_id}")
        response.raise_for_status()
        
        solutions = response.json()
        if not solutions:
            return None
        
        # Get the highest priority solution that's not failed
        valid_solutions = [s for s in solutions if s["status"] != "FAILED"]
        if not valid_solutions:
            return None
        
        # Sort by priority (highest first)
        valid_solutions.sort(key=lambda s: s["priority"], reverse=True)
        
        # Convert the top solution to a SolutionPath
        solution_data = valid_solutions[0]
        
        return SolutionPath.from_dict({
            "solution_id": solution_data["solution_id"],
            "bug_id": solution_data["bug_id"],
            "actions": solution_data["path_data"]["actions"],
            "metadata": solution_data["path_data"].get("metadata", {})
        })
    
    except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error fetching solution path: {e}")
        return None


def update_solution_status(solution_id: int, status: str) -> bool:
    """
    Update the status of a solution in the hub.
    
    Args:
        solution_id: The ID of the solution to update
        status: The new status ("PENDING", "IN_PROGRESS", "SUCCEEDED", "FAILED")
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.post(
            f"{HUB_API_BASE}/planner/solutions/{solution_id}/status",
            params={"status": status}
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Error updating solution status: {e}")
        return False


def record_outcome(solution_id: int, success: bool, metrics: Dict[str, Any] = None) -> bool:
    """
    Record the outcome of applying a solution path.
    
    Args:
        solution_id: The ID of the solution that was applied
        success: Whether the solution was successful
        metrics: Performance metrics for the solution (optional)
        
    Returns:
        True if the feedback was recorded successfully, False otherwise
    """
    metrics = metrics or {}
    
    try:
        response = requests.post(
            f"{HUB_API_BASE}/planner/feedback",
            json={
                "solution_id": solution_id,
                "success": success,
                "metrics": metrics
            }
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Error recording solution outcome: {e}")
        return False


def apply_solution_path(
    solution: SolutionPath,
    action_handlers: Dict[str, Callable[[SolutionAction], bool]]
) -> bool:
    """
    Apply a solution path by executing each action in sequence.
    
    Args:
        solution: The solution path to apply
        action_handlers: A dictionary mapping action types to handler functions
        
    Returns:
        True if all actions were successful, False otherwise
    """
    print(f"Applying solution path {solution.solution_id} for bug {solution.bug_id}")
    
    # Get solution ID as integer (for API calls)
    try:
        solution_id = int(solution.solution_id)
    except ValueError:
        solution_id = hash(solution.solution_id) % 1000000  # Fallback
    
    # Mark solution as in progress
    update_solution_status(solution_id, "IN_PROGRESS")
    
    start_time = time.time()
    success = True
    
    for i, action in enumerate(solution.actions):
        print(f"Executing action {i+1}/{len(solution.actions)}: {action}")
        
        handler = action_handlers.get(action.type)
        if not handler:
            print(f"No handler for action type: {action.type}")
            success = False
            break
        
        try:
            if not handler(action):
                print(f"Action failed: {action}")
                success = False
                break
        except Exception as e:
            print(f"Error executing action: {e}")
            success = False
            break
    
    # Calculate metrics
    execution_time = time.time() - start_time
    metrics = {
        "execution_time": execution_time,
        "num_actions": len(solution.actions),
        "actions_completed": i + 1 if success else i,
    }
    
    # Record outcome
    record_outcome(solution_id, success, metrics)
    
    return success


def create_solution_path(
    bug_id: str, 
    actions: List[Dict[str, Any]], 
    priority: float = 0.5,
    metadata: Dict[str, Any] = None
) -> Optional[str]:
    """
    Create a new solution path in the hub.
    
    Args:
        bug_id: The ID of the bug this solution addresses
        actions: List of action dictionaries (each with type, agent, params)
        priority: Priority score for this solution (0.0 to 1.0)
        metadata: Additional metadata for the solution
        
    Returns:
        The solution_id if creation was successful, None otherwise
    """
    solution_id = str(uuid.uuid4())
    
    try:
        response = requests.post(
            f"{HUB_API_BASE}/planner/solutions",
            json={
                "bug_id": bug_id,
                "solution_id": solution_id,
                "actions": actions,
                "priority": priority,
                "metadata": metadata or {}
            }
        )
        response.raise_for_status()
        return solution_id
    except requests.RequestException as e:
        print(f"Error creating solution path: {e}")
        return None


def generate_dag_based_solution(bug_id: str, file_paths: List[str]) -> Optional[str]:
    """
    Generate a solution path based on DAG analysis and register it with the hub.
    
    This creates a solution that follows the traditional DAG-based patching approach
    but wraps it in the planner solution interface.
    
    Args:
        bug_id: The ID of the bug to create a solution for
        file_paths: List of file paths to analyze for dependencies
        
    Returns:
        The solution_id if successful, None otherwise
    """
    # Build dependency graph and compute optimal patching order
    graph = build_dep_graph(file_paths)
    sccs = tarjan_scc(graph)
    sorted_sccs = ripple_sort(graph, sccs)
    
    # Create actions for each SCC
    actions = []
    for scc in sorted_sccs:
        # Create a PATCH action for each SCC
        actions.append({
            "type": "PATCH",
            "agent": "Analyst",
            "params": {
                "files": scc,
                "strategy": "atomic"
            }
        })
    
    # Add verification actions
    actions.append({
        "type": "TEST",
        "agent": "Verifier",
        "params": {
            "test_type": "unit"
        }
    })
    
    actions.append({
        "type": "VERIFY",
        "agent": "Verifier",
        "params": {
            "verification_type": "integration"
        }
    })
    
    # Register this solution with the hub
    return create_solution_path(
        bug_id=bug_id,
        actions=actions,
        priority=0.7,  # High priority but not maximum
        metadata={
            "source": "dag_analysis",
            "num_files": len(file_paths),
            "num_sccs": len(sorted_sccs)
        }
    )
