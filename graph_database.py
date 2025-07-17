"""
FixWurx Graph Database

This module implements a graph database for the Auditor agent to track relationships
between components, bugs, and fixes. It provides a foundation for impact analysis
and root cause identification.

See docs/auditor_agent_specification.md for full specification.
"""

import os
import json
import logging
import datetime
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [GraphDB] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('graph_database')


class Node:
    """
    Represents a node in the graph database.
    
    Each node has a unique ID, a type, and a set of properties.
    """
    
    def __init__(self, node_id: str, node_type: str, properties: Dict = None):
        """
        Initialize a node.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of the node (e.g., 'component', 'bug', 'patch')
            properties: Additional properties of the node
        """
        self.id = node_id
        self.type = node_type
        self.properties = properties or {}
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary representation"""
        return {
            "id": self.id,
            "type": self.type,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Node':
        """Create node from dictionary representation"""
        return cls(
            node_id=data["id"],
            node_type=data["type"],
            properties=data.get("properties", {})
        )


class Edge:
    """
    Represents an edge in the graph database.
    
    Each edge connects two nodes and has a type and an optional set of properties.
    """
    
    def __init__(self, source_id: str, target_id: str, edge_type: str, properties: Dict = None):
        """
        Initialize an edge.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            edge_type: Type of the edge (e.g., 'affects', 'fixes', 'depends_on')
            properties: Additional properties of the edge
        """
        self.source_id = source_id
        self.target_id = target_id
        self.type = edge_type
        self.properties = properties or {}
    
    def to_dict(self) -> Dict:
        """Convert edge to dictionary representation"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Edge':
        """Create edge from dictionary representation"""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=data["type"],
            properties=data.get("properties", {})
        )


class GraphDatabase:
    """
    Graph database implementation for the Auditor agent.
    
    Tracks relationships between components, bugs, fixes, and other entities.
    Provides methods for querying the graph and identifying impact and root causes.
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize the graph database.
        
        Args:
            storage_path: Path to the storage directory
        """
        self.storage_path = storage_path
        self.graph = nx.DiGraph()  # Directed graph
        self._load_graph()
    
    def add_node(self, node: Node) -> bool:
        """
        Add a node to the graph.
        
        Args:
            node: The node to add
            
        Returns:
            True if the node was added, False if it already exists
        """
        if node.id in self.graph.nodes:
            logger.warning(f"Node {node.id} already exists")
            return False
        
        self.graph.add_node(node.id, **{"data": node.to_dict()})
        logger.info(f"Added node {node.id} of type {node.type}")
        
        # Save the graph
        self._save_graph()
        
        return True
    
    def update_node(self, node: Node) -> bool:
        """
        Update a node in the graph.
        
        Args:
            node: The node to update
            
        Returns:
            True if the node was updated, False if it doesn't exist
        """
        if node.id not in self.graph.nodes:
            logger.warning(f"Node {node.id} does not exist")
            return False
        
        self.graph.nodes[node.id]["data"] = node.to_dict()
        logger.info(f"Updated node {node.id}")
        
        # Save the graph
        self._save_graph()
        
        return True
    
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node from the graph.
        
        Args:
            node_id: ID of the node to delete
            
        Returns:
            True if the node was deleted, False if it doesn't exist
        """
        if node_id not in self.graph.nodes:
            logger.warning(f"Node {node_id} does not exist")
            return False
        
        self.graph.remove_node(node_id)
        logger.info(f"Deleted node {node_id}")
        
        # Save the graph
        self._save_graph()
        
        return True
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Get a node from the graph.
        
        Args:
            node_id: ID of the node to get
            
        Returns:
            The node, or None if it doesn't exist
        """
        if node_id not in self.graph.nodes:
            logger.warning(f"Node {node_id} does not exist")
            return None
        
        node_data = self.graph.nodes[node_id]["data"]
        return Node.from_dict(node_data)
    
    def add_edge(self, edge: Edge) -> bool:
        """
        Add an edge to the graph.
        
        Args:
            edge: The edge to add
            
        Returns:
            True if the edge was added, False if the source or target node doesn't exist
        """
        if edge.source_id not in self.graph.nodes:
            logger.warning(f"Source node {edge.source_id} does not exist")
            return False
        
        if edge.target_id not in self.graph.nodes:
            logger.warning(f"Target node {edge.target_id} does not exist")
            return False
        
        # Check if edge already exists
        if self.graph.has_edge(edge.source_id, edge.target_id):
            logger.warning(f"Edge from {edge.source_id} to {edge.target_id} already exists")
            return False
        
        self.graph.add_edge(edge.source_id, edge.target_id, **{"data": edge.to_dict()})
        logger.info(f"Added edge from {edge.source_id} to {edge.target_id} of type {edge.type}")
        
        # Save the graph
        self._save_graph()
        
        return True
    
    def update_edge(self, edge: Edge) -> bool:
        """
        Update an edge in the graph.
        
        Args:
            edge: The edge to update
            
        Returns:
            True if the edge was updated, False if it doesn't exist
        """
        if not self.graph.has_edge(edge.source_id, edge.target_id):
            logger.warning(f"Edge from {edge.source_id} to {edge.target_id} does not exist")
            return False
        
        self.graph[edge.source_id][edge.target_id]["data"] = edge.to_dict()
        logger.info(f"Updated edge from {edge.source_id} to {edge.target_id}")
        
        # Save the graph
        self._save_graph()
        
        return True
    
    def delete_edge(self, source_id: str, target_id: str) -> bool:
        """
        Delete an edge from the graph.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            
        Returns:
            True if the edge was deleted, False if it doesn't exist
        """
        if not self.graph.has_edge(source_id, target_id):
            logger.warning(f"Edge from {source_id} to {target_id} does not exist")
            return False
        
        self.graph.remove_edge(source_id, target_id)
        logger.info(f"Deleted edge from {source_id} to {target_id}")
        
        # Save the graph
        self._save_graph()
        
        return True
    
    def get_edge(self, source_id: str, target_id: str) -> Optional[Edge]:
        """
        Get an edge from the graph.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            
        Returns:
            The edge, or None if it doesn't exist
        """
        if not self.graph.has_edge(source_id, target_id):
            logger.warning(f"Edge from {source_id} to {target_id} does not exist")
            return None
        
        edge_data = self.graph[source_id][target_id]["data"]
        return Edge.from_dict(edge_data)
    
    def get_neighbors(self, node_id: str, direction: str = "out", edge_type: str = None) -> List[Node]:
        """
        Get the neighbors of a node.
        
        Args:
            node_id: ID of the node
            direction: "out" for outgoing edges, "in" for incoming edges, "both" for both
            edge_type: Type of edges to consider, or None for all types
            
        Returns:
            List of neighboring nodes
        """
        if node_id not in self.graph.nodes:
            logger.warning(f"Node {node_id} does not exist")
            return []
        
        neighbors = []
        
        # Get outgoing neighbors
        if direction in ["out", "both"]:
            for neighbor_id in self.graph.successors(node_id):
                edge = self.get_edge(node_id, neighbor_id)
                if edge_type is None or edge.type == edge_type:
                    neighbor = self.get_node(neighbor_id)
                    if neighbor:
                        neighbors.append(neighbor)
        
        # Get incoming neighbors
        if direction in ["in", "both"]:
            for neighbor_id in self.graph.predecessors(node_id):
                edge = self.get_edge(neighbor_id, node_id)
                if edge_type is None or edge.type == edge_type:
                    neighbor = self.get_node(neighbor_id)
                    if neighbor:
                        neighbors.append(neighbor)
        
        return neighbors
    
    def find_path(self, source_id: str, target_id: str) -> List[Tuple[Node, Edge]]:
        """
        Find a path from source node to target node.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            
        Returns:
            List of (node, edge) tuples representing the path, or empty list if no path exists
        """
        if source_id not in self.graph.nodes:
            logger.warning(f"Source node {source_id} does not exist")
            return []
        
        if target_id not in self.graph.nodes:
            logger.warning(f"Target node {target_id} does not exist")
            return []
        
        try:
            # Find shortest path
            path_nodes = nx.shortest_path(self.graph, source_id, target_id)
            
            if not path_nodes or len(path_nodes) < 2:
                return []
            
            # Convert to list of (node, edge) tuples
            result = []
            for i in range(len(path_nodes) - 1):
                node = self.get_node(path_nodes[i])
                edge = self.get_edge(path_nodes[i], path_nodes[i + 1])
                result.append((node, edge))
            
            # Add the final node
            result.append((self.get_node(path_nodes[-1]), None))
            
            return result
        except nx.NetworkXNoPath:
            logger.info(f"No path from {source_id} to {target_id}")
            return []
    
    def find_impacted_components(self, node_id: str) -> List[Node]:
        """
        Find components impacted by a node (e.g., a bug or patch).
        
        Args:
            node_id: ID of the node
            
        Returns:
            List of impacted component nodes
        """
        if node_id not in self.graph.nodes:
            logger.warning(f"Node {node_id} does not exist")
            return []
        
        impacted = []
        
        # BFS to find impacted components
        visited = set()
        queue = [node_id]
        
        while queue:
            current_id = queue.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Get outgoing neighbors
            for neighbor_id in self.graph.successors(current_id):
                # Check if neighbor is a component
                neighbor = self.get_node(neighbor_id)
                if neighbor and neighbor.type == "component":
                    impacted.append(neighbor)
                
                # Add to queue for further exploration
                queue.append(neighbor_id)
        
        return impacted
    
    def find_root_causes(self, node_id: str) -> List[Node]:
        """
        Find potential root causes of a node (e.g., a bug or issue).
        
        Args:
            node_id: ID of the node
            
        Returns:
            List of potential root cause nodes
        """
        if node_id not in self.graph.nodes:
            logger.warning(f"Node {node_id} does not exist")
            return []
        
        # Get node
        node = self.get_node(node_id)
        if not node:
            return []
        
        # If not a bug or issue, return empty list
        if node.type not in ["bug", "issue"]:
            logger.warning(f"Node {node_id} is not a bug or issue")
            return []
        
        # Find root causes
        root_causes = []
        
        # BFS to find root causes
        visited = set()
        queue = [node_id]
        
        while queue:
            current_id = queue.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Get incoming neighbors
            for neighbor_id in self.graph.predecessors(current_id):
                # Get edge
                edge = self.get_edge(neighbor_id, current_id)
                
                # If edge type is "causes", add neighbor to root causes
                if edge and edge.type == "causes":
                    neighbor = self.get_node(neighbor_id)
                    if neighbor:
                        root_causes.append(neighbor)
                
                # Add to queue for further exploration
                queue.append(neighbor_id)
        
        return root_causes
    
    def find_related_bugs(self, component_id: str) -> List[Node]:
        """
        Find bugs related to a component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            List of related bug nodes
        """
        if component_id not in self.graph.nodes:
            logger.warning(f"Component {component_id} does not exist")
            return []
        
        # Get component
        component = self.get_node(component_id)
        if not component or component.type != "component":
            logger.warning(f"Node {component_id} is not a component")
            return []
        
        # Find related bugs
        related_bugs = []
        
        # BFS to find related bugs
        visited = set()
        queue = [component_id]
        
        while queue:
            current_id = queue.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Get incoming neighbors (bugs affecting the component)
            for neighbor_id in self.graph.predecessors(current_id):
                # Get neighbor
                neighbor = self.get_node(neighbor_id)
                
                # If neighbor is a bug, add to related bugs
                if neighbor and neighbor.type == "bug":
                    related_bugs.append(neighbor)
                
                # Add to queue for further exploration
                queue.append(neighbor_id)
        
        return related_bugs
    
    def find_fix_history(self, component_id: str) -> List[Tuple[Node, datetime.datetime]]:
        """
        Find the fix history of a component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            List of (fix_node, timestamp) tuples, sorted by timestamp
        """
        if component_id not in self.graph.nodes:
            logger.warning(f"Component {component_id} does not exist")
            return []
        
        # Get component
        component = self.get_node(component_id)
        if not component or component.type != "component":
            logger.warning(f"Node {component_id} is not a component")
            return []
        
        # Find fixes
        fixes = []
        
        # Find all patches that affect the component
        for node_id in self.graph.nodes:
            node = self.get_node(node_id)
            
            if node and node.type == "patch":
                # Check if patch affects the component
                if self.graph.has_edge(node_id, component_id):
                    # Get timestamp
                    timestamp_str = node.properties.get("timestamp")
                    if timestamp_str:
                        try:
                            timestamp = datetime.datetime.fromisoformat(timestamp_str)
                            fixes.append((node, timestamp))
                        except ValueError:
                            logger.warning(f"Invalid timestamp in patch {node_id}: {timestamp_str}")
        
        # Sort by timestamp
        fixes.sort(key=lambda x: x[1])
        
        return fixes
    
    def _load_graph(self) -> None:
        """Load graph from storage"""
        try:
            # Ensure storage path exists
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Path to graph file
            graph_file = os.path.join(self.storage_path, "graph.json")
            
            if os.path.exists(graph_file):
                with open(graph_file, 'r') as f:
                    data = json.load(f)
                
                # Create a new graph
                self.graph = nx.DiGraph()
                
                # Add nodes
                for node_data in data.get("nodes", []):
                    node = Node.from_dict(node_data)
                    self.graph.add_node(node.id, **{"data": node_data})
                
                # Add edges
                for edge_data in data.get("edges", []):
                    edge = Edge.from_dict(edge_data)
                    self.graph.add_edge(edge.source_id, edge.target_id, **{"data": edge_data})
                
                logger.info(f"Loaded graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            else:
                logger.info("No existing graph found, starting with empty graph")
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            # Start with empty graph
            self.graph = nx.DiGraph()
    
    def _save_graph(self) -> None:
        """Save graph to storage"""
        try:
            # Ensure storage path exists
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Path to graph file
            graph_file = os.path.join(self.storage_path, "graph.json")
            
            # Extract nodes and edges
            nodes = []
            for node_id in self.graph.nodes:
                node_data = self.graph.nodes[node_id]["data"]
                nodes.append(node_data)
            
            edges = []
            for source_id, target_id in self.graph.edges:
                edge_data = self.graph[source_id][target_id]["data"]
                edges.append(edge_data)
            
            # Create data
            data = {
                "nodes": nodes,
                "edges": edges
            }
            
            # Save to file
            with open(graph_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved graph with {len(nodes)} nodes and {len(edges)} edges")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")


# Example usage
if __name__ == "__main__":
    # Create graph database
    db = GraphDatabase("graph_data")
    
    # Add some nodes
    component1 = Node("component1", "component", {"name": "resource_manager.py"})
    component2 = Node("component2", "component", {"name": "load_balancer.py"})
    bug1 = Node("bug1", "bug", {"description": "Memory leak"})
    patch1 = Node("patch1", "patch", {"description": "Fix memory leak", "timestamp": datetime.datetime.now().isoformat()})
    
    db.add_node(component1)
    db.add_node(component2)
    db.add_node(bug1)
    db.add_node(patch1)
    
    # Add some edges
    db.add_edge(Edge("bug1", "component1", "affects"))
    db.add_edge(Edge("patch1", "bug1", "fixes"))
    db.add_edge(Edge("patch1", "component1", "modifies"))
    
    # Find impacted components
    impacted = db.find_impacted_components("bug1")
    print(f"Components impacted by bug1: {[c.properties['name'] for c in impacted]}")
    
    # Find related bugs
    related = db.find_related_bugs("component1")
    print(f"Bugs related to component1: {[b.properties['description'] for b in related]}")
    
    # Find fix history
    fixes = db.find_fix_history("component1")
    print(f"Fix history for component1: {[f[0].properties['description'] for f in fixes]}")
