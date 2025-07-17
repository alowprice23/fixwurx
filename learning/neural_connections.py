#!/usr/bin/env python3
"""
Family Tree Neural Connections Module

This module provides capabilities for managing neural connections between related concepts,
patterns, and solutions, forming a family tree structure for knowledge representation.
"""

import os
import sys
import json
import logging
import time
import random
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("neural_connections.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NeuralConnections")

class NeuralNode:
    """
    Represents a node in the neural network.
    """
    
    def __init__(self, node_id: str, node_type: str, name: str = None,
               metadata: Dict[str, Any] = None, activation: float = 0.0):
        """
        Initialize neural node.
        
        Args:
            node_id: Unique node ID
            node_type: Type of node (e.g., "pattern", "solution", "concept")
            name: Human-readable name
            metadata: Additional metadata
            activation: Initial activation level
        """
        self.node_id = node_id
        self.node_type = node_type
        self.name = name or node_id
        self.metadata = metadata or {}
        self.activation = activation
        self.connections = {}
        self.creation_time = time.time()
        self.last_updated = time.time()
        self.activation_history = [(time.time(), activation)]
        
        logger.debug(f"Created neural node: {node_id} ({node_type})")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "metadata": self.metadata,
            "activation": self.activation,
            "connections": self.connections,
            "creation_time": self.creation_time,
            "last_updated": self.last_updated,
            "activation_history": self.activation_history[-10:]  # Save only the last 10 activation points
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralNode':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Neural node
        """
        node = cls(
            node_id=data.get("node_id", ""),
            node_type=data.get("node_type", ""),
            name=data.get("name"),
            metadata=data.get("metadata", {}),
            activation=data.get("activation", 0.0)
        )
        
        # Restore timestamps and connections
        node.creation_time = data.get("creation_time", time.time())
        node.last_updated = data.get("last_updated", time.time())
        node.connections = data.get("connections", {})
        node.activation_history = data.get("activation_history", [(time.time(), node.activation)])
        
        return node
    
    def connect_to(self, target_id: str, weight: float = 1.0, connection_type: str = "related") -> None:
        """
        Connect this node to another node.
        
        Args:
            target_id: Target node ID
            weight: Connection weight
            connection_type: Type of connection
        """
        self.connections[target_id] = {
            "weight": weight,
            "type": connection_type,
            "created_at": time.time()
        }
        
        self.last_updated = time.time()
        
        logger.debug(f"Connected node {self.node_id} to {target_id} ({connection_type}, weight: {weight})")
    
    def disconnect_from(self, target_id: str) -> bool:
        """
        Disconnect this node from another node.
        
        Args:
            target_id: Target node ID
            
        Returns:
            Whether the connection was removed
        """
        if target_id in self.connections:
            del self.connections[target_id]
            self.last_updated = time.time()
            
            logger.debug(f"Disconnected node {self.node_id} from {target_id}")
            return True
        
        return False
    
    def update_connection(self, target_id: str, weight: float = None, connection_type: str = None) -> bool:
        """
        Update a connection to another node.
        
        Args:
            target_id: Target node ID
            weight: New connection weight
            connection_type: New connection type
            
        Returns:
            Whether the connection was updated
        """
        if target_id in self.connections:
            if weight is not None:
                self.connections[target_id]["weight"] = weight
            
            if connection_type is not None:
                self.connections[target_id]["type"] = connection_type
            
            self.last_updated = time.time()
            
            logger.debug(f"Updated connection from {self.node_id} to {target_id}")
            return True
        
        return False
    
    def get_connection(self, target_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a connection to another node.
        
        Args:
            target_id: Target node ID
            
        Returns:
            Connection data, or None if not connected
        """
        return self.connections.get(target_id)
    
    def is_connected_to(self, target_id: str) -> bool:
        """
        Check if this node is connected to another node.
        
        Args:
            target_id: Target node ID
            
        Returns:
            Whether the nodes are connected
        """
        return target_id in self.connections
    
    def get_connected_nodes(self, connection_type: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Get connected nodes.
        
        Args:
            connection_type: Filter by connection type
            
        Returns:
            Dictionary mapping node IDs to connection data
        """
        if connection_type is None:
            return self.connections
        
        return {node_id: conn for node_id, conn in self.connections.items() if conn["type"] == connection_type}
    
    def set_activation(self, activation: float) -> float:
        """
        Set activation level.
        
        Args:
            activation: New activation level
            
        Returns:
            New activation level
        """
        self.activation = activation
        self.last_updated = time.time()
        self.activation_history.append((time.time(), activation))
        
        # Trim history if needed
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]
        
        logger.debug(f"Set activation of node {self.node_id} to {activation}")
        
        return activation
    
    def activate(self, amount: float = 1.0, decay: bool = True) -> float:
        """
        Activate this node.
        
        Args:
            amount: Activation amount
            decay: Whether to apply decay to existing activation
            
        Returns:
            New activation level
        """
        if decay:
            # Apply decay to existing activation
            self.activation *= 0.9
        
        # Add new activation
        self.activation += amount
        
        # Ensure activation is within bounds (0.0 to 1.0)
        self.activation = max(0.0, min(1.0, self.activation))
        
        # Update timestamp and history
        self.last_updated = time.time()
        self.activation_history.append((time.time(), self.activation))
        
        # Trim history if needed
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]
        
        logger.debug(f"Activated node {self.node_id} by {amount} to {self.activation}")
        
        return self.activation

class NeuralNetwork:
    """
    Manages neural nodes and connections.
    """
    
    def __init__(self, network_file: str = None):
        """
        Initialize neural network.
        
        Args:
            network_file: Path to network database file
        """
        self.network_file = network_file or "neural_network.json"
        self.nodes = {}
        self.node_types = set()
        self.connection_types = set()
        self.activation_callbacks = []
        self.spreading_factor = 0.5
        self.activation_threshold = 0.1
        
        # Load network if file exists
        if os.path.exists(self.network_file):
            self._load_network()
        
        logger.info("Neural network initialized")
    
    def _load_network(self) -> None:
        """Load network from database file."""
        try:
            with open(self.network_file, "r") as f:
                data = json.load(f)
            
            # Load nodes
            nodes_data = data.get("nodes", {})
            
            for node_id, node_data in nodes_data.items():
                self.nodes[node_id] = NeuralNode.from_dict(node_data)
            
            # Extract node types and connection types
            self.node_types = set(node.node_type for node in self.nodes.values())
            
            self.connection_types = set()
            for node in self.nodes.values():
                for conn in node.connections.values():
                    self.connection_types.add(conn["type"])
            
            # Load configuration
            self.spreading_factor = data.get("spreading_factor", 0.5)
            self.activation_threshold = data.get("activation_threshold", 0.1)
            
            logger.info(f"Loaded {len(self.nodes)} nodes from {self.network_file}")
        except Exception as e:
            logger.error(f"Error loading network from {self.network_file}: {e}")
            self.nodes = {}
            self.node_types = set()
            self.connection_types = set()
    
    def save_network(self) -> None:
        """Save network to database file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.network_file)), exist_ok=True)
            
            data = {
                "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
                "spreading_factor": self.spreading_factor,
                "activation_threshold": self.activation_threshold,
                "last_updated": time.time()
            }
            
            with open(self.network_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.nodes)} nodes to {self.network_file}")
        except Exception as e:
            logger.error(f"Error saving network to {self.network_file}: {e}")
    
    def create_node(self, node_id: str, node_type: str, name: str = None,
                  metadata: Dict[str, Any] = None, activation: float = 0.0) -> NeuralNode:
        """
        Create a new node.
        
        Args:
            node_id: Node ID
            node_type: Node type
            name: Node name
            metadata: Node metadata
            activation: Initial activation
            
        Returns:
            Created node
        """
        # Create node
        node = NeuralNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            metadata=metadata,
            activation=activation
        )
        
        # Add to nodes
        self.nodes[node_id] = node
        
        # Update node types
        self.node_types.add(node_type)
        
        # Save network
        self.save_network()
        
        logger.info(f"Created node: {node_id} ({node_type})")
        
        return node
    
    def get_node(self, node_id: str) -> Optional[NeuralNode]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            Neural node, or None if not found
        """
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[NeuralNode]:
        """
        Get nodes by type.
        
        Args:
            node_type: Node type
            
        Returns:
            List of matching nodes
        """
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def get_nodes_by_activation(self, min_activation: float = 0.0, node_type: str = None) -> List[NeuralNode]:
        """
        Get nodes by activation level.
        
        Args:
            min_activation: Minimum activation level
            node_type: Filter by node type
            
        Returns:
            List of matching nodes
        """
        nodes = self.nodes.values()
        
        if node_type is not None:
            nodes = [node for node in nodes if node.node_type == node_type]
        
        return [node for node in nodes if node.activation >= min_activation]
    
    def connect_nodes(self, source_id: str, target_id: str, weight: float = 1.0,
                    connection_type: str = "related", bidirectional: bool = False) -> bool:
        """
        Connect two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            weight: Connection weight
            connection_type: Connection type
            bidirectional: Whether to create a bidirectional connection
            
        Returns:
            Whether the connection was created
        """
        source_node = self.get_node(source_id)
        target_node = self.get_node(target_id)
        
        if source_node and target_node:
            # Connect source to target
            source_node.connect_to(target_id, weight, connection_type)
            
            # Update connection types
            self.connection_types.add(connection_type)
            
            # Connect target to source if bidirectional
            if bidirectional:
                target_node.connect_to(source_id, weight, connection_type)
            
            # Save network
            self.save_network()
            
            return True
        
        return False
    
    def disconnect_nodes(self, source_id: str, target_id: str, bidirectional: bool = False) -> bool:
        """
        Disconnect two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            bidirectional: Whether to remove the bidirectional connection
            
        Returns:
            Whether the connection was removed
        """
        source_node = self.get_node(source_id)
        target_node = self.get_node(target_id)
        
        if source_node:
            # Disconnect source from target
            source_node.disconnect_from(target_id)
            
            # Disconnect target from source if bidirectional
            if bidirectional and target_node:
                target_node.disconnect_from(source_id)
            
            # Save network
            self.save_network()
            
            return True
        
        return False
    
    def update_connection(self, source_id: str, target_id: str, weight: float = None,
                        connection_type: str = None) -> bool:
        """
        Update a connection between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            weight: New connection weight
            connection_type: New connection type
            
        Returns:
            Whether the connection was updated
        """
        source_node = self.get_node(source_id)
        
        if source_node:
            # Update connection
            updated = source_node.update_connection(target_id, weight, connection_type)
            
            if updated:
                # Update connection types
                if connection_type is not None:
                    self.connection_types.add(connection_type)
                
                # Save network
                self.save_network()
            
            return updated
        
        return False
    
    def get_connection(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a connection between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            Connection data, or None if not connected
        """
        source_node = self.get_node(source_id)
        
        if source_node:
            return source_node.get_connection(target_id)
        
        return None
    
    def is_connected(self, source_id: str, target_id: str) -> bool:
        """
        Check if two nodes are connected.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            Whether the nodes are connected
        """
        source_node = self.get_node(source_id)
        
        if source_node:
            return source_node.is_connected_to(target_id)
        
        return False
    
    def activate_node(self, node_id: str, activation: float, spread: bool = True) -> Optional[Dict[str, float]]:
        """
        Activate a node and optionally spread activation to connected nodes.
        
        Args:
            node_id: Node ID
            activation: Activation amount
            spread: Whether to spread activation to connected nodes
            
        Returns:
            Dictionary mapping node IDs to activation levels
        """
        node = self.get_node(node_id)
        
        if node:
            # Activate node
            node.activate(activation)
            
            # Notify callbacks
            for callback in self.activation_callbacks:
                try:
                    callback(node_id, node.activation)
                except Exception as e:
                    logger.error(f"Error in activation callback: {e}")
            
            # Spread activation if requested
            if spread and node.activation >= self.activation_threshold:
                return self._spread_activation(node)
            
            # Save network
            self.save_network()
            
            return {node_id: node.activation}
        
        return None
    
    def _spread_activation(self, node: NeuralNode) -> Dict[str, float]:
        """
        Spread activation from a node to connected nodes.
        
        Args:
            node: Source node
            
        Returns:
            Dictionary mapping node IDs to activation levels
        """
        activations = {node.node_id: node.activation}
        
        # Get connected nodes
        for target_id, conn in node.connections.items():
            target_node = self.get_node(target_id)
            
            if target_node:
                # Calculate spread amount
                spread_amount = node.activation * self.spreading_factor * conn["weight"]
                
                # Activate target node
                target_node.activate(spread_amount)
                
                # Notify callbacks
                for callback in self.activation_callbacks:
                    try:
                        callback(target_id, target_node.activation)
                    except Exception as e:
                        logger.error(f"Error in activation callback: {e}")
                
                # Record activation
                activations[target_id] = target_node.activation
        
        # Save network
        self.save_network()
        
        return activations
    
    def add_activation_callback(self, callback: Callable[[str, float], None]) -> None:
        """
        Add a callback for node activations.
        
        Args:
            callback: Callback function that receives node ID and activation level
        """
        if callback not in self.activation_callbacks:
            self.activation_callbacks.append(callback)
            logger.debug(f"Added activation callback: {callback.__name__}")
    
    def remove_activation_callback(self, callback: Callable[[str, float], None]) -> bool:
        """
        Remove a callback.
        
        Args:
            callback: Callback function
            
        Returns:
            Whether the callback was removed
        """
        if callback in self.activation_callbacks:
            self.activation_callbacks.remove(callback)
            logger.debug(f"Removed activation callback: {callback.__name__}")
            return True
        
        return False
    
    def reset_activations(self) -> None:
        """Reset all node activations to zero."""
        for node in self.nodes.values():
            node.set_activation(0.0)
        
        logger.info("Reset all node activations")
        
        # Save network
        self.save_network()
    
    def decay_activations(self, decay_factor: float = 0.9) -> None:
        """
        Apply decay to all node activations.
        
        Args:
            decay_factor: Decay factor (0.0 to 1.0)
        """
        for node in self.nodes.values():
            new_activation = node.activation * decay_factor
            node.set_activation(new_activation)
        
        logger.debug(f"Applied decay of {decay_factor} to all node activations")
        
        # Save network
        self.save_network()
    
    def get_node_types(self) -> List[str]:
        """
        Get all node types.
        
        Returns:
            List of node types
        """
        return sorted(self.node_types)
    
    def get_connection_types(self) -> List[str]:
        """
        Get all connection types.
        
        Returns:
            List of connection types
        """
        return sorted(self.connection_types)
    
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Whether the node was deleted
        """
        if node_id in self.nodes:
            node = self.nodes.pop(node_id)
            
            # Remove connections to this node
            for other_node in self.nodes.values():
                other_node.disconnect_from(node_id)
            
            # Update node types
            self.node_types = set(node.node_type for node in self.nodes.values())
            
            # Save network
            self.save_network()
            
            logger.info(f"Deleted node: {node_id}")
            
            return True
        
        return False
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> Optional[List[Tuple[str, str, Dict[str, Any]]]]:
        """
        Find a path between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_depth: Maximum path depth
            
        Returns:
            List of (source_id, target_id, connection_data) tuples, or None if no path found
        """
        if source_id == target_id:
            return []
        
        # Check if source and target nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        # Use breadth-first search to find the shortest path
        visited = {source_id}
        queue = [(source_id, [])]
        
        while queue:
            current_id, path = queue.pop(0)
            current_node = self.nodes[current_id]
            
            # Check if we've reached the target
            if current_id == target_id:
                return path
            
            # Check if we've reached the maximum depth
            if len(path) >= max_depth:
                continue
            
            # Explore connected nodes
            for next_id, conn in current_node.connections.items():
                if next_id not in visited:
                    visited.add(next_id)
                    new_path = path + [(current_id, next_id, conn)]
                    queue.append((next_id, new_path))
        
        # No path found
        return None
    
    def find_related_nodes(self, node_id: str, relation_type: str = None, max_depth: int = 2) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        """
        Find nodes related to a given node.
        
        Args:
            node_id: Node ID
            relation_type: Relation type filter
            max_depth: Maximum relation depth
            
        Returns:
            Dictionary mapping node IDs to lists of (path_node_id, connection_data) tuples
        """
        if node_id not in self.nodes:
            return {}
        
        # Use breadth-first search to find related nodes
        visited = {node_id}
        queue = [(node_id, [], 0)]
        related = {}
        
        while queue:
            current_id, path, depth = queue.pop(0)
            current_node = self.nodes[current_id]
            
            # Add to related nodes if not the source node
            if current_id != node_id:
                if current_id not in related:
                    related[current_id] = []
                
                related[current_id].append((path[-1][0], path[-1][2]))
            
            # Check if we've reached the maximum depth
            if depth >= max_depth:
                continue
            
            # Explore connected nodes
            for next_id, conn in current_node.connections.items():
                # Filter by relation type if specified
                if relation_type is not None and conn["type"] != relation_type:
                    continue
                
                if next_id not in visited:
                    visited.add(next_id)
                    new_path = path + [(current_id, next_id, conn)]
                    queue.append((next_id, new_path, depth + 1))
        
        return related
    
    def create_family_tree(self, root_id: str, relation_types: List[str] = None, max_depth: int = 3) -> Dict[str, Any]:
        """
        Create a family tree starting from a root node.
        
        Args:
            root_id: Root node ID
            relation_types: Relation types to include
            max_depth: Maximum tree depth
            
        Returns:
            Tree structure as nested dictionaries
        """
        if root_id not in self.nodes:
            return {}
        
        root_node = self.nodes[root_id]
        
        # Create tree recursively
        tree = {
            "id": root_id,
            "type": root_node.node_type,
            "name": root_node.name,
            "activation": root_node.activation,
            "children": []
        }
        
        # Track visited nodes to avoid cycles
        visited = {root_id}
        
        # Build tree recursively
        self._build_tree(tree, visited, relation_types, max_depth, 0)
        
        return tree
    
    def _build_tree(self, node_data: Dict[str, Any], visited: Set[str],
                  relation_types: List[str], max_depth: int, current_depth: int) -> None:
        """
        Build a tree recursively.
        
        Args:
            node_data: Current node data
            visited: Set of visited node IDs
            relation_types: Relation types to include
            max_depth: Maximum tree depth
            current_depth: Current depth
        """
        if current_depth >= max_depth:
            return
        
        node_id = node_data["id"]
        node = self.nodes[node_id]
        
        # Get connected nodes
        for child_id, conn in node.connections.items():
            # Skip if already visited
            if child_id in visited:
                continue
            
            # Filter by relation type if specified
            if relation_types is not None and conn["type"] not in relation_types:
                continue
            
            child_node = self.nodes.get(child_id)
            
            if child_node:
                # Add child to tree
                child_data = {
                    "id": child_id,
                    "type": child_node.node_type,
                    "name": child_node.name,
                    "activation": child_node.activation,
                    "relation": conn["type"],
                    "weight": conn["weight"],
                    "children": []
                }
                
                node_data["children"].append(child_data)
                
                # Mark as visited
                visited.add(child_id)
                
                # Recursively build subtree
                self._build_tree(child_data, visited, relation_types, max_depth, current_depth + 1)
    
    def visualize_network(self, node_types: List[str] = None, min_activation: float = 0.0,
                        connection_types: List[str] = None, output_file: str = None) -> str:
        """
        Visualize the neural network.
        
        Args:
            node_types: Node types to include
            min_activation: Minimum activation level
            connection_types: Connection types to include
            output_file: Output file path
            
        Returns:
            Output file path
        """
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            # Filter by node type
            if node_types is not None and node.node_type not in node_types:
                continue
            
            # Filter by activation
            if node.activation < min_activation:
                continue
            
            # Add node
            G.add_node(node_id, type=node.node_type, name=node.name, activation=node.activation)
        
        # Add edges
        for node_id, node in self.nodes.items():
            # Skip if node not in graph
            if node_id not in G:
                continue
            
            for target_id, conn in node.connections.items():
                # Skip if target not in graph
                if target_id not in G:
                    continue
                
                # Filter by connection type
                if connection_types is not None and conn["type"] not in connection_types:
                    continue
                
                # Add edge
                G.add_edge(node_id, target_id, type=conn["type"], weight=conn["weight"])
        
        # Create figure
        plt.figure(figsize=(12, 12))
        
        # Set node positions
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        
        for node_id in G.nodes():
            node = self.nodes[node_id]
            node_colors.append(self._get_node_color(node.node_type))
            node_sizes.append(300 + 700 * node.activation)
