#!/usr/bin/env python3
"""
Neural Matrix Visualization

This module provides visualization capabilities for neural matrix patterns.
It supports ASCII-based visualizations in the terminal and exports to various formats.
"""

import os
import sys
import json
import math
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger("NeuralMatrixVisualization")

class NeuralMatrixVisualizer:
    """Neural Matrix visualization capabilities."""
    
    def __init__(self, matrix=None):
        """
        Initialize the visualizer.
        
        Args:
            matrix: Neural matrix instance (optional)
        """
        self.matrix = matrix
        self.colors_enabled = True
        self.unicode_enabled = True
        self.max_width = 80
        self.max_height = 40
    
    def set_matrix(self, matrix):
        """Set the neural matrix instance."""
        self.matrix = matrix
    
    def disable_colors(self):
        """Disable color output."""
        self.colors_enabled = False
    
    def enable_colors(self):
        """Enable color output."""
        self.colors_enabled = True
    
    def disable_unicode(self):
        """Disable Unicode characters."""
        self.unicode_enabled = False
    
    def enable_unicode(self):
        """Enable Unicode characters."""
        self.unicode_enabled = True
    
    def set_dimensions(self, width: int, height: int):
        """
        Set maximum dimensions for visualizations.
        
        Args:
            width: Maximum width
            height: Maximum height
        """
        self.max_width = max(20, width)
        self.max_height = max(10, height)
    
    def visualize_pattern(self, pattern_id: str = None, pattern_data: Dict = None) -> str:
        """
        Visualize a neural matrix pattern.
        
        Args:
            pattern_id: Pattern ID (optional if pattern_data provided)
            pattern_data: Pattern data (optional if pattern_id provided)
            
        Returns:
            ASCII visualization as a string
        """
        if not self.matrix and not pattern_data:
            return "Error: No neural matrix instance or pattern data provided"
        
        # Get pattern data
        if pattern_id and self.matrix:
            try:
                pattern_data = self.matrix.get_pattern(pattern_id)
            except Exception as e:
                return f"Error getting pattern: {e}"
        
        if not pattern_data:
            return f"Pattern not found: {pattern_id}"
        
        # Extract visualization data
        try:
            pattern_type = pattern_data.get("type", "unknown")
            structure = pattern_data.get("structure", {})
            
            if pattern_type == "bug":
                return self._visualize_bug_pattern(structure)
            elif pattern_type == "solution":
                return self._visualize_solution_pattern(structure)
            elif pattern_type == "code":
                return self._visualize_code_pattern(structure)
            else:
                return self._visualize_generic_pattern(structure)
        except Exception as e:
            return f"Error visualizing pattern: {e}"
    
    def visualize_family_tree(self, family_name: str = None) -> str:
        """
        Visualize a neural matrix family tree.
        
        Args:
            family_name: Family name (optional)
            
        Returns:
            ASCII visualization as a string
        """
        if not self.matrix:
            return "Error: No neural matrix instance provided"
        
        try:
            # Get family data
            if family_name:
                family_data = self.matrix.get_family(family_name)
                if not family_data:
                    return f"Family not found: {family_name}"
                families = [family_data]
            else:
                families = self.matrix.get_families()
            
            # Generate visualization
            result = []
            
            for family in families:
                name = family.get("name", "Unknown")
                result.append(f"Family: {name}")
                result.append("=" * len(f"Family: {name}"))
                
                # Add family description
                description = family.get("description", "No description")
                result.append(description)
                result.append("")
                
                # Add pattern relationships
                patterns = family.get("patterns", [])
                if not patterns:
                    result.append("No patterns in this family")
                    continue
                
                # Build relationship tree
                tree = self._build_family_tree(patterns)
                tree_viz = self._render_tree(tree)
                result.append(tree_viz)
                result.append("")
            
            return "\n".join(result)
        except Exception as e:
            return f"Error visualizing family tree: {e}"
    
    def visualize_weights(self, family_name: str = None) -> str:
        """
        Visualize neural matrix weights.
        
        Args:
            family_name: Family name (optional)
            
        Returns:
            ASCII visualization as a string
        """
        if not self.matrix:
            return "Error: No neural matrix instance provided"
        
        try:
            # Get weights
            weight_filter = {}
            if family_name:
                weight_filter["family"] = family_name
            
            weights = self.matrix.get_weights(filter=weight_filter)
            
            if not weights:
                return "No weights found"
            
            # Group weights by family
            families = {}
            for weight in weights:
                family = weight.get("family", "Unknown")
                if family not in families:
                    families[family] = []
                families[family].append(weight)
            
            # Generate visualization
            result = []
            
            for family, family_weights in families.items():
                result.append(f"Family: {family}")
                result.append("=" * len(f"Family: {family}"))
                
                # Create weight table
                table = []
                max_category_len = max(len(w.get("category", "")) for w in family_weights)
                
                for weight in family_weights:
                    category = weight.get("category", "Unknown")
                    value = weight.get("value", 0)
                    
                    # Create bar representation
                    bar_width = int(min(value * 40, 40))
                    bar = "█" * bar_width if self.unicode_enabled else "#" * bar_width
                    
                    # Add color if enabled
                    if self.colors_enabled:
                        if value > 0.8:
                            bar = f"\033[92m{bar}\033[0m"  # Green
                        elif value > 0.5:
                            bar = f"\033[93m{bar}\033[0m"  # Yellow
                        elif value > 0.2:
                            bar = f"\033[94m{bar}\033[0m"  # Blue
                        else:
                            bar = f"\033[91m{bar}\033[0m"  # Red
                    
                    # Add to table
                    table.append(f"  {category.ljust(max_category_len)} │ {value:.4f} │ {bar}")
                
                result.extend(table)
                result.append("")
            
            return "\n".join(result)
        except Exception as e:
            return f"Error visualizing weights: {e}"
    
    def visualize_activations(self, pattern_id: str = None) -> str:
        """
        Visualize neural matrix pattern activations.
        
        Args:
            pattern_id: Pattern ID to filter by (optional)
            
        Returns:
            ASCII visualization as a string
        """
        if not self.matrix:
            return "Error: No neural matrix instance provided"
        
        try:
            # Get activations
            if pattern_id:
                activations = self.matrix.get_pattern_activations(pattern_id)
            else:
                activations = self.matrix.get_recent_activations()
            
            if not activations:
                return "No activations found"
            
            # Generate visualization
            result = []
            result.append("Pattern Activations")
            result.append("=" * 20)
            
            # Create activation grid
            grid_height = min(len(activations), self.max_height)
            grid_width = self.max_width
            
            grid = []
            for _ in range(grid_height):
                grid.append([" "] * grid_width)
            
            # Place activations on grid
            for i, activation in enumerate(activations[:grid_height]):
                pattern_id = activation.get("pattern_id", "Unknown")
                similarity = activation.get("similarity", 0)
                timestamp = activation.get("timestamp", "Unknown")
                
                # Position based on similarity and time
                x = int((i / grid_height) * grid_width)
                y = i
                
                # Character based on similarity
                if self.unicode_enabled:
                    if similarity > 0.8:
                        char = "●"
                    elif similarity > 0.5:
                        char = "◉"
                    elif similarity > 0.2:
                        char = "○"
                    else:
                        char = "·"
                else:
                    if similarity > 0.8:
                        char = "@"
                    elif similarity > 0.5:
                        char = "O"
                    elif similarity > 0.2:
                        char = "o"
                    else:
                        char = "."
                
                # Add color if enabled
                if self.colors_enabled:
                    if similarity > 0.8:
                        char = f"\033[92m{char}\033[0m"  # Green
                    elif similarity > 0.5:
                        char = f"\033[93m{char}\033[0m"  # Yellow
                    elif similarity > 0.2:
                        char = f"\033[94m{char}\033[0m"  # Blue
                    else:
                        char = f"\033[91m{char}\033[0m"  # Red
                
                # Place on grid
                grid[y][x] = char
                
                # Add pattern ID
                id_start = max(0, x - len(pattern_id) // 2)
                for j, c in enumerate(pattern_id[:10]):
                    if 0 <= id_start + j < grid_width:
                        grid[y][id_start + j] = c
            
            # Render grid
            for row in grid:
                result.append("".join(row))
            
            return "\n".join(result)
        except Exception as e:
            return f"Error visualizing activations: {e}"
    
    def export_visualization(self, 
                             visualization: str, 
                             output_path: str, 
                             format: str = "text") -> bool:
        """
        Export a visualization to a file.
        
        Args:
            visualization: Visualization string
            output_path: Output file path
            format: Output format (text, html, svg)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Export based on format
            if format == "text":
                with open(output_path, "w") as f:
                    f.write(visualization)
            elif format == "html":
                html = self._convert_to_html(visualization)
                with open(output_path, "w") as f:
                    f.write(html)
            elif format == "svg":
                svg = self._convert_to_svg(visualization)
                with open(output_path, "w") as f:
                    f.write(svg)
            else:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error exporting visualization: {e}")
            return False
    
    def _visualize_bug_pattern(self, structure: Dict) -> str:
        """Visualize a bug pattern."""
        result = []
        result.append("Bug Pattern")
        result.append("=" * 10)
        
        # Extract bug details
        bug_type = structure.get("bug_type", "Unknown")
        severity = structure.get("severity", "Unknown")
        context = structure.get("context", {})
        
        result.append(f"Type: {bug_type}")
        result.append(f"Severity: {severity}")
        
        # Add context visualization
        if context:
            result.append("\nContext:")
            for key, value in context.items():
                result.append(f"  {key}: {value}")
        
        # Add code snippet if available
        code = structure.get("code", "")
        if code:
            result.append("\nCode Snippet:")
            result.append("```")
            result.append(code)
            result.append("```")
        
        return "\n".join(result)
    
    def _visualize_solution_pattern(self, structure: Dict) -> str:
        """Visualize a solution pattern."""
        result = []
        result.append("Solution Pattern")
        result.append("=" * 16)
        
        # Extract solution details
        solution_type = structure.get("solution_type", "Unknown")
        bug = structure.get("bug", {})
        steps = structure.get("steps", [])
        
        result.append(f"Type: {solution_type}")
        
        # Add bug reference
        if bug:
            result.append("\nRelated Bug:")
            bug_type = bug.get("bug_type", "Unknown")
            severity = bug.get("severity", "Unknown")
            result.append(f"  Type: {bug_type}")
            result.append(f"  Severity: {severity}")
        
        # Add solution steps
        if steps:
            result.append("\nSolution Steps:")
            for i, step in enumerate(steps, 1):
                result.append(f"  {i}. {step}")
        
        # Add code changes if available
        before = structure.get("before_code", "")
        after = structure.get("after_code", "")
        
        if before and after:
            result.append("\nCode Changes:")
            result.append("Before:")
            result.append("```")
            result.append(before)
            result.append("```")
            
            result.append("After:")
            result.append("```")
            result.append(after)
            result.append("```")
        
        return "\n".join(result)
    
    def _visualize_code_pattern(self, structure: Dict) -> str:
        """Visualize a code pattern."""
        result = []
        result.append("Code Pattern")
        result.append("=" * 12)
        
        # Extract code details
        pattern_name = structure.get("name", "Unknown")
        language = structure.get("language", "Unknown")
        code = structure.get("code", "")
        
        result.append(f"Name: {pattern_name}")
        result.append(f"Language: {language}")
        
        # Add code
        if code:
            result.append("\nPattern:")
            result.append("```")
            result.append(code)
            result.append("```")
        
        # Add examples if available
        examples = structure.get("examples", [])
        if examples:
            result.append("\nExamples:")
            for i, example in enumerate(examples, 1):
                result.append(f"Example {i}:")
                result.append("```")
                result.append(example)
                result.append("```")
        
        return "\n".join(result)
    
    def _visualize_generic_pattern(self, structure: Dict) -> str:
        """Visualize a generic pattern."""
        result = []
        result.append("Pattern Visualization")
        result.append("=" * 20)
        
        # Create a simple tree view of the structure
        def add_structure(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        result.append(f"{prefix}{key}:")
                        add_structure(value, prefix + "  ")
                    else:
                        result.append(f"{prefix}{key}: {value}")
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, (dict, list)):
                        result.append(f"{prefix}{i}:")
                        add_structure(item, prefix + "  ")
                    else:
                        result.append(f"{prefix}- {item}")
            else:
                result.append(f"{prefix}{data}")
        
        add_structure(structure)
        
        return "\n".join(result)
    
    def _build_family_tree(self, patterns: List[Dict]) -> Dict:
        """Build a family tree from patterns."""
        # Create tree structure
        tree = {"name": "Root", "children": []}
        
        # Add patterns to tree
        for pattern in patterns:
            pattern_id = pattern.get("id", "Unknown")
            pattern_type = pattern.get("type", "Unknown")
            
            # Find parent node
            parent = None
            related = pattern.get("related_patterns", [])
            
            if related:
                # Find first related pattern in the tree
                for rel_id in related:
                    parent = self._find_node(tree, rel_id)
                    if parent:
                        break
            
            # If no parent found, add to root
            if not parent:
                parent = tree
            
            # Add node
            node = {
                "name": pattern_id,
                "type": pattern_type,
                "children": []
            }
            
            parent["children"].append(node)
        
        return tree
    
    def _find_node(self, tree: Dict, node_name: str) -> Optional[Dict]:
        """Find a node in the tree by name."""
        if tree["name"] == node_name:
            return tree
        
        for child in tree.get("children", []):
            result = self._find_node(child, node_name)
            if result:
                return result
        
        return None
    
    def _render_tree(self, tree: Dict, prefix: str = "", is_last: bool = True) -> str:
        """Render a tree structure as ASCII."""
        lines = []
        
        # Add root node
        node_name = tree.get("name", "Unknown")
        node_type = tree.get("type", "")
        
        type_str = f" ({node_type})" if node_type else ""
        lines.append(f"{prefix}{'└── ' if is_last else '├── '}{node_name}{type_str}")
        
        # Add children
        children = tree.get("children", [])
        count = len(children)
        
        for i, child in enumerate(children):
            is_last_child = i == count - 1
            new_prefix = f"{prefix}{'    ' if is_last else '│   '}"
            child_lines = self._render_tree(child, new_prefix, is_last_child)
            lines.append(child_lines)
        
        return "\n".join(lines)
    
    def _convert_to_html(self, visualization: str) -> str:
        """Convert ASCII visualization to HTML."""
        # Replace special characters
        html = visualization.replace("<", "&lt;").replace(">", "&gt;")
        
        # Convert newlines to <br>
        html = html.replace("\n", "<br>\n")
        
        # Replace code blocks
        html = html.replace("```", "<pre>")
        html = html.replace("```", "</pre>")
        
        # Create HTML document
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Neural Matrix Visualization</title>
    <style>
        body {{ font-family: monospace; white-space: pre; }}
        pre {{ background-color: #f0f0f0; padding: 10px; }}
    </style>
</head>
<body>
{html}
</body>
</html>
"""
    
    def _convert_to_svg(self, visualization: str) -> str:
        """Convert ASCII visualization to SVG."""
        lines = visualization.split("\n")
        height = len(lines) * 20  # 20px per line
        max_width = max(len(line) for line in lines) * 8  # 8px per character
        
        # Create SVG header
        svg = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{max_width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
"""
        
        # Add lines as text elements
        for i, line in enumerate(lines):
            y = (i + 1) * 20
            svg += f'  <text x="0" y="{y}" font-family="monospace">{line}</text>\n'
        
        # Close SVG
        svg += "</svg>"
        
        return svg


# Command handler for neural matrix visualization
def visualize_command(args: str) -> int:
    """
    Visualize neural matrix patterns.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize neural matrix patterns")
    parser.add_argument("type", nargs="?", choices=["pattern", "family", "weights", "activations"], 
                        default="pattern", help="Type of visualization")
    parser.add_argument("id", nargs="?", help="Pattern ID or family name")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--no-unicode", action="store_true", help="Disable Unicode characters")
    parser.add_argument("--export", help="Export visualization to file")
    parser.add_argument("--format", choices=["text", "html", "svg"], default="text", 
                       help="Export format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    viz_type = cmd_args.type
    viz_id = cmd_args.id
    no_color = cmd_args.no_color
    no_unicode = cmd_args.no_unicode
    export_path = cmd_args.export
    export_format = cmd_args.format
    
    # Get registry and neural matrix
    registry = sys.modules.get("__main__").registry
    neural_matrix = registry.get_component("neural_matrix")
    
    if not neural_matrix:
        print("Error: Neural Matrix not available")
        return 1
    
    # Create visualizer
    visualizer = NeuralMatrixVisualizer(neural_matrix)
    
    # Configure visualizer
    if no_color:
        visualizer.disable_colors()
    
    if no_unicode:
        visualizer.disable_unicode()
    
    try:
        # Generate visualization
        if viz_type == "pattern":
            visualization = visualizer.visualize_pattern(pattern_id=viz_id)
        elif viz_type == "family":
            visualization = visualizer.visualize_family_tree(family_name=viz_id)
        elif viz_type == "weights":
            visualization = visualizer.visualize_weights(family_name=viz_id)
        elif viz_type == "activations":
            visualization = visualizer.visualize_activations(pattern_id=viz_id)
        else:
            print(f"Unknown visualization type: {viz_type}")
            return 1
        
        # Display visualization
        print(visualization)
        
        # Export if requested
        if export_path:
            success = visualizer.export_visualization(
                visualization, export_path, format=export_format)
            
            if success:
                print(f"Visualization exported to {export_path}")
            else:
                print(f"Error exporting visualization to {export_path}")
                return 1
        
        return 0
    except Exception as e:
        print(f"Error visualizing neural matrix: {e}")
        return 1


# Register visualization command in neural_matrix_commands.py
def register_visualization_command(registry):
    """Register visualization command."""
    registry.register_command_handler("visualize", visualize_command, "neural_matrix")
    registry.register_command_handler("neural_matrix:visualize", visualize_command, "neural_matrix")
    
    logger.info("Neural Matrix visualization command registered")
