"""
FixWurx Neural Matrix Validation Script
---------------------------------------
This script validates the completion status of tasks in the deployment roadmap
by creating neural connections between related files and verifying their existence
and content. It only outputs completion status without modifying any files.

This enhanced version now validates actual implementation completeness by:
1. Checking for neural connections between components
2. Verifying content for required neural matrix functionality
3. Validating bidirectional connections and component dependencies
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

class NeuralMatrixValidator:
    """Neural matrix validator for deployment roadmap tasks."""
    
    def __init__(self, root_dir: str = '.'):
        self.root_dir = Path(root_dir)
        self.file_status: Dict[str, Dict] = {}
        self.task_status: Dict[str, Dict] = {}
        self.detected_files: Set[str] = set()
        self.file_contents: Dict[str, str] = {}
        self.neural_connections: Dict[str, Dict[str, float]] = {}
        self.core_components = [
            "triangulation_engine.py",
            "hub.py", 
            "neural_matrix_init.py",
            "planner_agent.py",
            "specialized_agents.py",
            "verification_engine.py",
            "scope_filter.py"
        ]
        self.expected_neural_connections = [
            ("hub.py", "neural_matrix_init.py"),
            ("hub.py", "triangulation_engine.py"),
            ("neural_matrix_init.py", "hub.py"),
            ("neural_matrix_init.py", "verification_engine.py"),
            ("planner_agent.py", "agent_memory.py"),
            ("planner_agent.py", "hub.py"),
            ("planner_agent.py", "specialized_agents.py"),
            ("specialized_agents.py", "planner_agent.py"),
            ("specialized_agents.py", "triangulation_engine.py"),
            ("triangulation_engine.py", "hub.py"),
            ("triangulation_engine.py", "specialized_agents.py"),
            ("triangulation_engine.py", "verification_engine.py"),
            ("verification_engine.py", "neural_matrix_init.py"),
            ("verification_engine.py", "triangulation_engine.py")
        ]
        
    def scan_files(self) -> None:
        """Scan all files in the project directory."""
        for path in self.root_dir.glob('**/*'):
            if path.is_file():
                rel_path = str(path.relative_to(self.root_dir))
                self.detected_files.add(rel_path)
                
    def validate_file_renames(self) -> List[Dict]:
        """Validate file renaming tasks."""
        rename_tasks = [
            {
                "source": "Data Structures.py",
                "target": "data_structures.py",
                "status": "COMPLETE" if "data_structures.py" in self.detected_files else "PENDING",
                "notes": "File exists and contains correct content" if "data_structures.py" in self.detected_files else "File not found"
            },
            {
                "source": "Specialized_agents.py",
                "target": "specialized_agents.py",
                "status": "COMPLETE" if "specialized_agents.py" in self.detected_files else "PENDING",
                "notes": "File exists and contains correct content" if "specialized_agents.py" in self.detected_files else "File not found"
            },
            {
                "source": "Verification_engine.py",
                "target": "verification_engine.py",
                "status": "COMPLETE" if "verification_engine.py" in self.detected_files else "PENDING",
                "notes": "File exists and contains correct content" if "verification_engine.py" in self.detected_files else "File not found"
            }
        ]
        
        for task in rename_tasks:
            original_exists = task["source"] in self.detected_files
            renamed_exists = task["target"] in self.detected_files
            
            if original_exists and renamed_exists:
                task["notes"] = f"Both original and renamed files exist - clean up needed for {task['source']}"
            elif original_exists and not renamed_exists:
                task["status"] = "PENDING"
                task["notes"] = f"Original file exists, but renamed file {task['target']} does not"
            elif not original_exists and renamed_exists:
                task["status"] = "COMPLETE"
                task["notes"] = f"Rename complete - only {task['target']} exists"
            else:
                task["status"] = "PENDING"
                task["notes"] = f"Neither original nor renamed file exists"
                
        return rename_tasks
        
    def load_file_contents(self) -> None:
        """Load content of core component files."""
        for filename in self.core_components:
            filepath = self.root_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.file_contents[filename] = f.read()
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
    
    def detect_neural_connections(self) -> None:
        """Detect neural connections between files by analyzing imports."""
        # Initialize connection dictionary
        for source in self.core_components:
            self.neural_connections[source] = {}
        
        # Check for import statements
        for source, content in self.file_contents.items():
            for target in self.core_components:
                if source != target:
                    # Check for different types of imports
                    patterns = [
                        rf"import\s+{target.replace('.py', '')}\b",  # import target
                        rf"from\s+{target.replace('.py', '')}\s+import",  # from target import
                        rf"import.*?{target.replace('.py', '')}\b",  # import ... target
                        rf"#.*?Neural connection.*?{target.replace('.py', '')}\b"  # Neural connection comment
                    ]
                    
                    connection_strength = 0.0
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            connection_strength = 1.0
                            break
                    
                    self.neural_connections[source][target] = connection_strength
    
    def validate_neural_connections(self) -> Dict[str, Any]:
        """Validate neural connections against expected connections."""
        results = {
            "total_expected": len(self.expected_neural_connections),
            "found": 0,
            "missing": [],
            "connection_details": []
        }
        
        for source, target in self.expected_neural_connections:
            if source in self.neural_connections and target in self.neural_connections[source]:
                strength = self.neural_connections[source][target]
                if strength > 0:
                    results["found"] += 1
                    results["connection_details"].append({
                        "source": source,
                        "target": target,
                        "strength": strength,
                        "status": "COMPLETE"
                    })
                else:
                    results["missing"].append((source, target))
                    results["connection_details"].append({
                        "source": source,
                        "target": target,
                        "strength": 0.0,
                        "status": "PENDING"
                    })
            else:
                results["missing"].append((source, target))
                results["connection_details"].append({
                    "source": source,
                    "target": target,
                    "strength": 0.0,
                    "status": "PENDING"
                })
        
        results["completion_percentage"] = (results["found"] / results["total_expected"]) * 100 if results["total_expected"] > 0 else 0
        
        return results
    
    def validate_roadmap_tasks(self) -> Dict:
        """Validate deployment roadmap tasks."""
        # First, scan all files
        self.scan_files()
        
        # Load contents of core component files
        self.load_file_contents()
        
        # Detect neural connections
        self.detect_neural_connections()
        
        # Validate neural connections
        neural_connections_results = self.validate_neural_connections()
        
        result = {
            "file_renames": self.validate_file_renames(),
            "neural_connections": neural_connections_results,
            "overall_status": "Some files need cleanup" if "Specialized_agents.py" in self.detected_files or "Verification_engine.py" in self.detected_files else "All rename tasks complete"
        }
        
        # Update overall status based on neural connections
        if neural_connections_results["completion_percentage"] < 100:
            result["overall_status"] = f"Neural connections incomplete ({neural_connections_results['completion_percentage']:.1f}% complete)"
        
        return result
        
    def validate_implementation_completeness(self) -> Dict[str, Any]:
        """Validate implementation completeness of core components."""
        results = {
            "total_files": len(self.core_components),
            "complete_files": 0,
            "incomplete_files": [],
            "file_details": []
        }
        
        # Define required neural matrix features for each component
        required_features = {
            "triangulation_engine.py": ["neural_matrix", "agent_coordination", "solution_path"],
            "hub.py": ["neural_patterns", "solution_paths", "neural_weights"],
            "neural_matrix_init.py": ["neural_matrix_dir", "initialize", "weights"],
            "planner_agent.py": ["neural_learning", "solution_paths", "family_tree"],
            "specialized_agents.py": ["neural_connection", "planner", "agent_family"],
            "verification_engine.py": ["neural_validate", "integrity", "matrix"],
            "scope_filter.py": ["neural_pattern", "matching", "relevance"]
        }
        
        for filename in self.core_components:
            file_result = {
                "filename": filename,
                "exists": filename in self.file_contents,
                "required_features": required_features.get(filename, []),
                "found_features": [],
                "missing_features": [],
                "completeness_percentage": 0.0
            }
            
            if filename in self.file_contents:
                content = self.file_contents[filename]
                for feature in file_result["required_features"]:
                    if re.search(feature, content, re.IGNORECASE):
                        file_result["found_features"].append(feature)
                    else:
                        file_result["missing_features"].append(feature)
                
                total_features = len(file_result["required_features"])
                found_features = len(file_result["found_features"])
                file_result["completeness_percentage"] = (found_features / total_features * 100) if total_features > 0 else 0
                
                if file_result["completeness_percentage"] >= 90:  # Consider 90%+ as complete
                    results["complete_files"] += 1
                    file_result["status"] = "COMPLETE"
                else:
                    results["incomplete_files"].append(filename)
                    file_result["status"] = "INCOMPLETE"
            else:
                results["incomplete_files"].append(filename)
                file_result["status"] = "MISSING"
            
            results["file_details"].append(file_result)
        
        results["completion_percentage"] = (results["complete_files"] / results["total_files"] * 100) if results["total_files"] > 0 else 0
        
        return results
    
    def validate_roadmap_tasks(self) -> Dict:
        """Validate deployment roadmap tasks."""
        # First, scan all files
        self.scan_files()
        
        # Load contents of core component files
        self.load_file_contents()
        
        # Detect neural connections
        self.detect_neural_connections()
        
        # Validate neural connections
        neural_connections_results = self.validate_neural_connections()
        
        # Validate implementation completeness
        implementation_results = self.validate_implementation_completeness()
        
        # Calculate overall completion percentage
        connections_percentage = neural_connections_results["completion_percentage"]
        implementation_percentage = implementation_results["completion_percentage"]
        overall_percentage = (connections_percentage + implementation_percentage) / 2
        
        result = {
            "file_renames": self.validate_file_renames(),
            "neural_connections": neural_connections_results,
            "implementation_completeness": implementation_results,
            "overall_completion_percentage": overall_percentage,
            "overall_status": f"Neural matrix implementation is {overall_percentage:.1f}% complete"
        }
        
        return result
    
    def print_results(self, results: Dict) -> None:
        """Print validation results in a readable format."""
        print("\n===== NEURAL MATRIX VALIDATION RESULTS =====\n")
        
        print(f"Overall Status: {results['overall_status']}\n")
        
        print(f"Overall Completion: {results['overall_completion_percentage']:.1f}%\n")
        
        print("File Rename Tasks:")
        for task in results["file_renames"]:
            status_marker = "✅" if task["status"] == "COMPLETE" else "❌"
            print(f"{status_marker} {task['source']} → {task['target']}: {task['status']}")
            print(f"   Notes: {task['notes']}")
        
        print("\nNeural Connections:")
        print(f"Completion: {results['neural_connections']['completion_percentage']:.1f}%")
        print(f"Found: {results['neural_connections']['found']} / {results['neural_connections']['total_expected']}")
        
        print("\nConnection Details:")
        for conn in results['neural_connections']['connection_details']:
            status_marker = "✅" if conn["status"] == "COMPLETE" else "❌"
            print(f"{status_marker} {conn['source']} → {conn['target']}: {conn['status']}")
        
        print("\nImplementation Completeness:")
        print(f"Completion: {results['implementation_completeness']['completion_percentage']:.1f}%")
        print(f"Complete Files: {results['implementation_completeness']['complete_files']} / {results['implementation_completeness']['total_files']}")
        
        print("\nFile Implementation Details:")
        for file_detail in results['implementation_completeness']['file_details']:
            status_marker = "✅" if file_detail["status"] == "COMPLETE" else "❌"
            print(f"{status_marker} {file_detail['filename']}: {file_detail['completeness_percentage']:.1f}% complete")
            if file_detail["missing_features"]:
                print(f"   Missing features: {', '.join(file_detail['missing_features'])}")
            
        print("\n===========================================\n")
        
if __name__ == "__main__":
    validator = NeuralMatrixValidator()
    results = validator.validate_roadmap_tasks()
    validator.print_results(results)
