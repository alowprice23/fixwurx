#!/usr/bin/env python3
"""
test_memory_advanced.py
───────────────────────
Advanced integration testing for the enhanced AgentMemory system.

This script demonstrates and validates the enhanced features:
- Compressed storage for efficient token usage
- Family tree storage and retrieval
- Solution path versioning with rollback capability
- Cross-session learning storage
- Integrity verification
"""

import os
import json
import time
import tempfile
import shutil
from pathlib import Path

from agent_memory import AgentMemory, SolutionPathVersion
from compress import Compressor

# Create temporary test directory
TEST_DIR = Path(tempfile.mkdtemp())
MEMORY_PATH = TEST_DIR / "memory.json"
KV_PATH = TEST_DIR / "kv_store.json"
COMPRESSED_PATH = TEST_DIR / "compressed_store.json"
FAMILY_TREE_PATH = TEST_DIR / "family_tree.json"

def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_test_result(name, success):
    """Print test result."""
    result = "✅ PASSED" if success else "❌ FAILED"
    print(f"{result} - {name}")

def print_section(title):
    """Print a section title."""
    print(f"\n--- {title} ---")

def demo_family_tree():
    """Demonstrate family tree storage and retrieval."""
    print_header("Family Tree Storage")
    
    # Create an agent memory instance
    memory = AgentMemory(
        mem_path=MEMORY_PATH,
        kv_path=KV_PATH,
        compressed_path=COMPRESSED_PATH,
        family_tree_path=FAMILY_TREE_PATH
    )
    
    # Create a sample family tree
    family_tree = {
        "root": "planner_agent",
        "created_at": time.time(),
        "version": "1.0.0",
        "children": {
            "observer_agent": {
                "role": "analysis",
                "created_at": time.time(),
                "parent": "planner_agent",
                "capabilities": ["file_system_monitoring", "log_analysis", "bug_reproduction"],
                "children": {}
            },
            "analyst_agent": {
                "role": "solution",
                "created_at": time.time(),
                "parent": "planner_agent",
                "capabilities": ["code_analysis", "patch_generation"],
                "children": {
                    "specific_analyst": {
                        "role": "specialized_solution",
                        "parent": "analyst_agent",
                        "capabilities": ["database_patching"],
                        "children": {}
                    }
                }
            },
            "verifier_agent": {
                "role": "validation",
                "created_at": time.time(),
                "parent": "planner_agent",
                "capabilities": ["test_execution", "patch_validation"],
                "children": {}
            }
        }
    }
    
    # Store the family tree
    memory.store_family_tree(family_tree)
    print(f"Stored family tree with {len(family_tree['children'])} top-level agents")
    
    # Verify it was written to disk
    tree_file_exists = FAMILY_TREE_PATH.exists()
    print_test_result("Family tree persisted to disk", tree_file_exists)
    
    # Create a new memory instance to test loading
    new_memory = AgentMemory(
        mem_path=MEMORY_PATH,
        kv_path=KV_PATH,
        compressed_path=COMPRESSED_PATH,
        family_tree_path=FAMILY_TREE_PATH
    )
    
    # Retrieve the family tree
    retrieved_tree = new_memory.get_family_tree()
    
    # Verify content
    root_match = retrieved_tree["root"] == family_tree["root"]
    child_count_match = len(retrieved_tree["children"]) == len(family_tree["children"])
    
    print_test_result("Retrieved family tree matches original", 
                     root_match and child_count_match)
    
    # Return for use in other tests
    return memory

def demo_compressed_storage(memory):
    """Demonstrate compressed storage capabilities."""
    print_header("Compressed Storage")
    
    # Create a large, repetitive text for good compression
    large_context = "This is a sample of context with repetitive patterns. " * 200
    prompt_template = "Analyze the following code: " * 100
    code_sample = """
    def calculate_factorial(n):
        if n <= 1:
            return 1
        return n * calculate_factorial(n-1)
    
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """ * 20
    
    combined_text = large_context + prompt_template + code_sample
    original_size = len(combined_text)
    
    print(f"Original text size: {original_size} bytes")
    
    # Store with compression
    compression_result = memory.store_compressed("sample_context", combined_text)
    compressed_size = compression_result["compressed_size"]
    ratio = compression_result["compression_ratio"]
    
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {ratio:.2f}")
    print(f"Space saved: {(1-ratio)*100:.2f}%")
    
    # Retrieve compressed data
    retrieved_text = memory.retrieve_compressed("sample_context")
    content_intact = retrieved_text == combined_text
    
    print_test_result("Compressed content retrieved correctly", content_intact)
    
    # Test with more data to demonstrate storage efficiency
    for i in range(5):
        sample = f"Sample text {i}: " + ("This is some repeating content. " * (50 + i * 20))
        memory.store_compressed(f"sample_{i}", sample)
    
    # Get statistics
    stats = memory.get_compression_stats()
    print(f"Total items in compressed storage: {stats['count']}")
    print(f"Average compression ratio: {stats['avg_compression_ratio']:.2f}")
    print(f"Total original size: {stats['total_original_size']} bytes")
    print(f"Total compressed size: {stats['total_compressed_size']} bytes")
    print(f"Overall space saved: {(1 - stats['total_compressed_size']/stats['total_original_size']) * 100:.2f}%")
    
    all_retrieved_correctly = True
    for i in range(5):
        sample = f"Sample text {i}: " + ("This is some repeating content. " * (50 + i * 20))
        retrieved = memory.retrieve_compressed(f"sample_{i}")
        if retrieved != sample:
            all_retrieved_correctly = False
            break
    
    print_test_result("All compressed items retrieved correctly", all_retrieved_correctly)

def demo_solution_path_versioning(memory):
    """Demonstrate solution path versioning capabilities."""
    print_header("Solution Path Versioning")
    
    # Create an initial solution path
    path_id = "solution-path-123"
    initial_solution = {
        "bug_id": "BUG-456",
        "description": "Fix null pointer in UserService.java",
        "actions": [
            {"type": "analyze", "agent": "observer", "description": "Identify root cause"},
            {"type": "patch", "agent": "analyst", "description": "Generate fix"}
        ],
        "dependencies": ["config-service"],
        "priority": 0.8,
        "estimated_complexity": "medium"
    }
    
    # Store the initial solution
    result = memory.store_solution_path(path_id, initial_solution, {
        "creator": "planner_agent",
        "description": "Initial solution proposal",
        "tags": ["null-check", "critical"]
    })
    
    print(f"Created solution path: {path_id}, revision: {result['revision_number']}")
    
    # Create revisions with different approaches
    for i in range(3):
        # Make a copy and modify it
        import copy
        solution_copy = copy.deepcopy(initial_solution)
        
        # Add a new action for this approach
        solution_copy["actions"].append({
            "type": "approach",
            "agent": "analyst",
            "description": f"Alternative approach {i+1}"
        })
        
        # Change some properties
        solution_copy["priority"] = 0.8 - (i * 0.1)  # Decreasing priority
        solution_copy["estimated_complexity"] = ["medium", "high", "low"][i]
        
        # Store this revision
        result = memory.store_solution_path(path_id, solution_copy, {
            "creator": "planner_agent",
            "description": f"Alternative approach {i+1}",
            "tags": ["null-check", f"approach-{i+1}"]
        })
        
        print(f"Added revision {result['revision_number']}: {solution_copy['estimated_complexity']} complexity")
    
    # List all revisions
    print_section("Revision History")
    history = memory.get_revision_history(path_id)
    for rev in history:
        print(f"  Rev {rev['revision_number']}: {rev['description']} (created: {time.ctime(rev['timestamp'])})")
    
    # Test specific revision retrieval
    print_section("Retrieving Specific Revisions")
    
    # Get the initial version (revision 0)
    initial = memory.get_solution_path(path_id, 0)
    action_count_match = len(initial["actions"]) == 2
    print_test_result("Initial revision has correct action count", action_count_match)
    
    # Get the current (latest) version
    current = memory.get_solution_path(path_id)
    # At this point, the current version should be revision 3 (the latest) with 3 actions
    # (2 initial + 1 from the last approach since each revision is independent)
    latest_rev_match = len(current["actions"]) == 3
    print_test_result("Latest revision has correct action count", latest_rev_match)
    
    # Test rollback functionality
    print_section("Rollback Testing")
    
    # Rollback to revision 1
    rollback_result = memory.rollback_solution_path(path_id, 1)
    rollback_actions_match = len(rollback_result["actions"]) == 3  # 2 initial + 1 alternative
    print_test_result("Rollback to revision 1 successful", rollback_actions_match)
    
    # Verify current revision is now revision 1
    current_after_rollback = memory.get_solution_path(path_id)
    current_is_rev1 = len(current_after_rollback["actions"]) == 3
    print_test_result("Current revision is now revision 1", current_is_rev1)
    
    # Check that deeper rollback works
    rollback_result = memory.rollback_solution_path(path_id, 0)
    original_actions_match = len(rollback_result["actions"]) == 2
    print_test_result("Rollback to original version successful", original_actions_match)

def demo_cross_session_learning(memory):
    """Demonstrate cross-session learning storage."""
    print_header("Cross-Session Learning Storage")
    
    # Create learning data for different models
    models = ["gpt-4", "claude-3", "local-llm"]
    
    for i, model in enumerate(models):
        # Create model-specific learning data
        learning_data = {
            "model_version": f"{model}-v1",
            "training_iterations": 100 + (i * 50),
            "successful_patterns": [
                {"pattern": "null check before method call", "success_rate": 0.95 - (i * 0.05)},
                {"pattern": "boundary condition validation", "success_rate": 0.87 - (i * 0.04)},
                {"pattern": "type conversion safety", "success_rate": 0.78 - (i * 0.03)}
            ],
            "failure_patterns": [
                {"pattern": "tight coupling dependencies", "failure_rate": 0.65 + (i * 0.05)},
                {"pattern": "ignoring return values", "failure_rate": 0.58 + (i * 0.04)}
            ],
            "memory_examples": [
                {
                    "context": f"Example context for {model} #{j}",
                    "solution": f"Example solution for {model} #{j}",
                    "effectiveness": 0.9 - (j * 0.1)
                }
                for j in range(3)
            ],
            "last_updated": time.time()
        }
        
        # Store the learning data
        memory.store_learning_data(model, learning_data)
        print(f"Stored learning data for {model} with {len(learning_data['successful_patterns'])} success patterns")
    
    # List all models with learning data
    stored_models = memory.list_learning_models()
    models_match = set(stored_models) == set(models)
    print_test_result("All models stored correctly", models_match)
    
    # Retrieve learning data for a specific model
    gpt4_data = memory.get_learning_data("gpt-4")
    patterns_match = len(gpt4_data["successful_patterns"]) == 3
    print_test_result("Retrieved correct learning data", 
                      patterns_match and gpt4_data["model_version"] == "gpt-4-v1")
    
    # Persistence test
    print_section("Learning Data Persistence")
    
    # Create a new memory instance
    new_memory = AgentMemory(
        mem_path=MEMORY_PATH,
        kv_path=KV_PATH,
        compressed_path=COMPRESSED_PATH,
        family_tree_path=FAMILY_TREE_PATH
    )
    
    # Check if learning data was persisted
    retrieved_models = new_memory.list_learning_models()
    models_persist = set(retrieved_models) == set(models)
    print_test_result("Learning data persisted correctly", models_persist)
    
    # Retrieve and verify a specific model's data
    claude_data = new_memory.get_learning_data("claude-3")
    data_integrity = (
        claude_data["model_version"] == "claude-3-v1" and
        len(claude_data["successful_patterns"]) == 3 and
        len(claude_data["failure_patterns"]) == 2
    )
    print_test_result("Learning data integrity maintained", data_integrity)

def demo_integration(memory):
    """Demonstrate integration of all features."""
    print_header("Feature Integration")
    
    # Create a solution path with embedded compressed content
    path_id = "integrated-solution-1"
    
    # Generate some large context
    large_context = "This is a large context block that will be compressed. " * 100
    
    # Store it in compressed storage
    context_key = "context:integrated-1"
    memory.store_compressed(context_key, large_context)
    
    # Create a solution that references the compressed content
    solution = {
        "bug_id": "BUG-789",
        "title": "Integrated solution demonstration",
        "context_ref": context_key,
        "actions": [
            {"type": "analyze", "agent": "observer", "description": "Complex analysis"}
        ],
        "family_agents": [
            "planner_agent",
            "observer_agent",
            "analyst_agent"
        ],
        "learning_models": [
            "gpt-4",
            "claude-3"
        ],
        "metrics": {
            "estimated_success_rate": 0.92,
            "complexity": "high",
            "priority": 0.85
        }
    }
    
    # Store this integrated solution
    memory.store_solution_path(path_id, solution, {
        "creator": "integrated_test",
        "description": "Solution with integrated features",
        "tags": ["integration", "test"]
    })
    
    # Retrieve the solution
    retrieved = memory.get_solution_path(path_id)
    
    # Verify references
    context_ref = retrieved["context_ref"]
    context_content = memory.retrieve_compressed(context_ref)
    context_match = context_content == large_context
    print_test_result("Compressed content reference works", context_match)
    
    # Get learning data for referenced models
    learning_references_valid = True
    for model in retrieved["learning_models"]:
        model_data = memory.get_learning_data(model)
        if not model_data or "successful_patterns" not in model_data:
            learning_references_valid = False
            break
    
    print_test_result("Learning data references valid", learning_references_valid)
    
    # Verify family agent references
    tree = memory.get_family_tree()
    agents_in_tree = True
    for agent in retrieved["family_agents"]:
        if agent == "planner_agent":
            if tree["root"] != agent:
                agents_in_tree = False
                break
        elif agent not in tree["children"]:
            agents_in_tree = False
            break
    
    print_test_result("Family agent references valid", agents_in_tree)
    
    # Comprehensive integration
    print(f"Integrated solution successfully links:")
    print(f"  - {len(context_content)} bytes of compressed content")
    print(f"  - {len(retrieved['family_agents'])} family tree agents")
    print(f"  - {len(retrieved['learning_models'])} learning models")
    print(f"  - {len(retrieved['actions'])} solution actions")

def cleanup():
    """Clean up test files."""
    print_header("Cleaning Up")
    
    try:
        shutil.rmtree(TEST_DIR)
        print(f"Removed test directory: {TEST_DIR}")
    except Exception as e:
        print(f"Error removing test directory: {e}")

def main():
    """Run the demonstration."""
    print_header("ENHANCED AGENT MEMORY DEMONSTRATION")
    
    try:
        # Demo family tree storage
        memory = demo_family_tree()
        
        # Demo compressed storage
        demo_compressed_storage(memory)
        
        # Demo solution path versioning
        demo_solution_path_versioning(memory)
        
        # Demo cross-session learning storage
        demo_cross_session_learning(memory)
        
        # Demo integration of all features
        demo_integration(memory)
    
    finally:
        # Clean up
        cleanup()
    
    print_header("DEMONSTRATION COMPLETE")

if __name__ == "__main__":
    main()
