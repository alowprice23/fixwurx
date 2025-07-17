#!/usr/bin/env python3
"""
Collaborative Improvement Commands

Command handlers for the Collaborative Improvement Framework.
This module registers commands for pattern detection, script proposals,
peer review, and decision tree growth.
"""

import os
import sys
import json
import logging
import importlib
import tabulate
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger("CollaborativeImprovementCommands")

def register_collaborative_commands(registry):
    """
    Register collaborative improvement commands with the registry.
    
    Args:
        registry: Component registry
    """
    # Register commands
    registry.register_command_handler("show_patterns", show_patterns_command, "collaborative")
    registry.register_command_handler("show_proposals", show_proposals_command, "collaborative")
    registry.register_command_handler("review_proposal", review_proposal_command, "collaborative")
    registry.register_command_handler("commit_script", commit_script_command, "collaborative")
    registry.register_command_handler("add_decision_path", add_decision_path_command, "collaborative")
    registry.register_command_handler("get_recommendation", get_recommendation_command, "collaborative")
    registry.register_command_handler("prune_tree", prune_tree_command, "collaborative")
    
    # Register aliases
    registry.register_alias("patterns", "show_patterns")
    registry.register_alias("proposals", "show_proposals")
    registry.register_alias("review", "review_proposal")
    registry.register_alias("commit", "commit_script")
    
    logger.info("Registered collaborative improvement commands")

def show_patterns_command(args: str) -> int:
    """
    Show detected patterns.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Show detected patterns")
    parser.add_argument("--all", action="store_true", help="Show all patterns (including those already proposed)")
    parser.add_argument("--id", help="Show details for a specific pattern ID")
    parser.add_argument("--user", help="Filter patterns by user ID")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get pattern detector
    pattern_detector = sys.modules["__main__"].registry.get_component("pattern_detector")
    if not pattern_detector:
        print("Error: Pattern Detector not available")
        return 1
    
    pattern_id = cmd_args.id
    show_all = cmd_args.all
    user_filter = cmd_args.user
    
    # Show details for a specific pattern
    if pattern_id:
        pattern = pattern_detector.get_pattern(pattern_id)
        if not pattern:
            print(f"Error: Pattern {pattern_id} not found")
            return 1
        
        print(f"\nPattern: {pattern_id}")
        print("-" * 60)
        print(f"Commands:")
        for i, command in enumerate(pattern.get("commands", []), 1):
            print(f"  {i}. {command}")
        
        print(f"\nOccurrences: {pattern.get('occurrences', 0)}")
        print(f"Created: {pattern.get('created', 'Unknown')}")
        print(f"User ID: {pattern.get('user_id', 'Unknown')}")
        print(f"Proposed: {'Yes' if pattern.get('proposed', False) else 'No'}")
        print(f"Approved: {'Yes' if pattern.get('approved', False) else 'No'}")
        print(f"Committed: {'Yes' if pattern.get('committed', False) else 'No'}")
        
        if pattern.get("script_id"):
            print(f"Script ID: {pattern.get('script_id')}")
        
        return 0
    
    # Show all patterns
    patterns = pattern_detector.get_patterns()
    
    if not patterns:
        print("No patterns detected yet")
        return 0
    
    # Filter patterns
    filtered_patterns = []
    for pid, p in patterns.items():
        # Skip patterns that have been proposed unless showing all
        if not show_all and p.get("proposed", False):
            continue
        
        # Filter by user ID if specified
        if user_filter and p.get("user_id") != user_filter:
            continue
        
        filtered_patterns.append((pid, p))
    
    if not filtered_patterns:
        print("No matching patterns found")
        return 0
    
    # Sort patterns by created timestamp (newest first)
    filtered_patterns.sort(key=lambda x: x[1].get("created", 0), reverse=True)
    
    # Create table data
    table_data = []
    for pid, p in filtered_patterns:
        commands = " -> ".join(p.get("commands", [])[:2])
        if len(p.get("commands", [])) > 2:
            commands += " -> ..."
        
        status = []
        if p.get("proposed", False):
            status.append("Proposed")
        if p.get("approved", False):
            status.append("Approved")
        if p.get("committed", False):
            status.append("Committed")
        
        if not status:
            status = ["New"]
        
        table_data.append([
            pid,
            commands,
            p.get("occurrences", 0),
            p.get("user_id", "Unknown"),
            ", ".join(status)
        ])
    
    # Print table
    print("\nDetected Patterns:")
    print(tabulate.tabulate(
        table_data,
        headers=["ID", "Commands", "Occurrences", "User", "Status"],
        tablefmt="grid"
    ))
    
    print(f"\nTotal: {len(filtered_patterns)} pattern(s)")
    print("Use 'show_patterns --id <pattern_id>' to see details for a specific pattern")
    
    return 0

def show_proposals_command(args: str) -> int:
    """
    Show script proposals.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Show script proposals")
    parser.add_argument("--all", action="store_true", help="Show all proposals (including those already committed)")
    parser.add_argument("--id", help="Show details for a specific proposal ID")
    parser.add_argument("--pattern", help="Filter proposals by pattern ID")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get script proposer
    script_proposer = sys.modules["__main__"].registry.get_component("script_proposer")
    if not script_proposer:
        print("Error: Script Proposer not available")
        return 1
    
    proposal_id = cmd_args.id
    show_all = cmd_args.all
    pattern_filter = cmd_args.pattern
    
    # Show details for a specific proposal
    if proposal_id:
        proposal = script_proposer.get_proposal(proposal_id)
        if not proposal:
            print(f"Error: Proposal {proposal_id} not found")
            return 1
        
        print(f"\nProposal: {proposal_id}")
        print("-" * 60)
        print(f"Pattern ID: {proposal.get('pattern_id', 'Unknown')}")
        print(f"Created: {proposal.get('created', 'Unknown')}")
        print(f"Approved: {'Yes' if proposal.get('approved', False) else 'No'}")
        print(f"Committed: {'Yes' if proposal.get('committed', False) else 'No'}")
        
        if proposal.get("script_id"):
            print(f"Script ID: {proposal.get('script_id')}")
        
        print("\nScript:")
        print("-" * 60)
        print(proposal.get("script", "No script available"))
        
        if proposal.get("reviews"):
            print("\nReviews:")
            print("-" * 60)
            for i, review in enumerate(proposal.get("reviews", []), 1):
                print(f"Review {i}:")
                print(f"  Reviewer: {review.get('reviewer_id', 'Unknown')}")
                print(f"  Approved: {'Yes' if review.get('approved', False) else 'No'}")
                print(f"  Comments: {review.get('comments', 'No comments')}")
                print(f"  Timestamp: {review.get('timestamp', 'Unknown')}")
                print()
        
        return 0
    
    # Show all proposals
    proposals = script_proposer.get_proposals()
    
    if not proposals:
        print("No proposals created yet")
        return 0
    
    # Filter proposals
    filtered_proposals = []
    for pid, p in proposals.items():
        # Skip proposals that have been committed unless showing all
        if not show_all and p.get("committed", False):
            continue
        
        # Filter by pattern ID if specified
        if pattern_filter and p.get("pattern_id") != pattern_filter:
            continue
        
        filtered_proposals.append((pid, p))
    
    if not filtered_proposals:
        print("No matching proposals found")
        return 0
    
    # Sort proposals by created timestamp (newest first)
    filtered_proposals.sort(key=lambda x: x[1].get("created", 0), reverse=True)
    
    # Create table data
    table_data = []
    for pid, p in filtered_proposals:
        status = []
        if p.get("approved", False):
            status.append("Approved")
        if p.get("committed", False):
            status.append("Committed")
        
        if not status:
            status = ["Pending"]
        
        num_reviews = len(p.get("reviews", []))
        approvals = sum(1 for r in p.get("reviews", []) if r.get("approved", False))
        
        table_data.append([
            pid,
            p.get("pattern_id", "Unknown"),
            f"{approvals}/{num_reviews}",
            ", ".join(status)
        ])
    
    # Print table
    print("\nScript Proposals:")
    print(tabulate.tabulate(
        table_data,
        headers=["ID", "Pattern ID", "Reviews", "Status"],
        tablefmt="grid"
    ))
    
    print(f"\nTotal: {len(filtered_proposals)} proposal(s)")
    print("Use 'show_proposals --id <proposal_id>' to see details for a specific proposal")
    
    return 0

def review_proposal_command(args: str) -> int:
    """
    Review a script proposal.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Review a script proposal")
    parser.add_argument("--id", required=True, help="Proposal ID to review")
    parser.add_argument("--reviewer", help="Reviewer ID")
    parser.add_argument("--approve", action="store_true", help="Approve the proposal")
    parser.add_argument("--reject", action="store_true", help="Reject the proposal")
    parser.add_argument("--comment", help="Review comments")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get peer review workflow
    peer_review = sys.modules["__main__"].registry.get_component("peer_review_workflow")
    if not peer_review:
        print("Error: Peer Review Workflow not available")
        return 1
    
    proposal_id = cmd_args.id
    reviewer_id = cmd_args.reviewer or f"user_{os.getuid() if hasattr(os, 'getuid') else os.getpid()}"
    approved = cmd_args.approve
    rejected = cmd_args.reject
    comments = cmd_args.comment or ""
    
    # Check for conflicting options
    if approved and rejected:
        print("Error: Cannot both approve and reject a proposal")
        return 1
    
    if not approved and not rejected:
        print("Error: Must either approve or reject the proposal")
        return 1
    
    # Check if proposal exists
    script_proposer = sys.modules["__main__"].registry.get_component("script_proposer")
    if not script_proposer:
        print("Error: Script Proposer not available")
        return 1
    
    proposal = script_proposer.get_proposal(proposal_id)
    if not proposal:
        print(f"Error: Proposal {proposal_id} not found")
        return 1
    
    # Handle special case for "latest"
    if proposal_id.lower() == "latest":
        proposals = script_proposer.get_proposals()
        if not proposals:
            print("Error: No proposals available")
            return 1
        
        # Get the latest proposal
        latest_proposal = max(proposals.items(), key=lambda x: x[1].get("created", 0))
        proposal_id = latest_proposal[0]
        proposal = latest_proposal[1]
    
    # Submit review
    result = peer_review.submit_review(
        proposal_id,
        reviewer_id,
        approved,
        comments
    )
    
    if not result.get("success", False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    print(f"Review submitted for proposal {proposal_id}")
    
    if result.get("approved", False):
        print("The proposal has been approved and is ready to be committed")
    else:
        print("The proposal is still pending approval")
    
    return 0

def commit_script_command(args: str) -> int:
    """
    Commit an approved script proposal to the script library.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Commit an approved script proposal")
    parser.add_argument("--id", required=True, help="Proposal ID to commit")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get peer review workflow
    peer_review = sys.modules["__main__"].registry.get_component("peer_review_workflow")
    if not peer_review:
        print("Error: Peer Review Workflow not available")
        return 1
    
    proposal_id = cmd_args.id
    
    # Handle special case for "latest"
    if proposal_id.lower() == "latest":
        script_proposer = sys.modules["__main__"].registry.get_component("script_proposer")
        if not script_proposer:
            print("Error: Script Proposer not available")
            return 1
        
        proposals = script_proposer.get_proposals()
        if not proposals:
            print("Error: No proposals available")
            return 1
        
        # Get the latest proposal
        latest_proposal = max(proposals.items(), key=lambda x: x[1].get("created", 0))
        proposal_id = latest_proposal[0]
    
    # Commit script
    result = peer_review.commit_script(proposal_id)
    
    if not result.get("success", False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    print(f"Script committed for proposal {proposal_id}")
    print(f"Script ID: {result.get('script_id')}")
    print(f"Script Name: {result.get('script_name')}")
    
    return 0

def add_decision_path_command(args: str) -> int:
    """
    Add a decision path to the decision tree.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Add a decision path to the decision tree")
    parser.add_argument("--commands", required=True, help="Comma-separated list of commands")
    parser.add_argument("--outcome", required=True, help="Outcome of the commands")
    parser.add_argument("--success", action="store_true", help="Whether the outcome was successful")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get decision tree growth
    decision_tree = sys.modules["__main__"].registry.get_component("decision_tree_growth")
    if not decision_tree:
        print("Error: Decision Tree Growth not available")
        return 1
    
    commands = [cmd.strip() for cmd in cmd_args.commands.split(",")]
    outcome = cmd_args.outcome
    success = cmd_args.success
    
    # Add decision path
    result = decision_tree.add_decision_path(commands, outcome, success)
    
    if not result.get("success", False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    print(f"Decision path added: {result.get('path')}")
    
    return 0

def get_recommendation_command(args: str) -> int:
    """
    Get a recommendation from the decision tree.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Get a recommendation from the decision tree")
    parser.add_argument("--commands", required=True, help="Comma-separated list of commands")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get decision tree growth
    decision_tree = sys.modules["__main__"].registry.get_component("decision_tree_growth")
    if not decision_tree:
        print("Error: Decision Tree Growth not available")
        return 1
    
    commands = [cmd.strip() for cmd in cmd_args.commands.split(",")]
    
    # Get recommendation
    result = decision_tree.get_decision_recommendation(commands)
    
    if not result.get("success", False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    print("\nRecommendation:")
    print("-" * 60)
    print(f"Match: {'Exact' if result.get('exact_match', False) else 'Partial'}")
    print(f"Path: {result.get('path', 'Unknown')}")
    print(f"Recommendation: {result.get('recommendation', 'No recommendation')}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    print(f"Success Rate: {result.get('success_rate', 0):.2f}")
    print(f"Count: {result.get('count', 0)}")
    
    return 0

def prune_tree_command(args: str) -> int:
    """
    Prune the decision tree.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Prune the decision tree")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    # Get decision tree growth
    decision_tree = sys.modules["__main__"].registry.get_component("decision_tree_growth")
    if not decision_tree:
        print("Error: Decision Tree Growth not available")
        return 1
    
    # Prune tree
    result = decision_tree.prune_decision_tree()
    
    if not result.get("success", False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    print("\nDecision Tree Pruned:")
    print("-" * 60)
    print(f"Before: {result.get('before_count', 0)} nodes")
    print(f"After: {result.get('after_count', 0)} nodes")
    print(f"Pruned: {result.get('pruned_count', 0)} nodes")
    
    return 0
