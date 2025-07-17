#!/usr/bin/env python3
"""
Collaborative Improvement Framework

This module provides pattern detection, script proposal, peer review workflow,
and decision tree growth for collaborative improvement of shell interactions.
"""

import os
import sys
import json
import time
import logging
import re
import uuid
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set

logger = logging.getLogger("CollaborativeImprovement")

class PatternDetector:
    """
    Pattern Detector for identifying command patterns in user interactions.
    """
    
    def __init__(self, registry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Pattern Detector.
        
        Args:
            registry: Component registry
            config: Optional configuration dictionary
        """
        self.registry = registry
        self.config = config or {}
        self.initialized = False
        
        # Configuration parameters
        self.min_pattern_length = self.config.get("min_pattern_length", 2)
        self.min_occurrences = self.config.get("min_occurrences", 2)
        self.max_patterns = self.config.get("max_patterns", 100)
        
        # Command history by user
        self.command_history = {}
        
        # Detected patterns
        self.patterns = {}
        
        # Register with registry
        registry.register_component("pattern_detector", self)
        
        logger.info("Pattern Detector initialized with default settings")
    
    def initialize(self) -> bool:
        """
        Initialize the Pattern Detector.
        
        Returns:
            True if initialization was successful
        """
        if self.initialized:
            logger.warning("Pattern Detector already initialized")
            return True
        
        try:
            # Load patterns from file if available
            self._load_patterns()
            
            self.initialized = True
            logger.info("Pattern Detector initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing Pattern Detector: {e}")
            return False
    
    def _load_patterns(self) -> None:
        """Load patterns from file."""
        patterns_path = os.path.join("collaborative_improvement", "patterns.json")
        if os.path.exists(patterns_path):
            try:
                with open(patterns_path, "r") as f:
                    self.patterns = json.load(f)
                logger.info(f"Loaded {len(self.patterns)} patterns")
            except Exception as e:
                logger.error(f"Error loading patterns: {e}")
                self.patterns = {}
    
    def _save_patterns(self) -> None:
        """Save patterns to file."""
        patterns_path = os.path.join("collaborative_improvement", "patterns.json")
        os.makedirs(os.path.dirname(patterns_path), exist_ok=True)
        try:
            with open(patterns_path, "w") as f:
                json.dump(self.patterns, f, indent=2)
            logger.info(f"Saved {len(self.patterns)} patterns")
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def add_command(self, command: str, user_id: str) -> None:
        """
        Add a command to the history.
        
        Args:
            command: Command string
            user_id: User ID
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Pattern Detector initialization failed")
                return
        
        try:
            # Add user if not exists
            if user_id not in self.command_history:
                self.command_history[user_id] = []
            
            # Add command to history
            self.command_history[user_id].append({
                "command": command,
                "timestamp": time.time()
            })
            
            # Detect patterns
            self._detect_patterns(user_id)
        except Exception as e:
            logger.error(f"Error adding command: {e}")
    
    def _detect_patterns(self, user_id: str) -> None:
        """
        Detect patterns in command history.
        
        Args:
            user_id: User ID
        """
        try:
            # Get command history
            history = self.command_history.get(user_id, [])
            
            # Need at least min_pattern_length commands
            if len(history) < self.min_pattern_length:
                return
            
            # Extract commands
            commands = [entry["command"] for entry in history]
            
            # Look for repeating patterns
            for pattern_length in range(self.min_pattern_length, min(len(commands) // 2 + 1, 5)):
                for i in range(len(commands) - pattern_length * 2 + 1):
                    pattern = tuple(commands[i:i+pattern_length])
                    
                    # Count occurrences
                    occurrences = 0
                    for j in range(len(commands) - pattern_length + 1):
                        if tuple(commands[j:j+pattern_length]) == pattern:
                            occurrences += 1
                    
                    # Add pattern if it occurs at least min_occurrences times
                    if occurrences >= self.min_occurrences:
                        pattern_id = f"pattern_{uuid.uuid4().hex[:8]}"
                        
                        # Check if pattern already exists
                        pattern_exists = False
                        for pid, p in self.patterns.items():
                            if p.get("commands") == list(pattern):
                                pattern_exists = True
                                break
                        
                        if not pattern_exists:
                            self.patterns[pattern_id] = {
                                "id": pattern_id,
                                "commands": list(pattern),
                                "occurrences": occurrences,
                                "created": time.time(),
                                "user_id": user_id,
                                "proposed": False,
                                "approved": False,
                                "committed": False
                            }
                            
                            logger.info(f"Detected pattern {pattern_id}: {pattern}")
                            
                            # Save patterns
                            self._save_patterns()
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
    
    def get_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detected patterns.
        
        Returns:
            Dictionary of patterns
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Pattern Detector initialization failed")
                return {}
        
        return self.patterns
    
    def get_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get a pattern by ID.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Pattern dictionary
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Pattern Detector initialization failed")
                return {}
        
        return self.patterns.get(pattern_id, {})
    
    def shutdown(self) -> None:
        """
        Shutdown the Pattern Detector.
        """
        if not self.initialized:
            return
        
        # Save patterns
        self._save_patterns()
        
        self.initialized = False
        logger.info("Pattern Detector shutdown complete")

class ScriptProposer:
    """
    Script Proposer for proposing scripts based on detected patterns.
    """
    
    def __init__(self, registry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Script Proposer.
        
        Args:
            registry: Component registry
            config: Optional configuration dictionary
        """
        self.registry = registry
        self.config = config or {}
        self.initialized = False
        
        # Configuration parameters
        self.max_proposals = self.config.get("max_proposals", 100)
        
        # Proposals
        self.proposals = {}
        
        # Register with registry
        registry.register_component("script_proposer", self)
        
        logger.info("Script Proposer initialized with default settings")
    
    def initialize(self) -> bool:
        """
        Initialize the Script Proposer.
        
        Returns:
            True if initialization was successful
        """
        if self.initialized:
            logger.warning("Script Proposer already initialized")
            return True
        
        try:
            # Load proposals from file if available
            self._load_proposals()
            
            self.initialized = True
            logger.info("Script Proposer initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing Script Proposer: {e}")
            return False
    
    def _load_proposals(self) -> None:
        """Load proposals from file."""
        proposals_path = os.path.join("collaborative_improvement", "proposals.json")
        if os.path.exists(proposals_path):
            try:
                with open(proposals_path, "r") as f:
                    self.proposals = json.load(f)
                logger.info(f"Loaded {len(self.proposals)} proposals")
            except Exception as e:
                logger.error(f"Error loading proposals: {e}")
                self.proposals = {}
    
    def _save_proposals(self) -> None:
        """Save proposals to file."""
        proposals_path = os.path.join("collaborative_improvement", "proposals.json")
        os.makedirs(os.path.dirname(proposals_path), exist_ok=True)
        try:
            with open(proposals_path, "w") as f:
                json.dump(self.proposals, f, indent=2)
            logger.info(f"Saved {len(self.proposals)} proposals")
        except Exception as e:
            logger.error(f"Error saving proposals: {e}")
    
    def propose_script(self, pattern_id: str) -> Dict[str, Any]:
        """
        Propose a script for a pattern.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Dictionary with proposal information
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Script Proposer initialization failed")
                return {"success": False, "error": "Script Proposer initialization failed"}
        
        try:
            # Get pattern detector
            pattern_detector = self.registry.get_component("pattern_detector")
            if not pattern_detector:
                return {"success": False, "error": "Pattern Detector not available"}
            
            # Get pattern
            pattern = pattern_detector.get_pattern(pattern_id)
            if not pattern:
                return {"success": False, "error": f"Pattern {pattern_id} not found"}
            
            # Generate script from pattern
            script = self._generate_script(pattern)
            
            # Create proposal
            proposal_id = f"proposal_{uuid.uuid4().hex[:8]}"
            proposal = {
                "id": proposal_id,
                "pattern_id": pattern_id,
                "script": script,
                "created": time.time(),
                "reviews": [],
                "approved": False,
                "committed": False
            }
            
            # Add to proposals
            self.proposals[proposal_id] = proposal
            
            # Save proposals
            self._save_proposals()
            
            # Update pattern
            pattern["proposed"] = True
            pattern_detector._save_patterns()
            
            logger.info(f"Proposed script {proposal_id} for pattern {pattern_id}")
            
            return {
                "success": True,
                "proposal_id": proposal_id,
                "pattern_id": pattern_id,
                "script": script
            }
        except Exception as e:
            logger.error(f"Error proposing script: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_script(self, pattern: Dict[str, Any]) -> str:
        """
        Generate a script from a pattern.
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            Script content
        """
        try:
            # Get commands
            commands = pattern.get("commands", [])
            
            # Generate script name
            script_name = self._generate_script_name(commands)
            
            # Generate script content
            script_lines = [
                "#!/usr/bin/env bash",
                f"# {script_name}",
                "#",
                "# This script was automatically generated from a detected command pattern.",
                "#",
                "",
                "# Error handling",
                "set -e",
                "",
                "# Define variables",
                "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
                "",
                "# Command sequence",
            ]
            
            # Add commands
            for i, command in enumerate(commands):
                script_lines.append(f"echo \"Step {i+1}: {command}\"")
                script_lines.append(f"{command}")
                script_lines.append("")
            
            script_lines.append("echo \"Script completed successfully\"")
            
            return "\n".join(script_lines)
        except Exception as e:
            logger.error(f"Error generating script: {e}")
            return "#!/usr/bin/env bash\necho \"Error generating script\""
    
    def _generate_script_name(self, commands: List[str]) -> str:
        """
        Generate a script name from commands.
        
        Args:
            commands: List of commands
            
        Returns:
            Script name
        """
        try:
            # Try to extract meaningful words from commands
            words = []
            for command in commands:
                words.extend(re.findall(r'\b[a-zA-Z]{3,}\b', command))
            
            # Remove duplicates and limit to 3 words
            unique_words = []
            for word in words:
                if word not in unique_words and word not in ["echo", "printf", "cat", "the", "and", "for", "while", "done"]:
                    unique_words.append(word)
                if len(unique_words) >= 3:
                    break
            
            if unique_words:
                return "Automated " + " ".join(w.capitalize() for w in unique_words) + " Script"
            else:
                return "Automated Command Sequence Script"
        except Exception as e:
            logger.error(f"Error generating script name: {e}")
            return "Automated Command Sequence Script"
    
    def get_proposal(self, proposal_id: str) -> Dict[str, Any]:
        """
        Get a proposal by ID.
        
        Args:
            proposal_id: Proposal ID
            
        Returns:
            Proposal dictionary
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Script Proposer initialization failed")
                return {}
        
        return self.proposals.get(proposal_id, {})
    
    def get_proposals(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all proposals.
        
        Returns:
            Dictionary of proposals
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Script Proposer initialization failed")
                return {}
        
        return self.proposals
    
    def shutdown(self) -> None:
        """
        Shutdown the Script Proposer.
        """
        if not self.initialized:
            return
        
        # Save proposals
        self._save_proposals()
        
        self.initialized = False
        logger.info("Script Proposer shutdown complete")

class PeerReviewWorkflow:
    """
    Peer Review Workflow for reviewing and approving script proposals.
    """
    
    def __init__(self, registry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Peer Review Workflow.
        
        Args:
            registry: Component registry
            config: Optional configuration dictionary
        """
        self.registry = registry
        self.config = config or {}
        self.initialized = False
        
        # Configuration parameters
        self.min_reviews = self.config.get("min_reviews", 2)
        self.approval_threshold = self.config.get("approval_threshold", 0.75)
        
        # Register with registry
        registry.register_component("peer_review_workflow", self)
        
        logger.info("Peer Review Workflow initialized with default settings")
    
    def initialize(self) -> bool:
        """
        Initialize the Peer Review Workflow.
        
        Returns:
            True if initialization was successful
        """
        if self.initialized:
            logger.warning("Peer Review Workflow already initialized")
            return True
        
        try:
            self.initialized = True
            logger.info("Peer Review Workflow initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing Peer Review Workflow: {e}")
            return False
    
    def submit_review(self, proposal_id: str, reviewer_id: str, approved: bool, comments: str) -> Dict[str, Any]:
        """
        Submit a review for a script proposal.
        
        Args:
            proposal_id: Proposal ID
            reviewer_id: Reviewer ID
            approved: Whether the proposal is approved
            comments: Review comments
            
        Returns:
            Dictionary with review information
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Peer Review Workflow initialization failed")
                return {"success": False, "error": "Peer Review Workflow initialization failed"}
        
        try:
            # Get script proposer
            script_proposer = self.registry.get_component("script_proposer")
            if not script_proposer:
                return {"success": False, "error": "Script Proposer not available"}
            
            # Get proposal
            proposal = script_proposer.get_proposal(proposal_id)
            if not proposal:
                return {"success": False, "error": f"Proposal {proposal_id} not found"}
            
            # Check if reviewer already submitted a review
            for review in proposal.get("reviews", []):
                if review.get("reviewer_id") == reviewer_id:
                    return {"success": False, "error": f"Reviewer {reviewer_id} already submitted a review"}
            
            # Add review
            review = {
                "id": f"review_{uuid.uuid4().hex[:8]}",
                "reviewer_id": reviewer_id,
                "approved": approved,
                "comments": comments,
                "timestamp": time.time()
            }
            
            if "reviews" not in proposal:
                proposal["reviews"] = []
            
            proposal["reviews"].append(review)
            
            # Check if proposal is approved
            self._check_approval_status(proposal)
            
            # Save proposals
            script_proposer._save_proposals()
            
            logger.info(f"Submitted review for proposal {proposal_id} by reviewer {reviewer_id}")
            
            return {
                "success": True,
                "proposal_id": proposal_id,
                "review_id": review["id"],
                "approved": proposal.get("approved", False)
            }
        except Exception as e:
            logger.error(f"Error submitting review: {e}")
            return {"success": False, "error": str(e)}
    
    def _check_approval_status(self, proposal: Dict[str, Any]) -> None:
        """
        Check if a proposal is approved.
        
        Args:
            proposal: Proposal dictionary
        """
        try:
            # Get reviews
            reviews = proposal.get("reviews", [])
            
            # Check if enough reviews
            if len(reviews) < self.min_reviews:
                proposal["approved"] = False
                return
            
            # Count approvals
            approvals = sum(1 for review in reviews if review.get("approved", False))
            
            # Calculate approval ratio
            approval_ratio = approvals / len(reviews)
            
            # Set approved flag
            proposal["approved"] = approval_ratio >= self.approval_threshold
            
            if proposal["approved"]:
                logger.info(f"Proposal {proposal.get('id')} approved")
                
                # Update pattern
                pattern_detector = self.registry.get_component("pattern_detector")
                if pattern_detector:
                    pattern = pattern_detector.get_pattern(proposal.get("pattern_id"))
                    if pattern:
                        pattern["approved"] = True
                        pattern_detector._save_patterns()
        except Exception as e:
            logger.error(f"Error checking approval status: {e}")
    
    def commit_script(self, proposal_id: str) -> Dict[str, Any]:
        """
        Commit an approved script to the script library.
        
        Args:
            proposal_id: Proposal ID
            
        Returns:
            Dictionary with commit information
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Peer Review Workflow initialization failed")
                return {"success": False, "error": "Peer Review Workflow initialization failed"}
        
        try:
            # Get script proposer
            script_proposer = self.registry.get_component("script_proposer")
            if not script_proposer:
                return {"success": False, "error": "Script Proposer not available"}
            
            # Get proposal
            proposal = script_proposer.get_proposal(proposal_id)
            if not proposal:
                return {"success": False, "error": f"Proposal {proposal_id} not found"}
            
            # Check if proposal is approved
            if not proposal.get("approved", False):
                return {"success": False, "error": f"Proposal {proposal_id} is not approved"}
            
            # Check if proposal is already committed
            if proposal.get("committed", False):
                return {"success": False, "error": f"Proposal {proposal_id} is already committed"}
            
            # Get pattern detector
            pattern_detector = self.registry.get_component("pattern_detector")
            if not pattern_detector:
                return {"success": False, "error": "Pattern Detector not available"}
            
            # Get pattern
            pattern = pattern_detector.get_pattern(proposal.get("pattern_id"))
            if not pattern:
                return {"success": False, "error": f"Pattern {proposal.get('pattern_id')} not found"}
            
            # Get script library
            script_library = self.registry.get_component("script_library")
            if not script_library:
                return {"success": False, "error": "Script Library not available"}
            
            # Generate script name
            commands = pattern.get("commands", [])
            script_name = script_proposer._generate_script_name(commands)
            
            # Create script metadata
            metadata = {
                "name": script_name,
                "description": f"Generated from pattern {pattern.get('id')}",
                "author": "collaborative_improvement",
                "version": "1.0",
                "tags": ["auto-generated", "pattern"],
                "source_pattern": pattern.get("id"),
                "source_proposal": proposal_id
            }
            
            # Add script to library
            script_result = script_library.add_script(proposal.get("script", ""), metadata)
            if not script_result.get("success", False):
                return {"success": False, "error": f"Failed to add script to library: {script_result.get('error')}"}
            
            # Update proposal
            proposal["committed"] = True
            proposal["script_id"] = script_result.get("script_id")
            script_proposer._save_proposals()
            
            # Update pattern
            pattern["committed"] = True
            pattern["script_id"] = script_result.get("script_id")
            pattern_detector._save_patterns()
            
            logger.info(f"Committed script for proposal {proposal_id}")
            
            return {
                "success": True,
                "proposal_id": proposal_id,
                "script_id": script_result.get("script_id"),
                "script_name": script_name
            }
        except Exception as e:
            logger.error(f"Error committing script: {e}")
            return {"success": False, "error": str(e)}
    
    def shutdown(self) -> None:
        """
        Shutdown the Peer Review Workflow.
        """
        if not self.initialized:
            return
        
        self.initialized = False
        logger.info("Peer Review Workflow shutdown complete")

class DecisionTreeGrowth:
    """
    Decision Tree Growth for enhancing decision-making based on command patterns.
    """
    
    def __init__(self, registry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Decision Tree Growth.
        
        Args:
            registry: Component registry
            config: Optional configuration dictionary
        """
        self.registry = registry
        self.config = config or {}
        self.initialized = False
        
        # Configuration parameters
        self.growth_rate = self.config.get("growth_rate", 0.1)
        self.pruning_threshold = self.config.get("pruning_threshold", 0.05)
        
        # Decision tree
        self.decision_tree = {}
        
        # Register with registry
        registry.register_component("decision_tree_growth", self)
        
        logger.info("Decision Tree Growth initialized with default settings")
    
    def initialize(self) -> bool:
        """
        Initialize the Decision Tree Growth.
        
        Returns:
            True if initialization was successful
        """
        if self.initialized:
            logger.warning("Decision Tree Growth already initialized")
            return True
        
        try:
            # Load decision tree from file if available
            self._load_decision_tree()
            
            self.initialized = True
            logger.info("Decision Tree Growth initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing Decision Tree Growth: {e}")
            return False
    
    def _load_decision_tree(self) -> None:
        """Load decision tree from file."""
        tree_path = os.path.join("collaborative_improvement", "decision_tree.json")
        if os.path.exists(tree_path):
            try:
                with open(tree_path, "r") as f:
                    self.decision_tree = json.load(f)
                logger.info(f"Loaded decision tree with {len(self.decision_tree)} nodes")
            except Exception as e:
                logger.error(f"Error loading decision tree: {e}")
                self.decision_tree = {}
    
    def _save_decision_tree(self) -> None:
        """Save decision tree to file."""
        tree_path = os.path.join("collaborative_improvement", "decision_tree.json")
        os.makedirs(os.path.dirname(tree_path), exist_ok=True)
        try:
            with open(tree_path, "w") as f:
                json.dump(self.decision_tree, f, indent=2)
            logger.info(f"Saved decision tree with {len(self.decision_tree)} nodes")
        except Exception as e:
            logger.error(f"Error saving decision tree: {e}")
    
    def add_decision_path(self, commands: List[str], outcome: str, success: bool) -> Dict[str, Any]:
        """
        Add a decision path to the tree.
        
        Args:
            commands: List of commands
            outcome: Outcome of the commands
            success: Whether the outcome was successful
            
        Returns:
            Dictionary with result information
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Decision Tree Growth initialization failed")
                return {"success": False, "error": "Decision Tree Growth initialization failed"}
        
        try:
            # Create path string
            path = " -> ".join(commands)
            
            # Check if path exists
            if path in self.decision_tree:
                # Update existing path
                node = self.decision_tree[path]
                node["count"] += 1
                node["outcomes"][outcome] = node["outcomes"].get(outcome, 0) + 1
                node["successes"] += 1 if success else 0
                node["success_rate"] = node["successes"] / node["count"]
            else:
                # Create new path
                node = {
                    "path": path,
                    "commands": commands,
                    "count": 1,
                    "outcomes": {outcome: 1},
                    "successes": 1 if success else 0,
                    "success_rate": 1.0 if success else 0.0,
                    "created": time.time(),
                    "updated": time.time()
                }
                self.decision_tree[path] = node
            
            # Update timestamp
            node["updated"] = time.time()
            
            # Save decision tree
            self._save_decision_tree()
            
            logger.info(f"Added decision path: {path}")
            
            return {
                "success": True,
                "path": path,
                "node": node
            }
        except Exception as e:
            logger.error(f"Error adding decision path: {e}")
            return {"success": False, "error": str(e)}
    
    def get_decision_recommendation(self, commands: List[str]) -> Dict[str, Any]:
        """
        Get a recommendation for a decision path.
        
        Args:
            commands: List of commands
            
        Returns:
            Dictionary with recommendation information
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Decision Tree Growth initialization failed")
                return {"success": False, "error": "Decision Tree Growth initialization failed"}
        
        try:
            # Create path string
            path = " -> ".join(commands)
            
            # Check if exact path exists
            if path in self.decision_tree:
                node = self.decision_tree[path]
                
                # Get best outcome
                best_outcome = max(node["outcomes"].items(), key=lambda x: x[1])
                
                return {
                    "success": True,
                    "exact_match": True,
                    "path": path,
                    "recommendation": best_outcome[0],
                    "confidence": best_outcome[1] / node["count"],
                    "success_rate": node["success_rate"],
                    "count": node["count"]
                }
            
            # Look for partial matches
            matches = []
            for tree_path, node in self.decision_tree.items():
                # Check if commands are a prefix of this path
                tree_commands = node["commands"]
                if len(commands) <= len(tree_commands):
                    if all(c1 == c2 for c1, c2 in zip(commands, tree_commands)):
                        matches.append((node, len(commands) / len(tree_commands)))
            
            if matches:
                # Sort by match score and count
                matches.sort(key=lambda x: (x[1], x[0]["count"]), reverse=True)
                best_match = matches[0][0]
                
                # Get best outcome
                best_outcome = max(best_match["outcomes"].items(), key=lambda x: x[1])
                
                return {
                    "success": True,
                    "exact_match": False,
                    "path": best_match["path"],
                    "recommendation": best_outcome[0],
                    "confidence": best_outcome[1] / best_match["count"] * matches[0][1],
                    "success_rate": best_match["success_rate"],
                    "count": best_match["count"]
                }
            
            return {
                "success": False,
                "error": "No matching decision paths found"
            }
        except Exception as e:
            logger.error(f"Error getting decision recommendation: {e}")
            return {"success": False, "error": str(e)}
    
    def prune_decision_tree(self) -> Dict[str, Any]:
        """
        Prune the decision tree.
        
        Returns:
            Dictionary with pruning information
        """
        if not self.initialized:
            if not self.initialize():
                logger.error("Decision Tree Growth initialization failed")
                return {"success": False, "error": "Decision Tree Growth initialization failed"}
        
        try:
            # Count nodes before pruning
            before_count = len(self.decision_tree)
            
            # Get current time
            now = time.time()
            
            # Define pruning criteria
            max_age = 90 * 24 * 60 * 60  # 90 days in seconds
            min_count = 3  # Minimum number of occurrences
            
            # Collect nodes to prune
            to_prune = []
            for path, node in self.decision_tree.items():
                # Prune old nodes
                if now - node.get("updated", now) > max_age:
                    to_prune.append(path)
                    continue
                
                # Prune nodes with low counts
                if node.get("count", 0) < min_count:
                    to_prune.append(path)
                    continue
                
                # Prune nodes with low success rate
                if node.get("success_rate", 0) < self.pruning_threshold:
                    to_prune.append(path)
                    continue
            
            # Prune nodes
            for path in to_prune:
                del self.decision_tree[path]
            
            # Save decision tree
            self._save_decision_tree()
            
            # Count nodes after pruning
            after_count = len(self.decision_tree)
            
            logger.info(f"Pruned {before_count - after_count} nodes from decision tree")
            
            return {
                "success": True,
                "before_count": before_count,
                "after_count": after_count,
                "pruned_count": before_count - after_count
            }
        except Exception as e:
            logger.error(f"Error pruning decision tree: {e}")
            return {"success": False, "error": str(e)}
    
    def shutdown(self) -> None:
        """
        Shutdown the Decision Tree Growth.
        """
        if not self.initialized:
            return
        
        # Save decision tree
        self._save_decision_tree()
        
        self.initialized = False
        logger.info("Decision Tree Growth shutdown complete")


# Factory functions for the Launchpad's component registry

def get_pattern_detector(registry, config: Optional[Dict[str, Any]] = None) -> PatternDetector:
    """
    Get the singleton instance of the Pattern Detector.
    
    Args:
        registry: Component registry
        config: Optional configuration dictionary
        
    Returns:
        PatternDetector instance
    """
    return PatternDetector(registry, config)

def get_script_proposer(registry, config: Optional[Dict[str, Any]] = None) -> ScriptProposer:
    """
    Get the singleton instance of the Script Proposer.
    
    Args:
        registry: Component registry
        config: Optional configuration dictionary
        
    Returns:
        ScriptProposer instance
    """
    return ScriptProposer(registry, config)

def get_peer_review_workflow(registry, config: Optional[Dict[str, Any]] = None) -> PeerReviewWorkflow:
    """
    Get the singleton instance of the Peer Review Workflow.
    
    Args:
        registry: Component registry
        config: Optional configuration dictionary
        
    Returns:
        PeerReviewWorkflow instance
    """
    return PeerReviewWorkflow(registry, config)

def get_decision_tree_growth(registry, config: Optional[Dict[str, Any]] = None) -> DecisionTreeGrowth:
    """
    Get the singleton instance of the Decision Tree Growth.
    
    Args:
        registry: Component registry
        config: Optional configuration dictionary
        
    Returns:
        DecisionTreeGrowth instance
    """
    return DecisionTreeGrowth(registry, config)
