#!/usr/bin/env python3
"""
improvement_feedback_loop.py
───────────────────────────
System improvement feedback loop for continuously enhancing the system.

This module provides a continuous feedback loop that takes insights from blocker patterns
and preventative strategies to make automatic improvements to the system.
"""

import os
import sys
import time
import json
import logging
import datetime
import traceback
import threading
import schedule
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from enum import Enum
from collections import defaultdict

# Internal imports
from shell_environment import register_command, emit_event, EventType
from meta_agent import request_agent_task
from blocker_detection import BlockerType, BlockerSeverity, BlockerDetector
from response_protocol import ResponseStrategy
from blocker_learning import BlockerPattern, PatternType, BlockerLearningSystem
from preventative_strategy import StrategyScope, StrategyStatus, PreventativeStrategyManager

# Configure logging
logger = logging.getLogger("ImprovementFeedbackLoop")
handler = logging.FileHandler("improvement_feedback_loop.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ImprovementSource(Enum):
    """Sources of system improvements."""
    BLOCKER_PATTERN = "blocker_pattern"           # Improvement derived from blocker pattern
    STRATEGY_EVALUATION = "strategy_evaluation"   # Improvement derived from strategy evaluation
    USER_FEEDBACK = "user_feedback"               # Improvement derived from user feedback
    AGENT_SUGGESTION = "agent_suggestion"         # Improvement derived from agent suggestion
    SYSTEM_METRICS = "system_metrics"             # Improvement derived from system metrics
    AUDIT_REPORT = "audit_report"                 # Improvement derived from audit report

class ImprovementStatus(Enum):
    """Status of system improvements."""
    PROPOSED = "proposed"               # Improvement has been proposed
    APPROVED = "approved"               # Improvement has been approved
    IMPLEMENTED = "implemented"         # Improvement has been implemented
    VERIFIED = "verified"               # Improvement has been verified
    REVERTED = "reverted"               # Improvement has been reverted
    REJECTED = "rejected"               # Improvement has been rejected

class ImprovementImpact(Enum):
    """Impact level of system improvements."""
    LOW = "low"                         # Low impact on system
    MEDIUM = "medium"                   # Medium impact on system
    HIGH = "high"                       # High impact on system
    CRITICAL = "critical"               # Critical impact on system

class ImprovementType(Enum):
    """Types of system improvements."""
    CONFIGURATION = "configuration"     # Changes to system configuration
    CODE = "code"                       # Changes to system code
    PROCESS = "process"                 # Changes to system processes
    RESOURCE = "resource"               # Changes to system resources
    DEPENDENCY = "dependency"           # Changes to system dependencies
    MONITORING = "monitoring"           # Changes to system monitoring
    DOCUMENTATION = "documentation"     # Changes to system documentation

class SystemImprovementFeedbackLoop:
    """
    System improvement feedback loop for continuously enhancing the system.
    
    Provides a continuous feedback loop that takes insights from blocker patterns
    and preventative strategies to make automatic improvements to the system.
    """
    
    def __init__(self, 
                storage_file: str = "system_improvements.json",
                blocker_detector: Optional[BlockerDetector] = None,
                blocker_learning_system: Optional[BlockerLearningSystem] = None,
                preventative_strategy_manager: Optional[PreventativeStrategyManager] = None):
        """
        Initialize the system improvement feedback loop.
        
        Args:
            storage_file: File to store system improvements
            blocker_detector: Blocker detector to use
            blocker_learning_system: Blocker learning system to use
            preventative_strategy_manager: Preventative strategy manager to use
        """
        self.storage_file = storage_file
        self.improvements = {}
        self.blocker_detector = blocker_detector or BlockerDetector()
        self.blocker_learning_system = blocker_learning_system or BlockerLearningSystem()
        self.preventative_strategy_manager = preventative_strategy_manager or PreventativeStrategyManager()
        self.auto_improve = True
        self.scheduler_thread = None
        self.shutdown_event = threading.Event()
        
        # Load improvements
        self.load_improvements()
        
        # Register commands
        try:
            register_command("propose_improvement", self.propose_improvement_command, 
                            "Propose a system improvement")
            register_command("list_improvements", self.list_improvements_command,
                            "List system improvements")
            register_command("implement_improvement", self.implement_improvement_command,
                            "Implement a system improvement")
            register_command("verify_improvement", self.verify_improvement_command,
                            "Verify a system improvement")
            register_command("start_feedback_loop", self.start_feedback_loop_command,
                            "Start the system improvement feedback loop")
            register_command("stop_feedback_loop", self.stop_feedback_loop_command,
                            "Stop the system improvement feedback loop")
            register_command("run_feedback_cycle", self.run_feedback_cycle_command,
                            "Run a feedback cycle")
            logger.info("System improvement feedback loop commands registered")
        except Exception as e:
            logger.error(f"Failed to register commands: {e}")
    
    def load_improvements(self):
        """Load improvements from storage file."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                
                for improvement_id, improvement_data in data.items():
                    self.improvements[improvement_id] = improvement_data
                
                logger.info(f"Loaded {len(self.improvements)} system improvements from {self.storage_file}")
            else:
                logger.info(f"No improvement storage file found at {self.storage_file}")
        except Exception as e:
            logger.error(f"Error loading improvements: {e}")
    
    def save_improvements(self):
        """Save improvements to storage file."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.improvements, f, indent=2)
            
            logger.info(f"Saved {len(self.improvements)} system improvements to {self.storage_file}")
        except Exception as e:
            logger.error(f"Error saving improvements: {e}")
    
    def start_feedback_loop(self):
        """Start the system improvement feedback loop."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Feedback loop already running")
            return False
        
        # Reset shutdown event
        self.shutdown_event.clear()
        
        # Create and start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("System improvement feedback loop started")
        return True
    
    def stop_feedback_loop(self):
        """Stop the system improvement feedback loop."""
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            logger.warning("Feedback loop not running")
            return False
        
        # Set shutdown event to stop the loop
        self.shutdown_event.set()
        
        # Wait for the thread to stop
        self.scheduler_thread.join(timeout=5)
        
        logger.info("System improvement feedback loop stopped")
        return True
    
    def _scheduler_loop(self):
        """Scheduler loop for running feedback cycles."""
        # Schedule feedback cycle to run every hour
        schedule.every(1).hours.do(self.run_feedback_cycle)
        
        # Schedule daily report generation
        schedule.every().day.at("00:00").do(self._generate_improvement_report)
        
        logger.info("Scheduler loop started")
        
        try:
            while not self.shutdown_event.is_set():
                # Run pending scheduled tasks
                schedule.run_pending()
                
                # Sleep for a short time to avoid high CPU usage
                time.sleep(10)
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
        finally:
            logger.info("Scheduler loop stopped")
    
    def run_feedback_cycle(self) -> Dict[str, Any]:
        """
        Run a feedback cycle.
        
        Returns:
            Results of the feedback cycle
        """
        logger.info("Running feedback cycle")
        
        results = {
            "start_time": time.time(),
            "improvements_proposed": 0,
            "improvements_implemented": 0,
            "improvements_verified": 0,
            "successful_improvements": [],
            "failed_improvements": [],
            "end_time": None
        }
        
        try:
            # Step 1: Analyze blocker patterns for improvement opportunities
            pattern_improvements = self._analyze_blocker_patterns()
            results["improvements_proposed"] += len(pattern_improvements)
            
            # Step 2: Analyze strategy evaluations for improvement opportunities
            strategy_improvements = self._analyze_strategy_evaluations()
            results["improvements_proposed"] += len(strategy_improvements)
            
            # Step 3: Get improvement suggestions from agent
            agent_improvements = self._get_agent_suggestions()
            results["improvements_proposed"] += len(agent_improvements)
            
            # Combine all improvements
            all_improvements = pattern_improvements + strategy_improvements + agent_improvements
            
            # Step 4: Implement high-priority improvements if auto-improve is enabled
            if self.auto_improve and all_improvements:
                for improvement_id in all_improvements:
                    # Check improvement priority
                    improvement = self.improvements[improvement_id]
                    
                    if improvement["status"] == ImprovementStatus.APPROVED.value:
                        # Implement improvement
                        implementation_result = self.implement_improvement(improvement_id)
                        
                        if implementation_result.get("success", False):
                            results["improvements_implemented"] += 1
                            
                            # Verify improvement
                            verification_result = self.verify_improvement(improvement_id)
                            
                            if verification_result.get("success", False):
                                results["improvements_verified"] += 1
                                results["successful_improvements"].append(improvement_id)
                            else:
                                results["failed_improvements"].append(improvement_id)
            
            # Set end time
            results["end_time"] = time.time()
            results["duration"] = results["end_time"] - results["start_time"]
            
            # Log results
            logger.info(f"Feedback cycle completed: {results['improvements_proposed']} proposed, {results['improvements_implemented']} implemented, {results['improvements_verified']} verified")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running feedback cycle: {e}")
            results["error"] = str(e)
            results["end_time"] = time.time()
            results["duration"] = results["end_time"] - results["start_time"]
            return results
    
    def _analyze_blocker_patterns(self) -> List[str]:
        """
        Analyze blocker patterns for improvement opportunities.
        
        Returns:
            List of improvement IDs
        """
        improvement_ids = []
        
        # Get patterns from blocker learning system
        patterns = self.blocker_learning_system.patterns
        
        # Filter for recurring patterns (seen at least 3 times)
        recurring_patterns = [p for p in patterns.values() if p.occurrence_count >= 3]
        
        for pattern in recurring_patterns:
            # Check if we already have an improvement for this pattern
            pattern_improvements = [i for i in self.improvements.values() 
                                   if i.get("source_id") == pattern.pattern_id and 
                                      i.get("source") == ImprovementSource.BLOCKER_PATTERN.value]
            
            if not pattern_improvements:
                # Generate improvement from pattern
                improvement = self._generate_improvement_from_pattern(pattern)
                
                if improvement:
                    self.improvements[improvement["id"]] = improvement
                    improvement_ids.append(improvement["id"])
        
        # Save improvements
        self.save_improvements()
        
        logger.info(f"Generated {len(improvement_ids)} improvements from blocker patterns")
        
        return improvement_ids
    
    def _generate_improvement_from_pattern(self, pattern: BlockerPattern) -> Optional[Dict[str, Any]]:
        """
        Generate an improvement from a blocker pattern.
        
        Args:
            pattern: Blocker pattern
            
        Returns:
            Improvement information or None if generation failed
        """
        # Try to generate an improvement using an agent
        try:
            # Prepare data for the agent
            pattern_data = {
                "id": pattern.pattern_id,
                "type": pattern.pattern_type.value,
                "description": pattern.description,
                "occurrences": pattern.occurrence_count,
                "resolved_count": pattern.resolved_count,
                "success_rate": pattern.resolved_count / max(1, pattern.occurrence_count),
                "first_seen": pattern.first_seen,
                "last_seen": pattern.last_seen
            }
            
            # Get sample instances
            sample_instances = pattern.instances[-5:] if len(pattern.instances) > 5 else pattern.instances
            
            # Prepare prompt for agent
            prompt = f"""
            Generate a system improvement based on the following blocker pattern:
            
            Pattern ID: {pattern.pattern_id}
            Pattern Type: {pattern.pattern_type.value}
            Description: {pattern.description}
            Occurrences: {pattern.occurrence_count}
            Success Rate: {pattern.resolved_count / max(1, pattern.occurrence_count):.2f}
            
            Sample Instances:
            {json.dumps(sample_instances, indent=2)}
            
            Please provide a system improvement that includes:
            
            1. Title: A clear, descriptive title for the improvement
            2. Description: A detailed description of the improvement
            3. Type: The type of improvement (configuration, code, process, resource, dependency, monitoring, documentation)
            4. Impact: The impact level of the improvement (low, medium, high, critical)
            5. Implementation Steps: Detailed steps to implement the improvement
            6. Verification Steps: Steps to verify the improvement was successful
            7. Rollback Steps: Steps to rollback the improvement if it fails
            
            Return the improvement as a structured object.
            """
            
            # Request improvement from agent
            result = request_agent_task("system_improvement_generation", prompt, timeout=60)
            
            if result.get("success", False) and result.get("improvement"):
                improvement = result["improvement"]
                
                # Add metadata
                improvement_id = f"improvement_{int(time.time())}_{len(self.improvements)}"
                improvement["id"] = improvement_id
                improvement["source"] = ImprovementSource.BLOCKER_PATTERN.value
                improvement["source_id"] = pattern.pattern_id
                improvement["status"] = ImprovementStatus.PROPOSED.value
                improvement["created_at"] = time.time()
                improvement["updated_at"] = time.time()
                
                logger.info(f"Generated improvement {improvement_id} from pattern {pattern.pattern_id}: {improvement.get('title', 'Untitled improvement')}")
                
                return improvement
            
            logger.warning(f"Failed to generate improvement from pattern {pattern.pattern_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error generating improvement from pattern {pattern.pattern_id}: {e}")
            return None
    
    def _analyze_strategy_evaluations(self) -> List[str]:
        """
        Analyze strategy evaluations for improvement opportunities.
        
        Returns:
            List of improvement IDs
        """
        improvement_ids = []
        
        # Get strategies from preventative strategy manager
        strategies = self.preventative_strategy_manager.strategies
        
        # Filter for effective strategies (effectiveness > 0.7)
        effective_strategies = [s for s in strategies.values() 
                               if s.get("effectiveness", 0) > 0.7 and
                                  s.get("status") == StrategyStatus.EVALUATED.value]
        
        for strategy in effective_strategies:
            # Check if we already have an improvement for this strategy
            strategy_improvements = [i for i in self.improvements.values() 
                                    if i.get("source_id") == strategy["id"] and 
                                       i.get("source") == ImprovementSource.STRATEGY_EVALUATION.value]
            
            if not strategy_improvements:
                # Generate improvement from strategy
                improvement = self._generate_improvement_from_strategy(strategy)
                
                if improvement:
                    self.improvements[improvement["id"]] = improvement
                    improvement_ids.append(improvement["id"])
        
        # Save improvements
        self.save_improvements()
        
        logger.info(f"Generated {len(improvement_ids)} improvements from strategy evaluations")
        
        return improvement_ids
    
    def _generate_improvement_from_strategy(self, strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate an improvement from a preventative strategy.
        
        Args:
            strategy: Preventative strategy
            
        Returns:
            Improvement information or None if generation failed
        """
        # Try to generate an improvement using an agent
        try:
            # Prepare prompt for agent
            prompt = f"""
            Generate a system improvement based on the following preventative strategy:
            
            Strategy ID: {strategy["id"]}
            Name: {strategy.get("name", "Unnamed strategy")}
            Description: {strategy.get("description", "No description")}
            Scope: {strategy.get("scope", "Unknown scope")}
            Effectiveness: {strategy.get("effectiveness", 0):.2f}
            
            Implementation Steps:
            {json.dumps(strategy.get("implementation_steps", []), indent=2)}
            
            Please provide a system improvement that would make this strategy's approach
            a permanent part of the system, rather than a reactive measure. The improvement should include:
            
            1. Title: A clear, descriptive title for the improvement
            2. Description: A detailed description of the improvement
            3. Type: The type of improvement (configuration, code, process, resource, dependency, monitoring, documentation)
            4. Impact: The impact level of the improvement (low, medium, high, critical)
            5. Implementation Steps: Detailed steps to implement the improvement
            6. Verification Steps: Steps to verify the improvement was successful
            7. Rollback Steps: Steps to rollback the improvement if it fails
            
            Return the improvement as a structured object.
            """
            
            # Request improvement from agent
            result = request_agent_task("system_improvement_generation", prompt, timeout=60)
            
            if result.get("success", False) and result.get("improvement"):
                improvement = result["improvement"]
                
                # Add metadata
                improvement_id = f"improvement_{int(time.time())}_{len(self.improvements)}"
                improvement["id"] = improvement_id
                improvement["source"] = ImprovementSource.STRATEGY_EVALUATION.value
                improvement["source_id"] = strategy["id"]
                improvement["status"] = ImprovementStatus.PROPOSED.value
                improvement["created_at"] = time.time()
                improvement["updated_at"] = time.time()
                
                logger.info(f"Generated improvement {improvement_id} from strategy {strategy['id']}: {improvement.get('title', 'Untitled improvement')}")
                
                return improvement
            
            logger.warning(f"Failed to generate improvement from strategy {strategy['id']}")
            return None
            
        except Exception as e:
            logger.error(f"Error generating improvement from strategy {strategy['id']}: {e}")
            return None
    
    def _get_agent_suggestions(self) -> List[str]:
        """
        Get improvement suggestions from agent.
        
        Returns:
            List of improvement IDs
        """
        improvement_ids = []
        
        # Try to get suggestions using an agent
        try:
            # Prepare prompt for agent
            prompt = """
            Based on your knowledge of the system, suggest potential improvements that could enhance
            system performance, reliability, or user experience. Consider areas such as:
            
            1. Error handling and recovery
            2. Resource optimization
            3. User interface and experience
            4. Code structure and organization
            5. Documentation and help
            
            For each suggestion, provide:
            
            1. Title: A clear, descriptive title for the improvement
            2. Description: A detailed description of the improvement
            3. Type: The type of improvement (configuration, code, process, resource, dependency, monitoring, documentation)
            4. Impact: The impact level of the improvement (low, medium, high, critical)
            5. Implementation Steps: Detailed steps to implement the improvement
            6. Verification Steps: Steps to verify the improvement was successful
            7. Rollback Steps: Steps to rollback the improvement if it fails
            
            Return 1-3 improvements as a list of structured objects.
            """
            
            # Request suggestions from agent
            result = request_agent_task("improvement_suggestions", prompt, timeout=60)
            
            if result.get("success", False) and result.get("improvements"):
                suggestions = result["improvements"]
                
                for suggestion in suggestions:
                    # Add metadata
                    improvement_id = f"improvement_{int(time.time())}_{len(self.improvements)}"
                    suggestion["id"] = improvement_id
                    suggestion["source"] = ImprovementSource.AGENT_SUGGESTION.value
                    suggestion["source_id"] = None
                    suggestion["status"] = ImprovementStatus.PROPOSED.value
                    suggestion["created_at"] = time.time()
                    suggestion["updated_at"] = time.time()
                    
                    # Add to improvements
                    self.improvements[improvement_id] = suggestion
                    improvement_ids.append(improvement_id)
                
                # Save improvements
                self.save_improvements()
                
                logger.info(f"Generated {len(improvement_ids)} improvements from agent suggestions")
            
            return improvement_ids
            
        except Exception as e:
            logger.error(f"Error getting agent suggestions: {e}")
            return []
    
    def propose_improvement(self, title: str, description: str, improvement_type: str, 
                           impact: str, implementation_steps: List[str], 
                           verification_steps: List[str], rollback_steps: List[str]) -> Dict[str, Any]:
        """
        Propose a system improvement.
        
        Args:
            title: Title of the improvement
            description: Description of the improvement
            improvement_type: Type of improvement
            impact: Impact of the improvement
            implementation_steps: Steps to implement the improvement
            verification_steps: Steps to verify the improvement
            rollback_steps: Steps to rollback the improvement
            
        Returns:
            Improvement information
        """
        # Create improvement
        improvement_id = f"improvement_{int(time.time())}_{len(self.improvements)}"
        
        improvement = {
            "id": improvement_id,
            "title": title,
            "description": description,
            "type": improvement_type,
            "impact": impact,
            "implementation_steps": implementation_steps,
            "verification_steps": verification_steps,
            "rollback_steps": rollback_steps,
            "source": ImprovementSource.USER_FEEDBACK.value,
            "source_id": None,
            "status": ImprovementStatus.PROPOSED.value,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # Add to improvements
        self.improvements[improvement_id] = improvement
        
        # Save improvements
        self.save_improvements()
        
        logger.info(f"Proposed improvement {improvement_id}: {title}")
        
        return {
            "success": True,
            "improvement_id": improvement_id,
            "improvement": improvement
        }
    
    def approve_improvement(self, improvement_id: str) -> Dict[str, Any]:
        """
        Approve a system improvement.
        
        Args:
            improvement_id: ID of the improvement to approve
            
        Returns:
            Approval information
        """
        # Check if improvement exists
        if improvement_id not in self.improvements:
            return {
                "success": False,
                "error": f"Improvement {improvement_id} not found"
            }
        
        improvement = self.improvements[improvement_id]
        
        # Check if improvement is in the right state
        if improvement["status"] != ImprovementStatus.PROPOSED.value:
            return {
                "success": False,
                "error": f"Improvement {improvement_id} is not in the proposed state"
            }
        
        # Update improvement
        improvement["status"] = ImprovementStatus.APPROVED.value
        improvement["updated_at"] = time.time()
        improvement["approved_at"] = time.time()
        
        # Save improvements
        self.save_improvements()
        
        logger.info(f"Approved improvement {improvement_id}: {improvement['title']}")
        
        return {
            "success": True,
            "improvement_id": improvement_id
        }
    
    def implement_improvement(self, improvement_id: str) -> Dict[str, Any]:
        """
        Implement a system improvement.
        
        Args:
            improvement_id: ID of the improvement to implement
            
        Returns:
            Implementation information
        """
        # Check if improvement exists
        if improvement_id not in self.improvements:
            return {
                "success": False,
                "error": f"Improvement {improvement_id} not found"
            }
        
        improvement = self.improvements[improvement_id]
        
        # Check if improvement is in the right state
        if improvement["status"] != ImprovementStatus.APPROVED.value:
            return {
                "success": False,
                "error": f"Improvement {improvement_id} is not in the approved state"
            }
        
        # Get implementation steps
        implementation_steps = improvement.get("implementation_steps", [])
        
        if not implementation_steps:
            return {
                "success": False,
                "error": f"Improvement {improvement_id} has no implementation steps"
            }
        
        # Execute implementation steps
        implementation_results = []
        
        try:
            for i, step in enumerate(implementation_steps, 1):
                logger.info(f"Implementing step {i}/{len(implementation_steps)} for improvement {improvement_id}: {step}")
                
                # Here you would implement the actual step
                # For now, we'll just simulate it
                
                implementation_results.append({
                    "step": i,
                    "description": step,
                    "status": "implemented",
                    "timestamp": time.time()
                })
            
            # Update improvement
            improvement["status"] = ImprovementStatus.IMPLEMENTED.value
            improvement["updated_at"] = time.time()
            improvement["implemented_at"] = time.time()
            improvement["implementation_results"] = implementation_results
            
            # Save improvements
            self.save_improvements()
            
            logger.info(f"Implemented improvement {improvement_id}: {improvement['title']}")
            
            return {
                "success": True,
                "improvement_id": improvement_id,
                "implementation_results": implementation_results
            }
            
        except Exception as e:
            logger.error(f"Error implementing improvement {improvement_id}: {e}")
            
            # Update improvement with partial results
            improvement["implementation_results"] = implementation_results
            improvement["implementation_error"] = str(e)
            
            # Save improvements
            self.save_improvements()
            
            return {
                "success": False,
                "error": f"Error implementing improvement {improvement_id}: {e}",
                "implementation_results": implementation_results
            }
    
    def verify_improvement(self, improvement_id: str) -> Dict[str, Any]:
        """
        Verify a system improvement.
        
        Args:
            improvement_id: ID of the improvement to verify
            
        Returns:
            Verification information
        """
        # Check if improvement exists
        if improvement_id not in self.improvements:
            return {
                "success": False,
                "error": f"Improvement {improvement_id} not found"
            }
        
        improvement = self.improvements[improvement_id]
        
        # Check if improvement is in the right state
        if improvement["status"] != ImprovementStatus.IMPLEMENTED.value:
            return {
                "success": False,
                "error": f"Improvement {improvement_id} is not in the implemented state"
            }
        
        # Get verification steps
        verification_steps = improvement.get("verification_steps", [])
        
        if not verification_steps:
            return {
                "success": False,
                "error": f"Improvement {improvement_id} has no verification steps"
            }
        
        # Execute verification steps
        verification_results = []
        
        try:
            all_verified = True
            
            for i, step in enumerate(verification_steps, 1):
                logger.info(f"Verifying step {i}/{len(verification_steps)} for improvement {improvement_id}: {step}")
                
                # Here you would verify the actual step
                # For now, we'll just simulate it
                
                # Simulate verification success (90% chance)
                verified = (hash(f"{improvement_id}_{i}") % 10) < 9
                
                verification_results.append({
                    "step": i,
                    "description": step,
                    "verified": verified,
                    "timestamp": time.time()
                })
                
                if not verified:
                    all_verified = False
            
            # Update improvement
            improvement["updated_at"] = time.time()
            improvement["verification_results"] = verification_results
            
            if all_verified:
                improvement["status"] = ImprovementStatus.VERIFIED.value
                improvement["verified_at"] = time.time()
                logger.info(f"Verified improvement {improvement_id}: {improvement['title']}")
            else:
                improvement["status"] = ImprovementStatus.REVERTED.value
                improvement["reverted_at"] = time.time()
                logger.warning(f"Reverted improvement {improvement_id}: {improvement['title']} - verification failed")
                
                # Execute rollback
                self._rollback_improvement(improvement_id)
            
            # Save improvements
            self.save_improvements()
            
            return {
                "success": all_verified,
                "improvement_id": improvement_id,
                "verification_results": verification_results,
                "status": improvement["status"]
            }
            
        except Exception as e:
            logger.error(f"Error verifying improvement {improvement_id}: {e}")
            
            # Update improvement with partial results
            improvement["verification_results"] = verification_results
            improvement["verification_error"] = str(e)
            
            # Save improvements
            self.save_improvements()
            
            return {
                "success": False,
                "error": f"Error verifying improvement {improvement_id}: {e}",
                "verification_results": verification_results
            }
    
    def _rollback_improvement(self, improvement_id: str) -> Dict[str, Any]:
        """
        Rollback a system improvement.
        
        Args:
            improvement_id: ID of the improvement to rollback
            
        Returns:
            Rollback information
        """
        # Check if improvement exists
        if improvement_id not in self.improvements:
            return {
                "success": False,
                "error": f"Improvement {improvement_id} not found"
            }
        
        improvement = self.improvements[improvement_id]
        
        # Get rollback steps
        rollback_steps = improvement.get("rollback_steps", [])
        
        if not rollback_steps:
            logger.warning(f"Improvement {improvement_id} has no rollback steps")
            rollback_steps = ["Revert changes made by improvement"]
        
        # Execute rollback steps
        rollback_results = []
        
        try:
            for i, step in enumerate(rollback_steps, 1):
                logger.info(f"Rolling back step {i}/{len(rollback_steps)} for improvement {improvement_id}: {step}")
                
                # Here you would implement the actual rollback step
                # For now, we'll just simulate it
                
                rollback_results.append({
                    "step": i,
                    "description": step,
                    "status": "reverted",
                    "timestamp": time.time()
                })
            
            # Update improvement
            improvement["status"] = ImprovementStatus.REVERTED.value
            improvement["updated_at"] = time.time()
            improvement["reverted_at"] = time.time()
            improvement["rollback_results"] = rollback_results
            
            # Save improvements
            self.save_improvements()
            
            logger.info(f"Rolled back improvement {improvement_id}: {improvement['title']}")
            
            return {
                "success": True,
                "improvement_id": improvement_id,
                "rollback_results": rollback_results
            }
            
        except Exception as e:
            logger.error(f"Error rolling back improvement {improvement_id}: {e}")
            
            # Update improvement with partial results
            improvement["rollback_results"] = rollback_results
            improvement["rollback_error"] = str(e)
            
            # Save improvements
            self.save_improvements()
            
            return {
                "success": False,
                "error": f"Error rolling back improvement {improvement_id}: {e}",
                "rollback_results": rollback_results
            }
    
    def _generate_improvement_report(self) -> Dict[str, Any]:
        """
        Generate a report of system improvements.
        
        Returns:
            Report information
        """
        logger.info("Generating improvement report")
        
        # Get improvements
        improvements = self.improvements
        
        # Count improvements by status
        status_counts = defaultdict(int)
        for improvement in improvements.values():
            status_counts[improvement.get("status", "unknown")] += 1
        
        # Count improvements by type
        type_counts = defaultdict(int)
        for improvement in improvements.values():
            type_counts[improvement.get("type", "unknown")] += 1
        
        # Count improvements by source
        source_counts = defaultdict(int)
        for improvement in improvements.values():
            source_counts[improvement.get("source", "unknown")] += 1
        
        # Count improvements by impact
        impact_counts = defaultdict(int)
        for improvement in improvements.values():
            impact_counts[improvement.get("impact", "unknown")] += 1
        
        # Get successful improvements
        successful_improvements = [i for i in improvements.values() 
                                  if i.get("status") == ImprovementStatus.VERIFIED.value]
        
        # Get failed improvements
        failed_improvements = [i for i in improvements.values() 
                              if i.get("status") == ImprovementStatus.REVERTED.value]
        
        # Create report
        report = {
            "timestamp": time.time(),
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "total_improvements": len(improvements),
            "status_counts": dict(status_counts),
            "type_counts": dict(type_counts),
            "source_counts": dict(source_counts),
            "impact_counts": dict(impact_counts),
            "successful_improvements": len(successful_improvements),
            "failed_improvements": len(failed_improvements),
            "success_rate": len(successful_improvements) / max(1, len(successful_improvements) + len(failed_improvements))
        }
        
        # Save report to file
        report_file = f"improvement_report_{report['date']}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved improvement report to {report_file}")
        except Exception as e:
            logger.error(f"Error saving improvement report: {e}")
        
        return report
    
    def propose_improvement_command(self, args: str) -> int:
        """
        Handle the propose_improvement command.
        
        Args:
            args: Command arguments (title description type impact implementation_steps verification_steps rollback_steps)
            
        Returns:
            Exit code
        """
        try:
            # Load arguments from JSON file
            if args.strip().endswith(".json"):
                try:
                    with open(args.strip(), 'r') as f:
                        improvement_data = json.load(f)
                    
                    title = improvement_data.get("title", "")
                    description = improvement_data.get("description", "")
                    improvement_type = improvement_data.get("type", "")
                    impact = improvement_data.get("impact", "")
                    implementation_steps = improvement_data.get("implementation_steps", [])
                    verification_steps = improvement_data.get("verification_steps", [])
                    rollback_steps = improvement_data.get("rollback_steps", [])
                except Exception as e:
                    print(f"Error loading improvement data from file: {e}")
                    print("Usage: propose_improvement <improvement_file.json>")
                    print("  or:  propose_improvement interactive")
                    return 1
            elif args.strip() == "interactive":
                # Interactive mode
                print("Proposing a system improvement (interactive mode)")
                
                title = input("Title: ")
                description = input("Description: ")
                
                print("\nAvailable improvement types:")
                for t in ImprovementType:
                    print(f"  {t.value}")
                improvement_type = input("Type: ")
                
                print("\nAvailable impact levels:")
                for i in ImprovementImpact:
                    print(f"  {i.value}")
                impact = input("Impact: ")
                
                implementation_steps = []
                print("\nImplementation steps (enter empty line to finish):")
                while True:
                    step = input(f"Step {len(implementation_steps) + 1}: ")
                    if not step:
                        break
                    implementation_steps.append(step)
                
                verification_steps = []
                print("\nVerification steps (enter empty line to finish):")
                while True:
                    step = input(f"Step {len(verification_steps) + 1}: ")
                    if not step:
                        break
                    verification_steps.append(step)
                
                rollback_steps = []
                print("\nRollback steps (enter empty line to finish):")
                while True:
                    step = input(f"Step {len(rollback_steps) + 1}: ")
                    if not step:
                        break
                    rollback_steps.append(step)
            else:
                print("Error: Invalid arguments.")
                print("Usage: propose_improvement <improvement_file.json>")
                print("  or:  propose_improvement interactive")
                return 1
            
            # Validate inputs
            if not title:
                print("Error: Title is required.")
                return 1
            
            if not description:
                print("Error: Description is required.")
                return 1
            
            if not improvement_type:
                print("Error: Type is required.")
                return 1
            
            if not impact:
                print("Error: Impact is required.")
                return 1
            
            if not implementation_steps:
                print("Error: Implementation steps are required.")
                return 1
            
            # Propose improvement
            result = self.propose_improvement(
                title=title,
                description=description,
                improvement_type=improvement_type,
                impact=impact,
                implementation_steps=implementation_steps,
                verification_steps=verification_steps,
                rollback_steps=rollback_steps
            )
            
            if result.get("success", False):
                improvement_id = result["improvement_id"]
                
                print(f"Improvement proposed: {improvement_id}")
                print(f"Title: {title}")
                print(f"Type: {improvement_type}")
                print(f"Impact: {impact}")
                
                # Automatically approve low and medium impact improvements
                if impact in [ImprovementImpact.LOW.value, ImprovementImpact.MEDIUM.value]:
                    approval_result = self.approve_improvement(improvement_id)
                    
                    if approval_result.get("success", False):
                        print(f"Improvement automatically approved (low/medium impact)")
                    else:
                        print(f"Failed to automatically approve: {approval_result.get('error', 'Unknown error')}")
                else:
                    print(f"Improvement requires manual approval (high/critical impact)")
                
                return 0
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1
                
        except Exception as e:
            print(f"Error proposing improvement: {e}")
            logger.error(f"Error proposing improvement: {e}")
            return 1
    
    def list_improvements_command(self, args: str) -> int:
        """
        Handle the list_improvements command.
        
        Args:
            args: Command arguments (optional status filter)
            
        Returns:
            Exit code
        """
        try:
            status_filter = args.strip() if args.strip() else None
            
            if status_filter:
                try:
                    status_filter = ImprovementStatus(status_filter).value
                except ValueError:
                    print(f"Error: Invalid status filter: {status_filter}")
                    print("Valid statuses:")
                    for status in ImprovementStatus:
                        print(f"  {status.value}")
                    return 1
            
            # Filter improvements
            if status_filter:
                improvements = [i for i in self.improvements.values() if i.get("status") == status_filter]
            else:
                improvements = list(self.improvements.values())
            
            # Sort improvements by creation time
            improvements.sort(key=lambda i: i.get("created_at", 0), reverse=True)
            
            # Print improvements
            print(f"Found {len(improvements)} improvements:")
            
            for i, improvement in enumerate(improvements, 1):
                print(f"\n{i}. ID: {improvement.get('id', 'unknown')}")
                print(f"   Title: {improvement.get('title', 'Untitled improvement')}")
                print(f"   Type: {improvement.get('type', 'Unknown type')}")
                print(f"   Impact: {improvement.get('impact', 'Unknown impact')}")
                print(f"   Status: {improvement.get('status', 'Unknown status')}")
                print(f"   Source: {improvement.get('source', 'Unknown source')}")
                print(f"   Created: {datetime.datetime.fromtimestamp(improvement.get('created_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
                
                if improvement.get("status") == ImprovementStatus.IMPLEMENTED.value:
                    print(f"   Implemented: {datetime.datetime.fromtimestamp(improvement.get('implemented_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
                
                if improvement.get("status") == ImprovementStatus.VERIFIED.value:
                    print(f"   Verified: {datetime.datetime.fromtimestamp(improvement.get('verified_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
                
                if improvement.get("status") == ImprovementStatus.REVERTED.value:
                    print(f"   Reverted: {datetime.datetime.fromtimestamp(improvement.get('reverted_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
            
            return 0
                
        except Exception as e:
            print(f"Error listing improvements: {e}")
            logger.error(f"Error listing improvements: {e}")
            return 1
    
    def implement_improvement_command(self, args: str) -> int:
        """
        Handle the implement_improvement command.
        
        Args:
            args: Command arguments (improvement_id)
            
        Returns:
            Exit code
        """
        try:
            improvement_id = args.strip()
            
            if not improvement_id:
                print("Error: Improvement ID required.")
                print("Usage: implement_improvement <improvement_id>")
                return 1
            
            # Implement improvement
            print(f"Implementing improvement {improvement_id}...")
            
            result = self.implement_improvement(improvement_id)
            
            if result.get("success", False):
                print("Improvement implemented successfully.")
                print(f"Implementation steps completed: {len(result.get('implementation_results', []))}")
                
                return 0
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1
                
        except Exception as e:
            print(f"Error implementing improvement: {e}")
            logger.error(f"Error implementing improvement: {e}")
            return 1
    
    def verify_improvement_command(self, args: str) -> int:
        """
        Handle the verify_improvement command.
        
        Args:
            args: Command arguments (improvement_id)
            
        Returns:
            Exit code
        """
        try:
            improvement_id = args.strip()
            
            if not improvement_id:
                print("Error: Improvement ID required.")
                print("Usage: verify_improvement <improvement_id>")
                return 1
            
            # Verify improvement
            print(f"Verifying improvement {improvement_id}...")
            
            result = self.verify_improvement(improvement_id)
            
            if result.get("success", False):
                print("Improvement verified successfully.")
                print(f"Verification steps completed: {len(result.get('verification_results', []))}")
                print(f"Status: {result.get('status', 'Unknown status')}")
                
                return 0
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                if "verification_results" in result:
                    print("Verification results:")
                    for i, step in enumerate(result["verification_results"], 1):
                        verified = step.get("verified", False)
                        status = "✅" if verified else "❌"
                        print(f"  {status} Step {i}: {step.get('description', 'Unknown step')}")
                
                return 1
                
        except Exception as e:
            print(f"Error verifying improvement: {e}")
            logger.error(f"Error verifying improvement: {e}")
            return 1
    
    def start_feedback_loop_command(self, args: str) -> int:
        """
        Handle the start_feedback_loop command.
        
        Args:
            args: Command arguments (none required)
            
        Returns:
            Exit code
        """
        try:
            # Start feedback loop
            print("Starting system improvement feedback loop...")
            
            result = self.start_feedback_loop()
            
            if result:
                print("Feedback loop started successfully.")
                return 0
            else:
                print("Error: Feedback loop already running.")
                return 1
                
        except Exception as e:
            print(f"Error starting feedback loop: {e}")
            logger.error(f"Error starting feedback loop: {e}")
            return 1
    
    def stop_feedback_loop_command(self, args: str) -> int:
        """
        Handle the stop_feedback_loop command.
        
        Args:
            args: Command arguments (none required)
            
        Returns:
            Exit code
        """
        try:
            # Stop feedback loop
            print("Stopping system improvement feedback loop...")
            
            result = self.stop_feedback_loop()
            
            if result:
                print("Feedback loop stopped successfully.")
                return 0
            else:
                print("Error: Feedback loop not running.")
                return 1
                
        except Exception as e:
            print(f"Error stopping feedback loop: {e}")
            logger.error(f"Error stopping feedback loop: {e}")
            return 1
    
    def run_feedback_cycle_command(self, args: str) -> int:
        """
        Handle the run_feedback_cycle command.
        
        Args:
            args: Command arguments (none required)
            
        Returns:
            Exit code
        """
        try:
            # Run feedback cycle
            print("Running system improvement feedback cycle...")
            
            result = self.run_feedback_cycle()
            
            if "error" not in result:
                print("Feedback cycle completed successfully.")
                print(f"Improvements proposed: {result.get('improvements_proposed', 0)}")
                print(f"Improvements implemented: {result.get('improvements_implemented', 0)}")
                print(f"Improvements verified: {result.get('improvements_verified', 0)}")
                print(f"Duration: {result.get('duration', 0):.2f} seconds")
                
                if result.get("successful_improvements"):
                    print("\nSuccessful improvements:")
                    for improvement_id in result["successful_improvements"]:
                        improvement = self.improvements.get(improvement_id, {})
                        print(f"  {improvement_id}: {improvement.get('title', 'Untitled improvement')}")
                
                if result.get("failed_improvements"):
                    print("\nFailed improvements:")
                    for improvement_id in result["failed_improvements"]:
                        improvement = self.improvements.get(improvement_id, {})
                        print(f"  {improvement_id}: {improvement.get('title', 'Untitled improvement')}")
                
                return 0
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1
                
        except Exception as e:
            print(f"Error running feedback cycle: {e}")
            logger.error(f"Error running feedback cycle: {e}")
            return 1


# Initialize system improvement feedback loop
system_improvement_feedback_loop = SystemImprovementFeedbackLoop()
logger.info("System improvement feedback loop initialized")
