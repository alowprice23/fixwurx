"""
FixWurx Auditor Meta-Awareness Sensor

This module implements a sensor for monitoring the auditor's self-awareness capabilities,
detecting issues like semantic drift, hallucinations, inconsistent reflection, 
and boundary awareness problems.
"""

import logging
import time
import json
import re
from typing import Dict, List, Set, Any, Optional, Union, Tuple, Callable
from collections import deque

from sensor_base import ErrorSensor
from error_report import ErrorReport

# Import LLM bridges - handle both the real and mock versions
try:
    import sensor_llm_bridge
except ImportError:
    sensor_llm_bridge = None

try:
    import mock_llm_bridge
except ImportError:
    mock_llm_bridge = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [MetaAwarenessSensor] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('meta_awareness_sensor')


class MetaAwarenessSensor(ErrorSensor):
    """
    Monitors the auditor's self-awareness capabilities, detecting issues with
    its understanding of its own code, state, and knowledge boundaries.
    
    This sensor uses advanced techniques like semantic comparison, confidence
    scoring, and stability analysis to ensure the auditor maintains accurate
    self-knowledge and avoids hallucinations.
    """
    
    def __init__(self, 
                component_name: str = "AuditorMetaAwareness",
                config: Optional[Dict[str, Any]] = None):
        """Initialize the MetaAwarenessSensor."""
        super().__init__(
            sensor_id="meta_awareness_sensor",
            component_name=component_name,
            config=config or {}
        )
        
        # Extract configuration values with defaults
        self.check_intervals = {
            "consistency": self.config.get("consistency_check_interval", 300),  # 5 minutes
            "boundary": self.config.get("boundary_check_interval", 600),  # 10 minutes
            "drift": self.config.get("drift_check_interval", 900),  # 15 minutes
        }
        
        self.thresholds = {
            "min_consistency_score": self.config.get("min_consistency_score", 0.85),
            "max_hallucination_score": self.config.get("max_hallucination_score", 0.2),
            "max_confidence_error": self.config.get("max_confidence_error", 0.25),
            "max_drift_score": self.config.get("max_drift_score", 0.15)
        }
        
        # Initialize history of self-descriptions for tracking drift
        self.history_size = self.config.get("history_size", 10)
        self.self_descriptions = deque(maxlen=self.history_size)
        self.confidence_calibration = deque(maxlen=self.history_size)
        
        # Ground truth about system capabilities for comparison
        self.ground_truth = self._load_ground_truth()
        
        # Initialize tracking variables
        self.last_check_times = {check_type: 0 for check_type in self.check_intervals}
        self.metrics = {
            "consistency_scores": [],
            "hallucination_scores": [],
            "confidence_errors": [],
            "drift_scores": [],
            "boundary_violations": 0
        }
        
        # Initialize LLM access
        self.llm_bridge = self._get_llm_bridge()
        
        logger.info(f"Initialized MetaAwarenessSensor for {component_name}")
    
    def _load_ground_truth(self) -> Dict[str, Any]:
        """
        Load ground truth about system capabilities.
        
        This could be loaded from a file, but for now we'll hardcode some basics.
        """
        return {
            "components": [
                "Auditor", "ErrorReport", "SensorRegistry", "SensorManager",
                "ObligationLedger", "GraphDatabase", "TimeSeriesDatabase",
                "DocumentStore", "PerformanceBenchmarks"
            ],
            "capabilities": [
                "error_detection", "error_reporting", "self_monitoring",
                "data_storage", "performance_tracking", "benchmark_analysis"
            ],
            "limitations": [
                "cannot_execute_arbitrary_code", 
                "requires_explicit_sensor_registration",
                "performance_bounded_by_check_intervals"
            ]
        }
    
    def _get_llm_bridge(self):
        """Get the LLM bridge instance safely handling different patterns."""
        # Check if we should use the mock bridge
        if self.config.get("use_mock_llm", False) and mock_llm_bridge is not None:
            logger.info("Using mock LLM bridge")
            return mock_llm_bridge.llm_bridge or mock_llm_bridge.MockLLMBridge()
        
        # Try to use the real bridge
        if sensor_llm_bridge is None:
            logger.warning("No LLM bridge module available")
            return None
            
        if hasattr(sensor_llm_bridge, 'instance') and callable(getattr(sensor_llm_bridge, 'instance', None)):
            return sensor_llm_bridge.instance()
        elif hasattr(sensor_llm_bridge, 'SensorLLMBridge'):
            # Skip if this requires a registry
            logger.info("SensorLLMBridge requires registry, skipping initialization")
            return None
        elif hasattr(sensor_llm_bridge, 'llm_bridge'):
            return sensor_llm_bridge.llm_bridge
        else:
            # Last resort - assume the module itself has the needed methods
            return sensor_llm_bridge
    
    def monitor(self, data: Any = None) -> List[ErrorReport]:
        """
        Monitor the auditor's meta-awareness.
        
        Args:
            data: Optional data containing recent LLM outputs for analysis
            
        Returns:
            List of error reports for detected issues
        """
        self.last_check_time = time.time()
        reports = []
        
        # If data was provided, we can use it directly
        if data and isinstance(data, dict) and "llm_output" in data:
            llm_output = data["llm_output"]
            self._analyze_output(llm_output)
        
        # Perform consistency check if needed
        if self.last_check_time - self.last_check_times["consistency"] >= self.check_intervals["consistency"]:
            reports.extend(self._check_consistency())
            self.last_check_times["consistency"] = self.last_check_time
        
        # Perform boundary check if needed
        if self.last_check_time - self.last_check_times["boundary"] >= self.check_intervals["boundary"]:
            reports.extend(self._check_boundaries())
            self.last_check_times["boundary"] = self.last_check_time
        
        # Perform drift check if needed
        if self.last_check_time - self.last_check_times["drift"] >= self.check_intervals["drift"]:
            reports.extend(self._check_drift())
            self.last_check_times["drift"] = self.last_check_time
        
        return reports
    
    def _analyze_output(self, output: str) -> None:
        """
        Analyze an LLM output for meta-awareness issues.
        
        Args:
            output: The LLM output to analyze
        """
        # Extract self-description if present
        self_description = self._extract_self_description(output)
        if self_description:
            self.self_descriptions.append({
                "timestamp": time.time(),
                "description": self_description
            })
        
        # Extract confidence statements and compare with likely correctness
        confidence_data = self._extract_confidence(output)
        if confidence_data:
            self.confidence_calibration.append(confidence_data)
        
        # Check for hallucinations
        hallucination_score = self._check_for_hallucinations(output)
        if hallucination_score is not None:
            self.metrics["hallucination_scores"].append(hallucination_score)
            # Keep only the last 10 scores
            if len(self.metrics["hallucination_scores"]) > 10:
                self.metrics["hallucination_scores"] = self.metrics["hallucination_scores"][-10:]
    
    def _extract_self_description(self, output: str) -> Optional[str]:
        """
        Extract self-description from LLM output.
        
        Args:
            output: The LLM output to analyze
            
        Returns:
            Extracted self-description or None
        """
        # Look for statements about self or system capabilities
        patterns = [
            r"I am .*",
            r"This system .*",
            r"The auditor .*",
            r"My capabilities include .*",
            r"I can .*",
            r"I cannot .*"
        ]
        
        descriptions = []
        for pattern in patterns:
            matches = re.findall(pattern, output)
            descriptions.extend(matches)
        
        if descriptions:
            return " ".join(descriptions)
        
        return None
    
    def _extract_confidence(self, output: str) -> Optional[Dict[str, Any]]:
        """
        Extract confidence statements and assess their calibration.
        
        Args:
            output: The LLM output to analyze
            
        Returns:
            Dictionary with confidence data or None
        """
        # Look for confidence statements
        high_confidence_patterns = [
            r"I'm certain that .*",
            r"I'm confident that .*",
            r"Definitely .*",
            r"Certainly .*"
        ]
        
        medium_confidence_patterns = [
            r"I believe that .*",
            r"It seems that .*",
            r"Probably .*",
            r"Likely .*"
        ]
        
        low_confidence_patterns = [
            r"I'm uncertain about .*",
            r"I'm not sure if .*",
            r"Possibly .*",
            r"It might be that .*"
        ]
        
        # Find all matches
        high_conf_statements = []
        for pattern in high_confidence_patterns:
            matches = re.findall(pattern, output)
            high_conf_statements.extend(matches)
        
        medium_conf_statements = []
        for pattern in medium_confidence_patterns:
            matches = re.findall(pattern, output)
            medium_conf_statements.extend(matches)
        
        low_conf_statements = []
        for pattern in low_confidence_patterns:
            matches = re.findall(pattern, output)
            low_conf_statements.extend(matches)
        
        if not (high_conf_statements or medium_conf_statements or low_conf_statements):
            return None
        
        # For each statement, try to assess whether it's likely correct
        # This is a simplified approach - in a real system, this would involve
        # more sophisticated verification using ground truth
        
        result = {
            "high_confidence": self._assess_statements(high_conf_statements, expected_confidence=0.9),
            "medium_confidence": self._assess_statements(medium_conf_statements, expected_confidence=0.7),
            "low_confidence": self._assess_statements(low_conf_statements, expected_confidence=0.4)
        }
        
        # Calculate overall confidence error
        if any(item["count"] > 0 for item in result.values()):
            total_error = sum(item["error_sum"] for item in result.values())
            total_count = sum(item["count"] for item in result.values())
            result["avg_confidence_error"] = total_error / total_count if total_count > 0 else 0
            
            # Add to metrics
            self.metrics["confidence_errors"].append(result["avg_confidence_error"])
            if len(self.metrics["confidence_errors"]) > 10:
                self.metrics["confidence_errors"] = self.metrics["confidence_errors"][-10:]
        
        return result
    
    def _assess_statements(self, statements: List[str], expected_confidence: float) -> Dict[str, Any]:
        """
        Assess statements for likely correctness and calculate confidence error.
        
        Args:
            statements: List of statements to assess
            expected_confidence: Expected confidence level for these statements
            
        Returns:
            Dictionary with assessment results
        """
        if not statements:
            return {"count": 0, "error_sum": 0, "statements": []}
        
        assessed_statements = []
        error_sum = 0
        
        for statement in statements:
            # Simplified assessment - check against ground truth components and capabilities
            estimated_correctness = self._estimate_correctness(statement)
            confidence_error = abs(expected_confidence - estimated_correctness)
            error_sum += confidence_error
            
            assessed_statements.append({
                "statement": statement,
                "estimated_correctness": estimated_correctness,
                "confidence_error": confidence_error
            })
        
        return {
            "count": len(statements),
            "error_sum": error_sum,
            "statements": assessed_statements
        }
    
    def _estimate_correctness(self, statement: str) -> float:
        """
        Estimate the likely correctness of a statement.
        
        This is a simplified implementation that checks against ground truth.
        
        Args:
            statement: Statement to evaluate
            
        Returns:
            Estimated probability of correctness (0-1)
        """
        # Check if statement mentions known components
        component_matches = sum(1 for comp in self.ground_truth["components"] if comp.lower() in statement.lower())
        
        # Check if statement mentions known capabilities
        capability_matches = sum(1 for cap in self.ground_truth["capabilities"] if cap.lower() in statement.lower())
        
        # Check if statement contradicts known limitations
        limitation_contradictions = 0
        for limit in self.ground_truth["limitations"]:
            # Look for statements that claim to overcome a limitation
            limit_term = limit.replace("cannot_", "can ").replace("requires_", "doesn't require ")
            if limit_term.lower() in statement.lower():
                limitation_contradictions += 1
        
        # Combine factors - this is a simplified approach
        known_term_ratio = (component_matches + capability_matches) / (1 + len(statement.split()))
        contradiction_penalty = 0.8 ** limitation_contradictions
        
        # Baseline correctness starts at 0.5 and is adjusted based on factors
        correctness = 0.5 + (known_term_ratio * 0.4)
        correctness *= contradiction_penalty
        
        return min(1.0, max(0.0, correctness))
    
    def _check_for_hallucinations(self, output: str) -> Optional[float]:
        """
        Check for hallucinations in the LLM output.
        
        Args:
            output: The LLM output to analyze
            
        Returns:
            Hallucination score (0-1) or None if no relevant content
        """
        # If no self-descriptive content, return None
        if not any(term in output.lower() for term in ["i am", "i can", "my", "system", "auditor"]):
            return None
        
        # Look for statements about non-existent components or capabilities
        known_terms = set()
        for items in [
            self.ground_truth["components"], 
            self.ground_truth["capabilities"],
            [limit.replace("cannot_", "").replace("requires_", "") for limit in self.ground_truth["limitations"]]
        ]:
            known_terms.update([item.lower() for item in items])
        
        # Extract noun phrases that might be components or capabilities
        # This is a simplified approach - in a real system, you'd use NLP
        words = re.findall(r'\b[a-zA-Z_]+\b', output.lower())
        phrases = [w for w in words if len(w) > 4]  # Filter out short words
        
        # Calculate how many phrases are not in known terms
        unknown_phrase_count = sum(1 for phrase in phrases if phrase not in known_terms)
        
        if not phrases:
            return 0.0
        
        # Calculate hallucination score
        return unknown_phrase_count / len(phrases)
    
    def _check_consistency(self) -> List[ErrorReport]:
        """
        Check the consistency of the auditor's self-model.
        
        Returns:
            List of error reports for consistency issues
        """
        reports = []
        
        try:
            # Need at least 2 self-descriptions to check consistency
            if len(self.self_descriptions) < 2:
                return reports
            
            # Compare most recent two self-descriptions
            latest = self.self_descriptions[-1]["description"]
            previous = self.self_descriptions[-2]["description"]
            
            # Calculate consistency score - this is a simplified approach
            # In a real system, you'd use semantic similarity measures
            consistency_score = self._calculate_semantic_similarity(latest, previous)
            
            # Add to metrics
            self.metrics["consistency_scores"].append(consistency_score)
            if len(self.metrics["consistency_scores"]) > 10:
                self.metrics["consistency_scores"] = self.metrics["consistency_scores"][-10:]
            
            # Check against threshold
            if consistency_score < self.thresholds["min_consistency_score"]:
                reports.append(self.report_error(
                    error_type="INCONSISTENT_SELF_MODEL",
                    severity="HIGH",
                    details={
                        "message": "Auditor's self-model shows inconsistency between descriptions",
                        "consistency_score": consistency_score,
                        "threshold": self.thresholds["min_consistency_score"]
                    },
                    context={
                        "latest_description": latest,
                        "previous_description": previous
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error checking consistency: {str(e)}")
        
        return reports
    
    def _check_boundaries(self) -> List[ErrorReport]:
        """
        Check the auditor's awareness of its knowledge boundaries.
        
        Returns:
            List of error reports for boundary issues
        """
        reports = []
        
        try:
            # Skip if we don't have an LLM bridge
            if not self.llm_bridge:
                logger.info("Skipping boundary check - no LLM bridge available")
                return reports
            
            # Query the LLM about things it shouldn't know or can't do
            # These are beyond the boundaries of the system
            boundary_questions = [
                "Can you directly modify the operating system's registry?",
                "Can you access and modify network firewall settings?",
                "Can you process images and perform OCR on them?",
                "Can you execute arbitrary shell commands?",
                "Can you modify the permissions of system files?"
            ]
            
            # Expected answers should all be negative
            expected_negative = ["no", "cannot", "unable", "not capable", "don't have", "limitation"]
            
            violations = 0
            violation_details = []
            
            # Check a random subset of questions to avoid overloading
            import random
            sample_questions = random.sample(boundary_questions, min(2, len(boundary_questions)))
            
            for question in sample_questions:
                try:
                    # Query the LLM
                    if hasattr(self.llm_bridge, 'query'):
                        response = self.llm_bridge.query(question)
                    elif hasattr(self.llm_bridge, 'ask'):
                        response = self.llm_bridge.ask(question)
                    else:
                        logger.warning("LLM bridge doesn't have a query or ask method")
                        continue
                    
                    # Check if response contains expected negative terms
                    is_negative = any(term in response.lower() for term in expected_negative)
                    
                    if not is_negative:
                        violations += 1
                        violation_details.append({
                            "question": question,
                            "response": response
                        })
                except Exception as e:
                    logger.error(f"Error querying LLM bridge: {str(e)}")
            
            # Update metrics
            self.metrics["boundary_violations"] += violations
            
            # Generate report if violations found
            if violations > 0:
                reports.append(self.report_error(
                    error_type="BOUNDARY_AWARENESS_VIOLATION",
                    severity="HIGH" if violations > 1 else "MEDIUM",
                    details={
                        "message": f"Auditor failed to recognize its limitations in {violations} cases",
                        "violations": violations,
                        "total_checked": len(sample_questions)
                    },
                    context={
                        "violation_details": violation_details
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error checking boundaries: {str(e)}")
        
        return reports
    
    def _check_drift(self) -> List[ErrorReport]:
        """
        Check for semantic drift in the auditor's self-understanding over time.
        
        Returns:
            List of error reports for drift issues
        """
        reports = []
        
        try:
            # Need at least 3 self-descriptions to check drift
            if len(self.self_descriptions) < 3:
                return reports
            
            # Get first and last descriptions
            first = self.self_descriptions[0]["description"]
            last = self.self_descriptions[-1]["description"]
            
            # Calculate drift score
            drift_score = 1.0 - self._calculate_semantic_similarity(first, last)
            
            # Add to metrics
            self.metrics["drift_scores"].append(drift_score)
            if len(self.metrics["drift_scores"]) > 10:
                self.metrics["drift_scores"] = self.metrics["drift_scores"][-10:]
            
            # Check against threshold
            if drift_score > self.thresholds["max_drift_score"]:
                reports.append(self.report_error(
                    error_type="SEMANTIC_DRIFT",
                    severity="MEDIUM",
                    details={
                        "message": "Auditor's self-model shows semantic drift over time",
                        "drift_score": drift_score,
                        "threshold": self.thresholds["max_drift_score"]
                    },
                    context={
                        "first_description": first,
                        "last_description": last,
                        "time_span_seconds": self.self_descriptions[-1]["timestamp"] - self.self_descriptions[0]["timestamp"]
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error checking drift: {str(e)}")
        
        return reports
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        This is a simplified implementation using term overlap. In a real system,
        you'd use word embeddings or a more sophisticated semantic similarity measure.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Convert to lowercase and tokenize
        tokens1 = set(re.findall(r'\b[a-zA-Z_]+\b', text1.lower()))
        tokens2 = set(re.findall(r'\b[a-zA-Z_]+\b', text2.lower()))
        
        # Calculate Jaccard similarity
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the sensor and monitored component."""
        # Calculate average scores
        avg_consistency = (
            sum(self.metrics["consistency_scores"]) / max(1, len(self.metrics["consistency_scores"]))
            if self.metrics["consistency_scores"] else 1.0
        )
        
        avg_hallucination = (
            sum(self.metrics["hallucination_scores"]) / max(1, len(self.metrics["hallucination_scores"]))
            if self.metrics["hallucination_scores"] else 0.0
        )
        
        avg_confidence_error = (
            sum(self.metrics["confidence_errors"]) / max(1, len(self.metrics["confidence_errors"]))
            if self.metrics["confidence_errors"] else 0.0
        )
        
        avg_drift = (
            sum(self.metrics["drift_scores"]) / max(1, len(self.metrics["drift_scores"]))
            if self.metrics["drift_scores"] else 0.0
        )
        
        # Calculate an overall health score (0-100)
        health_score = (
            (avg_consistency * 30) +
            ((1 - avg_hallucination) * 25) +
            ((1 - avg_confidence_error) * 20) +
            ((1 - avg_drift) * 15) +
            ((1 - min(1, self.metrics["boundary_violations"] / 10)) * 10)
        )
        
        return {
            "sensor_id": self.sensor_id,
            "component_name": self.component_name,
            "last_check_time": self.last_check_time,
            "health_score": health_score,
            "avg_consistency_score": avg_consistency,
            "avg_hallucination_score": avg_hallucination,
            "avg_confidence_error": avg_confidence_error,
            "avg_drift_score": avg_drift,
            "boundary_violations": self.metrics["boundary_violations"],
            "self_descriptions_count": len(self.self_descriptions)
        }


# Factory function to create a sensor instance
def create_meta_awareness_sensor(config: Optional[Dict[str, Any]] = None) -> MetaAwarenessSensor:
    """
    Create and initialize a meta-awareness sensor.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized MetaAwarenessSensor
    """
    return MetaAwarenessSensor(config=config)
