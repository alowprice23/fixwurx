"""
FixWurx Auditor Proof Metrics Sensor

This module implements a sensor for monitoring and analyzing proof metrics
of the auditor system, including coverage, consistency, and completeness metrics.
"""

import logging
import time
import math
import random
import os
import json
import hashlib
from typing import Dict, List, Set, Any, Optional, Union, Tuple

from sensor_base import ErrorSensor
from error_report import ErrorReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [ProofMetrics] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('proof_metrics_sensor')


class ProofMetricsSensor(ErrorSensor):
    """
    Monitors and evaluates formal proof metrics for the auditor system.
    
    This sensor tracks coverage, completeness, consistency, and soundness of 
    verification proofs in the system. It ensures that bug fixes and system changes
    are properly verified and that the proof system maintains its integrity.
    """
    
    def __init__(self, 
                component_name: str = "VerificationProofs",
                config: Optional[Dict[str, Any]] = None):
        """Initialize the ProofMetricsSensor."""
        super().__init__(
            sensor_id="proof_metrics_sensor",
            component_name=component_name,
            config=config or {}
        )
        
        # Extract configuration values with defaults
        self.check_intervals = {
            "coverage": self.config.get("coverage_check_interval", 300),  # 5 minutes
            "consistency": self.config.get("consistency_check_interval", 600),  # 10 minutes
            "soundness": self.config.get("soundness_check_interval", 900),  # 15 minutes
        }
        
        self.thresholds = {
            "min_proof_coverage": self.config.get("min_proof_coverage", 0.75),
            "min_critical_coverage": self.config.get("min_critical_coverage", 0.95),
            "min_consistency_score": self.config.get("min_consistency_score", 0.9),
            "min_soundness_score": self.config.get("min_soundness_score", 0.95),
            "max_unproven_changes": self.config.get("max_unproven_changes", 5)
        }
        
        # Path settings
        self.proofs_path = self.config.get("proofs_path", "auditor_data/proofs")
        self.code_path = self.config.get("code_path", ".")
        
        # Ensure proofs directory exists
        os.makedirs(self.proofs_path, exist_ok=True)
        
        # Initialize metrics and tracking
        self.last_check_times = {check_type: 0 for check_type in self.check_intervals}
        self.metrics = {
            "coverage": {
                "total": 0.0,
                "by_component": {},
                "critical_paths": 0.0
            },
            "consistency": {
                "score": 1.0,
                "conflicts": []
            },
            "soundness": {
                "score": 1.0,
                "issues": []
            },
            "unproven_changes": [],
            "proof_history": []
        }
        
        # Critical paths that require high coverage
        self.critical_paths = self.config.get("critical_paths", [
            "error_detection",
            "security_verification",
            "resource_allocation",
            "data_integrity",
            "consistency_checks"
        ])
        
        # Register known proofs
        self.known_proofs = self._load_known_proofs()
        self.component_hashes = {}
        
        logger.info(f"Initialized ProofMetricsSensor for {component_name}")
    
    def monitor(self, data: Any = None) -> List[ErrorReport]:
        """
        Monitor proof metrics.
        
        Args:
            data: Optional data for monitoring, such as recent code changes
            
        Returns:
            List of error reports for detected issues
        """
        self.last_check_time = time.time()
        reports = []
        
        # If data was provided with changes, record them
        if data and isinstance(data, dict) and "changes" in data:
            self._record_changes(data["changes"])
        
        # Perform coverage check if needed
        if self.last_check_time - self.last_check_times["coverage"] >= self.check_intervals["coverage"]:
            coverage_reports = self._check_coverage()
            if coverage_reports:
                reports.extend(coverage_reports)
            self.last_check_times["coverage"] = self.last_check_time
        
        # Perform consistency check if needed
        if self.last_check_time - self.last_check_times["consistency"] >= self.check_intervals["consistency"]:
            consistency_reports = self._check_consistency()
            if consistency_reports:
                reports.extend(consistency_reports)
            self.last_check_times["consistency"] = self.last_check_time
        
        # Perform soundness check if needed
        if self.last_check_time - self.last_check_times["soundness"] >= self.check_intervals["soundness"]:
            soundness_reports = self._check_soundness()
            if soundness_reports:
                reports.extend(soundness_reports)
            self.last_check_times["soundness"] = self.last_check_time
        
        return reports
    
    def _load_known_proofs(self) -> Dict[str, Any]:
        """
        Load known proofs from the proofs directory.
        
        Returns:
            Dictionary of known proofs
        """
        known_proofs = {}
        try:
            # Get all proof files
            if os.path.exists(self.proofs_path):
                for filename in os.listdir(self.proofs_path):
                    if filename.endswith('.json'):
                        filepath = os.path.join(self.proofs_path, filename)
                        try:
                            with open(filepath, 'r') as f:
                                proof_data = json.load(f)
                                if "proof_id" in proof_data:
                                    known_proofs[proof_data["proof_id"]] = proof_data
                        except Exception as e:
                            logger.error(f"Error loading proof file {filepath}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading known proofs: {str(e)}")
            
            # Fallback to demo data if loading fails
            return self._generate_demo_proofs()
        
        # If no proofs found, use demo data
        if not known_proofs:
            known_proofs = self._generate_demo_proofs()
        
        logger.info(f"Loaded {len(known_proofs)} known proofs")
        return known_proofs
    
    def _generate_demo_proofs(self) -> Dict[str, Any]:
        """
        Generate demo proof data for testing.
        
        Returns:
            Dictionary of demo proofs
        """
        demo_proofs = {}
        
        # Generate some demo proofs for common components
        components = [
            "error_detection", 
            "security_verification",
            "resource_allocation",
            "data_integrity",
            "consistency_checks",
            "memory_management",
            "thread_safety"
        ]
        
        for component in components:
            # Create a proof for this component
            proof_id = f"proof_{component}_{int(time.time())}"
            
            proof_data = {
                "proof_id": proof_id,
                "component": component,
                "created_at": time.time(),
                "version": "1.0",
                "coverage": random.uniform(0.7, 0.98),
                "verified_by": "automated_verifier",
                "properties": [
                    {
                        "property_id": f"{component}_safety",
                        "description": f"Safety property for {component}",
                        "verified": True,
                        "proof_method": "model_checking"
                    },
                    {
                        "property_id": f"{component}_liveness",
                        "description": f"Liveness property for {component}",
                        "verified": random.choice([True, True, False]),  # 2/3 chance of being verified
                        "proof_method": "theorem_proving"
                    }
                ],
                "dependencies": [],
                "hash": hashlib.md5(component.encode()).hexdigest()
            }
            
            demo_proofs[proof_id] = proof_data
        
        # Add dependencies between proofs
        for proof_id, proof_data in demo_proofs.items():
            # Add 1-3 random dependencies
            for _ in range(random.randint(0, 2)):
                dependent_id = random.choice(list(demo_proofs.keys()))
                if dependent_id != proof_id and dependent_id not in proof_data["dependencies"]:
                    proof_data["dependencies"].append(dependent_id)
        
        return demo_proofs
    
    def _record_changes(self, changes: Dict[str, Any]) -> None:
        """
        Record code changes for tracking unproven changes.
        
        Args:
            changes: Dictionary containing code changes
        """
        try:
            current_time = time.time()
            
            for component, change_data in changes.items():
                # Add to unproven changes if not already present
                if not any(c["component"] == component for c in self.metrics["unproven_changes"]):
                    self.metrics["unproven_changes"].append({
                        "component": component,
                        "timestamp": current_time,
                        "description": change_data.get("description", "No description"),
                        "lines_changed": change_data.get("lines_changed", 0),
                        "criticality": change_data.get("criticality", "MEDIUM")
                    })
            
            # Sort unproven changes by timestamp (newest first)
            self.metrics["unproven_changes"].sort(key=lambda x: x["timestamp"], reverse=True)
            
            logger.info(f"Recorded {len(changes)} code changes")
        
        except Exception as e:
            logger.error(f"Error recording changes: {str(e)}")
    
    def _get_component_hash(self, component: str) -> str:
        """
        Calculate a hash of the component's code to detect changes.
        
        In a real implementation, this would analyze the actual code.
        For this demo, we'll use a simulated hash.
        
        Args:
            component: Component name
            
        Returns:
            Hash string representing the component's current state
        """
        try:
            # For a real implementation, we would analyze the actual code
            # For this demo, we'll use a simulated hash that occasionally changes
            if component not in self.component_hashes or random.random() < 0.05:
                self.component_hashes[component] = hashlib.md5(
                    f"{component}_{time.time()}_{random.randint(1000, 9999)}".encode()
                ).hexdigest()
            
            return self.component_hashes[component]
            
        except Exception as e:
            logger.error(f"Error calculating component hash for {component}: {str(e)}")
            return hashlib.md5(f"{component}_{time.time()}".encode()).hexdigest()
    
    def _check_coverage(self) -> List[ErrorReport]:
        """
        Check proof coverage metrics.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # Calculate coverage metrics
            coverage_by_component = {}
            critical_coverage = 0.0
            critical_count = 0
            
            for proof_id, proof_data in self.known_proofs.items():
                component = proof_data.get("component", "unknown")
                coverage = proof_data.get("coverage", 0.0)
                
                # Update component coverage
                if component not in coverage_by_component:
                    coverage_by_component[component] = []
                coverage_by_component[component].append(coverage)
                
                # Check if this is a critical component
                if any(critical in component for critical in self.critical_paths):
                    critical_coverage += coverage
                    critical_count += 1
            
            # Calculate average coverage by component
            avg_coverage_by_component = {}
            for component, coverages in coverage_by_component.items():
                avg_coverage_by_component[component] = sum(coverages) / len(coverages)
            
            # Calculate overall average coverage
            total_coverage = 0.0
            if avg_coverage_by_component:
                total_coverage = sum(avg_coverage_by_component.values()) / len(avg_coverage_by_component)
            
            # Calculate critical paths coverage
            avg_critical_coverage = 0.0
            if critical_count > 0:
                avg_critical_coverage = critical_coverage / critical_count
            
            # Update metrics
            self.metrics["coverage"]["total"] = total_coverage
            self.metrics["coverage"]["by_component"] = avg_coverage_by_component
            self.metrics["coverage"]["critical_paths"] = avg_critical_coverage
            
            # Check overall coverage
            if total_coverage < self.thresholds["min_proof_coverage"]:
                reports.append(self.report_error(
                    error_type="LOW_PROOF_COVERAGE",
                    severity="MEDIUM",
                    details={
                        "message": f"Overall proof coverage is below threshold: {total_coverage:.1%} < {self.thresholds['min_proof_coverage']:.1%}",
                        "current_coverage": total_coverage,
                        "threshold": self.thresholds["min_proof_coverage"]
                    },
                    context={
                        "coverage_by_component": avg_coverage_by_component,
                        "suggestion": "Increase formal verification coverage for components with low coverage"
                    }
                ))
            
            # Check critical paths coverage
            if avg_critical_coverage < self.thresholds["min_critical_coverage"]:
                reports.append(self.report_error(
                    error_type="LOW_CRITICAL_PATH_COVERAGE",
                    severity="HIGH",
                    details={
                        "message": f"Critical path proof coverage is below threshold: {avg_critical_coverage:.1%} < {self.thresholds['min_critical_coverage']:.1%}",
                        "current_coverage": avg_critical_coverage,
                        "threshold": self.thresholds["min_critical_coverage"]
                    },
                    context={
                        "critical_paths": self.critical_paths,
                        "suggestion": "Prioritize verification of critical path components"
                    }
                ))
            
            # Check for components with very low coverage
            low_coverage_components = []
            for component, coverage in avg_coverage_by_component.items():
                if coverage < self.thresholds["min_proof_coverage"] * 0.7:  # 70% of the minimum threshold
                    low_coverage_components.append((component, coverage))
            
            if low_coverage_components:
                reports.append(self.report_error(
                    error_type="COMPONENT_PROOF_COVERAGE_CRITICAL",
                    severity="HIGH" if any(component[0] in self.critical_paths for component in low_coverage_components) else "MEDIUM",
                    details={
                        "message": f"{len(low_coverage_components)} components have critically low proof coverage",
                        "components": [c[0] for c in low_coverage_components],
                        "threshold": self.thresholds["min_proof_coverage"] * 0.7
                    },
                    context={
                        "component_details": dict(low_coverage_components),
                        "suggestion": "These components need immediate attention for verification"
                    }
                ))
            
            # Check for too many unproven changes
            if len(self.metrics["unproven_changes"]) > self.thresholds["max_unproven_changes"]:
                reports.append(self.report_error(
                    error_type="EXCESSIVE_UNPROVEN_CHANGES",
                    severity="MEDIUM",
                    details={
                        "message": f"Too many unproven code changes: {len(self.metrics['unproven_changes'])} > {self.thresholds['max_unproven_changes']}",
                        "unproven_count": len(self.metrics["unproven_changes"]),
                        "threshold": self.thresholds["max_unproven_changes"]
                    },
                    context={
                        "recent_changes": self.metrics["unproven_changes"][:5],
                        "suggestion": "Verify recent code changes before making additional changes"
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error in coverage check: {str(e)}")
        
        return reports
    
    def _check_consistency(self) -> List[ErrorReport]:
        """
        Check proof consistency metrics.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # In a real implementation, this would perform sophisticated checks
            # to ensure that proofs are consistent with each other.
            # For this demo, we'll simulate some consistency checks.
            
            # Check for dependency conflicts
            dependency_conflicts = []
            
            for proof_id, proof_data in self.known_proofs.items():
                component = proof_data.get("component", "unknown")
                dependencies = proof_data.get("dependencies", [])
                
                # Check if any dependency has been updated more recently than this proof
                for dep_id in dependencies:
                    if dep_id in self.known_proofs:
                        dep_data = self.known_proofs[dep_id]
                        if dep_data.get("created_at", 0) > proof_data.get("created_at", 0):
                            dependency_conflicts.append({
                                "proof_id": proof_id,
                                "component": component,
                                "dependency_id": dep_id,
                                "dependency_component": dep_data.get("component", "unknown"),
                                "time_difference": dep_data.get("created_at", 0) - proof_data.get("created_at", 0)
                            })
            
            # Check for property conflicts
            property_conflicts = []
            property_map = {}
            
            for proof_id, proof_data in self.known_proofs.items():
                properties = proof_data.get("properties", [])
                
                for prop in properties:
                    prop_id = prop.get("property_id", "")
                    if prop_id:
                        if prop_id in property_map:
                            # Check if the verification status conflicts
                            if prop.get("verified", False) != property_map[prop_id].get("verified", False):
                                property_conflicts.append({
                                    "property_id": prop_id,
                                    "proof1": proof_id,
                                    "proof2": property_map[prop_id]["proof_id"],
                                    "status1": prop.get("verified", False),
                                    "status2": property_map[prop_id].get("verified", False)
                                })
                        else:
                            property_map[prop_id] = {
                                "proof_id": proof_id,
                                "verified": prop.get("verified", False)
                            }
            
            # Calculate consistency score
            total_proofs = len(self.known_proofs)
            total_dependencies = sum(len(p.get("dependencies", [])) for p in self.known_proofs.values())
            total_properties = sum(len(p.get("properties", [])) for p in self.known_proofs.values())
            
            if total_dependencies > 0 and total_properties > 0:
                dependency_score = 1.0 - (len(dependency_conflicts) / total_dependencies)
                property_score = 1.0 - (len(property_conflicts) / total_properties)
                consistency_score = (dependency_score + property_score) / 2
            else:
                consistency_score = 1.0  # No dependencies or properties means no conflicts
            
            # Update metrics
            self.metrics["consistency"]["score"] = consistency_score
            self.metrics["consistency"]["conflicts"] = dependency_conflicts + property_conflicts
            
            # Check if consistency score is below threshold
            if consistency_score < self.thresholds["min_consistency_score"]:
                reports.append(self.report_error(
                    error_type="LOW_PROOF_CONSISTENCY",
                    severity="HIGH" if consistency_score < 0.7 else "MEDIUM",
                    details={
                        "message": f"Proof consistency score is below threshold: {consistency_score:.2f} < {self.thresholds['min_consistency_score']}",
                        "consistency_score": consistency_score,
                        "threshold": self.thresholds["min_consistency_score"],
                        "dependency_conflicts": len(dependency_conflicts),
                        "property_conflicts": len(property_conflicts)
                    },
                    context={
                        "dependency_conflict_examples": dependency_conflicts[:3],
                        "property_conflict_examples": property_conflicts[:3],
                        "suggestion": "Resolve conflicting proofs and ensure proofs are updated when dependencies change"
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error in consistency check: {str(e)}")
        
        return reports
    
    def _check_soundness(self) -> List[ErrorReport]:
        """
        Check proof soundness metrics.
        
        Returns:
            List of error reports for detected issues
        """
        reports = []
        
        try:
            # In a real implementation, this would analyze the actual proofs
            # to ensure they are sound (i.e., valid and correct).
            # For this demo, we'll simulate some soundness checks.
            
            # Check for components with changed code but unchanged proofs
            code_hash_mismatches = []
            
            for proof_id, proof_data in self.known_proofs.items():
                component = proof_data.get("component", "unknown")
                stored_hash = proof_data.get("hash", "")
                
                # Calculate current hash
                current_hash = self._get_component_hash(component)
                
                # Compare hashes
                if stored_hash and current_hash != stored_hash:
                    code_hash_mismatches.append({
                        "proof_id": proof_id,
                        "component": component,
                        "stored_hash": stored_hash,
                        "current_hash": current_hash
                    })
            
            # Check for failed properties
            failed_properties = []
            
            for proof_id, proof_data in self.known_proofs.items():
                component = proof_data.get("component", "unknown")
                properties = proof_data.get("properties", [])
                
                for prop in properties:
                    if not prop.get("verified", True):
                        failed_properties.append({
                            "proof_id": proof_id,
                            "component": component,
                            "property_id": prop.get("property_id", "unknown"),
                            "description": prop.get("description", "No description")
                        })
            
            # Calculate soundness score
            total_proofs = len(self.known_proofs)
            total_properties = sum(len(p.get("properties", [])) for p in self.known_proofs.values())
            
            if total_proofs > 0 and total_properties > 0:
                hash_score = 1.0 - (len(code_hash_mismatches) / total_proofs)
                property_score = 1.0 - (len(failed_properties) / total_properties)
                soundness_score = (hash_score * 0.7) + (property_score * 0.3)  # Hash mismatches are more critical
            else:
                soundness_score = 1.0  # No proofs means no soundness issues
            
            # Update metrics
            self.metrics["soundness"]["score"] = soundness_score
            self.metrics["soundness"]["issues"] = {
                "hash_mismatches": code_hash_mismatches,
                "failed_properties": failed_properties
            }
            
            # Check if soundness score is below threshold
            if soundness_score < self.thresholds["min_soundness_score"]:
                reports.append(self.report_error(
                    error_type="LOW_PROOF_SOUNDNESS",
                    severity="HIGH",
                    details={
                        "message": f"Proof soundness score is below threshold: {soundness_score:.2f} < {self.thresholds['min_soundness_score']}",
                        "soundness_score": soundness_score,
                        "threshold": self.thresholds["min_soundness_score"],
                        "hash_mismatches": len(code_hash_mismatches),
                        "failed_properties": len(failed_properties)
                    },
                    context={
                        "hash_mismatch_examples": code_hash_mismatches[:3],
                        "failed_property_examples": failed_properties[:3],
                        "suggestion": "Update proofs for changed components and fix failed properties"
                    }
                ))
            
            # Check for critical components with hash mismatches
            critical_hash_mismatches = [m for m in code_hash_mismatches 
                                       if any(critical in m["component"] for critical in self.critical_paths)]
            
            if critical_hash_mismatches:
                reports.append(self.report_error(
                    error_type="CRITICAL_COMPONENT_PROOF_OUTDATED",
                    severity="HIGH",
                    details={
                        "message": f"{len(critical_hash_mismatches)} critical components have outdated proofs",
                        "components": [m["component"] for m in critical_hash_mismatches],
                    },
                    context={
                        "hash_mismatches": critical_hash_mismatches,
                        "suggestion": "Immediately update proofs for these critical components"
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error in soundness check: {str(e)}")
        
        return reports
    
    def register_proof(self, proof_data: Dict[str, Any]) -> bool:
        """
        Register a new or updated proof.
        
        Args:
            proof_data: Proof data dictionary
            
        Returns:
            True if registered successfully, False otherwise
        """
        try:
            # Validate proof data
            if "proof_id" not in proof_data or "component" not in proof_data:
                logger.error("Invalid proof data: missing proof_id or component")
                return False
            
            # Add creation timestamp if not present
            if "created_at" not in proof_data:
                proof_data["created_at"] = time.time()
            
            # Calculate and add hash if not present
            if "hash" not in proof_data:
                proof_data["hash"] = self._get_component_hash(proof_data["component"])
            
            # Update known proofs
            self.known_proofs[proof_data["proof_id"]] = proof_data
            
            # Save proof to disk
            self._save_proof(proof_data)
            
            # Mark component as proven
            self.metrics["unproven_changes"] = [
                c for c in self.metrics["unproven_changes"]
                if c["component"] != proof_data["component"]
            ]
            
            # Add to proof history
            self.metrics["proof_history"].append({
                "proof_id": proof_data["proof_id"],
                "component": proof_data["component"],
                "timestamp": time.time(),
                "coverage": proof_data.get("coverage", 0.0)
            })
            
            logger.info(f"Registered proof {proof_data['proof_id']} for component {proof_data['component']}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering proof: {str(e)}")
            return False
    
    def _save_proof(self, proof_data: Dict[str, Any]) -> None:
        """
        Save a proof to disk.
        
        Args:
            proof_data: Proof data dictionary
        """
        try:
            proof_id = proof_data["proof_id"]
            filepath = os.path.join(self.proofs_path, f"{proof_id}.json")
            
            with open(filepath, 'w') as f:
                json.dump(proof_data, f, indent=2)
                
            logger.info(f"Saved proof {proof_id} to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving proof: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the sensor and monitored component."""
        # Calculate health score (0-100)
        coverage_score = self.metrics["coverage"]["total"] * 40  # 0-40 points
        critical_coverage_bonus = max(0, self.metrics["coverage"]["critical_paths"] - 0.8) * 50  # Up to 10 bonus points
        consistency_score = self.metrics["consistency"]["score"] * 30  # 0-30 points
        soundness_score = self.metrics["soundness"]["score"] * 30  # 0-30 points
        unproven_penalty = min(15, len(self.metrics["unproven_changes"]) * 3)  # Up to 15 point penalty
        
        health_score = coverage_score + critical_coverage_bonus + consistency_score + soundness_score - unproven_penalty
        health_score = max(0, min(100, health_score))
        
        return {
            "sensor_id": self.sensor_id,
            "component_name": self.component_name,
            "last_check_time": self.last_check_time,
            "health_score": health_score,
            "proof_coverage": {
                "total": self.metrics["coverage"]["total"],
                "critical_paths": self.metrics["coverage"]["critical_paths"]
            },
            "consistency_score": self.metrics["consistency"]["score"],
            "soundness_score": self.metrics["soundness"]["score"],
            "unproven_changes": len(self.metrics["unproven_changes"]),
            "proof_count": len(self.known_proofs)
        }


# Factory function to create a sensor instance
def create_proof_metrics_sensor(config: Optional[Dict[str, Any]] = None) -> ProofMetricsSensor:
    """
    Create and initialize a proof metrics sensor.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized ProofMetricsSensor
    """
    return ProofMetricsSensor(config=config)
