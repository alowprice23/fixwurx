"""
FixWurx Auditor ObligationLedger Sensor

This module implements the sensor for monitoring the ObligationLedger component,
which is responsible for tracking Δ-Closure computation, rule application, and
ensuring obligations are properly managed.
"""

import logging
import datetime
from typing import Dict, List, Set, Any, Optional, Union, Tuple

# Import sensor base class
from sensor_base import ErrorSensor
from error_report import ErrorReport

# Import auditor components for monitoring
from auditor import ObligationLedger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [ObligationLedgerSensor] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('obligation_ledger_sensor')


class ObligationLedgerSensor(ErrorSensor):
    """
    Monitors the ObligationLedger component for errors related to
    Δ-Closure computation, rule application, and missing obligations.
    """
    
    def __init__(self, component_name: str = "ObligationLedger", config: Dict[str, Any] = None):
        """Initialize the ObligationLedger sensor."""
        super().__init__(
            sensor_id="obligation_ledger_sensor",
            component_name=component_name,
            config=config
        )
        # Specific thresholds for this sensor
        self.rule_application_threshold = self._get_threshold("rule_application_threshold", 0.95)
        self.max_missing_obligations = self._get_threshold("max_missing_obligations", 0)
        self.max_circular_dependencies = self._get_threshold("max_circular_dependencies", 0)
        
    def monitor(self, data: Any) -> List[ErrorReport]:
        """
        Monitor ObligationLedger data for errors.
        
        Args:
            data: ObligationLedger instance or data dictionary
            
        Returns:
            List of error reports
        """
        self.last_check_time = datetime.datetime.now()
        reports = []
        
        # Check data type
        if isinstance(data, ObligationLedger):
            ledger = data
            # Extract relevant data
            obligations = ledger.get_all() if hasattr(ledger, 'get_all') else set()
            delta_rules = ledger.delta_rules if hasattr(ledger, 'delta_rules') else []
        elif isinstance(data, dict):
            # Data is already a dictionary
            ledger = None
            obligations = data.get('obligations', set())
            delta_rules = data.get('delta_rules', [])
        else:
            # Unknown data type
            logger.warning(f"Unknown data type for ObligationLedger monitoring: {type(data)}")
            return reports
        
        # Check for empty obligations
        if not obligations:
            reports.append(self.report_error(
                error_type="EMPTY_OBLIGATIONS",
                severity="HIGH",
                details={"message": "No obligations found in ledger"},
                context={"obligations_count": 0}
            ))
        
        # Check for empty delta rules
        if not delta_rules:
            reports.append(self.report_error(
                error_type="EMPTY_DELTA_RULES",
                severity="HIGH",
                details={"message": "No Δ-rules found in ledger"},
                context={"rules_count": 0}
            ))
        
        # If we have both obligations and rules, check for rule application issues
        if obligations and delta_rules:
            reports.extend(self._check_rule_application(obligations, delta_rules))
            reports.extend(self._check_circular_dependencies(delta_rules))
        
        return reports
    
    def _check_rule_application(self, obligations: Set[str], delta_rules: List[Dict]) -> List[ErrorReport]:
        """
        Check for rule application issues.
        
        Args:
            obligations: Set of obligations
            delta_rules: List of delta rules
            
        Returns:
            List of error reports
        """
        reports = []
        
        # Count how many obligations have matching rules
        matched_obligations = 0
        
        for obligation in obligations:
            for rule in delta_rules:
                pattern = rule.get("pattern", "")
                if pattern in obligation:
                    matched_obligations += 1
                    break
        
        # Calculate match ratio
        if obligations:
            match_ratio = matched_obligations / len(obligations)
            
            # Check if below threshold
            if match_ratio < self.rule_application_threshold:
                reports.append(self.report_error(
                    error_type="RULE_APPLICATION_INSUFFICIENT",
                    severity="MEDIUM",
                    details={
                        "message": f"Rule application ratio below threshold",
                        "match_ratio": match_ratio,
                        "threshold": self.rule_application_threshold
                    },
                    context={
                        "total_obligations": len(obligations),
                        "matched_obligations": matched_obligations
                    }
                ))
        
        return reports
    
    def _check_circular_dependencies(self, delta_rules: List[Dict]) -> List[ErrorReport]:
        """
        Check for circular dependencies in delta rules.
        
        Args:
            delta_rules: List of delta rules
            
        Returns:
            List of error reports
        """
        reports = []
        
        # Build dependency graph
        dependencies = {}
        
        for rule in delta_rules:
            pattern = rule.get("pattern", "")
            transforms_to = rule.get("transforms_to", [])
            
            if pattern and transforms_to:
                if pattern not in dependencies:
                    dependencies[pattern] = set()
                
                for transform in transforms_to:
                    dependencies[pattern].add(transform)
        
        # Check for circular dependencies
        circular = self._find_circular_dependencies(dependencies)
        
        if circular and len(circular) > self.max_circular_dependencies:
            reports.append(self.report_error(
                error_type="CIRCULAR_DEPENDENCIES",
                severity="HIGH",
                details={
                    "message": "Circular dependencies detected in Δ-rules",
                    "circular_count": len(circular)
                },
                context={"circular_paths": list(circular)}
            ))
        
        return reports
    
    def _find_circular_dependencies(self, dependencies: Dict[str, Set[str]]) -> Set[Tuple[str, ...]]:
        """
        Find circular dependencies in the dependency graph.
        
        Args:
            dependencies: Dependency graph
            
        Returns:
            Set of circular dependency paths
        """
        circular = set()
        visited = set()
        path = []
        
        def dfs(node):
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                circular.add(tuple(path[cycle_start:]))
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for neighbor in dependencies.get(node, set()):
                dfs(neighbor)
            
            path.pop()
        
        # Run DFS from each node
        for node in dependencies:
            dfs(node)
        
        return circular
