"""
core/rollback_manager.py
────────────────────────
Loss-less, atomic rollback layer.

High-level contract
───────────────────
1. **Every successful patch application** ▫ stores the *reverse diff* in
   `.triangulum/rollback/{bug_id}.patch`.  (Store once; overwrite forbidden.)
2. **rollback_patch(bug_id)** ▫ applies that reverse diff with
   `git apply -R`, restoring the work-tree exactly to the pre-patch SHA.
3. **Atomicity** ▫ two-step algorithm  
      a) `git apply --check -R patch`   → verify clean-apply  
      b) `git apply        -R patch`    → mutate work-tree  
      c) if any step fails → work-tree unchanged, function raises `RollbackError`.

No third-party dependencies—only `subprocess` and `pathlib`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple

# ------------------------------------------------------------------------------
ROLLBACK_DIR = Path(".triangulum") / "rollback"
ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
LOG_FILE = ROLLBACK_DIR / "rollback.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rollback_manager")

REGISTRY_FILE = ROLLBACK_DIR / "registry.json"  # maps bug-id → patch filename
HISTORY_FILE = ROLLBACK_DIR / "history.json"    # rollback history
CONFIG_FILE = ROLLBACK_DIR / "config.json"      # rollback configuration


# ───────────────────────────────────────────────────────────────────────────────
# Enums
# ───────────────────────────────────────────────────────────────────────────────
class RollbackTrigger(Enum):
    """Reasons that can trigger an automatic rollback."""
    TEST_FAILURE = "test_failure"           # Tests started failing after patch
    REGRESSION = "regression"               # Regression detected in related functionality
    PERFORMANCE_DEGRADATION = "performance" # Significant performance degradation
    SECURITY_RISK = "security"              # Security vulnerability introduced
    MANUAL = "manual"                       # Manually triggered rollback
    PLANNER_DECISION = "planner"            # Planner agent decided to rollback
    TIMEOUT = "timeout"                     # Patch exceeded time threshold without validation


class RollbackPolicy(Enum):
    """Policy for automatic rollbacks."""
    CONSERVATIVE = "conservative"   # Only roll back on critical issues (security, tests)
    BALANCED = "balanced"           # Default balanced approach
    AGGRESSIVE = "aggressive"       # Roll back at any sign of issues
    PLANNER_GUIDED = "planner"      # Let planner make decisions
    MANUAL_ONLY = "manual_only"     # Only manual rollbacks allowed


class RollbackStatus(Enum):
    """Status of a rollback operation."""
    SUCCESS = "success"             # Rollback completed successfully
    FAILED = "failed"               # Rollback failed
    PARTIAL = "partial"             # Rollback partially completed
    PENDING = "pending"             # Rollback is pending
    CANCELLED = "cancelled"         # Rollback was cancelled


# ───────────────────────────────────────────────────────────────────────────────
# Exceptions
# ───────────────────────────────────────────────────────────────────────────────
class RollbackError(RuntimeError):
    """Raised if registry missing, git apply fails, or path tampered."""


# ───────────────────────────────────────────────────────────────────────────────
# Configuration & History
# ───────────────────────────────────────────────────────────────────────────────
class RollbackConfig:
    """Configuration for automatic rollback behavior."""
    
    DEFAULT_CONFIG = {
        "policy": RollbackPolicy.BALANCED.value,
        "triggers": {
            RollbackTrigger.TEST_FAILURE.value: True,
            RollbackTrigger.REGRESSION.value: True,
            RollbackTrigger.PERFORMANCE_DEGRADATION.value: False,
            RollbackTrigger.SECURITY_RISK.value: True,
            RollbackTrigger.PLANNER_DECISION.value: True,
            RollbackTrigger.TIMEOUT.value: False
        },
        "thresholds": {
            "test_failure_count": 3,          # Number of test failures before rollback
            "regression_severity": 0.7,        # Severity threshold (0-1) for regression
            "performance_threshold": 1.5,      # Performance degradation factor
            "validation_timeout_hours": 24,    # Hours to wait for validation
            "max_auto_rollbacks_per_day": 5    # Safety limit
        },
        "notifications": {
            "email_on_rollback": False,
            "slack_on_rollback": False,
            "notify_planner": True
        },
        "auto_rollback_enabled": True,
        "last_updated": datetime.now().isoformat()
    }
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if CONFIG_FILE.exists():
            try:
                with CONFIG_FILE.open("r", encoding="utf-8") as f:
                    config = json.load(f)
                logger.info("Loaded rollback configuration from %s", CONFIG_FILE)
                return config
            except (json.JSONDecodeError, IOError) as e:
                logger.error("Failed to load rollback config: %s", e)
                
        # Create default config
        logger.info("Creating default rollback configuration")
        config = self.DEFAULT_CONFIG.copy()
        self._save_config(config)
        return config
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        config["last_updated"] = datetime.now().isoformat()
        try:
            with CONFIG_FILE.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info("Saved rollback configuration to %s", CONFIG_FILE)
        except IOError as e:
            logger.error("Failed to save rollback config: %s", e)
    
    def save(self) -> None:
        """Save current configuration."""
        self._save_config(self.config)
    
    def get_policy(self) -> RollbackPolicy:
        """Get current rollback policy."""
        policy_name = self.config.get("policy", RollbackPolicy.BALANCED.value)
        try:
            return RollbackPolicy(policy_name)
        except ValueError:
            logger.warning("Invalid policy %s, using BALANCED", policy_name)
            return RollbackPolicy.BALANCED
    
    def set_policy(self, policy: Union[RollbackPolicy, str]) -> None:
        """Set rollback policy."""
        if isinstance(policy, str):
            try:
                policy = RollbackPolicy(policy)
            except ValueError:
                raise ValueError(f"Invalid policy: {policy}")
        
        self.config["policy"] = policy.value
        logger.info("Set rollback policy to %s", policy.value)
        self.save()
    
    def is_trigger_enabled(self, trigger: Union[RollbackTrigger, str]) -> bool:
        """Check if a specific trigger is enabled."""
        if isinstance(trigger, str):
            try:
                trigger = RollbackTrigger(trigger)
            except ValueError:
                logger.warning("Invalid trigger: %s", trigger)
                return False
        
        triggers = self.config.get("triggers", {})
        return triggers.get(trigger.value, False)
    
    def set_trigger_enabled(self, trigger: Union[RollbackTrigger, str], enabled: bool) -> None:
        """Enable or disable a trigger."""
        if isinstance(trigger, str):
            try:
                trigger = RollbackTrigger(trigger)
            except ValueError:
                raise ValueError(f"Invalid trigger: {trigger}")
        
        if "triggers" not in self.config:
            self.config["triggers"] = {}
        
        self.config["triggers"][trigger.value] = bool(enabled)
        logger.info("%s trigger %s", "Enabled" if enabled else "Disabled", trigger.value)
        self.save()
    
    def get_threshold(self, name: str) -> Union[int, float, None]:
        """Get a threshold value."""
        thresholds = self.config.get("thresholds", {})
        return thresholds.get(name)
    
    def set_threshold(self, name: str, value: Union[int, float]) -> None:
        """Set a threshold value."""
        if "thresholds" not in self.config:
            self.config["thresholds"] = {}
        
        self.config["thresholds"][name] = value
        logger.info("Set threshold %s to %s", name, value)
        self.save()
    
    def is_auto_rollback_enabled(self) -> bool:
        """Check if automatic rollback is enabled."""
        return self.config.get("auto_rollback_enabled", True)
    
    def set_auto_rollback_enabled(self, enabled: bool) -> None:
        """Enable or disable automatic rollback."""
        self.config["auto_rollback_enabled"] = bool(enabled)
        logger.info("%s automatic rollback", "Enabled" if enabled else "Disabled")
        self.save()


class RollbackHistory:
    """Track history of rollback operations."""
    
    def __init__(self):
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load rollback history from file."""
        if HISTORY_FILE.exists():
            try:
                with HISTORY_FILE.open("r", encoding="utf-8") as f:
                    history = json.load(f)
                return history
            except (json.JSONDecodeError, IOError) as e:
                logger.error("Failed to load rollback history: %s", e)
        
        return []
    
    def _save_history(self) -> None:
        """Save rollback history to file."""
        try:
            with HISTORY_FILE.open("w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        except IOError as e:
            logger.error("Failed to save rollback history: %s", e)
    
    def add_entry(
        self, 
        bug_id: str, 
        status: RollbackStatus, 
        trigger: RollbackTrigger, 
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add an entry to the rollback history.
        
        Args:
            bug_id: Bug identifier
            status: Status of the rollback
            trigger: What triggered the rollback
            details: Additional details
            
        Returns:
            The created history entry
        """
        entry = {
            "bug_id": bug_id,
            "status": status.value,
            "trigger": trigger.value,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.history.append(entry)
        self._save_history()
        logger.info(
            "Added rollback history entry: bug=%s status=%s trigger=%s", 
            bug_id, status.value, trigger.value
        )
        
        return entry
    
    def get_entries_for_bug(self, bug_id: str) -> List[Dict[str, Any]]:
        """Get all history entries for a specific bug."""
        return [entry for entry in self.history if entry["bug_id"] == bug_id]
    
    def get_recent_entries(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent history entries within specified hours."""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        
        recent = []
        for entry in self.history:
            try:
                entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if entry_time >= cutoff:
                    recent.append(entry)
            except (ValueError, KeyError):
                continue
                
        return recent
    
    def get_entries_by_trigger(self, trigger: RollbackTrigger) -> List[Dict[str, Any]]:
        """Get history entries for a specific trigger."""
        return [entry for entry in self.history if entry.get("trigger") == trigger.value]
    
    def get_entries_by_status(self, status: RollbackStatus) -> List[Dict[str, Any]]:
        """Get history entries for a specific status."""
        return [entry for entry in self.history if entry.get("status") == status.value]
    
    def clear_history(self, older_than_days: int = 30) -> int:
        """
        Clear history entries older than specified days.
        
        Returns:
            Number of entries removed
        """
        cutoff = datetime.now().timestamp() - (older_than_days * 86400)
        
        old_len = len(self.history)
        new_history = []
        
        for entry in self.history:
            try:
                entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if entry_time >= cutoff:
                    new_history.append(entry)
            except (ValueError, KeyError):
                # Keep entries with invalid timestamps (shouldn't happen)
                new_history.append(entry)
        
        self.history = new_history
        self._save_history()
        
        removed = old_len - len(self.history)
        if removed > 0:
            logger.info("Cleared %d rollback history entries older than %d days", 
                       removed, older_than_days)
            
        return removed


# ───────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ───────────────────────────────────────────────────────────────────────────────
def _calculate_patch_hash(content: str) -> str:
    """
    Calculate a secure hash for patch content.
    
    Args:
        content: The patch content
        
    Returns:
        SHA-256 hash of the content
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _verify_patch_hash(patch_path: Path, expected_hash: str) -> bool:
    """
    Verify that a patch file matches its expected hash.
    
    Args:
        patch_path: Path to the patch file
        expected_hash: Expected SHA-256 hash
        
    Returns:
        True if the hash matches, False otherwise
    """
    if not patch_path.exists():
        return False
        
    content = patch_path.read_text(encoding='utf-8')
    actual_hash = _calculate_patch_hash(content)
    
    return actual_hash == expected_hash


def _load_registry() -> Dict[str, Any]:
    """Load the patch registry file."""
    if REGISTRY_FILE.exists():
        with REGISTRY_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_registry(reg: Dict[str, Any]) -> None:
    """Save the patch registry file."""
    with REGISTRY_FILE.open("w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=2)


# ───────────────────────────────────────────────────────────────────────────────
# Automatic Rollback
# ───────────────────────────────────────────────────────────────────────────────
class AutomaticRollbackManager:
    """
    Manages automatic rollbacks based on configured triggers and conditions.
    
    This class monitors for conditions that should trigger a rollback, such as:
    - Test failures after a patch was applied
    - Performance degradation
    - Security issues
    - Regressions in functionality
    - Timeout waiting for validation
    
    It integrates with the planner agent for intelligent rollback decisions.
    The planner can provide strategic guidance on:
    - When to roll back (based on wider system context)
    - When to wait longer (if a solution is imminent)
    - Alternative solutions to try before rolling back
    - Coordinated rollbacks of multiple related bugs
    """
    
    def __init__(
        self, 
        planner=None, 
        verification_engine=None, 
        monitor=None
    ):
        """
        Initialize automatic rollback manager.
        
        Args:
            planner: Optional planner agent for intelligent decisions
            verification_engine: Optional verification engine for validating patches
            monitor: Optional system monitor for metrics
        """
        self.planner = planner
        self.verification_engine = verification_engine
        self.monitor = monitor
        
        self.config = RollbackConfig()
        self.history = RollbackHistory()
        
        # Active rollback monitors
        self._monitored_bugs = {}  # bug_id -> monitoring_info
        self._pending_rollbacks = {}  # bug_id -> rollback_info
        
        # Load state
        self._load_state()
    
    def _load_state(self) -> None:
        """Load monitored bugs and pending rollbacks state."""
        # Could add persistent state loading here
        pass
    
    def _save_state(self) -> None:
        """Save monitored bugs and pending rollbacks state."""
        # Could add persistent state saving here
        pass
    
    def register_patch(self, bug_id: str, forward_diff: str) -> Path:
        """
        Register a patch and start monitoring it for automatic rollback.
        
        Args:
            bug_id: Bug identifier
            forward_diff: Forward diff of the patch
            
        Returns:
            Path to the patch file
        """
        # Register the patch using the existing function
        patch_path = register_patch(bug_id, forward_diff)
        
        # Set up monitoring for this patch
        self._setup_monitoring(bug_id)
        
        return patch_path
    
    def _setup_monitoring(self, bug_id: str) -> None:
        """
        Setup monitoring for a patch.
        
        Args:
            bug_id: Bug identifier
        """
        if not self.config.is_auto_rollback_enabled():
            logger.info("Automatic rollback disabled, not monitoring bug %s", bug_id)
            return
        
        # Get validation timeout threshold (default to 24 hours)
        validation_timeout = self.config.get_threshold("validation_timeout_hours") or 24
        timeout_time = datetime.now().timestamp() + (validation_timeout * 3600)
        
        # Create monitoring info
        monitor_info = {
            "bug_id": bug_id,
            "start_time": datetime.now().isoformat(),
            "validation_timeout": timeout_time,
            "baseline_metrics": self._collect_baseline_metrics(bug_id),
            "test_failures": 0,
            "performance_metrics": [],
            "security_issues": [],
            "regressions": [],
            "last_checked": datetime.now().isoformat()
        }
        
        # Store monitoring info
        self._monitored_bugs[bug_id] = monitor_info
        logger.info("Started monitoring bug %s for automatic rollback", bug_id)
    
    def _collect_baseline_metrics(self, bug_id: str) -> Dict[str, Any]:
        """
        Collect baseline metrics for comparison.
        
        Args:
            bug_id: Bug identifier
            
        Returns:
            Dictionary with baseline metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "test_status": {},
            "performance": {},
            "security": {}
        }
        
        # Collect metrics from system monitor if available
        if self.monitor:
            try:
                # The actual implementation would collect metrics from the monitor
                # Here we just use placeholders
                metrics["performance"] = {
                    "response_time_ms": 200,
                    "memory_usage_mb": 100,
                    "cpu_usage_percent": 10
                }
            except Exception as e:
                logger.warning("Failed to collect baseline metrics from monitor: %s", e)
        
        return metrics
    
    def check_for_rollbacks(self) -> List[str]:
        """
        Check all monitored bugs for conditions that should trigger a rollback.
        
        Returns:
            List of bug IDs that were rolled back
        """
        if not self.config.is_auto_rollback_enabled():
            return []
        
        # Check daily limit
        if self._check_daily_limit_reached():
            logger.warning("Daily automatic rollback limit reached, skipping checks")
            return []
        
        rolled_back = []
        for bug_id, info in list(self._monitored_bugs.items()):
            # Skip bugs that are already pending rollback
            if bug_id in self._pending_rollbacks:
                continue
                
            # Check various triggers
            trigger = self._check_rollback_conditions(bug_id, info)
            
            if trigger:
                # Trigger rollback
                try:
                    self.trigger_rollback(bug_id, trigger)
                    rolled_back.append(bug_id)
                except Exception as e:
                    logger.error("Failed to trigger rollback for bug %s: %s", bug_id, e)
                    
                # Remove from monitored bugs
                self._monitored_bugs.pop(bug_id, None)
        
        return rolled_back
    
    def _check_daily_limit_reached(self) -> bool:
        """Check if daily automatic rollback limit has been reached."""
        max_rollbacks = self.config.get_threshold("max_auto_rollbacks_per_day") or 5
        
        # Count recent automatic rollbacks
        recent_entries = self.history.get_recent_entries(hours=24)
        auto_rollbacks = [
            e for e in recent_entries 
            if e.get("trigger") != RollbackTrigger.MANUAL.value
        ]
        
        return len(auto_rollbacks) >= max_rollbacks
    
    def _check_rollback_conditions(
        self, bug_id: str, info: Dict[str, Any]
    ) -> Optional[RollbackTrigger]:
        """
        Check if a bug meets any conditions for automatic rollback.
        
        Args:
            bug_id: Bug identifier
            info: Monitoring info for the bug
            
        Returns:
            Trigger that should cause a rollback, or None
        """
        # Update last checked time
        info["last_checked"] = datetime.now().isoformat()
        
        # Check timeout
        if self._check_timeout_trigger(info):
            return RollbackTrigger.TIMEOUT
        
        # Check test failures
        if self._check_test_failure_trigger(info):
            return RollbackTrigger.TEST_FAILURE
        
        # Check performance degradation
        if self._check_performance_trigger(info):
            return RollbackTrigger.PERFORMANCE_DEGRADATION
        
        # Check regression
        if self._check_regression_trigger(info):
            return RollbackTrigger.REGRESSION
        
        # Check security
        if self._check_security_trigger(info):
            return RollbackTrigger.SECURITY_RISK
        
        # Check planner decision
        if self._check_planner_trigger(bug_id, info):
            return RollbackTrigger.PLANNER_DECISION
        
        return None
    
    def _check_timeout_trigger(self, info: Dict[str, Any]) -> bool:
        """Check if a bug has timed out waiting for validation."""
        # Skip if timeout trigger is disabled
        if not self.config.is_trigger_enabled(RollbackTrigger.TIMEOUT):
            return False
            
        # Check if validation timeout has been reached
        if "validation_timeout" in info:
            now = datetime.now().timestamp()
            timeout = info["validation_timeout"]
            
            return now > timeout
            
        return False
    
    def _check_test_failure_trigger(self, info: Dict[str, Any]) -> bool:
        """Check if a bug has caused test failures."""
        # Skip if test failure trigger is disabled
        if not self.config.is_trigger_enabled(RollbackTrigger.TEST_FAILURE):
            return False
            
        # Get test failure threshold
        threshold = self.config.get_threshold("test_failure_count") or 3
        
        # Check if test failures exceed threshold
        return info.get("test_failures", 0) >= threshold
    
    def _check_performance_trigger(self, info: Dict[str, Any]) -> bool:
        """Check if a bug has caused performance degradation."""
        # Skip if performance trigger is disabled
        if not self.config.is_trigger_enabled(RollbackTrigger.PERFORMANCE_DEGRADATION):
            return False
            
        # Check for performance degradation (implementation would depend on metrics)
        # This is a simplified check
        if "baseline_metrics" in info and "performance" in info.get("baseline_metrics", {}):
            baseline = info["baseline_metrics"]["performance"]
            threshold = self.config.get_threshold("performance_threshold") or 1.5
            
            # The actual implementation would compare current metrics to baseline
            # For simplicity, we just return False here
            return False
            
        return False
    
    def _check_regression_trigger(self, info: Dict[str, Any]) -> bool:
        """Check if a bug has caused regressions in functionality."""
        # Skip if regression trigger is disabled
        if not self.config.is_trigger_enabled(RollbackTrigger.REGRESSION):
            return False
            
        # Check for regressions
        severity_threshold = self.config.get_threshold("regression_severity") or 0.7
        
        # The actual implementation would analyze regressions
        # For simplicity, we just return False here
        return False
    
    def _check_security_trigger(self, info: Dict[str, Any]) -> bool:
        """Check if a bug has introduced security issues."""
        # Skip if security trigger is disabled
        if not self.config.is_trigger_enabled(RollbackTrigger.SECURITY_RISK):
            return False
            
        # Check for security issues
        # The actual implementation would analyze security scans
        # For simplicity, we just return False here
        return False
    
    def _check_planner_trigger(self, bug_id: str, info: Dict[str, Any]) -> bool:
        """Check if the planner recommends rolling back."""
        # Skip if planner trigger is disabled or no planner is available
        if not self.config.is_trigger_enabled(RollbackTrigger.PLANNER_DECISION) or not self.planner:
            return False
            
        try:
            # Create decision context with bug info and system state
            context = {
                "bug_id": bug_id,
                "monitoring_info": info,
                "system_state": self._get_system_state(),
                "policy": self.config.get_policy().value,
                "thresholds": self.config.config.get("thresholds", {})
            }
            
            # Consult planner for rollback recommendation
            if hasattr(self.planner, "get_rollback_recommendation"):
                recommendation = self.planner.get_rollback_recommendation(bug_id, context)
            else:
                # Default behavior if planner doesn't have specific rollback method
                recommendation = {"decision": False, "reason": "Planner doesn't support rollback decisions"}
            
            # Extract decision from recommendation
            should_rollback = recommendation.get("decision", False)
            
            # Store recommendation for reference
            info["planner_recommendation"] = recommendation
            
            # Log planner's recommendation
            if should_rollback:
                logger.info(
                    "Planner recommended rollback for bug %s: %s", 
                    bug_id, recommendation.get("reason", "No reason provided")
                )
            
            return should_rollback
        except Exception as e:
            logger.error("Error consulting planner for rollback decision: %s", e)
            return False
            
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for planner context."""
        # This would normally be populated with actual system state
        state = {
            "total_bugs": len(self._monitored_bugs) + len(self._pending_rollbacks),
            "available_resources": 0.8,  # 80% resources available
            "system_health": 0.9,  # 90% healthy
            "current_phase": "analysis"  # Current system phase
        }
        
        # Add system monitor metrics if available
        if self.monitor:
            try:
                # The actual implementation would get real metrics
                state["metrics"] = {
                    "memory_usage": 0.6,  # 60% memory usage
                    "cpu_usage": 0.4,     # 40% CPU usage
                    "response_time_ms": 250  # 250ms average response time
                }
            except Exception as e:
                logger.warning("Failed to get system metrics: %s", e)
                
        return state
    
    def trigger_rollback(
        self, 
        bug_id: str, 
        trigger: Union[RollbackTrigger, str], 
        manual: bool = False,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Trigger a rollback for a bug.
        
        Args:
            bug_id: Bug identifier
            trigger: What triggered the rollback
            manual: Whether this is a manual rollback
            details: Additional details about the rollback
            
        Returns:
            Whether the rollback was successfully triggered
        """
        if isinstance(trigger, str):
            try:
                trigger = RollbackTrigger(trigger)
            except ValueError:
                logger.error("Invalid rollback trigger: %s", trigger)
                return False
        
        # For manual rollbacks, override trigger
        if manual:
            trigger = RollbackTrigger.MANUAL
        
        logger.info("Triggering rollback for bug %s: %s", bug_id, trigger.value)
        
        try:
            # Perform the actual rollback
            rollback_patch(bug_id)
            
            # Record in history
            self.history.add_entry(
                bug_id=bug_id,
                status=RollbackStatus.SUCCESS,
                trigger=trigger,
                details=details
            )
            
            # Remove from pending rollbacks if present
            self._pending_rollbacks.pop(bug_id, None)
            
            # Remove from monitored bugs if present
            self._monitored_bugs.pop(bug_id, None)
            
            return True
            
        except RollbackError as e:
            # Record failure in history
            self.history.add_entry(
                bug_id=bug_id,
                status=RollbackStatus.FAILED,
                trigger=trigger,
                details={"error": str(e), **(details or {})}
            )
            
            logger.error("Rollback failed for bug %s: %s", bug_id, e)
            return False
    
    def schedule_rollback(
        self, 
        bug_id: str, 
        trigger: Union[RollbackTrigger, str],
        delay_seconds: int = 0,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Schedule a rollback to happen after a delay.
        
        Args:
            bug_id: Bug identifier
            trigger: What triggered the rollback
            delay_seconds: Delay before rollback in seconds
            details: Additional details about the rollback
            
        Returns:
            Whether the rollback was successfully scheduled
        """
        if isinstance(trigger, str):
            try:
                trigger = RollbackTrigger(trigger)
            except ValueError:
                logger.error("Invalid rollback trigger: %s", trigger)
                return False
        
        # Calculate rollback time
        rollback_time = datetime.now().timestamp() + delay_seconds
        
        # Record pending rollback
        self._pending_rollbacks[bug_id] = {
            "bug_id": bug_id,
            "trigger": trigger.value,
            "scheduled_time": rollback_time,
            "details": details or {}
        }
        
        # Record in history
        self.history.add_entry(
            bug_id=bug_id,
            status=RollbackStatus.PENDING,
            trigger=trigger,
            details={
                "scheduled_time": datetime.fromtimestamp(rollback_time).isoformat(),
                "delay_seconds": delay_seconds,
                **(details or {})
            }
        )
        
        logger.info(
            "Scheduled rollback for bug %s: %s in %d seconds", 
            bug_id, trigger.value, delay_seconds
        )
        
        return True
    
    def process_pending_rollbacks(self) -> List[str]:
        """
        Process any pending rollbacks that are due.
        
        Returns:
            List of bug IDs that were rolled back
        """
        now = datetime.now().timestamp()
        rolled_back = []
        
        for bug_id, info in list(self._pending_rollbacks.items()):
            if now >= info["scheduled_time"]:
                # Rollback is due
                try:
                    trigger = RollbackTrigger(info["trigger"])
                    success = self.trigger_rollback(
                        bug_id=bug_id,
                        trigger=trigger,
                        details=info["details"]
                    )
                    
                    if success:
                        rolled_back.append(bug_id)
                except (ValueError, RollbackError) as e:
                    logger.error("Failed to process pending rollback for bug %s: %s", bug_id, e)
                    
                # Remove from pending rollbacks
                self._pending_rollbacks.pop(bug_id, None)
        
        return rolled_back
    
    def update_test_status(self, bug_id: str, test_results: Dict[str, Any]) -> None:
        """
        Update test status for a monitored bug.
        
        Args:
            bug_id: Bug identifier
            test_results: Test results data
        """
        if bug_id not in self._monitored_bugs:
            return
            
        info = self._monitored_bugs[bug_id]
        
        # Extract test failures
        failures = test_results.get("failures", 0)
        errors = test_results.get("errors", 0)
        total_failures = failures + errors
        
        # Update test failure count
        info["test_failures"] = info.get("test_failures", 0) + total_failures
        
        logger.info(
            "Updated test status for bug %s: failures=%d total=%d", 
            bug_id, total_failures, info["test_failures"]
        )
    
    def update_performance_metrics(self, bug_id: str, metrics: Dict[str, Any]) -> None:
        """
        Update performance metrics for a monitored bug.
        
        Args:
            bug_id: Bug identifier
            metrics: Performance metrics data
        """
        if bug_id not in self._monitored_bugs:
            return
            
        info = self._monitored_bugs[bug_id]
        
        # Store metrics with timestamp
        metrics["timestamp"] = datetime.now().isoformat()
        info["performance_metrics"].append(metrics)
        
        logger.info("Updated performance metrics for bug %s", bug_id)
    
    def update_security_status(self, bug_id: str, security_data: Dict[str, Any]) -> None:
        """
        Update security status for a monitored bug.
        
        Args:
            bug_id: Bug identifier
            security_data: Security scan data
        """
        if bug_id not in self._monitored_bugs:
            return
            
        info = self._monitored_bugs[bug_id]
        
        # Store security data with timestamp
        security_data["timestamp"] = datetime.now().isoformat()
        info["security_issues"].append(security_data)
        
        logger.info("Updated security status for bug %s", bug_id)
    
    def get_rollback_status(self, bug_id: str) -> Optional[Dict[str, Any]]:
        """
        Get rollback status for a bug.
        
        Args:
            bug_id: Bug identifier
            
        Returns:
            Dictionary with rollback status or None if not found
        """
        # Check if bug is being monitored
        if bug_id in self._monitored_bugs:
            return {
                "status": "monitored",
                "details": self._monitored_bugs[bug_id]
            }
            
        # Check if rollback is pending
        if bug_id in self._pending_rollbacks:
            pending_info = self._pending_rollbacks[bug_id]
            scheduled_time = pending_info["scheduled_time"]
            
            return {
                "status": "pending",
                "scheduled_time": datetime.fromtimestamp(scheduled_time).isoformat(),
                "trigger": pending_info["trigger"],
                "details": pending_info["details"]
            }
            
        # Check history
        entries = self.history.get_entries_for_bug(bug_id)
        if entries:
            latest = max(entries, key=lambda e: e.get("timestamp", ""))
            return {
                "status": latest.get("status"),
                "timestamp": latest.get("timestamp"),
                "trigger": latest.get("trigger"),
                "details": latest.get("details", {})
            }
            
        return None
    
    def cancel_pending_rollback(self, bug_id: str) -> bool:
        """
        Cancel a pending rollback.
        
        Args:
            bug_id: Bug identifier
            
        Returns:
            Whether a pending rollback was cancelled
        """
        if bug_id in self._pending_rollbacks:
            # Get info before removing
            info = self._pending_rollbacks.pop(bug_id)
            
            # Record cancellation in history
            self.history.add_entry(
                bug_id=bug_id,
                status=RollbackStatus.CANCELLED,
                trigger=RollbackTrigger(info["trigger"]),
                details={
                    "cancelled_at": datetime.now().isoformat(),
                    "original_details": info["details"]
                }
            )
            
            logger.info("Cancelled pending rollback for bug %s", bug_id)
            return True
            
        return False
    
    def stop_monitoring(self, bug_id: str) -> bool:
        """
        Stop monitoring a bug for automatic rollback.
        
        Args:
            bug_id: Bug identifier
            
        Returns:
            Whether monitoring was stopped
        """
        if bug_id in self._monitored_bugs:
            self._monitored_bugs.pop(bug_id)
            logger.info("Stopped monitoring bug %s for automatic rollback", bug_id)
            return True
            
        return False


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
# Global automatic rollback manager
_auto_rollback_manager = None

def get_rollback_manager(
    planner=None, 
    verification_engine=None, 
    monitor=None
) -> AutomaticRollbackManager:
    """
    Get the global automatic rollback manager instance.
    
    The manager integrates with the planner for strategic rollback decisions.
    When a planner is provided, the rollback manager will consult it for:
    - Whether to roll back a problematic patch
    - When to wait for an imminent fix instead of rolling back
    - How to coordinate multiple related rollbacks
    - Alternative strategies to try before rolling back
    
    Args:
        planner: Optional planner agent for strategic rollback decisions
        verification_engine: Optional verification engine for validating patches
        monitor: Optional system monitor for system health metrics
        
    Returns:
        Automatic rollback manager instance
    """
    global _auto_rollback_manager
    
    if _auto_rollback_manager is None:
        _auto_rollback_manager = AutomaticRollbackManager(
            planner=planner,
            verification_engine=verification_engine,
            monitor=monitor
        )
        
    # Update components if provided
    if planner is not None:
        _auto_rollback_manager.planner = planner
    if verification_engine is not None:
        _auto_rollback_manager.verification_engine = verification_engine
    if monitor is not None:
        _auto_rollback_manager.monitor = monitor
        
    return _auto_rollback_manager


def register_patch(bug_id: str, forward_diff: str) -> Path:
    """
    Called by the Analyst / patch bundle code **once** after a unit-green fix.
    Stores the *forward diff* so the reverse diff can be generated when needed.
    Returns the patch file path.
    
    Also calculates and stores a hash of the patch for integrity verification.
    """
    dest = ROLLBACK_DIR / f"{bug_id}.patch"
    if dest.exists():
        raise RollbackError(f"Patch for bug {bug_id} already registered")
    
    # Calculate patch hash before writing to file
    patch_hash = _calculate_patch_hash(forward_diff)
    
    # Write patch to file
    dest.write_text(forward_diff, encoding="utf-8")

    # Update registry with patch name and hash
    reg = _load_registry()
    reg[bug_id] = {
        "file": dest.name,
        "hash": patch_hash,
        "timestamp": datetime.now().isoformat()
    }
    _save_registry(reg)
    
    logger.info(f"Registered patch for bug {bug_id} with hash verification")
    
    # If automatic rollback manager exists, setup monitoring
    if _auto_rollback_manager is not None:
        _auto_rollback_manager._setup_monitoring(bug_id)
    
    return dest


def rollback_patch(bug_id: str) -> None:
    """
    Atomically restore pre-patch state for `bug_id`.

    Raises RollbackError if:
        • no patch registered,
        • patch hash verification fails (indicating tampering),
        • reverse apply fails,
        • git not available.
    """
    reg = _load_registry()
    patch_info = reg.get(bug_id)
    if patch_info is None:
        raise RollbackError(f"No patch recorded for bug {bug_id}")
    
    # Handle both old and new registry formats
    if isinstance(patch_info, str):
        # Old format: patch_info is just the filename
        patch_name = patch_info
        patch_hash = None  # No hash verification for old format
    else:
        # New format: patch_info is a dictionary with file, hash, and timestamp
        patch_name = patch_info.get("file")
        patch_hash = patch_info.get("hash")
        
        if not patch_name:
            raise RollbackError(f"Invalid patch info for bug {bug_id}")

    patch_path = ROLLBACK_DIR / patch_name
    if not patch_path.exists():
        raise RollbackError(f"Patch file missing: {patch_path}")
    
    # Verify patch hash if available
    if patch_hash:
        logger.info(f"Verifying patch integrity for bug {bug_id}")
        if not _verify_patch_hash(patch_path, patch_hash):
            raise RollbackError(f"Patch hash verification failed for bug {bug_id} - possible tampering detected")
        logger.info(f"Patch integrity verified for bug {bug_id}")

    # 1) dry-run check
    _git_apply(["--check", "-R", str(patch_path)])

    # 2) real revert (atomic at filesystem level—either succeeds or git exits non-zero)
    _git_apply(["-R", str(patch_path)])

    # 3) cleanup registry (idempotent)
    reg.pop(bug_id, None)
    _save_registry(reg)
    
    # 4) Log the rollback
    logger.info(f"Successfully rolled back patch for bug {bug_id}")


def trigger_automatic_rollback(
    bug_id: str, 
    trigger: Union[RollbackTrigger, str], 
    details: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Trigger an automatic rollback for a bug.
    
    Args:
        bug_id: Bug identifier
        trigger: What triggered the rollback
        details: Additional details about the rollback
        
    Returns:
        Whether the rollback was successful
    """
    manager = get_rollback_manager()
    return manager.trigger_rollback(bug_id, trigger, details=details)


def schedule_rollback(
    bug_id: str, 
    trigger: Union[RollbackTrigger, str],
    delay_seconds: int = 0,
    details: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Schedule a rollback to happen after a delay.
    
    Args:
        bug_id: Bug identifier
        trigger: What triggered the rollback
        delay_seconds: Delay before rollback in seconds
        details: Additional details about the rollback
        
    Returns:
        Whether the rollback was successfully scheduled
    """
    manager = get_rollback_manager()
    return manager.schedule_rollback(bug_id, trigger, delay_seconds, details)


def process_pending_rollbacks() -> List[str]:
    """
    Process any pending rollbacks that are due.
    
    Returns:
        List of bug IDs that were rolled back
    """
    manager = get_rollback_manager()
    return manager.process_pending_rollbacks()


def check_for_automatic_rollbacks() -> List[str]:
    """
    Check all monitored bugs for conditions that should trigger a rollback.
    
    Returns:
        List of bug IDs that were rolled back
    """
    manager = get_rollback_manager()
    return manager.check_for_rollbacks()


def update_test_status(bug_id: str, test_results: Dict[str, Any]) -> None:
    """
    Update test status for a monitored bug.
    
    Args:
        bug_id: Bug identifier
        test_results: Test results data
    """
    manager = get_rollback_manager()
    manager.update_test_status(bug_id, test_results)


def get_rollback_status(bug_id: str) -> Optional[Dict[str, Any]]:
    """
    Get rollback status for a bug.
    
    Args:
        bug_id: Bug identifier
        
    Returns:
        Dictionary with rollback status or None if not found
    """
    manager = get_rollback_manager()
    return manager.get_rollback_status(bug_id)


def cancel_pending_rollback(bug_id: str) -> bool:
    """
    Cancel a pending rollback.
    
    Args:
        bug_id: Bug identifier
        
    Returns:
        Whether a pending rollback was cancelled
    """
    manager = get_rollback_manager()
    return manager.cancel_pending_rollback(bug_id)


def set_rollback_policy(policy: Union[RollbackPolicy, str]) -> None:
    """
    Set the rollback policy.
    
    Args:
        policy: Rollback policy to set
    """
    manager = get_rollback_manager()
    manager.config.set_policy(policy)


def get_rollback_policy() -> RollbackPolicy:
    """
    Get the current rollback policy.
    
    Returns:
        Current rollback policy
    """
    manager = get_rollback_manager()
    return manager.config.get_policy()


# ───────────────────────────────────────────────────────────────────────────────
# CLI hook
# ───────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Enhanced CLI entry (invoked via `python -m core.rollback_manager [command] [args]`)
    with support for automatic rollback management.
    """
    import argparse

    # Create parent parser
    parser = argparse.ArgumentParser(
        description="Triangulum rollback manager - handles patch rollbacks and monitoring"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback a patch")
    rollback_parser.add_argument("bug_id", help="Bug identifier to revert")
    
    # Schedule command
    schedule_parser = subparsers.add_parser("schedule", help="Schedule a rollback")
    schedule_parser.add_argument("bug_id", help="Bug identifier to revert")
    schedule_parser.add_argument(
        "--delay", type=int, default=0, 
        help="Delay in seconds before rollback"
    )
    schedule_parser.add_argument(
        "--trigger", choices=[t.value for t in RollbackTrigger], 
        default=RollbackTrigger.MANUAL.value,
        help="Rollback trigger reason"
    )
    
    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a pending rollback")
    cancel_parser.add_argument("bug_id", help="Bug identifier")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get rollback status")
    status_parser.add_argument("bug_id", nargs="?", help="Bug identifier (optional)")
    
    # Process command
    process_parser = subparsers.add_parser(
        "process", help="Process pending rollbacks"
    )
    
    # Check command
    check_parser = subparsers.add_parser(
        "check", help="Check for automatic rollback conditions"
    )
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage rollback configuration")
    config_parser.add_argument(
        "--policy", choices=[p.value for p in RollbackPolicy], 
        help="Set rollback policy"
    )
    config_parser.add_argument(
        "--enable-trigger", choices=[t.value for t in RollbackTrigger], 
        help="Enable trigger"
    )
    config_parser.add_argument(
        "--disable-trigger", choices=[t.value for t in RollbackTrigger], 
        help="Disable trigger"
    )
    config_parser.add_argument(
        "--set-threshold", nargs=2, metavar=("NAME", "VALUE"),
        help="Set threshold value (e.g., test_failure_count 3)"
    )
    config_parser.add_argument(
        "--auto-rollback", choices=["enable", "disable"], 
        help="Enable or disable automatic rollback"
    )
    config_parser.add_argument(
        "--show", action="store_true", help="Show current configuration"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize rollback manager
    manager = get_rollback_manager()
    
    try:
        # Process commands
        if args.command == "rollback":
            # Rollback a patch
            rollback_patch(args.bug_id)
            print(f"✓ Rolled back patch for {args.bug_id}")
            
        elif args.command == "schedule":
            # Schedule a rollback
            success = schedule_rollback(
                args.bug_id, 
                args.trigger, 
                args.delay
            )
            if success:
                print(f"✓ Scheduled rollback for {args.bug_id} in {args.delay} seconds")
            else:
                print(f"✗ Failed to schedule rollback for {args.bug_id}")
                raise SystemExit(1)
                
        elif args.command == "cancel":
            # Cancel a pending rollback
            success = cancel_pending_rollback(args.bug_id)
            if success:
                print(f"✓ Cancelled pending rollback for {args.bug_id}")
            else:
                print(f"✗ No pending rollback found for {args.bug_id}")
                raise SystemExit(1)
                
        elif args.command == "status":
            # Get rollback status
            if args.bug_id:
                # Status for specific bug
                status = get_rollback_status(args.bug_id)
                if status:
                    print(f"Status for {args.bug_id}:")
                    for key, value in status.items():
                        if key == "details" and isinstance(value, dict):
                            print(f"  {key}:")
                            for k, v in value.items():
                                print(f"    {k}: {v}")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"No rollback information found for {args.bug_id}")
            else:
                # Overall status
                monitored = len(manager._monitored_bugs)
                pending = len(manager._pending_rollbacks)
                history = len(manager.history.history)
                print(f"Rollback manager status:")
                print(f"  Policy: {manager.config.get_policy().value}")
                print(f"  Auto-rollback: {'Enabled' if manager.config.is_auto_rollback_enabled() else 'Disabled'}")
                print(f"  Monitored bugs: {monitored}")
                print(f"  Pending rollbacks: {pending}")
                print(f"  History entries: {history}")
                
                # List monitored bugs
                if monitored > 0:
                    print("\nMonitored bugs:")
                    for bug_id in manager._monitored_bugs:
                        print(f"  - {bug_id}")
                
                # List pending rollbacks
                if pending > 0:
                    print("\nPending rollbacks:")
                    now = datetime.now().timestamp()
                    for bug_id, info in manager._pending_rollbacks.items():
                        scheduled_time = info["scheduled_time"]
                        remaining = max(0, scheduled_time - now)
                        print(f"  - {bug_id}: {info['trigger']} (in {remaining:.1f}s)")
                
        elif args.command == "process":
            # Process pending rollbacks
            rolled_back = process_pending_rollbacks()
            if rolled_back:
                print(f"✓ Processed {len(rolled_back)} rollbacks: {', '.join(rolled_back)}")
            else:
                print("No pending rollbacks to process")
                
        elif args.command == "check":
            # Check for automatic rollback conditions
            rolled_back = check_for_automatic_rollbacks()
            if rolled_back:
                print(f"✓ Rolled back {len(rolled_back)} bugs: {', '.join(rolled_back)}")
            else:
                print("No automatic rollbacks triggered")
                
        elif args.command == "config":
            # Configure rollback manager
            changed = False
            
            # Set policy
            if args.policy:
                manager.config.set_policy(args.policy)
                print(f"✓ Set rollback policy to {args.policy}")
                changed = True
                
            # Enable trigger
            if args.enable_trigger:
                manager.config.set_trigger_enabled(args.enable_trigger, True)
                print(f"✓ Enabled {args.enable_trigger} trigger")
                changed = True
                
            # Disable trigger
            if args.disable_trigger:
                manager.config.set_trigger_enabled(args.disable_trigger, False)
                print(f"✓ Disabled {args.disable_trigger} trigger")
                changed = True
                
            # Set threshold
            if args.set_threshold:
                name, value = args.set_threshold
                try:
                    # Try to convert to number
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                        
                    manager.config.set_threshold(name, value)
                    print(f"✓ Set threshold {name} to {value}")
                    changed = True
                except ValueError:
                    print(f"✗ Invalid threshold value: {value}")
                    raise SystemExit(1)
                    
            # Set auto-rollback
            if args.auto_rollback:
                enabled = args.auto_rollback == "enable"
                manager.config.set_auto_rollback_enabled(enabled)
                print(f"✓ {'Enabled' if enabled else 'Disabled'} automatic rollback")
                changed = True
                
            # Show configuration
            if args.show or not changed:
                config = manager.config.config
                print("Current configuration:")
                print(f"  Policy: {config['policy']}")
                print(f"  Auto-rollback: {'Enabled' if config['auto_rollback_enabled'] else 'Disabled'}")
                
                print("\n  Enabled triggers:")
                for trigger, enabled in config['triggers'].items():
                    print(f"    {trigger}: {'Enabled' if enabled else 'Disabled'}")
                
                print("\n  Thresholds:")
                for name, value in config['thresholds'].items():
                    print(f"    {name}: {value}")
                
        else:
            # Default to help if no command specified
            parser.print_help()
            
    except RollbackError as e:
        print(f"✗ Rollback failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print(f"✗ Error: {e}")
        raise SystemExit(1) from e


# ───────────────────────────────────────────────────────────────────────────────
# Git helper
# ───────────────────────────────────────────────────────────────────────────────
def _git_apply(extra_args: list[str]) -> None:
    cmd = ["git", "apply", "--whitespace=nowarn"] + extra_args
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RollbackError(
            f"`{' '.join(cmd)}` failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
