#!/usr/bin/env python3
"""
mttr_optimizer.py
───────────────────
Optimizes Mean Time To Repair (MTTR) to achieve the <15min target.

This component:
1. Tracks bug lifecycle timings from detection to resolution
2. Implements parallel repair strategies
3. Provides automated repair templates for common issues
4. Optimizes resource allocation for faster fixes
5. Continuously measures and refines the repair process
"""

import os
import sys
import json
import time
import logging
import statistics
import datetime
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/mttr.log")
    ]
)
logger = logging.getLogger("mttr_optimizer")

# ─── DATA STRUCTURES ────────────────────────────────────────────────────────────

@dataclass
class RepairTemplate:
    """Template for common bug fixes."""
    id: str
    name: str
    bug_pattern: str
    fix_pattern: str
    success_rate: float = 0.0
    avg_repair_time: float = 0.0
    usage_count: int = 0
    last_used: Optional[float] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class BugLifecycle:
    """Lifecycle of a bug from detection to resolution."""
    bug_id: str
    title: str
    description: str
    severity: str
    detected_time: float
    assigned_time: Optional[float] = None
    diagnosis_start_time: Optional[float] = None
    diagnosis_end_time: Optional[float] = None
    repair_start_time: Optional[float] = None
    repair_end_time: Optional[float] = None
    verification_start_time: Optional[float] = None
    verification_end_time: Optional[float] = None
    resolved_time: Optional[float] = None
    total_time: Optional[float] = None
    repair_time: Optional[float] = None
    successful: bool = False
    repair_attempts: int = 0
    templates_used: List[str] = field(default_factory=list)
    repair_agents: List[str] = field(default_factory=list)

@dataclass
class MTTRStats:
    """Statistics about MTTR performance."""
    time_period: str  # 'day', 'week', 'month', 'all'
    avg_mttr: float
    median_mttr: float
    min_mttr: float
    max_mttr: float
    p90_mttr: float  # 90th percentile
    total_bugs: int
    resolved_bugs: int
    success_rate: float
    by_severity: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class RepairAction:
    """Action to repair a bug."""
    action_id: str
    bug_id: str
    description: str
    type: str  # 'template', 'agent', 'manual'
    template_id: Optional[str] = None
    agent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    successful: bool = False
    result: Optional[str] = None

# ─── DATABASE OPERATIONS ─────────────────────────────────────────────────────────

class MTTRDatabase:
    """Database for storing MTTR-related information."""
    
    def __init__(self, db_path: str = "data/mttr_db.json"):
        self.db_path = db_path
        self.templates: Dict[str, RepairTemplate] = {}
        self.bugs: Dict[str, BugLifecycle] = {}
        self.actions: Dict[str, RepairAction] = {}
        self.stats: List[MTTRStats] = []
        self._load_db()
    
    def _load_db(self) -> None:
        """Load database from disk."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    
                    # Deserialize templates
                    self.templates = {
                        template_id: RepairTemplate(**template_data)
                        for template_id, template_data in data.get("templates", {}).items()
                    }
                    
                    # Deserialize bugs
                    self.bugs = {
                        bug_id: BugLifecycle(**bug_data)
                        for bug_id, bug_data in data.get("bugs", {}).items()
                    }
                    
                    # Deserialize actions
                    self.actions = {
                        action_id: RepairAction(**action_data)
                        for action_id, action_data in data.get("actions", {}).items()
                    }
                    
                    # Deserialize stats
                    self.stats = [MTTRStats(**stat_data) for stat_data in data.get("stats", [])]
                    
                logger.info(f"Loaded MTTR database from {self.db_path}")
            else:
                # Initialize with basic templates
                self._init_basic_templates()
                logger.info(f"No existing database found at {self.db_path}, initialized with basic templates")
        except Exception as e:
            logger.error(f"Error loading MTTR database: {str(e)}")
            # Initialize with basic templates
            self._init_basic_templates()
    
    def _init_basic_templates(self) -> None:
        """Initialize database with basic repair templates."""
        basic_templates = [
            RepairTemplate(
                id="missing-import",
                name="Missing Import",
                bug_pattern=r"ImportError|ModuleNotFoundError|NameError: name '(\w+)' is not defined",
                fix_pattern="import {module}",
                success_rate=0.8,
                avg_repair_time=60.0,
                tags=["python", "import", "common"]
            ),
            RepairTemplate(
                id="undefined-variable",
                name="Undefined Variable",
                bug_pattern=r"NameError: name '(\w+)' is not defined",
                fix_pattern="{var_name} = {default_value}",
                success_rate=0.6,
                avg_repair_time=120.0,
                tags=["python", "variable", "common"]
            ),
            RepairTemplate(
                id="type-error",
                name="Type Error",
                bug_pattern=r"TypeError: .*",
                fix_pattern="# Convert {var} to the correct type\n{var} = {type}({var})",
                success_rate=0.5,
                avg_repair_time=180.0,
                tags=["python", "type", "conversion"]
            ),
            RepairTemplate(
                id="index-error",
                name="Index Error",
                bug_pattern=r"IndexError: .*",
                fix_pattern="if len({collection}) > {index}:\n    {var} = {collection}[{index}]\nelse:\n    # Handle the case when index is out of bounds\n    {var} = {default}",
                success_rate=0.7,
                avg_repair_time=150.0,
                tags=["python", "index", "bounds-check"]
            ),
            RepairTemplate(
                id="key-error",
                name="Key Error",
                bug_pattern=r"KeyError: .*",
                fix_pattern="if {key} in {dict}:\n    {var} = {dict}[{key}]\nelse:\n    # Handle the case when key is not present\n    {var} = {default}",
                success_rate=0.7,
                avg_repair_time=150.0,
                tags=["python", "dict", "key-check"]
            )
        ]
        
        for template in basic_templates:
            self.templates[template.id] = template
    
    def save(self) -> None:
        """Save database to disk."""
        try:
            data = {
                "templates": {template_id: asdict(template) for template_id, template in self.templates.items()},
                "bugs": {bug_id: asdict(bug) for bug_id, bug in self.bugs.items()},
                "actions": {action_id: asdict(action) for action_id, action in self.actions.items()},
                "stats": [asdict(stat) for stat in self.stats]
            }
            
            # Create temporary file to avoid corruption
            import tempfile
            with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
                json.dump(data, tmp, indent=2)
            
            # Replace original file with new one
            os.replace(tmp.name, self.db_path)
            logger.info(f"Saved MTTR database to {self.db_path}")
        except Exception as e:
            logger.error(f"Error saving MTTR database: {str(e)}")
    
    def add_template(self, template: RepairTemplate) -> None:
        """Add a repair template to the database."""
        self.templates[template.id] = template
    
    def add_bug(self, bug: BugLifecycle) -> None:
        """Add a bug lifecycle to the database."""
        self.bugs[bug.bug_id] = bug
    
    def add_action(self, action: RepairAction) -> None:
        """Add a repair action to the database."""
        self.actions[action.action_id] = action
    
    def add_stats(self, stats: MTTRStats) -> None:
        """Add MTTR statistics to the database."""
        self.stats.append(stats)
        # Keep at most 1000 stats records
        if len(self.stats) > 1000:
            self.stats = self.stats[-1000:]
    
    def get_template(self, template_id: str) -> Optional[RepairTemplate]:
        """Get a repair template by ID."""
        return self.templates.get(template_id)
    
    def get_bug(self, bug_id: str) -> Optional[BugLifecycle]:
        """Get a bug lifecycle by ID."""
        return self.bugs.get(bug_id)
    
    def get_action(self, action_id: str) -> Optional[RepairAction]:
        """Get a repair action by ID."""
        return self.actions.get(action_id)
    
    def get_latest_stats(self, time_period: str) -> Optional[MTTRStats]:
        """Get the latest MTTR statistics for a time period."""
        for stat in reversed(self.stats):
            if stat.time_period == time_period:
                return stat
        return None
    
    def get_templates_by_tag(self, tag: str) -> List[RepairTemplate]:
        """Get repair templates by tag."""
        return [template for template in self.templates.values() if tag in template.tags]
    
    def get_templates_by_success_rate(self, min_rate: float = 0.0, max_rate: float = 1.0) -> List[RepairTemplate]:
        """Get repair templates by success rate range."""
        return [template for template in self.templates.values() 
                if min_rate <= template.success_rate <= max_rate]
    
    def get_bugs_in_timeframe(self, start_time: float, end_time: float) -> List[BugLifecycle]:
        """Get bugs detected within a timeframe."""
        return [bug for bug in self.bugs.values() 
                if start_time <= bug.detected_time <= end_time]
    
    def get_bugs_by_severity(self, severity: str) -> List[BugLifecycle]:
        """Get bugs by severity."""
        return [bug for bug in self.bugs.values() if bug.severity == severity]
    
    def get_avg_mttr(self, severities: Optional[List[str]] = None, time_period: Optional[float] = None) -> float:
        """
        Get average MTTR for bugs matching criteria.
        
        Args:
            severities: Optional list of severities to filter by
            time_period: Optional time period in seconds to consider (e.g., 7*24*60*60 for 7 days)
            
        Returns:
            Average MTTR in seconds
        """
        now = time.time()
        bugs = list(self.bugs.values())
        
        # Filter by time period if specified
        if time_period is not None:
            bugs = [bug for bug in bugs if now - bug.detected_time <= time_period]
        
        # Filter by severities if specified
        if severities is not None:
            bugs = [bug for bug in bugs if bug.severity in severities]
        
        # Filter only resolved bugs
        resolved_bugs = [bug for bug in bugs if bug.resolved_time is not None and bug.total_time is not None]
        
        if not resolved_bugs:
            return 0.0
        
        return sum(bug.total_time for bug in resolved_bugs) / len(resolved_bugs)

# ─── MTTR OPTIMIZATION ─────────────────────────────────────────────────────────────

class MTTROptimizer:
    """Main class for optimizing Mean Time To Repair."""
    
    def __init__(self, db_path: str = "data/mttr_db.json", target_mttr: float = 15 * 60):
        """
        Initialize MTTR optimizer.
        
        Args:
            db_path: Path to MTTR database
            target_mttr: Target MTTR in seconds (default: 15 minutes)
        """
        self.db = MTTRDatabase(db_path)
        self.target_mttr = target_mttr
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        logger.info(f"MTTR optimizer initialized with target of {target_mttr} seconds")
    
    def start_bug_lifecycle(self, bug_id: str, title: str, description: str, severity: str) -> BugLifecycle:
        """
        Start tracking a bug lifecycle.
        
        Args:
            bug_id: Unique identifier for the bug
            title: Short title describing the bug
            description: Detailed description of the bug
            severity: Severity level of the bug (e.g., 'critical', 'high', 'medium', 'low')
            
        Returns:
            BugLifecycle object for the bug
        """
        bug = BugLifecycle(
            bug_id=bug_id,
            title=title,
            description=description,
            severity=severity,
            detected_time=time.time()
        )
        
        self.db.add_bug(bug)
        self.db.save()
        
        logger.info(f"Started tracking bug {bug_id}: {title} (severity: {severity})")
        
        return bug
    
    def assign_bug(self, bug_id: str) -> None:
        """
        Mark a bug as assigned for repair.
        
        Args:
            bug_id: Unique identifier for the bug
        """
        bug = self.db.get_bug(bug_id)
        if bug is None:
            logger.error(f"Bug {bug_id} not found")
            return
        
        bug.assigned_time = time.time()
        self.db.save()
        
        logger.info(f"Bug {bug_id} assigned for repair")
    
    def start_diagnosis(self, bug_id: str) -> None:
        """
        Start the diagnosis phase for a bug.
        
        Args:
            bug_id: Unique identifier for the bug
        """
        bug = self.db.get_bug(bug_id)
        if bug is None:
            logger.error(f"Bug {bug_id} not found")
            return
        
        bug.diagnosis_start_time = time.time()
        self.db.save()
        
        logger.info(f"Started diagnosis for bug {bug_id}")
    
    def end_diagnosis(self, bug_id: str) -> None:
        """
        End the diagnosis phase for a bug.
        
        Args:
            bug_id: Unique identifier for the bug
        """
        bug = self.db.get_bug(bug_id)
        if bug is None:
            logger.error(f"Bug {bug_id} not found")
            return
        
        bug.diagnosis_end_time = time.time()
        self.db.save()
        
        logger.info(f"Completed diagnosis for bug {bug_id}")
    
    def start_repair(self, bug_id: str) -> None:
        """
        Start the repair phase for a bug.
        
        Args:
            bug_id: Unique identifier for the bug
        """
        bug = self.db.get_bug(bug_id)
        if bug is None:
            logger.error(f"Bug {bug_id} not found")
            return
        
        bug.repair_start_time = time.time()
        self.db.save()
        
        logger.info(f"Started repair for bug {bug_id}")
    
    def record_repair_attempt(self, bug_id: str, action: RepairAction) -> None:
        """
        Record a repair attempt for a bug.
        
        Args:
            bug_id: Unique identifier for the bug
            action: RepairAction object describing the attempt
        """
        bug = self.db.get_bug(bug_id)
        if bug is None:
            logger.error(f"Bug {bug_id} not found")
            return
        
        bug.repair_attempts += 1
        
        if action.template_id and action.template_id not in bug.templates_used:
            bug.templates_used.append(action.template_id)
        
        if action.agent_id and action.agent_id not in bug.repair_agents:
            bug.repair_agents.append(action.agent_id)
        
        # Update template statistics if applicable
        if action.template_id and action.successful:
            template = self.db.get_template(action.template_id)
            if template:
                # Update template success rate with exponential moving average
                alpha = 0.1  # Smoothing factor
                template.success_rate = (1 - alpha) * template.success_rate + alpha * (1.0 if action.successful else 0.0)
                
                # Update average repair time
                if action.duration is not None:
                    if template.usage_count > 0:
                        template.avg_repair_time = (template.avg_repair_time * template.usage_count + action.duration) / (template.usage_count + 1)
                    else:
                        template.avg_repair_time = action.duration
                
                template.usage_count += 1
                template.last_used = time.time()
                
                # Save template updates
                self.db.add_template(template)
        
        # Save action and bug updates
        self.db.add_action(action)
        self.db.add_bug(bug)
        self.db.save()
        
        logger.info(f"Recorded repair attempt for bug {bug_id} (success: {action.successful})")
    
    def end_repair(self, bug_id: str, successful: bool) -> None:
        """
        End the repair phase for a bug.
        
        Args:
            bug_id: Unique identifier for the bug
            successful: Whether the repair was successful
        """
        bug = self.db.get_bug(bug_id)
        if bug is None:
            logger.error(f"Bug {bug_id} not found")
            return
        
        bug.repair_end_time = time.time()
        if bug.repair_start_time is not None:
            bug.repair_time = bug.repair_end_time - bug.repair_start_time
        
        self.db.save()
        
        logger.info(f"Completed repair for bug {bug_id} (success: {successful})")
    
    def start_verification(self, bug_id: str) -> None:
        """
        Start the verification phase for a bug.
        
        Args:
            bug_id: Unique identifier for the bug
        """
        bug = self.db.get_bug(bug_id)
        if bug is None:
            logger.error(f"Bug {bug_id} not found")
            return
        
        bug.verification_start_time = time.time()
        self.db.save()
        
        logger.info(f"Started verification for bug {bug_id}")
    
    def end_verification(self, bug_id: str, successful: bool) -> None:
        """
        End the verification phase for a bug.
        
        Args:
            bug_id: Unique identifier for the bug
            successful: Whether the verification was successful
        """
        bug = self.db.get_bug(bug_id)
        if bug is None:
            logger.error(f"Bug {bug_id} not found")
            return
        
        bug.verification_end_time = time.time()
        self.db.save()
        
        logger.info(f"Completed verification for bug {bug_id} (success: {successful})")
    
    def resolve_bug(self, bug_id: str, successful: bool) -> None:
        """
        Mark a bug as resolved.
        
        Args:
            bug_id: Unique identifier for the bug
            successful: Whether the bug was successfully resolved
        """
        bug = self.db.get_bug(bug_id)
        if bug is None:
            logger.error(f"Bug {bug_id} not found")
            return
        
        bug.resolved_time = time.time()
        bug.successful = successful
        
        # Calculate total time
        if bug.detected_time is not None:
            bug.total_time = bug.resolved_time - bug.detected_time
        
        self.db.save()
        
        # Log MTTR for this bug
        if bug.total_time is not None:
            mttr_minutes = bug.total_time / 60.0
            logger.info(f"Bug {bug_id} resolved (success: {successful}) with MTTR of {mttr_minutes:.2f} minutes")
            
            # Check if we met target MTTR
            if mttr_minutes <= self.target_mttr / 60.0:
                logger.info(f"✓ Bug {bug_id} met target MTTR of {self.target_mttr / 60.0} minutes")
            else:
                logger.warning(f"✗ Bug {bug_id} exceeded target MTTR of {self.target_mttr / 60.0} minutes")
        else:
            logger.info(f"Bug {bug_id} resolved (success: {successful})")
        
        # Update statistics
        self.update_statistics()
    
    def find_matching_templates(self, error_message: str, language: str = "python") -> List[RepairTemplate]:
        """
        Find repair templates that match an error message.
        
        Args:
            error_message: Error message to match against templates
            language: Programming language (used for filtering templates)
            
        Returns:
            List of matching RepairTemplate objects
        """
        import re
        matching_templates = []
        
        # First get templates for the specified language
        language_templates = self.db.get_templates_by_tag(language)
        
        # Check each template for pattern match
        for template in language_templates:
            pattern = re.compile(template.bug_pattern)
            if pattern.search(error_message):
                matching_templates.append(template)
        
        # Sort by success rate (descending)
        matching_templates.sort(key=lambda t: t.success_rate, reverse=True)
        
        return matching_templates
    
    def apply_template(self, template: RepairTemplate, context: Dict[str, Any], file_path: str) -> Tuple[bool, str]:
        """
        Apply a repair template to fix a bug.
        
        Args:
            template: RepairTemplate to apply
            context: Dictionary of context variables to use in the template
            file_path: Path to the file to modify
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Read the file
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Format the fix pattern with the context
            fix = template.fix_pattern.format(**context)
            
            # Apply the fix (this is a simplified implementation)
            # In a real system, this would be more sophisticated
            if '{insert_line}' in context:
                line_num = context['insert_line']
                lines = content.splitlines()
                lines.insert(line_num - 1, fix)
                new_content = '\n'.join(lines)
            elif '{replace_line}' in context:
                line_num = context['replace_line']
                lines = content.splitlines()
                lines[line_num - 1] = fix
                new_content = '\n'.join(lines)
            else:
                # Append to the end of the file
                new_content = content + '\n\n' + fix
            
            # Write the modified content back to the file
            with open(file_path, 'w') as f:
                f.write(new_content)
            
            return True, f"Applied template {template.id} to {file_path}"
        except Exception as e:
            return False, f"Error applying template: {str(e)}"
    
    def update_statistics(self) -> None:
        """Update MTTR statistics."""
        now = time.time()
        
        # Calculate statistics for different time periods
        for period_name, seconds in [
            ("day", 24 * 60 * 60),
            ("week", 7 * 24 * 60 * 60),
            ("month", 30 * 24 * 60 * 60),
            ("all", float('inf'))
        ]:
            # Get bugs in the time period
            if period_name == "all":
                bugs = list(self.db.bugs.values())
            else:
                bugs = [bug for bug in self.db.bugs.values() if now - bug.detected_time <= seconds]
            
            # Only consider resolved bugs for MTTR calculation
            resolved_bugs = [bug for bug in bugs if bug.resolved_time is not None and bug.total_time is not None]
            
            if not resolved_bugs:
                continue
            
            # Calculate MTTR statistics
            mttrs = [bug.total_time for bug in resolved_bugs]
            mttrs.sort()
            
            avg_mttr = sum(mttrs) / len(mttrs)
            median_mttr = statistics.median(mttrs)
            min_mttr = min(mttrs)
            max_mttr = max(mttrs)
            
            # Calculate 90th percentile
            p90_index = int(len(mttrs) * 0.9)
            p90_mttr = mttrs[p90_index]
            
            # Calculate success rate
            success_count = sum(1 for bug in resolved_bugs if bug.successful)
            success_rate = success_count / len(resolved_bugs)
            
            # Calculate MTTR by severity
            by_severity = {}
            for severity in set(bug.severity for bug in resolved_bugs):
                severity_bugs = [bug for bug in resolved_bugs if bug.severity == severity]
                if severity_bugs:
                    by_severity[severity] = sum(bug.total_time for bug in severity_bugs) / len(severity_bugs)
            
            # Create stats object
            stats = MTTRStats(
                time_period=period_name,
                avg_mttr=avg_mttr,
                median_mttr=median_mttr,
                min_mttr=min_mttr,
                max_mttr=max_mttr,
                p90_mttr=p90_mttr,
                total_bugs=len(bugs),
                resolved_bugs=len(resolved_bugs),
                success_rate=success_rate,
                by_severity=by_severity
            )
            
            # Add to database
            self.db.add_stats(stats)
        
        # Save the database
        self.db.save()
        
        # Log current MTTR
        latest_stats = self.db.get_latest_stats("day")
        if latest_stats:
            logger.info(f"Current MTTR: {latest_stats.avg_mttr / 60.0:.2f} minutes (target: {self.target_mttr / 60.0} minutes)")
            
            # Check if we're meeting the target
            if latest_stats.avg_mttr <= self.target_mttr:
                logger.info(f"✓ Meeting target MTTR of {self.target_mttr / 60.0} minutes")
            else:
                logger.warning(f"✗ Exceeding target MTTR of {self.target_mttr / 60.0} minutes by {(latest_stats.avg_mttr - self.target_mttr) / 60.0:.2f} minutes")
    
    def generate_mttr_report(self) -> Dict[str, Any]:
        """
        Generate a report on MTTR performance.
        
        Returns:
            Dictionary containing MTTR statistics and trends
        """
        # Get latest statistics for different time periods
        day_stats = self.db.get_latest_stats("day")
        week_stats = self.db.get_latest_stats("week")
        month_stats = self.db.get_latest_stats("month")
        all_stats = self.db.get_latest_stats("all")
        
        # Get historical data for trend analysis
        all_stats_list = [stat for stat in self.db.stats if stat.time_period == "day"]
        all_stats_list.sort(key=lambda s: s.timestamp)
        
        # Calculate trend (simple linear regression)
        trend = 0.0
        if len(all_stats_list) >= 2:
            x = list(range(len(all_stats_list)))
            y = [stat.avg_mttr for stat in all_stats_list]
            
            # Calculate slope of trend line
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
            sum_xx = sum(x_i * x_i for x_i in x)
            
            # Avoid division by zero
            if n * sum_xx - sum_x * sum_x != 0:
                trend = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        
        # Format report
        report = {
            "timestamp": time.time(),
            "target_mttr": self.target_mttr,
            "current_mttr": day_stats.avg_mttr if day_stats else 0.0,
            "trend": trend,
            "meeting_target": day_stats.avg_mttr <= self.target_mttr if day_stats else False,
            "statistics": {
                "day": asdict(day_stats) if day_stats else None,
                "week": asdict(week_stats) if week_stats else None,
                "month": asdict(month_stats) if month_stats else None,
                "all_time": asdict(all_stats) if all_stats else None
            },
            "bottlenecks": self._identify_bottlenecks(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify bottlenecks in the repair process."""
        bottlenecks = []
        
        # Get resolved bugs
        resolved_bugs = [bug for bug in self.db.bugs.values() 
                        if bug.resolved_time is not None and bug.total_time is not None]
        
        if not resolved_bugs:
            return bottlenecks
        
        # Calculate phase durations
        diagnosis_times = []
        repair_times = []
        verification_times = []
        
        for bug in resolved_bugs:
            # Calculate diagnosis time
            if bug.diagnosis_start_time is not None and bug.diagnosis_end_time is not None:
                diagnosis_time = bug.diagnosis_end_time - bug.diagnosis_start_time
                diagnosis_times.append((bug.bug_id, diagnosis_time))
            
            # Calculate repair time
            if bug.repair_start_time is not None and bug.repair_end_time is not None:
                repair_time = bug.repair_end_time - bug.repair_start_time
                repair_times.append((bug.bug_id, repair_time))
            
            # Calculate verification time
            if bug.verification_start_time is not None and bug.verification_end_time is not None:
                verification_time = bug.verification_end_time - bug.verification_start_time
                verification_times.append((bug.bug_id, verification_time))
        
        # Sort by duration (descending)
        diagnosis_times.sort(key=lambda x: x[1], reverse=True)
        repair_times.sort(key=lambda x: x[1], reverse=True)
        verification_times.sort(key=lambda x: x[1], reverse=True)
        
        # Identify bottlenecks (top 3 in each phase)
        if diagnosis_times:
            bottlenecks.append({
                "phase": "diagnosis",
                "avg_time": sum(t[1] for t in diagnosis_times) / len(diagnosis_times),
                "worst_cases": diagnosis_times[:3]
            })
        
        if repair_times:
            bottlenecks.append({
                "phase": "repair",
                "avg_time": sum(t[1] for t in repair_times) / len(repair_times),
                "worst_cases": repair_times[:3]
            })
        
        if verification_times:
            bottlenecks.append({
                "phase": "verification",
                "avg_time": sum(t[1] for t in verification_times) / len(verification_times),
                "worst_cases": verification_times[:3]
            })
        
        # Sort bottlenecks by average time (descending)
        bottlenecks.sort(key=lambda x: x["avg_time"], reverse=True)
        
        return bottlenecks
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving MTTR."""
        recommendations = []
        
        # Get latest statistics
        day_stats = self.db.get_latest_stats("day")
        
        # Check if we're meeting the target
        if day_stats and day_stats.avg_mttr > self.target_mttr:
            # Calculate how far we are from target
            minutes_over = (day_stats.avg_mttr - self.target_mttr) / 60.0
            recommendations.append(f"Current MTTR exceeds target by {minutes_over:.2f} minutes.")
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks()
            if bottlenecks:
                worst_phase = bottlenecks[0]["phase"]
                recommendations.append(f"Focus on optimizing the {worst_phase} phase, which is the biggest bottleneck.")
            
            # Check repair template usage
            if any(template.usage_count > 0 for template in self.db.templates.values()):
                # Find most successful templates
                successful_templates = sorted(
                    [t for t in self.db.templates.values() if t.usage_count > 0],
                    key=lambda t: t.success_rate * t.usage_count,
                    reverse=True
                )[:3]
                
                if successful_templates:
                    recommendations.append("Use these high-success repair templates more frequently:")
                    for template in successful_templates:
                        recommendations.append(f"  - {template.name}: {template.success_rate * 100:.1f}% success rate")
            
            # Suggest parallelization if there are many bugs
            if len(self.db.bugs) > 10:
                recommendations.append("Consider parallel repair strategies for multiple bugs.")
            
            # Suggest more templates if few exist
            if len(self.db.templates) < 10:
                recommendations.append("Add more repair templates to handle common bug patterns.")
        else:
            recommendations.append("Meeting or exceeding MTTR target. Continue current practices.")
        
        return recommendations

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MTTR Optimizer")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start bug command
    start_parser = subparsers.add_parser("start-bug", help="Start tracking a bug")
    start_parser.add_argument("--id", required=True, help="Bug ID")
    start_parser.add_argument("--title", required=True, help="Bug title")
    start_parser.add_argument("--description", required=True, help="Bug description")
    start_parser.add_argument("--severity", required=True, choices=["critical", "high", "medium", "low"], help="Bug severity")
    
    # Update bug command
    update_parser = subparsers.add_parser("update-bug", help="Update bug status")
    update_parser.add_argument("--id", required=True, help="Bug ID")
    update_parser.add_argument("--phase", required=True, choices=["assign", "diagnosis-start", "diagnosis-end", "repair-start", "repair-end", "verification-start", "verification-end", "resolve"], help="Bug phase")
    update_parser.add_argument("--successful", action="store_true", help="Whether the phase was successful")
    
    # Record action command
    action_parser = subparsers.add_parser("record-action", help="Record a repair action")
    action_parser.add_argument("--bug-id", required=True, help="Bug ID")
    action_parser.add_argument("--action-id", required=True, help="Action ID")
    action_parser.add_argument("--description", required=True, help="Action description")
    action_parser.add_argument("--type", required=True, choices=["template", "agent", "manual"], help="Action type")
    action_parser.add_argument("--template-id", help="Template ID (for template actions)")
    action_parser.add_argument("--agent-id", help="Agent ID (for agent actions)")
    action_parser.add_argument("--successful", action="store_true", help="Whether the action was successful")
    action_parser.add_argument("--result", help="Action result")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate MTTR report")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = MTTROptimizer()
    
    if args.command == "start-bug":
        optimizer.start_bug_lifecycle(args.id, args.title, args.description, args.severity)
        print(f"Started tracking bug {args.id}")
    
    elif args.command == "update-bug":
        if args.phase == "assign":
            optimizer.assign_bug(args.id)
        elif args.phase == "diagnosis-start":
            optimizer.start_diagnosis(args.id)
        elif args.phase == "diagnosis-end":
            optimizer.end_diagnosis(args.id)
        elif args.phase == "repair-start":
            optimizer.start_repair(args.id)
        elif args.phase == "repair-end":
            optimizer.end_repair(args.id, args.successful)
        elif args.phase == "verification-start":
            optimizer.start_verification(args.id)
        elif args.phase == "verification-end":
            optimizer.end_verification(args.id, args.successful)
        elif args.phase == "resolve":
            optimizer.resolve_bug(args.id, args.successful)
        
        print(f"Updated bug {args.id} to phase {args.phase}")
    
    elif args.command == "record-action":
        action = RepairAction(
            action_id=args.action_id,
            bug_id=args.bug_id,
            description=args.description,
            type=args.type,
            template_id=args.template_id,
            agent_id=args.agent_id,
            start_time=time.time(),
            end_time=time.time(),
            duration=0.0,
            successful=args.successful,
            result=args.result
        )
        
        optimizer.record_repair_attempt(args.bug_id, action)
        print(f"Recorded repair action {args.action_id} for bug {args.bug_id}")
    
    elif args.command == "report":
        report = optimizer.generate_mttr_report()
        
        print("\nMTTR Report")
        print("===========")
        print(f"Current MTTR: {report['current_mttr'] / 60.0:.2f} minutes")
        print(f"Target MTTR: {report['target_mttr'] / 60.0:.2f} minutes")
        print(f"Status: {'✓ Meeting target' if report['meeting_target'] else '✗ Exceeding target'}")
        print(f"Trend: {'Improving' if report['trend'] < 0 else 'Worsening' if report['trend'] > 0 else 'Stable'}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"- {rec}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
