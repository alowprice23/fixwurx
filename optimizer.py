"""
learning/optimizer.py
─────────────────────
**Enhanced adaptive ML tuner** that optimizes multiple parameters:

    ──  `TriangulationEngine.timer_default`  (phase-timer initial value)
    ──  `Planner.exploration_rate`  (balance between exploration/exploitation)
    ──  `Scheduler.info_gain`  (information gain rate for entropy-based scheduling)

It uses a multi-armed bandit approach combined with entropy-based feedback
to maximize per-bug reward and overall system performance.

Design goals
────────────
▪ **No heavyweight ML stack** – online bandit with incremental averages.  
▪ **Safe** – changes are bounded to predefined ranges; never touches proofs.  
▪ **Planner-integrated** – considers feedback from planner agent.
▪ **Entropy-aware** – uses information theory to guide optimization.
▪ **Plug-and-play** – just instantiate with a live `TriangulationEngine`
  and call `push_metric()` every time results are published.

Reward signal
─────────────
reward = 1.0 if success and mean_tokens ≤ 1500
= 0.5 if success and mean_tokens > 1500
= 0.0 if failure


Algorithm
─────────
• ε-greedy bandit over the discrete set {2, 3, 4}.  
• Keeps running average Q(timer) and visit count N(timer).  
• Every `UPDATE_FREQ` episodes, with probability ε choose random timer,
  else exploit best Q.  
• Writes new value into `engine.timer_default` **between bugs** – the State
  Machine picks it up on the next WAIT→REPRO promotion, so no invariants
  break mid-flight.

Public API
──────────
"""
import json
import math
import secrets
import statistics
import time
import hashlib
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Any, Union, Callable
from dataclasses import dataclass

from replay_buffer import ReplayBuffer, CompressedReplayBuffer

# Parameter definitions
class OptimizableParameter:
    """Defines a parameter that can be optimized with valid range and step size."""
    def __init__(
        self, 
        name: str, 
        min_value: Union[int, float], 
        max_value: Union[int, float], 
        step: Union[int, float], 
        initial_value: Union[int, float] = None,
        discrete: bool = False,
        description: str = ""
    ):
        """
        Initialize a parameter definition.
        
        Args:
            name: Parameter name
            min_value: Minimum valid value
            max_value: Maximum valid value
            step: Step size for discrete values or adjustment magnitude
            initial_value: Starting value (defaults to mid-point if None)
            discrete: Whether parameter has discrete values (vs continuous)
            description: Human-readable description
        """
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.discrete = discrete
        self.description = description
        
        # Set initial value (default to midpoint if not specified)
        if initial_value is None:
            if discrete:
                # For discrete parameters, use minimum value as default
                self.initial_value = min_value
            else:
                # For continuous parameters, use midpoint as default
                self.initial_value = min_value + (max_value - min_value) / 2
        else:
            # Ensure initial value is within bounds
            self.initial_value = max(min_value, min(max_value, initial_value))
    
    def get_valid_values(self) -> List[Union[int, float]]:
        """Get list of valid values for discrete parameters."""
        if not self.discrete:
            return []
            
        values = []
        current = self.min_value
        while current <= self.max_value:
            values.append(current)
            current += self.step
        return values
    
    def validate_value(self, value: Union[int, float]) -> Union[int, float]:
        """Validate and adjust a value to be within parameter bounds."""
        if self.discrete:
            # Find closest valid discrete value
            valid_values = self.get_valid_values()
            return min(valid_values, key=lambda x: abs(x - value))
        else:
            # Clamp continuous value to range
            return max(self.min_value, min(self.max_value, value))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "initial_value": self.initial_value,
            "discrete": self.discrete,
            "description": self.description,
            "valid_values": self.get_valid_values() if self.discrete else []
        }


# Standard optimizable parameters
TIMER_PARAMETER = OptimizableParameter(
    name="timer_default",
    min_value=2,
    max_value=4,
    step=1,
    initial_value=3,
    discrete=True,
    description="Phase timer initial value"
)

EXPLORATION_PARAMETER = OptimizableParameter(
    name="exploration_rate",
    min_value=0.05,
    max_value=0.3,
    step=0.05,
    initial_value=0.1,
    discrete=True,
    description="Planner exploration rate"
)

INFO_GAIN_PARAMETER = OptimizableParameter(
    name="info_gain",
    min_value=0.5,
    max_value=2.0,
    step=0.1,
    initial_value=1.0,
    discrete=True,
    description="Information gain per attempt"
)

# Global tunables
TIMER_CANDIDATES = (2, 3, 4)  # Legacy setting
EPSILON = 0.10                # exploration probability
UPDATE_FREQ = 30              # recompute after this many new episodes
BUFFER_CAPACITY = 500         # replay buffer capacity
ENTROPY_WEIGHT = 0.5          # weight for entropy reduction in reward
PATH_COMPLEXITY_WEIGHT = 0.3  # weight for path complexity in parameter adjustment
FALLBACK_PENALTY = 0.2        # penalty when fallbacks are used

T = TypeVar('T')

def secure_choice(sequence: Sequence[T]) -> T:
    """
    Cryptographically secure random choice from a sequence.
    Replacement for random.choice() using the secrets module.
    """
    if not sequence:
        raise IndexError("Cannot choose from an empty sequence")
    return sequence[secrets.randbelow(len(sequence))]


class ParameterState:
    """
    Tracks the optimization state of a single parameter.
    
    This includes Q-values, visit counts, and current value.
    """
    def __init__(
        self, 
        parameter: OptimizableParameter, 
        target_object: Any = None, 
        attribute_name: Optional[str] = None
    ):
        """
        Initialize parameter state.
        
        Args:
            parameter: Parameter definition
            target_object: Object to update when parameter changes
            attribute_name: Attribute name on target object (defaults to parameter.name)
        """
        self.parameter = parameter
        self.target_object = target_object
        self.attribute_name = attribute_name or parameter.name
        
        # Initialize Q-values and visit counts
        self.valid_values = parameter.get_valid_values() if parameter.discrete else []
        self.q_values = {v: 0.0 for v in self.valid_values} if parameter.discrete else {}
        self.visit_counts = {v: 0 for v in self.valid_values} if parameter.discrete else {}
        
        # Current value
        self.current_value = parameter.initial_value
        self.last_update_time = time.time()
        
        # Apply initial value to target object if specified
        if target_object is not None:
            try:
                setattr(target_object, self.attribute_name, self.current_value)
            except (AttributeError, ValueError, TypeError) as e:
                print(f"[optimizer] Warning: Failed to set {self.attribute_name} on target object: {e}")
    
    def update_q_value(self, value: Union[int, float], reward: float) -> None:
        """
        Update Q-value for a parameter value.
        
        Args:
            value: Parameter value
            reward: Reward received
        """
        if self.parameter.discrete:
            # For discrete parameters, update exact value
            value = self.parameter.validate_value(value)
            if value not in self.q_values:
                self.q_values[value] = 0.0
                self.visit_counts[value] = 0
            
            self.visit_counts[value] += 1
            alpha = 1.0 / self.visit_counts[value]
            self.q_values[value] += alpha * (reward - self.q_values[value])
        else:
            # For continuous parameters, this would be more complex
            # For simplicity, we're focusing on discrete parameters in this implementation
            pass
    
    def set_value(self, value: Union[int, float]) -> bool:
        """
        Set parameter value and update target object.
        
        Args:
            value: New parameter value
            
        Returns:
            Whether the value was successfully applied
        """
        validated_value = self.parameter.validate_value(value)
        if validated_value == self.current_value:
            return True  # No change needed
        
        self.current_value = validated_value
        self.last_update_time = time.time()
        
        # Apply to target object if specified
        if self.target_object is not None:
            try:
                setattr(self.target_object, self.attribute_name, validated_value)
                return True
            except (AttributeError, ValueError, TypeError) as e:
                print(f"[optimizer] Warning: Failed to set {self.attribute_name} on target object: {e}")
                return False
        
        return True
    
    def choose_value(self, epsilon: float = EPSILON) -> Union[int, float]:
        """
        Choose next parameter value using epsilon-greedy policy.
        
        Args:
            epsilon: Exploration probability
            
        Returns:
            Chosen parameter value
        """
        if not self.parameter.discrete or not self.valid_values:
            return self.current_value
        
        # Use secure random number generation
        if (secrets.randbelow(100) / 100.0) < epsilon:
            # Explore - choose random value
            return secure_choice(self.valid_values)
        else:
            # Exploit - choose best value
            if not self.q_values:
                return self.current_value
                
            best_q = max(self.q_values.values())
            best_values = [v for v, q in self.q_values.items() if q == best_q]
            
            if self.parameter.name == "timer_default":
                # For timer, prefer smaller values (faster) on ties
                return min(best_values)
            else:
                # For other parameters, prefer values closer to current on ties
                return min(best_values, key=lambda v: abs(v - self.current_value))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parameter": self.parameter.to_dict(),
            "current_value": self.current_value,
            "q_values": {str(k): v for k, v in self.q_values.items()},
            "visit_counts": {str(k): v for k, v in self.visit_counts.items()},
            "last_update_time": self.last_update_time
        }


@dataclass
class PlannerPathData:
    """Data about a planner path for optimization purposes."""
    path_id: str
    bug_id: str
    complexity: float = 0.0  # Normalized complexity (0.0-1.0)
    success: bool = False
    fallback_used: bool = False
    execution_time: float = 0.0
    num_actions: int = 0
    num_dependencies: int = 0
    
    @staticmethod
    def from_path(path: Dict[str, Any]) -> 'PlannerPathData':
        """Create from a planner path dictionary."""
        # Extract actions and dependencies if available
        actions = path.get("actions", [])
        dependencies = path.get("dependencies", [])
        
        # Calculate complexity based on actions and dependencies
        action_complexity = min(len(actions) / 10.0, 1.0)  # Cap at 1.0
        dependency_complexity = min(len(dependencies) / 5.0, 1.0)
        complexity = (action_complexity + dependency_complexity) / 2.0
        
        return PlannerPathData(
            path_id=path.get("path_id", "unknown"),
            bug_id=path.get("bug_id", "unknown"),
            complexity=complexity,
            success=path.get("success", False),
            fallback_used=path.get("fallback_used", False),
            execution_time=path.get("execution_time", 0.0),
            num_actions=len(actions),
            num_dependencies=len(dependencies)
        )


class EnhancedAdaptiveOptimizer:
    """
    Enhanced multi-parameter optimizer with planner integration and entropy awareness.
    
    Features:
    - Optimizes multiple parameters simultaneously
    - Integrates with planner agent for strategic parameter adjustment
    - Uses entropy reduction as part of reward signal
    - Provides more sophisticated optimization strategies
    - Persists learned parameter values
    - Considers path complexity in optimization decisions
    - Adjusts parameters based on planner's historical data
    - Handles fallback strategies in parameter selection
    """
    
    def __init__(
        self, 
        engine, 
        planner=None, 
        scheduler=None, 
        *,
        buffer_capacity: int = BUFFER_CAPACITY,
        storage_path: Union[str, Path] = ".triangulum/optimizer_state.json",
        epsilon: float = EPSILON,
        update_freq: int = UPDATE_FREQ,
        entropy_weight: float = ENTROPY_WEIGHT
    ) -> None:
        """
        Initialize enhanced optimizer.
        
        Args:
            engine: TriangulationEngine instance
            planner: Optional planner agent
            scheduler: Optional scheduler
            buffer_capacity: Replay buffer capacity
            storage_path: Path to save optimizer state
            epsilon: Exploration probability
            update_freq: How often to update parameters
            entropy_weight: Weight for entropy reduction in reward calculation
        """
        self.engine = engine
        self.planner = planner
        self.scheduler = scheduler
        self.storage_path = Path(storage_path)
        self.epsilon = epsilon
        self.update_freq = update_freq
        self.entropy_weight = entropy_weight
        
        # Use compressed replay buffer for efficiency
        self.buffer = CompressedReplayBuffer(buffer_capacity)
        
        # Parameter states
        self.parameters: Dict[str, ParameterState] = {}
        
        # Counter for updates
        self.updates_since_change = 0
        self.total_updates = 0
        self.last_entropy_values: Dict[str, float] = {}
        
        # Planner-specific tracking
        self.path_history: Dict[str, PlannerPathData] = {}  # path_id -> data
        self.bug_paths: Dict[str, List[str]] = {}  # bug_id -> list of path_ids
        self.path_complexity_stats = {
            "mean": 0.0,
            "max": 0.0,
            "min": 1.0,
            "total_paths": 0
        }
        self.planner_callbacks: Dict[str, Callable] = {}
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Load previous state if available
        self._load_state()
    
    def _initialize_parameters(self) -> None:
        """Initialize parameter states with targets."""
        # Timer parameter for engine
        self.parameters[TIMER_PARAMETER.name] = ParameterState(
            TIMER_PARAMETER, self.engine, "timer_default"
        )
        
        # Exploration rate for planner (if available)
        if self.planner is not None:
            self.parameters[EXPLORATION_PARAMETER.name] = ParameterState(
                EXPLORATION_PARAMETER, self.planner, "exploration_rate"
            )
        
        # Info gain for scheduler (if available)
        if self.scheduler is not None:
            self.parameters[INFO_GAIN_PARAMETER.name] = ParameterState(
                INFO_GAIN_PARAMETER, self.scheduler, "info_gain"
            )
    
    def _load_state(self) -> None:
        """Load optimizer state from disk if available."""
        if not self.storage_path.exists():
            return
            
        try:
            with open(self.storage_path, 'r') as f:
                state = json.load(f)
                
            # Load parameter states
            for param_name, param_data in state.get('parameters', {}).items():
                if param_name in self.parameters:
                    param_state = self.parameters[param_name]
                    
                    # Restore Q-values and visit counts
                    param_state.q_values = {float(k): v for k, v in param_data.get('q_values', {}).items()}
                    param_state.visit_counts = {float(k): v for k, v in param_data.get('visit_counts', {}).items()}
                    
                    # Set current value
                    current_value = param_data.get('current_value', param_state.current_value)
                    param_state.set_value(current_value)
            
            # Load global state
            self.total_updates = state.get('total_updates', 0)
            
            print(f"[optimizer] Loaded state from {self.storage_path}")
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"[optimizer] Warning: Failed to load state: {e}")
    
    def _save_state(self) -> None:
        """Save optimizer state to disk."""
        # Create parent directory if needed
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            state = {
                'parameters': {name: param.to_dict() for name, param in self.parameters.items()},
                'total_updates': self.total_updates,
                'last_updated': time.time()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(state, f, indent=2)
        except (IOError, OSError) as e:
            print(f"[optimizer] Warning: Failed to save state: {e}")
    
    def push_metric(self, metric: Dict[str, Any]) -> None:
        """
        Process a new metric and update parameters.
        
        Args:
            metric: Dictionary with metric data including success, mean_tokens,
                   optional reward, and optional entropy values
        """
        # Add to replay buffer
        self.buffer.add(metric)
        self.updates_since_change += 1
        self.total_updates += 1
        
        # Extract bug_id if available
        bug_id = metric.get('bug_id', 'unknown')
        
        # Calculate reward
        reward = self._calculate_reward(metric, bug_id)
        
        # Update Q-values for current parameter values
        for param_name, param_state in self.parameters.items():
            param_state.update_q_value(param_state.current_value, reward)
        
        # Possibly update parameters
        if self.updates_since_change >= self.update_freq:
            self.updates_since_change = 0
            self._update_parameters()
            self._save_state()
    
    def _calculate_reward(self, metric: Dict[str, Any], bug_id: str) -> float:
        """
        Calculate reward from metric.
        
        Args:
            metric: Metric dictionary
            bug_id: Bug identifier
            
        Returns:
            Calculated reward
        """
        # Base reward from success/tokens
        base_reward = metric.get("reward")
        if base_reward is None:
            success = metric.get("success", False)
            mean_tokens = metric.get("mean_tokens", float("inf"))
            if success and mean_tokens <= 1500:
                base_reward = 1.0
            elif success:
                base_reward = 0.5
            else:
                base_reward = 0.0
        
        # Entropy component (if available)
        entropy_reward = 0.0
        if 'entropy' in metric and bug_id in self.last_entropy_values:
            # Calculate entropy reduction
            current_entropy = metric['entropy']
            last_entropy = self.last_entropy_values[bug_id]
            reduction = max(0, last_entropy - current_entropy)
            
            # Normalize and weight entropy reduction
            entropy_reward = min(1.0, reduction) * self.entropy_weight
            
            # Update last entropy value
            self.last_entropy_values[bug_id] = current_entropy
        elif 'entropy' in metric:
            # First occurrence of this bug, just store entropy
            self.last_entropy_values[bug_id] = metric['entropy']
        
        # Path complexity and fallback component (if available from planner)
        planner_reward = 0.0
        
        # Check for path data
        path_id = metric.get("path_id")
        if path_id and path_id in self.path_history:
            path_data = self.path_history[path_id]
            
            # Adjust based on path complexity - more complex paths that succeed get higher reward
            if path_data.success:
                planner_reward += path_data.complexity * PATH_COMPLEXITY_WEIGHT
                
            # Penalize for fallback use
            if path_data.fallback_used:
                planner_reward -= FALLBACK_PENALTY
        
        # Process planner-specific metrics if available
        if "planner" in metric:
            planner_metrics = metric["planner"]
            
            # Reward high successful_fixes rate
            if "successful_fixes" in planner_metrics and "failed_fixes" in planner_metrics:
                successful = planner_metrics["successful_fixes"]
                failed = planner_metrics["failed_fixes"]
                total = successful + failed
                if total > 0:
                    success_rate = successful / total
                    planner_reward += success_rate * 0.2  # Small boost for good success rate
            
            # Penalize high fallbacks_used
            if "fallbacks_used" in planner_metrics and "paths_generated" in planner_metrics:
                fallbacks = planner_metrics["fallbacks_used"]
                paths = planner_metrics["paths_generated"]
                if paths > 0:
                    fallback_rate = fallbacks / paths
                    planner_reward -= fallback_rate * 0.2  # Small penalty for high fallback rate
        
        # Combined reward
        return base_reward + entropy_reward + planner_reward
    
    def _update_parameters(self) -> None:
        """Update all parameters using current policy."""
        changes = []
        
        # Adapt epsilon based on planner metrics if available
        adapted_epsilon = self.epsilon
        if self.planner is not None and hasattr(self.planner, "get_metrics"):
            planner_metrics = self.planner.get_metrics()
            
            # If fallbacks are being used frequently, increase exploration
            if "fallbacks_used" in planner_metrics and "paths_generated" in planner_metrics:
                fallbacks = planner_metrics.get("fallbacks_used", 0)
                paths = planner_metrics.get("paths_generated", 1)  # Avoid division by zero
                if paths > 0:
                    fallback_rate = fallbacks / paths
                    if fallback_rate > 0.3:  # High fallback rate
                        adapted_epsilon = min(self.epsilon * 1.5, 0.3)  # Increase exploration but cap it
        
        for param_name, param_state in self.parameters.items():
            # Choose new value - use adaptive epsilon
            new_value = param_state.choose_value(adapted_epsilon)
            
            # Apply path complexity adjustments for specific parameters
            if param_name == "exploration_rate" and self.path_complexity_stats["total_paths"] > 0:
                # Adjust exploration rate based on average path complexity
                mean_complexity = self.path_complexity_stats["mean"]
                if mean_complexity > 0.7:  # Complex paths
                    # Increase exploration for complex paths
                    complexity_adjustment = mean_complexity * 0.05  # Max +5% to exploration
                    new_value = min(new_value + complexity_adjustment, EXPLORATION_PARAMETER.max_value)
            
            # Apply if different
            if new_value != param_state.current_value:
                old_value = param_state.current_value
                if param_state.set_value(new_value):
                    changes.append((param_name, old_value, new_value))
        
        # Log changes
        for param_name, old_value, new_value in changes:
            param_state = self.parameters[param_name]
            print(
                f"[optimizer] {param_name} updated: {old_value} → {new_value} "
                f"(Q={param_state.q_values.get(new_value, 0.0):.3f}, "
                f"N={param_state.visit_counts.get(new_value, 0)})"
            )
            
        # Notify planner of parameter changes if registered
        if changes and "on_parameters_updated" in self.planner_callbacks:
            # Create parameters dict to pass to callback
            updated_params = {
                name: self.parameters[name].current_value 
                for name in self.parameters
            }
            try:
                self.planner_callbacks["on_parameters_updated"](updated_params)
            except Exception as e:
                print(f"[optimizer] Error in planner callback: {e}")
    
    def set_planner(self, planner) -> None:
        """
        Set or update planner reference.
        
        Args:
            planner: Planner agent
        """
        self.planner = planner
        
        # Initialize or update planner parameter
        if EXPLORATION_PARAMETER.name not in self.parameters and planner is not None:
            self.parameters[EXPLORATION_PARAMETER.name] = ParameterState(
                EXPLORATION_PARAMETER, planner, "exploration_rate"
            )
        elif planner is not None:
            self.parameters[EXPLORATION_PARAMETER.name].target_object = planner
            
        # Register for planner callbacks if supported
        if planner is not None:
            # Try to register callbacks
            if hasattr(planner, "register_optimizer_callback"):
                try:
                    planner.register_optimizer_callback(
                        "path_completed", 
                        self.handle_path_completion
                    )
                except (AttributeError, TypeError) as e:
                    print(f"[optimizer] Warning: Failed to register path completion callback: {e}")
    
    def set_scheduler(self, scheduler) -> None:
        """
        Set or update scheduler reference.
        
        Args:
            scheduler: Scheduler
        """
        self.scheduler = scheduler
        
        # Initialize or update scheduler parameter
        if INFO_GAIN_PARAMETER.name not in self.parameters and scheduler is not None:
            self.parameters[INFO_GAIN_PARAMETER.name] = ParameterState(
                INFO_GAIN_PARAMETER, scheduler, "info_gain"
            )
        elif scheduler is not None:
            self.parameters[INFO_GAIN_PARAMETER.name].target_object = scheduler
    
    def get_parameter_value(self, param_name: str) -> Optional[Union[int, float]]:
        """
        Get current value of a parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Current parameter value or None if not found
        """
        if param_name in self.parameters:
            return self.parameters[param_name].current_value
        return None
    
    def set_parameter_value(self, param_name: str, value: Union[int, float]) -> bool:
        """
        Manually set a parameter value.
        
        Args:
            param_name: Parameter name
            value: New parameter value
            
        Returns:
            Whether the value was successfully set
        """
        if param_name in self.parameters:
            return self.parameters[param_name].set_value(value)
        return False
    
    def register_planner_callback(self, event_name: str, callback: Callable) -> bool:
        """
        Register a callback function for a planner event.
        
        Args:
            event_name: Event name
            callback: Callback function
            
        Returns:
            Whether registration was successful
        """
        if not callable(callback):
            return False
            
        self.planner_callbacks[event_name] = callback
        return True
        
    def handle_path_completion(self, path_data: Dict[str, Any]) -> None:
        """
        Handle path completion event from planner.
        
        Args:
            path_data: Data about the completed path
        """
        # Extract path ID and bug ID
        path_id = path_data.get("path_id")
        bug_id = path_data.get("bug_id")
        
        if not path_id or not bug_id:
            return
            
        # Create path data object
        path_obj = PlannerPathData.from_path(path_data)
        
        # Store in path history
        self.path_history[path_id] = path_obj
        
        # Update bug -> paths mapping
        if bug_id not in self.bug_paths:
            self.bug_paths[bug_id] = []
        if path_id not in self.bug_paths[bug_id]:
            self.bug_paths[bug_id].append(path_id)
            
        # Update complexity stats
        total_paths = self.path_complexity_stats["total_paths"]
        if total_paths == 0:
            # First path
            self.path_complexity_stats["mean"] = path_obj.complexity
            self.path_complexity_stats["max"] = path_obj.complexity
            self.path_complexity_stats["min"] = path_obj.complexity
            self.path_complexity_stats["total_paths"] = 1
        else:
            # Update running stats
            new_total = total_paths + 1
            old_mean = self.path_complexity_stats["mean"]
            new_mean = ((old_mean * total_paths) + path_obj.complexity) / new_total
            
            self.path_complexity_stats["mean"] = new_mean
            self.path_complexity_stats["max"] = max(self.path_complexity_stats["max"], path_obj.complexity)
            self.path_complexity_stats["min"] = min(self.path_complexity_stats["min"], path_obj.complexity)
            self.path_complexity_stats["total_paths"] = new_total
            
    def verify_state_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the optimizer's state using secure hashing.
        
        Returns:
            Dictionary with verification results
        """
        # Create a verification report
        report = {
            "timestamp": time.time(),
            "verified": True,
            "details": {}
        }
        
        try:
            # Create a serializable state representation
            state = {
                "parameters": {name: param.to_dict() for name, param in self.parameters.items()},
                "buffer_size": len(self.buffer),
                "total_updates": self.total_updates,
                "updates_since_change": self.updates_since_change,
                "entropy_tracking": {str(k): v for k, v in self.last_entropy_values.items()},
                "path_complexity_stats": self.path_complexity_stats,
            }
            
            # Compute hashes for each component
            for key, value in state.items():
                if isinstance(value, dict):
                    # Hash each component separately
                    component_hash = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                    report["details"][key] = {"hash": component_hash}
                else:
                    # For simple values, include directly
                    report["details"][key] = {"value": value}
            
            # Create overall hash
            full_hash = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
            report["full_hash"] = full_hash
            
        except Exception as e:
            report["verified"] = False
            report["error"] = str(e)
            
        return report
    
    def summary(self) -> Dict[str, Any]:
        """Get optimizer summary."""
        return {
            "parameters": {name: param.to_dict() for name, param in self.parameters.items()},
            "buffer_size": len(self.buffer),
            "total_updates": self.total_updates,
            "updates_since_change": self.updates_since_change,
            "entropy_tracking": dict(self.last_entropy_values),
            "path_complexity_stats": self.path_complexity_stats,
            "planner_integration": {
                "planner_available": self.planner is not None,
                "tracked_paths": len(self.path_history),
                "tracked_bugs": len(self.bug_paths),
                "registered_callbacks": list(self.planner_callbacks.keys())
            },
            "integrity": self.verify_state_integrity()["verified"]
        }


# Legacy compatibility
class AdaptiveOptimizer(EnhancedAdaptiveOptimizer):
    """Legacy compatibility wrapper for EnhancedAdaptiveOptimizer."""
    
    def __init__(self, engine, *, buffer_capacity: int = BUFFER_CAPACITY) -> None:
        """Initialize with legacy interface."""
        super().__init__(engine, buffer_capacity=buffer_capacity)
