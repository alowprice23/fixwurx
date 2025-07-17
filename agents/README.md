# Agent System

The Agent System provides a family tree of specialized agents for debugging and fixing issues, with the Meta Agent overseeing all operations and the Planner Agent acting as the root coordinator. The system also includes the Auditor Agent for monitoring and quality assurance.

## Directory Structure

```
agents/
├── __init__.py                # Package initialization
├── core/                      # Core agent classes
│   ├── __init__.py            # Core package initialization
│   ├── agent_system.py        # Main agent system API
│   ├── coordinator.py         # Agent coordination
│   ├── handoff.py             # Agent handoff mechanism
│   ├── launchpad/             # Launchpad Agent 
│   │   ├── __init__.py        # Launchpad package initialization
│   │   └── agent.py           # Launchpad Agent implementation
│   ├── meta_agent.py          # Meta Agent implementation
│   └── planner_agent.py       # Planner Agent implementation
├── specialized/               # Specialized agent classes
│   ├── __init__.py            # Specialized package initialization
│   └── specialized_agents.py  # Observer, Analyst, and Verifier agents
├── auditor/                   # Auditor agent classes
│   ├── __init__.py            # Auditor package initialization
│   ├── agent.py               # Auditor Agent implementation
│   ├── sensors/               # Sensor components for monitoring
│   │   ├── __init__.py        # Sensors package initialization
│   │   └── activity_sensor.py # Agent activity sensor
│   └── interface/             # Integration with shell and other components
│       ├── __init__.py        # Interface package initialization
│       ├── commands.py        # Auditor command handlers
│       └── shell_integration.py # Shell integration for auditor
├── interface/                 # Interface with other components
│   ├── __init__.py            # Interface package initialization
│   ├── commands.py            # Shell commands for agent interactions
│   └── shell_integration.py   # Integration with the FixWurx shell
├── tests/                     # Test modules
│   ├── __init__.py            # Tests package initialization
│   └── test_meta_agent.py     # Test suite for the Meta Agent
└── utils/                     # Utility modules
    ├── __init__.py            # Utils package initialization
    └── memory.py              # Agent memory implementation
```

## Agent Types

### Core Agents

1. **Launchpad Agent** (`agents/core/launchpad/agent.py`)
   - Bootstraps and initializes the entire FixWurx system
   - Starting point for all system operations
   - Ensures proper initialization of other agents and components
   - Manages component lifecycle (startup, shutdown, restart)

2. **Meta Agent** (`agents/core/meta_agent.py`)
   - Provides oversight and coordination across all agents
   - Manages agent registration and communication
   - Monitors agent health and performance

3. **Planner Agent** (`agents/core/planner_agent.py`)
   - Coordinates the overall debugging strategy
   - Manages solution path generation and selection
   - Orchestrates the work of specialized agents

4. **Auditor Agent** (`agents/auditor/agent.py`)
   - Monitors system components and activities
   - Logs and reports system events and errors
   - Enforces quality assurance standards
   - Collects metrics and performance data

### Specialized Agents

Located in `agents/specialized/specialized_agents.py`:

1. **Observer Agent**
   - Analyzes bugs and reproduces issues
   - Gathers information about bug context and behavior
   - Documents bug behavior and impact

2. **Analyst Agent**
   - Generates patches and solutions
   - Analyzes code to identify root causes
   - Creates fix implementations

3. **Verifier Agent**
   - Tests fixes and ensures correctness
   - Verifies that patches resolve the bugs
   - Runs regression tests to ensure no new issues

## Integration

The Agent System integrates with other FixWurx components through:

1. **Shell Integration** (`agents/interface/shell_integration.py`)
   - Registers agent commands with the shell
   - Provides the interface between agents and the shell environment

2. **Command Handlers** (`agents/interface/commands.py`)
   - Implements command handlers for agent operations
   - Provides a command-line interface for interacting with agents

## Utilities

1. **Agent Memory** (`agents/utils/memory.py`)
   - Provides memory capabilities for agents
   - Stores and retrieves agent state and information

## Auditor

1. **Sensor System** (`agents/auditor/sensors/`)
   - Activity Sensor: Monitors agent activities and system events
   - Collects telemetry and performance metrics
   - Detects anomalies and potential issues

2. **Interface Components** (`agents/auditor/interface/`)
   - Command Handlers: Provides shell commands for auditing functionality
   - Shell Integration: Connects auditor to the shell environment

## Usage

### Initialization

The Agent System is automatically initialized when the FixWurx shell environment starts. It can also be initialized programmatically:

```python
from agents.core.agent_system import get_instance

# Get the Agent System instance
agent_system = get_instance()

# Initialize the system
agent_system.initialize()
```

### Basic Operations

```python
# Create a bug
bug_info = agent_system.create_bug(
    "login-bug", 
    "User cannot log in with correct credentials",
    "The login form returns an error even when valid credentials are provided",
    "high", 
    ["critical", "security"]
)

# Analyze the bug
analysis = agent_system.analyze_bug("login-bug")

# Generate solution paths
paths = agent_system.generate_solution_paths("login-bug")

# Select the best path
best_path = agent_system.select_solution_path("login-bug")

# Generate a patch
patch = agent_system.generate_patch("login-bug")

# Verify the patch
verification = agent_system.verify_patch("login-bug", patch["patch_id"])

# Fix the bug (performs all steps)
result = agent_system.fix_bug("login-bug")

# Shutdown
agent_system.shutdown()
```

### Shell Commands

The Agent System provides various shell commands through the FixWurx shell:

```
# Agent management
fx> agent status
fx> agent tree

# Bug management
fx> bug create login-bug "User cannot log in with correct credentials"
fx> bug list
fx> bug show login-bug

# Solution path management
fx> plan generate login-bug
fx> plan select login-bug

# Agent interactions
fx> observe analyze login-bug
fx> analyze patch login-bug
fx> verify test login-bug
```

## Testing

The Agent System includes test modules for verifying the functionality of agents:

```
python -m unittest agents.tests.test_meta_agent
```

## Configuration

The Agent System can be configured through the FixWurx configuration system:

```yaml
agent_system:
  meta_agent:
    oversight_interval: 60
    log_level: INFO
  planner_agent:
    planning_depth: 3
    path_scoring_weights:
      confidence: 0.3
      complexity: 0.2
      estimated_time: 0.1
      success_rate: 0.4
  observer_agent:
    analysis_timeout: 120
  analyst_agent:
    patch_generation_timeout: 300
  verifier_agent:
    test_timeout: 180
