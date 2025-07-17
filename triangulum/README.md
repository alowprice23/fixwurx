# Triangulum Engine

The Triangulum Engine is the core execution engine for bug fixing in the FixWurx system. It provides deterministic phase transitions, path-based execution, bug state tracking, and neural matrix integration.

## Directory Structure

```
triangulum/
├── __init__.py              # Package initialization
├── core/                    # Core engine components
│   ├── __init__.py          # Core package initialization
│   ├── engine.py            # Triangulation engine implementation
│   ├── engine_patch.py      # Engine patches and hotfixes
│   └── commands.py          # Command implementations
├── client/                  # Client components
│   ├── __init__.py          # Client package initialization
│   ├── client.py            # Client implementation
│   └── integration.py       # Integration with other components
├── components/              # System components
│   ├── __init__.py          # Components package initialization
│   ├── system_monitor.py    # System monitoring component
│   ├── queue_manager.py     # Queue management component
│   ├── dashboard.py         # Dashboard component
│   ├── rollback_manager.py  # Rollback management component
│   └── plan_executor.py     # Plan execution component
├── daemon/                  # Daemon functionality
│   ├── __init__.py          # Daemon package initialization
│   ├── daemon.py            # Daemon implementation
│   ├── process.py           # Process management
│   └── start.py             # Daemon startup
├── utils/                   # Utilities
│   ├── __init__.py          # Utils package initialization
│   ├── resource_manager.py  # Resource management
│   └── helpers.py           # Helper functions
└── tests/                   # Tests
    ├── __init__.py          # Tests package initialization
    ├── test_engine.py       # Engine tests
    └── test_integration.py  # Integration tests
```

## Modules

### Core

- **engine.py**: Implements the core triangulation engine that handles bug state transitions, paths, and execution.
- **engine_patch.py**: Contains patches and hotfixes for the engine.
- **commands.py**: Implements commands for interacting with the triangulation engine.

### Client

- **client.py**: Provides a client interface for interacting with the triangulation engine.
- **integration.py**: Handles integration with other FixWurx components.

### Components

- **system_monitor.py**: Monitors system health and performance.
- **queue_manager.py**: Manages queues for tasks and jobs.
- **dashboard.py**: Provides a dashboard for visualizing system state.
- **rollback_manager.py**: Handles rollbacks for failed operations.
- **plan_executor.py**: Executes plans for bug fixing.

### Daemon

- **daemon.py**: Implements the daemon for running the triangulation engine as a service.
- **process.py**: Handles process management for the daemon.
- **start.py**: Handles startup of the daemon.

### Utils

- **resource_manager.py**: Manages resources for the triangulation engine.
- **helpers.py**: Provides helper functions for the triangulation engine.

### Tests

- **test_engine.py**: Contains tests for the triangulation engine.
- **test_integration.py**: Contains integration tests for the triangulation engine.

## Usage

The Triangulum Engine is used by the FixWurx system to provide deterministic phase transitions, path-based execution, bug state tracking, and neural matrix integration. It can be used directly through the client interface or through the daemon.

```python
from triangulum.client.client import TriangulumClient

# Create a client
client = TriangulumClient()

# Connect to the daemon
client.connect()

# Execute a command
result = client.execute_command("status")

# Print the result
print(result)
```

## Daemon

The Triangulum Engine can be run as a daemon using the following command:

```bash
python -m triangulum.daemon.start
```

This will start the daemon and make it available for clients to connect to.
