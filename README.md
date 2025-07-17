# FixWurx

A comprehensive system for automated bug detection, diagnosis, and repair with advanced intent classification.

## Overview

FixWurx is an intelligent system designed to streamline the software development and maintenance process. It combines neural networks, agent-based architectures, and natural language processing to create a powerful platform for detecting, diagnosing, and fixing software issues.

## Key Components

- **Intent Classification System**: Natural language understanding for developer commands
- **Neural Matrix**: Pattern recognition for bug detection and solution recommendation
- **Agent System**: Specialized agents for different tasks in the bug fixing workflow
- **Triangulum**: Distributed processing and communication framework
- **Shell Interface**: Interactive command-line interface for developers

## Architecture

The system follows a modular architecture with the following main components:

- Core system components (shell, launchpad, command execution)
- Neural processing modules (neural matrix, pattern recognition)
- Agent-based workflow (auditor, planner, executor agents)
- Integration layer (triangulum client, intent classification)

## Intent Classification

The advanced intent classification system allows developers to interact with FixWurx using natural language. The system:

1. Interprets developer intent from natural language queries
2. Routes the request to the appropriate system component
3. Provides context-aware assistance and suggestions
4. Learns from interactions to improve future responses

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

### Installation

```bash
# Clone the repository
git clone https://github.com/alowprice23/fixwurx.git
cd fixwurx

# Install dependencies
pip install -r requirements.txt

# Run the system
python start_fixwurx_with_intent.py
```

## Usage

FixWurx provides multiple interfaces for interaction:

- Interactive shell for direct commands
- Script-based automation with .fx files
- Integration with existing development workflows

Example usage:

```
> analyze bug in login_system.py
> suggest fix for memory leak in resource_manager.py
> run verification tests on patch #1234
```

## Documentation

For more information, refer to the following documentation files:

- [Intent Classification Guide](INTENT_CLASSIFICATION_README.md)
- [Agent System Architecture](README_agent_system.md)
- [Shell Integration](README_shell.md)
- [Neural Matrix](neural_matrix/README.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
