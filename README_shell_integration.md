# FixWurx Auditor Shell Integration

## Overview

The Auditor Shell Interface provides an interactive command-line interface for managing the sensor system, viewing error reports, performing real-time monitoring, and accessing the auditor's self-diagnostic capabilities. This shell environment serves as the primary interface for operators and maintainers to interact with the auditor system.

## Shell Features

- **Sensor Management**: List, enable/disable, and configure sensors
- **Error Reporting**: View, query, and resolve error reports
- **Real-time Monitoring**: Monitor sensor activity as it happens
- **Error Analysis**: Analyze error patterns and trends
- **Self-Diagnostics**: Access the auditor's self-diagnostic capabilities
- **Data Export**: Export error data for external analysis

## Getting Started

### Launching the Shell

To launch the Auditor Shell:

```python
from auditor_shell_interface import create_shell, run_shell
from sensor_registry import create_sensor_registry
from llm_sensor_integration import create_llm_integration
from llm_integrations import LLMManager

# Create sensor registry and manager
registry, manager = create_sensor_registry(config)

# Create LLM components
llm_manager = LLMManager()
llm_components = create_llm_integration(registry, manager, llm_manager)

# Create shell
shell = create_shell(
    registry=registry,
    sensor_manager=manager,
    llm_components=llm_components,
    config={"output_dir": "auditor_data/reports"}
)

# Run shell
run_shell(shell)
```

Or use the provided script:

```bash
python run_auditor_shell.py
```

## Available Commands

### General Commands

| Command | Description |
|---------|-------------|
| `help` | Display help information |
| `help <command>` | Display help for a specific command |
| `exit` | Exit the shell |
| `quit` | Alias for exit |

### Sensor Management Commands

#### List Sensors

Display all registered sensors:

```
auditor> sensors list
```

Example output:
```
Sensors (8):
Sensor ID                  | Component            | Enabled | Sensitivity
---------------------------|----------------------|---------|------------
obligation_ledger_sensor   | ObligationLedger     | Yes     | 0.90
energy_calculator_sensor   | EnergyCalculator     | Yes     | 0.90
proof_metrics_sensor       | ProofMetrics         | Yes     | 0.80
meta_awareness_sensor      | MetaAwareness        | Yes     | 0.90
graph_database_sensor      | GraphDatabase        | Yes     | 0.80
time_series_db_sensor      | TimeSeriesDatabase   | Yes     | 0.70
document_store_sensor      | DocumentStore        | Yes     | 0.80
benchmarking_sensor        | BenchmarkingSystem   | Yes     | 0.70
```

#### Show Sensor Status

Show status for all sensors or a specific sensor:

```
auditor> sensors status
auditor> sensors status --id obligation_ledger_sensor
```

Example output for a specific sensor:
```
Sensor: obligation_ledger_sensor
Component: ObligationLedger
Enabled: Yes
Sensitivity: 0.90
Error Count: 2
Last Check: 2025-07-13T01:20:45
```

#### Enable/Disable Sensors

Enable or disable a sensor:

```
auditor> sensors enable obligation_ledger_sensor
auditor> sensors disable energy_calculator_sensor
```

#### Set Sensor Sensitivity

Set the sensitivity level for a sensor (0.0 to 1.0):

```
auditor> sensors sensitivity obligation_ledger_sensor 0.8
```

### Error Reporting Commands

#### List Errors

List all errors or filter by criteria:

```
auditor> errors list
auditor> errors list --component ObligationLedger
auditor> errors list --type CIRCULAR_DEPENDENCIES
auditor> errors list --severity HIGH
auditor> errors list --status OPEN
```

Example output:
```
Errors (3):
ID                        | Component        | Type                    | Severity | Status | Timestamp
--------------------------|------------------|-------------------------|----------|--------|-------------------
ERR-20250713010000-abcd12 | ObligationLedger | CIRCULAR_DEPENDENCIES   | HIGH     | OPEN   | 2025-07-13 01:00:00
ERR-20250713010235-efgh34 | EnergyCalculator | LAMBDA_EXCEEDS_THRESHOLD| MEDIUM   | OPEN   | 2025-07-13 01:02:35
ERR-20250713011530-ijkl56 | GraphDatabase    | ORPHANED_NODES          | LOW      | RESOLVED| 2025-07-13 01:15:30
```

#### Show Error Details

Show detailed information about a specific error:

```
auditor> errors show ERR-20250713010000-abcd12
```

Example output:
```
Error ID: ERR-20250713010000-abcd12
Timestamp: 2025-07-13T01:00:00
Component: ObligationLedger
Type: CIRCULAR_DEPENDENCIES
Severity: HIGH
Status: OPEN

Details:
  message: Circular dependency detected in obligation chain
  dependencies: ['A depends on B', 'B depends on C', 'C depends on A']

Context:
  module_version: 1.2.3
  system_state: initializing

Explanation:
This error indicates a circular dependency in the obligation chain, where
obligations A, B, and C form a cycle. Circular dependencies prevent the
proper application of delta rules, as the system cannot determine a valid
order to process the obligations.

Recommendations:
1. Review the obligation definitions and identify the cycle
2. Break the cycle by modifying one of the dependencies
3. Consider introducing an intermediate obligation to resolve the cycle
```

#### Analyze Errors

Analyze error patterns:

```
auditor> errors analyze
auditor> errors analyze --hours 48
```

Example output:
```
Analyzing error patterns over the past 24 hours...
Analysis completed at: 2025-07-13T01:35:45
Errors analyzed: 25

Component Distribution:
- ObligationLedger: 8
- EnergyCalculator: 6
- GraphDatabase: 5
- TimeSeriesDatabase: 4
- DocumentStore: 2

Error Type Distribution:
- CIRCULAR_DEPENDENCIES: 7
- LAMBDA_EXCEEDS_THRESHOLD: 6
- ORPHANED_NODES: 5
- DATA_GAP: 4
- INVALID_DOCUMENT: 2
- NEGATIVE_GRADIENT: 1

Severity Distribution:
- HIGH: 10
- MEDIUM: 8
- LOW: 7

Identified Patterns:
1. Most circular dependency errors occur during initialization
2. Lambda threshold violations often follow energy recalculation
3. Orphaned nodes and data gaps appear to be correlated

Root Cause Hypotheses:
1. Circular dependencies may be caused by recent changes to rule definitions
2. Lambda threshold violations suggest energy convergence issues
3. Orphaned nodes could indicate incomplete graph cleanup
```

#### Show Error Trends

Show error trends over time:

```
auditor> errors trends
```

Example output:
```
Total Errors: 45

Errors by Component:
- ObligationLedger: 15
- EnergyCalculator: 12
- GraphDatabase: 8
- TimeSeriesDatabase: 6
- DocumentStore: 4

Errors by Type:
- CIRCULAR_DEPENDENCIES: 12
- LAMBDA_EXCEEDS_THRESHOLD: 10
- ORPHANED_NODES: 8
- DATA_GAP: 6
- INVALID_DOCUMENT: 4
- NEGATIVE_GRADIENT: 3
- INSUFFICIENT_DATA: 2

Errors by Severity:
- CRITICAL: 5
- HIGH: 15
- MEDIUM: 18
- LOW: 7

Errors by Date:
- 2025-07-11: 10
- 2025-07-12: 20
- 2025-07-13: 15
```

#### Export Errors

Export errors to a file:

```
auditor> errors export
auditor> errors export --format json
auditor> errors export --format yaml
auditor> errors export --format csv
auditor> errors export --output /path/to/output/file.json
```

#### Resolve Error

Resolve an error:

```
auditor> errors resolve ERR-20250713010000-abcd12 "Fixed by removing circular dependency"
```

#### Acknowledge Error

Acknowledge an error:

```
auditor> errors acknowledge ERR-20250713010235-efgh34
```

### Monitoring Commands

#### Start Monitoring

Start real-time monitoring:

```
auditor> monitor start
```

Example output:
```
Real-time monitoring started
Use 'monitor stop' to stop monitoring
```

#### Stop Monitoring

Stop real-time monitoring:

```
auditor> monitor stop
```

#### Collect Errors

Manually collect errors from sensors:

```
auditor> monitor collect
```

Example output:
```
Collected 3 new error reports

ID                        | Component        | Type                    | Severity
--------------------------|------------------|-------------------------|----------
ERR-20250713013000-mnop78 | ObligationLedger | CIRCULAR_DEPENDENCIES   | HIGH
ERR-20250713013002-qrst90 | EnergyCalculator | LAMBDA_EXCEEDS_THRESHOLD| MEDIUM
ERR-20250713013005-uvwx12 | GraphDatabase    | ORPHANED_NODES          | LOW
```

#### Show Monitoring Status

Show monitoring status:

```
auditor> monitor status
```

Example output:
```
Monitoring Active: Yes
Sensor Manager Enabled: Yes
Collection Interval: 60 seconds
Last Collection: 2025-07-13T01:30:00
```

### Self-Diagnostic Commands

#### Diagnose Issue

Diagnose a specific issue:

```
auditor> diagnose issue "Auditor fails to apply delta rules correctly"
```

Example output:
```
Diagnosing issue: Auditor fails to apply delta rules correctly
This may take a moment...

Diagnosis:
The issue appears to be related to circular dependencies in the obligation
definitions. When the auditor attempts to apply delta rules, it encounters
cycles in the dependency graph, preventing it from determining a valid order
for rule application. This is confirmed by the presence of CIRCULAR_DEPENDENCIES
errors in the ObligationLedger component. To resolve this, review the delta
rules defined in delta_rules.json and ensure there are no circular references
between obligations.
```

#### Explain Internal State

Get an explanation of the current internal state:

```
auditor> diagnose state
```

Example output:
```
Analyzing internal state...
This may take a moment...

System State:
The auditor is currently in a partially degraded state due to issues with the
ObligationLedger and EnergyCalculator components. There are 5 open errors,
including 2 HIGH severity issues that require immediate attention. The most
critical issue is a circular dependency in the obligation definitions that
is preventing proper delta rule application. Energy calculation is also affected,
with lambda values exceeding thresholds, suggesting convergence issues.

Component States:

ObligationLedger:
The ObligationLedger contains 25 obligations and 10 delta rules. It is currently
unable to properly apply all rules due to circular dependencies between obligations
A, B, and C. This is causing cascading issues in the verification process.

EnergyCalculator:
The EnergyCalculator is experiencing convergence issues, with lambda values
exceeding thresholds. This may be related to the obligation issues, as the
energy calculation depends on a valid obligation state.

GraphDatabase:
The GraphDatabase is functioning normally but contains 3 orphaned nodes that
should be addressed. These nodes are not currently impacting system operation.
```

#### Suggest Corrections

Get suggestions for fixing an error:

```
auditor> diagnose suggest ERR-20250713010000-abcd12
```

Example output:
```
Generating correction suggestions for error 'ERR-20250713010000-abcd12'...
This may take a moment...

Suggested corrections for error 'ERR-20250713010000-abcd12':

1. Break Circular Dependency
   Complexity: MEDIUM
   Steps:
     1. Identify the circular dependency chain (A → B → C → A)
     2. Choose one dependency to remove (e.g., C → A)
     3. Update the delta rules in delta_rules.json
     4. Restart the auditor to apply changes
   Expected Impact: Will resolve the circular dependency and allow proper rule application

2. Introduce Intermediate Obligation
   Complexity: HIGH
   Steps:
     1. Create a new intermediate obligation D
     2. Replace dependency C → A with C → D → A
     3. Update the delta rules in delta_rules.json
     4. Restart the auditor to apply changes
   Expected Impact: Will resolve the circular dependency while preserving semantic relationships

3. Restructure Obligation Hierarchy
   Complexity: HIGH
   Steps:
     1. Review the entire obligation hierarchy
     2. Reorganize obligations into logical groups
     3. Ensure no group creates circular dependencies
     4. Update the delta rules in delta_rules.json
     5. Restart the auditor to apply changes
   Expected Impact: Comprehensive solution that may prevent future circular dependencies
```

## Best Practices

### Effective Shell Usage

1. **Regular Monitoring**: Use `monitor start` during critical operations
2. **Error Analysis**: Regularly use `errors analyze` to identify patterns
3. **Proactive Resolution**: Address high-severity errors promptly
4. **Export Data**: Export error data for long-term analysis
5. **Customize Sensors**: Adjust sensor sensitivity based on system needs

### Shell Integration Patterns

1. **Interactive Diagnostics**: Use the shell for interactive troubleshooting
2. **Scheduled Tasks**: Create scripts that execute shell commands on a schedule
3. **Error Alerts**: Configure the shell to alert on critical errors
4. **Performance Monitoring**: Use the shell to monitor system performance
5. **Continuous Improvement**: Use error analysis to guide system improvements

## Customizing the Shell

The Auditor Shell can be customized by extending the `AuditorShell` class:

```python
from auditor_shell_interface import AuditorShell

class CustomAuditorShell(AuditorShell):
    def __init__(self, registry, sensor_manager, llm_components, config=None):
        super().__init__(registry, sensor_manager, llm_components, config)
        
    def do_custom_command(self, arg):
        """
        Custom command description.
        
        Usage: custom_command ARG
        """
        # Command implementation
        self.stdout.write(f"Custom command executed with argument: {arg}\n")
```

## Troubleshooting

### Common Shell Issues

1. **Command not found**:
   - Check command spelling
   - Ensure the command is available in the current shell version

2. **Error accessing data**:
   - Verify storage paths in configuration
   - Check file permissions

3. **LLM features not working**:
   - Ensure LLM manager is properly configured
   - Check LLM service availability

4. **Slow command execution**:
   - For complex queries, consider adding filters
   - Export data for offline analysis of large datasets

## Shell Command Reference

| Command | Subcommand | Options | Description |
|---------|------------|---------|-------------|
| `help` | | | Display help information |
| `exit` | | | Exit the shell |
| `quit` | | | Alias for exit |
| `sensors` | `list` | | List all sensors |
| `sensors` | `status` | `--id SENSOR_ID` | Show sensor status |
| `sensors` | `enable` | `SENSOR_ID` | Enable a sensor |
| `sensors` | `disable` | `SENSOR_ID` | Disable a sensor |
| `sensors` | `sensitivity` | `SENSOR_ID VALUE` | Set sensor sensitivity |
| `errors` | `list` | `--component COMP` `--type TYPE` `--severity SEV` `--status STATUS` | List errors |
| `errors` | `show` | `ERROR_ID` | Show error details |
| `errors` | `analyze` | `--hours HOURS` | Analyze error patterns |
| `errors` | `trends` | | Show error trends |
| `errors` | `export` | `--format FORMAT` `--output FILE` | Export errors |
| `errors` | `resolve` | `ERROR_ID RESOLUTION` | Resolve an error |
| `errors` | `acknowledge` | `ERROR_ID` | Acknowledge an error |
| `monitor` | `start` | | Start real-time monitoring |
| `monitor` | `stop` | | Stop real-time monitoring |
| `monitor` | `collect` | | Manually collect errors |
| `monitor` | `status` | | Show monitoring status |
| `diagnose` | `issue` | `DESCRIPTION` | Diagnose a specific issue |
| `diagnose` | `state` | | Explain internal state |
| `diagnose` | `suggest` | `ERROR_ID` | Get correction suggestions |
