# FixWurx Auditor Sensor System

## Overview

The FixWurx Auditor Sensor System provides comprehensive error detection, reporting, and analysis capabilities across all components of the auditor framework. Sensors monitor their respective components for anomalies, errors, and issues, generating standardized error reports that can be collected, analyzed, and acted upon.

## Architecture

The sensor system consists of the following key components:

1. **SensorRegistry**: Central registry for all sensors in the system
2. **SensorManager**: Coordinates error collection and monitoring
3. **ErrorSensor**: Base interface for all component-specific sensors
4. **ErrorReport**: Standard error report format
5. **Component-specific sensors**: Specialized sensors for each auditor component

```
┌────────────────┐     ┌─────────────────┐     ┌───────────────────┐
│ SensorRegistry │◄────┤ SensorManager   │────►│ Component Sensors │
└────────┬───────┘     └─────────────────┘     └────────┬──────────┘
         │                                              │
         │                                              │
         ▼                                              ▼
┌────────────────┐                           ┌───────────────────┐
│ Error Reports  │                           │ Component Data    │
└────────────────┘                           └───────────────────┘
```

## Core Components

### SensorRegistry

Central registry for all error sensors in the system, handling sensor registration, error collection, and storage.

Key responsibilities:
- Sensor registration and management
- Error report collection and storage
- Error querying and retrieval
- Error trend analysis

```python
# Create a sensor registry
from sensor_registry import SensorRegistry
registry = SensorRegistry(storage_path="auditor_data/sensors")

# Register a sensor
registry.register_sensor(my_sensor)

# Collect errors from all sensors
new_errors = registry.collect_errors()

# Query errors
component_errors = registry.query_errors(component_name="GraphDatabase")
```

### SensorManager

Coordinates error collection and monitoring across components.

Key responsibilities:
- Scheduled error collection
- Component monitoring
- Sensor registration for components

```python
# Create a sensor manager
from sensor_registry import SensorManager
manager = SensorManager(registry=registry, config={"collection_interval_seconds": 60})

# Monitor a component
errors = manager.monitor_component("GraphDatabase", graph_db_instance)

# Collect errors
manager.collect_errors()
```

### ErrorSensor

Base interface for all error sensors in the system.

Key responsibilities:
- Component monitoring
- Error detection
- Error report generation

```python
# Create a custom sensor
from sensor_registry import ErrorSensor

class CustomSensor(ErrorSensor):
    def monitor(self, data):
        # Check for errors in data
        if error_condition_detected:
            return [self.report_error(
                error_type="CUSTOM_ERROR",
                severity="HIGH",
                details={"message": "Error detected"}
            )]
        return []
```

### ErrorReport

Standard error report format for all sensors.

Key fields:
- `error_id`: Unique identifier
- `timestamp`: Time when the error was detected
- `sensor_id`: ID of the sensor that detected the error
- `component_name`: Name of the component where the error occurred
- `error_type`: Type of error detected
- `severity`: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
- `details`: Specific details about the error
- `context`: Additional context information
- `status`: Current status (OPEN, ACKNOWLEDGED, RESOLVED)

## Component-Specific Sensors

### ObligationLedgerSensor

Monitors the ObligationLedger component for:
- Empty obligations
- Empty delta rules
- Circular dependencies
- Missing obligations
- Rule application failures

### EnergyCalculatorSensor

Monitors the EnergyCalculator component for:
- Energy divergence
- Lambda threshold violations
- Negative gradients
- Energy oscillations
- Calculation errors

### ProofMetricsSensor

Monitors the ProofMetrics component for:
- Coverage below threshold
- High bug probability
- Insufficient verification
- Proof failures
- Invalid proofs

### MetaAwarenessSensor

Monitors the MetaAwareness component for:
- Excessive drift
- Excessive perturbation
- Phi increases
- Consistency violations
- Awareness degradation

### GraphDatabaseSensor

Monitors the GraphDatabase component for:
- Orphaned nodes
- Dangling edges
- Invalid relationships
- Circular references
- Graph inconsistencies

### TimeSeriesDatabaseSensor

Monitors the TimeSeriesDatabase component for:
- Data gaps
- Anomalous values
- Timestamp violations
- Insufficient data
- Trend violations

### DocumentStoreSensor

Monitors the DocumentStore component for:
- Invalid documents
- Missing references
- Schema violations
- Duplicate documents
- Document conflicts

### BenchmarkingSensor

Monitors the BenchmarkingSystem component for:
- Performance regressions
- High variance
- Insufficient iterations
- Timeouts
- Inconsistent results

## Configuration

Sensors can be configured through the `auditor_config.yaml` file:

```yaml
# Sensor configuration
sensors:
  # Global sensor settings
  enabled: true
  storage_path: "auditor_data/sensors"
  collection_interval_seconds: 60
  
  # Default thresholds for all sensors
  default_thresholds:
    sensitivity: 0.8
    max_dangling_edges: 0
    max_orphaned_nodes: 0
    max_circular_dependencies: 0
    
  # Component-specific sensor configurations
  components:
    ObligationLedger:
      enabled: true
      sensitivity: 0.9
      rule_application_threshold: 0.95
      
    EnergyCalculator:
      enabled: true
      sensitivity: 0.9
      energy_delta_threshold: 1.0e-7
```

## Usage Examples

### Creating and Registering Sensors

```python
from sensor_registry import create_sensor_registry
from component_sensors import (
    ObligationLedgerSensor, EnergyCalculatorSensor, GraphDatabaseSensor
)

# Create registry and manager
registry, manager = create_sensor_registry(config)

# Create sensors
obligation_sensor = ObligationLedgerSensor(
    component_name="ObligationLedger",
    config={"rule_application_threshold": 0.95}
)

energy_sensor = EnergyCalculatorSensor(
    component_name="EnergyCalculator",
    config={"energy_delta_threshold": 1.0e-7}
)

# Register sensors
registry.register_sensor(obligation_sensor)
registry.register_sensor(energy_sensor)
```

### Monitoring Components

```python
# Monitor the obligation ledger
errors = manager.monitor_component("ObligationLedger", obligation_ledger)

# Monitor the energy calculator
errors = manager.monitor_component("EnergyCalculator", energy_calculator)

# Collect all errors
new_errors = manager.collect_errors()
```

### Querying and Analyzing Errors

```python
# Get all errors for a component
component_errors = registry.query_errors(component_name="ObligationLedger")

# Get all high severity errors
high_severity_errors = registry.query_errors(severity="HIGH")

# Get error trends
trends = registry.get_error_trends()
```

### Resolving Errors

```python
# Resolve an error
registry.resolve_error("ERR-20250713-abcd1234", "Fixed by updating delta rules")

# Acknowledge an error
registry.acknowledge_error("ERR-20250713-efgh5678")
```

## Integration with Shell Interface

The sensor system is integrated with the shell interface through the `auditor_shell_interface.py` module, providing commands for:

- Listing and managing sensors
- Querying and analyzing errors
- Real-time monitoring
- Error resolution

See `README_shell_integration.md` for details on shell commands.

## Integration with LLM Components

The sensor system is integrated with LLM components through the `llm_sensor_integration.py` module, providing:

- Error contextualization
- Pattern recognition
- Self-diagnosis
- Recommendation generation

See `README_error_reporting.md` for details on LLM integration.

## Extending the Sensor System

### Creating Custom Sensors

To create a custom sensor:

1. Subclass the `ErrorSensor` class
2. Implement the `monitor` method
3. Use `report_error` to generate error reports

```python
from sensor_registry import ErrorSensor

class CustomSensor(ErrorSensor):
    def __init__(self, component_name, config=None):
        sensor_id = f"{component_name.lower()}_custom_sensor"
        super().__init__(sensor_id, component_name, config)
        
    def monitor(self, data):
        # Initialize error reports list
        reports = []
        
        # Check for error conditions
        if some_error_condition(data):
            # Report an error
            report = self.report_error(
                error_type="CUSTOM_ERROR_TYPE",
                severity="HIGH",
                details={"message": "Detailed error message", "value": 42},
                context={"additional_context": "value"}
            )
            reports.append(report)
        
        # Return all generated reports
        return reports
```

### Registering Custom Sensors

```python
# Create and register a custom sensor
custom_sensor = CustomSensor("CustomComponent", {"custom_threshold": 0.5})
registry.register_sensor(custom_sensor)
```

## Performance Considerations

- The sensor system is designed to have minimal performance impact
- Sensors should be efficient in their monitoring
- Adjust the collection interval based on system needs
- Disable sensors that are not needed
- Set appropriate sensitivity levels

## Troubleshooting

### Common Issues

1. **Sensors not detecting errors**
   - Check if the sensor is enabled
   - Verify the sensitivity setting
   - Ensure the component data is correct

2. **High sensor overhead**
   - Increase the collection interval
   - Disable unnecessary sensors
   - Optimize sensor implementations

3. **Missing error reports**
   - Check the storage path
   - Verify the sensor manager is enabled
   - Ensure errors are being collected

## API Reference

For a complete API reference, see the module documentation:

- `sensor_registry.py`: Core sensor registry and management
- `component_sensors.py`: Component-specific sensor implementations
- `llm_sensor_integration.py`: LLM integration for error analysis
