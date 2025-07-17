# FixWurx Auditor Error Reporting System

## Overview

The FixWurx Auditor Error Reporting System provides a comprehensive framework for detecting, analyzing, reporting, and resolving errors across all components of the auditor system. It integrates with LLM components to provide advanced error analysis, pattern recognition, and self-diagnostic capabilities.

## Architecture

The error reporting system consists of the following key components:

1. **Error Detection**: Component-specific sensors detect error conditions
2. **Error Reporting**: Standardized error reports are generated and stored
3. **Error Analysis**: LLM components analyze errors for root causes and patterns
4. **Error Resolution**: Tools for resolving and tracking error resolutions
5. **Error Visualization**: Shell interface for viewing and interacting with errors

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Error Detection │────►│ Error Reporting │────►│  Error Analysis │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│Error Resolution │◄────┤Error Visualization◄────┤  LLM Integration│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Standardized Error Format

All errors in the system follow a standardized format defined in `error_format_standardization.md`, which ensures consistency and interoperability across components.

Key fields include:
- `error_id`: Unique identifier
- `timestamp`: Time when the error was detected
- `component_name`: Name of the component where the error occurred
- `error_type`: Type of error detected
- `severity`: Severity level
- `details`: Specific details about the error
- `root_cause`: Root cause analysis result (added during analysis)
- `impact`: Impact assessment result (added during analysis)

## LLM Integration Components

### SensorDataProvider

Provides sensor and error data to LLM components for analysis.

Key capabilities:
- Collect sensor data from all components
- Retrieve error reports from the registry
- Format data for LLM processing
- Cache frequently used data for performance

```python
from llm_sensor_integration import SensorDataProvider

# Create a data provider
data_provider = SensorDataProvider(registry, sensor_manager)

# Get sensor data for a component
component_data = data_provider.get_component_data("ObligationLedger")

# Get error data
error_data = data_provider.get_error_data("ERR-20250713-abcd1234")

# Get trend data
trend_data = data_provider.get_error_trends(days=7)
```

### ErrorContextualizer

Enhances error reports with contextual information, explanations, and recommendations.

Key capabilities:
- Generate natural language explanations of errors
- Identify potential causes and impacts
- Provide recommendations for resolution
- Link to related errors and context

```python
from llm_sensor_integration import ErrorContextualizer

# Create an error contextualizer
contextualizer = ErrorContextualizer(
    data_provider=data_provider,
    llm_manager=llm_manager
)

# Contextualize an error
enhanced_error = contextualizer.contextualize(error_report)

# Access enhanced information
explanation = enhanced_error["explanation"]
recommendations = enhanced_error["recommendations"]
related_components = enhanced_error["related_components"]
```

### ErrorPatternRecognizer

Analyzes error patterns across time and components to identify systemic issues.

Key capabilities:
- Identify recurring error patterns
- Detect temporal patterns (time-based trends)
- Recognize cross-component patterns
- Generate root cause hypotheses

```python
from llm_sensor_integration import ErrorPatternRecognizer

# Create a pattern recognizer
recognizer = ErrorPatternRecognizer(
    data_provider=data_provider,
    llm_manager=llm_manager
)

# Analyze patterns over the last 24 hours
analysis = recognizer.analyze_patterns(timeframe_hours=24)

# Access analysis results
statistical_patterns = analysis["statistical_patterns"]
llm_patterns = analysis["llm_patterns"]
identified_patterns = llm_patterns["identified_patterns"]
root_cause_hypotheses = llm_patterns["root_cause_hypotheses"]
```

### SelfDiagnosisProvider

Enables the auditor to diagnose itself and provide explanations of its internal state.

Key capabilities:
- Diagnose specific issues based on sensor data
- Explain the current internal state of the system
- Suggest corrections for reported errors
- Provide ongoing self-monitoring

```python
from llm_sensor_integration import SelfDiagnosisProvider

# Create a self-diagnosis provider
self_diagnosis = SelfDiagnosisProvider(
    data_provider=data_provider,
    llm_manager=llm_manager
)

# Diagnose an issue
diagnosis = self_diagnosis.diagnose_issue(
    "Auditor fails to apply delta rules correctly"
)

# Get explanation of internal state
state = self_diagnosis.explain_internal_state()

# Get correction suggestions for an error
corrections = self_diagnosis.suggest_corrections(error_report)
```

## Error Processing Workflow

### 1. Error Detection

Sensors monitor components for error conditions:

```python
class GraphDatabaseSensor(ErrorSensor):
    def monitor(self, graph_db):
        # Initialize error reports list
        reports = []
        
        # Check for orphaned nodes
        orphaned_nodes = graph_db.get_orphaned_nodes()
        if len(orphaned_nodes) > self.max_orphaned_nodes:
            # Report an error
            report = self.report_error(
                error_type="ORPHANED_NODES",
                severity="MEDIUM",
                details={
                    "message": f"Detected {len(orphaned_nodes)} orphaned nodes",
                    "orphaned_nodes": orphaned_nodes
                }
            )
            reports.append(report)
        
        return reports
```

### 2. Error Collection

The SensorManager collects errors from all sensors:

```python
# Collect errors from all sensors
new_errors = manager.collect_errors()

# Process new errors
for error in new_errors:
    process_error(error)
```

### 3. Error Analysis

LLM components analyze errors for root causes and patterns:

```python
# Contextualize an error
enhanced_error = contextualizer.contextualize(error)

# Update the error with enhanced information
error.root_cause = enhanced_error["root_cause"]
error.impact = enhanced_error["impact"]
error.recommendations = enhanced_error["recommendations"]

# Store the updated error
registry._store_report(error)
```

### 4. Error Reporting

Errors are reported through the shell interface:

```
auditor> errors show ERR-20250713-abcd1234
```

### 5. Error Resolution

Errors are resolved through the shell interface:

```
auditor> errors resolve ERR-20250713-abcd1234 "Fixed by updating delta rules"
```

## Advanced Error Analysis Capabilities

### Root Cause Analysis

The system uses LLM-powered analysis to determine the root cause of errors:

1. **Pattern Matching**: Match error messages and stack traces against known patterns
2. **Contextual Analysis**: Analyze the context in which the error occurred
3. **Component Relationships**: Consider relationships between components
4. **Historical Data**: Compare with similar errors in the past

Example root cause analysis result:

```json
{
  "cause_type": "circular_dependency",
  "confidence": 0.92,
  "details": {
    "description": "Circular dependencies in obligation definitions",
    "source_file": "delta_rules.json",
    "source_function": "apply_delta_rules",
    "dependency_chain": ["A → B → C → A"]
  },
  "potential_causes": [
    {
      "cause_type": "rule_definition_error",
      "confidence": 0.75,
      "details": {
        "description": "Error in rule definition format"
      }
    }
  ]
}
```

### Impact Assessment

The system assesses the impact of errors on the system and users:

1. **Component Impact**: Determine which components are affected
2. **Functional Impact**: Identify affected functionality
3. **User Impact**: Assess impact on users
4. **System Impact**: Evaluate overall system impact

Example impact assessment result:

```json
{
  "severity": "HIGH",
  "scope": "multi_component",
  "affected_components": [
    "ObligationLedger",
    "EnergyCalculator",
    "ProofMetrics"
  ],
  "affected_functionality": [
    "rule_application",
    "energy_calculation",
    "proof_verification"
  ],
  "user_impact": "Users cannot verify rule applications",
  "system_impact": "System cannot complete verification process"
}
```

### Pattern Recognition

The system recognizes patterns across errors:

1. **Statistical Patterns**: Frequency, distribution, and correlation analysis
2. **Temporal Patterns**: Time-based trends and cyclical patterns
3. **Cross-Component Patterns**: Patterns across different components
4. **Message Patterns**: Similarities in error messages and details

Example pattern recognition result:

```json
{
  "statistical_patterns": {
    "component_distribution": {
      "ObligationLedger": 15,
      "EnergyCalculator": 12,
      "GraphDatabase": 8
    },
    "temporal_patterns": {
      "peak_hours": [9, 14, 18],
      "daily_distribution": {
        "Monday": 25,
        "Tuesday": 18,
        "Wednesday": 22
      }
    },
    "co_occurrences": [
      {
        "error1": {"component": "ObligationLedger", "type": "CIRCULAR_DEPENDENCIES"},
        "error2": {"component": "EnergyCalculator", "type": "ENERGY_DIVERGENCE"},
        "correlation": 0.85
      }
    ]
  },
  "llm_patterns": {
    "identified_patterns": [
      "Circular dependencies trigger energy divergence",
      "Graph inconsistencies follow database updates",
      "Most errors occur during rule application phase"
    ],
    "root_cause_hypotheses": [
      "Recent changes to delta rules may have introduced circular dependencies",
      "Database schema changes may be causing graph inconsistencies",
      "Energy calculation parameters may need adjustment"
    ]
  }
}
```

## Integration with LLM Manager

The error reporting system integrates with the LLM Manager to leverage LLM capabilities for error analysis:

```python
from llm_integrations import LLMManager
from llm_sensor_integration import create_llm_integration

# Create LLM manager
llm_manager = LLMManager()

# Create LLM integration components
llm_components = create_llm_integration(
    registry=registry,
    sensor_manager=manager,
    llm_manager=llm_manager
)

# Access individual components
data_provider = llm_components["data_provider"]
contextualizer = llm_components["error_contextualizer"]
pattern_recognizer = llm_components["pattern_recognizer"]
self_diagnosis = llm_components["self_diagnosis"]
```

## Error Data Storage and Retrieval

Errors are stored in YAML format in the error storage directory:

```yaml
error_id: "ERR-20250713010000-abcd1234"
timestamp: "2025-07-13T01:00:00.000Z"
sensor_id: "obligation_ledger_sensor"
component_name: "ObligationLedger"
error_type: "CIRCULAR_DEPENDENCIES"
severity: "HIGH"
details:
  message: "Circular dependency detected in obligation chain"
  dependencies:
    - "A depends on B"
    - "B depends on C"
    - "C depends on A"
context:
  module_version: "1.2.3"
  system_state: "initializing"
status: "OPEN"
root_cause:
  cause_type: "logic_error"
  confidence: 0.9
  details:
    description: "Circular dependencies in obligation definitions"
    source_file: "obligation_ledger.py"
    source_function: "apply_delta_rules"
impact:
  severity: "HIGH"
  scope: "single_component"
  affected_components: ["ObligationLedger"]
  affected_functionality: ["rule_application", "verification"]
  user_impact: "Users cannot verify rule applications"
  system_impact: "System cannot complete verification process"
```

## Configuration

LLM integration for error reporting can be configured in `auditor_config.yaml`:

```yaml
# LLM Integration configuration
llm_integration:
  enabled: true
  provider: "openai"
  model: "gpt-4"
  temperature: 0.2
  max_tokens: 1000
  api_key_env: "OPENAI_API_KEY"
  integration_points:
    - "obligation_extraction"
    - "delta_rule_generation"
    - "error_contextualization"
    - "gap_analysis"
  
  # Custom LLM settings for specific integration points
  custom_settings:
    error_contextualization:
      temperature: 0.1
      max_tokens: 500
    gap_analysis:
      temperature: 0.3
      max_tokens: 2000
```

## Best Practices

### Effective Error Handling

1. **Standardized Error Types**: Use standardized error types defined in `error_format_standardization.md`
2. **Appropriate Severity Levels**: Assign appropriate severity levels based on impact
3. **Contextual Information**: Include relevant contextual information in error reports
4. **Proactive Monitoring**: Monitor components proactively to detect errors early
5. **Timely Resolution**: Address high-severity errors promptly

### LLM Integration Optimization

1. **Efficient Data Providers**: Optimize data providers to minimize LLM token usage
2. **Cached Analysis**: Cache analysis results for similar errors
3. **Batched Processing**: Process multiple errors in batches when possible
4. **Selective Enhancement**: Only enhance errors that require detailed analysis
5. **Feedback Loop**: Incorporate resolution feedback to improve future analysis

## Extending the Error Reporting System

### Adding New Error Types

To add a new error type:

1. Add the error type to the standardized error types list in `error_format_standardization.md`
2. Update relevant sensors to detect and report the new error type
3. Update LLM components to analyze the new error type

### Creating Custom Analysis Components

To create a custom analysis component:

```python
from llm_sensor_integration import BaseAnalysisProvider

class CustomAnalysisProvider(BaseAnalysisProvider):
    def __init__(self, data_provider, llm_manager):
        super().__init__(data_provider, llm_manager)
    
    def analyze_error(self, error_report):
        # Custom analysis logic
        # ...
        return {
            "custom_analysis": "result",
            "recommendations": ["custom recommendation"]
        }
```

## Troubleshooting

### Common Issues

1. **LLM Integration Not Working**:
   - Check LLM service availability
   - Verify API keys and configuration
   - Check network connectivity

2. **Missing Error Reports**:
   - Verify sensor configuration
   - Check error storage path
   - Ensure collection interval is appropriate

3. **Incorrect Analysis**:
   - Review LLM prompt templates
   - Check data provider functionality
   - Verify error report format

## API Reference

For a complete API reference, see the module documentation:

- `sensor_registry.py`: Core error reporting functionality
- `llm_sensor_integration.py`: LLM integration components
- `component_sensors.py`: Component-specific error detection
