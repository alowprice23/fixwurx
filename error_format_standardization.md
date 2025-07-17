# Standardized Error Format Specification

## Overview

This document defines the standardized error format to be used across the FixWurx Auditor system. All components that detect, report, or process errors should conform to this specification to ensure consistency, interoperability, and comprehensive error handling.

## Core Error Report Structure

Every error in the system must include the following core fields:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `error_id` | String | Unique identifier in format `ERR-YYYYMMDDHHMMSS-RANDOM` | Yes |
| `timestamp` | ISO8601 | Time when the error was detected/generated | Yes |
| `sensor_id` | String | ID of the sensor that detected the error | Yes |
| `component_name` | String | Name of the component where the error occurred | Yes |
| `error_type` | String | Type of the error (standardized error types listed below) | Yes |
| `severity` | String | One of: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW` | Yes |
| `details` | Object | Detailed information about the error (structure varies by error type) | Yes |
| `context` | Object | Additional contextual information | No |
| `status` | String | One of: `OPEN`, `ACKNOWLEDGED`, `RESOLVED` | Yes |
| `resolution` | String | Description of how the error was resolved | No |
| `resolution_timestamp` | ISO8601 | Time when the error was resolved | No |

## Extended Error Information

The following extended fields should be included when an error has been analyzed:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `root_cause` | Object | Root cause analysis result | No |
| `impact` | Object | Impact assessment result | No |
| `related_errors` | Array | IDs of related error reports | No |
| `recommendations` | Array | Recommendations for resolving the error | No |

### Root Cause Structure

```json
{
  "cause_type": "String: Type of root cause",
  "confidence": "Float: Confidence level (0.0-1.0)",
  "details": {
    "description": "String: Human-readable description",
    "source_file": "String: Source file where the error originated",
    "source_function": "String: Function where the error originated",
    "line_number": "Integer: Line number where the error originated"
  },
  "potential_causes": [
    {
      "cause_type": "String: Type of potential cause",
      "confidence": "Float: Confidence level (0.0-1.0)",
      "details": { "..." }
    }
  ]
}
```

### Impact Structure

```json
{
  "severity": "String: Severity level",
  "scope": "String: 'single_component' or 'multi_component'",
  "affected_components": ["String: List of affected component names"],
  "affected_functionality": ["String: List of affected functionality"],
  "user_impact": "String: Description of impact on users",
  "system_impact": "String: Description of impact on system"
}
```

## Standardized Error Types

The following standardized error types should be used:

### System-Level Errors

- `RESOURCE_EXHAUSTION`: Resources like memory, disk space, or connections exhausted
- `NETWORK_ERROR`: Network-related errors (connection, timeout, etc.)
- `PERFORMANCE_DEGRADATION`: System performing below expected thresholds
- `SECURITY_VIOLATION`: Security policy violations
- `CONFIGURATION_ERROR`: System misconfiguration issues

### Data-Level Errors

- `DATA_CORRUPTION`: Data is corrupt or inconsistent
- `DATA_MISSING`: Required data is missing
- `SCHEMA_VIOLATION`: Data doesn't match expected schema
- `VALIDATION_FAILURE`: Data failed validation rules
- `INTEGRITY_VIOLATION`: Data integrity constraints violated

### Component-Specific Errors

#### ObligationLedger Errors

- `EMPTY_OBLIGATIONS`: No obligations found
- `EMPTY_DELTA_RULES`: No delta rules found
- `CIRCULAR_DEPENDENCIES`: Circular dependencies detected in obligations
- `MISSING_OBLIGATIONS`: Required obligations missing
- `RULE_APPLICATION_FAILURE`: Rule application failed

#### EnergyCalculator Errors

- `ENERGY_DIVERGENCE`: Energy not converging
- `LAMBDA_EXCEEDS_THRESHOLD`: Lambda value exceeds threshold
- `NEGATIVE_GRADIENT`: Negative gradient detected
- `ENERGY_OSCILLATION`: Energy oscillating rather than converging
- `CALCULATION_ERROR`: General calculation error

#### ProofMetrics Errors

- `COVERAGE_BELOW_THRESHOLD`: Proof coverage below threshold
- `HIGH_BUG_PROBABILITY`: Bug probability above threshold
- `INSUFFICIENT_VERIFICATION`: Insufficient verification count
- `PROOF_FAILURE`: Proof generation failed
- `INVALID_PROOF`: Invalid proof detected

#### MetaAwareness Errors

- `EXCESSIVE_DRIFT`: Semantic drift exceeds threshold
- `EXCESSIVE_PERTURBATION`: Perturbation exceeds threshold
- `PHI_INCREASE`: Phi value increasing
- `CONSISTENCY_VIOLATION`: Consistency violated
- `AWARENESS_DEGRADATION`: Meta-awareness degraded

#### GraphDatabase Errors

- `ORPHANED_NODES`: Orphaned nodes detected
- `DANGLING_EDGES`: Dangling edges detected
- `INVALID_RELATIONSHIP`: Invalid relationship detected
- `CIRCULAR_REFERENCE`: Circular reference detected
- `GRAPH_INCONSISTENCY`: Graph consistency violated

#### TimeSeriesDatabase Errors

- `DATA_GAP`: Gap in time series data
- `ANOMALOUS_VALUE`: Anomalous value detected
- `TIMESTAMP_VIOLATION`: Invalid timestamp
- `INSUFFICIENT_DATA`: Insufficient data points
- `TREND_VIOLATION`: Trend violated

#### DocumentStore Errors

- `INVALID_DOCUMENT`: Invalid document detected
- `MISSING_REFERENCE`: Missing document reference
- `SCHEMA_VIOLATION`: Document schema violated
- `DUPLICATE_DOCUMENT`: Duplicate document detected
- `DOCUMENT_CONFLICT`: Document conflict detected

#### BenchmarkingSystem Errors

- `PERFORMANCE_REGRESSION`: Performance regression detected
- `HIGH_VARIANCE`: High variance in benchmark results
- `INSUFFICIENT_ITERATIONS`: Insufficient benchmark iterations
- `TIMEOUT`: Benchmark timeout
- `INCONSISTENT_RESULTS`: Inconsistent benchmark results

## Error Serialization

Errors should be serializable to both JSON and YAML formats using the following structure:

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

## Integration with Error Detection System

The sensor system should use this standardized format:

1. **Sensor Detection**: Sensors detect error conditions and create reports
2. **Report Creation**: Reports are created using the standardized format
3. **Storage**: Reports are stored in YAML format
4. **Analysis**: Reports are analyzed to determine root cause and impact
5. **Enhancement**: Reports are enhanced with analysis results
6. **Query**: Reports can be queried by component, type, severity, etc.

## Integration with Error Analysis System

The error analysis system should use this standardized format:

1. **Load Reports**: Load error reports in standardized format
2. **Analyze Reports**: Analyze reports to determine root cause and impact
3. **Update Reports**: Update reports with analysis results
4. **Store Reports**: Store updated reports in standardized format

## Implementation Guidelines

1. All new error reporting should use this standardized format
2. Existing error reporting should be migrated to this format
3. Error detection, reporting, and analysis components should be updated to use this format
4. Storage and retrieval mechanisms should support this format
5. Visualization and reporting tools should display information from this format

## Transition Plan

1. Update `ErrorReport` class to include all required fields
2. Enhance sensors to use the standardized error types
3. Update storage mechanisms to support the extended format
4. Migrate existing error reports to the new format
5. Update analysis components to work with the new format
