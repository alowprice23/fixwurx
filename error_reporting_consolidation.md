# Error Reporting System Consolidation Plan

## Duplicate Functionality Analysis

This document outlines the duplicate functionality between the new sensor-based error reporting system and the existing advanced error analysis system, with recommendations for consolidation.

### Primary Duplicate Components

| Functionality | `sensor_registry.py` | `advanced_error_analysis.py` | Recommendation |
|---------------|----------------------|------------------------------|----------------|
| Error Representation | `ErrorReport` class with standard fields | Dictionary-based error format | **Keep sensor_registry implementation** - More structured, object-oriented approach |
| Error Storage | YAML files in storage directory | Document store and local cache | **Merge capabilities** - Keep YAML storage but add document store option |
| Error Query | Filter by component, type, severity, status | Similar filtering capabilities | **Keep sensor_registry implementation** - Cleaner API |
| Error Resolution | `resolve_error()` and `acknowledge_error()` | Similar methods with document store updates | **Keep sensor_registry implementation** - More consistent |
| Trend Analysis | Basic error trends by component, type, severity | More sophisticated trend analysis | **Enhance sensor_registry** with advanced analysis features |

### Advanced Features to Migrate

The following advanced capabilities from `advanced_error_analysis.py` should be migrated to the sensor system:

1. **Root Cause Analysis**
   - Pattern-based identification of error causes
   - Confidence scoring for potential causes
   - Multi-level cause hierarchy

2. **Impact Assessment**
   - Severity determination
   - Scope analysis (single vs. multi-component)
   - User and system impact estimation

3. **Pattern Recognition**
   - Message pattern detection
   - Stack trace pattern analysis
   - Cross-component pattern identification

4. **Graph Relationship Management**
   - Error-to-component relationships
   - Error-to-error relationships
   - Causal relationship tracking

## Standardization Plan

### 1. Error Format Standardization

```python
class ErrorReport:
    """
    Standard error report format for all components.
    """
    
    def __init__(self,
                 sensor_id: str,
                 component_name: str,
                 error_type: str,
                 severity: str,
                 details: Dict[str, Any],
                 context: Optional[Dict[str, Any]] = None):
        # Core fields
        self.error_id = f"ERR-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"
        self.timestamp = datetime.datetime.now().isoformat()
        self.sensor_id = sensor_id
        self.component_name = component_name
        self.error_type = error_type
        self.severity = severity
        self.details = details
        self.context = context or {}
        
        # Status tracking
        self.status = "OPEN"  # OPEN, ACKNOWLEDGED, RESOLVED
        self.resolution = None
        self.resolution_timestamp = None
        
        # Advanced analysis fields (migrated from ErrorAnalyzer)
        self.root_cause = None
        self.impact = None
        self.related_errors = []
```

### 2. Repository Enhancement

Enhance `SensorRegistry` to incorporate the advanced features from `ErrorRepository`:

1. Add document store and graph database integration options
2. Add relationship tracking capabilities
3. Implement advanced query capabilities

### 3. Analysis Integration

Create an enhanced `ErrorAnalyzer` that works with the sensor system:

1. Migrate root cause analysis logic from existing analyzer
2. Integrate impact assessment capabilities
3. Add pattern recognition features
4. Implement relationship discovery

### 4. Implementation Phases

| Phase | Description | Tasks |
|-------|-------------|-------|
| 1 | Error Format Standardization | Update `ErrorReport` class with advanced fields |
| 2 | Storage Integration | Add document store and graph DB options to `SensorRegistry` |
| 3 | Analysis Migration | Port analysis capabilities to sensor-compatible versions |
| 4 | Testing | Verify all migrated capabilities work correctly |
| 5 | Old System Retirement | Remove `advanced_error_analysis.py` after migration |

## Integration with LLM Components

The LLM-based error analysis components (`ErrorContextualizer`, `ErrorPatternRecognizer`, `SelfDiagnosisProvider`) should use the standardized error format. These components can leverage the more detailed error information from the enhanced reports for better analysis.

## Code Refactoring Examples

### Example 1: Enhanced SensorRegistry

```python
class SensorRegistry:
    def __init__(self, storage_path: str = "auditor_data/sensors", 
                document_store = None, graph_db = None):
        self.sensors = {}
        self.error_reports = []
        self.storage_path = storage_path
        self.document_store = document_store
        self.graph_db = graph_db
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        logger.info(f"Initialized SensorRegistry with storage at {storage_path}")
```

### Example 2: Enhanced Error Analysis

```python
def analyze_error(self, error_id: str) -> Dict[str, Any]:
    """
    Analyze an error to determine root cause, impact, and relationships.
    
    Args:
        error_id: Error ID
    
    Returns:
        Analysis results
    """
    # Get error report
    report = self.registry.get_error_report(error_id)
    if not report:
        return {"error": "Error not found"}
    
    # Perform root cause analysis
    report.root_cause = self._perform_root_cause_analysis(report)
    
    # Assess impact
    report.impact = self._assess_impact(report)
    
    # Find related errors
    related_errors = self._find_related_errors(report)
    report.related_errors = [e.error_id for e in related_errors]
    
    # Update the report
    self.registry._store_report(report)
    
    return {
        "error_id": error_id,
        "root_cause": report.root_cause,
        "impact": report.impact,
        "related_errors": [e.error_id for e in related_errors]
    }
```

## Conclusion

By consolidating the error reporting functionality, we will achieve:

1. A more unified and consistent error handling approach
2. Reduced code duplication and maintenance overhead
3. Enhanced analysis capabilities leveraging both sensor-based detection and sophisticated analysis
4. Better integration with LLM components for advanced error understanding

The consolidated system will retain the strengths of both implementations: the structured sensor-based detection and the advanced analysis capabilities of the existing system.
