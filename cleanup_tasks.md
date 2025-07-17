# Auditor System Cleanup Tasks

## Unnecessary Scripts to Remove

The following scripts appear to be ad-hoc fixes or utilities that are now superseded by the new error reporting and sensor system:

1. **Direct Fix Scripts**
   - `direct_auditor_fix.py` - Ad-hoc fixes now handled by component sensors
   - `fix_auditor_components.py` - Now replaced by component-specific sensors
   - `fix_auditor_methods.py` - Method fixes now covered by error detection
   - `auditor_implementation_fixes.py` - Implementation fixes now systematized
   - `direct_method_addition.py` - Ad-hoc method additions
   - `direct_method_injection.py` - Ad-hoc method injections
   - `direct_replace_methods.py` - Ad-hoc method replacements
   - `direct_test_fix.py` - Test-specific fixes

2. **Comprehensive Fix Scripts**
   - `comprehensive_fix.py` - One-time comprehensive fixes 
   - `complete_functionality_fix.py` - One-time functionality fixes
   - `fix_functionality_verifier.py` - Specific verifier fixes

3. **Test-Related Fix Scripts**
   - `fix_test_config_manager.py` - Config manager test fixes
   - `fix_config_manager_test.py` - Similar to above
   - `fix_upgrade_test.py` - Upgrade test fixes

4. **Backup Files**
   - `advanced_error_analysis.py.bak` - Backup file
   - `functionality_verification.py.bak` - Backup file
   - `functionality_verification.py.bak2` - Another backup file

5. **Specific Component Fixes**
   - `scaling_coordinator_fix.py` - Specific component fix
   - `scaling_coordinator_fix_v2.py` - Updated specific component fix
   - `fixed_functionality_verification.py` - Fixed functionality verification
   - `fixed_key_rotation_test.py` - Fixed key rotation test
   - `fixed_log_retention_test.py` - Fixed log retention test
   - `fixed_model_updater_test.py` - Fixed model updater test

## Files Exceeding 800 Lines to Refactor

The following files need to be examined and potentially refactored into smaller components:

1. Check auditor.py line count and consider splitting into:
   - `auditor_core.py` - Core auditor functionality
   - `auditor_verification.py` - Verification-specific code
   - `auditor_reporting.py` - Reporting functionality

2. Check component_sensors.py line count and consider splitting by component:
   - `obligation_ledger_sensor.py`
   - `energy_calculator_sensor.py`
   - `proof_metrics_sensor.py`
   - `meta_awareness_sensor.py`
   - `graph_database_sensor.py`
   - `time_series_database_sensor.py`
   - `document_store_sensor.py`
   - `benchmarking_sensor.py`

## Duplicate Functionality to Consolidate

1. Error reporting functionality in:
   - New `sensor_registry.py` vs. existing `advanced_error_analysis.py`
   - Consider keeping only the new implementation

2. Verification functionality in:
   - `functionality_verification.py` vs. sensor-based verification
   - Consider migrating necessary functionality to sensors

## Documentation to Add

1. Create comprehensive README files for all new components:
   - `README_sensors.md` - Overview of the sensor system
   - `README_error_reporting.md` - Guide to error reporting
   - `README_shell_integration.md` - Guide to shell commands
   
2. Update existing documentation:
   - Update `README_auditor.md` to reference new sensing capabilities
   - Update module docstrings in all new files
   - Add diagrams showing the sensor system architecture
