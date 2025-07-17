# FixWurx Auditor Agent

The FixWurx Auditor Agent provides a mathematically rigorous framework for ensuring completeness, correctness, and meta-awareness of system changes. It serves as the mathematical brake that transforms FixWurx from an enthusiastic coder into a disciplined engineer with provable guarantees.

## Overview

The Auditor certifies:

1. **Completeness** - Every Δ-derived obligation exists in the implementation
2. **Correctness** - Global energy is at optimum and proofs have ≥ 90% coverage
3. **Meta-awareness** - All watch-dog invariants remain intact

It then emits a single authoritative Audit-Stamp that either approves the build or rejects it with a minimal defect log.

## Installation

1. Ensure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Configure the system using `auditor_config.yaml`
4. Define your delta rules in `delta_rules.json`

## Usage

### Basic Usage

Run the auditor using:

```bash
python run_auditor.py --config auditor_config.yaml
```

This will:
1. Scan the repository for implemented modules
2. Compute the Δ-closure of obligations
3. Verify that all obligations are fulfilled
4. Check energy metrics and proof coverage
5. Validate meta-awareness guards
6. Emit an Audit-Stamp

### Configuration

The `auditor_config.yaml` file controls the auditor's behavior:

```yaml
# Example configuration
repo_path: "."
data_path: "auditor_data"
delta_rules_file: "delta_rules.json"
thresholds:
  energy_delta: 1e-7
  lambda: 0.9
  bug_probability: 1.1e-4
  drift: 0.02
```

### Delta Rules

Define your term-rewriting system in `delta_rules.json`:

```json
[
  {
    "pattern": "authenticate_user",
    "transforms_to": ["validate_credentials", "manage_sessions"]
  },
  {
    "pattern": "store_data",
    "transforms_to": ["validate_data", "persist_data", "backup_data"]
  }
]
```

## Core Components

### Mathematical Foundations

- **Δ-Closure Algorithm**: Computes the minimal fixed point of obligation closure
- **PL Inequality**: Ensures global convergence of energy function
- **Chernoff Risk Bound**: Calculates residual bug probability
- **Lyapunov Martingale**: Monitors system stability

### Storage Systems

- **Graph Database**: Tracks relationships between components, bugs, and fixes
- **Time-Series Database**: Stores and analyzes metrics over time
- **Document Store**: Manages structured documents like error reports and benchmarks

### Analysis Systems

- **Benchmarking System**: Measures and tracks component performance
- **Trend Analysis**: Detects patterns and anomalies in metrics
- **Error Reporting**: Provides detailed context for failures

## Examples

### Verifying Completeness

```python
from auditor import Auditor

# Initialize auditor
auditor = Auditor("auditor_config.yaml")

# Run completeness check
result = auditor.check_completeness()
if result["success"]:
    print("All obligations fulfilled!")
else:
    print(f"Missing obligations: {result['missing']}")
```

### Running Benchmarks

```python
from benchmarking_system import BenchmarkingSystem, BenchmarkConfig

# Create benchmarking system
benchmarking = BenchmarkingSystem("benchmark_data")

# Define benchmark configuration
config = BenchmarkConfig(
    name="api_latency",
    target="api_server",
    benchmark_type="PERFORMANCE",
    command="curl -s -w '%{time_total}' http://localhost:8000/api/v1/status -o /dev/null",
    iterations=10
)

# Run benchmark
result = benchmarking.run_benchmark(config)
print(f"Average latency: {result.statistics['execution_time']['mean']} seconds")
```

### Querying Relationships

```python
from graph_database import GraphDatabase, Node, Edge

# Create graph database
db = GraphDatabase("graph_data")

# Find components impacted by a bug
bug_id = "bug-123"
impacted = db.find_impacted_components(bug_id)
print(f"Components impacted by {bug_id}: {[c.properties['name'] for c in impacted]}")
```

## Advanced Features

### LLM Integration

The Auditor Agent integrates with Large Language Models (LLMs) at key points in the workflow:

1. **Initial Obligation Extraction**: LLMs analyze user requirements to extract initial obligations
2. **Delta Rule Generation**: LLMs help generate transformation rules for the term-rewriting system
3. **Error Contextualization**: LLMs provide human-readable explanations for audit failures
4. **Gap Analysis Enhancement**: LLMs provide detailed context about potential implementation approaches

For full details, see [Auditor LLM Integration](docs/auditor_llm_integration.md).

### Audit Stamping

The Auditor generates a YAML Audit-Stamp for each audit:

```yaml
audit_stamp:
  status: PASS
  timestamp: "2025-07-12T23:30:00.000Z"
```

Or for failures:

```yaml
audit_stamp:
  status: FAIL
  reason: MISSING_OBLIGATION
  details:
    missing:
      - validate_user_input
      - handle_api_errors
  timestamp: "2025-07-12T23:30:00.000Z"
```

### Historical Context

Error reports include detailed historical context:

```
Error: ENERGY_NOT_MINIMUM
Related errors: ERR-20250710-1, ERR-20250711-3
Frequency: 3rd occurrence
Common components: ["resource_manager.py", "load_balancer.py"]
Last occurrence: 2 days ago
```

### Gap Analysis

The system can identify functionality gaps by comparing implementations against requirements:

```
Gap: Required functionality 'rate_limiting' not implemented
Component: api_server
Impact: 
  - Severity: HIGH
  - Affected systems: ["api_server", "auth_service"]
  - Performance delta: N/A
```

## Mathematical Background

The Auditor implements several key mathematical concepts:

1. **Δ-Closure (Lemma 14-Pieces)**: Ensures the minimal fixed point of rewriting is unique and finite.

2. **Functor-Injectivity**: Guarantees that two distinct obligations cannot survive AST-normalisation.

3. **PL Inequality**: For energy function E and optimum E*, the gradient norm satisfies:
   ‖∇E(S)‖² ≥ μ(E(S)−E*)

4. **Chernoff Risk Bound**: The residual bug probability is bounded by:
   P_bug ≤ (1−f)ρ + f⋅2e^(−2nε²)

5. **Lyapunov Martingale**: For state Φₜ, we ensure:
   E[Φ_{t+1}|F_t] ≤ Φ_t−δ

## Contributing

The Auditor is designed as a fixed verification engine. If you encounter issues or have suggestions for improving the verification logic, please open an issue with a detailed description of the mathematical reasoning behind your proposed change.

## License

Copyright (c) 2025 FixWurx. All Rights Reserved.
