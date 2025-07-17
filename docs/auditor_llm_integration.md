# Auditor and LLM Integration

This document explains how the FixWurx Auditor Agent integrates with Large Language Models (LLMs) to create a hybrid system that combines mathematical verification with natural language capabilities.

## Architecture Overview

The Auditor Agent is designed as a mathematical verification framework with specific integration points for LLM assistance. This creates a powerful hybrid system where:

1. The Auditor provides rigorous mathematical verification through formal methods
2. LLMs provide natural language understanding, context, and generation capabilities
3. Together, they form a complete system that can both verify and explain code changes

```
┌───────────────────┐                ┌───────────────────┐
│                   │                │                   │
│   LLM Services    │◄──────────────►│   Auditor Agent   │
│                   │                │                   │
└─────────┬─────────┘                └─────────┬─────────┘
          │                                    │
          │                                    │
          │         ┌───────────────┐          │
          └────────►│               │◄─────────┘
                    │  Orchestrator │
                    │               │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │               │
                    │  FixWurx Core │
                    │               │
                    └───────────────┘
```

## LLM Integration Points

The Auditor integrates with LLMs at several key points in the workflow:

### 1. Initial Obligation Extraction

- **Component**: `llm_integrations.py` → `ObligationExtractor` class
- **Process**: 
  - LLM analyzes user requirements or code changes
  - Extracts initial "goals" or high-level obligations
  - These become the starting points for Δ-closure calculations
- **Example**:
  ```python
  # From a user requirement like:
  # "Add secure login functionality to the system"
  
  # The LLM extracts:
  initial_obligations = ["authenticate_user", "authorize_user"]
  ```

### 2. Delta Rule Generation

- **Component**: `llm_integrations.py` → `DeltaRuleGenerator` class
- **Process**:
  - LLM analyzes coding patterns and best practices
  - Generates transformation rules for `delta_rules.json`
  - These define how high-level obligations transform into implementation requirements
- **Example**:
  ```json
  {
    "pattern": "authenticate_user",
    "transforms_to": ["validate_credentials", "manage_sessions", "handle_auth_errors"]
  }
  ```

### 3. Error Contextualization

- **Component**: `llm_integrations.py` → `ErrorContextualizer` class
- **Process**:
  - When the Auditor produces a FAIL result, the error details are sent to an LLM
  - The LLM provides human-readable explanations and suggested fixes
- **Example**:
  ```
  AUDITOR: Missing obligation "validate_credentials"
  LLM CONTEXT: "This means you need to implement a function that checks user credentials 
  against a secure database. Consider using bcrypt for password hashing."
  ```

### 4. Gap Analysis Enhancement

- **Component**: `llm_integrations.py` → `GapAnalyzer` class
- **Process**:
  - LLMs analyze the gaps identified by the Auditor
  - They provide detailed context about potential implementation approaches
- **Example**:
  ```
  AUDITOR: Identified gap in "rate_limiting" functionality
  LLM ANALYSIS: "Consider implementing a token bucket algorithm. This would 
  require tracking request timestamps and implementing a sliding window."
  ```

## Communication Mechanism

The Auditor and LLMs communicate through:

1. **File-based Communication**:
   - Auditor writes detailed YAML/JSON reports to disk
   - LLM components read and process these files
   - Orchestration layer manages the flow of information

2. **Direct API Integration**:
   - `llm_integrations.py` contains API connections to external LLM services
   - Supports various LLM providers (OpenAI, Anthropic, local models)
   - Handles authentication, rate limiting, and response parsing

3. **Event-based System**:
   - Orchestration layer uses an event system to coordinate components
   - `agent_coordinator.py` routes messages between components
   - `meta_agent.py` decides when to involve LLMs in the process

## Code Integration Example

Here's a simplified example of how the Auditor interacts with LLMs:

```python
# In auditor.py
def check_completeness(self):
    # Perform mathematical verification
    missing_obligations = self._calculate_missing_obligations()
    
    if missing_obligations:
        # Report error to LLM for contextualization
        from llm_integrations import ErrorContextualizer
        
        contextualizer = ErrorContextualizer()
        enhanced_error = contextualizer.contextualize(
            error_type="MISSING_OBLIGATION",
            details={"missing": missing_obligations}
        )
        
        # Log the enhanced error
        self.error_reporting.record_error(
            "MISSING_OBLIGATION",
            enhanced_error
        )
        
        return {
            "success": False,
            "details": {"missing": missing_obligations, "context": enhanced_error}
        }
    
    return {"success": True}
```

## Benefits of Hybrid Approach

1. **Mathematical Rigor + Natural Language Understanding**:
   - Auditor: Provides provable guarantees through formal methods
   - LLM: Provides natural language interpretation and generation

2. **Separation of Concerns**:
   - Verification logic remains pure and testable
   - Interpretation/suggestion logic can evolve independently

3. **Explainability**:
   - Mathematical verification results are translated into understandable explanations
   - Technical findings become actionable insights

4. **Enhanced Decision Making**:
   - Formal verification identifies what's missing or incorrect
   - LLM suggests how to address the issues

## Configuration

The LLM integration is configured in `auditor_config.yaml` under the `llm_integration` section:

```yaml
llm_integration:
  enabled: true
  provider: "openai"  # or "anthropic", "local", etc.
  model: "gpt-4"
  temperature: 0.2
  max_tokens: 1000
  api_key_env: "OPENAI_API_KEY"
  integration_points:
    - "obligation_extraction"
    - "delta_rule_generation"
    - "error_contextualization"
    - "gap_analysis"
```

## Future Enhancements

1. **Feedback Loop Integration**:
   - LLM learns from past audit results to improve future analyses
   - Auditor refines mathematical models based on LLM insights

2. **Proactive Suggestion**:
   - LLM suggests improvements before verification failures occur
   - Predicts likely issues based on code patterns

3. **Interactive Debugging**:
   - LLM provides interactive guidance during audit failure resolution
   - Step-by-step assistance for complex fixes
