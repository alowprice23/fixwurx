# Collaborative Improvement Framework (CIF)

This document outlines the implementation of the Collaborative Improvement Framework (CIF) components from the LLM Shell Integration Plan v4.

## Overview

The Collaborative Improvement Framework (CIF) identifies patterns in user interactions, proposes improvements, and enables peer review workflows for enhancing the system's capabilities over time. It serves as the self-improvement mechanism of the system, allowing it to learn from user behavior and evolve.

## Components Implemented

### 1. Pattern Detector (`PatternDetector`)

The Pattern Detector identifies repetitive sequences of commands in user interactions:

- **Command History Tracking**: Records command execution history with context
- **Sequence Analysis**: Identifies repeating patterns in command sequences
- **Similarity Detection**: Uses difflib to detect similar but not identical patterns
- **Pattern Persistence**: Stores detected patterns for long-term learning

### 2. Script Proposer (`ScriptProposer`)

The Script Proposer generates script proposals based on detected patterns:

- **Automated Script Generation**: Creates scripts to automate repetitive tasks
- **Neural Matrix Integration**: Uses LLM for intelligent script creation
- **Proposal Management**: Tracks and manages script proposals
- **Rate Limiting**: Prevents overwhelming users with too many proposals

### 3. Peer Review Workflow (`PeerReviewWorkflow`)

The Peer Review Workflow manages the review and approval process for proposed scripts:

- **Review Management**: Tracks reviews, comments, and approval status
- **Approval Thresholds**: Implements configurable approval requirements
- **Comment System**: Enables collaborative discussion of proposals
- **Stalled Review Detection**: Identifies and handles stalled reviews

### 4. Decision Tree Growth (`DecisionTreeGrowth`)

The Decision Tree Growth component integrates approved scripts into the decision tree:

- **Tree Integration**: Adds approved scripts to the decision tree
- **Metadata Association**: Links patterns with scripts in the decision tree
- **Node Creation**: Creates appropriate nodes for script execution

## Implementation Details

### Pattern Detection

The Pattern Detector uses a sliding window approach to identify repeating command sequences:

```python
def _detect_patterns(self) -> None:
    """
    Detect repetitive patterns in the command history.
    """
    # Extract commands from history
    commands = [entry["command"] for entry in self.command_history]
    
    # Skip if not enough commands
    if len(commands) < self.min_sequence_length * self.min_repetitions:
        return
    
    # Find sequences of commands that repeat
    for sequence_length in range(self.min_sequence_length, min(10, len(commands) // 2)):
        self._find_repeating_sequences(commands, sequence_length)
```

The similarity detection mechanism allows for flexible pattern matching:

```python
def _calculate_similarity(self, sequence1: Tuple[str, ...], sequence2: Tuple[str, ...]) -> float:
    """
    Calculate similarity between two command sequences.
    """
    # Calculate string similarity using difflib
    similarity_sum = 0
    for cmd1, cmd2 in zip(sequence1[:min(len1, len2)], sequence2[:min(len1, len2)]):
        similarity = difflib.SequenceMatcher(None, cmd1, cmd2).ratio()
        similarity_sum += similarity
    
    return similarity_sum / min(len1, len2)
```

### Script Generation

The Script Proposer generates scripts based on detected patterns using the Neural Matrix:

```python
def _generate_script(self, pattern: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a script based on a pattern.
    """
    # Get neural matrix for script generation
    neural_matrix = self.registry.get_component("neural_matrix")
    if neural_matrix:
        # Format pattern for prompt
        sequence_str = "\n".join([f"  {i+1}. {cmd}" for i, cmd in enumerate(pattern["sequence"])])
        
        # Generate script using neural matrix
        prompt = f"""
        Generate a script that automates the following sequence of commands:
        
        {sequence_str}
        
        The script should:
        1. Accept appropriate parameters for flexibility
        2. Include error handling
        3. Provide helpful usage information
        4. Follow best practices for shell scripting
        """
        
        script_content = neural_matrix.generate_text(prompt=prompt, max_tokens=1000)
        # ...
```

### Peer Review Process

The Peer Review Workflow manages the entire review lifecycle:

```python
def submit_review(self, proposal_id: str, reviewer_id: str, 
                 approved: bool, comments: str) -> Dict[str, Any]:
    """
    Submit a review for a script proposal.
    """
    # ...
    # Add review
    review = {
        "reviewer_id": reviewer_id,
        "approved": approved,
        "comments": comments,
        "timestamp": time.time()
    }
    
    proposal["reviews"][reviewer_id] = review
    
    # Check if we have enough approvals or rejections
    approval_count = sum(1 for r in proposal["reviews"].values() if r["approved"])
    rejection_count = sum(1 for r in proposal["reviews"].values() if not r["approved"])
    
    if approval_count >= self.min_approvals:
        proposal["status"] = "approved"
        proposal["approved"] = True
    # ...
```

### Decision Tree Integration

The Decision Tree Growth component extends the decision tree with new scripts:

```python
def add_script_to_tree(self, script_id: str) -> Dict[str, Any]:
    """
    Add a script to the decision tree.
    """
    # ...
    # Get decision tree
    decision_tree = self.registry.get_component("decision_tree")
    if not decision_tree:
        return {"success": False, "error": "Decision Tree not available"}
    
    # Add script to decision tree
    tree_result = decision_tree.add_script_node(script_id, script["metadata"])
    # ...
```

## Configuration Options

The Collaborative Improvement Framework provides various configuration options:

### Pattern Detector Configuration

```json
{
  "pattern_detector": {
    "min_sequence_length": 3,
    "min_repetitions": 2,
    "similarity_threshold": 0.8,
    "max_history_size": 1000
  }
}
```

### Script Proposer Configuration

```json
{
  "script_proposer": {
    "proposal_threshold": 3,
    "max_proposals_per_day": 5
  }
}
```

### Peer Review Workflow Configuration

```json
{
  "peer_review_workflow": {
    "min_approvals": 2,
    "review_timeout_days": 7
  }
}
```

## Integration with Other Components

The Collaborative Improvement Framework integrates with other system components:

- **Neural Matrix**: Used for script generation and pattern analysis
- **Script Library**: Stores approved scripts
- **Decision Tree**: Integrates approved scripts for execution
- **Command Executor**: Executes scripts and records commands for pattern detection

## Workflow Example

1. **Pattern Detection**:
   ```
   User executes commands:
   $ ls -la
   $ grep "error" log.txt
   $ cat log.txt | grep "error" > errors.txt
   ...
   User later repeats similar sequence
   ```

2. **Pattern Identification**:
   ```
   Pattern detector identifies repeating sequence: 
   [grep "error" log.txt, cat log.txt | grep "error" > errors.txt]
   ```

3. **Script Proposal**:
   ```
   Script proposer generates a script:
   #!/usr/bin/env bash
   # Extract errors from log file
   
   LOG_FILE="${1:-log.txt}"
   OUTPUT_FILE="${2:-errors.txt}"
   
   if [ ! -f "$LOG_FILE" ]; then
     echo "Error: Log file $LOG_FILE not found"
     exit 1
   fi
   
   grep "error" "$LOG_FILE" > "$OUTPUT_FILE"
   echo "Errors extracted to $OUTPUT_FILE"
   ```

4. **Peer Review**:
   ```
   Review 1: ✅ Approved - "Good script, handles file existence checks"
   Review 2: ✅ Approved - "Added parameters for flexibility"
   → Status: Approved
   ```

5. **Script Committal**:
   ```
   Script added to library with metadata:
   {
     "name": "extract_errors",
     "description": "Extracts error lines from a log file",
     "tags": ["log", "error", "auto-generated"],
     "pattern_id": "pattern_1234567"
   }
   ```

6. **Decision Tree Growth**:
   ```
   Script added to decision tree for future suggestion when user 
   performs similar operations
   ```

## Future Enhancements

1. **Advanced Pattern Recognition**: Implement more sophisticated pattern recognition using ML techniques
2. **Interactive Script Refinement**: Allow users to refine proposed scripts interactively
3. **Usage Analytics**: Track script usage and effectiveness over time
4. **Script Evolution**: Allow scripts to evolve based on user feedback and changing patterns
5. **Multi-Modal Patterns**: Detect patterns that span different interfaces (CLI, GUI, etc.)

## Test Results

The tests (`test_collaborative_improvement.py`) verify the following functionality:

1. **Pattern Detection**: Correctly identifies repeating command sequences
2. **Similarity Detection**: Accurately calculates similarity between command sequences
3. **Script Generation**: Generates appropriate scripts based on patterns
4. **Review Process**: Properly manages the review workflow
5. **Script Committal**: Successfully commits approved scripts to the library

All implemented features pass their respective tests, demonstrating the robustness of the Collaborative Improvement Framework implementation.
