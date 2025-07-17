# FixWurx Test Cases: Calculator Application

This document provides detailed test cases for the FixWurx testing process using the intentionally buggy calculator application. Each test case includes specific steps, commands, and expected outcomes.

## Bug Inventory Summary

### Basic Operations (4 bugs)
| ID | File | Function | Bug Description |
|----|----|----------|----------------|
| B1 | basic_operations.py | subtract | Incorrect order of operands (`b - a` instead of `a - b`) |
| B2 | basic_operations.py | multiply | Using addition instead of multiplication |
| B3 | basic_operations.py | divide | No zero division check |
| B4 | basic_operations.py | power | Incorrect implementation (using `a * b` instead of `a ** b`) |

### Advanced Operations (6 bugs)
| ID | File | Function | Bug Description |
|----|----|----------|----------------|
| A1 | advanced_operations.py | N/A | Missing random import |
| A2 | advanced_operations.py | square_root | No negative number check |
| A3 | advanced_operations.py | factorial | Incorrect base case (returns 0 instead of 1 for n=0) |
| A4 | advanced_operations.py | logarithm | Wrong math function (using log10 instead of log with base) |
| A5 | advanced_operations.py | sine | Using cosine instead of sine function |
| A6 | advanced_operations.py | random_operation | Using undefined 'random' module |

### Validation Utilities (6 bugs)
| ID | File | Function | Bug Description |
|----|----|----------|----------------|
| V1 | validation.py | is_number | Only checking float conversion, not integer |
| V2 | validation.py | is_number | Too broad exception handling |
| V3 | validation.py | is_positive | Wrong comparison operator (>= instead of >) |
| V4 | validation.py | is_integer | Logic error in integer check |
| V5 | validation.py | is_in_range | Marked as a bug but implementation is correct |
| V6 | validation.py | validate_operation_inputs | Incomplete validation logic |

### Memory Utilities (6 bugs)
| ID | File | Function | Bug Description |
|----|----|----------|----------------|
| M1 | memory.py | CalculationHistory.__init__ | Using wrong variable name (size instead of max_size) |
| M2 | memory.py | add_calculation | Missing timestamp for each calculation |
| M3 | memory.py | add_calculation | Not enforcing max size correctly |
| M4 | memory.py | get_last_calculation | No empty check |
| M5 | memory.py | clear_history | Inefficient way to clear list |
| M6 | memory.py | get_calculations_by_operation | Marked as a bug but implementation is correct |

### CLI Interface (10 bugs)
| ID | File | Function | Bug Description |
|----|----|----------|----------------|
| C1 | cli.py | CalculatorCLI.__init__ | Incorrect initialization (minor issue) |
| C2 | cli.py | CalculatorCLI.__init__ | Missing operation mapping for 'random' |
| C3 | cli.py | get_operation | No error handling for invalid operation |
| C4 | cli.py | calculate | Not validating inputs properly |
| C5 | cli.py | calculate | Not handling missing second parameter |
| C6 | cli.py | calculate | Recording history inconsistently |
| C7 | cli.py | calculate | Swallowing exceptions |
| C8 | cli.py | display_menu | Incomplete menu |
| C9 | cli.py | parse_input | Overly simplistic parsing |
| C10 | cli.py | parse_input | Not providing specific error messages |

### Main Module (2 bugs)
| ID | File | Function | Bug Description |
|----|----|----------|----------------|
| MM1 | main.py | main | Infinite loop with no exit condition |
| MM2 | main.py | __main__ | No exception handling for the main function |

## Detailed Test Cases

### Test Case Group 1: Basic Operations

#### Test Case B1: Subtraction Bug
**Command:**
```
fixwurx --analyze calculator/operations/basic_operations.py --focus subtract
```

**Expected FixWurx Actions:**
1. Identify incorrect operand order in the subtract function
2. Suggest changing `return b - a` to `return a - b`
3. Apply the fix after user confirmation

**Verification Command:**
```
python -m unittest calculator/tests/test_basic_operations.py::TestBasicOperations::test_subtract
```

**Expected Outcome:**
- Tests pass after FixWurx applies the fix
- FixWurx logs the successful fix

#### Test Case B2: Multiplication Bug
**Command:**
```
fixwurx --analyze calculator/operations/basic_operations.py --focus multiply
```

**Expected FixWurx Actions:**
1. Identify incorrect operation (addition instead of multiplication)
2. Suggest changing `return a + b` to `return a * b`
3. Apply the fix after user confirmation

**Verification Command:**
```
python -m unittest calculator/tests/test_basic_operations.py::TestBasicOperations::test_multiply
```

**Expected Outcome:**
- Tests pass after FixWurx applies the fix
- FixWurx logs the successful fix

#### Test Case B3: Division Zero Check Bug
**Command:**
```
fixwurx --analyze calculator/operations/basic_operations.py --focus divide
```

**Expected FixWurx Actions:**
1. Identify missing zero division check
2. Suggest adding code to check if b is zero before division
3. Apply the fix after user confirmation

**Verification Command:**
```
python -m unittest calculator/tests/test_basic_operations.py::TestBasicOperations::test_divide
```

**Expected Outcome:**
- Tests pass after FixWurx applies the fix
- FixWurx logs the successful fix

#### Test Case B4: Power Implementation Bug
**Command:**
```
fixwurx --analyze calculator/operations/basic_operations.py --focus power
```

**Expected FixWurx Actions:**
1. Identify incorrect power implementation
2. Suggest changing `return a * b` to `return a ** b`
3. Apply the fix after user confirmation

**Verification Command:**
```
python -m unittest calculator/tests/test_basic_operations.py::TestBasicOperations::test_power
```

**Expected Outcome:**
- Tests pass after FixWurx applies the fix
- FixWurx logs the successful fix

### Test Case Group 2: Advanced Operations

#### Test Case A1 & A6: Random Import and Usage Bugs
**Command:**
```
fixwurx --analyze calculator/operations/advanced_operations.py --focus random_operation
```

**Expected FixWurx Actions:**
1. Identify missing random import
2. Suggest adding `import random` at the top of the file
3. Apply the fix after user confirmation

**Verification Command:**
```
python -m unittest calculator/tests/test_advanced_operations.py::TestAdvancedOperations::test_random_operation
```

**Expected Outcome:**
- Tests pass after FixWurx applies the fix
- FixWurx logs the successful fix

#### Test Case A2: Square Root Negative Check Bug
**Command:**
```
fixwurx --analyze calculator/operations/advanced_operations.py --focus square_root
```

**Expected FixWurx Actions:**
1. Identify missing negative number validation
2. Suggest adding code to check if x is negative before calling math.sqrt
3. Apply the fix after user confirmation

**Verification Command:**
```
python -m unittest calculator/tests/test_advanced_operations.py::TestAdvancedOperations::test_square_root
```

**Expected Outcome:**
- Tests pass after FixWurx applies the fix
- FixWurx logs the successful fix

### Test Case Group 3: Integration Tests

#### Integration Test 1: CLI and Operations Interdependencies
**Command:**
```
fixwurx --analyze calculator/ui/cli.py calculator/operations/advanced_operations.py
```

**Expected FixWurx Actions:**
1. Identify the missing random operation mapping in CLI (C2)
2. Identify the missing random import in advanced_operations.py (A1)
3. Recognize the dependency between these bugs
4. Suggest fixing the import issue first, then the mapping issue
5. Apply fixes in the correct order after user confirmation

**Verification Command:**
```
python -m unittest calculator/tests/test_cli.py::TestCalculatorCLI::test_initialization calculator/tests/test_advanced_operations.py::TestAdvancedOperations::test_random_operation
```

**Expected Outcome:**
- FixWurx identifies and addresses the interdependent bugs in the correct order
- Both tests pass after fixes are applied
- FixWurx logs show dependency awareness

#### Integration Test 2: Input Validation Chain
**Command:**
```
fixwurx --analyze calculator/utils/validation.py calculator/ui/cli.py
```

**Expected FixWurx Actions:**
1. Identify validation issues in validation.py
2. Identify how these impact the CLI's input handling
3. Suggest fixes in logical order (fix validation utilities first, then CLI usage)
4. Apply fixes after user confirmation

**Verification Command:**
```
python -m unittest calculator/tests/test_validation.py calculator/tests/test_cli.py::TestCalculatorCLI::test_calculate
```

**Expected Outcome:**
- FixWurx addresses the validation bugs first, then the CLI bugs
- All tests pass after fixes are applied
- FixWurx logs show awareness of dependency chain

### Test Case Group 4: System-Wide Tests

#### Full Application Test
**Command:**
```
fixwurx --analyze calculator/ --comprehensive
```

**Expected FixWurx Actions:**
1. Analyze the entire application structure
2. Identify all 34 bugs
3. Create a dependency graph for fixes
4. Prioritize critical bugs (e.g., those causing crashes)
5. Suggest a comprehensive fix plan
6. Apply fixes in optimal order after user confirmation

**Verification Command:**
```
python -m unittest discover calculator/tests
```

**Expected Outcome:**
- FixWurx proposes a logical fix sequence
- All tests pass after fixes are applied
- FixWurx logs show comprehensive analysis
- Performance metrics are within acceptable ranges

## Data Collection Template

For each test case, record the following information:

```
Test Case ID: [ID]
Date/Time: [Timestamp]
FixWurx Version: [Version]
Command Used: [Command]

Analysis Phase:
- Time to initial bug detection: [seconds]
- Number of bugs correctly identified: [count]
- Number of bugs missed: [count]
- Dependencies correctly identified: [yes/no]

Interaction Phase:
- Number of clarification questions: [count]
- Quality of explanations provided: [1-5 rating]
- User guidance clarity: [1-5 rating]

Fix Implementation Phase:
- Time to implement fixes: [seconds]
- Fix correctness: [correct/partially correct/incorrect]
- Code quality of fixes: [1-5 rating]

Verification Phase:
- Tests passing before fix: [count]/[total]
- Tests passing after fix: [count]/[total]
- Any new issues introduced: [yes/no + details]

Outcome Category: [Success/Failure category]
Error Information Quality: [High/Medium/Low]
Notes: [Any additional observations]
```

## Testing Schedule

The suggested order for executing these tests:

1. Individual basic operation bugs (B1-B4)
2. Individual advanced operation bugs (A1-A6)
3. Individual validation bugs (V1-V6)
4. Individual memory management bugs (M1-M6)
5. Individual CLI interface bugs (C1-C10)
6. Integration tests
7. Full application test

Record all results following the data collection template and compile into a comprehensive report after completing all tests.
