# Calculator Application for FixWurx Testing

This intentionally buggy calculator application is designed to test and validate the bug detection and fixing capabilities of the FixWurx framework. The application contains 34 strategically placed bugs across different components to provide a comprehensive testing environment.

## Project Structure

```
calculator/
├── __init__.py
├── main.py                      # Main entry point (2 bugs)
├── operations/                  # Math operations
│   ├── __init__.py
│   ├── basic_operations.py      # Basic arithmetic (4 bugs)
│   └── advanced_operations.py   # Advanced math functions (6 bugs)
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── validation.py            # Input validation (6 bugs)
│   └── memory.py                # Calculation history (6 bugs)
├── ui/                          # User interface
│   ├── __init__.py
│   └── cli.py                   # Command-line interface (10 bugs)
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_basic_operations.py
│   ├── test_advanced_operations.py
│   ├── test_validation.py
│   ├── test_memory.py
│   └── test_cli.py
├── FIXWURX_TESTING_PLAN.md      # High-level testing strategy
├── FIXWURX_TEST_CASES.md        # Detailed test cases
└── README.md                    # This file
```

## Bug Summary

The application contains 34 intentional bugs distributed across different modules:

| Component | # of Bugs | Types of Bugs |
|-----------|-----------|---------------|
| Basic Operations | 4 | Logic errors, missing validation |
| Advanced Operations | 6 | Missing imports, incorrect implementations |
| Validation Utilities | 6 | Incorrect checks, error handling issues |
| Memory Utilities | 6 | Variable naming, functionality issues |
| CLI Interface | 10 | UX problems, error handling, validation |
| Main Module | 2 | Infinite loop, missing exception handling |

For a full inventory of bugs with detailed descriptions, refer to `FIXWURX_TEST_CASES.md`.

## How to Use This Test Environment

### Running the Application (Without Fixes)

If you want to run the application to observe the bugs in action:

```bash
python -m calculator.main
```

Note: Due to Bug MM1 (infinite loop), you'll need to use Ctrl+C to exit.

### Running the Test Suite

To run the entire test suite (most tests will fail due to the bugs):

```bash
python -m unittest discover calculator/tests
```

To run tests for a specific module:

```bash
python -m unittest calculator/tests/test_basic_operations.py
```

### Using with FixWurx

Follow the procedures outlined in `FIXWURX_TESTING_PLAN.md` to systematically test FixWurx's capabilities. The typical workflow is:

1. Feed a buggy module to FixWurx
2. Document the interaction (prompts, responses, actions)
3. Verify if the bug was fixed using the corresponding test
4. Record the outcome according to the data collection template

For example:

```bash
# Step 1: Run FixWurx on a specific module
fixwurx --analyze calculator/operations/basic_operations.py --focus subtract

# Step 2: After FixWurx interaction, verify the fix
python -m unittest calculator/tests/test_basic_operations.py::TestBasicOperations::test_subtract
```

## Testing Documentation

This repository includes two key documents:

1. **FIXWURX_TESTING_PLAN.md**: High-level testing strategy, including testing phases, expected outcomes, and the iterative improvement process.

2. **FIXWURX_TEST_CASES.md**: Detailed test cases for each bug, including:
   - Commands to run
   - Expected FixWurx actions
   - Verification methods
   - Data collection templates

## Expected Outcomes and Learning Process

Through this testing process, we expect to:

1. Validate FixWurx's bug detection and fixing capabilities
2. Identify strengths and weaknesses in the current implementation
3. Gather data for iterative improvement
4. Develop a robust error reporting system

The six possible outcomes for each test:
1. **Success with proper logging** - FixWurx correctly identifies and fixes the bug
2. **Failure without proper error information** - FixWurx fails silently
3. **Crash with limited error information** - FixWurx crashes but provides some error context
4. **Failure with detailed error information** - FixWurx fails but provides actionable information
5. **False positive** - FixWurx reports success but bug remains
6. **False negative** - FixWurx reports failure but bug is actually fixed

By systematically documenting these outcomes, we can improve FixWurx's capabilities over time.

## Contributing

To extend this test environment:
1. Add new modules with different types of bugs
2. Add corresponding test cases that will fail due to the bugs
3. Update the testing documentation accordingly

## License

This testing environment is provided for internal use only.
