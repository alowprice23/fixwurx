from calculator.operations import basic_operations, advanced_operations
from calculator.utils import validation, memory


class CalculatorCLI:
    """Command-line interface for the calculator."""

    # Argument arity definitions ------------------------------------------------
    ZERO_ARG_OPERATIONS = {"random"}
    SINGLE_ARG_OPERATIONS = {"sqrt", "factorial", "sin"}

    def __init__(self):
        """Initialise a new calculator CLI instance with its own history."""
        self.history = memory.CalculationHistory()
        self.operations = {
            "add": basic_operations.add,
            "subtract": basic_operations.subtract,
            "multiply": basic_operations.multiply,
            "divide": basic_operations.divide,
            "power": basic_operations.power,
            "modulus": basic_operations.modulus,
            "sqrt": advanced_operations.square_root,
            "factorial": advanced_operations.factorial,
            "log": advanced_operations.logarithm,
            "sin": advanced_operations.sine,
            "random": advanced_operations.random_operation,  # newly exposed
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def get_operation(self, operation_name):
        """Return the function associated with *operation_name*."""
        try:
            return self.operations[operation_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported operation '{operation_name}'.") from exc

    def _validate_operands(self, op_name, a, b):
        """Validate and return a tuple of operands suitable for *op_name*."""

        # Zero-argument operations -------------------------------------
        if op_name in self.ZERO_ARG_OPERATIONS:
            if a is not None or b is not None:
                raise ValueError(f"Operation '{op_name}' does not take any operands.")
            return tuple()

        # Single-argument operations -----------------------------------
        if op_name in self.SINGLE_ARG_OPERATIONS:
            if not validation.is_number(a):
                raise ValueError("Operand must be a valid number")
            return (a,)

        # Two-argument operations --------------------------------------
        if b is None:
            raise ValueError(f"Operation '{op_name}' requires two operands.")

        if not validation.is_number(a):
            raise ValueError("First operand must be a valid number")
        if not validation.is_number(b):
            raise ValueError("Second operand must be a valid number")

        # Special cases ------------------------------------------------
        if op_name == "divide" and b == 0:
            raise ValueError("Cannot divide by zero")
        if op_name == "modulus" and b == 0:
            raise ValueError("Cannot perform modulus by zero")

        return (a, b)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calculate(self, operation_name, a=None, b=None):
        """Perform *operation_name* on the supplied operands and store the result."""
        try:
            operation = self.get_operation(operation_name)
            operands = self._validate_operands(operation_name, a, b)

            # Execute ---------------------------------------------------
            result = operation(*operands)

            # Uniform history recording --------------------------------
            if len(operands) == 0:
                self.history.add_calculation(operation_name, None, None, result)
            elif len(operands) == 1:
                self.history.add_calculation(operation_name, operands[0], None, result)
            else:
                self.history.add_calculation(operation_name, operands[0], operands[1], result)

            return result
        except Exception as e:
            return f"Error: {str(e)}"

    # ------------------------------------------------------------------
    # User-facing helpers
    # ------------------------------------------------------------------
    def display_menu(self):
        """Print a friendly help/usage screen to the console."""
        print("=== Calculator Menu ===")
        print("Enter commands in the following format:\n")
        print("    <operation> <number1> [<number2>]\n")
        print("Examples:")
        print("    add 5 10        → 15")
        print("    sqrt 9          → 3\n")
        print("Available operations:")
        for op in sorted(self.operations):
            if op in self.ZERO_ARG_OPERATIONS:
                req = "0 args"
            elif op in self.SINGLE_ARG_OPERATIONS:
                req = "1 arg"
            else:
                req = "2 args"
            print(f" - {op:<10} ({req})")

    def parse_input(self, user_input):
        """Parse *user_input* and return *(operation, a, b)*."""
        parts = user_input.strip().split()
        if not parts:
            raise ValueError("Empty input.")

        operation = parts[0].lower()
        if operation not in self.operations:
            raise ValueError(f"Unsupported operation '{operation}'.")

        # Zero-argument handling --------------------------------------
        if operation in self.ZERO_ARG_OPERATIONS:
            if len(parts) > 1:
                raise ValueError(f"Operation '{operation}' does not accept operands.")
            return operation, None, None

        # First operand ------------------------------------------------
        try:
            a = float(parts[1])
        except IndexError:
            raise ValueError("Missing first operand.")
        except ValueError as exc:
            raise ValueError("First operand is not a valid number.") from exc

        # Second operand (if required) ---------------------------------
        b = None
        if operation not in self.SINGLE_ARG_OPERATIONS:
            try:
                b = float(parts[2])
            except IndexError:
                raise ValueError(f"Operation '{operation}' requires two operands.")
            except ValueError as exc:
                raise ValueError("Second operand is not a valid number.") from exc

        return operation, a, b
