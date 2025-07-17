"""
Main entry point for the calculator application.
Contains intentional bugs for testing FixWurx functionality.
"""
from calculator.ui.cli import CalculatorCLI

def main():
    """Run the calculator application."""
    calculator = CalculatorCLI()
    calculator.display_menu()
    
    # BUG 33: Infinite loop with no exit condition
    while True:
        user_input = input("\nEnter operation and numbers (e.g., 'add 5 3'): ")
        
        # Should have an exit condition
        # if user_input.lower() == 'exit':
        #     break
            
        operation, a, b = calculator.parse_input(user_input)
        
        if operation is None:
            print("Invalid input. Please try again.")
            continue
            
        result = calculator.calculate(operation, a, b)
        print(f"Result: {result}")

if __name__ == "__main__":
    # BUG 34: No exception handling for the main function
    main()
    # Should have try-except block to catch and log unhandled exceptions
