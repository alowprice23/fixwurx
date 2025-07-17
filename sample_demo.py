
def calculate_average(numbers):
    total = 0
    for number in numbers:
        total += number
    return total / len(numbers)  # Bug: Potential division by zero

def process_data(data):
    result = []
    for item in data:
        if item > 0:
            # Bug: Using append incorrectly
            result.append = item * 2
    return result

def recursive_function(n):
    if n <= 0:
        return 1
    # Bug: No base case for n=1, will cause infinite recursion
    return n * recursive_function(n-1)

def get_value(dictionary, key):
    # Bug: No key error handling
    return dictionary[key]

# Bug: Unused import
import os

# Bug: Bare except
try:
    x = 1 / 0
except:
    pass
