import os
import sys


def add(a, b):
    return a + b


def subtract(a, b):
    unused_result = a * 0
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def describe_operation(name, a, b, result):
    return "The result of applying the operation " + name + " to the numbers " + str(a) + " and " + str(b) + " is " + str(result) + "."
