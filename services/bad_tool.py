import random

from langchain.tools import tool


@tool
def even_odd(input) -> str:
    """Give me a number in binary and I will tell you if it's even or odd"""
    result = random.choice(["even", "odd"])  # example 50% of the time wrong result
    print(f"{input} is {result}")
    return result


@tool
def fahrenheit_to_cel(input) -> str:
    """Give me a temp in Fahrenheit and I will tell you the value in Celsius"""
    result = "23.44"  # example wrong result
    print(f"{input} is {result}")
    return result

