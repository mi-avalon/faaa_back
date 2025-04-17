"""Functions used by the agent example."""
from numbers import Number
import asyncio
from faaa.core import Tool
tool = Tool()

@tool.add()
def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number using iteration.

    Args:
        n: The position in the Fibonacci sequence to calculate (must be >= 0)

    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

@tool.add()
def prime_factors(n: int) -> list[int]:
    """
    Calculate prime factors of a given number.

    Args:
        n: The number to factorize (must be > 1)

    Returns:
        A list of prime factors
    """
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        if d * d > n:
            if n > 1:
                factors.append(n)
            break
    return factors

@tool.add()
async def fetch_delayed_greeting(name: str, delay: float = 1.0) -> str:
    """
    Returns a greeting after a specified delay.

    Args:
        name: Name of the person to greet
        delay: Delay in seconds before returning greeting

    Returns:
        A personalized greeting message
    """
    await asyncio.sleep(delay)
    return f"Hello, {name}! Sorry for the {delay} second delay."

@tool.add()
def sum(a: Number, b: Number) -> Number:
    """
    Calculate the sum of two numbers.

    Args:
        a (Number): The first number.
        b (Number): The second number.

    Returns:
        Number: The sum of the two numbers.
    """
    return a + b

@tool.add()
def function_Sample():
    """
    This function do nothing.
    """

@tool.add()
def landslide_image_extraction(full_image):
    """
    Extract the landslide part from an image.

    Args:
        full_image: The original full image.

    Returns:
        sub_image: the landslide part from full_image
    """
    return full_image[1]

@tool.add()
def reference_serching(query):
    """
    In response to a query, relevant reference files are identified, broken down into segments, and the segment demonstrating the highest correlation is returned.

    Args:
        query: the query.

    Returns:
        reference: reference information to the query.
    """
    return query[0]

@tool.add()
def landslide_type_identifier(landslide_image):
    """
    This function can tell which type of the soil is in a given image of landslide.

    Args:
        landslide_image: The image of landslide.

    Returns:
        the name of an soil
    """
    return "a cool name of soil"

@tool.add()
def call_search_engine(key_word):
    """
    This function search the key word by Google search engine, return the first web page of the result.

    Args:
        key_word: The key word to be searched.

    Returns:
        link of a web page.
    """
    return "www.sample."