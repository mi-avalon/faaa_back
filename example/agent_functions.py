"""Functions used by the agent example."""

import asyncio


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
