import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from agent_functions import calculate_fibonacci, fetch_delayed_greeting, prime_factors

from faaa.decorator.agent import Agent

# Initialize agent with default prefix path
agent = Agent()

# Example 1: Synchronous function using thread pool
agent.register(use_process=False)(calculate_fibonacci)

# Example 2: CPU-intensive task using process pool
agent.register(use_process=True)(prime_factors)

# Example 3: Asynchronous function
agent.register()(fetch_delayed_greeting)


async def main():
    # Initialize thread and process pools
    agent._thread_pool_executor = ThreadPoolExecutor()
    agent._process_pool_executor = ProcessPoolExecutor()

    # Register all tools with LLM
    await agent.register_tools()

    # Example usage
    print("Registered endpoints:")
    for schema in agent.tools.values():
        print(f"- {schema.entry_points}")

    # Test the functions
    print("\nTesting functions:")

    # Get wrapped functions from agent
    wrapped_fib = agent.tools[
        next(k for k, v in agent.tools.items() if "calculate_fibonacci" in v.entry_points)
    ].func
    wrapped_prime = agent.tools[
        next(k for k, v in agent.tools.items() if "prime_factors" in v.entry_points)
    ].func
    wrapped_greeting = agent.tools[
        next(k for k, v in agent.tools.items() if "fetch_delayed_greeting" in v.entry_points)
    ].func

    # Test fibonacci
    n = 10
    start = time.time()
    result = await wrapped_fib(n)
    duration = time.time() - start
    print(f"Fibonacci({n}) = {result} (took {duration:.2f}s)")

    # Test prime factors
    n = 84
    start = time.time()
    result = await wrapped_prime(n)
    duration = time.time() - start
    print(f"Prime factors of {n} = {result} (took {duration:.2f}s)")

    # Test async greeting
    name = "Alice"
    delay = 0.5
    start = time.time()
    result = await wrapped_greeting(name, delay)
    duration = time.time() - start
    print(f"Greeting result: {result} (took {duration:.2f}s)")

    # Cleanup
    agent._thread_pool_executor.shutdown()
    agent._process_pool_executor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
