# FAAA - Flexible AI Agent Architecture

FAAA is a powerful Python framework for building AI agents that seamlessly integrates with Large Language Models (LLMs). It provides a flexible and intuitive way to create, manage, and execute AI-powered tools and agents.

## Features

- **Decorator-based Agent Creation**: Simple and intuitive way to create AI agents using Python decorators
- **Automatic Tool Description Generation**: Uses LLM to automatically generate tool descriptions and schemas
- **Flexible Execution Models**: 
  - Support for both synchronous and asynchronous functions
  - Thread pool execution for I/O-bound operations
  - Process pool execution for CPU-intensive tasks
- **Robust LLM Integration**:
  - Built-in OpenAI API support
  - Chat completions with structured outputs
  - Embeddings generation
  - Function calling capabilities
- **Error Handling & Reliability**:
  - Automatic retries with exponential backoff
  - Comprehensive error handling
  - Token limit management

## Installation

```bash
pip install faaa
```

## Quick Start

Here's a simple example of creating an AI agent:

```python
from faaa.decorator.agent import Agent

# Initialize agent
agent = Agent()

# Register a synchronous function
@agent.register(use_process=False)
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

# Register a CPU-intensive task
@agent.register(use_process=True)
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

# Register an async function
@agent.register()
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
```

## Usage

1. Initialize thread and process pools:
```python
agent._thread_pool_executor = ThreadPoolExecutor()
agent._process_pool_executor = ProcessPoolExecutor()
```

2. Register tools with LLM:
```python
await agent.register_tools()
```

3. Access registered tools:
```python
# Get all registered tools
for schema in agent.tools.values():
    print(f"- {schema.entry_points}")

# Get a specific tool
wrapped_fib = agent.tools[
    next(k for k, v in agent.tools.items() if "calculate_fibonacci" in v.entry_points)
].func
```

4. Execute tools:
```python
# Execute synchronous function
result = await wrapped_fib(10)

# Execute CPU-intensive task
result = await wrapped_prime(84)

# Execute async function
result = await wrapped_greeting("Alice", 0.5)
```

## Advanced Features

### LLM Integration

FAAA provides comprehensive LLM integration through the `LLMClient` class:

```python
from faaa.core.llm import LLMClient

client = LLMClient()

# Chat completion
response = await client.chat("Hello, how are you?")

# Generate embeddings
embeddings = await client.embeddings("Hello world")

# Structured output parsing
from pydantic import BaseModel

class MyStructure(BaseModel):
    field1: str
    field2: int

result = await client.structured_output(
    "Parse this message",
    structured_outputs=MyStructure
)
```

### Tool Schema Generation

FAAA automatically generates tool descriptions using LLM analysis of your functions:

```python
schema = await client.generate_tool_description(my_function)
```

## Configuration

Configure the LLM client using environment variables:

```bash
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # Optional
```

## License

MIT License - see LICENSE file for details.
