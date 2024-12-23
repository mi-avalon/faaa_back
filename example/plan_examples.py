"""Examples of using FaaA's generate_plan functionality.

Before running this example:
1. Create a .env file in the project root with:
   OPENAI_API_KEY="your-api-key"
   OPENAI_BASE_URL="https://openrouter.ai/api/v1"

2. Replace "your-api-key" with your actual OpenAI API key
"""

import asyncio

from agent_functions import calculate_fibonacci, fetch_delayed_greeting, prime_factors

from faaa.app import FaaA
from faaa.decorator.agent import Agent


async def main():
    # Create and configure agents
    agent = Agent()

    agent.register(use_process=False)(calculate_fibonacci)
    agent.register(use_process=True)(prime_factors)
    agent.register()(fetch_delayed_greeting)

    # Initialize FaaA
    async with FaaA() as fa:
        # Include our agent
        fa.include_agents(agent)
        # Register all tools
        await fa.register_agents()

        print("Simple Example: Calculate Fibonacci")
        # Simple query using one tool
        simple_plans = await fa.generate_plan("I need to calculate the 10th number in the Fibonacci sequence")
        for simple_plan in simple_plans:
            print(f"\nPlan ID: {simple_plan.id}")
            print(f"Description: {simple_plan.description}")
            print(f"Recommendation score: {simple_plan.recommendation_score}")
            print("\nSteps:")
            for i, step in enumerate(simple_plan.steps or [], 1):
                print(f"\nStep {i}:")
                print(f"Description: {step.description}")
                print(f"Tool: {step.suggested_tool}")
                print(f"Query: {step.sub_query}")
                print(f"Explanation: {step.explanation}")

            print("\n" + "=" * 50 + "\n")

        print("Example with Retry: Delayed Greetings")
        # Example showing retry for potentially failing operations
        retry_plans = await fa.generate_plan("""
        Send a delayed greeting to Alice with a 2 second delay. 
        Since network operations might fail, we should retry if needed.
        """)

        for retry_plan in retry_plans:
            print(f"\nPlan ID: {retry_plan.id}")
            print(f"Description: {retry_plan.description}")
            print(f"Recommendation score: {simple_plan.recommendation_score}")
            print("\nSteps:")
            for i, step in enumerate(retry_plan.steps or [], 1):
                print(f"\nStep {i}:")
                print(f"Description: {step.description}")
                print(f"Tool: {step.suggested_tool}")
                print(f"Query: {step.sub_query}")
                print(f"Explanation: {step.explanation}")
                print(f"Retry: {step.retry}")

            print("\n" + "=" * 50 + "\n")

        print("Example with Insufficient Tools")
        # Example showing tool recommendations when tools are insufficient
        insufficient_plan = (
            await fa.generate_plan("""
        I need to:
        1. Calculate the square root of 16
        2. Round the result to 2 decimal places
        """)
        )[0]

        print(f"\nPlan ID: {insufficient_plan.id}")
        print(f"Description: {insufficient_plan.description}")
        print(f"Recommendation score: {insufficient_plan.recommendation_score}")

        if insufficient_plan.recommendation_tools:
            print("\nRecommended Tools Needed:")
            for tool in insufficient_plan.recommendation_tools:
                print(f"\nTool: {tool.name}")
                print(f"Description: {tool.description}")
                print(f"Why: {tool.reason}")
                print("Parameters:")
                for param in tool.parameters:
                    print(f"- {param.name}: {param.type} ({'required' if param.required else 'optional'})")
                    print(f"  Description: {param.description}")
        else:
            print("Should no recommendations available but:")
            print("\nSteps:")
            for i, step in enumerate(insufficient_plan.steps or [], 1):
                print(f"\nStep {i}:")
                print(f"Description: {step.description}")
                print(f"Tool: {step.suggested_tool}")
                print(f"Query: {step.sub_query}")
                print(f"Explanation: {step.explanation}")
                print(f"Retry: {step.retry}")

        print("\n" + "=" * 50 + "\n")

        print("Complex Example: Mathematical Analysis")
        # Complex query combining multiple tools
        complex_plans = await fa.generate_plan("""
        I need to perform the following mathematical analysis:
        1. Calculate the 15th Fibonacci number
        2. Find its prime factors
        3. For each prime factor, get a delayed greeting with that number as the delay
        """)

        for complex_plan in complex_plans:
            print(f"\nPlan ID: {complex_plan.id}")
            print(f"Description: {complex_plan.description}")
            print(f"Recommendation score: {complex_plan.recommendation_score}")
            print("\nSteps:")
            for i, step in enumerate(complex_plan.steps or [], 1):
                print(f"\nStep {i}:")
                print(f"Description: {step.description}")
                print(f"Tool: {step.suggested_tool}")
                print(f"Query: {step.sub_query}")
                print(f"Explanation: {step.explanation}")

            if complex_plan.recommendation_tools:
                print("\nRecommended Tools:")
                for tool in complex_plan.recommendation_tools:
                    print(f"- {tool.name}: {tool.description}")


if __name__ == "__main__":
    asyncio.run(main())
