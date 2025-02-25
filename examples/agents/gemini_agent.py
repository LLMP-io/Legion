"""Basic Agent Example with Gemini

This example demonstrates how to create a simple agent using Legion's decorator syntax.
The agent can use both internal (nested) tools and external tools.
"""

from typing import Annotated, List

from dotenv import load_dotenv
from pydantic import Field

from legion import agent, tool

load_dotenv()


@tool
async def add_numbers(
    numbers: Annotated[List[float], Field(description="List of numbers to add together")]
) -> float:
    """Add a list of numbers together and return the sum."""
    return sum(numbers)


@tool
async def multiply(
    a: Annotated[float, Field(description="First number to multiply")],
    b: Annotated[float, Field(description="Second number to multiply")]
) -> float:
    """Multiply two numbers together."""
    return a * b


@agent(
    model="gemini:gemini-2.0-flash-exp",
    temperature=0.2,
    tools=[add_numbers, multiply],  # Bind external tools
    debug=True
)
class MathHelper:
    """An agent that helps with basic arithmetic and string operations.

    I can perform calculations and manipulate text based on your requests.
    I have both external tools for math operations and internal tools for formatting.
    """

    @tool
    async def format_result(
        self,
        number: Annotated[float, Field(description="Number to format")],
        prefix: Annotated[str, Field(description="Text to add before the number")] = (
            "Result: "
        )
    ) -> str:
        """Format a number with a custom prefix."""
        return f"{prefix}{number:.2f}"


async def main():
    # Create an instance of our agent
    agent = MathHelper()

    try:
        # Example 1: Using external add_numbers tool with internal format_result
        response = await agent.aprocess(
            "I have the numbers 1.5, 2.5, and 3.5. Can you add them together and format "
            "the result nicely?"
        )
        print("Example 1 Response:")
        print(response.content if response and response.content else "No response")
        print()

        # Example 2: Using external multiply tool with internal format_result
        response = await agent.aprocess(
            "Can you multiply 4.2 by 2.0 and then format the result with the prefix "
            "'The product is: '?"
        )
        print("Example 2 Response:")
        print(response.content if response and response.content else "No response")
        print()

        # Example 3: Complex operation using both external and internal tools
        response = await agent.aprocess(
            "I need to add the numbers 10.5 and 20.5, then multiply the result by 2, "
            "and format it nicely."
        )
        print("Example 3 Response:")
        print(response.content if response and response.content else "No response")
        print()

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
