"""Basic Tools Example

This example demonstrates how to create and use tools with a Legion agent.
It shows both standalone tools and agent-specific tools.
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List  # noqa: F401

from dotenv import load_dotenv
from pydantic import Field

from legion import agent, tool

load_dotenv()


# Standalone tool that can be shared between agents
@tool
def get_current_time(
    format: Annotated[str, Field(description="Format string for the datetime")] = (
        "%Y-%m-%d %H:%M:%S"
    )
) -> str:
    """Get the current time in a specified format."""
    return datetime.now().strftime(format)


@tool
def list_formatter(
    items: Annotated[List[str], Field(description="List of items to format")],
    prefix: Annotated[str, Field(description="Prefix for each item")] = "- "
) -> str:
    """Format a list of items with a prefix."""
    return "\n".join(f"{prefix}{item}" for item in items)


@agent(
    model="openai:gpt-4o-mini",
    temperature=0.2,
    tools=[get_current_time, list_formatter]  # Bind external tools
)
class NoteTaker:
    """An agent that helps with taking and formatting notes.

    I can create timestamps, format lists, and organize information.
    """

    # Tools can also be defined within the agent class, and as such,
    # are only available to that agent.
    # This makes it easy when reading Legion code to know which tools
    # are specific to an agent.
    @tool
    def create_note(
        self,
        title: Annotated[str, Field(description="Title of the note")],
        content: Annotated[str, Field(description="Content of the note")],
        add_timestamp: Annotated[bool, Field(
            description="Whether to add a timestamp"
        )] = True
    ) -> str:
        """Create a formatted note with an optional timestamp."""
        timestamp = (
            f"\nCreated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            if add_timestamp else ""
        )
        return f"# {title}\n\n{content}{timestamp}"


async def main():
    # Create an instance of our agent
    agent = NoteTaker()

    # Example 1: Create a simple note with current time
    response = await agent.aprocess(
        "Create a note titled 'Meeting Summary' with the content being a list of: "
        "'Discussed project timeline', 'Assigned tasks', 'Set next meeting date'. "
        "Make sure to format it as a nice list and include the timestamp."
    )
    print("Example 1 Response:")
    print(response.content)
    print("\n" + "=" * 50 + "\n")

    # Example 2: Get current time in a specific format
    response = await agent.aprocess(
        "What's the current time in the format: Month Day, Year (HH:MM)?"
    )
    print("Example 2 Response:")
    print(response.content)
    print("\n" + "=" * 50 + "\n")

    # Example 3: Complex note with multiple tool usage
    response = await agent.aprocess(
        "Create a note titled 'Daily Tasks' with today's date. Include a list of: "
        "'Check emails', 'Team standup', 'Code review'. Format it nicely with bullets "
        "and add the current time at the end."
    )
    print("Example 3 Response:")
    print(response.content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
