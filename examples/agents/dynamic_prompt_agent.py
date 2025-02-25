"""Dynamic System Prompt Example

This example demonstrates how to create an agent with a dynamic system prompt that changes
based on context and user preferences. It shows how to provide dynamic values during
process/aprocess calls.

Best Practices:
1. When using a dynamic system prompt, don't include a docstring on the agent class
2. Use descriptive section IDs for dynamic fields
3. Provide meaningful default values
4. Use callable defaults for dynamic values that should be computed at runtime
"""

from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field

from legion import agent, tool, system_prompt

load_dotenv()


def get_current_time() -> str:
    """Get formatted current time."""
    return datetime.now().strftime("%I:%M %p")


# Create a dynamic system prompt with section IDs for runtime updates
SYSTEM_PROMPT = system_prompt(
    sections=[
        system_prompt.SystemPromptSection(
            content="I am a helpful assistant that adapts my communication style "
                    "based on context.",
            is_dynamic=False
        ),
        system_prompt.SystemPromptSection(
            content="{mood}",
            is_dynamic=True,
            section_id="mood",
            default_value="neutral and ready to help"
        ),
        system_prompt.SystemPromptSection(
            content="{context}",
            is_dynamic=True,
            section_id="context",
            default_value="general assistance"
        ),
        system_prompt.SystemPromptSection(
            content="{time}",
            is_dynamic=True,
            section_id="time",
            default_value=get_current_time
        )
    ]
)


@tool
def get_weather(
    location: Annotated[str, Field(description="Location to get weather for")]
) -> str:
    """Simulate getting weather (in a real app, you'd call a weather API)."""
    # This is just a mock response
    return f"It's sunny and 72°F in {location}"


@agent(
    model="openai:gpt-4o-mini",
    temperature=0.7,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather]  # Binding external tools
)
class DynamicAssistant:
    # Note: Since we are using a dynamic system prompt, we don't need to define
    # the system prompt using a docstring

    # Internal tool specific to this agent
    @tool
    def get_current_time(self) -> str:
        """Get the current time."""
        return datetime.now().strftime("%I:%M %p")


async def main():
    # Create an instance
    assistant = DynamicAssistant()

    # Example 1: Using defaults (no dynamic values provided)
    print("Example 1 (Default Values):")
    print("System Prompt Before Process:")
    print(assistant._memory.messages[0].content)
    print("\nMaking Request...")

    response = await assistant.aprocess(
        "How are you feeling right now?",
        dynamic_values=None
    )

    print("\nSystem Prompt During Process:")
    print(assistant._memory.messages[0].content)
    print("\nResponse:")
    print(response.content)
    print("\n" + "=" * 50 + "\n")

    # Example 2: Provide dynamic values during process
    print("Example 2 (With Dynamic Values):")
    dynamic_values = {
        "mood": "energetic and enthusiastic, you really like to speak like a pirate",
        "context": "casual conversation",
        "time": "9:00 AM"
    }

    print("System Prompt Before Process:")
    print(assistant._memory.messages[0].content)
    print("\nMaking Request with dynamic values:", dynamic_values)

    response = await assistant.aprocess(
        "How are you feeling right now?",
        dynamic_values=dynamic_values
    )

    print("\nSystem Prompt During Process:")
    print(assistant._memory.messages[0].content)
    print("\nResponse:")
    print(response.content)
    print("\n" + "=" * 50 + "\n")

    # Example 3: Different context, different mood
    print("Example 3 (Different Context):")
    dynamic_values = {
        "mood": "professional and focused",
        "context": "weather reporting",
        "time": datetime.now().strftime("%I:%M %p")
    }

    print("System Prompt Before Process:")
    print(assistant._memory.messages[0].content)
    print("\nMaking Request with dynamic values:", dynamic_values)

    response = await assistant.aprocess(
        "What's the weather in San Francisco?",
        dynamic_values=dynamic_values
    )

    print("\nSystem Prompt During Process:")
    print(assistant._memory.messages[0].content)
    print("\nResponse:")
    print(response.content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
