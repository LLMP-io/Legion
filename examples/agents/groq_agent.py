"""Groq Agent Example

This example demonstrates using Legion's agent system with the Groq provider.
It tests various capabilities including:
- Basic text completion
- Tool usage (both external and internal tools)
- JSON output with schema validation
"""

import asyncio
import os
from typing import Annotated, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from legion import agent, tool

# Load environment variables
load_dotenv()

# Verify API key is present
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is not set")


# Define some tools for testing
@tool
def calculate_average(
    numbers: Annotated[List[float], Field(description="List of numbers to average")]
) -> float:
    """Calculate the average (mean) of a list of numbers."""
    return sum(numbers) / len(numbers)


@tool
def analyze_text(
    text: Annotated[str, Field(description="Text to analyze")],
    include_word_count: Annotated[bool, Field(description="Whether to include word count")] = True
) -> dict:
    """Analyze text and return statistics."""
    stats = {
        "length": len(text),
        "uppercase_count": sum(1 for c in text if c.isupper())
    }
    if include_word_count:
        stats["word_count"] = len(text.split())
    return stats


# Define a schema for JSON output
class WeatherInfo(BaseModel):
    """Schema for weather information"""

    temperature: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions (e.g., sunny, rainy)")
    precipitation_chance: Optional[float] = Field(
        description="Chance of precipitation (0-1)",
        ge=0,
        le=1
    )


@agent(
    model="groq:llama-3.3-70b-versatile",  # Use Groq's LLaMA model
    temperature=0.2,
    tools=[calculate_average, analyze_text]  # Bind external tools
)
class GroqAssistant:
    """A versatile assistant powered by Groq's LLaMA model.

    I can help with various tasks including:
    - Mathematical calculations
    - Text analysis
    - Weather information (in JSON format)
    - Custom text formatting
    """

    @tool
    def format_number(
        self,
        number: Annotated[float, Field(description="Number to format")],
        decimal_places: Annotated[int, Field(description="Number of decimal places")] = 2,
        prefix: Annotated[str, Field(description="Text to add before number")] = ""
    ) -> str:
        """Format a number with specified decimal places and optional prefix."""
        return f"{prefix}{number:.{decimal_places}f}"

    async def get_weather(self, location: str) -> WeatherInfo:
        """Get weather information in a structured format."""
        # Use JSON output with schema
        response = await self.aprocess(
            f"Generate realistic weather information for {location}",
            response_schema=WeatherInfo
        )
        return WeatherInfo.model_validate_json(response.content)


async def main():
    # Create an instance of our Groq-powered agent
    agent = GroqAssistant()

    print("1. Testing basic text completion:")
    response = await agent.aprocess(
        "What are the key benefits of using Groq's LLM services?"
    )
    print(response.content)
    print("\n" + "="*50 + "\n")

    print("2. Testing external tool (calculate_average):")
    response = await agent.aprocess(
        "Calculate the average of these numbers: 15.5, 20.5, 25.5, 30.5 "
        "and format the result to 1 decimal place"
    )
    print(response.content)
    print("\n" + "="*50 + "\n")

    print("3. Testing external tool (analyze_text):")
    response = await agent.aprocess(
        "Analyze this text: 'The QUICK brown FOX jumps over the lazy dog'"
    )
    print(response.content)
    print("\n" + "="*50 + "\n")

    print("4. Testing JSON output with schema:")
    weather = await agent.get_weather("San Francisco")
    print(f"Temperature: {weather.temperature}°C")
    print(f"Conditions: {weather.conditions}")
    if weather.precipitation_chance is not None:
        print(f"Precipitation chance: {weather.precipitation_chance*100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
