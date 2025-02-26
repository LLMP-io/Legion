"""Schema Output Example

This example demonstrates how to use output schemas with Legion agents.
It shows two scenarios:
1. Basic schema output without tools
2. Schema output with tool integration

Output schemas help ensure the agent's responses are structured consistently,
making them easier to process programmatically.
"""

from datetime import datetime
from typing import Annotated, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from legion import agent, schema, tool

load_dotenv()


# Define output schemas
@schema
class MovieReview(BaseModel):
    """Schema for a movie review"""

    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    rating: float = Field(description="Rating out of 10", ge=0, le=10)
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    recommended: bool = Field(description="Whether the movie is recommended")


@schema
class WeatherReport(BaseModel):
    """Schema for a detailed weather report"""

    location: str = Field(description="The location for the weather report")
    temperature: float = Field(description="Current temperature in Fahrenheit")
    conditions: str = Field(description="Current weather conditions")
    forecast: List[str] = Field(description="Weather forecast for next 3 days")
    last_updated: str = Field(description="When this report was generated")
    warnings: Optional[List[str]] = Field(
        description="Any active weather warnings",
        default=None
    )


# Example 1: Basic schema output without tools
@agent(
    model="openai:gpt-4o-mini",
    temperature=0.3
)
class MovieReviewAgent:
    """I am a movie critic that provides structured reviews."""

    pass


# Example 2: Schema output with tools
@tool
def get_weather(
    location: Annotated[str, Field(description="Location to get weather for")]
) -> str:
    """Simulate getting weather (in a real app, you'd call a weather API)"""
    return f"It's sunny and 72°F in {location}. Forecast: Clear skies for next 3 days."


@tool
def get_weather_warnings(
    location: Annotated[str, Field(description="Location to check warnings for")]
) -> List[str]:
    """Simulate getting weather warnings (in a real app, you'd call a weather API)"""
    # Just a mock response
    return [] if location.lower() != "miami" else ["Heat advisory in effect"]


@agent(
    model="openai:gpt-4o-mini",
    temperature=0.3,
    tools=[get_weather, get_weather_warnings]
)
class WeatherReportAgent:
    """I am a weather reporter that provides detailed, structured weather reports."""

    @tool
    def format_time(self) -> str:
        """Get current time in a formatted string"""
        return datetime.now().strftime("%Y-%m-%d %I:%M %p")


async def main():
    # Example 1: Movie review without tools
    print("Example 1: Movie Review (No Tools)")
    print("=" * 50)

    reviewer = MovieReviewAgent()
    response = await reviewer.aprocess(
        "Review the movie 'The Matrix' (1999)",
        response_schema=MovieReview
    )

    # Parse the response into our schema
    review = MovieReview.model_validate_json(response.content)

    print(f"Movie: {review.title} ({review.year})")
    print(f"Rating: {review.rating}/10")
    print("\nPros:")
    for pro in review.pros:
        print(f"- {pro}")
    print("\nCons:")
    for con in review.cons:
        print(f"- {con}")
    print(f"\nRecommended: {'Yes' if review.recommended else 'No'}")

    print("\n" + "=" * 50 + "\n")

    # Example 2: Weather Report (With Tools)
    print("Example 2: Weather Report (With Tools)")
    print("=" * 50)

    reporter = WeatherReportAgent()

    # Test for a location without warnings
    response = await reporter.aprocess(
        "Give me a weather report for San Francisco",
        response_schema=WeatherReport
    )

    # Parse the response into our schema
    report = WeatherReport.model_validate_json(response.content)

    print(f"Weather Report for {report.location}")
    print(f"Temperature: {report.temperature}°F")
    print(f"Conditions: {report.conditions}")
    print("\nForecast:")
    for day in report.forecast:
        print(f"- {day}")
    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"- {warning}")
    print(f"\nLast Updated: {report.last_updated}")

    print("\n" + "=" * 50 + "\n")

    # Test for a location with warnings
    print("Example 3: Weather Report with Warnings")
    print("=" * 50)

    response = await reporter.aprocess(
        "Give me a weather report for Miami during hurricane season",
        response_schema=WeatherReport
    )

    # Parse the response into our schema
    report = WeatherReport.model_validate_json(response.content)

    print(f"Weather Report for {report.location}")
    print(f"Temperature: {report.temperature}°F")
    print(f"Conditions: {report.conditions}")
    print("\nForecast:")
    for day in report.forecast:
        print(f"- {day}")
    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"- {warning}")
    print(f"\nLast Updated: {report.last_updated}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
