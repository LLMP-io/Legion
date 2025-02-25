"""Basic Team Example

This example demonstrates how to create a team of agents using Legion's decorator syntax.
The team consists of:
1. A research leader that coordinates tasks
2. A data analyst that processes numbers
3. A writer that creates reports
"""

from typing import Annotated, Dict, List

from dotenv import load_dotenv
from pydantic import Field

from legion import agent, leader, team, tool
from legion.memory.providers.memory import InMemoryProvider

load_dotenv()


# Create specialized tools for team members
@tool
def analyze_numbers(
    numbers: Annotated[List[float], Field(description="List of numbers to analyze")]
) -> Dict[str, float]:
    """Analyze a list of numbers and return basic statistics."""
    if not numbers:
        return {"mean": 0, "min": 0, "max": 0}
    return {
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers)
    }


@tool
def format_report(
    title: Annotated[str, Field(description="Report title")],
    sections: Annotated[List[str], Field(description="List of report sections")],
    summary: Annotated[str, Field(description="Executive summary")] = None
) -> str:
    """Format a professional report with sections."""
    parts = [f"# {title}\n"]
    if summary:
        parts.extend(["\n## Executive Summary", summary])
    for i, section in enumerate(sections, 1):
        parts.extend([f"\n## Section {i}", section])
    return "\n".join(parts)


# Define the research team using decorator syntax
@team
class ResearchTeam:
    """A team that collaborates on research tasks."""

    # Shared memory provider for the team
    memory = InMemoryProvider()

    @leader(
        model="openai:gpt-4o-mini",
        temperature=0.2
    )
    class Leader:
        """Research team coordinator who delegates tasks and synthesizes results."""

        pass

    @agent(
        model="openai:gpt-4o-mini",
        temperature=0.1,
        tools=[analyze_numbers]
    )
    class Analyst:
        """Data analyst who processes numerical data and provides statistical insights."""

        @tool
        def interpret_stats(
            self,
            stats: Annotated[Dict[str, float], Field(
                description="Statistics to interpret"
            )]
        ) -> str:
            """Interpret statistical results in plain language."""
            return (
                f"The data shows an average of {stats['mean']:.2f}, "
                f"ranging from {stats['min']:.2f} to {stats['max']:.2f}."
            )

    @agent(
        model="openai:gpt-4o-mini",
        temperature=0.7,
        tools=[format_report]
    )
    class Writer:
        """Technical writer who creates clear and professional reports."""

        @tool
        def create_report(
            self,
            content: Annotated[str, Field(
                description="The content to include in the report"
            )],
            report_type: Annotated[str, Field(
                description="Type of report (e.g., analysis, summary, technical)"
            )] = "analysis"
        ) -> str:
            """Create a professional report from the given content."""
            # Generate a title based on the report type
            title = f"{report_type.title()} Report"

            # Split content into sections
            sections = [
                "Introduction",
                "Analysis",
                content,
                "Conclusion"
            ]

            # Format the report using the format_report tool
            return format_report(
                title=title,
                sections=sections,
                summary=f"This report provides a {report_type} of the given data."
            )


async def main():
    # Create an instance of our research team
    team = ResearchTeam()

    # Example 1: Basic research task
    print("Example 1: Basic Research Task")
    print("=" * 50)

    response = await team.aprocess(
        "We need to analyze these numbers: [10.5, 20.5, 15.0, 30.0, 25.5] "
        "and create a report for stakeholders."
    )
    print(response.content)
    print("\n" + "=" * 50 + "\n")

    # Example 2: Complex research project
    print("Example 2: Complex Research Project")
    print("=" * 50)

    response = await team.aprocess(
        "Let's conduct a research project on quarterly sales data:\n"
        "Q1: [100.0, 120.0, 95.0]\n"
        "Q2: [115.0, 125.0, 105.0]\n"
        "Q3: [130.0, 140.0, 125.0]\n"
        "Q4: [150.0, 160.0, 145.0]\n\n"
        "Analyze the trends and prepare a detailed report."
    )
    print(response.content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
