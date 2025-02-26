"""Collaborative Team Example

This example demonstrates a more advanced team setup with specialized agents
collaborating on a complex task. The team includes:
1. A project manager that coordinates the overall process
2. A researcher that gathers information
3. A developer that creates code solutions
4. A tester that validates the solutions

This example shows how agents can collaborate, share context, and build on each other's work.
"""

import asyncio
from typing import Annotated, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from legion import agent, leader, team, tool
from legion.memory.providers.memory import InMemoryProvider 

load_dotenv()


# Define specialized tools for team members
@tool
def search_documentation(
    query: Annotated[str, Field(description="Search query for documentation")]
) -> str:
    """Search documentation for relevant information."""
    # This is a mock implementation
    documentation = {
        "python": "Python is a high-level, interpreted programming language known for its readability and versatility.",
        "api": "API (Application Programming Interface) allows different software applications to communicate with each other.",
        "database": "Databases are organized collections of structured information or data, typically stored electronically.",
        "algorithm": "An algorithm is a step-by-step procedure for solving a problem or accomplishing a task.",
        "framework": "A framework is a pre-built, reusable software environment that provides a foundation for developing applications."
    }
    
    # Simple keyword matching
    results = []
    for keyword, info in documentation.items():
        if keyword.lower() in query.lower():
            results.append(f"- {keyword.title()}: {info}")
    
    if not results:
        return "No relevant documentation found. Try a different search term."
    
    return "Documentation search results:\n" + "\n".join(results)


@tool
def validate_code(
    code: Annotated[str, Field(description="Code to validate")],
    language: Annotated[str, Field(description="Programming language")] = "python"
) -> Dict[str, any]:
    """Validate code for syntax and basic logic issues."""
    # This is a mock implementation
    if language.lower() != "python":
        return {
            "valid": False,
            "errors": [f"Validation for {language} is not supported. Only Python is supported."]
        }
    
    # Very basic Python syntax checking
    common_errors = [
        ("print ", "print(", "Missing parentheses in print function"),
        ("import pandas", "import pandas as pd", "Pandas is typically imported as pd"),
        ("if x = ", "if x == ", "Assignment operator used in condition instead of comparison"),
        ("except:", "except Exception as e:", "Bare except clause should specify exception type")
    ]
    
    errors = []
    warnings = []
    
    for pattern, suggestion, message in common_errors:
        if pattern in code:
            if "print" in pattern:  # Critical error
                errors.append(f"Error: {message}. Suggestion: Use {suggestion}")
            else:  # Just a warning
                warnings.append(f"Warning: {message}. Suggestion: Use {suggestion}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


class CodeSolution(BaseModel):
    """Schema for code solutions."""
    
    language: str = Field(description="Programming language used")
    code: str = Field(description="The actual code solution")
    explanation: str = Field(description="Explanation of how the code works")
    requirements: Optional[List[str]] = Field(description="Required libraries or dependencies", default=None)


# Define the collaborative team
@team
class DevelopmentTeam:
    """A team that collaborates on software development tasks."""

    # Shared memory provider for the team
    memory = InMemoryProvider()

    @leader(
        model="openai:gpt-4o-mini",
        temperature=0.2
    )
    class ProjectManager:
        """Project manager who coordinates the team and ensures the final deliverable meets requirements."""
        
        @tool
        def create_project_plan(
            self,
            task: Annotated[str, Field(description="The task to create a plan for")],
            steps: Annotated[int, Field(description="Number of steps in the plan")] = 3
        ) -> str:
            """Create a structured project plan with defined steps."""
            if steps < 2:
                steps = 2  # Minimum 2 steps
            elif steps > 5:
                steps = 5  # Maximum 5 steps
                
            return f"Project Plan for: {task}\n" + "\n".join([f"{i+1}. [Step {i+1} description]" for i in range(steps)])

    @agent(
        model="openai:gpt-4o-mini",
        temperature=0.4,
        tools=[search_documentation]
    )
    class Researcher:
        """Research specialist who gathers information and provides context for solutions."""
        
        @tool
        def summarize_findings(
            self,
            information: Annotated[List[str], Field(description="List of information pieces to summarize")]
        ) -> str:
            """Summarize research findings into a concise format."""
            if not information:
                return "No information provided to summarize."
                
            return "Research Summary:\n" + "\n".join([f"- {item}" for item in information])

    @agent(
        model="openai:gpt-4o-mini",
        temperature=0.7
    )
    class Developer:
        """Software developer who creates code solutions based on requirements and research."""
        
        @tool
        def generate_solution(
            self,
            requirements: Annotated[str, Field(description="Requirements for the solution")],
            language: Annotated[str, Field(description="Programming language to use")] = "python"
        ) -> CodeSolution:
            """Generate a code solution based on requirements."""
            # This tool doesn't actually generate code - the LLM will do that
            # This is just a structured way to return the solution
            return CodeSolution(
                language=language,
                code="# Code will be generated by the LLM",
                explanation="Explanation will be provided by the LLM",
                requirements=[]
            )

    @agent(
        model="openai:gpt-4o-mini",
        temperature=0.3,
        tools=[validate_code]
    )
    class Tester:
        """Quality assurance specialist who tests and validates solutions."""
        
        @tool
        def provide_feedback(
            self,
            solution: Annotated[str, Field(description="Solution to provide feedback on")],
            issues: Annotated[List[str], Field(description="List of identified issues")] = None
        ) -> str:
            """Provide structured feedback on a solution."""
            if not issues:
                return "Feedback: The solution looks good with no obvious issues."
                
            return "Feedback:\n" + "\n".join([f"- Issue: {issue}" for issue in issues])


async def main():
    # Create an instance of our development team
    team = DevelopmentTeam()

    # Example 1: Simple programming task
    print("Example 1: Simple Programming Task")
    print("=" * 50)

    response = await team.aprocess(
        "Create a Python function that calculates the Fibonacci sequence up to n terms. "
        "Make sure it's efficient and well-documented."
    )
    print(response.content)
    print("\n" + "=" * 50 + "\n")

    # Example 2: More complex task requiring research
    print("Example 2: Complex Task with Research")
    print("=" * 50)

    response = await team.aprocess(
        "We need to create a simple API endpoint using Flask that connects to a SQLite database "
        "and returns data in JSON format. Please research, develop, and test this solution."
    )
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main()) 